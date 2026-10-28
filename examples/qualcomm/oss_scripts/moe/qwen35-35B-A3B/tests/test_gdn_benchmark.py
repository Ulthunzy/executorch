import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "gdn_benchmark", Path(__file__).resolve().parents[1] / "tools/gdn_benchmark.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class GdnBenchmarkTest(unittest.TestCase):
    def test_moe_prefill_traffic_distinguishes_resident_state(self):
        decode = benchmark.traffic_model(1, 1, 32, 128, 128)
        prefill = benchmark.traffic_model(1, 128, 32, 128, 128)
        self.assertEqual(prefill["state_bytes"], 1024 * 1024)
        self.assertEqual(
            prefill["mathematical_operations"],
            128 * decode["mathematical_operations"],
        )
        self.assertGreater(
            prefill["estimated_loop_bytes"],
            50 * prefill["minimum_external_bytes"],
        )
        self.assertEqual(
            benchmark.rates(prefill, 1)["estimated_loop_GB_per_s"],
            prefill["estimated_loop_bytes"] / 1e6,
        )

    def test_runner_timing_excludes_total_and_warmup(self):
        log = (
            "Perform 5 inferences for warming up\n"
            "I 100 inferences took 123.45 ms, avg 1.2345 ms\n"
            "I Total 100 inferences took 500.00 ms, avg 5.00000 ms\n"
        )
        self.assertEqual(benchmark.parse_latency(log), 1.2345)
        with self.assertRaises(RuntimeError):
            benchmark.parse_latency("runner failed to load graph")

    def test_latency_and_peak_validation(self):
        model = benchmark.traffic_model(1, 1, 32, 128, 128)
        for latency in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                benchmark.rates(model, latency)
        with self.assertRaises(ValueError):
            benchmark.rates(model, 1, peak_gbps=0)

    def test_missing_outputs_fail_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "output_0.raw").write_bytes(b"attention")
            with self.assertRaises(RuntimeError):
                benchmark.output_hashes(path)
            (path / "output_1.raw").write_bytes(b"state")
            first = benchmark.output_hashes(path)
            (path / "output_1.raw").write_bytes(b"wrong")
            self.assertNotEqual(first, benchmark.output_hashes(path))

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is required for binary output comparisons",
    )
    def test_numerical_checks_reject_nonfinite_and_compare_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual, baseline, reference = (
                root / name for name in ("actual", "baseline", "reference")
            )
            for path in (actual, baseline, reference):
                path.mkdir()
            values = struct.pack("ff", 1.0, 2.0)
            (reference / "output_0.raw").write_bytes(values)
            (baseline / "output_0.raw").write_bytes(values)
            (actual / "output_0.raw").write_bytes(struct.pack("ff", 1.0, 2.0001))
            specs = [{"file": "output_0.raw", "dtype": "float32"}]
            compatible, metrics = benchmark.compare_outputs(
                actual, baseline, reference, specs, 0, 0
            )
            self.assertFalse(compatible)
            self.assertGreater(metrics[0]["reference_rmse"], 0)
            compatible, _ = benchmark.compare_outputs(
                actual, baseline, reference, specs, 0.001, 0
            )
            self.assertTrue(compatible)
            (actual / "output_0.raw").write_bytes(struct.pack("ff", 1.0, float("nan")))
            compatible, metrics = benchmark.compare_outputs(
                actual, baseline, reference, specs, 0.001, 0
            )
            self.assertFalse(compatible)
            json.dumps(metrics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
