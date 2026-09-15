"""ADB transport for the isolated, parameterized teacher-forcing PPL driver."""

import shlex
import subprocess
import uuid
from pathlib import Path, PurePosixPath

from ..ppl.artifacts import digest, file_hash, read_json, write_json

PROTOCOL = "executorch.ppl.driver.v1"
DECODERS = ("lincal", "qwen3", "qwen2_5", "llama3", "llama2")


def device_path(value):
    """Validate an explicit absolute device path before quoting it for a shell."""
    if (
        not value
        or not PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or "\n" in value
        or "\x00" in value
    ):
        raise ValueError("Device paths must be absolute without '..', newlines or NUL")
    return str(PurePosixPath(value))


class PTEEngine:
    """Run only an explicitly configured decoder and validated driver protocol."""

    def __init__(self, args, model, tokenizer, manifest, identity):
        if (
            not args.serial
            or not args.device_lib_dir
            or args.driver is None
            or args.decoder not in DECODERS
        ):
            raise ValueError(
                "PTE requires --serial, --device-lib-dir, --driver and explicit --decoder"
            )
        if args.dtype != "float32":
            raise ValueError(
                "PTE dtype is exported in the model; --dtype bfloat16 is unsupported"
            )
        if args.mode not in ("hybrid", "kv"):
            raise ValueError("PTE supports hybrid or AR1 kv mode only")
        self.serial = args.serial
        self.libdir = device_path(args.device_lib_dir)
        self.model = Path(model)
        self.tokenizer = Path(tokenizer)
        self.driver = args.driver.resolve()
        self.device_model = (
            device_path(args.device_model) if args.device_model else None
        )
        self.mode, self.decoder = args.mode, args.decoder
        self.manifest = manifest
        self.model_hash = identity["model"]["files"].get(self.model.name)
        self.source = Path(__file__).resolve().parent.parent / "ppl" / "ppl_driver.cpp"
        if not self.model.is_file() or self.model.suffix != ".pte":
            raise ValueError("PTE --model must name one local .pte file")
        if not self.tokenizer.is_file() or self.tokenizer.name != "tokenizer.json":
            raise ValueError("PTE --tokenizer must be a local tokenizer.json")
        if not self.driver.is_file():
            raise ValueError(
                "PTE --driver must name the locally built evaluation binary"
            )
        sidecar = self.driver.parent / "driver_source.sha256"
        if not sidecar.is_file() or sidecar.read_text().strip() != file_hash(
            self.source
        ):
            raise ValueError(
                "Driver source fingerprint mismatch; build with --entry-point ppl/ppl_driver.cpp --define ET_QNN_PPL_EVALUATION"
            )

    def _adb(self, *args, timeout=180):
        return subprocess.run(
            ["adb", "-s", self.serial, *map(str, args)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout

    def _remote_hash(self, path):
        value = self._adb("shell", "sha256sum " + shlex.quote(path)).split()
        if not value or len(value[0]) != 64:
            raise ValueError(f"Cannot hash device artifact: {path}")
        return value[0]

    def _push(self, source, destination):
        self._adb("push", source, destination, timeout=600)
        if self._remote_hash(destination) != file_hash(source):
            raise ValueError(f"Device artifact hash mismatch: {destination}")

    def _libraries(self):
        command = (
            f"for f in {shlex.quote(self.libdir)}/*.so {shlex.quote(self.libdir)}/*.so.[0-9]*; do "
            'if test -f "$f"; then sha256sum "$f" || exit 1; fi; done'
        )
        lines = sorted(self._adb("shell", command).splitlines())
        if not lines or not any(".so" in line for line in lines):
            raise ValueError("No device libraries found in --device-lib-dir")
        if any(len(line.split()[0]) != 64 for line in lines):
            raise ValueError("Invalid device library hashes")
        return lines

    def identity(self, *, offline=False):
        """Bind binary, tokenizer and device libraries; offline mode never calls ADB."""
        result = dict(
            protocol=PROTOCOL,
            serial=self.serial,
            mode=self.mode,
            decoder=self.decoder,
            device_lib_dir=self.libdir,
            device_model=self.device_model,
            driver_sha256=file_hash(self.driver),
            driver_source_sha256=file_hash(self.source),
            tokenizer_file_sha256=file_hash(self.tokenizer),
            dtype="exported_float32_or_uint16",
            libraries=None,
        )
        if not offline:
            result["libraries"] = self._libraries()
            result["device_build"] = self._adb(
                "shell", "getprop ro.build.fingerprint"
            ).strip()
            if (
                self.device_model
                and self._remote_hash(self.device_model) != self.model_hash
            ):
                raise ValueError(
                    "--device-model hash differs from --model; refusing to overwrite it"
                )
        return result

    def plan(self, identity, windows, remote="/data/local/tmp/executorch-ppl/ATTEMPT"):
        """Generate reviewable config and argv without loading or pushing a model."""
        remote = device_path(remote)
        config = dict(
            protocol=PROTOCOL,
            model=self.device_model or remote + "/model.pte",
            tokenizer=remote + "/tokenizer.json",
            seq_len=self.manifest["context_len"],
            mode=self.mode,
            decoder=self.decoder,
            windows=windows,
            tokenizer_max_id=self.manifest["tokenizer"]["max_token_id"],
            corpus_hash=identity["tokens_sha256"],
            model_hash=self.model_hash,
            identity_sha256=digest(identity),
            tokenizer_hash=file_hash(self.tokenizer),
            driver_hash=file_hash(self.driver),
            libraries=identity["runtime"]["libraries"],
        )
        uploads = [
            dict(local=str(self.driver), remote=remote + "/ppl_driver"),
            dict(local=str(self.tokenizer), remote=remote + "/tokenizer.json"),
            dict(local="CORPUS/tokens.u64", remote=remote + "/tokens.u64"),
            dict(local="RUN/device-config.json", remote=remote + "/config.json"),
        ]
        if not self.device_model:
            uploads.append(dict(local=str(self.model), remote=remote + "/model.pte"))
        command = (
            f"cd {shlex.quote(self.libdir)} && "
            f"env LD_LIBRARY_PATH={shlex.quote(self.libdir)} ADSP_LIBRARY_PATH={shlex.quote(self.libdir)} "
            + shlex.join(
                [
                    remote + "/ppl_driver",
                    remote + "/config.json",
                    remote + "/tokens.u64",
                    remote + "/results",
                ]
            )
        )
        return dict(
            config=config,
            uploads=uploads,
            argv=["adb", "-s", self.serial, "shell", command],
            device_output=remote + "/results",
            cleanup="Device attempt is retained for inspection; no automatic deletion",
        )

    def run(self, corpus_root, output, identity, windows):
        """Use a fresh device attempt and return validated-protocol partial results."""
        if self.identity() != identity["runtime"]:
            raise ValueError("PTE device identity changed before execution")
        remote = "/data/local/tmp/executorch-ppl/" + uuid.uuid4().hex
        plan = self.plan(identity, windows, remote)
        config_path = output / "device-config.json"
        write_json(config_path, plan["config"])
        self._adb("shell", "mkdir -p " + shlex.quote(plan["device_output"]))
        sources = [self.driver, self.tokenizer, corpus_root / "tokens.u64", config_path]
        if not self.device_model:
            sources.append(self.model)
        for source, upload in zip(sources, plan["uploads"]):
            self._push(source, upload["remote"])
        self._adb("shell", "chmod 755 " + shlex.quote(remote + "/ppl_driver"))
        attempt = output / ("device-" + remote.rsplit("/", 1)[1])
        attempt.mkdir()
        write_json(attempt / "plan.json", plan)
        failure = None
        try:
            with (attempt / "device.log").open("w") as log:
                subprocess.run(
                    plan["argv"], check=True, stdout=log, stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError as error:
            failure = str(error)
        finally:
            try:
                self._adb("pull", plan["device_output"] + "/.", attempt, timeout=600)
            except subprocess.SubprocessError as error:
                raise ValueError(
                    f"Cannot retrieve PPL results; driver failure: {failure or 'unknown'}. "
                    f"Inspect {attempt / 'device.log'}; device attempt retained at {remote}"
                ) from error
        if self.identity() != identity["runtime"]:
            raise ValueError("PTE device identity changed during execution")
        if not (attempt / "metadata.json").is_file():
            raise ValueError(
                f"Driver produced no metadata; no results are eligible. Inspect {attempt / 'device.log'}"
            )
        metadata = read_json(attempt / "metadata.json")
        if (
            metadata.get("protocol") != PROTOCOL
            or metadata.get("config") != plan["config"]
        ):
            raise ValueError(
                "Driver protocol/config mismatch; rebuild the generic PPL driver"
            )
        results = {
            w["id"]: read_json(attempt / (w["id"] + ".json"))
            for w in windows
            if (attempt / (w["id"] + ".json")).is_file()
        }
        return results, dict(driver_metadata=metadata, device_attempt=remote), failure
