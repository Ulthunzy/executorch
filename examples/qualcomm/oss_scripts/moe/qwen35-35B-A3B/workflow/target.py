"""Mobile targets supported by the Native N2/HMX implementation."""

from dataclasses import dataclass

from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    HtpArch,
    QcomChipset,
)


SUPPORTED_ARCHES = (HtpArch.V79, HtpArch.V81, HtpArch.V85)
SOC_MODELS = tuple(
    chipset.name
    for chipset, info in _soc_info_table.items()
    if info.htp_info.htp_arch in SUPPORTED_ARCHES
)
DSP_ARCHES = tuple(arch.name.lower() for arch in SUPPORTED_ARCHES)


@dataclass(frozen=True)
class TargetConfig:
    soc_model: str = QcomChipset.SM8975.name

    @property
    def dsp_arch(self):
        return _soc_info_table[
            QcomChipset[self.soc_model]
        ].htp_info.htp_arch.name.lower()

    @classmethod
    def resolve(cls, soc_model=QcomChipset.SM8975.name, dsp_arch=None):
        if not isinstance(soc_model, str) or soc_model not in SOC_MODELS:
            raise ValueError(f"Native MoE requires an HTP architecture in {DSP_ARCHES}")
        target = cls(soc_model)
        expected = target.dsp_arch
        if dsp_arch is not None and dsp_arch != expected:
            raise ValueError(f"{soc_model} requires {expected}, received {dsp_arch}")
        return target

    def to_dict(self):
        return dict(soc_model=self.soc_model, dsp_arch=self.dsp_arch)

    def validate(self):
        self.resolve(self.soc_model)

    def check_manifest(self, metadata):
        # Existing frozen artifacts predate target metadata and are SM8975/V85.
        actual = TargetConfig.resolve(**metadata.get("target", {}))
        actual.validate()
        if actual != self:
            raise ValueError(
                f"Artifact target {actual.to_dict()} differs from {self.to_dict()}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soc-model", choices=SOC_MODELS, required=True)
    parser.add_argument("--dsp-arch", choices=DSP_ARCHES)
    args = parser.parse_args()
    try:
        target = TargetConfig.resolve(args.soc_model, args.dsp_arch)
        print(int(HtpArch[target.dsp_arch.upper()]))
    except ValueError as error:
        parser.error(str(error))
