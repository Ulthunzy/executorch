# Low-bit HTP compute

This directory contains the shared N2 expansion, HVX/HMX FFN kernels and VTCM tensor layouts. It owns numerical computation only. It does not map expert files, schedule expert IO, implement FastRPC services or orchestrate a model.

Prefill and Decode use different tensor layouts, so their compute implementations remain under `prefill/` and `decode/`. `expert_stream.h` describes the numerical tile handoff. Architecture selection uses `QCOM_HTP_ARCH=79|81|85` from [`../htp_arch.h`](../htp_arch.h).

The service in [`moe/runner/native`](../../oss_scripts/moe/runner/native/) links these sources and owns DSP workers, bank mapping and RPC. Build it with `runner/native/build.sh --arch 81 --output <build-dir>`, or use the MoE root `build.sh`.
