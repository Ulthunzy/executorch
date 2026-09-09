# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Dict, Optional, Union

import torch
from executorch.backends.qualcomm._passes import SeqMSE
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    TEXT_DECODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference import DecoderInference
from executorch.examples.qualcomm.oss_scripts.llama.quantize.strategy import (
    QuantizationStrategy,
)
from executorch.examples.qualcomm.oss_scripts.llama.utils import safe_dataloader_iter
from torch.utils.data import DataLoader
from tqdm import tqdm


class PTQStrategy(QuantizationStrategy):
    """Post-training quantization strategy: calibrate only."""

    def __init__(
        self,
        inference: DecoderInference,
        module: torch.fx.GraphModule,
        seq_mse_candidates: int = 0,
        tok_embedding: Optional[torch.fx.GraphModule] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            inference=inference, module=module, tok_embedding=tok_embedding
        )
        self._seq_mse_candidates = seq_mse_candidates
        self._device = torch.device(device)

    def _calibrate(self, calib_loader: Dict[str, DataLoader]) -> None:
        audio_dataloader = calib_loader.get(AUDIO_ENCODER)
        vision_dataloader = calib_loader.get(VISION_ENCODER)
        text_dataloader = calib_loader[TEXT_DECODER]

        for batch_idx, (audio_batch, vision_batch, text_batch) in tqdm(
            enumerate(
                zip(
                    safe_dataloader_iter(audio_dataloader),
                    safe_dataloader_iter(vision_dataloader),
                    text_dataloader,
                )
            ),
            total=len(text_dataloader),
        ):
            input_ids = text_batch["input_ids"]
            attn_mask = text_batch["attention_mask"]
            hidden_states = (*audio_batch, *vision_batch)
            self._inference.predict_step(
                self._module,
                input_ids=input_ids,
                attn_mask=attn_mask,
                hidden_states=hidden_states,
                tok_embedding=self._tok_embedding,
            )
            if batch_idx == 0 and self._seq_mse_candidates:
                with SeqMSE(self._module, self._seq_mse_candidates):
                    self._inference.predict_step(
                        self._module,
                        input_ids=input_ids,
                        attn_mask=attn_mask,
                        hidden_states=hidden_states,
                        tok_embedding=self._tok_embedding,
                    )

    @torch.no_grad()
    def quantize(self, calib_loader, **kwargs):
        use_cuda = self._device.type == "cuda"
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA calibration is unavailable; check CUDA_VISIBLE_DEVICES and the PyTorch CUDA build."
            )
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        start = time.perf_counter()
        try:
            if use_cuda:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.cuda.reset_peak_memory_stats(self._device)
            self._module.to(self._device)
            if self._tok_embedding is not None:
                self._tok_embedding.to(self._device)
            self._inference.to(self._device)
            logging.info("PTQ calibration device: %s", self._device)
            self._calibrate(calib_loader)
            if use_cuda:
                torch.cuda.synchronize(self._device)
                logging.info(
                    "CUDA calibration completed in %.2fs; peak allocated memory %.2f GiB",
                    time.perf_counter() - start,
                    torch.cuda.max_memory_allocated(self._device) / 1024**3,
                )
        finally:
            self._module.cpu()
            if self._tok_embedding is not None:
                self._tok_embedding.cpu()
            self._inference.to("cpu")
            if use_cuda:
                torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
                torch.backends.cudnn.allow_tf32 = cudnn_tf32
                torch.cuda.empty_cache()
        return self._module
