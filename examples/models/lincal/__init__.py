# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.llama.model import Llama2Model
from executorch.examples.models.lincal.convert_weights import convert_weights


class LincalModel(Llama2Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    "LincalModel",
    "convert_weights",
]
