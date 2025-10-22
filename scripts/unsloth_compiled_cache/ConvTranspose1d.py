
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False}
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.mllama.modeling_mllama import (F, List, Optional, Tuple, nn)

def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
    if self.padding_mode != "zeros":
        raise ValueError(
            "Only `zeros` padding mode is supported for ConvTranspose1d"
        )

    assert isinstance(self.padding, tuple)
    # One cannot replace List by Tuple or Sequence in "_output_padding" because
    # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
    num_spatial_dims = 1
    output_padding = self._output_padding(
        input,
        output_size,
        self.stride,  # type: ignore[arg-type]
        self.padding,  # type: ignore[arg-type]
        self.kernel_size,  # type: ignore[arg-type]
        num_spatial_dims,
        self.dilation,  # type: ignore[arg-type]
    )
    return F.conv_transpose1d(
        input,
        self.weight,
        self.bias,
        self.stride,
        self.padding,
        output_padding,
        self.groups,
        self.dilation,
    ).to(input.dtype)
