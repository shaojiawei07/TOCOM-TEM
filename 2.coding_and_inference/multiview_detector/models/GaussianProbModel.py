# import warnings
# from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from compressai.ops import LowerBound


def GaussianLikelihoodEstimation(inputs, scales, means):

    lower_bound_scale = LowerBound(0.11).to('cuda:0')
    likelihood_lower_bound = LowerBound(1e-9).to('cuda:0')

    def _likelihood(inputs, scales, means):

        half = float(0.5)
        values = inputs - means
        scales = lower_bound_scale(scales)
        values = torch.abs(values)
        upper = _standardized_cumulative((half - values) / scales)
        lower = _standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative( inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    likelihood = _likelihood(inputs, scales, means)
    likelihood = likelihood_lower_bound(likelihood)

    return likelihood



# class GaussianConditional(EntropyModel):
#     r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
#     S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
#     hyperprior" <https://arxiv.org/abs/1802.01436>`_.
#     This is a re-implementation of the Gaussian conditional layer in
#     *tensorflow/compression*. See the `tensorflow documentation
#     <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
#     for more information.
#     """

#     def __init__(
#         self,
#         scale_table: Optional[Union[List, Tuple]],
#         *args: Any,
#         scale_bound: float = 0.11,
#         tail_mass: float = 1e-9,
#         **kwargs: Any,
#     ):
#         super().__init__(*args, **kwargs)

#         if not isinstance(scale_table, (type(None), list, tuple)):
#             raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

#         if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
#             raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

#         if scale_table and (
#             scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
#         ):
#             raise ValueError(f'Invalid scale_table "({scale_table})"')

#         self.tail_mass = float(tail_mass)
#         if scale_bound is None and scale_table:
#             scale_bound = self.scale_table[0]
#         if scale_bound <= 0:
#             raise ValueError("Invalid parameters")
#         self.lower_bound_scale = LowerBound(scale_bound)

#         self.register_buffer(
#             "scale_table",
#             self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
#         )

#         self.register_buffer(
#             "scale_bound",
#             torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
#         )

#     @staticmethod
#     def _prepare_scale_table(scale_table):
#         return torch.Tensor(tuple(float(s) for s in scale_table))

#     def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
#         half = float(0.5)
#         const = float(-(2**-0.5))
#         # Using the complementary error function maximizes numerical precision.
#         return half * torch.erfc(const * inputs)

#     @staticmethod
#     def _standardized_quantile(quantile):
#         return scipy.stats.norm.ppf(quantile)

#     def update_scale_table(self, scale_table, force=False):
#         # Check if we need to update the gaussian conditional parameters, the
#         # offsets are only computed and stored when the conditonal model is
#         # updated.
#         if self._offset.numel() > 0 and not force:
#             return False
#         device = self.scale_table.device
#         self.scale_table = self._prepare_scale_table(scale_table).to(device)
#         self.update()
#         return True

#     def update(self):
#         multiplier = -self._standardized_quantile(self.tail_mass / 2)
#         pmf_center = torch.ceil(self.scale_table * multiplier).int()
#         pmf_length = 2 * pmf_center + 1
#         max_length = torch.max(pmf_length).item()

#         device = pmf_center.device
#         samples = torch.abs(
#             torch.arange(max_length, device=device).int() - pmf_center[:, None]
#         )
#         samples_scale = self.scale_table.unsqueeze(1)
#         samples = samples.float()
#         samples_scale = samples_scale.float()
#         upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
#         lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
#         pmf = upper - lower

#         tail_mass = 2 * lower[:, :1]

#         quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
#         quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
#         self._quantized_cdf = quantized_cdf
#         self._offset = -pmf_center
#         self._cdf_length = pmf_length + 2

#     def _likelihood(
#         self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
#     ) -> Tensor:
#         half = float(0.5)

#         if means is not None:
#             values = inputs - means
#         else:
#             values = inputs

#         scales = self.lower_bound_scale(scales)

#         values = torch.abs(values)
#         upper = self._standardized_cumulative((half - values) / scales)
#         lower = self._standardized_cumulative((-half - values) / scales)
#         likelihood = upper - lower

#         return likelihood

#     def forward(
#         self,
#         inputs: Tensor,
#         scales: Tensor,
#         means: Optional[Tensor] = None,
#         training: Optional[bool] = None,
#     ) -> Tuple[Tensor, Tensor]:
#         if training is None:
#             training = self.training
#         outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
#         likelihood = self._likelihood(outputs, scales, means)
#         if self.use_likelihood_bound:
#             likelihood = self.likelihood_lower_bound(likelihood)
#         return outputs, likelihood

#     def build_indexes(self, scales: Tensor) -> Tensor:
#         scales = self.lower_bound_scale(scales)
#         indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
#         for s in self.scale_table[:-1]:
#             indexes -= (scales <= s).int()
#         return indexes

# if __name__ == "__main__":


# test()
#     print(GaussianLikelihoodEstimation())