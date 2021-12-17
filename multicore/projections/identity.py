from torch import Tensor

from multicore.projections.projection import Projection


class Identity(Projection):

    def apply(self, data: Tensor) -> Tensor:
        return data

    def T_apply(self, data: Tensor) -> Tensor:
        return data
