import torch

from torch import Tensor
from pyro.nn import PyroModule
from src.dgp_rff.inner_layer import FirstLayer, SecondLayer


class SingleGP:

    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
            num_layer: int = 1,
            kernel_name: str = 'RBF',
    ) -> None:

        self.layers = [
            FirstLayer(in_dim=in_dim, hid_dim=2 * J, num_layer=num_layer, kernel_name=kernel_name),
            SecondLayer(hid_dim=2 * J, out_dim=out_dim, num_layer=num_layer, kernel_name=kernel_name)
        ]

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:

        x = self.layers[0].forward(x)
        mu = self.layers[1].forward(x)

        return mu
