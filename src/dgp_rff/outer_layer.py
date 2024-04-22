import torch

from torch import Tensor
from pyro.nn import PyroModule
from src.dgp_rff.inner_layer import FirstLayer, SecondLayer


class SingleGP(PyroModule):
    r"""
    A single random feature-based GP is equivalent to a two-layer Bayesian neural network.

    Attributes
    ----------
    layers: PyroModule
        The layers containing the FirstLayer and SecondLayer.
    """
    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
    ) -> None:
        """
        :param in_dim: int
            The input dimension
        :param out_dim:
            The output dimension
        :param J:
            The number of random features
        """
        super().__init__()

        assert in_dim > 0 and out_dim > 0 and J > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = [FirstLayer(in_dim, 2 * J), SecondLayer(2 * J, out_dim)]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        x = self.layers[0](x)
        mu = self.layers[1](x)

        return mu
