import torch

import torch.nn as nn
import pyro.distributions as dist

from torch import Tensor
from pyro.nn import PyroModule, PyroSample


class FirstLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.

    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)

        self.layer.weight = PyroSample(dist.Normal(0., 1.).expand([self.J, in_dim]).to_event(2))

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the FirstLayer.
        :return: Tensor
            The output of the FirstLayer, which is $\phi(\Omega \times x)$.
        """
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(torch.tensor(self.J))

        return mu


class SecondLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The second layer refers to produce the GP output plus noises.

    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The second layer is based on nn.Linear where the weight($\Theta$) and bias($\Epsilon$) is defined
        by the PyroSample.
    """

    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
    ) -> None:
        """
        :param out_dim: int
            The output dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=True)

        self.layer.weight = PyroSample(dist.Normal(0., 1.).expand([out_dim, hid_dim]).to_event(2))
        self.layer.bias = PyroSample(dist.Normal(0., 1.).expand([out_dim]).to_event(1))

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """
        mu = self.layer(x)

        return mu
