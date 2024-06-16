import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
from torch import Tensor
from torch import tensor


class FirstLayer:

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            num_layer: int = 1,
            kernel_name: str = 'RBF',
    ) -> None:

        self.J = hid_dim // 2
        self.weight = pyro.sample(
            f"{kernel_name}: layer-{num_layer} Omega",
            dist.Normal(tensor(0.), tensor(1.)).expand([in_dim, self.J]).to_event(2)
        )

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:

        hid = torch.matmul(x, self.weight)
        mu = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(tensor(self.J))
        return mu


class SecondLayer:

    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            num_layer: int = 1,
            kernel_name: str = 'RBF',
            noise: bool = True,

    ) -> None:

        self.J = hid_dim // 2

        self.weight = pyro.sample(
            f"{kernel_name}: Layer-{num_layer} Theta",
            dist.Normal(tensor(0.), tensor(1.)).expand([hid_dim, out_dim]).to_event(2)
        )

        if noise:
            self.bias = pyro.sample(
                f"{kernel_name}: Layer-{num_layer} Sigma",
                dist.Gamma(tensor(.5), tensor(1.)).expand([out_dim]).to_event(1)
            )
        else:
            self.bias = None

        self.noise = noise
        self.num_layer = num_layer
        self.kernel_name = kernel_name

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:

        mu = torch.matmul(x, self.weight)

        # if self.noise:
        #     with pyro.plate(f"{self.kernel_name}: Layer-{self.num_layer} Noise", x.shape[0]):
        #         mu = pyro.sample(
        #             f"{self.kernel_name}: Layer-{self.num_layer} Hidden State",
        #             dist.Normal(mu, self.bias).to_event(1),
        #         )

        return mu
