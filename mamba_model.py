import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union


# Mamba Model Args
@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    features: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


# Pure Mamba network
class KFGN_Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.encode = nn.Linear(args.features, args.d_model)
        self.encoder_layers = nn.ModuleList(
            [ResidualBlock(args) for _ in range(args.n_layer)]
        )
        self.encoder_norm = RMSNorm(args.d_model)
        self.decode = nn.Linear(args.d_model, args.features)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len, features)
        x = self.encode(input_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)
        x = self.decode(x)
        return x


# Residual Block in Mamba Model
class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x_res = x
        x_norm = self.norm(x)
        x_mixed = self.mixer(x_norm)
        # residual connection
        output = x_mixed + x_res
        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(
            args.d_inner, args.dt_rank + args.d_state * 2, bias=False
        )

        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), "n -> d n", d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(
            split_size=[self.args.d_inner, self.args.d_inner], dim=-1
        )

        # depthwise conv along sequence
        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        # x: (b, l, d_inner)
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)

        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Standard selective scan without any graph / adjacency mixing.
        u:      (b, l, d_in)
        delta:  (b, l, d_in)
        A:      (d_in, n)
        B, C:   (b, l, n)
        D:      (d_in,)
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(
            delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n"
        )

        x = torch.zeros((b, d_in, n), device=u.device, dtype=u.dtype)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # C[:, i, :] is (b, n)
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (b, l, d_in)

        y = y + u * D  # (b, l, d_in)

        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = x * rms * self.weight
        return output
