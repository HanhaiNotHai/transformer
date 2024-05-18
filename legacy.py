from copy import deepcopy
from math import inf, sqrt

import torch
from torch import Tensor, nn


class Project(nn.Linear):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dk: int = 64) -> None:
        super().__init__()

        self.scaling = 1 / sqrt(dk)
        self.softmax = nn.Softmax(-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        # [b, l, dk] -> [b, dk, l]
        k.transpose_(-1, -2)
        # [b, l, l]
        score = q @ k
        score *= self.scaling
        if mask is not None:
            score.masked_fill_(mask.logical_not(), -inf)
        # [b, l, dv]
        return self.softmax(score) @ v


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int = 512, h: int = 8) -> None:
        super().__init__()

        dk = dv = d_model // h

        project = Project(d_model, dk)
        self.Wq = nn.ModuleList(deepcopy(project) for _ in range(h))
        self.Wk = nn.ModuleList(deepcopy(project) for _ in range(h))
        project = Project(d_model, dv)
        self.Wv = nn.ModuleList(deepcopy(project) for _ in range(h))
        self.attention = ScaledDotProductAttention(dk)
        self.Wo = Project(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        # [b, l, d_model] -> [b, l, d_head] * h
        qs = [Wq(q) for Wq in self.Wq]
        ks = [Wk(k) for Wk in self.Wk]
        vs = [Wv(v) for Wv in self.Wv]

        # [b, l, dv] * h
        heads = [self.attention(q, k, v, mask) for q, k, v in zip(qs, ks, vs)]
        # [b, l, dv*h=d_model]
        concat = torch.cat(heads, -1)
        x = self.Wo(concat)

        return x
