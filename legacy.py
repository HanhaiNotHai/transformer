from copy import deepcopy
from math import inf, sqrt

import torch
from torch import Tensor, nn


class Project(nn.Linear):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 512, n_position: int = 100):
        super().__init__()

        # [n_position, 1]
        pos = torch.arange(n_position, dtype=torch.float).unsqueeze_(1)
        # [d_model // 2]
        i = torch.arange(0, d_model, 2, dtype=torch.float)
        # [n_position, d_model // 2]
        x = pos / torch.pow(10000, i / d_model)
        PE = torch.FloatTensor(torch.Size([n_position, d_model]))
        PE[:, ::2] = torch.sin(x)
        PE[:, 1::2] = torch.cos(x)
        # Not a parameter, but to(device) with nn.Module.
        self.PE: Tensor
        self.register_buffer('PE', PE, False)

    def forward(self, x: Tensor, i: int = None) -> Tensor:
        if i is None:
            x += self.PE[: x.shape[-2]]
        else:
            x += self.PE[i : i + 1]
        return x


class Embedder(nn.Module):
    '''Combine the two embedding and positional encoding layers into one.'''

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_position: int = 100,  # positional encoding length
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scaling = sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model, n_position)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, i: int = None) -> Tensor:
        # [b, l] -> [b, l, d_model]
        x = self.embedding(x)
        x *= self.embedding_scaling
        x = self.positional_encoding(x, i)
        x = self.dropout(x)
        return x


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


class FeedForward(nn.Module):

    def __init__(self, d_model: int = 512, d_ff: int = 2048) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # [b, l, d_model] -> [b, l, d_ff]
        x = self.linear1(x)
        self.relu(x)
        # [b, l, d_ff] -> [b, l, d_model]
        x = self.linear2(x)
        return x
