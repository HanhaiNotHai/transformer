from copy import deepcopy
from math import inf, sqrt

import torch
from torch import Tensor, nn

from config import Config


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

    def forward(self, x: Tensor) -> Tensor:
        x += self.PE[: x.shape[-2]]
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

    def forward(self, x: Tensor) -> Tensor:
        # [b, l] -> [b, l, d_model]
        x = self.embedding(x)
        x *= self.embedding_scaling
        x = self.positional_encoding(x)
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


class EncoderLayer(nn.Module):

    def __init__(
        self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(d_model, h)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, x_mask: Tensor = None) -> Tensor:
        residual = x
        x = self.mha(x, x, x, x_mask)
        x = self.mha_dropout(x)
        x += residual
        x = self.mha_norm(x)

        residual = x
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        x += residual
        x = self.ffn_norm(x)

        return x


class Encoder(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        h: int = 8,  # num of k,q,v heads
        d_ff: int = 2048,
        dropout: float = 0.1,
        N: int = 6,  # num of encoder,decoder layers
    ) -> None:
        super().__init__()

        encoder_layer = EncoderLayer(d_model, h, d_ff, dropout)
        self.layers = nn.ModuleList(deepcopy(encoder_layer) for _ in range(N))

    def forward(self, x: Tensor, x_mask: Tensor = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, x_mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(
        self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, h)
        self.mha1_dropout = nn.Dropout(dropout)
        self.mha1_norm = nn.LayerNorm(d_model)

        self.mha2 = MultiHeadAttention(d_model, h)
        self.mha2_dropout = nn.Dropout(dropout)
        self.mha2_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, y: Tensor, x: Tensor, y_mask: Tensor = None, x_mask: Tensor = None) -> Tensor:
        '''
        y: decoder input
        x: encoder output
        '''

        residual = y
        y = self.mha1(y, y, y, y_mask)
        x = self.mha1_dropout(x)
        y += residual
        y = self.mha1_norm(y)

        residual = y
        y = self.mha2(y, x, x, x_mask)
        x = self.mha2_dropout(x)
        y += residual
        y = self.mha2_norm(y)

        residual = y
        y = self.ffn(y)
        x = self.ffn_dropout(x)
        y += residual
        y = self.ffn_norm(y)

        return y


class Decoder(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        h: int = 8,  # num of k,q,v heads
        d_ff: int = 2048,
        dropout: float = 0.1,
        N: int = 6,  # num of encoder,decoder layers
    ) -> None:
        super().__init__()

        decoder_layer = DecoderLayer(d_model, h, d_ff, dropout)
        self.layers = nn.ModuleList(deepcopy(decoder_layer) for _ in range(N))

    def forward(self, y: Tensor, x: Tensor, y_mask: Tensor = None, x_mask: Tensor = None) -> Tensor:
        '''
        y: decoder input
        x: encoder output
        '''

        for layer in self.layers:
            y = layer(y, x, y_mask, x_mask)
        return y


class Transformer(nn.Module):

    def __init__(
        self,
        config: Config,
        vocab_size: int,
        d_model: int = 512,
        n_position: int = 100,  # positional encoding length
        dropout: float = 0.1,
        h: int = 8,  # num of k,q,v heads
        d_ff: int = 2048,
        N: int = 6,  # num of encoder,decoder layers
        ckpt_path: str = None,
    ) -> None:
        super().__init__()

        self.embedder = Embedder(vocab_size, d_model, n_position, dropout)
        self.encoder = Encoder(d_model, h, d_ff, dropout, N)
        self.decoder = Decoder(d_model, h, d_ff, dropout, N)
        self.linear = nn.Linear(d_model, vocab_size)

        # share the same weight matrix between the two embedding layers
        # and the pre-softmax linear transformation
        self.linear.weight = self.embedder.embedding.weight

        # Prevent leftward information flow in the decoder.
        subsequent_mask = (
            torch.triu(torch.ones([config.n_position, config.n_position]), diagonal=1) == 0
        )
        self.subsequent_mask: Tensor
        self.register_buffer('subsequent_mask', subsequent_mask, False)

        self.to(config.device)

        if ckpt_path is not None:
            # WHY: eval() or traversing children() is faster on cuda.
            self.eval()
            # load_state_dict() is faster when on the same device as nn.Module.
            # WHY: load_state_dict() is faster after eval().
            self.load_state_dict(torch.load(ckpt_path, config.device))
        else:
            # WHY: iterating parameters() is faster on cuda.
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        self.device = config.device
        self.max_len = config.n_position
        self.pad_id = config.pad_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.beam_size = config.beam_size
        self.length_penalty = config.length_penalty

    def get_subsequent_mask(self, seq_len: int) -> Tensor:
        return self.subsequent_mask[:seq_len, :seq_len]

    def forward(self, x: Tensor, y: Tensor, x_mask: Tensor = None, y_mask: Tensor = None) -> Tensor:
        # WHY: (subsequent_mask & y_mask) is faster than (y_mask & subsequent_mask).
        y_mask = self.get_subsequent_mask(y.shape[1]) & y_mask

        x = self.embedder(x)
        y = self.embedder(y)
        x = self.encoder(x, x_mask)
        y = self.decoder(y, x, y_mask, x_mask)
        y = self.linear(y)
        return y

    def save(self, ckpt_path: str) -> None:
        torch.save(self.state_dict(), ckpt_path)

    @torch.inference_mode()
    def inference(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.embedder(x)
        x = self.encoder(x)

        seq_len = min(x.shape[1] + 50, self.max_len)
        y = torch.empty([seq_len], dtype=torch.long, device=self.device)
        y[0] = self.bos_id

        for i in range(1, y.shape[0]):
            y_mask = self.get_subsequent_mask(i)

            y_emb = self.embedder(y[:i])
            dec_out = self.decoder(y_emb, x, y_mask)
            logits: Tensor = self.linear(dec_out[-1])

            y[i] = logits.argmax()
            if y[i] == self.eos_id:
                # Remove BOS and EOS ids.
                return y[1:i]
        # Remove BOS id.
        return y[1:]
