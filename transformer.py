from copy import deepcopy
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

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


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int = 512, h_q: int = 8, h_kv: int = 1, max_len: int = 100) -> None:
        super().__init__()

        self.d_head = d_model // h_q
        self.d_kv = h_kv * self.d_head
        self.G = h_q // h_kv  # groups of grouped-query attention
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv

        self.Wq = Project(d_model, d_model)
        self.Wk = Project(d_model, self.d_kv)
        self.Wv = Project(d_model, self.d_kv)
        self.Wo = Project(d_model, d_model)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor = None,
        i: int = None,
    ) -> Tensor:
        # [b, l, d_model] -> [b, l, h_q * d_head]
        q = self.Wq(q)
        # [b, l, h_q * d_head] -> [b, l, h_q, d_head]
        q = q.reshape(*q.shape[:-1], self.h_q, self.d_head)

        if not i:
            # [b, l, d_model] -> [b, l, h_kv * d_head]
            k = self.Wk(k)
            v = self.Wv(v)

            # [b, l,   * d_head] -> [b, l,  , d_head]
            k = k.reshape(*k.shape[:-1], self.h_kv, self.d_head)
            v = v.reshape(*v.shape[:-1], self.h_kv, self.d_head)

            self.k_cache = k
            self.v_cache = v
        else:
            k = self.k_cache
            v = self.v_cache

        if self.G > 1:
            # [b, l, h_kv, d_head] -> [b, l, h_q, d_head]
            k = torch.repeat_interleave(k, self.G, -2)
            v = torch.repeat_interleave(v, self.G, -2)

        # [b, l, h_q, d_head] -> [b, h_q, l, d_head]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        x = scaled_dot_product_attention(q, k, v, mask)
        # [b, h_q, l, d_head] -> [b, l, h_q, d_head]
        x = x.transpose(-2, -3)
        # [b, l, h_q, d_head] -> [b, l, d_model]
        x = x.reshape(*x.shape[:-2], self.d_model)
        x = self.Wo(x)

        return x


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        h_q: int = 8,
        h_kv: int = 1,
        max_len: int = 100,
        kv_cache: bool = False,
    ) -> None:
        super().__init__()

        self.d_head = d_model // h_q
        self.d_q = d_model
        self.d_kv = h_kv * self.d_head
        self.G = h_q // h_kv  # groups of grouped-query attention
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv

        self.Wqkv = Project(d_model, (h_q + 2 * h_kv) * self.d_head)
        self.Wo = Project(d_model, d_model)

        if kv_cache:
            k_cache = torch.zeros(max_len, h_kv, self.d_head)
            v_cache = torch.zeros(max_len, h_kv, self.d_head)
            self.k_cache: Tensor
            self.v_cache: Tensor
            self.register_buffer('k_cache', k_cache, False)
            self.register_buffer('v_cache', v_cache, False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        kv_cache: bool = False,
        i: int = None,
    ) -> Tensor:
        # [b, l, d_model] -> [b, l, (h_q + 2 * h_kv) * self.d_head]
        qkv: Tensor = self.Wqkv(x)
        # [b, l, (h_q + 2 * h_kv) * self.d_head] -> [b, l, {h_q, h_kv, h_kv} * d_head]
        q, k, v = qkv.split([self.d_q, self.d_kv, self.d_kv], -1)

        # [b, l, h * d_head] -> [b, l, h, d_head]
        q: Tensor = q.reshape(*q.shape[:-1], self.h_q, self.d_head)
        k: Tensor = k.reshape(*k.shape[:-1], self.h_kv, self.d_head)
        v: Tensor = v.reshape(*v.shape[:-1], self.h_kv, self.d_head)

        if kv_cache:
            self.k_cache[i] = k[0]
            self.v_cache[i] = v[0]
            k = self.k_cache[: i + 1]
            v = self.v_cache[: i + 1]

        if self.G > 1:
            # [b, l, h_kv, d_head] -> [b, l, h_q, d_head]
            k = torch.repeat_interleave(k, self.G, -2)
            v = torch.repeat_interleave(v, self.G, -2)

        # [b, l, h_q, d_head] -> [b, h_q, l, d_head]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        x = scaled_dot_product_attention(q, k, v, mask)
        # [b, h_q, l, d_head] -> [b, l, h_q, d_head]
        x = x.transpose(-2, -3)
        # [b, l, h_q, d_head] -> [b, l, d_model]
        x = x.reshape(*x.shape[:-2], self.d_model)
        x = self.Wo(x)

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
        self,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_mha = MultiHeadSelfAttention(d_model, h_q, h_kv)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, x_mask: Tensor = None) -> Tensor:
        residual = x
        x = self.self_mha(x, x_mask)
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
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        d_ff: int = 2048,
        dropout: float = 0.1,
        N: int = 6,  # num of encoder,decoder layers
    ) -> None:
        super().__init__()

        encoder_layer = EncoderLayer(d_model, h_q, h_kv, d_ff, dropout)
        self.layers = nn.ModuleList(deepcopy(encoder_layer) for _ in range(N))

    def forward(self, x: Tensor, x_mask: Tensor = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, x_mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 100,
    ) -> None:
        super().__init__()

        self.self_mha = MultiHeadSelfAttention(d_model, h_q, h_kv, max_len, kv_cache=True)
        self.self_mha_dropout = nn.Dropout(dropout)
        self.self_mha_norm = nn.LayerNorm(d_model)

        self.mha = MultiHeadAttention(d_model, h_q, h_kv, max_len)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        y: Tensor,
        x: Tensor,
        y_mask: Tensor = None,
        x_mask: Tensor = None,
        kv_cache: bool = False,
        i: int = None,
    ) -> Tensor:
        '''
        y: decoder input
        x: encoder output
        '''

        residual = y
        y = self.self_mha(y, y_mask, kv_cache, i)
        x = self.self_mha_dropout(x)
        y += residual
        y = self.self_mha_norm(y)

        residual = y
        y = self.mha(y, x, x, x_mask, i)
        x = self.mha_dropout(x)
        y += residual
        y = self.mha_norm(y)

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
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        d_ff: int = 2048,
        dropout: float = 0.1,
        N: int = 6,  # num of encoder,decoder layers
        max_len: int = 100,
    ) -> None:
        super().__init__()

        decoder_layer = DecoderLayer(d_model, h_q, h_kv, d_ff, dropout, max_len)
        self.layers = nn.ModuleList(deepcopy(decoder_layer) for _ in range(N))

    def forward(
        self,
        y: Tensor,
        x: Tensor,
        y_mask: Tensor = None,
        x_mask: Tensor = None,
        kv_cache: bool = False,
        i: int = None,
    ) -> Tensor:
        '''
        y: decoder input
        x: encoder output
        '''

        for layer in self.layers:
            y = layer(y, x, y_mask, x_mask, kv_cache, i)
        return y


class Transformer(nn.Module):

    def __init__(
        self,
        config: Config,
        vocab_size: int,
        d_model: int = 512,
        n_position: int = 100,  # positional encoding length
        dropout: float = 0.1,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        d_ff: int = 2048,
        N: int = 6,  # num of encoder,decoder layers
        ckpt_path: str = None,
    ) -> None:
        super().__init__()

        self.embedder = Embedder(vocab_size, d_model, n_position, dropout)
        self.encoder = Encoder(d_model, h_q, h_kv, d_ff, dropout, N)
        self.decoder = Decoder(d_model, h_q, h_kv, d_ff, dropout, N, max_len=n_position)
        self.linear = nn.Linear(d_model, vocab_size)

        # share the same weight matrix between the two embedding layers
        # and the pre-softmax linear transformation
        self.linear.weight = self.embedder.embedding.weight

        # Prevent leftward information flow in the decoder.
        subsequent_mask = torch.ones([config.n_position, config.n_position]).bool().tril()
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
        # [b, l] -> [b, 1, 1, l]
        x_mask.unsqueeze_(1).unsqueeze_(1)
        # WHY: (subsequent_mask & y_mask) is faster than (y_mask & subsequent_mask).
        # [l - 1, l - 1] & [b, 1, l - 1] -> [b, l - 1, l - 1]
        y_mask = self.get_subsequent_mask(y.shape[1]) & y_mask.unsqueeze(1)
        # [b, l - 1, l - 1] -> [b, 1, l - 1, l - 1]
        y_mask.unsqueeze_(1)

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

        for i in range(y.shape[0] - 1):
            y_emb = self.embedder(y[i : i + 1], i)
            dec_out = self.decoder(y_emb, x, kv_cache=True, i=i)
            logits: Tensor = self.linear(dec_out)

            y[i + 1] = logits.argmax()
            if y[i + 1] == self.eos_id:
                # Remove BOS and EOS ids.
                return y[1 : i + 1]
        # Remove BOS id.
        return y[1:]
