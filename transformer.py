import heapq
from copy import deepcopy
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from config import Config


def Singleton(cls):
    instance = dict()

    def singleton(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return singleton


class Module(nn.Module):
    __call__: Callable[..., Tensor]


class Project(nn.Linear, Module):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)


@Singleton
class RotaryPositionEmbedding(Module):

    def __init__(
        self, d_head: int = 64, n_position: int = 100  # length of rotary position embedding
    ) -> None:
        super().__init__()

        # [d_head // 2]
        theta = 1 / (10000 ** (torch.arange(0, d_head, 2) / d_head))
        # [n_position]
        m = torch.arange(n_position, dtype=theta.dtype)

        # [n_position, d_head // 2]
        x = torch.outer(m, theta)
        # [n_position, d_head // 2] -> [n_position, d_head]
        x = x.repeat_interleave(2, dim=1)
        # [n_position, d_head] -> [n_position, 1, d_head]
        x.unsqueeze_(1)

        cos = torch.cos(x)
        sin = torch.sin(x)

        self.cos: Tensor
        self.sin: Tensor
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x = torch.empty_like(x)
        x[..., ::2] = -x2
        x[..., 1::2] = x1
        return x

    def forward(self, x: Tensor, i: int | None = None) -> Tensor:
        if i is None:
            cos = self.cos[: x.shape[-3]]
            sin = self.sin[: x.shape[-3]]
        else:
            cos = self.cos[i]
            sin = self.sin[i]
        return x * cos + self.rotate_half(x) * sin


class MultiHeadAttention(Module):
    '''encoder-decoder mha'''

    def __init__(
        self,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        n_position: int = 100,  # length of rotary position embedding
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_head = d_model // h_q
        self.d_kv = h_kv * self.d_head
        self.G = h_q // h_kv  # groups of grouped-query attention
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.dropout = dropout

        self.Wq = Project(d_model, d_model)
        self.Wkv = Project(d_model, 2 * self.d_kv)
        self.rotary_position_embedding = RotaryPositionEmbedding(self.d_head, n_position)
        self.Wo = Project(d_model, d_model)

    def forward(
        self, y: Tensor, x: Tensor, x_mask: Tensor | None = None, i: int | None = None
    ) -> Tensor:
        # [b, l, d_model] -> [b, l, h_q * d_head]
        q = self.Wq(y)
        # [b, l, h_q * d_head] -> [b, l, h_q, d_head]
        q = q.reshape(*q.shape[:2], self.h_q, self.d_head)
        q = self.rotary_position_embedding(q, i)

        if not i:
            # [b, l, d_model] -> [b, l, 2 * h_kv * d_head]
            kv = self.Wkv(x)
            # [b, l, 2 * h_kv * d_head] -> [b, l, h_kv * d_head] * 2
            k, v = kv.split([self.d_kv, self.d_kv], dim=2)

            # [b, l, h_kv * d_head] -> [b, l, h_kv, d_head]
            k = k.reshape(*k.shape[:2], self.h_kv, self.d_head)
            v = v.reshape(*v.shape[:2], self.h_kv, self.d_head)

            k = self.rotary_position_embedding(k)

            self.k_cache = k
            self.v_cache = v
        else:
            k = self.k_cache
            v = self.v_cache

        if self.G > 1:
            # [b, l, h_kv, d_head] -> [b, l, h_q, d_head]
            k = k.repeat_interleave(self.G, dim=2)
            v = v.repeat_interleave(self.G, dim=2)

        # [b, l, h_q, d_head] -> [b, h_q, l, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = scaled_dot_product_attention(q, k, v, x_mask, self.dropout if self.training else 0)
        # [b, h_q, l, d_head] -> [b, l, h_q, d_head]
        x = x.transpose(1, 2)
        # [b, l, h_q, d_head] -> [b, l, d_model]
        x = x.reshape(*x.shape[:2], self.d_model)
        x = self.Wo(x)

        return x


class MultiHeadSelfAttention(Module):
    '''encoder and decoder self_mha'''

    def __init__(
        self,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        n_position: int = 100,  # length of rotary position embedding
        dropout: float = 0.1,
        inference_batch_size: int = 1,
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
        self.dropout = dropout

        self.Wqkv = Project(d_model, (h_q + 2 * h_kv) * self.d_head)
        self.rotary_position_embedding = RotaryPositionEmbedding(self.d_head, n_position)
        self.Wo = Project(d_model, d_model)

        if kv_cache:
            k_cache = torch.empty(inference_batch_size, n_position, h_kv, self.d_head)
            v_cache = torch.empty(inference_batch_size, n_position, h_kv, self.d_head)
            self.k_cache: Tensor
            self.v_cache: Tensor
            self.register_buffer('k_cache', k_cache, persistent=False)
            self.register_buffer('v_cache', v_cache, persistent=False)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, i: int | None = None, kv_cache: bool = False
    ) -> Tensor:
        # [b, l, d_model] -> [b, l, (h_q + 2 * h_kv) * d_head]
        qkv = self.Wqkv(x)
        # [b, l, (h_q + 2 * h_kv) * self.d_head] -> [b, l, {h_q, h_kv, h_kv} * d_head]
        q, k, v = qkv.split([self.d_q, self.d_kv, self.d_kv], dim=2)

        # [b, l, h * d_head] -> [b, l, h, d_head]
        q = q.reshape(*q.shape[:2], self.h_q, self.d_head)
        k = k.reshape(*k.shape[:2], self.h_kv, self.d_head)
        v = v.reshape(*v.shape[:2], self.h_kv, self.d_head)

        q = self.rotary_position_embedding(q, i)
        k = self.rotary_position_embedding(k, i)

        if kv_cache:
            self.k_cache[:, i] = k[:, 0]
            self.v_cache[:, i] = v[:, 0]
            k = self.k_cache[:, : i + 1]
            v = self.v_cache[:, : i + 1]

        if self.G > 1:
            # [b, l, h_kv, d_head] -> [b, l, h_q, d_head]
            k = k.repeat_interleave(self.G, dim=2)
            v = v.repeat_interleave(self.G, dim=2)

        # [b, l, h_q, d_head] -> [b, h_q, l, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = scaled_dot_product_attention(q, k, v, mask, self.dropout if self.training else 0)
        # [b, h_q, l, d_head] -> [b, l, h_q, d_head]
        x = x.transpose(1, 2)
        # [b, l, h_q, d_head] -> [b, l, d_model]
        x = x.reshape(*x.shape[:2], self.d_model)
        x = self.Wo(x)

        return x


class FeedForward(Module):
    '''SwiGLU'''

    def __init__(self, d_model: int = 512, d_ff: int = 2048) -> None:
        super().__init__()

        self.W = Project(d_model, d_ff)
        self.silu = nn.SiLU(inplace=True)
        self.V = Project(d_model, d_ff)
        self.W2 = Project(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # [b, l, d_model] -> [b, l, d_ff] -> [b, l, d_model]
        return self.W2(self.silu(self.W(x)) * self.V(x))


class MixtureOfExperts(Module):

    def __init__(
        self, d_model: int = 512, d_ff: int = 2048, num_experts: int = 8, topk: int = 2
    ) -> None:
        super().__init__()

        self.topk = topk

        self.gate = Project(d_model, num_experts)
        self.softmax: Callable[..., Tensor] = nn.Softmax(dim=-1)
        expert = FeedForward(d_model, d_ff)
        self.experts = nn.ModuleList(deepcopy(expert) for _ in range(num_experts))

    def forward(self, x: Tensor) -> Tensor:
        # [b, l, num_experts]
        gate_logits = self.gate(x)
        # [b, l, topk]
        weights, selected_experts = torch.topk(gate_logits, self.topk)

        weights = self.softmax(weights)
        # [b, l, topk, 1] for weight * expert
        weights = weights.unsqueeze(-1)

        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            *idx, nth_expert = torch.where(selected_experts == i)
            y[*idx] += weights[*idx, nth_expert] * expert(x[*idx])
        return y


class EncoderLayer(Module):

    def __init__(
        self,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        n_position: int = 100,  # length of rotary position embedding
        dropout: float = 0.1,
        d_ff: int = 2048,
        num_experts: int = 8,
        topk: int = 2,
    ) -> None:
        super().__init__()

        self.self_mha = MultiHeadSelfAttention(d_model, h_q, h_kv, n_position, dropout)
        self.mha_norm = nn.RMSNorm(d_model)

        self.ffn = MixtureOfExperts(d_model, d_ff, num_experts, topk)
        self.ffn_norm = nn.RMSNorm(d_model)

    def forward(self, x: Tensor, x_mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self.self_mha(x, x_mask)
        x += residual
        x = self.mha_norm(x)

        residual = x
        x = self.ffn(x)
        x += residual
        x = self.ffn_norm(x)

        return x


class Encoder(Module):

    def __init__(
        self,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        n_position: int = 100,  # length of rotary position embedding
        dropout: float = 0.1,
        d_ff: int = 2048,
        num_experts: int = 8,
        topk: int = 2,
        N: int = 6,  # num of encoder layers
    ) -> None:
        super().__init__()

        encoder_layer = EncoderLayer(
            d_model, h_q, h_kv, n_position, dropout, d_ff, num_experts, topk
        )
        self.layers = nn.ModuleList(deepcopy(encoder_layer) for _ in range(N))

    def forward(self, x: Tensor, x_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, x_mask)
        return x


class DecoderLayer(Module):

    def __init__(
        self,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        n_position: int = 100,  # length of rotary position embedding
        dropout: float = 0.1,
        inference_batch_size: int = 1,
        d_ff: int = 2048,
        num_experts: int = 8,
        topk: int = 2,
    ) -> None:
        super().__init__()

        self.self_mha = MultiHeadSelfAttention(
            d_model, h_q, h_kv, n_position, dropout, inference_batch_size, kv_cache=True
        )
        self.self_mha_norm = nn.RMSNorm(d_model)

        self.mha = MultiHeadAttention(d_model, h_q, h_kv, n_position, dropout)
        self.mha_norm = nn.RMSNorm(d_model)

        self.ffn = MixtureOfExperts(d_model, d_ff, num_experts, topk)
        self.ffn_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        y: Tensor,
        x: Tensor,
        y_mask: Tensor | None = None,
        x_mask: Tensor | None = None,
        i: int | None = None,
        kv_cache: bool = False,
    ) -> Tensor:
        '''
        y: decoder input
        x: encoder output
        '''

        residual = y
        y = self.self_mha(y, y_mask, i, kv_cache)
        y += residual
        y = self.self_mha_norm(y)

        residual = y
        y = self.mha(y, x, x_mask, i)
        y += residual
        y = self.mha_norm(y)

        residual = y
        y = self.ffn(y)
        y += residual
        y = self.ffn_norm(y)

        return y


class Decoder(Module):

    def __init__(
        self,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        n_position: int = 100,  # length of rotary position embedding
        dropout: float = 0.1,
        inference_batch_size: int = 1,
        d_ff: int = 2048,
        num_experts: int = 8,
        topk: int = 2,
        N: int = 6,  # num of decoder layers
    ) -> None:
        super().__init__()

        decoder_layer = DecoderLayer(
            d_model, h_q, h_kv, n_position, dropout, inference_batch_size, d_ff, num_experts, topk
        )
        self.layers = nn.ModuleList(deepcopy(decoder_layer) for _ in range(N))

    def forward(
        self,
        y: Tensor,
        x: Tensor,
        y_mask: Tensor | None = None,
        x_mask: Tensor | None = None,
        i: int | None = None,
        kv_cache: bool = False,
    ) -> Tensor:
        '''
        y: decoder input
        x: encoder output
        '''

        for layer in self.layers:
            y = layer(y, x, y_mask, x_mask, i, kv_cache)
        return y


class Transformer(Module):

    def __init__(
        self,
        config: Config,
        vocab_size: int,
        d_model: int = 512,
        h_q: int = 8,  # num of q heads
        h_kv: int = 1,  # num of k,v heads
        n_position: int = 100,  # length of rotary position embedding
        dropout: float = 0.1,
        d_ff: int = 2048,
        num_experts: int = 8,
        topk: int = 2,
        N: int = 6,  # num of encoder,decoder layers
        inference_batch_size: int = 1,
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()

        # share the same weight matrix between the two embedding layers
        # and the pre-softmax linear transformation
        self.embedding = nn.Embedding(vocab_size, d_model, config.pad_id)
        self.encoder = Encoder(d_model, h_q, h_kv, n_position, dropout, d_ff, num_experts, topk, N)
        self.decoder = Decoder(
            d_model,
            h_q,
            h_kv,
            n_position,
            dropout,
            inference_batch_size,
            d_ff,
            num_experts,
            topk,
            N,
        )

        self.to(config.device)

        if ckpt_path is not None:
            # WHY: eval() or traversing children() is faster on cuda.
            self.eval()
            # load_state_dict() is faster when on the same device as nn.Module.
            # WHY: load_state_dict() is faster after eval().
            self.load_state_dict(torch.load(ckpt_path, config.device, weights_only=True))
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

    def forward(self, x: Tensor, y: Tensor, x_mask: Tensor, y_mask: Tensor) -> Tensor:
        x = self.embedding(x)
        y = self.embedding(y)
        x = self.encoder(x, x_mask)
        y = self.decoder(y, x, y_mask, x_mask)
        y = y @ self.embedding.weight.T
        return y

    def save(self, ckpt_path: str) -> None:
        torch.save(self.state_dict(), ckpt_path)

    @torch.inference_mode()
    def inference(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.encoder(x)

        seq_len = min(x.shape[1] + 50, self.max_len)
        y = torch.empty([1, seq_len], dtype=torch.long).to(self.device)
        y[0, 0] = self.bos_id

        for i in range(seq_len - 1):
            y_emb = self.embedding(y[:, i : i + 1])
            dec_out = self.decoder(y_emb, x, i=i, kv_cache=True)
            logits = dec_out @ self.embedding.weight.T

            y[0, i + 1] = logits.argmax()
            if y[0, i + 1] == self.eos_id:
                # Remove BOS and EOS ids.
                return y[0, 1 : i + 1]
        # Remove BOS id.
        return y[0, 1:]

    @torch.inference_mode()
    def inference_batch(self, x: Tensor, x_mask: Tensor | None = None) -> list[list[int]]:
        x = self.embedding(x)
        x = self.encoder(x, x_mask)

        bs = x.shape[0]
        seq_len = min(x.shape[1] + 50, self.max_len)
        y = torch.empty([bs, seq_len], dtype=torch.long).to(self.device)
        y[:, 0] = self.bos_id
        y_mask = torch.ones(bs, 1, 1, 1).bool().to(self.device)
        EOSidx = torch.full([bs], seq_len).to(self.device)

        for i in range(seq_len - 1):
            y_emb = self.embedding(y[:, i : i + 1])
            dec_out = self.decoder(y_emb, x, y_mask, x_mask, i, kv_cache=True)
            logits = dec_out @ self.embedding.weight.T

            y[:, i + 1] = logits.argmax(dim=-1).squeeze()
            idx = torch.where((y[:, i + 1] == self.eos_id) & y_mask.squeeze())[0]
            y_mask[idx] = False
            EOSidx[idx] = i + 1

            if torch.all(y_mask == False):
                break

        # Remove BOS and EOS ids.
        return [y[i][1 : EOSidx[i]].tolist() for i in range(bs)]

    @torch.inference_mode()
    def beam_search(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.encoder(x)

        seq_len = min(x.shape[1] + 50, self.max_len)
        # (score, EOSidx, -log_prob, y)  EOSidx==0 means EOS is not found.
        beams = [(0, 0, 0, torch.empty([1, seq_len], dtype=torch.long).to(self.device))]
        for *_, y in beams:
            y[0, 0] = self.bos_id

        for i in range(seq_len - 1):
            new_beams = []
            for score, EOSidx, log_prob, y in beams:
                if EOSidx:
                    heapq.heappush(new_beams, (score, EOSidx, log_prob, y))
                    continue

                y_emb = self.embedding(y[:, i : i + 1])
                dec_out = self.decoder(y_emb, x, i=i, kv_cache=True)
                logits = dec_out @ self.embedding.weight.T

                logits.squeeze_()
                log_probs = logits.log_softmax(dim=-1)
                indecies = log_probs.argsort(descending=True)[: self.beam_size]
                for index in indecies:
                    new_log_prob = log_prob + log_probs[index].item()
                    new_score = new_log_prob / (i + 1) ** self.length_penalty
                    new_EOSidx = i + 1 if index == self.eos_id else 0
                    new_y = y.clone()
                    new_y[0, i + 1] = index
                    if len(new_beams) < self.beam_size:
                        heapq.heappush(new_beams, (new_score, new_EOSidx, new_log_prob, new_y))
                    else:
                        heapq.heappushpop(new_beams, (new_score, new_EOSidx, new_log_prob, new_y))

            beams = new_beams
            if all(EOSidx for _, EOSidx, *_ in beams):
                break

        beams.sort()
        _, EOSidx, _, y = beams[-1]
        # Remove BOS and EOS ids.
        return y[0, 1:EOSidx] if EOSidx else y[0, 1:]
