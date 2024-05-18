import os
from dataclasses import dataclass

import torch


@dataclass
class Config:
    # Set proxy if you are using.
    proxy = None  # '127.0.0.1:7890'
    # Use wandb or not.
    WANDB: bool = True

    # device
    cuda: bool = True

    # tokenizer
    vocab_size: int = 37000
    train_max_len: int = 16
    tokenizer_dir = 'tokenizer/'
    vocab_filename: str = tokenizer_dir + 'vocab.json'
    merges_filename: str = tokenizer_dir + 'merges.txt'
    special_tokens = ['<unk>', '<pad>', '<BOS>', '<EOS>']
    unk_id: int = 0
    pad_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    unk_token: str = special_tokens[unk_id]
    pad_token: str = special_tokens[pad_id]
    bos_token: str = special_tokens[bos_id]
    eos_token: str = special_tokens[eos_id]

    # data
    batch_size: int = 512 + 256 + 64
    dataset_dir = 'dataset/wmt14en_de/'
    train_src = dataset_dir + 'train.en'
    train_tgt = dataset_dir + 'train.de'
    test_src = dataset_dir + 'test.en'
    test_tgt = dataset_dir + 'test.de'

    # model
    ckpt_path: str = 'checkpoint/transformer.ckpt'
    d_model: int = 512
    n_position: int = 100  # positional encoding length
    dropout: float = 0.1
    N: int = 6  # num of encoder,decoder layers
    h: int = 8  # num of k,q,v heads
    d_ff: int = 2048

    # train
    warmup_steps: int = 14000
    eval_save_per_steps: int = 1000
    n_best_models: int = 5  # num of best models to save
    epochs: int = 200
    label_smoothing: float = 0.1
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9

    # inference
    n_test: int = 128
    beam_size: int = 4
    length_penalty: float = 0.6

    def __post_init__(self) -> None:
        assert self.d_model % self.h == 0, 'd_model must be divisible by h'
        self.device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')
        if self.cuda and not torch.cuda.is_available():
            print('Warning: CUDA is not available, using CPU instead.')
        if self.proxy is not None:
            os.environ['http_proxy'] = os.environ['https_proxy'] = self.proxy

    @property
    def model_config(self) -> dict:
        return dict(
            config=self,
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_position=self.n_position,
            dropout=self.dropout,
            h=self.h,
            d_ff=self.d_ff,
            N=self.N,
        )

    @property
    def wandb_config(self) -> dict:
        return dict(
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
        )


@dataclass
class BigConfig(Config):
    d_model: int = 1024
    dropout: float = 0.3
    N: int = 6  # num of encoder,decoder layers
    h: int = 16  # num of k,q,v heads
    d_ff: int = 4096