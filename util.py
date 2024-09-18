import heapq
import os
from itertools import batched
from time import strftime

import torch
from torch import Tensor
from tqdm import tqdm

from config import Config
from tokenizer.tokenizer import Tokenizer
from transformer import Transformer


class DataLoader:
    len: int

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> 'DataLoader':
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index >= len(self):
            raise StopIteration
        return self[self.index]


class TrainDataloader(DataLoader):

    def __init__(self, config: Config, tokenizer: Tokenizer) -> None:
        self.batch_size = config.batch_size
        self.device = torch.device(config.device)
        self.tokenizer = tokenizer
        self.n_position = config.n_position
        self.train_x_dataset = self.read(config.train_src, mask_subsequent=False)
        self.train_y_dataset = self.read(config.train_tgt, mask_subsequent=True)
        print('Done.')

        assert len(self.train_x_dataset) == len(
            self.train_y_dataset
        ), 'x and y of train dataset should have the same length.'
        self.len = len(self.train_x_dataset)

    def read(self, file: str, mask_subsequent: bool) -> list[tuple[Tensor, Tensor]]:
        encoded_file = file + f'.encoded{self.batch_size}'
        if os.path.exists(encoded_file):
            print(f'Loading {encoded_file} ...')
            return torch.load(encoded_file, self.device, weights_only=True)

        print(f'Reading {file} ...')
        with open(file) as f:
            text_dataset = f.read().splitlines()[::-1]

        text_batched_dataset = list(batched(text_dataset, self.batch_size))

        print(f'Encoding {file} ...')
        tensor_dataset = list(map(self.tokenizer.encode_batch, tqdm(text_batched_dataset)))

        if not mask_subsequent:
            for _, x_mask in tensor_dataset:
                # [b, l] -> [b, 1, 1, l]
                x_mask.unsqueeze_(1).unsqueeze_(1)
        else:
            # Prevent leftward information flow in the decoder.
            subsequent_mask = torch.ones([self.n_position, self.n_position]).bool().tril()
            for i, (y, y_mask) in enumerate(tensor_dataset):
                # [b, l] -> [b, l - 1]
                y_mask = y_mask[:, :-1]
                # [b, l - 1] -> [b, 1, l - 1]
                y_mask.unsqueeze_(1)
                # WHY: (subsequent_mask & y_mask) is faster than (y_mask & subsequent_mask).
                # [l - 1, l - 1] & [b, 1, l - 1] -> [b, l - 1, l - 1]
                y_mask = subsequent_mask[: y_mask.shape[2], : y_mask.shape[2]] & y_mask
                # [b, l - 1, l - 1] -> [b, 1, l - 1, l - 1]
                tensor_dataset[i] = (y, y_mask.unsqueeze_(1))

        tensor_dataset = [
            (tokens.to(self.device), mask.to(self.device)) for tokens, mask in tensor_dataset
        ]

        print(f'Saving {encoded_file} ...')
        torch.save(tensor_dataset, encoded_file)

        return tensor_dataset

    def __getitem__(self, index) -> tuple[Tensor, ...]:
        return *self.train_x_dataset[index], *self.train_y_dataset[index]


class TestBatchDataloader(DataLoader):

    def __init__(self, config: Config, tokenizer: Tokenizer) -> None:
        with open(config.test_src) as f:
            test_src_dataset = f.read().splitlines()
        test_src_batched_dataset = list(batched(test_src_dataset, config.batch_size))
        self.test_x_dataset = list(map(tokenizer.encode_batch, test_src_batched_dataset))

        with open(config.test_tgt) as f:
            test_tgt_dataset = f.read().splitlines()
        self.test_tgt_dataset = list(batched(test_tgt_dataset, config.batch_size))

        assert len(self.test_x_dataset) == len(
            self.test_tgt_dataset
        ), 'src and tgt of test dataset should have the same length'
        self.len = len(self.test_x_dataset)

        for _, x_mask in self.test_x_dataset:
            x_mask.unsqueeze_(1).unsqueeze_(1)
        self.test_x_dataset = [
            (x.to(config.device), x_mask.to(config.device)) for x, x_mask in self.test_x_dataset
        ]

    def __getitem__(self, index) -> tuple[Tensor, Tensor, tuple[str]]:
        return *self.test_x_dataset[index], self.test_tgt_dataset[index]


class TestDataloader(DataLoader):

    def __init__(self, config: Config, tokenizer: Tokenizer) -> None:
        with open(config.test_src) as f:
            self.test_src_dataset = f.read().splitlines()
        self.test_x_dataset = list(map(tokenizer.encode, self.test_src_dataset))

        with open(config.test_tgt) as f:
            self.test_tgt_dataset = f.read().splitlines()

        assert len(self.test_src_dataset) == len(
            self.test_tgt_dataset
        ), 'src and tgt of test dataset should have the same length'
        self.len = len(self.test_x_dataset)

        self.test_x_dataset = [x.to(config.device) for x in self.test_x_dataset]

    def __getitem__(self, index) -> tuple[Tensor, str, str]:
        return (
            self.test_x_dataset[index],
            self.test_src_dataset[index],
            self.test_tgt_dataset[index],
        )


class Saver:

    def __init__(self, transformer: Transformer, n_best_models: int | None = None) -> None:
        self.transformer = transformer
        self.save_dir = 'checkpoint/' + strftime('%m%d_%X/')

        if n_best_models is None:
            self.save = self.save0
        else:
            self.save = self.save1
            # (score, -step, save_path)
            self.best_models: list[tuple[float, int, str]] = []
            self.n_best_models = n_best_models

    def save0(self, score: float, epoch: int, step: int) -> None:
        '''save all checkpoints'''

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = self.save_dir + f'{score:.6f}_{epoch}_{step}.ckpt'
        torch.save(self.transformer.state_dict(), save_path)

    def save1(self, score: float, epoch: int, step: int) -> None:
        '''len(self.best_models) < self.n_best_models'''

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = self.save_dir + f'{score:.6f}_{epoch}_{step}.ckpt'
        heapq.heappush(self.best_models, (score, -step, save_path))
        torch.save(self.transformer.state_dict(), save_path)

        if len(self.best_models) >= self.n_best_models:
            self.save = self.save2

    def save2(self, score: float, epoch: int, step: int) -> None:
        '''len(self.best_models) >= self.n_best_models'''

        save_path = self.save_dir + f'{score:.6f}_{epoch}_{step}.ckpt'
        pop_save_path = heapq.heappushpop(self.best_models, (score, -step, save_path))[-1]
        if pop_save_path != save_path:
            os.remove(pop_save_path)
            torch.save(self.transformer.state_dict(), save_path)
