from torch import Tensor

from config import Config
from tokenizer.tokenizer import Tokenizer


def Singleton(cls):
    instance = dict()

    def singleton(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return singleton


class DataLoader:

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


class TestDataloader(DataLoader):

    def __init__(self, config: Config, tokenizer: Tokenizer, n: int = None) -> None:
        with open(config.test_src) as f:
            self.test_src_dataset = f.read().splitlines()
        self.test_x_dataset = list(map(tokenizer.encode, self.test_src_dataset))

        with open(config.test_tgt) as f:
            self.test_tgt_dataset = f.read().splitlines()

        assert len(self.test_x_dataset) == len(self.test_src_dataset) == len(self.test_tgt_dataset)
        if n is None:
            self.len = len(self.test_x_dataset)
        else:
            self.len = n
            self.test_x_dataset = self.test_x_dataset[:n]
            self.test_src_dataset = self.test_src_dataset[:n]
            self.test_tgt_dataset = self.test_tgt_dataset[:n]

    def __getitem__(self, index) -> tuple[Tensor, str, str]:
        return (
            self.test_x_dataset[index],
            self.test_src_dataset[index],
            self.test_tgt_dataset[index],
        )
