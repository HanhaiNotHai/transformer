import heapq
import os
from itertools import batched
from time import strftime

import evaluate
import torch
import wandb
from torch import Tensor
from tqdm import tqdm, trange

from tokenizer.tokenizer import Tokenizer
from transformer.config import Config
from transformer.transformer import Transformer
from util import DataLoader, TestDataloader


class TrainDataloader(DataLoader):

    def __init__(self, config: Config, tokenizer: Tokenizer) -> None:
        self.batch_size = config.batch_size
        self.tokenizer = tokenizer
        self.train_x_dataset = self.read(config.train_src)
        self.train_y_dataset = self.read(config.train_tgt)
        print('Done.')

        assert len(self.train_x_dataset) == len(self.train_y_dataset)
        self.len = len(self.train_x_dataset)

    def read(self, file: str) -> list[Tensor]:
        encoded_file = file + f'.encoded{self.batch_size}'
        if os.path.exists(encoded_file):
            print(f'Loading {encoded_file} ...')
            return torch.load(encoded_file)

        print(f'Reading {file} ...')
        with open(file) as f:
            text_dataset = f.read().splitlines()
        text_batched_dataset = list(batched(text_dataset, self.batch_size))
        print(f'Encoding {file} ...')
        tensor_dataset = list(map(self.tokenizer.encode_batch, tqdm(text_batched_dataset)))
        print(f'Saving {encoded_file} ...')
        torch.save(tensor_dataset, encoded_file)
        return tensor_dataset

    def __getitem__(self, index) -> tuple[Tensor, ...]:
        return *self.train_x_dataset[index], *self.train_y_dataset[index]


class Scheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(
        self, optimizer: torch.optim.Adam, d_model: int = 512, warmup_steps: int = 4000
    ) -> None:
        self.scaling = d_model**-0.5
        self.warmup_steps_ = warmup_steps**-1.5

        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        lr = self.scaling * min(self._step_count**-0.5, self._step_count * self.warmup_steps_)
        return [lr] * len(self.optimizer.param_groups)


class Saver:

    def __init__(self, transformer: Transformer, n_best_models: int = 10) -> None:
        self.transformer = transformer
        self.n_best_models = n_best_models

        self.save_dir = 'checkpoint/' + strftime('%m%d_%X/')

        # (score, -step, save_path)
        self.best_models = []
        self.save = self.save1

    def save1(self, score: float, epoch: int, step: int) -> None:
        '''len(self.best_models) < self.n_best_models'''

        # Make save_dir on first save.
        if not self.best_models:
            os.makedirs(self.save_dir)

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


def to(batch: tuple[Tensor, ...], device: torch.device) -> tuple[Tensor, ...]:
    return (x.to(device) for x in batch)


def main() -> None:
    config = Config()
    tokenizer = Tokenizer(config)
    train_dataloader = TrainDataloader(config, tokenizer)
    test_dataloader = TestDataloader(config, tokenizer)
    transformer = Transformer(**config.model_config)
    bleu = evaluate.load('sacrebleu')
    saver = Saver(transformer, config.n_best_models)

    cross_entropy_loss = torch.nn.CrossEntropyLoss(
        ignore_index=config.pad_id, label_smoothing=config.label_smoothing
    )
    # WHY: Adam() is faster on cuda.
    optimizer = torch.optim.Adam(transformer.parameters(), betas=config.betas, eps=config.eps)
    scheduler = Scheduler(optimizer, config.d_model, config.warmup_steps)

    wandb.init(project='transformer', config=config.wandb_config) if config.WANDB else None
    wandb.watch(transformer, cross_entropy_loss, log='all') if config.WANDB else None

    step = 0
    for epoch in trange(config.epochs, desc='epoch'):
        wandb.log({'epoch': epoch}) if config.WANDB else None

        for batch in tqdm(train_dataloader, 'train', leave=False):
            step += 1
            x, x_pad_mask, y, y_pad_mask = to(batch, transformer.device)
            # [b, 1, l]
            x_pad_mask.unsqueeze_(1)
            # [b * (l - 1)]
            target = y[:, 1:].reshape(-1)
            # [b, l - 1]
            y = y[:, :-1]
            # [b, 1, l - 1]
            y_pad_mask = y_pad_mask[:, :-1].unsqueeze(1)

            optimizer.zero_grad()
            logits: Tensor = transformer(x, y, x_pad_mask, y_pad_mask)
            # [b, l - 1, vocab_size] -> [b * (l - 1), vocab_size]
            logits = logits.reshape(-1, logits.shape[-1])
            loss: Tensor = cross_entropy_loss(logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if config.WANDB:
                wandb.log(
                    {'train_loss': loss, 'learning_rate': optimizer.param_groups[0]['lr']}, step
                )

            if step % config.eval_save_per_steps == 0:
                transformer.eval()
                for x, _, tgt in tqdm(test_dataloader, 'test', leave=False):
                    x = x.to(transformer.device)
                    y_hat = transformer.inference(x)
                    pred = tokenizer.decode(y_hat)
                    bleu.add_batch(predictions=[pred], references=[tgt])
                transformer.train()

                score = bleu.compute()['score']
                wandb.log({'bleu': score}, step) if config.WANDB else None
                saver.save(score, epoch, step)

    wandb.finish() if config.WANDB else None


if __name__ == '__main__':
    main()
