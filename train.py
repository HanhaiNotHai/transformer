import evaluate
import torch
import wandb
from torch import Tensor
from tqdm import tqdm, trange

from config import Config
from tokenizer.tokenizer import Tokenizer
from transformer import Transformer
from util import Saver, TestDataloader, TrainDataloader


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

        with tqdm(train_dataloader, desc='train', leave=False) as pbar:
            for x, x_pad_mask, y, y_pad_mask in pbar:
                pbar.set_postfix_str(
                    f'{torch.mps.current_allocated_memory() / 2**30:.1f} / '
                    f'{torch.mps.driver_allocated_memory() / 2**30:.1f} GB'
                )
                step += 1

                # [b, l - 1]
                target = y[:, 1:]
                y = y[:, :-1]
                y_pad_mask = y_pad_mask[:, :-1]

                optimizer.zero_grad()
                logits = transformer(x, y, x_pad_mask, y_pad_mask)
                # [b, l - 1, vocab_size] -> [b, vocab_size, l - 1]
                logits.transpose_(1, 2)
                loss: Tensor = cross_entropy_loss(logits, target)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if config.WANDB:
                    wandb.log(
                        {'train_loss': loss, 'learning_rate': optimizer.param_groups[0]['lr']},
                        step,
                    )

                if step % config.eval_save_per_steps == 0:
                    transformer.eval()
                    for x, _, tgt in tqdm(test_dataloader, 'test', leave=False):
                        y_hat = transformer.beam_search(x)
                        pred = tokenizer.decode(y_hat)
                        bleu.add_batch(predictions=[pred], references=[tgt])
                    transformer.train()

                    score = bleu.compute()['score']
                    wandb.log({'bleu': score}, step) if config.WANDB else None
                    saver.save(score, epoch, step)

    wandb.finish() if config.WANDB else None


if __name__ == '__main__':
    main()
