import os
from time import strftime

import evaluate
from tqdm import tqdm

from tokenizer.tokenizer import Tokenizer
from transformer.config import Config
from transformer.transformer import Transformer
from util import TestDataloader


def main() -> None:
    config = Config()
    tokenizer = Tokenizer(config, config.n_position)
    test_dataloader = TestDataloader(config, tokenizer)
    transformer = Transformer(**config.model_config, ckpt_path=config.ckpt_path)
    bleu = evaluate.load('sacrebleu')
    bleu_all = evaluate.load('sacrebleu')
    result = ['source\ttarget\tpred\tBLEU']

    for x, src, tgt in tqdm(test_dataloader):
        x = x.to(transformer.device)
        y_hat = transformer.inference(x)
        pred = tokenizer.decode(y_hat)

        score = bleu.compute(predictions=[pred], references=[tgt])['score']
        result.append(f'{src}\t{tgt}\t{pred}\t{score}')
        print(result[-1])

        bleu_all.add_batch(predictions=[pred], references=[tgt])

    score = bleu_all.compute()
    print(score)

    result_dir = 'result/'
    os.makedirs(result_dir, exist_ok=True)
    result_file = result_dir + strftime(f'{score['score']:.6f}_%m%d_%X')
    with open(result_file, 'w') as f:
        f.write(f'{score}\n')
        f.write('\n'.join(result))


if __name__ == '__main__':
    main()
