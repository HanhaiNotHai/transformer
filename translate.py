from tokenizer.tokenizer import Tokenizer
from transformer.config import Config
from transformer.transformer import Transformer


def main() -> None:
    config = Config()
    tokenizer = Tokenizer(config, config.n_position)
    transformer = Transformer(**config.model_config, ckpt_path=config.ckpt_path)

    print('Press Ctrl+D to exit. (^+D on Mac)')
    while True:
        try:
            en = input('English: ')
        except EOFError:
            break
        x = tokenizer.encode(en).to(transformer.device)
        y_hat = transformer.inference(x)
        pred = tokenizer.decode(y_hat)
        print(f'German: {pred}\n')


if __name__ == '__main__':
    main()
