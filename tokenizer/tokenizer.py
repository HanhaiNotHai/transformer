import torch
from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import TemplateProcessing
from torch import Tensor

from config import Config


class Tokenizer:

    def __init__(self, config: Config, max_len: int = None) -> None:
        '''
        Encoding process is as follows.
        encode -> truncate -> post process -> pad
        '''

        self.tokenizer = CharBPETokenizer.from_file(config.vocab_filename, config.merges_filename)

        if max_len is not None:
            self.tokenizer.enable_truncation(max_len)
        else:
            self.tokenizer.enable_truncation(config.train_max_len)

        # Add BOS token to the begin of sequence and EOS token to the end of sequence.
        self.tokenizer.post_processor = TemplateProcessing(
            single=f'{config.bos_token} $A {config.eos_token}',
            special_tokens=[
                (config.bos_token, config.bos_id),
                (config.eos_token, config.eos_id),
            ],
        )

        self.tokenizer.enable_padding(pad_id=config.pad_id, pad_token=config.pad_token)

    @classmethod
    def train(self) -> None:
        tokenizer = CharBPETokenizer()
        tokenizer.train(
            [Config.train_src, Config.train_tgt],
            vocab_size=Config.vocab_size,
            special_tokens=Config.special_tokens,
        )
        tokenizer.save_model(Config.tokenizer_dir)

    def encode_batch(self, inputs: list[str]) -> tuple[Tensor, Tensor]:
        encodings = self.tokenizer.encode_batch(inputs)
        tokens = torch.tensor([encoding.ids for encoding in encodings])
        pad_mask = torch.tensor([encoding.attention_mask for encoding in encodings]).bool()
        return tokens, pad_mask

    def encode(self, sequence: str) -> Tensor:
        encoding = self.tokenizer.encode(sequence)
        tokens = torch.tensor(encoding.ids)
        return tokens

    def decode(self, y_hat: Tensor) -> str:
        return self.tokenizer.decode(y_hat.tolist())


if __name__ == '__main__':
    Tokenizer.train()
