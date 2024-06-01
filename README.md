# Implement and optimize the vanilla Transformer

Transformer 8x67M. Total parameters: 332M. Activated parameters: 105M.

# Features

- Parallelly compute heads and qkv in multi-head attention.

- GQA: grouped-query attention / MQA: multi-query attention

- RoPE: rotary position embedding

- kv_cache

- SwiGLU

- MoE: mixture of experts

- RMSNorm: root mean square layer normalization

# Environment

You can create a conda environment with `conda env create -f environment.yml` or following commands.

```shell
conda create -n transformer python=3.12
conda activate transformer
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tokenizers evaluate sacrebleu wandb
```

# Inference

Download the [checkpoint](https://github.com/HanhaiNotHai/transformer/releases/download/v2.0/checkpoint.tar.xz) and `tar Jxvf checkpoint.tar.xz`.

`python inference.py` to inference the test dataset.

`python translate.py` to input your own sentence and translate.

# Train

You can `python -m dataset.prepare-wmt14en2de` to build your own tokenizer and datasets or download my [dataset](https://github.com/HanhaiNotHai/transformer/releases/download/v2.0/dataset.tar.xz) and `tar Jxvf dataset.tar.xz`.

Sentence pairs in the train dataset are sorted by sum of the length of the source and target tokens.

`python train.py` to train the Transformer.
