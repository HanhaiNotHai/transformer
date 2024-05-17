# Implement and optimize the original Transformer

# Environment

You can create a conda environment with `conda env create -f environment.yml` or following commands.

```shell
conda create -n transformer python=3.12
conda activate transformer
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tokenizers evaluate sacrebleu wandb
```

# Inference

Download the [checkpoint](https://github.com/HanhaiNotHai/transformer/releases/download/v1.0/checkpoint.tar.xz) and `tar Jxvf checkpoint.tar.xz`.

`python inference.py` to inference the test dataset.

`python translate.py` to input your own sentence and translate.

# Train

You can `python -m dataset.prepare-wmt14en2de` to build your own tokenizer and datasets or download the [dataset](https://github.com/HanhaiNotHai/transformer/releases/download/v1.0/dataset.tar.xz) and `tar Jxvf dataset.tar.xz` to get my train dataset.

Sentence pairs in the train dataset are sorted by sum of the length of the source and target tokens.

`python train.py` to train the Transformer.
