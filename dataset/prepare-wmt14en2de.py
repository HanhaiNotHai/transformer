import os
import random

from tqdm import tqdm

from config import Config
from tokenizer.tokenizer import Tokenizer

config = Config()

status = os.system('dataset/prepare-wmt14en2de.sh')
if status != 0:
    raise RuntimeError('prepare-wmt14en2de.sh failed')


print('prepare train files...')
src_file = config.train_src
tgt_file = config.train_tgt

with open(src_file) as src_io, open(tgt_file) as tgt_io:
    src = src_io.read().splitlines()
    tgt = tgt_io.read().splitlines()

src_out = []
tgt_out = []

for s, t in zip(src, tgt):
    if s and t and s != '.' and t != ',':
        src_out.append(s)
        tgt_out.append(t)

src = src_out
tgt = tgt_out

while True:
    Tokenizer.train()
    tokenizer = Tokenizer(config, config.train_max_len + 1)

    print('len_before:', len_before := len(src))

    src_out = []
    tgt_out = []

    for s, t in tqdm(zip(src, tgt), total=len(src)):
        s_ids = tokenizer.tokenizer.encode(s).ids
        t_ids = tokenizer.tokenizer.encode(t).ids
        if len(s_ids) <= config.train_max_len and len(t_ids) <= config.train_max_len:
            src_out.append(s)
            tgt_out.append(t)

    src = src_out
    tgt = tgt_out

    print('len_after:', len_after := len(src))
    print('diff:', diff := len_before - len_after)
    if diff == 0:
        break


# 'Sentence pairs were batched together by approximate sequence length' in paper.
# Here sort by sum of the length of the source and target tokens.
get_len = lambda text: len(tokenizer.tokenizer.encode(text).ids)
src, tgt = zip(*sorted(zip(src, tgt), key=lambda st: get_len(st[0]) + get_len(st[1])))

with open(src_file, 'w') as src_io, open(tgt_file, 'w') as tgt_io:
    src_io.write('\n'.join(src))
    tgt_io.write('\n'.join(tgt))


print('prepare test files...')
src_file = config.test_src
tgt_file = config.test_tgt

with open(src_file) as src_io, open(tgt_file) as tgt_io:
    src = src_io.read().splitlines()
    tgt = tgt_io.read().splitlines()

print('len_before:', len_before := len(src))

src_out = []
tgt_out = []

for s, t in tqdm(zip(src, tgt), total=len(src)):
    s_ids = tokenizer.tokenizer.encode(s).ids
    t_ids = tokenizer.tokenizer.encode(t).ids
    if len(s_ids) <= config.train_max_len and len(t_ids) <= config.train_max_len:
        src_out.append(s)
        tgt_out.append(t)

src = src_out
tgt = tgt_out

print('len_after:', len_after := len(src))
print('diff:', diff := len_before - len_after)

src, tgt = zip(*random.sample(list(zip(src, tgt)), config.n_test))

with open(src_file, 'w') as src_io, open(tgt_file, 'w') as tgt_io:
    src_io.write('\n'.join(src))
    tgt_io.write('\n'.join(tgt))
