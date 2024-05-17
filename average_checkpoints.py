import os

from transformer.config import Config
from transformer.transformer import Transformer

CHECKPINTS_DIR = 'checkpoint/0512_00:46:30/'

config = Config()
transformer = Transformer(**config.model_config)

checkpoints = os.listdir(CHECKPINTS_DIR)
checkpoints = [CHECKPINTS_DIR + ckpt for ckpt in checkpoints]
transformers = [Transformer(**config.model_config, ckpt_path=ckpt) for ckpt in checkpoints]
l = len(transformers)
parameters = [transformer.parameters() for transformer in transformers]
parameters = list(zip(*parameters))

for i, p in enumerate(transformer.parameters()):
    p = sum(parameters[i]) / l

transformer.save(CHECKPINTS_DIR + 'transformer.ckpt')
