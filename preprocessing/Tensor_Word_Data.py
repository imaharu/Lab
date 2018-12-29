import os
import tensor

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_src = '{}/{}'.format(pardir, "plain_data/train.txt.src")
train_tgt = '{}/{}'.format(pardir, "plain_data/train.txt.tgt")

PADDING = 0
UNK = 1
EOS = 2
BOS = 3

