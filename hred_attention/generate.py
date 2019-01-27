import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define import *
from loader import *
from generate_util import *

class GenerateDoc():
    def __init__(self, aritcle_data):
        decode_set = EvaluateDataset(aritcle_data)
        self.decode_iter = DataLoader(decode_set, batch_size=1, collate_fn=decode_set.collater)
        self.GenerateUtil = GenerateUtil(target_dict)

    def generate(self, generate_dir, model=False):
        model.eval()

        for index, iters in enumerate(self.decode_iter):
            doc = model(articles_sentences=iters, generate=True)
            doc = self.GenerateUtil.TranslateDoc(doc)
            doc = ' '.join(doc)
            doc = ' '.join(doc.split()[:args.max_summary_len])
            with open('{}/{:0=5}.txt'.format(generate_dir, index), mode='w') as f:
                f.write(doc)
