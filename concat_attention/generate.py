import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define import *
from loader import *
from decode_util import *

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = EncoderDecoder().to(device)
    checkpoint = torch.load("trained_model/{}".format(str(args.model_path)))
    model.load_state_dict(checkpoint)
    model.eval()

    decode_set = DecodeDataset(article_val_data)
    decode_iter = DataLoader(decode_set, batch_size=1, collate_fn=decode_set.collater)
    Evaluate = Evaluate(target_dict)
    generate_dir = "trained_model/{}".format(str(args.result_path))

    if not os.path.exists(generate_dir):
        os.mkdir(generate_dir)

    for index, iters in enumerate(decode_iter):
        doc = model(source=iters.cuda(), generate=True)
        doc = Evaluate.TranslateSentence(doc)
        doc = ' '.join(doc)
        with open('{}/{:0=5}.txt'.format(generate_dir, index), mode='w') as f:
            f.write(doc)
