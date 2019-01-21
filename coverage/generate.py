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
        decode_set = DecodeDataset(aritcle_data)
        self.decode_iter = DataLoader(decode_set, batch_size=1, collate_fn=decode_set.collater)
        self.GenerateUtil = GenerateUtil(target_dict)

    def generate(self, generate_dir, model=False, model_path=False,is_checkpoint=False):
        if is_checkpoint:
            device = torch.device('cuda:0')
            opts = { "bidirectional" : args.none_bid }
            model = EncoderDecoder(source_size, target_size, opts).cuda(device=device)
            checkpoint = torch.load("trained_model/{}".format(str(args.model_path)))
            model.load_state_dict(checkpoint)

        model.eval()

        for index, iters in enumerate(self.decode_iter):
            doc = model(article_docs=iters.cuda(), generate=True)
            doc = self.GenerateUtil.TranslateDoc(doc)
            doc = ' '.join(doc)
            with open('{}/{:0=5}.txt'.format(generate_dir, index), mode='w') as f:
                f.write(doc)

if __name__ == '__main__':
    save_dir = "{}/{}".format("trained_model", args.save_dir)
    generate_dir = "{}/{}".format(save_dir , args.generate_dir)

    device = torch.device('cuda:0')
    opts = { "bidirectional" : args.none_bid, "coverage_vector": args.coverage }
    model = EncoderDecoder(source_size, target_size, opts).cuda(device=device)
    checkpoint = torch.load("{}/{}".format(save_dir ,str(args.model_path)))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.Adagrad( model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    generate_module = GenerateDoc(article_val_data)
    generate_module.generate(generate_dir, model=model)
