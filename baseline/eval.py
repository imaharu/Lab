import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define import *

if __name__ == '__main__':
    device = torch.device('cuda:0')
    print(article_data)
    exit()
    opts = { "bidirectional" : args.none_bid, "coverage_vector": args.coverage }
    model = EncoderDecoder().to(device)
    checkpoint = torch.load("trained_model/{}".format(str(args.model_path)))
    model.load_state_dict(checkpoint)
    model.eval()

    data_set = EvaluateDataset(test_source)
    eval_iter = DataLoader(data_set, batch_size=1, collate_fn=data_set.collater)

    Evaluate = Evaluate(target_dict)
    pred_file = open("trained_model/{}".format(str(args.result_path)), 'w', encoding="utf-8")
    for iters in eval_iter:
        pred = model(source=iters.cuda(), generate=True)
        sentence = Evaluate.TranslateSentence(pred)
        sentence = ' '.join(sentence)
        pred_file.write(sentence + '\n')
    pred_file.close
