''' Pytorch '''
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

''' myfile'''
from model import *
from define import *
from loader import *
from evaluate import *
from generate import *

''' Python '''
import time
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

def train(model, articles, summaries):
    loss = model(article_docs=articles.cuda(), summary_docs=summaries.cuda(), train=True)
    return loss

def save(model, real_epoch, max_rouge, generate_module):

    model_dir = "{}/{}".format("trained_model", args.save_dir)
    generate_dir = "{}/{}".format(model_dir , args.generate_dir)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(generate_dir):
        os.mkdir(generate_dir)

    model_path = "/home/ochi/Lab/gold_summary/val_summaries"

    generate_module.generate(generate_dir, model=model)
    ROUGE1_f1, ROUGE2_f1, ROUGEL_f1 = EvaluateByPyrouge(generate_dir, model_dir)

    if max_rouge["ROUGE1_f1"] < ROUGE1_f1:
        max_rouge["ROUGE1_f1"] = ROUGE1_f1
        ROUGE1_best_model_filename = "{}/ROUGE1_best.model".format(model_dir)
        states = {
            'epoch': real_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, ROUGE1_best_model_filename)

    if (real_epoch) == args.epoch or (real_epoch) % 2 == 0:
        save_model_filename = "{}/epoch-{}.model".format(model_dir, str(real_epoch))
        states = {
            'epoch': real_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_model_filename)
    exit()

if __name__ == '__main__':
    start = time.time()
    max_rouge = {"ROUGE1_f1":0, "ROUGE2_f1":0, "ROUGEL_f1":0}
    device = "cuda:0"

    data_set = MyDataset(article_data, summary_data)
    train_iter = DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collater, shuffle=True)

    opts = { "bidirectional" : args.none_bid }
    model = EncoderDecoder(source_size, target_size, opts).cuda(device=device)

    generate_module = GenerateDoc(article_val_data)

    if args.set_state:
        optimizer = torch.optim.Adagrad( model.parameters(), lr=0.15,  initial_accumulator_value=0.1)
        set_epoch = 0
    else:
        checkpoint = torch.load("trained_model/{}".format(str(args.model_path)))
        epochs -= checkpoint['epoch']
        set_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = torch.optim.Adagrad( model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(max_epoch):
        real_epoch = epoch + set_epoch + 1
        tqdm_desc = "[Epoch{:>3}]".format(epoch)
        tqdm_bar_format = "{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        tqdm_kwargs = {'desc': tqdm_desc, 'smoothing': 0.1, 'ncols': 100,
                    'bar_format': tqdm_bar_format, 'leave': False}

        model.train()
        for iters in tqdm(train_iter, **tqdm_kwargs):
            optimizer.zero_grad()
            loss = train(model, iters[0], iters[1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        #if args.mode == "train":
        save(model, real_epoch, max_rouge, generate_module)

        elapsed_time = time.time() - start
        print("{0.days:02}days{0.hours:02}hours{0.minutes:02}minute{0.seconds:02}second".format(relativedelta(seconds=int(elapsed_time))))
