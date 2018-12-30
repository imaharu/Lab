import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# model
from model import *

# my function
from create_dict import *

# hyperparameter
from define import *

# Other
import time
import os

def train(model, source_doc, target_doc):
    loss = 0
    loss = torch.mean(torch.unsqueeze(model(source_doc, target_doc), 0))
    return loss

class MyDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __getitem__(self, index):
        get_source = self.source[index]
        get_target = self.target[index]
        return [get_source, get_target]

    def __len__(self):
        return len(self.source)

    def collater(self, items):
        source_items = [item[0] for item in items]
        target_items = [item[1] for item in items]
        source_padding = pad_sequence(source_items, batch_first=True)
        target_padding = pad_sequence(target_items, batch_first=True)
        return [source_padding, target_padding]

if __name__ == '__main__':
    start = time.time()
    device = "cuda:0"

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    path = os.path.dirname(os.getcwd())
    #article_data = torch.load(path + "/preprocessing/article50000.pt")
    #summary_data = torch.load(path + "/preprocessing/summary50000.pt")
    article_data = torch.load(path + "/preprocessing/article.pt")
    summary_data = torch.load(path + "/preprocessing/summary.pt")
    data_set = MyDataset(article_data, summary_data)
    train_iter = DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collater, shuffle=True)

    # PADDING UNK
    source_size = 50002
    target_size = 50002

    model = EncoderDecoder(source_size, target_size, hidden_size)
    model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=args.weightdecay)

    for epoch in range(args.epoch):
        flag = 0
        for iters in train_iter:
            optimizer.zero_grad()
            loss = train(model, iters[0], iters[1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
            optimizer.step()

        if (epoch + 1)  % 1 == 0 or epoch == 0:
            outfile = "trained_model/" + str(args.save_path) \
                + "-epoch-" + str(epoch + 1) +  ".model"
            torch.save(model.state_dict(), outfile)
            elapsed_time = time.time() - start
            print("時間:",elapsed_time / 60.0, "分")
