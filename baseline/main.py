import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

# model
from model import *

# my function
from create_dict import *

# hyperparameter
from define import *

# Other
import time

def train(model, source_doc, target_doc):
    loss = 0
    loss = torch.mean(torch.unsqueeze(model(source_doc, target_doc), 0))
    return loss

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
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
    model = HierachicalEncoderDecoder(source_size, target_size, hidden_size)
    model = nn.DataParallel(model).to(device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    optimizer = torch.optim.Adam( model.parameters(), weight_decay=args.weightdecay)

    import os
    path = os.path.dirname(os.getcwd())
    article_data = torch.load(path + "/preprocessing/article.pt")
    summary_data = torch.load(path + "/preprocessing/summary.pt")
    a = MyDataset(article_data, summary_data)
    train_iter = DataLoader(a, batch_size=4, collate_fn=a.collater, shuffle=True)

    for iters in train_iter:
        exit()
    indexes = torch.randperm(len(article_data))
    print(torch.tensor(article_data).size())

#    for epoch in range(args.epoch):
#        target_docs = []
#        source_docs = []
#        print("epoch",epoch + 1)
#        indexes = torch.randperm(train_doc_num)
#        for i in range(0, train_doc_num, batch_size):
#            source_docs = [ get_source_doc(english_paths[doc_num], english_vocab) for doc_num in indexes[i:i+batch_size]]
#            target_docs = [ get_target_doc(english_paths[doc_num], english_vocab) for doc_num in indexes[i:i+batch_size]]
#            # source_docs
#            max_doc_sentence_num =  max([*map(lambda x: len(x), source_docs )])
#            source_docs = [ [ s + [ english_vocab["[SEOS]"] ] for s in t_d ] for t_d in source_docs]
#            source_spadding = sentence_padding(source_docs, max_doc_sentence_num)
#            source_wpadding = word_padding(source_spadding, max_doc_sentence_num)
#            for source in source_wpadding:
#                source.append([ english_vocab["[EOD]"] ])
#
#            max_doc_target_num =  max([*map(lambda x: len(x), target_docs )])
#            # add <teos> to target_docs
#
#            target_docs = [ [ [english_vocab["[BOS]"]] + s + [ english_vocab["[TEOS]"] ] for s in t_d ] for t_d in target_docs]
#            target_spadding = sentence_padding(target_docs, max_doc_target_num)
#            target_wpadding = word_padding(target_spadding, max_doc_target_num)
#            for target in target_wpadding:
#                target.extend([ [english_vocab["[BOS]"] ,  english_vocab["[EOD]"]  ] ] )
#
#            optimizer.zero_grad()
#            loss = train(model, source_wpadding,target_wpadding)
#            loss.backward()
#            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
#            optimizer.step()
#
#        if (epoch + 1)  % 5 == 0 or epoch == 0:
#            outfile = "trained_model/" + str(args.save_path) \
#                + "-epoch-" + str(epoch + 1) +  ".model"
#            torch.save(model.state_dict(), outfile)
#        elapsed_time = time.time() - start
#        print("時間:",elapsed_time / 60.0, "分")
