import os
import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

PADDING = 0
UNK = 1
BOS = 2
EOS = 3

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

class Word_Data():
    def __init__(self):
        self.vocab_file = os.environ['cnn_vocab50000']
        self.dict = {"[UNK]": UNK, "[BOS]": BOS, "[EOS]": EOS}

        pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_src = '{}/{}'.format(pardir, "plain_data/train.txt.src")
        train_tgt = '{}/{}'.format(pardir, "plain_data/train.txt.tgt.tagged")
        self.article_path = train_src
        self.summary_path = train_tgt

    def getVocabSize(self):
        self.pushVocab()
        return len(self.dict)

    def save(self, article_file , summary_file):
        self.pushVocab()
        self.SaveTensorData(self.article_path, article_file)
        self.SaveTensorData(self.summary_path, summary_file)

    def pushVocab(self):
        with open(self.vocab_file) as f:
            for count, vocab in enumerate(f):
                self.dict[vocab.strip()] = len(self.dict) + 1

    def SaveTensorData(self, path, file_name):
        tensor_data = self.GetTensorData(path)
        torch.save(tensor_data, file_name)

    def GetTensorData(self, file_path):
        with open(file_path) as f:
            tensor_data = [ self.ConvertTensor(doc) for i, doc in enumerate(f)]
        return tensor_data

    def ConvertTensor(self, doc):
        doc = self.replaceWord(doc)
        words = self.DocToWord(doc)
        words_id = self.SentenceToDictID(words)
        return words_id

    def replaceWord(self, doc):
        doc = doc.replace("<t>", "")
        doc = doc.replace("</t>", "")
        return doc

    def DocToWord(self, strs):
        return strs.strip().split(' ')

    def SentenceToDictID(self, sentence):
        slist = []
        for word in sentence:
            if word in self.dict:
                slist.append(self.dict[word])
            else:
                slist.append(UNK)
        return torch.tensor(slist)
