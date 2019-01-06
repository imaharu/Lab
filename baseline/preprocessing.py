import os
import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import copy
PADDING = 0
UNK = 1
START_DECODING = 2
STOP_DECODING = 3

class Preprocess():
    def __init__(self, max_article_len, max_summary_len):
        self.init_dict = {"[PAD]": PADDING ,"[UNK]": UNK, "[START]": START_DECODING, "[STOP]": STOP_DECODING}
        self.max_article_len = max_article_len
        self.max_summary_len = max_summary_len

    def getVocab(self, vocab_file):
        return self.pushVocab(vocab_file)

    def pushVocab(self, vocab_file):
        vocab_dict = copy.copy(self.init_dict)
        with open(vocab_file) as f:
            for count, vocab in enumerate(f):
                vocab_dict[vocab.strip()] = len(vocab_dict)
                if len(vocab_dict) >= 50000:
                    break
        return vocab_dict

    def load(self, save_file):
        return torch.load(save_file)

    def save(self, data_path, mode, vocab_dict, save_file, debug=False):
        '''
            mode : 0 -> source
            mode : 1 -> target
        '''
        self.dict = vocab_dict
        with open(data_path) as data:
            if debug:
                tensor_data = [ self.ConvertTensor(doc, mode) for count , doc in enumerate(data) if count <= 20]
            else:
                tensor_data = [ self.ConvertTensor(doc, mode) for count , doc in enumerate(data)]
        torch.save(tensor_data, save_file)

    def ConvertTensor(self, doc, mode):
        '''
            mode : 0 -> source
            mode : 1 -> target
        '''
        doc = self.replaceWord(doc)
        if mode == 1:
            words = doc.strip().split(' ')[:self.max_summary_len]
            words = ["[START]"] + words + ["[STOP]"]
        else:
            words = doc.strip().split(' ')[:self.max_article_len]
        words_id = self.SentenceToDictID(words)
        return words_id

    def replaceWord(self, doc):
        doc = doc.replace("<t>", "")
        doc = doc.replace("</t>", "")
        return doc

    def SentenceToDictID(self, sentence):
        slist = []
        for word in sentence:
            if word in self.dict:
                slist.append(self.dict[word])
            else:
                slist.append(UNK)
        return torch.tensor(slist)
