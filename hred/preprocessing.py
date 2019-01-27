import os
import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import copy
import re

PADDING = 0
UNK = 1
START_DECODING = 2
STOP_DECODING = 3
EOD = 4
class Preprocess():
    def __init__(self, max_article_len=False, max_summary_len=False):
        self.init_dict = {"[PAD]": PADDING ,"[UNK]": UNK, "[START]": START_DECODING, "[STOP]": STOP_DECODING, "[EOD]": EOD}
        if max_article_len:
            self.max_article_len = max_article_len
            self.max_summary_len = max_summary_len

    def getVocab(self, vocab_file):
        return self.pushVocab(vocab_file)

    def pushVocab(self, vocab_file):
        vocab_dict = copy.copy(self.init_dict)
        with open(vocab_file) as f:
            for count, vocab in enumerate(f):
                if vocab not in vocab_dict:
                    vocab_dict[vocab.strip()] = len(vocab_dict)
                '''
                    50000 + 1 because [EOD] is added
                '''
                if len(vocab_dict) >= 50001:
                    break
        return vocab_dict

    def load(self, save_file):
        return torch.load(save_file)

    def save(self, data_path, mode, vocab_dict, save_file, debug=False):
        self.dict = vocab_dict
        with open(data_path) as data:
            if debug:
                tensor_data = [ self.ConvertTensor(doc, mode) for count , doc in enumerate(data) if count < 10]
            else:
                tensor_data = [ self.ConvertTensor(doc, mode) for count , doc in enumerate(data)]
        torch.save(tensor_data, save_file)

    def ConvertTensor(self, doc, mode):
        '''
            mode : 0 -> source
            mode : 1 -> target
        '''
        self.mode = mode
        if self.mode == 1:
            doc, max_summary_len = self.RemoveT(doc, self.max_summary_len)
            summaries = doc.strip().split(' ')[:max_summary_len]
            summaries = " ".join(summaries)
            summaries = summaries.strip().split('</t>')
            filter_summaries = list(filter(lambda summary: summary != "", summaries))
            summaries = [ ["[START]"] +  summary.strip().split(' ') +  ["[STOP]"] for summary in filter_summaries ]
            summaries.append(["[START]"] + ["[EOD]"])
            tensor_ids = self.DocToID(summaries)
        else:
            articles = doc.strip().split(' ')[:self.max_article_len]
            articles = [ str(self.dict[word]) if word in self.dict else str(UNK) for word in " ".join(articles).split(' ')]
            last_word = articles[len(articles) - 1]
            articles = " ".join(articles).split(" " + str(self.dict["."]) + " ")
            articles_len = len(articles)
            if last_word == str(self.dict["."]):
                articles = [ articles[index] + " " + str(self.dict["."]) for index in range(articles_len)]
            else:
                articles = [ articles[index] + " " + str(self.dict["."]) for index in range(articles_len)]
                last_sentence_len = len(articles[len(articles) - 1])
                articles[articles_len - 1] = articles[articles_len - 1][:-2]

            articles.append(str(EOD))
            articles = [ article.strip().split(' ') for article in articles ]
            tensor_ids = self.AleadyID(articles)
        return tensor_ids

    def RemoveT(self, doc, max_summary_len):
        doc = doc.replace("<t>", "")
        max_summary_len = max_summary_len + doc.count("</t>")
        return doc, max_summary_len

    def AleadyID(self, doc):
        doc_list = []
        for sentence in doc:
            doc_list.append(torch.tensor([int(word) for word in sentence]))
        return doc_list

    def DocToID(self, doc):
        doc_list = []
        for sentence in doc:
            slist = []
            for word in sentence:
                if word in self.dict:
                    slist.append(self.dict[word])
                else:
                    slist.append(UNK)
            doc_list.append(torch.tensor(slist))
        return doc_list
