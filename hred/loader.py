import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import itertools
class MyDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.padding = [ torch.tensor([0]) for _ in range(100) ]

    def __getitem__(self, index):
        get_source = self.source[index]
        get_target = self.target[index]
        return [get_source, get_target]

    def __len__(self):
        return len(self.source)

    def GetSentencePadding(self, datas, max_len):
        for index, data in enumerate(datas):
            tmp = self.padding[0:max_len]
            tmp[:len(data)] = data
            datas[index] = tmp

        chunk_sentences = [ [ datas[doc_id][index] for doc_id in range(len(datas)) ]
            for index in range(max_len) ]

        sentences_padding = [ pad_sequence(sentences , batch_first=True)
            for sentences in chunk_sentences ]

        return sentences_padding

    def collater(self, items):
        articles = [item[0] for item in items]
        summaries = [item[1] for item in items]
        max_article_len = max([ len(article) for article in articles ])
        max_summary_len = max([ len(summary) for summary in summaries ])
        articles_sentences_padding = self.GetSentencePadding(articles, max_article_len)
        summaries_sentences_padding = self.GetSentencePadding(summaries, max_summary_len)

        return [articles_sentences_padding, summaries_sentences_padding]

class EvaluateDataset(Dataset):
    def __init__(self, source):
        self.source = source

    def __getitem__(self, index):
        get_source = self.source[index]
        return [get_source]

    def __len__(self):
        return len(self.source)

    def collater(self, items):
        source_items = [item[0] for item in items]
        source_items.sort(key=lambda x: len(x), reverse=True)
        source_padding = pad_sequence(source_items, batch_first=True)
        return source_padding
