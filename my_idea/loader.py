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

    def __getitem__(self, index):
        get_source = self.source[index]
        get_target = self.target[index]
        return [get_source, get_target]

    def __len__(self):
        return len(self.source)

    def GetSentencePadding(self, docs, max_len, max_sentence_len):
        final_docs = []
        for index, doc in enumerate(docs):
            n_sentence, n_word = doc.size(0), doc.size(1)
            if n_sentence != max_len:
                mask = torch.zeros([ max_len - n_sentence, n_word ], dtype=torch.int64)
                doc = torch.cat((doc, mask), 0)
            if n_word != max_sentence_len:
                doc = torch.nn.functional.pad(doc, (0, max_sentence_len - n_word), "constant", 0)
            final_docs.append(doc)
        final_docs = torch.stack(final_docs)
        return final_docs.permute(1,0,2)

    def collater(self, items):
        articles = [item[0] for item in items]
        summaries = [item[1] for item in items]
        max_article_len = max([ article.size(0) for article in articles ])
        max_article_sentence_len = max([ article.size(1) for article in articles ])
        max_summary_len = max([ summary.size(0) for summary in summaries ])
        max_summary_sentence_len = max([ summary.size(1) for summary in summaries ])
        articles_sentences_padding = self.GetSentencePadding(articles, max_article_len, max_article_sentence_len)
        summaries_sentences_padding = self.GetSentencePadding(summaries, max_summary_len, max_summary_sentence_len)
        return [articles_sentences_padding, summaries_sentences_padding]

class EvaluateDataset(Dataset):
    def __init__(self, source):
        self.source = source

    def __getitem__(self, index):
        get_source = self.source[index]
        return get_source

    def __len__(self):
        return len(self.source)

    def collater(self, items):
        return items[0].unsqueeze(1).cuda()
