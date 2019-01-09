import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
class MyDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.article_padding = [ torch.zeros(1) for _ in range(30) ]
        self.summary_padding = [ torch.zeros(1) for _ in range(15) ]

    def __getitem__(self, index):
        get_source = self.source[index]
        get_target = self.target[index]
        return [get_source, get_target]

    def __len__(self):
        return len(self.source)

    def collater(self, items):
        articles = [item[0] for item in items]
        summaries = [item[1] for item in items]

        max_article_sentence_len = max([ len(article) for article in articles ])
        max_summary_sentence_len = max([ len(summary) for summary in summaries ])

        for index, article in enumerate(articles):
            if len(article) < max_article_sentence_len:
                tmp = self.article_padding[0:max_article_sentence_len]
                tmp[:len(article)] = article
                articles[index] = tmp
        for index, summary in enumerate(summaries):
            if len(summary) < max_summary_sentence_len:
                tmp = self.summary_padding[0:max_summary_sentence_len]
                tmp[:len(summary)] = summary
                summaries[index] = tmp
        ## need sort?
        articles_chunk_sentences = [ [ articles[doc_id][article_index] for doc_id in range(2) ]
                                        for article_index  in range(max_article_sentence_len) ]
        articles_sentences_padding = [ pad_sequence(article_sentences , batch_first=True, padding_value=0)
                                    for article_sentences in articles_chunk_sentences ]

        summaries_chunk_sentences = [ [ summaries[doc_id][sentence_index] for doc_id in range(2) ]
                                        for sentence_index  in range(max_summary_sentence_len) ]
        summaries_sentences_padding = [ pad_sequence(summary_sentences , batch_first=True)
                                    for summary_sentences in summaries_chunk_sentences ]

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
