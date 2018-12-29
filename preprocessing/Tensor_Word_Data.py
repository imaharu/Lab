import os
import torch
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_src = '{}/{}'.format(pardir, "plain_data/train.txt.src")
train_tgt = '{}/{}'.format(pardir, "plain_data/train.txt.tgt.tagged")

PADDING = 0
UNK = 1

class Word_Data():
    def __init__(self):
        self.vocab_file = os.environ['cnn_vocab50000']
        self.dict = {"[UNK]": UNK}

    def exec(self, article_path ,summary_path):
        self.pushVocab()
        self.article_path = article_path
        self.summary_path = summary_path
        self.SaveTensorData(self.article_path, "article10.pt")
        self.SaveTensorData(self.summary_path, "summary10.pt")

    def pushVocab(self):
        with open(self.vocab_file) as f:
            for count, vocab in enumerate(f):
                self.dict[vocab.strip()] = len(self.dict) + 1

    def SaveTensorData(self, path, file_name):
        tensor_data = self.GetTensorData(path)
        torch.save(tensor_data, file_name)

    def GetTensorData(self, file_path):
        with open(file_path) as f:
            tensor_data = [ self.ConvertTensor(doc) for i, doc in enumerate(f) if i < 10]
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

word_data = Word_Data()
word_data.exec(train_src, train_tgt)
