from define import *
from encoder import *
from decoder import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, target_size, opts):
        super(EncoderDecoder, self).__init__()
        self.opts = opts
        self.encoder = Encoder(source_size, self.opts)
        self.decoder = Decoder(target_size, self.opts)

    def forward(self, article_docs=None, summary_docs=None, train=False, generate=False):
        if train:
            loss = 0
            hx, cx = self.encoder(article_docs)
            summary_docs = summary_docs.t()

            for words_f, words_t in zip(summary_docs[:-1], summary_docs[1:]):
                hx, cx = self.decoder(words_f, hx, cx)
                loss += F.cross_entropy(
                   self.decoder.linear(hx), words_t , ignore_index=0)
            return loss

        elif generate:
            hx, cx = self.encoder(article_docs)
            word_id = torch.tensor( [ target_dict["[START]"] ] ).cuda()
            doc = []
            loop = 0
            while True:
                hx, cx = self.decoder(words_f, hx, cx)
                word_id = torch.tensor([ torch.argmax(
                        F.softmax(self.decoder.linear(final_dist), dim=1).data[0]) ]).cuda()
                loop += 1
                if loop >= 200 or int(word_id) == target_dict['[STOP]']:
                    break
                doc.append(word_id.item())
            return doc
