from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class WordEncoder(nn.Module):
    def __init__(self, opts):
        super(WordEncoder, self).__init__()
        self.opts = opts
        self.embed = nn.Embedding(source_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])

    def forward(self, sentences):
        input_lengths = sentences.ne(0).sum(-1)
        embed = self.embed(sentences)
        sequence = rnn.pack_padded_sequence(embed, input_lengths, batch_first=True)
        _, (w_hx, w_cx) = self.lstm(sequence)

        if self.opts["bidirectional"]:
            w_hx = w_hx.view(-1, 2 , sentences.size(0), hidden_size).sum(1)
            w_cx = w_cx.view(-1, 2 , sentences.size(0), hidden_size).sum(1)
        w_hx = w_hx.view(sentences.size(0) , -1)
        w_cx = w_cx.view( sentences.size(0) , -1)
        return w_hx, w_cx

class SentenceEncoder(nn.Module):
    def __init__(self, opts):
        super(SentenceEncoder, self).__init__()
        self.opts = opts
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, words_encoder_outputs):
        sentence_outputs, (s_hx, s_cx) = self.lstm(words_encoder_outputs)
        if self.opts["bidirectional"]:
            s_hx = s_hx.view(-1, 2 , words_encoder_outputs.size(1), hidden_size).sum(1)
            s_cx = s_cx.view(-1, 2 , words_encoder_outputs.size(1), hidden_size).sum(1)
        s_hx = s_hx.view(words_encoder_outputs.size(1) , -1)
        s_cx = s_cx.view(words_encoder_outputs.size(1) , -1)
        return s_hx, s_cx
