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
        self.drop = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])

    def forward(self, sentences):
        '''
            return
                w_hx, w_cx
            option
                bidirectional
        '''
        input_lengths = torch.tensor(
            [seq.size(-1) for seq in sentences])
        embed = self.embed(sentences)
        embed = self.drop(embed)
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
        self.drop = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, words_encoder_outputs):
        '''
            return
                encoder_ouput, hx, cx
            option
                bidirectional
        '''
        words_encoder_outs = self.drop(words_encoder_outputs)
        # need where
        sentence_outputs, (s_hx, s_cx) = self.lstm(words_encoder_outputs)
        if self.opts["bidirectional"]:
            sentence_outputs = sentence_outputs[:, :, :hidden_size] + sentence_outputs[:, :, hidden_size:]
            s_hx = s_hx.view(-1, 2 , words_encoder_outputs.size(1), hidden_size).sum(1)
            s_cx = s_cx.view(-1, 2 , words_encoder_outputs.size(1), hidden_size).sum(1)
        sentence_features = self.W_h(sentence_outputs)
        s_hx = s_hx.view(words_encoder_outputs.size(1) , -1)
        s_cx = s_cx.view(words_encoder_outputs.size(1) , -1)
        return sentence_outputs, sentence_features, s_hx, s_cx
