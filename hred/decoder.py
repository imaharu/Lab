from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class WordDecoder(nn.Module):
    def __init__(self):
        super(WordDecoder, self).__init__()
        self.embed = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.drop = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, summary_words, w_hx, w_cx):
        embed = self.embed(summary_words)
        embed = self.drop(embed)
        '''
            0があるときは、where
        '''
        if torch.nonzero(summary_words.eq(0)).size(0):
            before_w_hx , before_w_cx = w_hx, w_cx
            mask = torch.cat( [ summary_words.unsqueeze(-1) ] * hidden_size, 1)
            w_hx, w_cx = self.lstm(embed, (w_hx, w_cx) )
            w_hx = torch.where(mask == 0, before_w_hx, w_hx)
            w_cx = torch.where(mask == 0, before_w_cx, w_cx)
        else:
            w_hx, w_cx = self.lstm(embed, (w_hx, w_cx) )
        return w_hx, w_cx

class SentenceDecoder(nn.Module):
    def __init__(self):
        super(SentenceDecoder, self).__init__()
        self.drop = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.attention = Attention()

    def forward(self, w_hx, s_hx, s_cx, encoder_outputs, encoder_features, mask_tensor):
        w_hx = self.drop(w_hx)
        s_hx, s_cx = self.lstm(w_hx, (s_hx, s_cx) )
        final_dist = self.attention(
                s_hx, encoder_outputs, encoder_features, mask_tensor)
        return final_dist, s_hx ,s_cx

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_hx, encoder_outputs , encoder_feature , mask_tensor):
        '''
            encoder_outputs = sen_len x batch x hidden
        '''
        t_k, b, n = list(encoder_outputs.size())
        dec_feature = self.W_s(decoder_hx)
        dec_feature = dec_feature.unsqueeze(0).expand(t_k, b, n)
        att_features = encoder_feature + dec_feature
        e = torch.tanh(att_features)
        scores = self.v(e)
        attn_dist = torch.softmax(scores, dim=0) * mask_tensor # sen_len x Batch x 1
        content_vector = (attn_dist * encoder_outputs).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        final_dist = torch.tanh(self.linear(concat))
        return final_dist
