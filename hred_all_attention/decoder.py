from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class WordDecoder(nn.Module):
    def __init__(self):
        super(WordDecoder, self).__init__()
        self.embed = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, summary_words, w_hx, w_cx):
        embed = self.embed(summary_words)
        w_hx, w_cx = self.lstm(embed, (w_hx, w_cx) )
        return w_hx, w_cx

class SentenceDecoder(nn.Module):
    def __init__(self, opts):
        super(SentenceDecoder, self).__init__()
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.attention = Attention(opts)

    def forward(self, w_hx, s_hx, s_cx, encoder_outputs, encoder_features, coverage_vector, mask_tensor):
        s_hx, s_cx = self.lstm(w_hx, (s_hx, s_cx) )
        final_dist, align_weight, next_coverage_vector = self.attention(
                s_hx, encoder_outputs, encoder_features, coverage_vector, mask_tensor)
        return final_dist, s_hx ,s_cx, align_weight, next_coverage_vector

class Attention(nn.Module):
    def __init__(self, opts):
        super(Attention, self).__init__()
        self.opts = opts
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.coverage = Coverage()

    def forward(self, decoder_hx, encoder_outputs ,encoder_feature, coverage_vector, mask_tensor):
        '''
            encoder_outputs = sen_len x batch x hidden
        '''
        t_k, b, n = list(encoder_outputs.size())
        dec_feature = self.W_s(decoder_hx)
        dec_feature = dec_feature.unsqueeze(0).expand(t_k, b, n)
        att_features = encoder_feature + dec_feature
        if self.opts["coverage_vector"]:
            att_features = self.coverage.getFeature(coverage_vector, att_features)
        e = torch.tanh(att_features)
        scores = self.v(e)
        align_weight = torch.softmax(scores, dim=0) * mask_tensor # sen_len x Batch x 1
        if self.opts["coverage_vector"]:
            next_coverage_vector = self.coverage.getNextCoverage(coverage_vector, align_weight)
        else:
            next_coverage_vector = coverage_vector

        content_vector = (align_weight * encoder_outputs).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        final_dist = torch.tanh(self.linear(concat))
        return final_dist, align_weight, next_coverage_vector

class Coverage(nn.Module):
    def __init__(self):
        super(Coverage, self).__init__()
        self.W_c = nn.Linear(1, hidden_size)

    def getFeature(self, coverage_vector, att_features):
        coverage_input = coverage_vector.view(-1, 1)
        coverage_features = self.W_c(coverage_input).unsqueeze(-1)
        coverage_features = coverage_features.view(-1, att_features.size(1), hidden_size)
        att_features += coverage_features
        return att_features

    def getNextCoverage(self, coverage_vector, align_weight):
        next_coverage_vector = coverage_vector + align_weight
        return next_coverage_vector
