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
        b = sentences.size(0)
        input_lengths = sentences.ne(0).sum(-1)
        sorted_lengths, indices = torch.sort(input_lengths, descending=True)
        sentences = sentences[indices]
        masked_select = sorted_lengths.masked_select(sorted_lengths.ne(0))
        except_flag = False
        # if all 0 sentence is appeared
        if not torch.equal(sorted_lengths, masked_select):
            s_b = b
            b = masked_select.size(0)
            sentences = sentences.narrow(0, 0, b)
            sorted_lengths = masked_select
            except_flag = True
        embed = self.embed(sentences)
        sequence = rnn.pack_padded_sequence(embed, sorted_lengths, batch_first=True)
        _, (w_hx, w_cx) = self.lstm(sequence)
        if self.opts["bidirectional"]:
            w_hx = w_hx.view(-1, 2 , b, hidden_size).sum(1)
            w_cx = w_cx.view(-1, 2 , b, hidden_size).sum(1)
        w_hx = w_hx.view(b , -1)
        w_cx = w_cx.view(b , -1)

        if except_flag:
            zeros = torch.zeros((s_b - b, hidden_size)).cuda()
            w_hx = torch.cat((w_hx, zeros), 0)
            w_cx = torch.cat((w_cx, zeros), 0)
        inverse_indices = indices.sort()[1] # Inverse permutation
        w_hx = w_hx[inverse_indices]
        w_cx = w_cx[inverse_indices]
        return w_hx, w_cx

class SentenceEncoder(nn.Module):
    def __init__(self, opts):
        super(SentenceEncoder, self).__init__()
        self.opts = opts
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, words_encoder_outputs):
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
