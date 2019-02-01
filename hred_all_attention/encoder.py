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
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])

    def forward(self, sentences):
        except_flag = False
        b = sentences.size(0)

        input_lengths = sentences.ne(0).sum(-1)
        sorted_lengths, indices = torch.sort(input_lengths, descending=True)
        sentences = sentences[indices]
        masked_select = sorted_lengths.masked_select(sorted_lengths.ne(0))

        # if all 0 sentence is appeared
        if not torch.equal(sorted_lengths, masked_select):
            s_b = b
            b = masked_select.size(0)
            sentences = sentences.narrow(0, 0, b)
            sorted_lengths = masked_select
            except_flag = True

        embed = self.embed(sentences)
        sequence = rnn.pack_padded_sequence(embed, sorted_lengths, batch_first=True)
        _, w_hx = self.gru(sequence)

        if self.opts["bidirectional"]:
            w_hx = w_hx.view(-1, 2 , b, hidden_size).sum(1)
        w_hx = w_hx.view(b , -1)

        if except_flag:
            zeros = torch.zeros((s_b - b, hidden_size)).cuda()
            w_hx = torch.cat((w_hx, zeros), 0)

        inverse_indices = indices.sort()[1] # Inverse permutation
        w_hx = w_hx[inverse_indices]
        return w_hx

class SentenceEncoder(nn.Module):
    def __init__(self, opts):
        super(SentenceEncoder, self).__init__()
        self.opts = opts
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, words_encoder_outputs):
        b = words_encoder_outputs.size(1)
        input_sentences = words_encoder_outputs.transpose(0,1)
        input_lengths = input_sentences.ne(0).sum(1).t()[0]

        sorted_lengths, indices = torch.sort(input_lengths, descending=True)
        input_sentences = input_sentences[indices]

        sequence = rnn.pack_padded_sequence(input_sentences, sorted_lengths, batch_first=True)
        sentence_outputs, s_hx = self.gru(sequence)
        sentence_outputs, _ = rnn.pad_packed_sequence(
            sentence_outputs
        )

        inverse_indices = indices.sort()[1] # Inverse permutation
        sentence_outputs = sentence_outputs[:, inverse_indices]
        s_hx = s_hx[:, inverse_indices]

        if self.opts["bidirectional"]:
            sentence_outputs = sentence_outputs[:, :, :hidden_size] + sentence_outputs[:, :, hidden_size:]
            s_hx = s_hx.view(-1, 2 , b, hidden_size).sum(1)
        sentence_features = self.W_h(sentence_outputs)
        s_hx = s_hx.view(b, -1)
        return sentence_outputs, sentence_features, s_hx
