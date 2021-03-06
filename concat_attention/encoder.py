from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from model_util import *

class Encoder(nn.Module):
    def __init__(self, source_size, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.embed = nn.Embedding(source_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, sentences):
        b = sentences.size(0)
        input_lengths = sentences.ne(0).sum(-1)
        sorted_lengths, indices = torch.sort(input_lengths, descending=True)

        sentences = sentences[indices]

        embed = self.embed(sentences)
        sequence = rnn.pack_padded_sequence(embed, sorted_lengths, batch_first=True)

        packed_output, (hx, cx) = self.lstm(sequence)
        encoder_outputs, _ = rnn.pad_packed_sequence(
            packed_output
        )
        if self.opts["bidirectional"]:
            encoder_outputs = encoder_outputs[:, :, :hidden_size] + encoder_outputs[:, :, hidden_size:]
            hx = hx.view(-1, 2 , b, hidden_size).sum(1)
            cx = cx.view(-1, 2 , b, hidden_size).sum(1)

        inverse_indices = indices.sort()[1] # Inverse permutation
        encoder_outputs = encoder_outputs[:, inverse_indices]

        hx = hx[:, inverse_indices]
        cx = cx[:, inverse_indices]

        encoder_features = self.W_h(encoder_outputs)
        hx = hx.view(b, -1)
        cx = cx.view(b, -1)
        return encoder_outputs, encoder_features, hx, cx
