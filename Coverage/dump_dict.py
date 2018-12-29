from create_sin_dict import *

# Other
import os
import glob
import pickle

english_vocab = {"[PAD]": 0, "[UNK]" : 1, "[SEOS]": 2, "[TEOS]": 3 , "[BOS]" : 4, "[EOD]": 5}
def create_dict(vocab_file, vocab):
    with open(vocab_file) as f:
        for word in f:
            vocab[word.strip()] = len(vocab)
    return vocab
vocab_file = "/home/ochi/Lab/CNN_STORY/finished_files/vocab50000"
english_vocab = create_dict(vocab_file, english_vocab)
with open('cnn.dump', 'wb') as f:
    pickle.dump(english_vocab, f)
