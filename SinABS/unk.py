# my function
from create_sin_dict import *

# Other
import os
import glob
import sys
from collections import Counter
argvs = sys.argv
c_vocab = Counter()
train_doc_num = int(argvs[1]) # defalut 20000
co_num = int(argvs[2]) # defalut 30000
data_path = os.environ["cnn_data"]

english_paths = sorted(glob.glob(data_path + "/*.story"))[0:train_doc_num]

def count_dict(language_files, c_vocab):
    for filename in language_files:
        story_lines = [ line.split() for line in separate_source_data(filename) ]
        highlights_lines = [ line.split() for line in separate_target_data(filename) ]
        # story
        for lines in story_lines:
            for word in lines:
                if word in c_vocab:
                    c_vocab[word] += 1
                else:
                    c_vocab[word] = 1
        # highlights
        for lines in highlights_lines:
            for word in lines:
                if word in c_vocab:
                    c_vocab[word] += 1
                else:
                    c_vocab[word] = 1

    no_unk_word = c_vocab.most_common(co_num)

count_dict(english_paths, c_vocab)
