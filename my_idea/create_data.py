import os
import glob
import torch
import pickle
import argparse
from preprocessing import *

##### args #####
parser = argparse.ArgumentParser(description='Sequence to Sequence Model by using Pytorch')

parser.add_argument('--max_article_len', type=int, default=400,
                    help='max article length')
parser.add_argument('--max_summary_len', type=int, default=100,
                    help='max summary length')
parser.add_argument('--mode', type=str, default="dubug",
                    help='save debug train generate')

args = parser.parse_args()
##### end #####

vocab_path = os.environ['cnn_vocab50000']
preprocess = Preprocess(args.max_article_len, args.max_summary_len)

source_dict = preprocess.getVocab(vocab_path)
target_dict = preprocess.getVocab(vocab_path)

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
debug = False
if args.mode == "train":
    train_src = '{}/{}'.format(pardir, "plain_data/discard_a_train.txt")
    train_tgt = '{}/{}'.format(pardir, "plain_data/discard_s_train.txt")
    print("source data path: {} ".format(train_src))
    print("target data path: {} ".format(train_tgt))
    train_article_file = "data/article.pt"
    train_summary_file = "data/summary.pt"
    preprocess.save(train_src , 0, source_dict, train_article_file, debug)
    preprocess.save(train_tgt , 1, target_dict, train_summary_file, debug)
if args.mode == "val":
    val_src = '{}/{}'.format(pardir, "plain_data/fix_val_article.txt")
    print("source data path: {} ".format(val_src))
    val_article_file = "data/val_article.pt"
    preprocess.save(val_src , 0, source_dict, val_article_file, debug)
if args.mode == "test":
    test_src = '{}/{}'.format(pardir, "plain_data/fix_test_article.txt")
    print("source data path: {} ".format(test_src))
    test_article_file = "data/test_article.pt"
    preprocess.save(test_src , 0, source_dict, test_article_file, debug)
