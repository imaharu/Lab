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

parser.add_argument('--save_article_file', type=str, default="data/debug_article.pt",
                    help='save article file')
parser.add_argument('--save_summary_file', type=str, default="data/debug_summary.pt",
                    help='save article file')
parser.add_argument('--mode', type=str, default="train",
                    help='save debug train generate')
args = parser.parse_args()
##### end #####

vocab_path = os.environ['cnn_vocab50000']
preprocess = Preprocess(args.max_article_len, args.max_summary_len)

source_dict = preprocess.getVocab(vocab_path)
target_dict = preprocess.getVocab(vocab_path)
source_size = len(source_dict)
target_size = len(target_dict)

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
debug = False
if args.mode == "train":
    train_src = '{}/{}'.format(pardir, "plain_data/discard_a_train.txt")
    train_article_file = "data/train_article.pt"
    preprocess.save(train_src , 0, source_dict, train_article_file, debug)
    train_summary_file = "data/train_summary.pt"
    train_tgt = '{}/{}'.format(pardir, "plain_data/discard_s_train.txt")
    preprocess.save(train_tgt , 1, target_dict, train_summary_file, debug)
if args.mode == "val":
    val_src = '{}/{}'.format(pardir, "plain_data/discard/fix_val_article.txt")
    val_article_file = "data/val_article.pt"
    preprocess.save(val_src , 0, source_dict, val_article_file, debug)
if args.mode == "test":
    test_src = '{}/{}'.format(pardir, "plain_data/test.txt.src")
    print("source data path: {} ".format(test_src))
    test_article_file = "data/test_article.pt"
    preprocess.save(test_src , 0, source_dict, test_article_file, debug)
