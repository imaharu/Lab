# my function
# Other
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
parser.add_argument('--save_option', type=str, default="train",
                    help='train val test')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
##### end #####

vocab_path = os.environ['cnn_vocab50000']
preprocess = Preprocess(args.max_article_len, args.max_summary_len)

source_dict = preprocess.getVocab(vocab_path)
target_dict = preprocess.getVocab(vocab_path)

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if args.save_option == "train":
    train_src = '{}/{}'.format(pardir, "plain_data/train.txt.src")
    train_tgt = '{}/{}'.format(pardir, "plain_data/train.txt.tgt.tagged")
    print("source data path: {} ".format(train_src))
    print("target data path: {} ".format(train_tgt))
    train_source = preprocess.save(train_src , 0, source_dict, args.save_article_file, args.debug)
    train_target = preprocess.save(train_tgt , 1, target_dict, args.save_summary_file, args.debug)
if args.save_option == "val":
    val_src = '{}/{}'.format(pardir, "plain_data/val.txt.src")
    print("source data path: {} ".format(val_src))
    val_article_file = "data/val_article.pt"
    preprocess.save(val_src , 0, source_dict, val_article_file, args.debug)
if args.save_option == "test":
    test_src = '{}/{}'.format(pardir, "plain_data/test.txt.src")
    print("source data path: {} ".format(test_src))
    test_article_file = "data/test_article.pt"
    preprocess.save(test_src , 0, source_dict, test_article_file, args.debug)
