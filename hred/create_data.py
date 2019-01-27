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
parser.add_argument('--mode', type=str, default="dubug",
                    help='save debug train generate')
args = parser.parse_args()
##### end #####

vocab_path = os.environ['cnn_vocab50000']
preprocess = Preprocess(args.max_article_len, args.max_summary_len)

source_dict = preprocess.getVocab(vocab_path)
target_dict = preprocess.getVocab(vocab_path)
source_size = len(source_dict)
target_size = len(target_dict)

if args.mode == "save":
    debug = False
    pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.save_option == "train":
        train_src = '{}/{}'.format(pardir, "plain_data/train.txt.src")
        train_tgt = '{}/{}'.format(pardir, "plain_data/train.txt.tgt.tagged")
        print("source data path: {} ".format(train_src))
        print("target data path: {} ".format(train_tgt))
        preprocess.save(train_src , 0, source_dict, args.save_article_file, debug)
        preprocess.save(train_tgt , 1, target_dict, args.save_summary_file, debug)
    elif args.save_option == "val":
        val_src = '{}/{}'.format(pardir, "plain_data/val.txt.src")
        print("source data path: {} ".format(val_src))
        preprocess.save(val_src , 0, source_dict, args.save_article_file, debug)
    elif args.save_option == "test":
        test_src = '{}/{}'.format(pardir, "plain_data/test.txt.src")
        print("source data path: {} ".format(test_src))
        test_source = preprocess.save(test_src , 0, source_dict, args.save_article_file, debug)
