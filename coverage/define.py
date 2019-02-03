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


parser.add_argument('--epoch', '-e', type=int, default=33,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--embed', type=int, default=128,
                    help='size of embed size for word representation')

parser.add_argument('--hidden', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--batch', '-b', type=int, default=50,
                    help='Number of batchsize')

parser.add_argument('--max_article_len', type=int, default=400,
                    help='max article length')
parser.add_argument('--max_summary_len', type=int, default=100,
                    help='max summary length')

parser.add_argument('--generate_dir', '-g' ,type=str, default="val")
parser.add_argument('--model_path', '-m' , type=str)
parser.add_argument('--save_dir', '-s' , type=str, default="train")
parser.add_argument('--cuda', '-c' , type=str, default="0")

parser.add_argument('--load_article_file', type=str, default="data/article.pt",
                    help='load article file')
parser.add_argument('--load_summary_file', type=str, default="data/summary.pt",
                    help='load article file')
parser.add_argument('--save_article_file', type=str, default="data/article.pt",
                    help='save article file')
parser.add_argument('--save_summary_file', type=str, default="data/summary.pt",
                    help='save article file')
parser.add_argument('--mode', type=str, default="dubug",
                    help='save debug train evaluate')
parser.add_argument('--save_option', type=str, default="train",
                    help='save option')
parser.add_argument('--set_state', action='store_false')

parser.add_argument('--none_bid', action='store_false')
parser.add_argument('--coverage', action='store_true')

args = parser.parse_args()
##### end #####

vocab_path = os.environ['cnn_vocab50000']
preprocess = Preprocess(args.max_article_len, args.max_summary_len)
"""
    source_target dict and size is same
"""
source_dict = preprocess.getVocab(vocab_path)
target_dict = preprocess.getVocab(vocab_path)
source_size = len(source_dict)
target_size = len(target_dict)

if args.mode == "save":
    pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    debug = False
    if args.save_option == "train":
        train_src = '{}/{}'.format(pardir, "plain_data/train.txt.src")
        train_tgt = '{}/{}'.format(pardir, "plain_data/train.txt.tgt.tagged")
        print("source data path: {} ".format(train_src))
        print("target data path: {} ".format(train_tgt))
        train_source = preprocess.save(train_src , 0, source_dict, args.save_article_file, debug)
        train_target = preprocess.save(train_tgt , 1, target_dict, args.save_summary_file, debug)
    elif args.save_option == "val":
        val_src = '{}/{}'.format(pardir, "plain_data/val.txt.src")
        print("source data path: {} ".format(val_src))
        val_source = preprocess.save(val_src , 0, source_dict, args.save_article_file, debug)
    elif args.save_option == "test":
        test_src = '{}/{}'.format(pardir, "plain_data/test.txt.src")
        print("source data path: {} ".format(test_src))
        test_source = preprocess.save(test_src , 0, source_dict, args.save_article_file, debug)
    exit()

if args.mode == "debug":
    article_data = preprocess.load("data/debug_article.pt")
    summary_data = preprocess.load("data/debug_summary.pt")

elif args.mode == "train":
    article_data = preprocess.load(args.load_article_file)
    summary_data = preprocess.load(args.load_summary_file)

elif args.mode == "generate":
    generate_data = preprocess.load("data/val_article.pt")

hidden_size = args.hidden
embed_size = args.embed
max_epoch = args.epoch
batch_size = args.batch
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
