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
parser.add_argument('--embed_size', type=int, default=128,
                    help='size of embed size for word representation')

parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--batch_size', '-b', type=int, default=50,
                    help='Number of batchsize')

parser.add_argument('--generate_dir', '-g' ,type=str, default="val")
parser.add_argument('--model_path', '-m' , type=str)
parser.add_argument('--save_dir', '-s' , type=str, default="train")
parser.add_argument('--cuda', '-c' , type=str, default="0")

parser.add_argument('--load_article_file', type=str, default="data/article.pt",
                    help='load article file')
parser.add_argument('--load_summary_file', type=str, default="data/summary.pt",
                    help='load article file')
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
preprocess = Preprocess()
"""
    source_target dict and size is same
"""
source_dict = preprocess.getVocab(vocab_path)
target_dict = preprocess.getVocab(vocab_path)
source_size = len(source_dict)
target_size = len(target_dict)

if args.mode == "debug":
    article_data = preprocess.load("data/dubug_article.pt")
    summary_data = preprocess.load("data/dubug_summary.pt")

elif args.mode == "train":
    article_data = preprocess.load(args.load_article_file)
    summary_data = preprocess.load(args.load_summary_file)
elif args.mode == "val":
    generate_data = preprocess.load("data/val_article.pt")

hidden_size = args.hidden_size
embed_size = args.embed_size
max_epoch = args.epoch
batch_size = args.batch_size
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
