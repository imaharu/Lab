# my function
# Other
import os
import glob
import torch
import pickle
import argparse
from preprocessing import *

# Set logger
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


##### args #####
parser = argparse.ArgumentParser(description='Sequence to Sequence Model by using Pytorch')
'''train details'''
parser.add_argument('--epoch', '-e', type=int, default=35,
                    help='Number of sweeps over the dataset to train')

'''train_num embed hidden batch'''
parser.add_argument('--train_doc_num','-t', type=int,
                    help='train num')
parser.add_argument('--embed_size', type=int, default=256,
                    help='size of embed size for word representation')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='size of dropout')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--batch_size', '-b', type=int, default=50,
                    help='Number of batchsize')
parser.add_argument('--train_or_generate', '--tg', type=int, default=0, help='train is 0 : generete is 1')
parser.add_argument('--test_size',type=int, default=1000, help='test_size')

parser.add_argument('--result_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--load_article_file', type=str, default="data/article_word.pt",
                    help='load article file')
parser.add_argument('--load_summary_file', type=str, default="data/summary_word.pt",
                    help='load article file')
parser.add_argument('--create_data', type=int, default=0,
                    help='if over 0 like 1 it will create data')
args = parser.parse_args()

##### end #####

word_data = Word_Data()

if args.create_data:
    article_save_path = "data/article_word.pt"
    summary_save_path = "data/summary_word.pt"
    word_data.save(article_save_path, summary_save_path)
    exit()

if args.debug:
    train_doc_num = 6
    hidden_size = 4
    embed_size = 4
    batch_size = 2
    epoch = 2

else:
    train_doc_num = args.train_doc_num
    hidden_size = args.hidden_size
    embed_size = args.embed_size
    batch_size = args.batch_size
    epoch = args.epoch

if args.train_or_generate == 1:
    get_test_data_target(args.test_size, output_input_lines)

source_size = word_data.getVocabSize() + 1
target_size = source_size
if train_doc_num is None:
    article_data = torch.load(args.load_article_file)
    summary_data = torch.load(args.load_summary_file)
    train_doc_num = len(article_data)
else:
    article_data = torch.load(args.load_article_file)[0:train_doc_num]
    summary_data = torch.load(args.load_summary_file)[0:train_doc_num]
logger.debug("訓練文書数: " +  str(train_doc_num))
logger.debug("hidden_size: " + str(hidden_size))
logger.debug("embed_size: " +  str(embed_size))
logger.debug("epoch : " + str(epoch))
logger.debug("batch size : " +  str(batch_size))
