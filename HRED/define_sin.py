# my function
from create_sin_dict import *

# Other
import os
import glob
import torch
import pickle
import argparse

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
'''model param'''
parser.add_argument('--weightdecay', type=float, default=1.0e-6,
                    help='Weight Decay norm')
parser.add_argument('--gradclip', type=float, default=5.0,
                    help='Gradient norm threshold to clip')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Set dropout ratio in training')
'''train details'''
parser.add_argument('--epoch', '-e', type=int, default=30,
                    help='Number of sweeps over the dataset to train')

'''train_num embed hidden batch'''
parser.add_argument('--train_doc_num','-t', type=int, default=90000,
                    help='train num')
parser.add_argument('--embed_size', type=int, default=256,
                    help='size of embed size for word representation')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--batch_size', '-b', type=int, default=50,
                    help='Number of batchsize')
parser.add_argument('--num_layer', '-l', type=int, default=4,
                    help='Layer num')
parser.add_argument('--train_or_generate', '--tg', type=int, default=0, help='train is 0 : generete is 1')
parser.add_argument('--test_size',type=int, default=1000, help='test_size')

parser.add_argument('--result_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--unk', type=int, default=0)
parser.add_argument('--new', type=int, default=0)
parser.add_argument('--debug', type=int, default=0)
#parser.add_argument('--device', type=int, default=0)
parser.set_defaults(generate=False)
args = parser.parse_args()

##### end #####

english_vocab = {}

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

if args.device:
    device = torch.device('cuda:'+ str(args.device))
else:
    device = torch.device("cuda:0")

data_path = os.environ["cnn_unk"] + "/train"
data_path = "/home/ochi/Lab/CNN_STORY/cnn_stories_tokenized"
english_paths = sorted(glob.glob(data_path + "/*.story"))[0:train_doc_num]
if not args.new:
    with open('cnn.dump', 'rb') as f:
        english_vocab = pickle.load(f)
else:
    get_dict(english_paths, english_vocab)

source_size = len(english_vocab) + 1
target_size = len(english_vocab) + 1

logger.debug("訓練文書数: " +  str(train_doc_num))
logger.debug("hidden_size: " + str(hidden_size))
logger.debug("embed_size: " +  str(embed_size))
logger.debug("epoch : " + str(epoch))
logger.debug("num layer : " +  str(args.num_layer))
