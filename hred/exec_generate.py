import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define import *
from generate import *

def save(model, generate_module):
    save_dir = "{}/{}".format("trained_model", args.save_dir)
    generate_dir = "{}/{}".format(save_dir , args.generate_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(generate_dir):
        os.mkdir(generate_dir)

    generate_module.generate(generate_dir, model=model)

opts = { "bidirectional" : args.none_bid }
model = Hierachical(opts).cuda()
checkpoint = torch.load("trained_model/{}".format(str(args.model_path)))
model.load_state_dict(checkpoint['state_dict'])
optimizer = torch.optim.Adagrad( model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])

generate_module = GenerateDoc(generate_data)
save(model, generate_module)
