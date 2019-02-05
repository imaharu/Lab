import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define import *
from generate import *
from pyrouge import Rouge155

def save(model, generate_module):
    save_dir = "{}/{}".format("trained_model", args.save_dir)
    generate_dir = "{}/{}".format(save_dir , args.generate_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(generate_dir):
        os.mkdir(generate_dir)

    generate_module.generate(generate_dir, model=model)


def EvaluateByPyrouge(generate_path, model_dir):
    r = Rouge155()
    r.system_dir = generate_path
    r.model_dir = model_dir
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = 'gold_#ID#.txt'
    output = r.convert_and_evaluate()
    save_dir = "{}/{}".format("trained_model", args.save_dir)
    rouge_result = "{}/{}".format(save_dir, args.result_file)
    with open(rouge_result, "w") as f:
        print(output, file=f)
    output_dict = r.output_to_dict(output)
    return output_dict["rouge_1_f_score"], output_dict["rouge_2_f_score"], output_dict["rouge_l_f_score"]

save_dir = "{}/{}".format("trained_model", args.save_dir)
generate_dir = "{}/{}".format(save_dir , args.generate_dir)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(generate_dir):
    os.mkdir(generate_dir)

device = torch.device('cuda:0')
opts = { "bidirectional" : args.none_bid, "coverage_vector": args.coverage }
model = Hierachical(opts).cuda()
model_dir = "/home/ochi/Lab/gold_summary/val_summaries"
max_rouge1 = 0
max_rouge2 = 0
max_rougeL = 0

checkpoint = torch.load("{}/{}".format(save_dir , str(args.model_path) ))
model.load_state_dict(checkpoint['state_dict'])
generate_module = GenerateDoc(generate_data)
generate_module.generate(generate_dir, model=model)
rouge1, rouge2, rougeL = EvaluateByPyrouge(generate_dir, model_dir)
print("rouge1 : {} \nrouge2 : {} \nrougeL : {}\n".format(rouge1, rouge2, rougeL))
