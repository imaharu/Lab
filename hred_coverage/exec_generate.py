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

for index in range(args.epoch):
    index += 1
    if (index >= 25) or index == args.epoch:
        try:
            checkpoint = torch.load("{}/{}-{}.model".format(save_dir , str(args.model_path) , index))
        except:
            print("{}/{}-{}.model は存在しません".format(save_dir , str(args.model_path) , index))
            continue
        model.load_state_dict(checkpoint['state_dict'])

        generate_module = GenerateDoc(generate_data)
        generate_module.generate(generate_dir, model=model)
        rouge1, rouge2, rougeL = EvaluateByPyrouge(generate_dir, model_dir)
        print("index: {} \nrouge1 : {} \nrouge2 : {} \nrougeL : {}\n".format(index, rouge1, rouge2, rougeL))
        if max_rouge1 < rouge1:
            with open("{}/max_rouge1.txt".format(save_dir), 'w') as f:
                f.write("max_{}-{}.model\n".format(args.model_path, index))
                f.write("max_rouge1 score : {}\n".format(str(rouge1)))
            max_rouge1 = rouge1
        if max_rouge2 < rouge2:
            with open("{}/max_rouge2.txt".format(save_dir), 'w') as f:
                f.write("max_{}-{}.model \n".format(args.model_path, index))
                f.write("max_rouge2 score : {}\n".format(str(rouge2)))
            max_rouge2 = rouge2
        if max_rougeL < rougeL:
            with open("{}/max_rougeL.txt".format(save_dir), 'w') as f:
                f.write("max_{}-{}.model \n".format(args.model_path, index))
                f.write("max_rougeL score : {}\n".format(str(rougeL)))
            max_rougeL = rougeL
