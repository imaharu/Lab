import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *
from model import *
from define import *
from loader import *
from generate_util import *
from pyrouge import Rouge155

class GenerateDoc():
    def __init__(self, aritcle_data):
        decode_set = DecodeDataset(aritcle_data)
        self.decode_iter = DataLoader(decode_set, batch_size=1, collate_fn=decode_set.collater)
        self.GenerateUtil = GenerateUtil(target_dict)

    def generate(self, generate_dir, model=False):
        model.eval()
        for index, iters in enumerate(self.decode_iter):
            doc = model(article_docs=iters.cuda(), generate=True)
            doc = self.GenerateUtil.TranslateDoc(doc)
            doc = ' '.join(doc)
            doc = ' '.join(doc.split()[:120])
            with open('{}/{:0=5}.txt'.format(generate_dir, index), mode='w') as f:
                f.write(doc)

def EvaluateByPyrouge(generate_path, model_dir):
    r = Rouge155()
    r.system_dir = generate_path
    r.model_dir = model_dir
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = 'gold_#ID#.txt'
    output = r.convert_and_evaluate()
    save_dir = "{}/{}".format("trained_model", args.save_dir)
    rouge_result = "{}/{}".format(save_dir, "rouge_test.txt")
    with open(rouge_result, "w") as f:
        print(output, file=f)
    output_dict = r.output_to_dict(output)
    print(output)
    output_dict = r.output_to_dict(output)
    return output_dict["rouge_1_f_score"], output_dict["rouge_2_f_score"], output_dict["rouge_l_f_score"]

save_dir = "{}/{}".format("trained_model", args.save_dir)
generate_dir = "{}/{}".format(save_dir , args.generate_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(generate_dir):
    os.mkdir(generate_dir)

device = torch.device('cuda:0')
opts = { "bidirectional" : args.none_bid }
model = EncoderDecoder(source_size, target_size, opts).cuda(device=device)
model_dir = "/home/ochi/Lab/gold_summary/test_summaries"
max_rouge1 = 0
max_rouge2 = 0
max_rougeL = 0

checkpoint = torch.load("{}/{}".format(save_dir , str(args.model_path)))
model.load_state_dict(checkpoint['state_dict'])
generate_module = GenerateDoc(generate_data)
generate_module.generate(generate_dir, model=model)
rouge1, rouge2, rougeL = EvaluateByPyrouge(generate_dir, model_dir)
