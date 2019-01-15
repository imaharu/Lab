import argparse
from pyrouge import Rouge155

##### args #####
parser = argparse.ArgumentParser(description='Evaluate by pyrouge')
''' mode '''
parser.add_argument('--result_path', '-r', help='result path')
parser.add_argument('--mode', help='val test')
args = parser.parse_args()
##### end #####


r = Rouge155()
r.system_dir = "trained_model/{}".format(args.result_path)
if args.mode == "val":
    r.model_dir = '/home/ochi/Lab/gold_summary/val_summaries'
elif args.mode == "test":
    r.model_dir = '/home/ochi/Lab/gold_summary/test_summaries'
r.system_filename_pattern = '(\d+).txt'
r.model_filename_pattern = 'gold_#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
print(output_dict)
