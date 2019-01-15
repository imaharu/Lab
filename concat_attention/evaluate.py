import argparse
from pyrouge import Rouge155
from define import *

def EvaluateByPyrouge(result_path, model_dir):
    r = Rouge155()
    r.system_dir = result_path
    r.model_dir = model_dir
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = 'gold_#ID#.txt'
    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    return output_dict
