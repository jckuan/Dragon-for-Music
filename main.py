# coding: utf-8


"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start, quick_start_fix_params
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DRAGON', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of datasets')

    config_dict = {
        'learning_rate': [1e-4],
        'reg_weight': [0.001],
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start_fix_params(args.model, args.dataset, config_dict, save_model=True)


