# -*- coding: UTF-8 -*-
import argparse
import distutils.util
import json
import os

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def restore_args(path):
    with open(path, 'r') as f:
        args_dict = json.load(f)
    return Args(**args_dict)

def collect_args():
    parser = argparse.ArgumentParser()
    # attack type
    parser.add_argument('--attack_type', type=int, default=0, help='FineTuning for 1 | Prune for 2 | Overwrite for 3')
    # args.txt path
    parser.add_argument('--args_file_path', type=str, default='./result/VGG16/args.txt')
    # global model path
    parser.add_argument('--model_path', type=str, default='./result/VGG16/model_last_epochs_30.pth')
    # attack model dir
    parser.add_argument('--attack_model_dir', type=str, default='./result/VGG16/')
    # test
    parser.add_argument('--test', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help='whether test the model')

    args = parser.parse_args()
    return args