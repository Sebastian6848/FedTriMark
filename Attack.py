# -*- coding: UTF-8 -*-
import copy
import os.path
import time
import numpy as np
import torch
import random
from torch.backends import cudnn
from torch.utils.data import DataLoader
import json

from fed.client import create_clients
from fed.server import FedAvg
from utils.datasets import get_full_dataset
from utils.models import get_model
from utils.test import test_img
from utils.train import get_optim, gem_train, set_bn_eval
from utils.utils import printf, load_args
from watermark.fingerprint import *
from watermark.watermark import *
from tqdm import tqdm

from experiment.FineTuneAttack import *
from experiment.PruneAttack import *
from experiment.OverwriteAttack import *
from experiment.utils import *
from experiment.Testify import *

if __name__ == '__main__':
    apt = collect_args()
    args = restore_args(apt.args_file_path)  # 解析命令行参数

    args.attack_epochs = 5  # 微调攻击的轮次

    # 加载数据集
    train_dataset, test_dataset = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))

    # 加载模型
    model = get_model(args)
    model.load_state_dict(torch.load(apt.model_path))

    # 加载触发器数据集
    trigger_set = generate_waffle_pattern(args)
    global_model = copy.deepcopy(model)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    global_model.to(args.device)

    clients = create_clients(args, train_dataset)
    for client in clients:
        client.set_model(copy.deepcopy(global_model))

    # get weight size
    weight_size = get_embed_layers_length(global_model, args.embed_layer_names)
    # generate fingerprints
    local_fingerprints = generate_fingerprints(args.num_clients, args.lfp_length)
    # generate extracting matrix
    extracting_matrices = generate_extracting_matrices(weight_size, args.lfp_length, args.num_clients)

    if apt.test:
        Testify(global_model, clients, test_dataset, trigger_set, local_fingerprints, extracting_matrices, args, apt)


    # 执行微调攻击
    if apt.attack_type == 1:
        fine_tune_attack(global_model, train_dataset, test_dataset, apt, args, trigger_set)
    elif apt.attack_type == 2:
        pruning_attack(global_model, train_dataset, test_dataset, apt, args, trigger_set)
    elif apt.attack_type == 3:
        overwrite_attack(global_model, train_dataset, test_dataset, apt, args, trigger_set)



