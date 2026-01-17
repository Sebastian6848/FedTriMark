# -*- coding: UTF-8 -*-
import copy
import os
import numpy as np
from torchvision import transforms
from security.FGSM import *

from trigger.generate_waffle_pattern import NumpyLoader

def FedAvg(models, nums):
    model_avg = copy.deepcopy(models[0])
    total = sum(nums)
    for k in model_avg.keys():
        model_avg[k] = torch.div(model_avg[k], total / nums[0])
        for i in range(1, len(models)):
            model_avg[k] += torch.div(models[i][k], total / nums[i])
    return model_avg

def server_aggregate_trigger_sets(upload_dir="./trigger/upload_buffer", adv=False, model=None, eps=0.03, device='cuda'):
    all_data, all_labels = [], []

    for file in os.listdir(upload_dir):
        if "data.npy" in file:
            label_file = file.replace("data", "labels")
            data = np.load(os.path.join(upload_dir, file))  # shape: (N, H, W, C)
            labels = np.load(os.path.join(upload_dir, label_file))  # shape: (N,)
            all_data.append(data)
            all_labels.append(labels)

    all_data = np.concatenate(all_data, axis=0)    # (N_total, H, W, C)
    all_labels = np.concatenate(all_labels, axis=0)

    # 统一标准化（按通道）
    mean = np.mean(all_data, axis=(0, 1, 2))  # shape: (C,)
    std = np.std(all_data, axis=(0, 1, 2))  # shape: (C,)
    print(f"[Server] Aggregated Trigger Set | mean={mean}, std={std}")

    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean, std)
    ])
    dataset = NumpyLoader(all_data, all_labels, transformer=transform)

    return dataset



