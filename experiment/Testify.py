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


def Testify(global_model, clients, test_dataset, trigger_set, local_fingerprints, extracting_matrices, args, apt):
    """
    测试全局模型和客户端模型的性能，计算水印准确率和指纹提取分数，并打印结果。

    参数:
        global_model: 全局模型。
        clients: 客户端列表。
        test_dataset: 测试数据集。
        trigger_set: 触发器数据集（用于水印测试）。
        local_fingerprints: 本地指纹列表。
        extracting_matrices: 提取矩阵列表。
        args: 命令行参数。

    返回:
        acc_best: 全局模型的最佳准确率。
        client_acc_best: 更新后的客户端历史最佳准确率列表。
    """

    log_path = os.path.join(apt.attack_model_dir, "testify_log.txt")

    # 测试全局模型
    acc_test, acc_test_top5 = test_img(global_model, test_dataset, args)
    printf("[Global Model]Testing accuracy: Top1: {:.3f}, Top5: {:.3f}".format(acc_test, acc_test_top5), log_path)
    watermark_acc, _ = test_img(global_model, trigger_set, args)
    printf("[Global Model]Watermark accuracy: {:.3f}".format(watermark_acc), log_path)

    # 测试客户端模型
    if args.watermark:
        avg_watermark_acc = 0.0
        avg_fss = 0.0
        client_acc = []
        client_acc_top5 = []
        for client_idx in range(args.num_clients):
            # 测试客户端模型在测试数据集上的准确率
            acc, acc_top5 = test_img(clients[client_idx].model, test_dataset, args)
            client_acc.append(acc)
            client_acc_top5.append(acc_top5)

            # 测试客户端模型在触发器数据集上的水印准确率
            watermark_acc, _ = test_img(clients[client_idx].model, trigger_set, args)
            avg_watermark_acc += watermark_acc

            # 计算指纹提取分数
            if args.fingerprint:
                embed_layers = get_embed_layers(clients[client_idx].model, args.embed_layer_names)
                fss, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints, extracting_matrices)
                avg_fss += fss

        # 计算统计指标
        avg_acc = np.mean(client_acc)
        max_acc = np.max(client_acc)
        min_acc = np.min(client_acc)
        median_acc = np.median(client_acc)
        low_acc = np.percentile(client_acc, 25)
        high_acc = np.percentile(client_acc, 75)

        avg_acc_top5 = np.mean(client_acc_top5)
        max_acc_top5 = np.max(client_acc_top5)
        min_acc_top5 = np.min(client_acc_top5)
        median_acc_top5 = np.median(client_acc_top5)
        low_acc_top5 = np.percentile(client_acc_top5, 25)
        high_acc_top5 = np.percentile(client_acc_top5, 75)

        avg_watermark_acc = avg_watermark_acc / args.num_clients
        avg_fss = avg_fss / args.num_clients

        # 打印结果
        printf("[Clients] Average Testing accuracy: {:.2f}".format(avg_acc), log_path)
        printf("[Clients] Quantile Testing accuracy, Top1: min_acc{:.2f}, low_acc{:.2f}, median_acc{:.2f}, high_acc{:.2f}, max_acc{:.2f}".format(min_acc, low_acc, median_acc, high_acc, max_acc), log_path)
        printf("[Clients] Quantile Testing accuracy, Top5: min_acc{:.2f}, low_acc{:.2f}, median_acc{:.2f}, high_acc{:.2f}, max_acc{:.2f}".format(min_acc_top5, low_acc_top5, median_acc_top5, high_acc_top5, max_acc_top5), log_path)
        printf("Watermark Average Testing accuracy:{:.2f}".format(avg_watermark_acc), log_path)
        if args.fingerprint:
            printf("Average fss: {:.4f}".format(avg_fss), log_path)