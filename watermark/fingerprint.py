# -*- coding: UTF-8 -*-
import copy

import numpy as np
from torch import nn
import torch
import random
from geneal.genetic_algorithms import BinaryGenAlgSolver


def dec2bin(num, length):
    mid = []
    while True:
        if num == 0:
            break
        num, rem = divmod(num, 2)
        if int(rem) == 0:
            mid.append(-1)
        else:
            mid.append(1)
    while len(mid) < length:
        mid.insert(0, -1)
    return mid

"""
1. 生成一组长度为 length 的二进制向量，每个向量用于标识一个客户端。
2. 使用遗传算法优化指纹，确保它们之间的汉明距离最大化，从而减少指纹之间的相似性。
例如：num_clients = 3，length = 4 条件下生成如下指纹
fingerprints = [
    np.array([1, -1, 1, -1]),
    np.array([-1, 1, -1, 1]),
    np.array([1, 1, -1, -1])
]
"""
def generate_fingerprints(num_clients, length):
    # 设置随机种子，确保每次运行代码时生成的指纹相同
    np.random.seed(0)
    random.seed(0)
    # 使用 random.getrandbits(length) 生成一个长度为 length 的随机二进制数
    # 将这些二进制数存储在集合 fingerprints_int 中，确保每个指纹唯一
    fingerprints_int = set()
    while len(fingerprints_int) < num_clients:
        fingerprints_int.add(random.getrandbits(length))
    # 使用遗传算法优化指纹，确保它们之间的差异最大化
    solver = BinaryGenAlgSolver(
        n_genes=num_clients,
        fitness_function=get_minimum_hamming_distance_func(num_clients, length), 
        n_bits=length, # number of bits describing each gene (variable)
        pop_size=10, # population size (number of individuals)
        max_gen=50, # maximum number of generations
        mutation_rate=0.05, # mutation rate to apply to the population
        selection_rate=0.5, # percentage of the population to select for mating
    )
    solver.solve()
    fingerprints = []
    count = 0
    # 将每个指纹从二进制（0 和 1）转换为 -1 和 1 的格式，分配给每个客户端
    for i in range(num_clients):
        fingerprint = np.array(solver.best_individual_[count: count + length])  #遗传算法找到的最优个体（即最优指纹组合）
        fingerprint[fingerprint == 0.0] = -1.0
        fingerprints.append(fingerprint)
        count += length
    return fingerprints


def hamming_distance(a, b):
    return bin(int(a) ^ int(b)).count("1")


def get_minimum_hamming_distance_func(num_clients, length):
    def minimum_hamming_distance(fingerprints):
        x = fingerprints.reshape(num_clients, length)
        # n = len(fingerprints)
        n = num_clients
        min_hamming = 100000
        for i in range(n):
            if sum(x[i] == np.ones(x[i].shape)) == 0:
                return -100000
            for j in range(i + 1, n):
                distance = np.sum(x[i] != x[j])
                if distance == 0:
                    return -100000
                if distance < min_hamming:
                    min_hamming = distance
        return min_hamming
    return minimum_hamming_distance

"""
生成一组形状为 (total_length, weight_size) 的随机矩阵，每个矩阵用于从一个权重向量中提取信息
np.random.standard_normal：从标准正态分布（均值为 0，标准差为 1）中随机采样。
(total_length, weight_size)：每个矩阵的形状，total_length 是输出向量的长度，weight_size 是输入权重向量的长度。
astype(np.float32)：将矩阵的数据类型转换为 float32，以节省内存并提高计算效率。
extracting_matrices：存储生成的矩阵。
"""
def generate_extracting_matrices(weight_size, total_length, num_clients):
    np.random.seed(0)
    extracting_matrices = []
    for i in range(num_clients):
        extracting_matrices.append(np.random.standard_normal((total_length, weight_size)).astype(np.float32))
    return extracting_matrices


class HingeLikeLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, results, labels):
        loss = torch.mul(results, labels)
        loss = torch.mul(loss, -1)
        loss = torch.add(loss, self.epsilon)
        loss = torch.sum(torch.clamp(loss, min=0))
        return loss


def calculate_local_grad(layers, local_fingerprint, extracting_metrix, epsilon=0.5):
    for layer in layers:
        layer.zero_grad()
    weight = layers[0].weight.detach().numpy()
    for i in range(1, len(layers)):
        weight = np.append(weight, layers[i].weight.detach().numpy())
    weight = nn.Parameter(torch.from_numpy(weight))
    loss_func = HingeLikeLoss(epsilon=epsilon)
    matrix = torch.from_numpy(extracting_metrix).float()
    fingerprint = torch.from_numpy(local_fingerprint).float()
    result = torch.matmul(matrix, weight)
    loss = loss_func(result, fingerprint)
    loss.backward()
    return copy.deepcopy(weight.grad)

"""
这段代码的功能是从模型的权重中提取指纹，并与本地指纹进行比较，以找到最匹配的指纹。
具体来说，它通过计算权重与提取矩阵的乘积，然后与本地指纹进行比较，最终返回最匹配的指纹的得分或误码率（BER）。
假设：
提取的指纹：[0.2, -0.3, 0.8, -0.1]
本地指纹：[1, -1, 1, -1]
epsilon = 0.5

计算匹配得分：
逐元素乘积：[0.2, 0.3, 0.8, 0.1]
截断：[0.2, 0.3, 0.5, 0.1]
归一化得分：(0.2 + 0.3 + 0.5 + 0.1) / 4 / 0.5 = 0.55
返回：0.55, idx
"""

def extracting_fingerprints(layers, local_fingerprints, extracting_matrices, epsilon=0.5, hd=False):
    min_ber = 100000  # 最小误码率
    min_idx = 0  # 最小误码率对应的索引
    max_score = -100000  # 最大匹配得分，初始化为一个较小的值
    max_idx = 0 # 最大匹配得分对应的索引
    bit_length = local_fingerprints[0].shape[0]
    weight = layers[0].weight.detach().numpy()  # 提取模型的权重，并将其转换为 NumPy 数组
    for i in range(1, len(layers)):
        weight = np.append(weight, layers[i].weight.detach().numpy())
    for idx in range(len(local_fingerprints)):
        matrix = extracting_matrices[idx]
        # print(f"客户端：{idx} 行数: {len(matrix)} 列：{len(matrix[0])}")
        result = np.dot(matrix, weight)  # 计算提取矩阵与权重的点积，得到提取的指纹
        if hd:
            result[result >= 0] = 1
            result[result < 0] = -1
            ber = np.sum(result != local_fingerprints[idx]) / bit_length  # 计算提取的指纹与本地指纹之间的误码率（不匹配的比例）
            if ber < min_ber:
                min_ber = ber
                min_idx = idx
        else:
            result = np.multiply(result, local_fingerprints[idx])
            result[result > epsilon] = epsilon
            score = np.sum(result) / bit_length / epsilon  # 计算匹配得分
            if score > max_score:
                max_score = score
                max_idx = idx
    if hd:
        return min_ber, min_idx
    else:
        return max_score, max_idx

"""
将 embed_layer_names 按分号 ; 分隔成多个嵌入层名称。
对于每个嵌入层名称，按点号 . 分隔并逐层获取子模块。
将提取的嵌入层添加到列表中并返回。
"""
def get_embed_layers(model, embed_layer_names):
    embed_layers = []
    embed_layer_names = embed_layer_names.split(";")
    for embed_layer_name in embed_layer_names:
        embed_layer = model
        for name in embed_layer_name.split('.'):
            embed_layer = getattr(embed_layer, name)
        embed_layers.append(embed_layer)
    return embed_layers

"""
调用 get_embed_layers 提取嵌入层。
遍历每个嵌入层，累加其词汇表大小（embed_layer.weight.shape[0]）。
返回总长度。

def get_embed_layers_length(model, embed_layer_names):
    weight_size = 0
    embed_layers = get_embed_layers(model, embed_layer_names)
    for embed_layer in embed_layers:
        weight_size += embed_layer.weight.shape[0]
    return weight_size
"""
def get_embed_layers_length(model, embed_layer_names):
    weight_size = 0
    embed_layers = get_embed_layers(model, embed_layer_names)
    for embed_layer in embed_layers:
        # 计算权重的总参数量（所有维度乘积）
        weight_size += embed_layer.weight.numel()  # 关键修改
    return weight_size

def apply_gradient(embed_layers, client_grad):
    weight_count = 0
    for layer in embed_layers:
        weight_length = layer.weight.numel()
        grad_slice = client_grad[weight_count: weight_count + weight_length].view(layer.weight.shape)
        layer.weight.data.add_(grad_slice)
        weight_count += weight_length
