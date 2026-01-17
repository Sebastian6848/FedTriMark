import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import os.path

from utils.test import test_img
from experiment.utils import *

def overwrite_attack(model, train_dataset, test_dataset,apt ,args, trigger_set, attack_strength=0.1, num_iterations=10):
    """
    执行覆盖攻击并验证效果
    :param model: 待攻击的模型
    :param train_dataset: 用于微调恢复的原始数据集
    :param test_dataset: 测试数据集
    :param args: 配置参数
    :param trigger_set: 水印触发器数据集 (可选)
    :param attack_strength: 攻击强度 (控制噪声的大小)
    :param num_iterations: 攻击迭代次数
    """
    # 设备配置
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    model.to(args.device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.local_lr, momentum=args.local_momentum)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False)
    if trigger_set:
        trigger_loader = DataLoader(trigger_set, batch_size=args.local_bs, shuffle=True)

    # 记录原始性能
    original_acc, _ = test_img(model, test_dataset, args)
    print(f"[Original Model] Test Acc: {original_acc:.2f}%")
    if trigger_set:
        wm_acc, _ = test_img(model, trigger_set, args)
        print(f"[Original Model] Watermark Acc: {wm_acc:.2f}%")

    # 执行覆盖攻击
    print("\nStarting Overwrite Attack...")
    model.train()
    model = model.to(args.device)

    # 攻击过程
    for iteration in tqdm(range(num_iterations), desc="Overwrite Attack"):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 在梯度上添加噪声，试图破坏水印
            loss.backward()
            for param in model.parameters():
                noise = torch.randn_like(param.grad) * attack_strength
                param.grad += noise
            optimizer.step()

    # 攻击后测试
    attacked_acc, _ = test_img(model, test_dataset, args)
    print(f"[After Attack] Test Acc: {attacked_acc:.2f}%")
    if trigger_set:
        attacked_wm_acc, _ = test_img(model, trigger_set, args)
        print(f"[After Attack] Watermark Acc: {attacked_wm_acc:.2f}%")


    # 最终测试
    final_acc, _ = test_img(model, test_dataset, args)
    print(f"\n[Final Result] Test Acc: {final_acc:.2f}%")
    if trigger_set:
        final_wm_acc, _ = test_img(model, trigger_set, args)
        print(f"[Final Result] Watermark Acc: {final_wm_acc:.2f}%")

    # 保存被攻击的模型
    torch.save(model.state_dict(), os.path.join(apt.attack_model_dir, "model_OverwriteAttacked.pth"))

