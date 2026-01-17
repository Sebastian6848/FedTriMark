import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import os.path

from utils.test import test_img
from experiment.utils import *

def fine_tune_attack(model, train_dataset, test_dataset, apt, args, trigger_set):
    """
    对模型进行微调攻击
    :param model: 待攻击的模型
    :param train_dataset: 用于微调的训练数据集
    :param test_dataset: 用于测试的数据集
    :param args: 参数配置
    :param trigger_set: 水印触发器数据集（可选）
    """
    # 将模型移动到指定设备
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

    # 微调模型
    for epoch in range(args.attack_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 打印训练结果
        train_acc = 100. * correct / total
        print(f"Epoch {epoch + 1}/{args.attack_epochs}, Train Loss: {train_loss / (batch_idx + 1):.3f}, Train Acc: {train_acc:.2f}%")

        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # 打印测试结果
        test_acc = 100. * correct / total
        print(f"Epoch {epoch + 1}/{args.attack_epochs}, Test Loss: {test_loss / (batch_idx + 1):.3f}, Test Acc: {test_acc:.2f}%")

        # 测试水印（如果提供了触发器数据集）
        if trigger_set:
            watermark_acc = 0.0
            correct = 0
            total = 0

            for inputs, labels in trigger_loader:
                print("Labels:", labels)
                break
            model.eval()
            with torch.no_grad():
                for inputs, labels in trigger_loader:
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    outputs = model(inputs)
                    print("Model Outputs:", outputs.argmax(dim=1))
                    print("Labels:", labels)
                    break

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(trigger_loader):
                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            watermark_acc = 100. * correct / total
            #print(f"Epoch {epoch + 1}/{args.attack_epochs}, Watermark Acc: {watermark_acc:.2f}%")

    # 保存微调后的模型
    if args.save:
        torch.save(model.state_dict(), os.path.join(apt.attack_model_dir, "model_FineTuneAttacked.pth"))