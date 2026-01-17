import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import os.path

from utils.test import test_img
from experiment.utils import *


def prune_model(model, prune_rate=0.2):
    """
    对模型进行结构化剪枝
    :param model: 待剪枝的模型
    :param prune_rate: 剪枝比例 (0-1)
    """
    # 获取所有卷积和全连接层
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # 全局L1范数剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_rate,
    )

    # 永久移除剪枝掩码
    for module, param in parameters_to_prune:
        prune.remove(module, param)


def pruning_attack(model, train_dataset, test_dataset, apt, args, trigger_set):
    """
    执行剪枝攻击并验证效果
    :param model: 待攻击的模型
    :param train_dataset: 用于微调恢复的原始数据集
    :param test_dataset: 测试数据集
    :param args: 配置参数
    :param trigger_set: 水印触发器数据集 (可选)
    """
    # 设备配置
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    model.to(args.device)

    # 记录原始性能
    original_acc, _ = test_img(model, test_dataset, args)
    print(f"[orign model] Test Acc: {original_acc:.2f}%")
    if trigger_set:
        wm_acc, _ = test_img(model, trigger_set, args)
        print(f"[orign model] Watermark Acc: {wm_acc:.2f}%")

    # 执行设备感知的剪枝
    prune_model(model)
    model = model.to(args.device)

    # 剪枝后直接测试
    pruned_acc, _ = test_img(model, test_dataset, args)
    print(f"[after Prune] Test Acc: {pruned_acc:.2f}%")
    if trigger_set:
        pruned_wm_acc, _ = test_img(model, trigger_set, args)
        print(f"[after Prune] Watermark Acc: {pruned_wm_acc:.2f}%")

    # 微调恢复性能
    if args.attack_epochs > 0:
        print("\nPrune begin...")
        optimizer = optim.SGD(model.parameters(),
                              lr=args.local_lr,
                              momentum=args.local_momentum)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.local_bs,
                                  shuffle=True)

        # 微调训练循环
        for epoch in range(args.attack_epochs):
            model.train().to(args.device)  # 确保训练时在GPU
            train_loss = 0
            correct = 0
            total = 0

            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
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

            # 打印训练状态
            train_acc = 100. * correct / total
            print(f"Epoch {epoch + 1} | Train loss: {train_loss / len(train_loader):.3f} | Train Acc: {train_acc:.2f}%")

            # 实时测试
            if (epoch + 1) % args.test_interval == 0:
                test_acc, _ = test_img(model, test_dataset, args)
                print(f"Epoch {epoch + 1} | Test Acc: {test_acc:.2f}%")
                if trigger_set:
                    wm_acc, _ = test_img(model, trigger_set, args)
                    #print(f"Epoch {epoch + 1} | Watermark Acc: {wm_acc:.2f}%")

    # 最终测试
    final_acc, _ = test_img(model, test_dataset, args)
    print(f"\n[result] Test Acc: {final_acc:.2f}%")
    if trigger_set:
        final_wm_acc, _ = test_img(model, trigger_set, args)
        #print(f"[result] Watermark Acc: {final_wm_acc:.2f}%")

    # 保存被攻击的模型
    save_path = os.path.join(apt.attack_model_dir, "model_PruneAttacked.pth")
    torch.save(model.state_dict(), save_path)