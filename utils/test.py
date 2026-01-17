# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 计算模型的 Top-1 准确率（预测的最高概率类别是否正确）和 Top-5 准确率（真实类别是否在预测的前 5 个类别中）
def test_img(net_g, datatest, args):
    net_g.eval()  # 将模型设置为评估模式，关闭 dropout 和 batch normalization 等训练时的特殊行为。
    net_g.to(args.device)
    # testing
    correct = 0
    correct_top5 = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    # 将当前批次的数据和标签传入设备中测试
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)  # 使用模型对数据进行预测，得到输出的对数概率

        # 计算top1准确率
        _, y_pred = torch.max(log_probs.data, 1)
        correct += (y_pred == target).sum().item()  # 统计当前批次中预测正确的样本数量
        #计算Top5准确率
        _, pred = log_probs.topk(5, 1, True, True)
        target_resize = target.view(-1, 1)
        correct_top5 += torch.eq(pred, target_resize).sum().float().item()

    accuracy = 100.00 * correct / len(data_loader.dataset)
    accuracy_top5 = 100.00 * correct_top5 / len(data_loader.dataset)
    net_g.cpu()  # 将模型移回 CPU，释放 GPU 内存
    return accuracy, accuracy_top5


