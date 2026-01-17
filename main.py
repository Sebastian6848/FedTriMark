# -*- coding: UTF-8 -*-
# 程序入口
import copy
import os.path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.backends import cudnn
from torch.utils.data import DataLoader
import json


from fed.client import *
from fed.server import *
from utils.datasets import get_full_dataset
from utils.models import get_model
from utils.test import *
from utils.train import get_optim, gem_train, set_bn_eval
from utils.utils import printf, load_args
from watermark.fingerprint import *
from watermark.watermark import *
from tqdm import tqdm
from experiment.attack_tester import *
from security.FGSM import *

from trigger.generate_pattern import *
from trigger.generate_waffle_pattern import *
from trigger.generate_trigger_plan import *


if __name__ == '__main__':
    args = load_args() # 解析命令行参数
    # 创建保存log.txt和args.txt的目录
    log_path = os.path.join(args.save_dir, 'log.txt')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # 保存训练参数文件args.txt
    with open(os.path.join(args.save_dir, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # 定义训练模型的设备（GPU or CPU）
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # 下载训练集和测试集，并进行预处理
    train_dataset, test_dataset = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    # 创建客户端， 并且给客户端分配训练数据集
    clients = create_clients(args, train_dataset)
    # 创建全局模型:根据输入的神经网络名称初始化神经网络模型，并且将模型分配给每个客户端（每个客户端得到全局模型的副本）
    global_model = get_model(args)
    if args.pre_train:
        global_model.load_state_dict(torch.load(args.pre_train_path))
    for client in clients:
        client.set_model(copy.deepcopy(global_model))

    # 初始化全局水印和模型指纹
    if args.watermark or args.fingerprint:
        # 提取嵌入层并分析其大小，了解模型的参数分布
        weight_size = get_embed_layers_length(global_model, args.embed_layer_names)
        print(f"weight_size:{weight_size}")

        # 当需要从文件加载参数时
        if args.isParameter:
            trigger_set = torch.load(args.trigger_path)  # 从文件加载触发集
            local_fingerprints = np.load(args.fingerprints_path)  # 从文件加载本地指纹
            extracting_matrices = torch.load(args.matrices_path)  # 从文件加载提取矩阵
            print("load parameter success...")
        else:
            # 生成指纹集：返回一个二位数组，里面包含每个客户端的指纹数组，指纹由-1和1构成，通过遗传算法确保差异性
            local_fingerprints = generate_fingerprints(args.num_clients, args.lfp_length)
            # 生成提取矩阵：生成一组(num_clients个)形状为 (args.lfp_length, weight_size) 的随机矩阵，每个矩阵用于从一个权重向量中提取信息
            extracting_matrices = generate_extracting_matrices(weight_size, args.lfp_length, args.num_clients)
            # 生成全局水印触发集
            trigger_set = create_trigger(args, global_model)
        watermark_set = DataLoader(trigger_set, batch_size=args.local_bs, shuffle=True)

        if args.save == True and args.isParameter == False:
            # 创建专用保存目录
            watermark_dir = os.path.join(args.save_dir, 'watermark_data')
            if not os.path.exists(watermark_dir):
                os.makedirs(watermark_dir)

            # 保存触发集
            torch.save(trigger_set, os.path.join(watermark_dir, 'trigger_set.pth'))

            # 保存指纹相关数据
            np.save(os.path.join(watermark_dir, 'local_fingerprints.npy'), local_fingerprints)
            torch.save(extracting_matrices, os.path.join(watermark_dir, 'extracting_matrices.pth'))
            # np.save(os.path.join(watermark_dir, 'encrypted_fingerprints.npy'), [ef.to_bytes() for ef in encrypted_fingerprints])

            # 保存水印参数
            watermark_config = {
                'embed_layer_names': args.embed_layer_names,
                'lfp_length': args.lfp_length,
                'lambda2': args.lambda2
            }
            with open(os.path.join(watermark_dir, 'watermark_config.json'), 'w') as f:
                json.dump(watermark_config, f)

    # 设置种子，确保实验可复现
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    # 开始训练
    train_loss = []
    val_loss, val_acc = [], []
    acc_best = None
    client_acc_best = [0 for i in range(args.num_clients)]
    num_clients_each_iter = max(min(args.num_clients, args.num_clients_each_iter), 1)

    # 训练过程
    for epoch in tqdm(range(args.start_epochs, args.epochs)):
        start_time = time.time()
        local_losses = []
        local_models = []
        local_nums = []

        # 选择参与本轮训练的客户端，并调节联邦学习客户端的学习率，确保每个客户端参与训练的机会一致
        for client in clients:
            client.local_lr *= args.lr_decay  # 对每个客户端的学习率进行衰减，衰减因子为 args.lr_decay

        # 步骤 1：获取所有客户端的 bwm_acc（加上一个 epsilon，防止全是0）
        bwm_accs = np.array([getattr(client, 'bwm_acc', 50) for client in clients])
        bwm_accs = bwm_accs + 1e-3  # 防止全部为 0 导致除以 0 错误
        # 步骤 2：归一化为选择概率（行为水印准确率越高，越容易被选中）
        probabilities = bwm_accs / bwm_accs.sum()
        # 步骤 3：根据概率加权采样客户端（不放回采样）
        clients_idxs = np.random.choice(range(args.num_clients), num_clients_each_iter, replace=False, p=probabilities)

        # 生成行为水印
        img = create_behavior_pattern(img_size=(32, 32), line_width=5)
        behavior_wm = generate_behavior_trigger(args, img, random.randint(0, 9))

        # 客户端进行本地训练
        for idx in clients_idxs:  # 对每个被选中的客户端进行本地训练
            current_client = clients[idx]
            local_model, num_samples, local_loss = current_client.train_one_iteration()  # 客户端进行一轮训练，返回本地训练后的模型、样本数量和损失值
            test_model = current_client.model

            bwm_acc = embed_behavior_watermark(test_model, behavior_wm, args)  # 客户端嵌入行为水印
            current_client.bwm_acc = bwm_acc  # 将该值保存到客户端对象中
            activity_score = compute_gradient_activity_score(global_model.state_dict(), local_model)
            if activity_score < 0.1:
                print(f"客户端{id}的activity_score为{activity_score}，可能是潜在的搭便车者！")

            local_models.append(copy.deepcopy(local_model))  # 本地模型列表
            local_losses.append(local_loss)  # 本地损失列表
            local_nums.append(num_samples)   # 本地样本数量

        # 服务器聚合客户端上传的本地模型，将聚合后的模型加载到全局模型 global_model 中
        global_model.load_global_model(FedAvg(local_models, local_nums), args.device, args.gem)

        # 将损失写入日志文件
        avg_loss = sum(local_losses) / len(local_losses)  # 计算本轮次所有客户端的平均损失 avg_loss
        printf('Round {:3d}, Average loss {:.3f}'.format(epoch, avg_loss), log_path)
        printf('Time: {}'.format(time.time() - start_time), log_path)
        train_loss.append(avg_loss)

        # 测试全局模型，并保存最优模型
        if (epoch + 1) % args.test_interval == 0:  # 检查当前训练轮次是否是测试间隔（args.test_interval）的倍数
            acc_test, acc_test_top5 = test_img(global_model, test_dataset, args)  #调用test_img函数，使用当前全局模型（global_model）在测试数据集（test_dataset）上进行测试
            printf("Testing accuracy: Top1: {:.3f}, Top5: {:.3f}".format(acc_test, acc_test_top5), log_path)
            if acc_best is None or acc_test >= acc_best:
                acc_best = acc_test
                if args.save:
                    torch.save(global_model.state_dict(), args.save_dir + "model_best.pth")

        # 嵌入全局水印和本地指纹
        if args.watermark:
            # 嵌入全局水印：让模型多次训练携带触发集的图像，使得模型在对于携带触发集的图像识别上具有超高的准确率
            watermark_optim = get_optim(global_model, args.local_optim, args.lambda1)  # 初始化水印优化器
            watermark_acc, _ = test_img(global_model, trigger_set, args)  # 在触发器数据集（trigger_set）上测试当前模型的水印准确率（watermark_acc）
            watermark_loss_func = nn.CrossEntropyLoss()  # 定义交叉熵损失函数（CrossEntropyLoss），用于计算水印嵌入的损失
            watermark_embed_iters = 0  # 初始化一个计数器（watermark_embed_iters），用于记录水印嵌入的迭代次数

            while watermark_acc <= 98 and watermark_embed_iters <= args.watermark_max_iters:
                global_model.train()
                global_model.to(args.device)
                if args.freeze_bn:
                    global_model.apply(set_bn_eval)
                watermark_embed_iters += 1

                for batch_idx, (images, labels) in enumerate(watermark_set):
                    images, labels = images.to(args.device), labels.to(args.device)
                    # ========== 对抗样本生成 ==========
                    if args.adv_train:
                        adv_images = fgsm_attack(global_model, images, labels, epsilon=args.adv_eps)
                    else:
                        adv_images = None
                    global_model.zero_grad()
                    # ========== 模型前向传播 ==========
                    clean_logits = global_model(images)
                    loss_clean = watermark_loss_func(clean_logits, labels)
                    if adv_images is not None:
                        adv_logits = global_model(adv_images)
                        loss_adv = watermark_loss_func(adv_logits, labels)
                        loss_consistency = torch.nn.functional.mse_loss(clean_logits, adv_logits)

                        # 综合损失（防止扰动破坏预测）
                        total_loss = (
                                args.wm_clean_lambda * loss_clean +
                                args.wm_adv_lambda * loss_adv +
                                args.wm_consistency_lambda * loss_consistency
                        )
                    else:
                        total_loss = loss_clean
                    # ========== 梯度更新 ==========
                    total_loss.backward()
                    if args.gem:
                        global_model = gem_train(global_model)
                    watermark_optim.step()
                # ========== 验证水印准确率 ==========
                watermark_acc, _ = test_img(global_model, trigger_set, args)

        # 嵌入本地指纹：从模型中提取指纹，然后与所有客户端的指纹进行对比，相似度最高的就是模型的主人。因此本地嵌入时要不断更新梯度，确保fss > 0.85
        if args.fingerprint:
            if (epoch + 1) % args.change_fingerprint == 0:
                # 生成指纹集：返回一个二位数组，里面包含每个客户端的指纹数组，指纹由-1和1构成，通过遗传算法确保差异性
                local_fingerprints = generate_fingerprints(args.num_clients, args.lfp_length)
                # 生成提取矩阵：生成一组(num_clients个)形状为 (args.lfp_length, weight_size) 的随机矩阵，每个矩阵用于从一个权重向量中提取信息
                extracting_matrices = generate_extracting_matrices(weight_size, args.lfp_length, args.num_clients)
            for client_idx in range(len(clients)):  # 遍历所有客户端，每个客户端给自己的本地模型嵌入指纹
                client_fingerprint = local_fingerprints[client_idx]
                client_model = copy.deepcopy(global_model)  # 复制全局模型（global_model）作为当前客户端的模型（client_model）
                embed_layers = get_embed_layers(client_model, args.embed_layer_names)  # 获取嵌入层

                fss, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints, extracting_matrices)  # 从嵌入层中提取指纹，计算指纹相似度分数（fss）和提取的客户端索引（extract_idx）
                count = 0
                while (extract_idx != client_idx or (client_idx == extract_idx and fss < 0.85)) and count <= args.fingerprint_max_iters:  # 当提取的指纹索引不等于当前客户端索引，或者指纹相似度分数小于0.85且未超过最大迭代次数时，继续嵌入指纹。
                    if args.dynamic:
                        # 计算当前λ2衰减系数（FSS越低，λ2越大）
                        lambda_factor = 1.0 + (0.85 - fss) * 2  # 动态调整因子
                        current_lambda = args.lambda2 * lambda_factor
                        # 计算梯度并更新权重
                        client_grad = calculate_local_grad(embed_layers, client_fingerprint,
                                                           extracting_matrices[client_idx])
                        client_grad = torch.mul(client_grad, -current_lambda)
                        # 应用梯度到嵌入层
                        apply_gradient(embed_layers, client_grad)
                    else:
                        client_grad = calculate_local_grad(embed_layers, client_fingerprint, extracting_matrices[client_idx])  # 计算嵌入层的梯度
                        client_grad = torch.mul(client_grad, -args.lambda2)  # 将梯度乘以学习率 -args.lambda2

                    weight_count = 0
                    for embed_layer in embed_layers:  # 将梯度添加到嵌入层的权重中
                        weight_length = embed_layer.weight.numel()  # 修改，原为shape[0]
                        # 从 client_grad 中截取对应长度的梯度，并重塑为权重形状
                        grad_slice = client_grad[weight_count: weight_count + weight_length]
                        grad_slice = grad_slice.view(embed_layer.weight.shape)  # 关键：重塑形状
                        embed_layer.weight = torch.nn.Parameter(torch.add(embed_layer.weight, grad_slice))
                        weight_count += weight_length
                    count += 1
                    fss, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints, extracting_matrices)  # 提取指纹进行测试
                printf("(Client_idx:{}, Result_idx:{}, FSS:{})".format(client_idx, extract_idx, fss), log_path)
                if args.isDestory:
                    # 配置自毁模型
                    global_model.setup_self_destruct(
                        fingerprint=client_fingerprint,
                        extraction_matrix=extracting_matrices[client_idx],
                        embed_layers=args.embed_layer_names,
                        threshold=0.6
                    )
                clients[client_idx].set_model(client_model)

        # 服务器分发模型
        if not args.fingerprint:
            for client in clients:
                client.set_model(copy.deepcopy(global_model))

        # 测试嵌入水印和指纹的性能
        if (epoch + 1) % args.test_interval == 0:
            # 初始化公共变量
            client_acc = []  # 存储每个客户端在测试数据集上的 Top1 准确率
            client_acc_top5 = []  # 存储每个客户端在测试数据集上的 Top5 准确率
            
            # 测试主模型准确率（所有情况都需要）
            for client_idx in range(args.num_clients):
                acc, acc_top5 = test_img(clients[client_idx].model, test_dataset, args)
                client_acc.append(acc)
                client_acc_top5.append(acc_top5)
                
                if acc >= client_acc_best[client_idx]:
                    client_acc_best[client_idx] = acc  # 更新客户端最佳准确率
            
            # 计算客户端测试准确率的统计指标
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
            
            printf("Clients Average Testing accuracy: {:.2f}".format(avg_acc), log_path)
            printf("Clients Quantile Testing accuracy, Top1: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
                min_acc, low_acc, median_acc, high_acc, max_acc), log_path)
            printf("Clients Quantile Testing accuracy, Top5: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
                min_acc_top5, low_acc_top5, median_acc_top5, high_acc_top5, max_acc_top5), log_path)
        
            # 水印测试逻辑
            if args.watermark:
                avg_watermark_acc = 0.0  # 用于累加所有客户端的水印准确率，最终计算平均值
                
                for client_idx in range(args.num_clients):
                    watermark_acc, _ = test_img(clients[client_idx].model, trigger_set, args)
                    avg_watermark_acc += watermark_acc
                
                avg_watermark_acc = avg_watermark_acc / args.num_clients
                printf("Watermark Average Testing accuracy:{:.2f}".format(avg_watermark_acc), log_path)
        
            # 指纹测试逻辑
            if args.fingerprint:
                avg_fss = 0.0  # 用于累加所有客户端的指纹提取分数（FSS），最终计算平均值
                
                for client_idx in range(args.num_clients):
                    embed_layers = get_embed_layers(clients[client_idx].model, args.embed_layer_names)
                    fss, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints, extracting_matrices)
                    avg_fss += fss
                
                avg_fss = avg_fss / args.num_clients
                printf("Average fss: {:.4f}".format(avg_fss), log_path)

        if (epoch + 1) % 10 == 0:
            torch.save(global_model.state_dict(), args.save_dir + "model_last_epochs_" + str(epoch+1) + ".pth")

    printf("Best Acc of Global Model:" + str(acc_best), log_path)
    if args.watermark:
        printf("Clients' Best Acc:", log_path)
        for client_idx in range(args.num_clients):
            printf(client_acc_best[client_idx], log_path)
        avg_acc = np.mean(client_acc_best)
        max_acc = np.max(client_acc_best)
        min_acc = np.min(client_acc_best)
        median_acc = np.median(client_acc_best)
        low_acc = np.percentile(client_acc_best, 25)
        high_acc = np.percentile(client_acc_best, 75)
        printf("Clients Average Testing accuracy: {:.2f}".format(avg_acc), log_path)
        printf("Clients Quantile Testing accuracy: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(min_acc, low_acc, median_acc, high_acc, max_acc), log_path)
    if args.save:
        torch.save(global_model.state_dict(),
                   args.save_dir + "model_last_epochs_" + str((args.epochs + args.start_epochs)) + ".pth")

    # 当需要从文件加载参数时
    if args.isParameter == True and args.watermark == False:
        trigger_set = torch.load(args.trigger_path)  # 从文件加载触发集
        local_fingerprints = np.load(args.fingerprints_path)  # 从文件加载本地指纹
        extracting_matrices = torch.load(args.matrices_path)  # 从文件加载提取矩阵

    # 根据攻击类型选择测试器
    if args.attack_type == 'finetune':
        finetune_tester = FineTuningAttackTester(
            args, global_model, clients, test_dataset,
            trigger_set, local_fingerprints, extracting_matrices
        )
        finetune_tester.run_attack()
    elif args.attack_type == 'prune':
        prune_tester = PruningAttackTester(
            args, global_model, clients, test_dataset,
            trigger_set, local_fingerprints, extracting_matrices
        )
        prune_tester.run_attack()
    elif args.attack_type == 'overwrite':
        overwrite_tester = OverwritingAttackTester(
            args, global_model, clients, test_dataset,
            trigger_set, local_fingerprints, extracting_matrices
        )
        overwrite_tester.run_attack()
    elif args.attack_type == 'neuralclense':
        cleanse_tester = NeuralCleanseAttackTester(
            args, global_model, clients, test_dataset,
            trigger_set, local_fingerprints, extracting_matrices
        )
        cleanse_tester.run_attack()



