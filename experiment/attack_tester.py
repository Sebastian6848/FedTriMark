import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from utils.test import test_img
from watermark.fingerprint import get_embed_layers, extracting_fingerprints


class FineTuningAttackTester:
    def __init__(self, args, global_model, clients, test_dataset, trigger_set, local_fingerprints, extracting_matrices):
        self.args = args
        self.global_model = global_model
        self.clients = clients
        self.test_dataset = test_dataset
        self.trigger_set = trigger_set

        # 加载水印数据
        self.watermark_dir = os.path.join(args.save_dir, 'watermark_data')
        self.local_fingerprints = local_fingerprints
        self.extracting_matrices = extracting_matrices

    def run_attack(self):
        """执行完整的攻击测试流程"""
        # 1. 初始化攻击模型
        attacked_models = self._init_attack_models()
        self._evaluate(attacked_models)
        # 2. 执行微调攻击
        self._execute_fine_tuning(attacked_models)
        # 3. 评估攻击效果
        self._evaluate_attack(attacked_models)

    def _init_attack_models(self):
        """初始化攻击者模型副本"""
        return [copy.deepcopy(self.global_model) for _ in range(self.args.n_adversaries)]

    def _evaluate(self, global_model):
        report = []
        report.append("{:<15} {:<15} {:<15} {:<15}".format(
            "Client", "Main Acc(%)", "WM Acc(%)", "Fingerprint"
        ))

        for idx, model in enumerate(global_model):
            # 主任务准确率
            main_acc, _ = test_img(model, self.test_dataset, self.args)

            # 水印存活率
            wm_acc, _ = test_img(model, self.trigger_set, self.args)

            # 指纹检测
            embed_layers = get_embed_layers(model, self.args.embed_layer_names)
            fss, detected_idx = extracting_fingerprints(
                embed_layers,
                self.local_fingerprints,
                self.extracting_matrices
            )

            status = "✅" if detected_idx == idx else "❌"

            report.append("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                f"Client {idx}", main_acc, wm_acc, f"{status} ({fss:.2f})"
            ))
        print("\nOrignal Model Report:")
        print("\n".join(report))

    def _execute_fine_tuning(self, attacked_models):
        """执行微调攻击核心逻辑"""
        for client_idx in tqdm(range(self.args.n_adversaries), desc="Attacking"):
            model = attacked_models[client_idx].to(self.args.device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.finetune_lr,
                momentum=0.9
            )

            # 使用对应客户端的原始数据
            # client_data = self.clients[client_idx].dataset
            # attack_loader = DataLoader(client_data, batch_size=self.args.local_bs, shuffle=True)
            attack_loader = self.clients[client_idx].dataset

            # 微调训练
            for epoch in range(self.args.attack_epochs):
                model.train()
                for data, target in attack_loader:
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.CrossEntropyLoss()(output, target)
                    loss.backward()
                    optimizer.step()

    def _evaluate_attack(self, attacked_models):
        """生成攻击评估报告"""
        report = []
        # 表头
        report.append("{:<15} {:<15} {:<15} {:<15}".format(
            "Client", "Main Acc(%)", "WM Acc(%)", "Fingerprint"
        ))

        for idx, model in enumerate(attacked_models):
            # 主任务准确率
            main_acc, _ = test_img(model, self.test_dataset, self.args)
            # 水印存活率
            wm_acc, _ = test_img(model, self.trigger_set, self.args)
            # 指纹检测
            embed_layers = get_embed_layers(model, self.args.embed_layer_names)
            fss, detected_idx = extracting_fingerprints(embed_layers, self.local_fingerprints, self.extracting_matrices)

            # 判断攻击是否成功
            attack_status = "✅" if detected_idx == idx else "❌"

            report.append("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                f"Client {idx}", main_acc, wm_acc, f"{attack_status} ({fss:.2f})"
            ))

        # 保存并打印报告
        with open(os.path.join(self.args.save_dir, 'FineTuningAttack_report.txt'), 'w') as f:
            f.write("\n".join(report))
        print("\nAttack Report:")
        print("\n".join(report))


class PruningAttackTester:
    def __init__(self, args, global_model, clients, test_dataset, trigger_set, local_fingerprints, extracting_matrices):
        self.args = args
        self.global_model = global_model
        self.clients = clients
        self.test_dataset = test_dataset
        self.trigger_set = trigger_set

        # 加载水印数据
        self.watermark_dir = os.path.join(args.save_dir, 'watermark_data')
        self.local_fingerprints = local_fingerprints
        self.extracting_matrices = extracting_matrices

    def run_attack(self):
        """执行完整的剪枝攻击测试流程"""
        # 1. 初始化攻击模型
        attacked_models = self._init_attack_models()
        self._evaluate(attacked_models)
        # 2. 执行剪枝攻击
        self._execute_pruning(attacked_models)
        # 3. 评估攻击效果
        self._evaluate_attack(attacked_models)

    def _init_attack_models(self):
        """初始化攻击者模型副本（与微调攻击相同）"""
        return [copy.deepcopy(self.global_model) for _ in range(self.args.n_adversaries)]

    def _evaluate(self, global_model):
        report = []
        report.append("{:<15} {:<15} {:<15} {:<15}".format(
            "Client", "Main Acc(%)", "WM Acc(%)", "Fingerprint"
        ))

        for idx, model in enumerate(global_model):
            # 主任务准确率
            main_acc, _ = test_img(model, self.test_dataset, self.args)

            # 水印存活率
            wm_acc, _ = test_img(model, self.trigger_set, self.args)

            # 指纹检测
            embed_layers = get_embed_layers(model, self.args.embed_layer_names)
            fss, detected_idx = extracting_fingerprints(
                embed_layers,
                self.local_fingerprints,
                self.extracting_matrices
            )

            status = "✅" if detected_idx == idx else "❌"

            report.append("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                f"Client {idx}", main_acc, wm_acc, f"{status} ({fss:.2f})"
            ))
        print("\nOrignal Model Report:")
        print("\n".join(report))

    def _execute_pruning(self, attacked_models):
        """执行剪枝攻击核心逻辑"""
        for model in tqdm(attacked_models, desc="Pruning攻击"):
            model = model.to(self.args.device)

            # 遍历所有权重参数进行剪枝
            for name, param in model.named_parameters():
                if 'weight' in name:  # 仅处理权重矩阵
                    tensor = param.data
                    total_params = tensor.numel()

                    # 计算需要保留的参数数量
                    keep_num = int(total_params * (1 - self.args.pruning_rate))
                    if keep_num == 0:
                        continue  # 防止全部参数被剪枝

                    # 获取重要性阈值（绝对值排序）
                    flat_tensor = tensor.abs().view(-1)
                    threshold = flat_tensor.kthvalue(keep_num).values

                    # 创建二进制掩码
                    mask = tensor.abs().ge(threshold).float().to(self.args.device)
                    param.data.mul_(mask)  # 应用掩码

    def _evaluate_attack(self, attacked_models):
        """生成攻击评估报告（与微调攻击相同）"""
        report = []
        report.append("{:<15} {:<15} {:<15} {:<15}".format(
            "Client", "Main Acc(%)", "WM Acc(%)", "Fingerprint"
        ))

        for idx, model in enumerate(attacked_models):
            # 主任务准确率
            main_acc, _ = test_img(model, self.test_dataset, self.args)

            # 水印存活率
            wm_acc, _ = test_img(model, self.trigger_set, self.args)

            # 指纹检测
            embed_layers = get_embed_layers(model, self.args.embed_layer_names)
            fss, detected_idx = extracting_fingerprints(
                embed_layers,
                self.local_fingerprints,
                self.extracting_matrices
            )

            # 判断攻击是否成功
            attack_status = "✅" if detected_idx == idx else "❌"

            report.append("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                f"Client {idx}", main_acc, wm_acc, f"{attack_status} ({fss:.2f})"
            ))

        # 保存并打印报告
        with open(os.path.join(self.args.save_dir, 'PruningAttack_report.txt'), 'w') as f:
            f.write("\n".join(report))
        print("\nPruning Attack Report:")
        print("\n".join(report))


class OverwritingAttackTester:
    def __init__(self, args, global_model, clients, test_dataset, trigger_set, local_fingerprints, extracting_matrices):
        self.args = args
        self.global_model = global_model
        self.clients = clients
        self.test_dataset = test_dataset
        self.trigger_set = trigger_set

        # 加载水印数据
        self.watermark_dir = os.path.join(args.save_dir, 'watermark_data')
        self.local_fingerprints = local_fingerprints
        self.extracting_matrices = extracting_matrices

        # 加载恶意模型（需提前训练）
        if hasattr(args, 'malicious_model_path'):
            self.malicious_model = torch.load(args.malicious_model_path)
        else:
            self.malicious_model = None

    def run_attack(self):
        """执行完整的覆盖攻击测试流程"""
        # 1. 初始化攻击模型
        attacked_models = self._init_attack_models()
        self._evaluate(attacked_models)
        # 2. 执行参数覆盖
        self._execute_overwriting(attacked_models)
        # 3. 评估攻击效果
        self._evaluate_attack(attacked_models)

    def _init_attack_models(self):
        """初始化攻击者模型副本"""
        return [copy.deepcopy(self.global_model) for _ in range(self.args.n_adversaries)]

    def _evaluate(self, global_model):
        report = []
        report.append("{:<15} {:<15} {:<15} {:<15}".format(
            "Client", "Main Acc(%)", "WM Acc(%)", "Fingerprint"
        ))

        for idx, model in enumerate(global_model):
            # 主任务准确率
            main_acc, _ = test_img(model, self.test_dataset, self.args)

            # 水印存活率
            wm_acc, _ = test_img(model, self.trigger_set, self.args)

            # 指纹检测
            embed_layers = get_embed_layers(model, self.args.embed_layer_names)
            fss, detected_idx = extracting_fingerprints(
                embed_layers,
                self.local_fingerprints,
                self.extracting_matrices
            )

            status = "✅" if detected_idx == idx else "❌"

            report.append("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                f"Client {idx}", main_acc, wm_acc, f"{status} ({fss:.2f})"
            ))
        print("\nOrignal Model Report:")
        print("\n".join(report))

    def _execute_overwriting(self, attacked_models):
        """执行覆盖攻击核心逻辑"""
        for model_idx in tqdm(range(len(attacked_models)), desc="Overwriting攻击"):
            target_model = attacked_models[model_idx].to(self.args.device)

            # 模式选择：完全覆盖或混合覆盖
            if self.args.overwrite_mode == 'full':
                self._full_parameter_overwrite(target_model)
            elif self.args.overwrite_mode == 'partial':
                self._partial_parameter_overwrite(target_model)
            elif self.args.overwrite_mode == 'noise':
                self._noise_injection_overwrite(target_model)
            else:
                raise ValueError(f"未知的覆盖模式: {self.args.overwrite_mode}")

    def _full_parameter_overwrite(self, model):
        """全参数覆盖：用恶意模型完全替换"""
        if self.malicious_model is None:
            raise ValueError("未配置恶意模型路径！")
        malicious_state = self.malicious_model.state_dict()
        model.load_state_dict(malicious_state)

    def _partial_parameter_overwrite(self, model):
        """部分参数覆盖：选择关键层进行覆盖"""
        self.args.overwrite_layers = ['model.bn8', 'model.bn9']
        for name, param in model.named_parameters():
            # 只覆盖指定层的参数
            if name in self.args.overwrite_layers:
                if self.malicious_model is not None:
                    # 从恶意模型复制参数
                    malicious_param = self.malicious_model.state_dict()[name]
                    param.data.copy_(malicious_param)
                else:
                    # 随机生成覆盖参数
                    new_param = torch.randn_like(param) * self.args.noise_scale
                    param.data.add_(new_param)

    def _noise_injection_overwrite(self, model):
        """噪声注入覆盖：添加干扰噪声"""
        self.args.noise_scale = 0.1  # 噪声强度
        self.args.sensitive_layers = ['model.bn8']  # 敏感层列表
        self.args.use_param_clipping = True  # 启用参数裁剪
        self.args.clip_value = 1.0  # 参数裁剪阈值
        for name, param in model.named_parameters():
            if name in self.args.sensitive_layers:
                # 高斯噪声注入
                noise = torch.randn_like(param) * self.args.noise_scale
                param.data.add_(noise)
                # 参数裁剪
                if self.args.use_param_clipping:
                    param.data.clamp_(-self.args.clip_value, self.args.clip_value)

    def _evaluate_attack(self, attacked_models):
        """生成攻击评估报告（与之前架构一致）"""
        report = []
        report.append("{:<15} {:<15} {:<15} {:<15}".format(
            "Client", "Main Acc(%)", "WM Acc(%)", "Fingerprint"
        ))

        for idx, model in enumerate(attacked_models):
            main_acc, _ = test_img(model, self.test_dataset, self.args)
            wm_acc, _ = test_img(model, self.trigger_set, self.args)

            embed_layers = get_embed_layers(model, self.args.embed_layer_names)
            fss, detected_idx = extracting_fingerprints(
                embed_layers,
                self.local_fingerprints,
                self.extracting_matrices
            )

            attack_status = "✅" if detected_idx == idx else "❌"
            report.append("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                f"Client {idx}", main_acc, wm_acc, f"{attack_status} ({fss:.2f})"
            ))

        with open(os.path.join(self.args.save_dir, 'overwrite_attack_report.txt'), 'w') as f:
            f.write("\n".join(report))
        print("\nOverwriting Attack Report:")
        print("\n".join(report))


class NeuralCleanseAttackTester:
    def __init__(self, args, global_model, clients, test_dataset, trigger_set, local_fingerprints, extracting_matrices):
        self.args = args
        self.global_model = global_model
        self.clients = clients
        self.test_dataset = test_dataset
        self.trigger_set = trigger_set

        # 加载水印数据
        self.watermark_dir = os.path.join(args.save_dir, 'watermark_data')
        self.local_fingerprints = local_fingerprints
        self.extracting_matrices = extracting_matrices

        # 清洗参数
        self.cleanse_layers = get_embed_layers(global_model, args.embed_layer_names)  # 水印嵌入层名称
        self.cleanse_epochs = 50  # 清洗训练轮次
        self.args.cleanse_lr = 1e-4  # 学习率
        self.args.lambda_param = 0.5  # 损失平衡系数

        # 高级参数
        self.args.threshold = 0.3  # 指纹相似度阈值
        self.args.trigger_ratio = 0.2  # 触发样本混合比例

    def run_attack(self):
        """执行神经清洗攻击全流程"""
        attacked_models = self._init_attack_models()
        self._evaluate(attacked_models)
        self._execute_neural_cleanse(attacked_models)
        self._evaluate_attack(attacked_models)

    def _init_attack_models(self):
        """初始化攻击模型副本"""
        return [copy.deepcopy(self.global_model) for _ in range(self.args.n_adversaries)]

    def _evaluate(self, global_model):
        report = []
        report.append("{:<15} {:<15} {:<15} {:<15}".format(
            "Client", "Main Acc(%)", "WM Acc(%)", "Fingerprint"
        ))

        for idx, model in enumerate(global_model):
            # 主任务准确率
            main_acc, _ = test_img(model, self.test_dataset, self.args)

            # 水印存活率
            wm_acc, _ = test_img(model, self.trigger_set, self.args)

            # 指纹检测
            embed_layers = get_embed_layers(model, self.args.embed_layer_names)
            fss, detected_idx = extracting_fingerprints(
                embed_layers,
                self.local_fingerprints,
                self.extracting_matrices
            )

            status = "✅" if detected_idx == idx else "❌"

            report.append("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                f"Client {idx}", main_acc, wm_acc, f"{status} ({fss:.2f})"
            ))
        print("\nOrignal Model Report:")
        print("\n".join(report))

    def _execute_neural_cleanse(self, attacked_models):
        """核心清洗算法实现"""
        for client_idx in tqdm(range(len(attacked_models)), desc="神经清洗"):
            model = attacked_models[client_idx].to(self.args.device)

            # 准备混合数据集（主任务数据+触发集）
            client_data = self.clients[client_idx].dataset
            mixed_dataset = ConcatDataset([
                client_data,
                self.trigger_set
            ])
            mixed_loader = DataLoader(
                mixed_dataset,
                batch_size=self.args.local_bs,
                shuffle=True,
                sampler=self._get_balanced_sampler()
            )

            # 双目标优化器配置
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.args.cleanse_lr,
                weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.attack_epochs
            )

            # 对抗训练循环
            for epoch in range(self.args.attack_epochs):
                model.train()
                for data, target in client_data:
                    data, target = data.to(self.args.device), target.to(self.args.device)

                    # 前向传播
                    output = model(data)

                    # 双目标损失计算
                    main_loss = torch.nn.CrossEntropyLoss()(output, target)

                    # 水印对抗损失（最小化嵌入层参数的相关性）
                    embed_loss = self._calculate_embedding_loss(model)

                    # 组合损失函数
                    total_loss = main_loss + self.args.lambda_param * embed_loss

                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()

                    # 梯度手术（只更新指定层）
                    self._gradient_surgery(model, self.cleanse_layers)

                    optimizer.step()
                scheduler.step()

    def _calculate_embedding_loss(self, model):
        """计算嵌入层对抗损失"""
        loss = 0
        for layer_name in self.cleanse_layers:
            layer = dict(model.named_modules())[layer_name]
            if isinstance(layer, torch.nn.Linear):
                # 最小化权重矩阵的Frobenius范数
                loss += torch.norm(layer.weight, p='fro')
            elif isinstance(layer, torch.nn.Conv2d):
                # 破坏卷积核的通道相关性
                kernels = layer.weight.view(layer.out_channels, -1)
                gram_matrix = torch.mm(kernels, kernels.t())
                loss += torch.norm(gram_matrix, p='nuc')  # 核范数
        return loss

    def _gradient_surgery(self, model, target_layers):
        """梯度手术：仅保留指定层的梯度"""
        for name, param in model.named_parameters():
            if not any(layer in name for layer in target_layers):
                param.grad = None  # 冻结非目标层梯度

    def _get_balanced_sampler(self):
        """创建平衡采样器（控制触发样本比例）"""
        # 实现细节根据具体数据集结构调整
        # 返回一个能维持触发集比例的采样器

    def _evaluate_attack(self, attacked_models):
        """评估报告生成（保持统一格式）"""
        report = []
        report.append("{:<15} {:<15} {:<15} {:<15}".format(
            "Client", "Main Acc(%)", "WM Acc(%)", "Fingerprint"
        ))

        for idx, model in enumerate(attacked_models):
            main_acc, _ = test_img(model, self.test_dataset, self.args)
            wm_acc, _ = test_img(model, self.trigger_set, self.args)

            embed_layers = get_embed_layers(model, self.args.embed_layer_names)
            fss, detected_idx = extracting_fingerprints(
                embed_layers,
                self.local_fingerprints,
                self.extracting_matrices
            )

            # 判断清洗成功标准：指纹相似度低于阈值或检测失败
            attack_status = "✅" if fss < self.args.threshold or detected_idx == -1 else "❌"

            report.append("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                f"Client {idx}", main_acc, wm_acc,
                f"{attack_status} ({fss:.2f}/{self.args.threshold})"
            ))

        with open(os.path.join(self.args.save_dir, 'neural_cleanse_report.txt'), 'w') as f:
            f.write("\n".join(report))
        print("\nNeural Cleanse Report:")
        print("\n".join(report))