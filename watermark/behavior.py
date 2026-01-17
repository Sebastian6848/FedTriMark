from torch.utils.data import DataLoader
import torch
from utils.test import test_img
from utils.train import set_bn_eval, get_optim, gem_train
import torch.nn as nn

def embed_behavior_watermark(global_model, trigger_set, args):
    """
    在全局模型中嵌入水印
    Args:
        global_model: 待嵌入水印的模型
        watermark_set: 水印训练数据集（DataLoader）
        trigger_set: 水印验证数据集（DataLoader）
        args: 配置参数，需包含以下字段：
            - local_optim: 优化器类型（如'SGD'）
            - lambda1: 正则化系数
            - device: 训练设备（如'cuda'）
            - freeze_bn: 是否冻结BN层
            - gem: 是否使用GEM算法
    Returns:
        水印测试准确率
    """
    watermark_set = DataLoader(trigger_set, batch_size=args.local_bs, shuffle=True)
    # 初始化水印优化器和损失函数
    watermark_optim = get_optim(global_model, args.local_optim, args.lambda1)
    watermark_loss_func = nn.CrossEntropyLoss()

    # 训练模式设置
    global_model.train()
    global_model.to(args.device)
    if args.freeze_bn:
        global_model.apply(set_bn_eval)

    # 水印嵌入训练
    for batch_idx, (images, labels) in enumerate(watermark_set):
        images, labels = images.to(args.device), labels.to(args.device)

        global_model.zero_grad()
        probs = global_model(images)
        watermark_loss = watermark_loss_func(probs, labels)
        watermark_loss.backward()

        watermark_optim.step()

    # 验证水印效果
    watermark_acc, _ = test_img(global_model, trigger_set, args)
    return watermark_acc


def compute_gradient_activity_score(global_weights, client_weights, epsilon=1e-6):
    """
    计算客户端上传模型的参数更新的“活跃度”（sparsity + 强度）
    用于评估是否只更新了水印相关参数

    Returns:
        activity_score: float ∈ [0, 1] 越大表示更新越广泛/强
    """

    total_nonzero = 0
    total_params = 0
    total_strength = 0.0

    for name in global_weights:
        old_param = global_weights[name]
        new_param = client_weights[name]

        if not isinstance(old_param, torch.Tensor):
            continue

        update = new_param - old_param

        # 将 update 转换为浮动类型
        update = update.float()  # <--- 这里转换为 float

        # 非零参数数量
        nonzero_count = torch.count_nonzero(update)
        total_nonzero += nonzero_count.item()
        total_params += update.numel()

        # 更新强度（L2范数）
        strength = torch.norm(update, p=2).item()  # norm 计算更新的 L2 范数
        total_strength += strength

    # 非零比率：代表更新是否稀疏
    sparsity_ratio = total_nonzero / (total_params + epsilon)

    # 更新强度标准化：避免过拟合于小更新
    avg_strength = total_strength / (len(global_weights) + epsilon)

    # 组合得分（可以调整权重）
    activity_score = (0.6 * sparsity_ratio) + (0.4 * (avg_strength / 10.0))  # 可调比例

    return min(activity_score, 1.0)

