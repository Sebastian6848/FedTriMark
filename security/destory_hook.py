import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
from typing import Tuple


def get_embed_layers(model: nn.Module, embed_layer_names: str) -> List[nn.Module]:
    """根据点分路径获取模型中的指定层

    Args:
        model: 目标神经网络模型
        embed_layer_names: 分号分隔的层路径字符串，支持数字索引
            示例: "features.0;classifier.3"
    """
    embed_layers = []
    for path in embed_layer_names.split(";"):
        current_module = model
        try:
            # 逐级解析模块路径
            for part in path.split('.'):
                # 尝试作为属性访问
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                # 尝试作为数字索引访问
                elif part.isdigit() and isinstance(current_module, nn.ModuleList):
                    current_module = current_module[int(part)]
                else:
                    raise ValueError(f"无效的层路径: {path}")
            embed_layers.append(current_module)
        except Exception as e:
            raise RuntimeError(f"解析层路径失败: {path}") from e
    return embed_layers


class SelfDestructMixin:
    """自毁水印功能混入类"""

    def setup_self_destruct(
            self,
            fingerprint: np.ndarray,
            extraction_matrix: np.ndarray,
            embed_layers: Union[str, List[Union[str, nn.Module]]],
            threshold: float = 0.6,
            use_hd: bool = True,
            check_frequency: int = 100
    ):
        """配置自毁机制

        Args:
            fingerprint: 预期指纹 (-1/1 格式)
            extraction_matrix: 指纹提取矩阵
            embed_layers: 可以是以下格式：
                - 字符串: 分号分隔的层路径 (使用get_embed_layers解析)
                - 列表: 混合包含层对象或层路径字符串
            threshold: BER阈值 (0-1) 或得分阈值
            use_hd: 是否使用汉明距离验证
            check_frequency: 每N次前向传播检查一次
        """
        # 解析嵌入层
        final_layers = []
        if isinstance(embed_layers, str):
            final_layers = get_embed_layers(self, embed_layers)
        elif isinstance(embed_layers, list):
            for item in embed_layers:
                if isinstance(item, str):
                    final_layers.extend(get_embed_layers(self, item))
                elif isinstance(item, nn.Module):
                    final_layers.append(item)
        else:
            raise TypeError("embed_layers应为字符串或层对象列表")

        # 参数初始化
        self.fingerprint = fingerprint
        self.extraction_matrix = extraction_matrix
        self.embed_layers = final_layers
        self.self_destruct_threshold = threshold
        self.use_hd_verification = use_hd
        self.check_frequency = max(1, check_frequency)
        self.forward_counter = 0
        self.is_destroyed = False
        self._hook_handle = self.register_forward_pre_hook(self._self_destruct_hook)

    def _self_destruct_hook(self, module, inputs):
        """前向传播钩子函数"""
        if self.is_destroyed or not self.training:
            return

        self.forward_counter += 1
        if self.forward_counter % self.check_frequency == 0:
            verification_result, score = self._verify_fingerprint()
            if not verification_result:
                self.destroy_model(score)

    def _verify_fingerprint(self) -> Tuple[bool, float]:
        """执行指纹验证，返回验证结果和得分（兼容多个候选指纹+提取矩阵）"""
        try:
            # 提取模型中嵌入层的所有权重参数
            weight_vectors = []
            for layer in self.embed_layers:
                for param in layer.parameters():
                    weight_vectors.append(param.detach().cpu().numpy().flatten())
            weight_vector = np.concatenate(weight_vectors)

            # 检查是否是多个候选指纹 + 提取矩阵
            multiple_candidates = (
                    isinstance(self.fingerprint, list) or isinstance(self.fingerprint,
                                                                     np.ndarray) and self.fingerprint.ndim == 2
            )

            if multiple_candidates:
                # 多个候选指纹，使用扩展验证逻辑
                fingerprints = self.fingerprint  # shape: (N, L)
                matrices = self.extraction_matrix  # shape: (N, L, D)
                bit_length = fingerprints[0].shape[0]
                min_ber = float("inf")
                max_score = float("-inf")

                for idx in range(len(fingerprints)):
                    fp = fingerprints[idx]
                    mat = matrices[idx]
                    dim = min(mat.shape[1], len(weight_vector))
                    result = np.dot(mat[:, :dim], weight_vector[:dim])

                    if self.use_hd_verification:
                        result[result >= 0] = 1
                        result[result < 0] = -1
                        ber = np.mean(result != fp[:len(result)])
                        if ber < min_ber:
                            min_ber = ber
                    else:
                        score_vec = np.multiply(result, fp[:len(result)])
                        score_vec[score_vec > 0.5] = 0.5
                        score = np.sum(score_vec) / bit_length / 0.5
                        if score > max_score:
                            max_score = score

                if self.use_hd_verification:
                    return min_ber <= self.self_destruct_threshold, min_ber
                else:
                    return max_score >= self.self_destruct_threshold, max_score
            else:
                # 单个指纹 + 提取矩阵（原逻辑）
                fp = self.fingerprint
                mat = self.extraction_matrix
                dim = min(mat.shape[1], len(weight_vector))
                result = np.dot(mat[:, :dim], weight_vector[:dim])

                if self.use_hd_verification:
                    bits = np.sign(result)
                    ber = np.mean(bits != fp[:len(bits)])
                    return ber <= self.self_destruct_threshold, ber
                else:
                    score_vec = np.multiply(result, fp[:len(result)])
                    score_vec[score_vec > 0.5] = 0.5
                    score = np.sum(score_vec) / fp.shape[0] / 0.5
                    return score >= self.self_destruct_threshold, score

        except Exception as e:
            print(f"指纹验证错误: {e}")
            return False, 0.0

    def destroy_model(self, score: float):
        """破坏模型参数并输出得分"""
        print(f"\033[91m! 触发自毁机制 ! 当前得分: {score:.4f}\033[0m")
        with torch.no_grad():
            for param in self.parameters():
                param.data = torch.randn_like(param.data)
        self.is_destroyed = True
        if self._hook_handle:
            self._hook_handle.remove()





