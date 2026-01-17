# FedTriMark

## 简介 / Overview
FedTriMark 是一个用于联邦学习场景下水印/指纹与攻击实验的研究型代码仓库，包含数据准备、训练、攻击与评测等模块。

FedTriMark is a research-oriented codebase for watermark/fingerprint and attack experiments in federated learning, including data preparation, training, attack, and evaluation modules.

## 功能概览 / Key Features
- 水印/指纹生成与验证 / Watermark/Fingerprint generation and verification
- 多种攻击与防护实验 / Multiple attack and defense experiments
- 常见数据集支持（MNIST、CIFAR-10、CIFAR-100）/ Common datasets support (MNIST, CIFAR-10, CIFAR-100)
- 训练与评测脚本 / Training and evaluation scripts

## 项目结构 / Project Structure
- [main.py](main.py): 主入口 / Main entry
- [Attack.py](Attack.py): 攻击相关入口 / Attack entry
- data/: 数据与模式生成 / Data and pattern generation
- experiment/: 实验与测试脚本 / Experiments and testers
- fed/: 联邦学习核心逻辑 / Federated learning core
- script/: 常用训练与攻击脚本 / Training and attack scripts
- security/: 对抗攻击与防护 / Adversarial attack/defense
- trigger/: 触发器/模式生成 / Trigger & pattern generation
- utils/: 通用工具与训练测试 / Utilities, training and testing
- watermark/: 水印/指纹实现 / Watermark & fingerprint implementations

## 环境依赖 / Requirements
- Python 3.8+（建议 3.9 或 3.10）/ Python 3.8+ (recommended 3.9 or 3.10)
- 依赖见 [requirements.txt](requirements.txt)

## 快速开始 / Quick Start
1. 安装依赖 / Install dependencies
   - `pip install -r requirements.txt`
2. 准备数据集 / Prepare datasets
   - 将 MNIST/CIFAR 数据放置在 `data/` 目录下（见现有示例）
3. 运行主程序 / Run main
   - `python main.py`
4. 运行攻击实验 / Run attack experiments
   - `python Attack.py`

> 若使用脚本，请参考 `script/` 目录中的示例。
> For scripts, see examples under `script/`.

## 数据集 / Datasets
- MNIST: `data/mnist/`
- CIFAR-10: `data/cifar10/`
- CIFAR-100: `data/cifar100/`

## 实验说明 / Experiments
- 相关实验脚本位于 `experiment/` 目录。
- 触发器与模式生成在 `trigger/` 和 `data/` 中。

## 备注 / Notes
- 本仓库用于学术研究与实验复现。
- 如需调整训练参数，请查看 `utils/` 与 `fed/` 中的相关配置。

## 许可 / License
若未特别说明，默认遵循学术研究用途。请在使用前确认许可证要求。
If not otherwise specified, this project is intended for academic research. Please check licensing requirements before use.

This project partially references：
@article{shao2024fedtracker,
  title={Fedtracker: Furnishing Ownership Verification and Traceability for Federated Learning Model},
  author={Shao, Shuo and Yang, Wenyuan and Gu, Hanlin and Qin, Zhan and Fan, Lixin and Yang, Qiang and Ren, Kui},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2024}
}
