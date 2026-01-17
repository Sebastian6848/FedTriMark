# -*- coding: UTF-8 -*-
import argparse
import distutils.util

# 输出内容，若给定路径则写文件，无路径则控制台输出
def printf(content, path=None):
    if path is None:
        print(content)
    else:
        with open(path, 'a+') as f:
            print(content, file=f)

# 解析命令行参数，并返回一个包含所有参数的对象
def load_args():
    parser = argparse.ArgumentParser()
    # 全局设定
    parser.add_argument('--start_epochs', type=int, default=0, help='start epochs (only used in save model)')
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=10, help="number of clients: K")
    parser.add_argument('--clients_percent', type=float, default=0.4, help="the fraction of clients to train the local models in each iteration.")
    parser.add_argument('--pre_train', type=lambda x: bool(distutils.util.strtobool(x)), default=False, help="Intiate global model with pre-trained weight.")
    parser.add_argument('--pre_train_path', type=str, default='./result/CNN4/model_last_epochs_30.pth')
    parser.add_argument('--model_dir', type=str, default='./result/final/VGG16/10-4/')

    # 客户端本地设定
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--local_optim', type=str, default='sgd', help="local optimizer")
    parser.add_argument('--local_lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--local_momentum', type=float, default=0, help="SGD momentum (default: 0)")
    parser.add_argument('--local_loss', type=str, default="CE", help="Loss Function")
    parser.add_argument('--distribution', type=str, default='iid', help="the distribution used to split the dataset")
    parser.add_argument('--dniid_param', type=float, default=0.8)
    parser.add_argument('--lr_decay', type=float, default=0.999)

    # 测试相关设置
    parser.add_argument('--test_bs', type=int, default=512, help="test batch size")
    parser.add_argument('--test_interval', type=int, default=1)

    # 神经网络相关
    parser.add_argument('--model', type=str, default='AlexNet', help='model name')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    # 其他参数
    parser.add_argument('--classes_per_client', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--image_size', type=int, default=32, help="length or width of images")
    parser.add_argument('--gpu', type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default="./result/test/")
    parser.add_argument('--save', type=lambda x: bool(distutils.util.strtobool(x)), default=True)

    # 水印相关
    parser.add_argument("--watermark", type=lambda x: bool(distutils.util.strtobool(x)), default=True, help="whether embedding the watermark")
    parser.add_argument("--fingerprint", type=lambda x: bool(distutils.util.strtobool(x)), default=True, help="whether to embed the fingerprints")
    parser.add_argument('--lfp_length', type=int, default=128, help="Bit length of local fingerprints")
    parser.add_argument('--num_trigger_set', type=int, default=100, help='number of images used as trigger set')
    parser.add_argument('--num_trigger_each_class', type=int, default=5, help='number of images used as trigger set')
    parser.add_argument('--embed_layer_names', type=str, default='model.bn8')
    parser.add_argument('--freeze_bn', type=lambda x: bool(distutils.util.strtobool(x)), default=True)
    parser.add_argument('--change_fingerprint', type=int, default=100, help="多少轮更换一次更换模型指纹")

    parser.add_argument('--lambda1', type=float)
    parser.add_argument('--watermark_max_iters', type=int, default=100)
    parser.add_argument('--fingerprint_max_iters', type=int, default=5)
    parser.add_argument('--lambda2', type=float)
    parser.add_argument('--gem', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help="whether to use the CL-based watermark embedding methods.")

    # 攻击测试
    parser.add_argument('--attack_type', type=str, default='None', help='采取攻击类型[None|finetune|prune|overwrite|neuralcleanse]')
    parser.add_argument('--n_adversaries', type=int, default=3, help='恶意客户端数量')
    parser.add_argument('--finetune_lr', type=float, default=0.01, help='微调学习率')
    parser.add_argument('--pruning_rate', type=float, default=0.5, help='剪枝权重参数')
    parser.add_argument('--overwrite_mode', type=str, default='noise', help='覆盖攻击模式[noise|full|partial]')
    parser.add_argument('--attack_epochs', type=int, default=10, help='攻击训练轮次')

    # 保存参数地址
    parser.add_argument('--isParameter', type=lambda x: bool(distutils.util.strtobool(x)), default=False)
    parser.add_argument('--trigger_path', type=str, default="./result/CNN4/watermark_data/trigger_set.pth")
    parser.add_argument('--fingerprints_path', type=str, default="./result/CNN4/watermark_data/local_fingerprints.npy")
    parser.add_argument('--matrices_path', type=str, default="./result/CNN4/watermark_data/extracting_matrices.pth")

    # 对抗训练参数
    parser.add_argument('--adv_train', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help='是否继续对抗性训练')
    parser.add_argument('--adv_eps', type=float, default=0.03, help='FGSM attack epsilon for adversarial training')
    parser.add_argument('--wm_clean_lambda', type=float, default=1.0)
    parser.add_argument('--wm_adv_lambda', type=float, default=0.5)
    parser.add_argument('--wm_consistency_lambda', type=float, default=0.1)

    # 动态调整：固定λ2可能导致某些客户端指纹嵌入不足（如数据复杂度过高）或破坏模型性能（如数据过于简单）。
    parser.add_argument('--dynamic', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help='是否动态调整参数')

    # 自毁模型
    parser.add_argument('--isDestory', type=lambda x: bool(distutils.util.strtobool(x)), default=False)



    args = parser.parse_args()
    # 每次选多少个客户端参与训练
    args.num_clients_each_iter = int(args.num_clients * args.clients_percent)
    return args
