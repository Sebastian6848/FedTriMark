# -*- coding: UTF-8 -*-
from torch.utils.data import ConcatDataset, DataLoader
import json
from utils.train import *
from utils.datasets import *

from trigger.generate_pattern import *
from trigger.generate_waffle_pattern import generate_waffle
from watermark.behavior import *
from watermark.fingerprint import get_embed_layers


class Client:
    def __init__(self):
        self.model = None
        self.dataset = None

    def set_model(self, model):
        self.model = model

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_model(self):
        return self.model

    def get_dataset(self):
        return self.dataset

    def train_one_iteration(self):
        pass

class OrdinaryClient(Client):
    def __init__(self, args, dataset=None, idx=None, watermark_dataset=None):
        super().__init__()
        # self.model = get_model(args)
        self.loss = get_loss(args.local_loss)
        self.ep = args.local_ep
        self.device = args.device
        self.local_optim = args.local_optim
        self.local_lr = args.local_lr
        self.local_momentum = args.local_momentum
        self.embed_layer_names = args.embed_layer_names
        self.bwm_acc = 50
        # self.local_wd = args.local_wd
        self.dataset = DataLoader(DatasetSplit(dataset, idx), batch_size=args.local_bs, shuffle=True)  # 使用 DatasetSplit 和 DataLoader 创建客户端的数据集，支持分批加载数据


    '''
    1. 把“大脑”调整到学习模式，并准备好学习工具（优化器）。
    2. 按照学习计划，进行多轮学习（self.ep 轮）。
    3. 每轮学习中，它会从“练习册”中拿出一部分题目（一个批次的数据），认真思考（计算模型输出），检查错误（计算损失），并改正错误（反向传播更新参数）。
    4. 记录每轮的学习效果（平均损失）。
    5. 学习结束后，把“大脑”整理好，并返回学习成果（模型参数、数据集大小和平均损失）。
    '''
    def train_one_iteration(self):
        self.model.train()
        self.model = self.model.to(self.device)
        epoch_loss = []
        optim = get_optim(self.model, self.local_optim, self.local_lr, self.local_momentum)

        for _ in range(self.ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.dataset):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                probs = self.model(images)
                loss = self.loss(probs, labels)
                loss.backward()
                optim.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.model = self.model.cpu()
        return self.model.state_dict(), len(self.dataset.dataset), sum(epoch_loss) / len(epoch_loss)


'''
1. 根据参数 args.distribution 决定数据分布方式（如独立同分布 iid 或非独立同分布 dniid、pniid）。
2. 使用 iid_split、dniid_split 或 pniid_split 将数据集划分为多个子集，每个子集对应一个客户端。
3. 为每个客户端创建一个 OrdinaryClient 实例，并将其添加到客户端列表中。
'''
def create_clients(args, dataset):
    if args.distribution == 'iid':
        idxs = iid_split(dataset, args.num_clients)
    elif args.distribution == 'dniid':
        idxs = dniid_split(dataset, args.num_clients, args.dniid_param)
    elif args.distribution == 'pniid':
        idxs = pniid_split(dataset, args.num_clients)
    else:
        exit("Unknown Distribution!")
    clients = []
    for idx in idxs.values():
        client = OrdinaryClient(args, dataset, idx)
        clients.append(client)
    return clients

# 为指定的客户端生成触发样本子集
def client_generate_trigger_subset(client_id, args, plan_path="./trigger/trigger_plan/client_to_classes.json", save_dir="./trigger/upload_buffer"):
    with open(plan_path) as f:
        plan = json.load(f)
    class_list = plan[str(client_id)]

    pattern_path = os.path.join(args.save_dir, 'pattern')
    create_pattern(client_id=client_id, save_path=pattern_path, class_list=class_list)

    args.class_list = class_list  # 给 generate_waffle_pattern 使用
    generate_waffle(args, client_id)
