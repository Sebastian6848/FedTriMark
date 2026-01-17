# -*- coding: UTF-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from security.destory_hook import *


from utils.utils import load_args
class AlexNet(SelfDestructMixin, nn.Module):
    def __init__(self, args):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(args.num_channels, 64, kernel_size=11, stride=4, padding=2)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('bn2', nn.BatchNorm2d(192)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(384)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool3', nn.AdaptiveAvgPool2d((6, 6))),  # 使用自适应池化层
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(0.5)),
            ('fc1', nn.Linear(256 * 6 * 6, 4096)),
            ('relu6', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout(0.5)),
            ('fc2', nn.Linear(4096, 4096)),
            ('relu7', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(4096, args.num_classes)),
        ]))
        self.memory = dict()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)

class VGG16(SelfDestructMixin, nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(args.num_channels, 64, 3, padding="same", bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 64, 3, padding="same", bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('pool1', nn.MaxPool2d((2, 2), (2, 2))),
            ('dropout1', nn.Dropout(0.25)),
            ('conv3', nn.Conv2d(64, 128, 3, padding="same", bias=False)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(128, 128, 3, padding="same", bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU()),
            ('pool2', nn.MaxPool2d((2, 2), (2, 2))),
            ('dropout2', nn.Dropout(0.25)),
            ('conv5', nn.Conv2d(128, 256, 3, padding="same", bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU()),
            ('conv6', nn.Conv2d(256, 256, 3, padding="same", bias=False)),
            ('bn6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU()),
            ('conv7', nn.Conv2d(256, 256, 3, padding="same", bias=False)),
            ('bn7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU()),
            ('pool3', nn.MaxPool2d((2, 2), (2, 2))),
            ('dropout3', nn.Dropout(0.25)),
            ('conv8', nn.Conv2d(256, 512, 3, padding="same", bias=False)),
            ('bn8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU()),
            ('conv9', nn.Conv2d(512, 512, 3, padding="same", bias=False)),
            ('bn9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU()),
            ('conv10', nn.Conv2d(512, 512, 3, padding="same", bias=False)),
            ('bn10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU()),
            ('pool4', nn.MaxPool2d((2, 2), (2, 2))),
            ('dropout4', nn.Dropout(0.25)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ]))
        self.fc = nn.Linear(512, args.num_classes)
        self.memory = dict()

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.shape[0], -1)
        return self.fc(output)

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            # self.memory = dict()
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)

class CNN4(SelfDestructMixin, nn.Module):
    def __init__(self, args):
        super().__init__()
        self.extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(args.num_channels, 64, kernel_size=3)),  # 卷积层1 对扰动敏感
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d((2, 2))),   # 池化层 水印信息容易丢失
            ('conv2', nn.Conv2d(64, 128, kernel_size=3)),  # 卷积层2 推荐嵌入
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d((2, 2)))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4608, 512)),  # 全连接层1 次优选择
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(512, args.num_classes))  # 全连接层2 or 输出层 对模型输出影响大
        ]))
        self.memory = dict()

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)


class ResidualBlock(SelfDestructMixin, nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(outchannel)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(outchannel))
        ]))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(SelfDestructMixin, nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1,1],padding=[0,1,0],first=False) -> None:
        super(Bottleneck,self).__init__()
        self.bottleneck = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False)),
            ('bn3', nn.BatchNorm2d(out_channels*4))
        ]))

        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels*4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(SelfDestructMixin, nn.Module):
    def __init__(self, ResidualBlock, args):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, args.num_classes)
        self.memory = dict()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = OrderedDict()
        count = 1
        for stride in strides:
            name = "layer{}".format(count)
            layers[name] = block(self.in_channel, channels, stride)
            self.in_channel = channels
            count += 1
        return nn.Sequential(layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            # self.memory = dict()
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)

class LeNet5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.extractor = nn.Sequential(OrderedDict([
            # LeNet-5 原始架构中的第一层卷积：输入 1x32x32 -> 输出 6x28x28
            ('conv1', nn.Conv2d(args.num_channels, 6, kernel_size=5)),
            ('relu1', nn.Tanh()),
            ('pool1', nn.AvgPool2d(kernel_size=2, stride=2)),  # 输出 6x14x14

            # 第二层卷积：6x14x14 -> 16x10x10
            ('conv2', nn.Conv2d(6, 16, kernel_size=5)),
            ('relu2', nn.Tanh()),
            ('pool2', nn.AvgPool2d(kernel_size=2, stride=2))   # 输出 16x5x5
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            # LeNet-5 使用 16x5x5 = 400 个输入神经元
            ('fc1', nn.Linear(16 * 5 * 5, 120)),
            ('relu3', nn.Tanh()),
            ('fc2', nn.Linear(120, 84)),
            ('relu4', nn.Tanh()),
            ('fc3', nn.Linear(84, args.num_classes))  # 输出层
        ]))
        self.memory = dict()

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)

def ResNet18(args):
    return ResNet(ResidualBlock, args)

def get_model(args):
    if args.model == 'VGG16':
        return VGG16(args)
    elif args.model == 'LeNet5':
        return LeNet5(args)
    elif args.model == 'CNN4':
        return CNN4(args)
    elif args.model == 'ResNet18':
        return ResNet18(args)
    elif args.model == 'AlexNet':
        return AlexNet(args)
    else:
        exit("Unknown Model!")

