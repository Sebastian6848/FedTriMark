# -*- coding: UTF-8 -*-
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def high_pass_filter(images):
    new_images = []
    for i in range(images.shape[0]):
        image = images[i]
        rows, cols = image.shape
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        crows = rows // 2
        ccols = cols // 2
        fshift[(crows - rows // 10):(crows + rows // 10), (ccols - cols // 10):(ccols + cols // 10)] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        new_images.append(img_back)
    return np.array(new_images)


class NumpyLoader(Dataset):

    def __init__(self, x, y, transformer=None):
        self.x = x
        self.y = y
        self.transformer = transformer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        image = self.x[item]
        label = self.y[item]
        if self.transformer is not None:
            image = self.transformer(image)
        return image, label


'''
1. 从模板库取出模板pattern，并且按照要求调整模板尺寸为args.image_size，调整模板色彩模式为黑白orRGB
2. 使用每个模板生成相同数量的触发集（num_trigger_each_class），用于不同class，例如cifar-10有10个class，num_trigger_set为100，那么每个class分配10个触发集
3. 在模板pattern上加入随机噪声（np.random.randint），类似于给华夫饼撒糖霜（使得每个模板生成的图像有细微不同，防止模型死记硬背）
4. 使用每个pattern模板生成num_trigger_each_class个华夫饼，并且给同一模板制作的华夫饼打上标签label
4. 最后计算触发集的全局均值和全局标准差，命令行输出
'''
def generate_waffle_pattern(args):
    # np.random.seed(0)
    path = "./data/pattern/"

    base_patterns = []
    for i in range(args.num_classes):
        pattern_path = os.path.join(path, "{}.png".format(i))
        pattern = Image.open(pattern_path)

        if args.num_channels == 1:
            pattern = pattern.convert("L")
        else:
            pattern = pattern.convert("RGB")
        pattern = np.array(pattern)
        pattern = np.resize(pattern, (args.image_size, args.image_size, args.num_channels))
        base_patterns.append(pattern)

    trigger_set = []
    trigger_set_labels = []
    label = 0
    num_trigger_each_class = args.num_trigger_set // args.num_classes
    for pattern in base_patterns:
        for _ in range(num_trigger_each_class):
            image = (pattern + np.random.randint(0, 255, (args.image_size, args.image_size, args.num_channels)))\
                        .astype(np.float32) / 255 / 2
            trigger_set.append(image)
            trigger_set_labels.append(label)
        label += 1
    trigger_set = np.array(trigger_set)
    trigger_set_labels = np.array(trigger_set_labels)
    trigger_set_mean = np.mean(trigger_set, axis=(0, 1, 2))
    trigger_set_std = np.std(trigger_set, axis=(0, 1, 2))
    #print(trigger_set_mean, trigger_set_std)
    print("[Analysis] trigger_set_mean:", trigger_set_mean, "| trigger_set_std:", trigger_set_std)

    dataset = NumpyLoader(trigger_set, trigger_set_labels, transformer=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(trigger_set_mean, trigger_set_std)
                                ]))
    return dataset
