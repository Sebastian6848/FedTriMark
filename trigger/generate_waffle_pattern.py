# generate_waffle_pattern.py
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset

# data_loader
class NumpyLoader(Dataset):
    def __init__(self, data, labels, transformer=None):
        self.data = data
        self.labels = labels
        self.transformer = transformer

    def __getitem__(self, idx):
        x = self.data[idx]  # shape: (H, W, C)
        y = self.labels[idx]

        # Apply transformations if any
        if self.transformer:
            x = self.transformer(x)

        return x, y

    def __len__(self):
        return len(self.data)

# 生成华夫饼图案
def generate_waffle(args, client_id):
    path = os.path.join(args.save_dir, 'pattern')
    base_patterns = []

    # 遍历 class_list 生成每个 class_id 的 pattern
    for class_id in args.class_list:
        pattern_path = os.path.join(path, f"{client_id}_{class_id}.png")

        # 确保文件存在
        if not os.path.exists(pattern_path):
            print(f"[Warning] Pattern for class {class_id} not found for client {args.client_id}. Skipping...")
            continue

        # 加载图像
        pattern = Image.open(pattern_path)
        pattern = pattern.convert("L" if args.num_channels == 1 else "RGB")
        pattern = np.array(pattern)
        pattern = np.resize(pattern, (args.image_size, args.image_size, args.num_channels))

        base_patterns.append((class_id, pattern))

    trigger_set = []
    trigger_set_labels = []

    # 生成触发集
    for label, pattern in base_patterns:
        for _ in range(args.num_trigger_each_class):
            image = (pattern + np.random.randint(0, 255, (args.image_size, args.image_size, args.num_channels))) \
                        .astype(np.float32) / 255 / 2
            trigger_set.append(image)
            trigger_set_labels.append(label)
    trigger_set = np.array(trigger_set)
    trigger_set_labels = np.array(trigger_set_labels)

    # Print the shape of the trigger set
    print(f"[Debug] Trigger set shape (before saving): {trigger_set.shape}")
    np.save(f"./trigger/upload_buffer/client_{client_id}_data.npy", trigger_set)
    np.save(f"./trigger/upload_buffer/client_{client_id}_labels.npy", trigger_set_labels)

def generate_behavior_trigger(args, img, class_id):
    base_patterns = []

    # 加载图像
    pattern = img
    pattern = pattern.convert("L" if args.num_channels == 1 else "RGB")
    pattern = np.array(pattern)
    pattern = np.resize(pattern, (args.image_size, args.image_size, args.num_channels))

    base_patterns.append((class_id, pattern))

    trigger_set = []
    trigger_set_labels = []

    # 生成触发集
    for label, pattern in base_patterns:
        for _ in range(args.num_trigger_each_class):
            image = (pattern + np.random.randint(0, 255, (args.image_size, args.image_size, args.num_channels))) \
                        .astype(np.float32) / 255 / 2
            trigger_set.append(image)
            trigger_set_labels.append(label)

    trigger_set = np.array(trigger_set)
    trigger_set_labels = np.array(trigger_set_labels)
    trigger_set_mean = np.mean(trigger_set, axis=(0, 1, 2))
    trigger_set_std = np.std(trigger_set, axis=(0, 1, 2))
    # print(trigger_set_mean, trigger_set_std)
    #print("[Analysis] trigger_set_mean:", trigger_set_mean, "| trigger_set_std:", trigger_set_std)

    dataset = NumpyLoader(trigger_set, trigger_set_labels, transformer=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trigger_set_mean, trigger_set_std)
    ]))
    return dataset
