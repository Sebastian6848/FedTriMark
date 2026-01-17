import numpy as np
from PIL import Image, ImageDraw
import random
import math
import os


def generate_line_pattern(width, height, line_width=3):
    """生成随机方向的粗直线模式"""
    img = Image.new('F', (width, height), 0)
    draw = ImageDraw.Draw(img)

    angle = random.uniform(0, 2 * math.pi)
    length = max(width, height) * 0.8
    x_shift = random.uniform(-0.5, 0.5) * width
    y_shift = random.uniform(-0.5, 0.5) * height

    center_x = width / 2 + x_shift
    center_y = height / 2 + y_shift
    dx = length / 2 * math.cos(angle)
    dy = length / 2 * math.sin(angle)
    start = (center_x - dx, center_y - dy)
    end = (center_x + dx, center_y + dy)

    draw.line([start, end], fill=1.0, width=line_width)
    return np.array(img)


def generate_arc_pattern(width, height, line_width=3):
    """生成随机粗弧线模式"""
    img = Image.new('F', (width, height), 0)
    draw = ImageDraw.Draw(img)

    size = random.uniform(0.2, 0.4) * min(width, height)
    x_center = random.uniform(0, width)
    y_center = random.uniform(0, height)
    start_angle = random.uniform(0, 360)
    end_angle = start_angle + random.uniform(60, 270)

    left = x_center - size / 2
    top = y_center - size / 2
    right = x_center + size / 2
    bottom = y_center + size / 2

    draw.arc([left, top, right, bottom],
             int(start_angle), int(end_angle),
             fill=1.0, width=line_width)
    return np.array(img)


def create_pattern(client_id, class_list, save_path, img_size=(32, 32), line_width=5):
    """
    为客户端生成多个 class_id 的图案图（每类图案一张图）
    :param client_id: 客户ID（仅用于日志）
    :param class_list: 客户端负责生成的类ID列表
    :param save_path: 图案存储路径
    :param img_size: 图像尺寸（H, W）
    :param line_width: 图案线宽
    """
    os.makedirs(save_path, exist_ok=True)
    width, height = img_size

    patterns = [lambda w, h: generate_line_pattern(w, h, line_width),
                lambda w, h: generate_arc_pattern(w, h, line_width)]

    for class_id in class_list:
        canvas = np.zeros((height, width), dtype=np.float32)
        for _ in range(6):
            pattern = random.choice(patterns)(width, height)
            canvas += pattern

        binary_mask = (canvas > 0.5)
        pattern_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

        img_array = np.full((height, width, 3), 255, dtype=np.uint8)
        img_array[binary_mask] = pattern_color

        img = Image.fromarray(img_array, 'RGB')
        filename = f"{client_id}_{class_id}.png"
        save_file = os.path.join(save_path, filename)
        img.save(save_file)

    print(f"[Client {client_id}] 生成图案完毕")

def create_behavior_pattern(img_size=(32, 32), line_width=5):
    """
    生成单张随机图案图像
    :param img_size: 图像尺寸 (width, height)
    :param line_width: 线条粗细
    :return: PIL.Image 对象
    """
    width, height = img_size
    patterns = [lambda w, h: generate_line_pattern(w, h, line_width),
                lambda w, h: generate_arc_pattern(w, h, line_width)]

    canvas = np.zeros((height, width), dtype=np.float32)
    for _ in range(6):  # 叠加6次图案
        pattern = random.choice(patterns)(width, height)
        canvas += pattern

    binary_mask = (canvas > 0.5)  # 二值化
    pattern_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    img_array = np.full((height, width, 3), 255, dtype=np.uint8)  # 白色背景
    img_array[binary_mask] = pattern_color  # 填充图案颜色

    return Image.fromarray(img_array, 'RGB')