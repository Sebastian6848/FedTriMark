import numpy as np
from PIL import Image, ImageDraw
import random
import math
import os


def generate_line_pattern(width, height, line_width=3):
    """生成随机方向的粗直线模式"""
    img = Image.new('F', (width, height), 0)
    draw = ImageDraw.Draw(img)

    # 随机参数
    angle = random.uniform(0, 2 * math.pi)
    length = max(width, height) * 0.8
    x_shift = random.uniform(-0.5, 0.5) * width
    y_shift = random.uniform(-0.5, 0.5) * height

    # 计算线段端点
    center_x = width / 2 + x_shift
    center_y = height / 2 + y_shift
    dx = length / 2 * math.cos(angle)
    dy = length / 2 * math.sin(angle)
    start = (center_x - dx, center_y - dy)
    end = (center_x + dx, center_y + dy)

    draw.line([start, end], fill=1.0, width=line_width)  # 修改线条宽度
    return np.array(img)


def generate_arc_pattern(width, height, line_width=3):
    """生成随机粗弧线模式"""
    img = Image.new('F', (width, height), 0)
    draw = ImageDraw.Draw(img)

    # 随机参数
    size = random.uniform(0.2, 0.4) * min(width, height)
    x_center = random.uniform(0, width)
    y_center = random.uniform(0, height)
    start_angle = random.uniform(0, 360)
    end_angle = start_angle + random.uniform(60, 270)

    # 计算包围盒
    left = x_center - size / 2
    top = y_center - size / 2
    right = x_center + size / 2
    bottom = y_center + size / 2

    draw.arc([left, top, right, bottom],
             int(start_angle), int(end_angle),
             fill=1.0, width=line_width)  # 修改弧线宽度
    return np.array(img)


def create_pattern(output_path, img_size=(500, 500), line_width=75, ClientID=None):
    """
    创建彩色模式图像
    :param output_path: 输出路径
    :param img_size: 图像尺寸 (width, height)
    :param line_width: 线条粗细（像素）
    :param ClientID: 客户ID，用于创建子文件夹
    :return: 生成的PIL.Image对象
    """
    width, height = img_size

    # 生成二值模板
    canvas = np.zeros((height, width), dtype=np.float32)
    patterns = [lambda w, h: generate_line_pattern(w, h, line_width),
                lambda w, h: generate_arc_pattern(w, h, line_width)]

    for _ in range(6):  # 减少叠加次数避免过于密集
        pattern = random.choice(patterns)(width, height)
        canvas += pattern

    binary_mask = (canvas > 0.5)  # 得到布尔掩码

    # 生成随机颜色 (RGB)
    pattern_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    # 创建白色背景的RGB图像
    img_array = np.full((height, width, 3), 255, dtype=np.uint8)

    # 应用颜色到模板区域
    img_array[binary_mask] = pattern_color

    # 创建PIL图像对象
    img = Image.fromarray(img_array, 'RGB')

    # 处理ClientID文件夹
    if ClientID is not None:
        client_dir = os.path.join(output_path, str(ClientID))
        os.makedirs(client_dir, exist_ok=True)
        save_path = os.path.join(client_dir, os.path.basename(output_path))
    else:
        save_path = output_path

    # 保存图像
    img.save(save_path)
    return img


# 创建输出目录
os.makedirs("pattern", exist_ok=True)

# 生成90张图像（10.png到99.png）
for num in range(10, 100):
    filename = os.path.join("pattern", f"{num}.png")
    create_pattern(filename, img_size=(500, 500), line_width=75)  # 设置线条宽度为5像素
    print(f"已生成：{filename}")

print("所有图案生成完成！")