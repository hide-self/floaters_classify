import cv2
import numpy as np
import os
from pathlib import Path


def preprocess_image(img, enable_clahe=True, enable_sharpen=True,
                     enable_color_enhance=True, denoise_ksize=3):
    """
    对单张图像进行预处理
    Args:
        img: BGR格式图像 (numpy array)
        enable_clahe: 是否进行CLAHE增强
        enable_sharpen: 是否锐化
        enable_color_enhance: 是否进行HSV色彩增强
        denoise_ksize: 高斯滤波核大小（奇数）
    Returns:
        处理后的图像 (BGR格式)
    """
    # 1. 去噪（高斯滤波）
    if denoise_ksize > 0:
        # 传入参数解析：
        # img传入的图像
        # (denoise_ksize, denoise_ksize)表示卷积核大小
        # 0表示sigmaX,sigmaY没有传入默认等于sigmaX
        # sigmaX传入0表示标准差自适应
        img = cv2.GaussianBlur(img, (denoise_ksize, denoise_ksize), 0)

    # 2. 色彩增强（在HSV空间调整）
    if enable_color_enhance:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 提高饱和度（S）和亮度（V），让颜色更鲜明
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 30)  # 增加饱和度
        v = cv2.add(v, 20)  # 增加亮度
        hsv_enhanced = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # 3. 对比度增强（CLAHE，在亮度通道上操作,分块进行直方图均衡化）
    if enable_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 4. 锐化（非锐化掩模，USM）
    if enable_sharpen:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]]) / 1.0  # 简单锐化核
        img = cv2.filter2D(img, -1, kernel)

    return img


def batch_preprocess(input_dir, output_dir, **kwargs):
    """
    批量处理文件夹内所有图像
    Args:
        input_dir: 原始图像文件夹路径
        output_dir: 输出文件夹路径
        **kwargs: 传递给 preprocess_image 的参数
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图像扩展名
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

    for img_file in input_path.iterdir():
        if img_file.suffix.lower() not in img_exts:
            continue
        # 读取图像
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"无法读取: {img_file.name}")
            continue

        # 预处理
        processed = preprocess_image(img, **kwargs)

        # 保存到输出文件夹（保持原文件名）
        new_filename = f"{img_file.stem}_enhance{img_file.suffix}"
        out_path = output_path / new_filename
        cv2.imwrite(str(out_path), processed)
        print(f"处理完成: {input_path}\\{img_file.name} -> {out_path}")


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 设置路径（请修改为实际路径）
    class_type='no_floaters'
    input_folder = f"./Trash_floaters_exist_classify/{class_type}"  # 原始图像所在文件夹
    output_folder = f"./after_enhance/{class_type}"  # 处理后图像保存文件夹

    # 可调节参数
    batch_preprocess(input_folder, output_folder,
                     enable_clahe=True,
                     enable_sharpen=True,
                     enable_color_enhance=True,
                     denoise_ksize=3)  # 高斯滤波核大小，设为0则跳过

