"""
detect_floaters.py
用于对单张或文件夹内多张水面图片进行漂浮物检测。
直接运行脚本即可，需预先训练并保存好 'floaters_svm_model.joblib'。
"""

import os
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# 特征提取函数（与训练时完全一致）
def glcm_features(gray_img, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2]):
    glcm = graycomatrix(gray_img, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props:
        feat = graycoprops(glcm, prop)
        features.extend(feat.flatten())
    return np.array(features)

def lbp_hist(gray_img, P=8, R=1):
    lbp = local_binary_pattern(gray_img, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def hu_moments(img):
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    # 取对数以避免极值
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu

def color_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_rgb = img_bgr.mean(axis=(0, 1))
    std_rgb = img_bgr.std(axis=(0, 1))
    mean_hsv = hsv.mean(axis=(0, 1))
    std_hsv = hsv.std(axis=(0, 1))
    return np.concatenate([mean_rgb, std_rgb, mean_hsv, std_hsv])

def extract_features(img_path, size=(256, 256)):
    """读取单张图片并提取特征，与训练时使用完全相同的流程"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 纹理
    f_glcm = glcm_features(gray)
    f_lbp = lbp_hist(gray)
    # 形状（对Otsu二值图计算Hu矩）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    f_hu = hu_moments(binary)
    # 颜色
    f_color = color_stats(img)

    return np.concatenate([f_glcm, f_lbp, f_hu, f_color])

# 加载模型
def load_model(model_path='floaters_svm_model.joblib'):
    """加载保存的模型、标准化器和图像尺寸"""
    assets = joblib.load(model_path)
    return assets['svm'], assets['scaler'], assets['image_size']

# 检测函数
def predict_image(model, scaler, size, img_path):
    """对单张图片进行预测，返回标签和置信度"""
    features = extract_features(img_path, size)
    if features is None:
        return None, None
    # 标准化
    features = scaler.transform([features])
    # 预测
    prob = model.predict_proba(features)[0]  # [no_floaters, have_floaters]
    label = model.predict(features)[0]       # 0: no_floaters, 1: have_floaters
    confidence = prob[1]                     # have_floaters 的概率
    return label, confidence

def detect_floaters(input_path, model_path='floaters_svm_model.joblib'):
    """
    输入：单张图片路径 或 包含多张图片的文件夹路径
    输出：打印每张图的预测结果
    """
    # 加载模型
    model, scaler, size = load_model(model_path)
    class_names = {0: 'no_floaters', 1: 'have_floaters'}

    # 收集所有待检测图片路径
    if os.path.isfile(input_path):
        img_paths = [input_path]
    elif os.path.isdir(input_path):
        img_paths = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))
        ]
    else:
        raise ValueError("输入路径无效（不是文件或文件夹）")

    if not img_paths:
        print("未找到任何图片文件。")
        return

    print(f"共检测到 {len(img_paths)} 张图片\n")
    for path in img_paths:
        label, confidence = predict_image(model, scaler, size, path)
        if label is None:
            print(f"{os.path.basename(path)} -> 读取失败")
            continue
        print(f"{os.path.basename(path)} -> {class_names[label]} (have_floaters置信度: {confidence:.4f})")



# 主程序入口
if __name__ == '__main__':
    # 修改下面路径为你的图片或文件夹
    INPUT_PATH = "test_images/s1.jpg"   # 或文件夹，如 "test_folder/"
    # 开始检测
    detect_floaters(INPUT_PATH)