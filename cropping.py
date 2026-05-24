import cv2
import numpy as np
import os
import time

def get_center_water_mask(binary, center_ratio=0.4):
    """从 OTSU 二值图中提取位于图像中央的连通域作为水面掩膜"""
    h, w = binary.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    center_x1 = int(w * (1 - center_ratio) / 2)
    center_y1 = int(h * (1 - center_ratio) / 2)
    center_x2 = w - center_x1
    center_y2 = h - center_y1

    mask = np.zeros_like(binary)
    for label in range(1, num_labels):
        cx, cy = centroids[label]
        if center_x1 < cx < center_x2 and center_y1 < cy < center_y2:
            mask[labels == label] = 255
    return mask

def process_one_image(input_path, output_path, center_ratio=0.4, kernel_size=5, max_pixel_limit=3000000):
    """
    处理单张图像
    max_pixel_limit: 如果图像总像素超过此值，先缩小到该限制内再处理
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法读取图像: {input_path}")
        return False

    h, w = img.shape[:2]
    total_pixels = h * w
    scale = 1.0
    if total_pixels > max_pixel_limit:
        scale = (max_pixel_limit / total_pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  大图缩放: {w}x{h} -> {new_w}x{new_h} (缩放因子 {scale:.2f})")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"OTSU threshold = {ret} for {os.path.basename(input_path)}")

    water_mask = get_center_water_mask(binary_otsu, center_ratio)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

    if scale != 1.0:
        water_mask = cv2.resize(water_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY)

    water_gray = cv2.bitwise_and(gray, gray, mask=water_mask)
    cv2.imwrite(output_path, water_gray)
    print(f"已保存: {output_path}")
    return True

def batch_process(input_dir, output_dir, center_ratio=0.4, kernel_size=5, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif'), timer_enabled=False):
    """
    批量处理
    timer_enabled: 是否开启计时器，统计处理数量和平均耗时
    """
    os.makedirs(output_dir, exist_ok=True)

    if timer_enabled:
        total_start = time.time()
        processed_count = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + "_cropping" + os.path.splitext(filename)[1])
            success = process_one_image(input_path, output_path, center_ratio, kernel_size)
            if timer_enabled and success:
                processed_count += 1

    if timer_enabled:
        total_elapsed = time.time() - total_start
        avg_time = total_elapsed / processed_count if processed_count > 0 else 0
        print("\n========== 计时统计 ==========")
        print(f"成功处理图像数: {processed_count}")
        print(f"总耗时: {total_elapsed:.2f} 秒")
        print(f"平均每张耗时: {avg_time:.3f} 秒")
        print("==============================\n")

if __name__ == "__main__":
    class_type='no_floaters'
    INPUT_DIR = f"./Trash_floaters_exist_classify/{class_type}"
    OUTPUT_DIR = f"./after_cropping/{class_type}"
    CENTER_RATIO = 0.4
    KERNEL_SIZE = 5

    # 开启计时器：将 timer_enabled=True
    batch_process(INPUT_DIR, OUTPUT_DIR, CENTER_RATIO, KERNEL_SIZE, timer_enabled=True)
    print("批量处理完成！")