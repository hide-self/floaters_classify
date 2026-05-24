import os

# 根目录路径（根据实际情况修改）
root = "Trash_floaters_exist_classify"

# 要处理的子文件夹列表
sub_dirs = ["have_floaters", "no_floaters"]

# 支持的图片扩展名
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

for folder in sub_dirs:
    folder_path = os.path.join(root, folder)

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"文件夹 {folder_path} 不存在，跳过")
        continue

    # 获取文件夹中所有图片文件
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))
             and os.path.splitext(f)[1].lower() in img_exts]

    # 按文件名排序（可根据需要改成其他排序方式）
    files.sort()

    # 批量重命名
    for idx, filename in enumerate(files, start=1):
        old_path = os.path.join(folder_path, filename)
        # 提取扩展名
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # 构造新文件名：如 have_floaters_0001.jpg
        # 如果想强制改为 .jpg，可以把 ext 替换为 ".jpg"
        new_name = f"{folder}_{idx:04d}{ext}"
        new_path = os.path.join(folder_path, new_name)

        # 避免重名导致覆盖（如果新名已存在，可根据需要处理）
        if os.path.exists(new_path):
            print(f"警告：{new_path} 已存在，跳过重命名 {old_path}")
            continue

        os.rename(old_path, new_path)
        print(f"重命名：{old_path} -> {new_path}")

    print(f"{folder} 完成，共处理 {len(files)} 张图片")
