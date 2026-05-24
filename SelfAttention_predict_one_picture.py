import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
import os

# 计算数据集均值和标准差（用于归一化）
def compute_mean_std(data_root, cache_file=None):
    """
    计算数据集均值和标准差（每通道），若缓存文件存在则直接读取。
    参数:
        data_root: 数据集根目录
        cache_file: 缓存文件完整路径，若为None则默认保存为 data_root/mean_std.txt
    返回:
        mean (list), std (list)
    """
    if cache_file is None:
        cache_file = os.path.join(data_root, 'mean_std.txt')

    # 若缓存存在，直接读取
    if os.path.exists(cache_file):
        print(f"从缓存文件 {cache_file} 读取均值和标准差")
        with open(cache_file, 'r') as f:
            lines = f.readlines()
            mean = list(map(float, lines[0].strip().split()))
            std = list(map(float, lines[1].strip().split()))
        return mean, std

    # 否则计算并保存
    print("计算数据集均值和标准差...")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(dim=[0, 2]) * batch_samples
        std += images.std(dim=[0, 2]) * batch_samples
        total_pixels += batch_samples

    mean /= total_pixels
    std /= total_pixels
    mean_list = mean.tolist()
    std_list = std.tolist()

    # 保存为txt：第一行均值，第二行标准差（空格分隔）
    with open(cache_file, 'w') as f:
        f.write(' '.join(f'{v:.6f}' for v in mean_list) + '\n')
        f.write(' '.join(f'{v:.6f}' for v in std_list) + '\n')
    print(f"均值和标准差已保存至 {cache_file}")
    return mean_list, std_list




# 原 CNN 模型（通道3，尺寸512）
class myCNN_model(nn.Module):
    def __init__(self, num_conv_blocks=3, kernel_size=5, input_channels=3, input_size=512, num_classes=2):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 须为奇数"
        padding = kernel_size // 2
        out_channels_list = [6, 16, 32]
        self.blocks = nn.ModuleList()
        in_ch = input_channels
        for i in range(num_conv_blocks):
            conv = nn.Conv2d(in_channels=in_ch, out_channels=out_channels_list[i],
                             kernel_size=kernel_size, padding=padding)
            relu = nn.ReLU()
            pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.blocks.extend([conv, relu, pool])
            in_ch = out_channels_list[i]

        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            x = dummy
            for layer in self.blocks:
                x = layer(x)
            flatten_dim = self.flatten(x).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



# 自注意力 CNN（含自适应池化，降低序列长度）
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_ratio, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class SelfAttentionCNN(nn.Module):
    def __init__(self, num_conv_blocks=3, kernel_size=5, input_channels=3,
                 input_size=512, num_classes=2, num_heads=4, attn_dropout=0.1, attn_pool_size=8):
        super().__init__()
        self.cnn_backbone = myCNN_model(num_conv_blocks=num_conv_blocks, kernel_size=kernel_size,
                                        input_channels=input_channels, input_size=input_size,
                                        num_classes=num_classes)
        self.blocks = self.cnn_backbone.blocks
        self.spatial_pool = nn.AdaptiveAvgPool2d((attn_pool_size, attn_pool_size))

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            x = dummy
            for layer in self.blocks:
                x = layer(x)
            out_channels = x.shape[1]
        self.embed_dim = out_channels
        self.seq_len = attn_pool_size * attn_pool_size

        valid_heads = num_heads
        while valid_heads > 0 and self.embed_dim % valid_heads != 0:
            valid_heads -= 1
        if valid_heads == 0:
            valid_heads = 1
        self.num_heads = valid_heads

        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, self.embed_dim) * 0.02)
        self.attn_block = SelfAttentionBlock(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=attn_dropout)

        flatten_dim = self.seq_len * self.embed_dim
        self.fc1 = nn.Linear(flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.spatial_pool(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = x + self.pos_embed[:, :H * W, :]
        x = self.attn_block(x)
        x = x.reshape(B, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



def load_model(model_path, model_type='cnn', num_classes=2):
    """加载训练好的模型"""
    if model_type == 'cnn':
        model = myCNN_model(num_conv_blocks=3, kernel_size=5, input_channels=3,
                            input_size=512, num_classes=num_classes)
    elif model_type == 'sa_cnn':
        model = SelfAttentionCNN(num_conv_blocks=3, kernel_size=5, input_channels=3,
                                 input_size=512, num_classes=num_classes)
    else:
        raise ValueError("model_type 应为 'cnn' 或 'sa_cnn'")

    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def predict_image(image_path, model, class_names, mean, std):
    """
    预测单张图片，返回类别名称和置信度
    """
    # 预处理：与测试集完全一致（Resize → ToTensor → Normalize）
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 512, 512)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    class_name = class_names[predicted_idx.item()]
    confidence_value = confidence.item()
    return class_name, confidence_value


if __name__ == '__main__':
    # ====== 请修改以下参数 ======
    MODEL_PATH = './best_floaters_sa_cnn.pth'  # 训练保存的模型
    IMAGE_PATH = './test_images/s1.jpg'  # 待预测图片
    MODEL_TYPE = 'sa_cnn'  # 与训练时一致
    DATA_ROOT = './Trash_floaters_exist_classify'  # 数据集根目录（仅用于计算归一化参数）
    # ===========================

    # 类别名称（按 ImageFolder 字母序，通常 ['have_floaters', 'no_floaters']）
    class_names = ['have_floaters', 'no_floaters']

    # 计算与训练时相同的归一化参数（均值、标准差）
    print("正在计算数据集均值和标准差...")
    mean, std = compute_mean_std(DATA_ROOT)
    print(f"均值: {mean}, 标准差: {std}")

    # 加载模型并预测
    model = load_model(MODEL_PATH, MODEL_TYPE, num_classes=len(class_names))
    class_name, confidence = predict_image(IMAGE_PATH, model, class_names, mean, std)

    print(f'预测类别: {class_name}')
    print(f'置信度: {confidence:.4f} ({confidence * 100:.2f}%)')