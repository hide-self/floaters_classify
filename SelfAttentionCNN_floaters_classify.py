import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import time



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



# 数据加载模块（适配漂浮物数据集，含归一化）
def get_train_test_dataloader(batch_size=32, train_ratio=0.8):
    """
    返回训练集与测试集的DataLoader
    数据集结构：./Trash_floaters_exist_classify/have_floaters 和 no_floaters
    """
    data_root = './Trash_floaters_exist_classify'

    # 计算数据集的均值和标准差
    mean, std = compute_mean_std(data_root)
    print(f"数据集均值: {mean}")
    print(f"数据集标准差: {std}")

    # 训练数据增强（含归一化）
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # 加入归一化
    ])

    # 测试集：仅Resize + ToTensor + 同一归一化
    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 加载整个数据集（先用 train_transform 初始化，后面测试集会单独修改）
    full_dataset = datasets.ImageFolder(root=data_root, transform=train_transform)

    # 划分训练/测试集
    total = len(full_dataset)
    train_size = int(train_ratio * total)
    test_size = total - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 重要：将测试集的 transform 改为 test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def get_num_classes():
    data_root = './Trash_floaters_exist_classify'
    temp_dataset = datasets.ImageFolder(root=data_root)
    return len(temp_dataset.classes)


def get_class_names():
    data_root = './Trash_floaters_exist_classify'
    temp_dataset = datasets.ImageFolder(root=data_root)
    return temp_dataset.classes










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











# 训练与验证辅助函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / len(train_loader), 100. * correct / total


def test_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return test_loss / len(test_loader), 100. * correct / total


def whole_train_process(num_conv_blocks=3, kernel_size=5, model_type='cnn'):
    batch_size = 32
    learning_rate = 0.01
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = get_num_classes()
    input_channels = 3
    input_size = 512

    train_loader, test_loader = get_train_test_dataloader(batch_size=batch_size)

    if model_type == 'cnn':
        model = myCNN_model(num_conv_blocks=num_conv_blocks, kernel_size=kernel_size,
                            input_channels=input_channels, input_size=input_size, num_classes=num_classes)
    elif model_type == 'sa_cnn':
        model = SelfAttentionCNN(num_conv_blocks=num_conv_blocks, kernel_size=kernel_size,
                                 input_channels=input_channels, input_size=input_size, num_classes=num_classes)
    else:
        raise TypeError('仅支持 cnn 或 sa_cnn')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_test_acc = 0.0
    sum_time_use = 0

    print('开始训练...')
    for epoch in range(epochs):
        since = time.time()
        print(f'--- Epoch {epoch + 1}/{epochs} ---')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model_name = f'./best_floaters_{model_type}.pth'
            torch.save(model.state_dict(), model_name)

        time_use = time.time() - since
        print(f'耗时: {time_use / 60:.1f} min')
        sum_time_use += time_use

    print(f'总耗时: {sum_time_use / 60:.1f} min，最佳测试准确率: {best_test_acc:.2f}%')
    train_process = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
    })
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process['train_losses'], 'ro-', label='train loss')
    plt.plot(train_process['epoch'], train_process['test_losses'], 'bs-', label='test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_accuracies'], 'ro-', label='train acc')
    plt.plot(train_process['epoch'], train_process['test_accuracies'], 'bs-', label='test acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()



# 主程序：开始训练
if __name__ == '__main__':
    # 可选 'cnn' 或 'sa_cnn'
    train_df = whole_train_process(num_conv_blocks=3, kernel_size=5, model_type='sa_cnn')
    matplot_acc_loss(train_df)