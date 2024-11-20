import pickle
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
# 定义解压后的路径和数据批次文件列表
data_path = "/home/zpc/cifar-10-batches-py/cifar-10-batches-py"  # 解压后文件的目录
batch_files = [f"data_batch_{i}" for i in range(1, 6)]
test_batch = "test_batch"
def load_cifar_batch(filename):
    """加载单个 CIFAR 批次文件."""
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        # CIFAR-10 数据是以行展开的方式存储，将其变为 (N, 3, 32, 32) 形状
        data = data.reshape(-1, 3, 32, 32).astype("float32")
        return data, labels

# 加载训练数据
train_data = []
train_labels = []
for batch_file in batch_files:
    data, labels = load_cifar_batch(os.path.join(data_path, batch_file))
    train_data.append(data)
    train_labels.extend(labels)
train_data = np.concatenate(train_data)
train_labels = np.array(train_labels)

# 加载测试数据
test_data, test_labels = load_cifar_batch(os.path.join(data_path, test_batch))
test_data = np.array(test_data)
test_labels = np.array(test_labels)
# 定义标准化转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 将数据转换为张量，并应用 transform
train_data = torch.tensor(train_data).float() / 255.0
test_data = torch.tensor(test_data).float() / 255.0

# 调整数据维度为 (N, 32, 32, 3)，并使用 permute 转换到 (N, 3, 32, 32)
train_data = train_data.permute(0, 1, 2, 3)
test_data = test_data.permute(0, 1, 2, 3)

train_labels = torch.tensor(train_labels).long()
test_labels = torch.tensor(test_labels).long()

# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

subset_size = 256
indices = torch.randperm(len(test_dataset)).tolist()[:subset_size]
subset = Subset(test_dataset, indices)
min_test_loader = DataLoader(subset, batch_size=32, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
