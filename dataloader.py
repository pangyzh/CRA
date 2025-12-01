import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

class CNN_MNIST(nn.Module):
    def __init__(self, num_channels=1):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50) # MNIST 320, CIFAR 500 (approx for simple net)
        self.fc2 = nn.Linear(50, 10)
        self.is_cifar = (num_channels == 3)
        if self.is_cifar:
            self.fc1 = nn.Linear(20 * 5 * 5, 50) # 32x32 -> 28x28 -> 14x14 -> 10x10 -> 5x5

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        # 核心：第一个卷积层接收 3 个输入通道
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 展平后接全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def get_dataset(args):
    """读取数据集并进行划分"""
    data_dir = './data/'
    if args.name_dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        input_channels = 1
    elif args.name_dataset == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        input_channels = 3
    else:
        raise ValueError("Unsupported dataset")

    # 1. distribute Root Data (for server)
    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)
    
    root_indices = all_indices[:args.root_data_size]
    client_pool_indices = all_indices[args.root_data_size:]
    
    root_data = Subset(train_dataset, root_indices)
    
    # 2. distribute Client Data (IID 或 Non-IID)
    user_groups = {}
    if args.iid:
        # IID
        num_items = int(len(client_pool_indices) / args.num_clients)
        for i in range(args.num_clients):
            user_groups[i] = client_pool_indices[i*num_items : (i+1)*num_items]
    else:
        # Non-IID: Dirichlet
        min_size = 0
        while min_size < 10:
            idx_batch = [[] for _ in range(args.num_clients)]

            targets = np.array(train_dataset.targets)[client_pool_indices]
            for k in range(10):
                idx_k = np.where(targets == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.alpha, args.num_clients))
                proportions = np.array([p * (len(idx_j) < len(client_pool_indices) / args.num_clients) 
                                      for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                idx_split = np.split(idx_k, proportions)
                for i in range(args.num_clients):
                    idx_batch[i] = np.concatenate((idx_batch[i], client_pool_indices[idx_split[i]]), axis=0)
            
            min_size = min([len(idx_j) for idx_j in idx_batch])
            
        for i in range(args.num_clients):
            user_groups[i] = idx_batch[i].astype(int)

    return train_dataset, test_dataset, user_groups, root_data, input_channels