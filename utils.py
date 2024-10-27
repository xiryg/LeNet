from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_data(batch_size, val_split=0.2):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 数据集的标准化
        ]
    )

    train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # 划分训练集和验证集
    val_size = int(val_split * len(train_set))
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
