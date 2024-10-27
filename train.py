import torch
from torch import nn, optim
from model import LeNet
from utils import load_data
from config import Config

config = Config()
model = LeNet().to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.NAdam(model.parameters(), lr=config.lr)

# 加载数据集并划分训练集和验证集
train_loader, val_loader, _ = load_data(config.batch_size, val_split=0.2)  # 接收三个值，忽略测试集

best_val_acc = 0.0  # 保存最佳验证集准确率
best_model_path = 'best_lenet.pth'  # 最佳模型的保存路径

for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(config.device), labels.to(config.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{config.epochs}], '
          f'Train Loss: {train_loss / len(train_loader):.4f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Accuracy: {val_acc:.2f}%')

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'Saving Best Model with Val Accuracy: {best_val_acc:.2f}%')
