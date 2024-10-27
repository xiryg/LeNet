import torch
from model import LeNet
from utils import load_data
from config import Config

config = Config()
model = LeNet().to(config.device)
model.load_state_dict(torch.load('best_lenet.pth', weights_only=True))
model.eval()

_, _, test_loader = load_data(config.batch_size)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(config.device), labels.to(config.device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
