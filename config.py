import torch


class Config:
    batch_size = 256
    epochs = 10
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_split = 0.2
