"""
PyTorch 版本训练脚本，与原 TensorFlow main.py 等效。
数据集: dataset.csv (40 特征 + 1 标签，二分类)
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# --------------- 1. 数据加载与预处理 ---------------
def load_data(csv_path="dataset.csv", sample_frac=0.2, val_ratio=0.2):
    data = pd.read_csv(csv_path).sample(frac=sample_frac, random_state=42)
    features = data.iloc[:, 0:40].values.astype(np.float32)
    labels = data.iloc[:, 40].values.astype(np.int64)
    # reshape: (N, 40) -> (N, 1, 10, 4) [NCHW]
    features = features.reshape(-1, 1, 10, 4)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    dataset = TensorDataset(x, y)
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    return train_ds, val_ds


# --------------- 2. 模型定义（与 TF Sequential 对应） ---------------
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),  # 5x2 -> 2x1
            nn.Dropout(0.25),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# --------------- 3. 训练与评估 ---------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 超参与 TF 一致
    batch_size = 2048
    epochs = 10
    lr = 5e-5

    train_ds, val_ds = load_data(val_ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-10)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    os.makedirs("saved_model", exist_ok=True)
    torch.save(model.state_dict(), "saved_model/mix.ver1.0.pytorch.pt")
    print("Model saved to saved_model/mix.ver1.0.pytorch.pt (state_dict, for read_pytorch.py)")

    # 导出 TorchScript 供 LAMMPS compute voronoi/classify/atom 加载（torch::jit::load 只认此格式）
    model.eval()
    example = torch.randn(1, 1, 10, 4)  # (batch, channel, NGROUPS=10, NPER=4)，与 LAMMPS 输入一致
    traced = torch.jit.trace(model, example)
    traced.save("saved_model/mix.ver1.0.torchscript.pt")
    print("TorchScript (LAMMPS-loadable): saved_model/mix.ver1.0.torchscript.pt")


if __name__ == "__main__":
    main()
