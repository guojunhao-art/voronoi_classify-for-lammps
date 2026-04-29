"""
PyTorch 版本训练脚本，与原 TensorFlow main.py 等效。
数据集: dataset.csv (40 特征 + 1 标签，二分类)
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


# --------------- 1. 数据加载与预处理（流式，支持超大 CSV） ---------------
class CSVBatchedDataset(IterableDataset):
    """
    以 chunk 方式流式读取 CSV，避免将整个数据集一次性读入内存。
    split: "train" 或 "val"，通过行号哈希划分，确保每条数据都参与训练/验证之一。
    """
    def __init__(self, csv_path, split="train", val_ratio=0.2, chunksize=500_000, sample_frac=1.0,
                 split_seed=42, shuffle_within_chunk=True):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if not (0.0 < sample_frac <= 1.0):
            raise ValueError("sample_frac must be in (0, 1]")
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be in (0, 1)")
        self.csv_path = csv_path
        self.split = split
        self.val_ratio = val_ratio
        self.chunksize = chunksize
        self.sample_frac = sample_frac
        self.split_seed = int(split_seed)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)

    def __iter__(self):
        row_start = 0
        rng = np.random.default_rng(42)
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunksize):
            chunk = chunk.reset_index(drop=True)
            original_n = len(chunk)
            n = original_n
            if n == 0:
                continue

            # 可选抽样（默认 sample_frac=1.0 即全量）
            if self.sample_frac < 1.0:
                keep = rng.random(n) < self.sample_frac
                chunk = chunk.loc[keep].reset_index(drop=True)
                n = len(chunk)
                if n == 0:
                    row_start += original_n
                    continue

            global_ids = row_start + np.arange(n, dtype=np.int64)
            split_rng = np.random.default_rng(self.split_seed + row_start)
            is_val = split_rng.random(n) < self.val_ratio
            if self.split == "train":
                selected = chunk.loc[~is_val]
            else:
                selected = chunk.loc[is_val]

            if len(selected) > 0:
                if self.shuffle_within_chunk:
                    selected = selected.sample(frac=1.0, random_state=self.split_seed + row_start).reset_index(drop=True)
                features = selected.iloc[:, 0:40].to_numpy(dtype=np.float32).reshape(-1, 1, 10, 4)
                labels = selected.iloc[:, 40].to_numpy(dtype=np.int64)
                for i in range(len(selected)):
                    yield torch.from_numpy(features[i]), torch.tensor(labels[i], dtype=torch.long)

            row_start += original_n


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

    # 关键参数：sample_frac=1.0 表示全量数据参与训练/验证划分
    csv_path = "dataset.csv"
    val_ratio = 0.2
    sample_frac = 1.0
    chunksize = 500_000

    train_ds = CSVBatchedDataset(
        csv_path=csv_path, split="train", val_ratio=val_ratio, chunksize=chunksize, sample_frac=sample_frac,
        split_seed=42, shuffle_within_chunk=True
    )
    val_ds = CSVBatchedDataset(
        csv_path=csv_path, split="val", val_ratio=val_ratio, chunksize=chunksize, sample_frac=sample_frac,
        split_seed=42, shuffle_within_chunk=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
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
