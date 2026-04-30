"""
PyTorch 版本训练脚本，与原 TensorFlow main.py 等效。
数据集: dataset.csv (40 特征 + 1 标签，二分类)
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN model and/or export TorchScript.")
    parser.add_argument("--export-only", action="store_true",
                        help="仅导出 TorchScript，不进行训练")
    parser.add_argument("--state-dict-in", default="saved_model/mix.ver1.0.pytorch.pt",
                        help="导出模式下加载的 state_dict 路径")
    parser.add_argument("--state-dict-out", default="saved_model/mix.ver1.0.pytorch.pt",
                        help="训练模式下保存的 state_dict 路径")
    parser.add_argument("--torchscript-out", default="saved_model/mix.ver1.0.torchscript.pt",
                        help="TorchScript 输出路径")
    parser.add_argument("--csv-path", default="dataset.csv", help="训练 CSV 路径")
    parser.add_argument("--batch-size", type=int, default=2048, help="训练 batch size")
    parser.add_argument("--epochs", type=int, default=10, help="训练 epoch")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="训练抽样比例")
    parser.add_argument("--chunksize", type=int, default=500_000, help="CSV 分块大小")
    return parser.parse_args()


# --------------- 1. 数据加载与预处理（流式，支持超大 CSV） ---------------
class CSVBatchedDataset(IterableDataset):
    """
    以 chunk 方式流式读取 CSV，避免将整个数据集一次性读入内存。
    split: "train" 或 "val"，通过行号哈希划分，确保每条数据都参与训练/验证之一。
    """
    def __init__(self, csv_path, split="train", val_ratio=0.2, chunksize=500_000, sample_frac=1.0,
                 split_seed=42, shuffle_within_chunk=True, batch_size=2048):
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
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

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
                for start in range(0, len(selected), self.batch_size):
                    end = min(start + self.batch_size, len(selected))
                    x = torch.from_numpy(features[start:end])
                    y = torch.from_numpy(labels[start:end]).to(torch.long)
                    yield x, y

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
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CNNClassifier()
    if not args.export_only:
        train_ds = CSVBatchedDataset(
            csv_path=args.csv_path, split="train", val_ratio=args.val_ratio, chunksize=args.chunksize,
            sample_frac=args.sample_frac, split_seed=42, shuffle_within_chunk=True, batch_size=args.batch_size
        )
        val_ds = CSVBatchedDataset(
            csv_path=args.csv_path, split="val", val_ratio=args.val_ratio, chunksize=args.chunksize,
            sample_frac=args.sample_frac, split_seed=42, shuffle_within_chunk=False, batch_size=args.batch_size
        )

        # 数据集已经按 batch 产出，DataLoader 不再二次组 batch
        train_loader = DataLoader(train_ds, batch_size=None, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=None, shuffle=False, num_workers=0)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-10)

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        os.makedirs(os.path.dirname(args.state_dict_out) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.state_dict_out)
        print(f"Model saved to {args.state_dict_out} (state_dict, for read_pytorch.py)")
    else:
        state = torch.load(args.state_dict_in, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"Loaded state_dict from: {args.state_dict_in}")

    # 导出 TorchScript 供 LAMMPS compute voronoi/classify/atom 加载（torch::jit::load 只认此格式）
    # 为避免 cpu/cuda 设备不一致报错，导出时统一在 CPU 上进行
    model_cpu = CNNClassifier()
    model_cpu.load_state_dict(model.state_dict(), strict=True)
    model_cpu.eval()
    example = torch.randn(1, 1, 10, 4)  # CPU example
    traced = torch.jit.trace(model_cpu, example)
    os.makedirs(os.path.dirname(args.torchscript_out) or ".", exist_ok=True)
    traced.save(args.torchscript_out)
    print(f"TorchScript (LAMMPS-loadable): {args.torchscript_out}")


if __name__ == "__main__":
    main()
