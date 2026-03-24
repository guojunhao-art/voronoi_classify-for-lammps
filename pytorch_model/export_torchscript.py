"""
从 main_pytorch 保存的 state_dict（mix.ver1.0.pytorch.pt）导出 TorchScript，
供 LAMMPS compute voronoi/classify/atom 使用（LibTorch 只加载 TorchScript 格式）。
用法：python export_torchscript.py [state_dict.pt] [output.torchscript.pt]
默认：saved_model/mix.ver1.0.pytorch.pt -> saved_model/mix.ver1.0.torchscript.pt
"""
import sys
import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """与 main_pytorch.CNNClassifier 一致"""
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
            nn.MaxPool2d(2, stride=2, padding=0),
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


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "./mix.ver1.0.pytorch.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "./mix.ver1.0.torchscript.pt"

    model = CNNClassifier()
    state = torch.load(src, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    example = torch.randn(1, 1, 10, 4)
    traced = torch.jit.trace(model, example)
    traced.save(dst)
    print(f"TorchScript saved: {dst} (use this path in LAMMPS compute voronoi/classify/atom)")


if __name__ == "__main__":
    main()
