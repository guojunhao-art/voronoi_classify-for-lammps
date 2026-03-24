"""
PyTorch 版 read.py：加载训练好的模型，对 CSV（前 40 列特征）做二分类预测并保存结果。
与 read.py 行为一致：读入 100.csv -> 预测 -> 输出 150result
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 可选：限制线程，与 read.py 中 OMP/CUDA 设置类似
#os.environ["OMP_NUM_THREADS"] = "20"


class CNNClassifier(nn.Module):
    """与 main_pytorch.CNNClassifier 结构一致，用于加载 state_dict。"""
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


# 与 read.py 对应：模型路径、输入 CSV、输出文件
MODEL_PATH = "./mix.ver1.0.pytorch.pt"
INPUT_CSV = "data.csv"
OUTPUT_FILE = "150result"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    predict_data = pd.read_csv(INPUT_CSV)
    p_features = predict_data.iloc[:, 0:40]  # 与 read.py 一致：前 40 列
    p_F = p_features.values.astype(np.float32)
    # PyTorch NCHW: (N, 1, 10, 4)
    p_F1 = torch.from_numpy(p_F.reshape(-1, 1, 10, 4)).to(device)

    with torch.no_grad():
        prediction = model(p_F1)
    y_test_pred = prediction.argmax(dim=1).cpu().numpy()
    y_test_output = y_test_pred.reshape(-1, 1)
    pd.DataFrame(y_test_output).to_csv(OUTPUT_FILE, index=False, header=False)
    print(f"Predict: {INPUT_CSV} -> {OUTPUT_FILE} (n={len(y_test_output)})")


if __name__ == "__main__":
    main()
