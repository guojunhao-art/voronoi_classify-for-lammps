import torch, pandas as pd, numpy as np
from torch import jit

# 1. 读特征
F = pd.read_csv("data.csv", header=None).iloc[:, 0:40].values.astype(np.float32)
x = torch.from_numpy(F.reshape(-1, 1, 10, 4))

# 2. state_dict 模型（read_pytorch 中那一套）
from read_pytorch import CNNClassifier
m_sd = CNNClassifier()
m_sd.load_state_dict(torch.load("./mix.ver1.0.pytorch.pt", map_location="cpu"))
m_sd.eval()
y_sd = m_sd(x).argmax(1)

# 3. TorchScript 模型（给 LAMMPS 用的那一个）
m_ts = jit.load("./mix.ver1.0.torchscript.pt")
m_ts.eval()
y_ts = m_ts(x).argmax(1)

print("equal:", torch.equal(y_sd, y_ts))
