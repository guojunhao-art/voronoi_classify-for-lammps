## 项目概览

本仓库包含一套 **Voronoi + PyTorch 二分类** 的完整实现，用于在 **LAMMPS** 中在线计算 40 维局部结构特征，并可选地在 LAMMPS 内部直接调用 **LibTorch** 做二分类推理，避免后处理和巨大中间文件。

目录结构：

- `lammps_voronoi_classify/`：集成到 LAMMPS `src/VORONOI/` 包的 C++ 源码与 CMake 片段  
  - `compute_voronoi_atom.h/.cpp`：在官方 `compute voronoi/atom` 基础上增加 `comm_ghost` 选项，用于把体积/面数同步到 ghost 原子。  
  - `compute_voronoi_classify_atom.h/.cpp`：新增 `compute voronoi/classify/atom`，构造 40 维特征并可选用 LibTorch 推理。  
  - `README.txt`：功能说明、命令语法与使用示例。  
  - `BUILD_CMAKE.md`：如何用 CMake 把本模块编进 LAMMPS（仅特征模式 / 推理模式）。  
  - `VORONOI_cmake_append_snippet.cmake`：追加到 `cmake/Modules/Packages/VORONOI.cmake` 末尾，启用 `PKG_VORONOI_TORCH` 并正确链接 LibTorch。  
  - `VORONOI_voro_build_fix.cmake`：修复在 Intel/nvcc_wrapper 下构建 Voro++ 的兼容性问题（单独用 g++ 构建 Voro++）。

- `pytorch_model/`：用于训练与验证的 PyTorch 模型与数据脚本  
  - `dataset.csv` / `data.csv`：示例数据（40 维特征 + 1 维标签）。  
  - `dataset_structure.txt`：40 维特征与标签含义说明。  
  - `main_pytorch.py`：基于 40 维特征训练 CNN 分类器，同时导出 state_dict 与 TorchScript 模型。  
  - `read_pytorch.py`：加载 state_dict，对 CSV 特征做推理，行为与原 TensorFlow `read.py` 对齐。  
  - `export_torchscript.py`：从现有 state_dict 导出 TorchScript（无需重新训练）。  
  - `compare.py`：在 Python 端对比 state_dict 模型与 TorchScript 模型输出是否完全一致。  
  - 其它脚本（`main.py`, `read.py`, `process.py` 等）：原 TensorFlow 流程及特征构造相关代码，仅作为参考/对照。

---

## 一、40 维 Voronoi 特征与 LAMMPS 侧实现

### 1.1 特征定义（10 组 × 4 维）

每个中心原子一行特征，共 40 维，可视为 \(10 \times 4\)：

- **组 1（中心原子）**：  
  \[ 距离(=0), Voronoi 体积, 边数(面数), 原子类型 \]
- **组 2–10（前 9 个最近邻）**：  
  \[ 与中心原子的距离, Voronoi 体积, 边数, 原子类型 \]，按距离从近到远排序。

原子类型来源：

- 默认：使用 data 文件中的 `atom type`。  
- 若在命令中写 `type g1 g2 ...`，则：  
  - 在组 `g1` 的原子类型记为 1，`g2` 记为 2，…  
  - 不在任一指定 group 的原子类型记为 0。

详细列索引和结构说明见 `pytorch_model/dataset_structure.txt`。

### 1.2 LAMMPS 内部 compute 设计

- `compute voronoi/atom`：  
  - 增加 `comm_ghost` 关键字，将每个原子的 `[体积, 面数]` 传递到 ghost，以便 classify 在邻居为 ghost 时也能访问 Voronoi 数据。  
  - 未使用 `comm_ghost` 时，classify 遇到 ghost 邻居将无法获取其 Voronoi 信息，只能跳过或填 0。

- `compute voronoi/classify/atom`：  
  - 语法：  
    ```text
    compute ID group voronoi/classify/atom c_voronoi_id [type g1 g2 ...] [model.pt]
    ```  
  - `c_voronoi_id`：已有的 `compute voronoi/atom` 的 ID（推荐带 `comm_ghost`）。  
  - `type g1 g2 ...`：可选，使用 group 定义特征中的“原子类型”。  
  - `model.pt`：可选 TorchScript 模型路径（需以 LibTorch 编译 LAMMPS）。  
  - 使用当前 pair style 的 full 邻居列表，按距离排序取前 9 个近邻构造 40 维特征。  
  - **特征模式**：未提供 `model.pt` 时，`array_atom` 为 40 列特征，可用 `dump ... c_ID[*]` 输出。  
  - **推理模式**：提供 `model.pt` 且启用 LibTorch 时，`array_atom` 为单列 0/1 分类结果。

内部细节（neighbor request、ghost 通信、feature 填充顺序等）可参考 `lammps_voronoi_classify/README.txt`。

---

## 二、如何集成到 LAMMPS 并编译

### 2.1 复制源码到 LAMMPS 源码树

假设 LAMMPS 根目录为 `/path/to/lammps`：

1. 复制以下 4 个源码文件到 `src/VORONOI/`（覆盖原有同名文件）：  
   - `lammps_voronoi_classify/compute_voronoi_atom.h`  
   - `lammps_voronoi_classify/compute_voronoi_atom.cpp`  
   - `lammps_voronoi_classify/compute_voronoi_classify_atom.h`  
   - `lammps_voronoi_classify/compute_voronoi_classify_atom.cpp`
2. 打开 LAMMPS 的 `cmake/Modules/Packages/VORONOI.cmake`，在文件末尾追加  
   `lammps_voronoi_classify/VORONOI_cmake_append_snippet.cmake` 中的全部内容（用于 `PKG_VORONOI_TORCH` 和 LibTorch 链接）。

如需在 Intel 或 Kokkos+CUDA（`nvcc_wrapper`）环境下修复 Voro++ 的编译问题，可参考  
`lammps_voronoi_classify/VORONOI_voro_build_fix.cmake`，将其中逻辑插入 VORONOI.cmake 里 Voronoi 外部构建部分。

### 2.2 仅特征模式（不启用 LibTorch）

适用于只想得到 40 维特征、在 Python / 其它工具中自行推理的场景。

在 LAMMPS 根目录外创建 build 目录并配置：

```bash
cd /path/to/lammps
mkdir build && cd build

cmake -S cmake -B . \
  -DPKG_VORONOI=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)
```

生成的 `lmp` 可执行文件中已包含：

- `compute voronoi/atom`（含 `comm_ghost`）；  
- `compute voronoi/classify/atom`（仅特征模式）。

### 2.3 推理模式（启用 LibTorch）

在 `VORONOI.cmake` 追加 `VORONOI_cmake_append_snippet.cmake` 后，可以通过 CMake 选项启用：

```bash
cmake -S cmake -B . \
  -DPKG_VORONOI=ON \
  -DPKG_VORONOI_TORCH=ON \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)
```

注意事项：

- `/path/to/libtorch` 应指向 LibTorch 的根目录（其下有 `include/` 与 `share/cmake/Torch/` 等）。  
- CMake 片段会自动：  
  - `find_package(Torch CONFIG)`；  
  - 对 `lammps` 目标定义 `LAMMPS_TORCH`；  
  - 将 `Torch::Torch` 或 `Torch_LIBRARIES` 链接到 `lammps` 和最终的 `lmp`。

更多 CMake 细节与常见错误（如 `Torch::Torch not found`、Voro++ 编译问题）见  
`lammps_voronoi_classify/BUILD_CMAKE.md`。

---

## 三、PyTorch 侧：训练与导出模型

### 3.1 数据与特征

- **数据文件**：`pytorch_model/dataset.csv`（或同结构 CSV），每行：40 维特征 + 1 维标签。  
- 结构说明：`pytorch_model/dataset_structure.txt`。

### 3.2 训练模型

在 `pytorch_model/` 目录下：

```bash
cd pytorch_model
python main_pytorch.py
```

脚本会：

- 从 `dataset.csv` 读取数据（可在脚本中修改路径/采样比例）；  
- 训练一个 CNN 分类器（输入 reshape 为 `(N, 1, 10, 4)`）；  
- 在 `pytorch_model/saved_model/` 下保存：
  - `mix.ver1.0.pytorch.pt`：state_dict，给 `read_pytorch.py` 使用；  
  - `mix.ver1.0.torchscript.pt`：TorchScript 模型，给 LAMMPS 使用。

若已训练好模型、只想从 state_dict 导出 TorchScript，可运行：

```bash
python export_torchscript.py \
  saved_model/mix.ver1.0.pytorch.pt \
  saved_model/mix.ver1.0.torchscript.pt
```

### 3.3 Python 侧推理（对比/验证用）

在 `pytorch_model/` 中：

- 使用 state_dict 推理 CSV：

  ```bash
  python read_pytorch.py
  ```

  默认读取 `100.csv`，输出 `150result`，可在脚本内修改输入/输出文件名。

- 对比 state_dict 与 TorchScript 输出是否一致：

  ```bash
  python compare.py
  ```

  若打印 `equal: True`，说明两种保存方式下的模型行为一致。

---

## 四、LAMMPS 中的使用示例

下面示例假设：

- 已在 LAMMPS 中启用 `VORONOI` 包，且（可选）启用 `PKG_VORONOI_TORCH`；  
+- 已定义 pair_style / neighbor 等基本设置。

### 4.1 仅特征模式（40 维特征输出）

```lammps
compute voro all voronoi/atom comm_ghost
compute cc   all voronoi/classify/atom voro

dump d1 all custom 100 dump.feature id c_cc[*]
```

此时 `dump.feature` 中，每行为：`id c_cc[1] ... c_cc[40]`，可直接作为 `dataset.csv` 的特征部分。

### 4.2 推理模式（在 LAMMPS 内直接得到 0/1）

```lammps
compute voro all voronoi/atom comm_ghost
# 基于 data 中的 atom type
compute pred all voronoi/classify/atom voro \
  /absolute/path/to/pytorch_model/saved_model/mix.ver1.0.torchscript.pt

dump d2 all custom 100 dump.class id c_pred
```

或使用 group 映射类型：

```lammps
group C type 1 2 6
group H type 3 5
group O type 4

compute voro all voronoi/atom comm_ghost
compute pred all voronoi/classify/atom voro type C H O \
  /absolute/path/to/pytorch_model/saved_model/mix.ver1.0.torchscript.pt

dump d2 all custom 100 dump.class id c_pred
```

此时：

- `c_pred` 为单列 per-atom 数组（0 或 1）；  
- compute group 之外的原子，其输出会被置为 0。

---

## 五、调试与一致性验证

本项目提供了从 **TensorFlow → PyTorch → LibTorch (C++) → LAMMPS 内推理** 的一致性验证手段：

1. 使用特征模式在 LAMMPS 中生成 `dump.feature`；  
2. 在 `pytorch_model/` 中用 `read_pytorch.py` 或 TorchScript 对同一特征做推理；  
3. 在 LAMMPS 中启用推理模式，将 `compute voronoi/classify/atom` 的输出与 Python 端结果按 `id` 对齐比较。  

我们已在开发中验证：

- 当使用相同的 TorchScript 模型和 40 维特征时，Python-TorchScript 与 LAMMPS 内推理输出可做到完全一致（差异来自内存布局问题已在当前版本中修复）。  

如需进一步 debug，可在 LAMMPS 里同时定义：

```lammps
compute feat all voronoi/classify/atom voro
compute pred all voronoi/classify/atom voro /path/to/model.torchscript.pt
```

分别 dump `c_feat[*]` 与 `c_pred`，再在 Python 中进行对比。  

---

## 六、许可证与致谢

- LAMMPS 本身遵循 GPLv2 许可证，详见 LAMMPS 官方仓库。  
- 本项目中对 LAMMPS 的修改和新增文件，遵循与 LAMMPS 一致的许可证约定。  
- PyTorch / LibTorch 遵循其各自的开源许可证，使用时请参考官方文档。

