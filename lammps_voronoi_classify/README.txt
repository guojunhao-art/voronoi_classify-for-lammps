================================================================================
  LAMMPS 内嵌 Voronoi 二分类方案：compute voronoi/classify/atom
================================================================================
目标：在 LAMMPS 中直接使用 compute voronoi/atom + 邻居列表构建 40 维特征并
      可选地调用 PyTorch 模型做二分类，避免轨迹后处理、全局 top-K 与巨大中间文件。

--------------------------------------------------------------------------------
本文件夹为修改过的 LAMMPS 源码副本，便于备份或覆盖到 LAMMPS 源码树使用。
--------------------------------------------------------------------------------
- compute_voronoi_atom.h / .cpp  → 覆盖到 LAMMPS src/VORONOI/
- compute_voronoi_classify_atom.h / .cpp  → 放入 LAMMPS src/VORONOI/
- 使用 CMake 编译的步骤与可选 LibTorch 推理编译见：BUILD_CMAKE.md
- 推理模式 CMake 片段（可追加到 cmake/Modules/Packages/VORONOI.cmake）：VORONOI_cmake_append_snippet.cmake

--------------------------------------------------------------------------------
一、数据流与 40 维特征（与 dataset_structure.txt 一致）
--------------------------------------------------------------------------------
- 每个中心原子一行：10 组 × 4 维 = 40 维。
- 组 1：中心原子 [ 到自身距离(=0), Voronoi体积, 边数(面数), 原子类型 ]
- 组 2–10：距中心原子第 1～9 近的原子 [ 距离, Voronoi体积, 边数, 原子类型 ]，
  按距离从近到远。
- 来源：LAMMPS 内 compute voronoi/atom 得到体积与面数；邻居列表 + 距离排序得到 top-9。
- “原子类型”可选由 group 指定：命令中写 type g1 g2 ... 时，特征中的类型为 1,2,...
  （按所属 group 顺序；不属于任一指定 group 的原子类型为 0）。不写 type 时仍用
  data 文件中的 atom type。

--------------------------------------------------------------------------------
二、实现内容
--------------------------------------------------------------------------------
1) 修改 compute voronoi/atom（可选）
   - 增加关键字 "comm_ghost"：将每个原子的 [体积, 面数] 向 ghost 通信，便于
     classify 在邻居为 ghost 时也能取到 Voronoi 数据。
   - 若不用 comm_ghost，classify 仅基于本地原子邻居（邻居列表里 j 为 ghost 时
     无 Voronoi 数据则跳过或填 0）。

2) 新增 compute voronoi/classify/atom（VORONOI 包内）
   - 语法：compute ID group voronoi/classify/atom c_voronoi_id [type g1 g2 ...] [model.pt]
   - c_voronoi_id：已有的 compute voronoi/atom 的 ID（建议带 comm_ghost）。
   - type g1 g2 ...：可选；用 group 定义特征中的“原子类型”：在 g1 的原子类型=1，
     在 g2 的=2，…；不在任一中的=0。省略则用 data 里的 atom type。
   - model.pt：可选 TorchScript 模型路径（需 LAMMPS_TORCH 编译）。
   - 使用当前 pair style 的邻居列表，按距离排序取前 9 近邻构建 40 维特征。
   - 当前仅输出 40 维特征（array_atom，40 列），可用 dump 写出后由外部 Python
     脚本做推理；若将来提供 model_file 并启用 LibTorch，可在 LAMMPS 内输出 0/1。

3) 内部逻辑概要
   - 请求 full、occasional 邻居列表。
   - 每步：先调用 compute voronoi/atom；若启用了 comm_ghost，则对 voronoi 做
     forward_comm；然后对每个中心原子：
     a) 从邻居列表中取所有 j，算距离 rsq，按 rsq 排序；
     b) 取前 9 个（不足则用 0 填充）；
     c) 组 1：0, voro[i][0], voro[i][1], type[i]；
     d) 组 2–10：sqrt(rsq_j), voro[j][0], voro[j][1], type[j]；
     e) 若无 model_file：将 40 维写入 array_atom；若有且已编 LibTorch：reshape
        为 (1,1,10,4)，前向推理，argmax 写入 array_atom 单列。

--------------------------------------------------------------------------------
三、编译与依赖
--------------------------------------------------------------------------------
- 仅“特征模式”（不提供 model_file）：只需 VORONOI 包，无需 LibTorch。
- “推理模式”（提供 model_file）：已实现；需在编译时定义宏 LAMMPS_TORCH 并链接
  LibTorch，模型为 TorchScript（.pt）。

  编译推理模式示例（在 LAMMPS 的 build 目录或 Make 中）：
  · 定义宏：-DLAMMPS_TORCH
  · 包含路径：LibTorch 的 include（如 -I/path/to/libtorch/include）
  · 链接：libtorch、c10、torch_cpu（或 torch_cuda）等（见 LibTorch 文档）
  · CMake 可设：CMAKE_PREFIX_PATH=/path/to/libtorch，并在 VORONOI 或本模块的
    CMakeLists 里 find_package(Torch) 与 target_compile_definitions(.. LAMMPS_TORCH)

--------------------------------------------------------------------------------
四、使用示例（特征模式：只生成 40 维）
--------------------------------------------------------------------------------
  # 用 data 里的 atom type 作为特征中的类型：
  compute voro all voronoi/atom comm_ghost
  compute cc all voronoi/classify/atom voro

  # 用 group 指定类型（与 data 的 atom type 无关）：先定义 group，再写 type 组名列表
  group typeA type 1
  group typeB type 2
  compute cc all voronoi/classify/atom voro type typeA typeB

  # 需已定义 pair_style。dump 40 列特征：
  dump 1 all custom 100 dump.feature id c_cc[*]

--------------------------------------------------------------------------------
五、使用示例（推理模式：得到每原子 0/1 分类）
--------------------------------------------------------------------------------
（1）当前方式：LAMMPS 输出 40 维特征，再用 Python 脚本做推理

  · LAMMPS 输入脚本中（同上）：
      compute voro all voronoi/atom comm_ghost
      compute cc  all voronoi/classify/atom voro
      dump 1 all custom 100 dump.feature id c_cc[*]
      run 1000

  · 将 dump.feature 整理成“每行 40 列特征”的 CSV（无表头或表头不参与），
    列顺序与 c_cc[1]..c_cc[40] 一致。

  · 在 pytorch_model 目录下用训练好的模型做推理，例如：
      python read_pytorch.py
    默认会读 100.csv、输出 150result。若你的特征文件是别的名字，可修改
    read_pytorch.py 里 INPUT_CSV 与 OUTPUT_FILE，或使用 process_pytorch.py：
      python process_pytorch.py --pairs dump_40cols.csv result_class.txt

  · 得到 result_class.txt（每行一个 0 或 1）后，可按原子 id 写回 LAMMPS 的
    dump、或用于后处理统计/可视化。

（2）LAMMPS 内推理（已实现；需用 -DLAMMPS_TORCH 链接 LibTorch 编译）

  · 语法（第 5 个参数为 TorchScript 模型路径）：
      compute voro all voronoi/atom comm_ghost
      compute cc  all voronoi/classify/atom voro saved_model/mix.ver1.0.torchscript.pt

  · 此时 c_cc 为单列 per-atom 数组，取值为 0 或 1，可直接 dump：
      dump 2 all custom 100 dump.class id c_cc[1]

  · 模型需先用 PyTorch 导出为 TorchScript（.pt），输入形状 (N, 1, 10, 4)，例如：
      example = torch.randn(1, 1, 10, 4)
      traced = torch.jit.trace(model, example)
      traced.save("saved_model/mix.ver1.0.torchscript.pt")

--------------------------------------------------------------------------------
六、文件清单（本文件夹）
--------------------------------------------------------------------------------
  compute_voronoi_atom.h / .cpp
  compute_voronoi_classify_atom.h / .cpp
  README.txt（本说明）
  BUILD_CMAKE.md（CMake 编译说明：仅特征 / 推理模式）
  VORONOI_cmake_append_snippet.cmake（推理模式：可追加到 VORONOI.cmake 的片段）

================================================================================
