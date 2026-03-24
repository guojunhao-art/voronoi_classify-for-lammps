# 使用 CMake 编译本模块

本文说明如何用 CMake 编译带有 `compute voronoi/classify/atom` 的 LAMMPS：仅特征模式（不链接 LibTorch）与推理模式（链接 LibTorch）。

---

## 一、前置：把源码放进 LAMMPS 树

1. 将本目录下 4 个源码文件复制/覆盖到 LAMMPS 的 **VORONOI** 源码目录：
   - 目标目录：`<LAMMPS根目录>/src/VORONOI/`
   - 文件：
     - `compute_voronoi_atom.h`
     - `compute_voronoi_atom.cpp`
     - `compute_voronoi_classify_atom.h`
     - `compute_voronoi_classify_atom.cpp`
2. 确认 LAMMPS 根目录结构包含 `src/`、`cmake/` 等（标准发布或 git 克隆均可）。

---

## 二、仅特征模式（不启用 LibTorch）

只生成 40 维特征、不在 LAMMPS 内跑模型时，只需打开 **VORONOI** 包，无需 LibTorch。

### 2.1 配置

在 LAMMPS 根目录外新建并进入 build 目录，例如：

```bash
cd /path/to/lammps
mkdir build && cd build
```

运行 CMake（**源目录指向 `cmake` 子目录**）：

```bash
cmake -S cmake -B . \
  -DPKG_VORONOI=ON \
  -DCMAKE_BUILD_TYPE=Release
```

- 若需 MPI：保证环境已配置好 MPI，或使用 LAMMPS 预设（如 `-C ../cmake/presets/most.cmake` 再设 `-DPKG_VORONOI=ON`）。
- 安装路径默认在 `~/.local`，可改：`-DCMAKE_INSTALL_PREFIX=/your/install/path`。

### 2.2 编译与安装

```bash
cmake --build . -j$(nproc)
make install   # 可选
```

可执行文件在 build 目录下，通常名为 `lmp`。

---

## 三、推理模式（启用 LibTorch）

要在 LAMMPS 内用 `compute voronoi/classify/atom ... model.pt` 做推理，需在编译时定义 **LAMMPS_TORCH** 并链接 LibTorch。

### 3.1 准备 LibTorch

- **推荐**：从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 下载与系统/CUDA 匹配的 **LibTorch** 预编译包（C++/CPU 或 CUDA），解压到某目录，例如 `/opt/libtorch`。
- 或自行从源码编译 LibTorch，并记下安装目录。

### 3.2 让 CMake 找到 LibTorch 并打开 LAMMPS_TORCH

有两种做法。

#### 方法 A：在 VORONOI 的 CMake 里增加可选 LibTorch（推荐）

1. 打开 LAMMPS 的  
   `<LAMMPS根目录>/cmake/Modules/Packages/VORONOI.cmake`
2. 在 **文件末尾**（最后一个 `endif()` 之后）**整段追加**本目录下的  
   `VORONOI_cmake_append_snippet.cmake` 中的内容（该片段会处理 `Torch::Torch` 目标及部分仅提供 `Torch_LIBRARIES` 的安装）。
3. 配置时打开 VORONOI 与 LibTorch，并指定 LibTorch 路径（二选一）：

```bash
# 方式 1：指定 LibTorch 根目录（推荐）
cmake -S cmake -B . \
  -DPKG_VORONOI=ON \
  -DPKG_VORONOI_TORCH=ON \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_BUILD_TYPE=Release

# 方式 2：直接指定 Torch 的 CMake 配置目录（若方式 1 仍报 target 未找到）
cmake -S cmake -B . \
  -DPKG_VORONOI=ON \
  -DPKG_VORONOI_TORCH=ON \
  -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
  -DCMAKE_BUILD_TYPE=Release
```

4. 编译：

```bash
cmake --build . -j$(nproc)
```

若 LibTorch 不在标准路径，一般设置 **CMAKE_PREFIX_PATH** 即可；若需指定 CUDA 版 LibTorch，保证与当前 CUDA 驱动/工具链一致。

#### 方法 B：不改 VORONOI.cmake，手动加定义和链接

若不修改 `VORONOI.cmake`，可自行传编译定义和链接库（需根据实际 LibTorch 安装调整）：

```bash
cmake -S cmake -B . \
  -DPKG_VORONOI=ON \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCMAKE_CXX_FLAGS="-DLAMMPS_TORCH" \
  -DCMAKE_BUILD_TYPE=Release
```

并在 CMake 配置后，在生成的构建系统中为 `lammps` 目标增加对 LibTorch 的链接（例如在 `CMakeLists.txt` 或通过 `target_link_libraries` 的额外配置）。由于 LAMMPS 主 CMake 不会自动为 VORONOI 加 Torch，**更推荐用方法 A**，一次改好 `VORONOI.cmake` 即可。

### 3.3 常见问题

- **Torch not found**：检查 `CMAKE_PREFIX_PATH` 指向 LibTorch 的**根目录**（其下应有 `share/cmake/Torch/` 等）。
- **Target "lammps" links to target "Torch::Torch" but the target was not found**  
  表示 CMake 找到了 Torch 包但当前 LibTorch 未导出 `Torch::Torch` 目标。处理方式：
  1. 用 **`Torch_DIR`** 明确指定配置目录再配置一次：  
     `-DTorch_DIR=/path/to/libtorch/share/cmake/Torch`（把 `/path/to/libtorch` 换成你的 LibTorch 解压路径）。
  2. 确保使用的是**官方预编译 LibTorch**（从 PyTorch 官网下载的 libtorch 压缩包解压后的目录），其下应有 `share/cmake/Torch/TorchConfig.cmake`。
  3. 若已使用本目录提供的 **最新 `VORONOI_cmake_append_snippet.cmake`** 追加到 `VORONOI.cmake`，片段会在无 `Torch::Torch` 时尝试用 `Torch_LIBRARIES` / `TORCH_LIBRARIES` 等变量链接；若仍报错，请确认 LibTorch 版本与 CMake 版本兼容。
- **CUDA 版本**：LibTorch 的 CUDA 版本宜与系统 CUDA 一致或兼容，否则可能运行时报错。
- **C++14/17**：LibTorch 通常需要 C++14 及以上；LAMMPS 默认已满足。

---

## 四、编译选项小结

| 目标           | CMake 配置要点 |
|----------------|----------------|
| 仅特征模式     | `-DPKG_VORONOI=ON` |
| 推理模式       | `-DPKG_VORONOI=ON`，并在 VORONOI.cmake 中启用 `PKG_VORONOI_TORCH` 且 `find_package(Torch)` 成功，或手动 `-DLAMMPS_TORCH` 并链接 LibTorch |
| LibTorch 路径  | `-DCMAKE_PREFIX_PATH=/path/to/libtorch` |

---

## 五、验证

- **特征模式**：运行 LAMMPS 输入脚本，其中包含  
  `compute cc all voronoi/classify/atom voro`  
  再 `dump ... c_cc[*]`，检查输出是否为 40 列。
- **推理模式**：使用  
  `compute cc all voronoi/classify/atom voro type g1 g2 model.pt`  
  再 `dump ... c_cc[1]`，检查是否得到单列 0/1。

文档版本与 `README.txt` 一致，如有 LAMMPS 或 LibTorch 版本差异，可据此调整路径或选项。

---

## 六、编译阶段报错：Voro++ 与 Intel / nvcc_wrapper 编译器

若在 **编译阶段**（非 configure）出现类似：

```text
Performing build step for 'voro_build'
...
/tmp/...cell.cpp4.ii(1): error: this declaration has no storage class or type specifier
  cell.o: cell.cc \
  ^
```

说明当前用 **Intel（icpc）** 或 **Kokkos 的 nvcc_wrapper** 作为 CXX 构建 Voro++，Voro++ 的 Makefile 与这些编译器不兼容，依赖行被误当成了源码。

**处理方式（二选一）：**

1. **推荐：Voro++ 单独用 g++ 编译**  
   打开 LAMMPS 的 `cmake/Modules/Packages/VORONOI.cmake`，在 **`set(VORO_BUILD_OPTIONS ...)` 之后、`find_program(HAVE_PATCH patch)` 之前** 插入 **本目录下 `VORONOI_voro_build_fix.cmake` 中的整段内容**（该片段会检测 Intel 与 nvcc/nvcc_wrapper，并自动改用 g++ 编 Voro++）。

   插入后清掉 Voro++ 的构建缓存再重新配置、编译，例如：

   ```bash
   rm -rf build/voro_build-prefix
   cmake -C ../cmake/presets/basic.cmake -C ../cmake/presets/kokkos-cuda.cmake \
         -DKokkos_ENABLE_CUDA=on -DCMAKE_CXX_COMPILER=.../nvcc_wrapper \
         -DPKG_VORONOI=ON -DPKG_VORONOI_TORCH=ON -DCMAKE_PREFIX_PATH=.../libtorch ../cmake
   cmake --build build -j$(nproc)
   ```

   LAMMPS 主体仍用 nvcc_wrapper/Kokkos，只有 Voro++ 用 g++。

2. **改用 GCC 编整个 LAMMPS**  
   若不需要 Kokkos-CUDA，可改用 g++ 编 LAMMPS，则 Voro++ 也会用 g++，不会再触发该错误。
