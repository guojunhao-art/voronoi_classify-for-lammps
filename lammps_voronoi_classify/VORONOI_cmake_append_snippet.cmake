# -----------------------------------------------------------------------------
# 将本段追加到 LAMMPS 的 cmake/Modules/Packages/VORONOI.cmake 文件末尾
# （在最后一个 endif() 之后），即可在 CMake 配置时用 PKG_VORONOI_TORCH=ON
# 启用 LibTorch 推理。
#
# 配置示例（二选一）：
#   cmake -S cmake -B build -DPKG_VORONOI=ON -DPKG_VORONOI_TORCH=ON \
#         -DCMAKE_PREFIX_PATH=/path/to/libtorch
#   cmake -S cmake -B build -DPKG_VORONOI=ON -DPKG_VORONOI_TORCH=ON \
#         -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
# -----------------------------------------------------------------------------

option(PKG_VORONOI_TORCH "Build voronoi/classify/atom with LibTorch (model.pt)" OFF)
if(PKG_VORONOI AND PKG_VORONOI_TORCH)
  if(NOT Torch_DIR AND CMAKE_PREFIX_PATH)
    list(GET CMAKE_PREFIX_PATH 0 _torch_hint)
    if(_torch_hint)
      set(Torch_DIR "${_torch_hint}/share/cmake/Torch" CACHE PATH "LibTorch CMake config dir")
    endif()
    unset(_torch_hint)
  endif()
  find_package(Torch CONFIG)
  if(NOT Torch_FOUND)
    message(FATAL_ERROR "PKG_VORONOI_TORCH=ON but Torch not found. "
      "Set CMAKE_PREFIX_PATH to LibTorch root, or Torch_DIR to libtorch/share/cmake/Torch")
  endif()
  message(STATUS "VORONOI: LibTorch found, enabling LAMMPS_TORCH for classify/atom")
  target_compile_definitions(lammps PRIVATE LAMMPS_TORCH)
  # 使用 PUBLIC 使链接 lammps 的可执行文件（如 lmp）自动继承 LibTorch
  if(TARGET Torch::Torch)
    target_link_libraries(lammps PUBLIC Torch::Torch)
    if(TARGET lmp)
      target_link_libraries(lmp PRIVATE Torch::Torch)
    endif()
  elseif(DEFINED Torch_LIBRARIES)
    target_include_directories(lammps PRIVATE ${Torch_INCLUDE_DIRS})
    target_link_libraries(lammps PUBLIC ${Torch_LIBRARIES})
    if(DEFINED Torch_CXX_FLAGS)
      target_compile_options(lammps PRIVATE ${Torch_CXX_FLAGS})
    endif()
    if(TARGET lmp)
      target_link_libraries(lmp PRIVATE ${Torch_LIBRARIES})
    endif()
  elseif(DEFINED TORCH_LIBRARIES)
    target_include_directories(lammps PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(lammps PUBLIC ${TORCH_LIBRARIES})
    if(DEFINED TORCH_CXX_FLAGS)
      target_compile_options(lammps PRIVATE ${TORCH_CXX_FLAGS})
    endif()
    if(TARGET lmp)
      target_link_libraries(lmp PRIVATE ${TORCH_LIBRARIES})
    endif()
  else()
    message(FATAL_ERROR "PKG_VORONOI_TORCH: find_package(Torch) succeeded but no target Torch::Torch and no TORCH_LIBRARIES. Check your LibTorch installation.")
  endif()
endif()
