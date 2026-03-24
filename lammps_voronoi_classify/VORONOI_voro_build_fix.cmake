# -----------------------------------------------------------------------------
# Voro++ 用 Intel / nvcc_wrapper 等编译器构建时可能报错：
#   this declaration has no storage class or type specifier
#   cell.o: cell.cc \
# 原因是 Voro++ 的 Makefile 与 icpc/nvcc 不兼容，依赖行被当成源码编译。
#
# 解决：让 Voro++ 单独用 g++ 编译，LAMMPS 主体仍可用 Intel 或 Kokkos/nvcc_wrapper。
# 用法：在 LAMMPS 的 cmake/Modules/Packages/VORONOI.cmake 里，
#       在 set(VORO_BUILD_OPTIONS ...) 之后、find_program(HAVE_PATCH ...) 之前插入下面整段。
# -----------------------------------------------------------------------------

# 若主编译器为 Intel 或 nvcc/nvcc_wrapper（Kokkos-CUDA），则 Voro++ 强制用 g++ 构建
set(VORO_USE_GXX OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(VORO_USE_GXX ON)
  set(VORO_USE_GXX_REASON "main compiler is Intel")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVIDIA")
  set(VORO_USE_GXX ON)
  set(VORO_USE_GXX_REASON "main compiler is NVIDIA (nvcc)")
endif()
if(CMAKE_CXX_COMPILER AND (CMAKE_CXX_COMPILER MATCHES "nvcc" OR CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper"))
  set(VORO_USE_GXX ON)
  set(VORO_USE_GXX_REASON "CXX is nvcc/nvcc_wrapper (Kokkos-CUDA)")
endif()
if(VORO_USE_GXX)
  find_program(VORO_GXX NAMES g++ gcc)
  if(VORO_GXX)
    message(STATUS "VORONOI: Building Voro++ with g++ (${VORO_USE_GXX_REASON})")

    # 从 VORO_BUILD_CFLAGS 中过滤掉 nvcc 专用的 CUDA 选项（如 -Xcudafe / --diag_suppress 等），
    # 否则传给 g++ 会报 “unrecognized command line option”。
    set(VORO_BUILD_CFLAGS_FILTERED "")
    foreach(_flag IN LISTS VORO_BUILD_CFLAGS)
      if(_flag MATCHES "^-X" OR _flag MATCHES "cudafe" OR _flag MATCHES "diag_suppress")
        # 跳过 nvcc 专用选项
      else()
        list(APPEND VORO_BUILD_CFLAGS_FILTERED "${_flag}")
      endif()
    endforeach()

    set(VORO_BUILD_OPTIONS CXX=${VORO_GXX} CFLAGS=${VORO_BUILD_CFLAGS_FILTERED})
  endif()
  unset(VORO_USE_GXX_REASON)
endif()
unset(VORO_USE_GXX)
