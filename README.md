HCC : An open source C++ compiler for heterogeneous devices
===========================================================
This repository hosts HCC compiler implementation project. The goal is to 
implement a compiler that takes a program conforming parallel programming 
standards such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP and transforms it 
into AMD GCN ISA.

The project is based on LLVM+CLANG. For more information, please visit the 
[hcc wiki][1]:

[https://github.com/RadeonOpenCompute/hcc/wiki][1]

Download HCC
============
The project now employs git submodules to manage external components it depends 
upon. It it advised to add `--recursive` when you clone the project so all 
submodules are fetched automatically.

For example:
```bash
# automatically fetches all submodules
git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git
```

For more information about git submodules, please refer to [git documentation][2].

Build HCC from source
=====================
To configure and build HCC from source, use the following steps:
```bash
mkdir -p build; cd build
# NUM_BUILD_THREADS is optional
# set the number to your CPU core numbers time 2 is recommended
# in this example we set it to 96
cmake -DNUM_BUILD_THREADS=96 \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make
```

To install it, use the following steps:
```bash
sudo make install
```

Use HCC
=======
For C++AMP source codes:
```bash
hcc `clamp-config --cxxflags --ldflags` foo.cpp
```

For HC source codes:
```bash
hcc `hcc-config --cxxflags --ldflags` foo.cpp
```

In case you build HCC from source and want to use the compiled binaries
directly in the build directory:

For C++AMP source codes:
```bash
# notice the --build flag
bin/hcc `bin/clamp-config --build --cxxflags --ldflags` foo.cpp
```

For HC source codes:
```bash
# notice the --build flag
bin/hcc `bin/hcc-config --build --cxxflags --ldflags` foo.cpp
```

Multiple ISA
============

HCC now supports having multiple GCN ISAs in one executable file. You can do it 
in different ways:

use `--amdgpu-target=` command line option
------------------------------------------
It's possible to specify multiple `--amdgpu-target=` option. Example:

```bash
# ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would 
# be produced
hcc `hcc-config --cxxflags --ldflags` \
    --amdgpu-target=gfx701 \
    --amdgpu-target=gfx801 \
    --amdgpu-target=gfx802 \
    --amdgpu-target=gfx803 \
    foo.cpp
```

use `HCC_AMDGPU_TARGET` env var
------------------------------------------
Use `,` to delimit each AMDGPU target in HCC. Example:

```bash
export HCC_AMDGPU_TARGET=gfx701,gfx801,gfx802,gfx803
# ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would 
# be produced
hcc `hcc-config --cxxflags --ldflags` foo.cpp
```

configure HCC use CMake `HSA_AMDGPU_GPU_TARGET` variable
---------------------------------------------------------
If you build HCC from source, it's possible to configure it to automatically
produce multiple ISAs via `HSA_AMDGPU_GPU_TARGET` CMake variable.

Use `;` to delimit each AMDGPU target. Example:

```bash
# ISA for Hawaii(gfx701), Carrizo(gfx801), Tonga(gfx802) and Fiji(gfx803) would 
# be produced by default
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DROCM_DEVICE_LIB_DIR=~hcc/ROCm-Device-Libs/build/dist/lib \
    -DHSA_AMDGPU_GPU_TARGET="gfx701;gfx801;gfx802;gfx803" \
    ../hcc
```

CodeXL Activity Logger Profiling
================================
To enable [CodeXL Activity Logger][7] profiling, enable the `HCC_CODEXL_PROFILING` environment variable and the `CODEXL_PROFILING` CMake flag.

Configure the build in the following way:
```bash
mkdir -p build; cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCODEXL_PROFILING=ON \
    ..
make
```

In your application compiled using HCC, include the profiling header file:
```cpp
#include <hc_profile.hpp>
```

To profile a function, place the following marker in the function:
```cpp
cxlMarker f1 = CXL_MARKER;
```

To profile a class, place the following marker as a data member of that class:
```cpp
cxlMarker c1 = CXL_MARKER_CLASS;
```

For information about the usage of the Activity Logger for profiling, please refer to its [documentation][8].

[//]: # (References)
[1]: https://github.com/RadeonOpenCompute/hcc/wiki
[2]: https://git-scm.com/book/en/v2/Git-Tools-Submodules
[7]: https://github.com/RadeonOpenCompute/ROCm-Profiler/tree/master/CXLActivityLogger
[8]: https://github.com/RadeonOpenCompute/ROCm-Profiler/blob/master/CXLActivityLogger/doc/AMDTActivityLogger.pdf
