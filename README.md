HCC : An open source C++ compiler for heterogeneous devices
===========================================================
This repository hosts the HCC compiler implementation project. The goal is to 
implement a compiler that takes a program that conforms to a parallel programming 
standard such as HC, C++ 17 ParallelSTL and transforms it 
into the AMD GCN ISA.

The project is based on LLVM+CLANG. For more information, please visit the 
[hcc wiki][1]:

[https://github.com/RadeonOpenCompute/hcc/wiki][1]

Deprecation Notice
==================
AMD is deprecating HCC to put more focus on HIP development and on other languages supporting heterogeneous compute.    We will no longer develop any new feature in HCC and we will stop maintaining HCC after its final release, which is planned for June 2019.  If your application was developed with the hc C++ API, we would encourage you to transition it to other languages supported by AMD, such as [HIP](https://github.com/ROCm-Developer-Tools/HIP) or OpenCL.    HIP and hc language share the same compiler technology, so many hc kernel language features (including inline assembly) are also available through the HIP compilation path.

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
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

To install it, use the following steps:
```bash
sudo make install
```

Use HCC
=======

For HC source codes:
```bash
hcc -hc foo.cpp -o foo
```

Multiple ISA
============

HCC now supports having multiple GCN ISAs in one executable file. You can do it 
in different ways:

use `--amdgpu-target=` command line option
------------------------------------------
It's possible to specify multiple `--amdgpu-target=` option. Example:

```bash
# ISA for Fiji(gfx803) and Vega10(gfx900) would 
# be produced
hcc -hc \
    --amdgpu-target=gfx803 \
    --amdgpu-target=gfx900 \
    foo.cpp
```

configure HCC use CMake `HSA_AMDGPU_GPU_TARGET` variable
---------------------------------------------------------
If you build HCC from source, it's possible to configure it to automatically
produce multiple ISAs via `HSA_AMDGPU_GPU_TARGET` CMake variable.

Use `;` to delimit each AMDGPU target. Example:

```bash
# ISA for Fiji(gfx803) and Vega10(gfx900) would 
# be produced by default
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET="gfx803;gfx900" \
    ../hcc
```

CodeXL Activity Logger
======================
To enable the [CodeXL Activity Logger][7], use the `USE_CODEXL_ACTIVITY_LOGGER` 
environment variable.

Configure the build in the following way: 

```bash
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CODEXL_ACTIVITY_LOGGER=1 \
    <ToT HCC checkout directory>
```

In your application compiled using hcc, include the CodeXL Activity Logger header:
```
#include <CXLActivityLogger.h>
```

For information about the usage of the Activity Logger for profiling, please 
refer to its [documentation][8].

HCC with ThinLTO Linking
========================
To enable the ThinLTO link time, use the `KMTHINLTO` environment variable.

Set up your environment in the following way:
```
export KMTHINLTO=1
```
ThinLTO Phase 1 - Implemented
-----------------------------
For applications compiled using hcc, ThinLTO could significantly improve link-time
performance. This implementation will maintain kernels in their .bc file format,
create module-summaries for each, perform llvm-lto's cross-module function importing
and then perform clamp-device (which uses opt and llc tools) on each of the
kernel files. These files are linked with lld into one .hsaco per target specified.

ThinLTO Phase 2 - Under development
-----------------------------------
This ThinLTO implementation which will use llvm-lto LLVM tool to replace
clamp-device bash script. It adds an optllc option into ThinLTOGenerator,
which will perform in-program opt and codegen in parallel.

To use HCC Printf Functions
===========================
Set up environmental variable:
```bash
export HCC_ENABLE_PRINTF=1
```

Then compile the printf kernel with HCC_ENABLE_ACCELERATOR_PRINTF macro defined.
```bash
~/build/bin/hcc -hc -DHCC_ENABLE_ACCELERATOR_PRINTF -lhc_am -o printf.out ~/hcc/tests/Unit/HSA/printf.cpp
```

For more examples on how to use printf, see tests in `tests/Unit/HSA/printf*.cpp`.

[//]: # (References)
[1]: https://github.com/RadeonOpenCompute/hcc/wiki
[2]: https://git-scm.com/book/en/v2/Git-Tools-Submodules
[7]: https://github.com/RadeonOpenCompute/ROCm-Profiler/tree/master/CXLActivityLogger
[8]: https://github.com/RadeonOpenCompute/ROCm-Profiler/blob/master/CXLActivityLogger/doc/AMDTActivityLogger.pdf
