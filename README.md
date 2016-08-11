HCC : An open source C++ compiler for heterogeneous devices
===========================================================
This repository hosts HCC compiler implementation project. The goal is to implement a compiler that takes a program conforming parallel programming standards such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP and transforms it into the following targets:

1. AMD GCN ISA
2. HSA BRIG

The project is based on LLVM+CLANG.  For more information, please visit the [hcc wiki][1]:

[https://github.com/RadeonOpenCompute/hcc/wiki][1]

Git submodules
--------------
The project now employs git submodules to manage external components it depends upon. It it advised to add `--recursive` when you clone the project so all submodules are fetched automatically.

For example:
```
# automatically fetches all submodules
git clone --recursive -b clang_tot_upgrade git@github.com:RadeonOpenCompute/hcc.git
```

For more information about git submodules, please refer to [git documentation][2].

[//]: # (References)
[1]: https://github.com/RadeonOpenCompute/hcc/wiki
[2]: https://git-scm.com/book/en/v2/Git-Tools-Submodules

Device libraries
----------------
HCC device library is a part of [ROCm-Device-Libs](https://github.com/RadeonOpenCompute/ROCm-Device-Libs).
When compiling device code with hcc, rocm-device-libs package needs to be installed.
