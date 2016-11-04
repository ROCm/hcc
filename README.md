## HCC : An open source C++ compiler for heterogeneous compute

This repository hosts HCC compiler implementation project. The goal is to implement a compiler that takes a program conforming parallel programming standards such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP and transforms it into AMD GCN ISA.

The project is based on LLVM+CLANG.  For more information, please visit the [hcc wiki][1]:

[https://github.com/RadeonOpenCompute/hcc/wiki][1]


### What's new in the ROCm 1.3 release

1. A lot of bug fixes
1. Added compiler and HCC runtime support for new HIP APIs
1. New device linking algorithm to support static library
1. Support for Ubuntu 16.04 and Fedora 23
1. Performance optimizations
1. Polaris 10 and Polaris 11 support

