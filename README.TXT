HCC : An open source C++ compiler for heterogeneous devices

Introduction
--------------------------------------------------------------------------------
This repository hosts HCC compiler implementation project. The goal is to
implement a compiler that takes a program conforming parallel programming
standards such as C++AMP, HC, C++17 ParallelSTL, or OpenMP and transforms it
into the following targets:

1. AMD GCN ISA
2. HSA BRIG

The project is based on LLVM+CLANG.  For more information, please visit the hcc wiki:

https://github.com/RadeonOpenCompute/hcc/wiki
