# HC API : An HSA-extension to C++AMP

HC is a C++ API derived from C++AMP. It has nearly identical features with C++AMP in:

- abstract modeling of GPU devices: accelerator / acclerator_view
- multi-dimensional array / array_view
- multi-dimensional index / extent
- parallel_for_each kernel launching interface
- math & atomic functions

And it comes with a few HSA-specific features:

- relaxed rules in operations allowed in kernels
- new syntax of tiled_extent and tiled_index
- dynamic group segment memory allocation
- true asynchronous kernel launching behavior
- additional HSA-specific APIs

---

# HC API

HC comes with two header files as of now:

- <hc.hpp> : Main header file for HC
- <hc_math.hpp> : Math functions for HC

Most HC APIs are stored under "hc" namespace, and the class name is the same as their counterpart in C++AMP "Concurrency" namespace.  Users of C++AMP should find it easy to switch from C++AMP to HC.

| HC | C++AMP |
|----|--------|
| Concurrency::accelerator | hc::accelerator |
| Concurrency::accelerator_view | hc::accelerator_view |
| Concurrency::extent | hc::extent |
| Concurrency::index | hc::index |
| Concurrency::completion_future | hc::completion_future |
| Concurrency::array | hc::array |
| Concurrency::array_view | hc::array_view |

---

# How to build programs with HC API

Use "hcc-config", instead of "clamp-config", or you could manually add "-hc" when you invoke clang++. Also, "hcc" is added as an alias for "clang++".

For example:

```
hcc `hcc-config --cxxflags --ldflags` foo.cpp -o foo
```

---

# HCC built-in macros

Built-in macros:

| Macro | Meaning |
|----|--------|
| ```__HCC__``` | always be 1 |
| ```__hcc_major__``` | major version number of HCC |
| ```__hcc_minor__``` | minor version number of HCC |
| ```__hcc_patchlevel__``` | patchlevel of HCC |
| ```__hcc_version__``` | combined string of ```__hcc_major__```, ```__hcc_minor__```, ```__hcc_patchlevel__``` |

The rule for ```__hcc_patchlevel__``` is: yyWW-(HCC driver git commit #)-(HCC clang git commit #)
- yy stands for the last 2 digits of the year
- WW stands for the week number of the year

Macros for language modes in use:

| Macro | Meaning |
|----|--------|
| ```__KALMAR_AMP__``` | 1 in case in C++ AMP mode (-std=c++amp) |
| ```__KALMAR_HC__``` | 1 in case in HC mode (-hc) |

Compilation mode:
HCC is a single-source compiler where kernel codes and host codes can reside in the same file. Internally HCC would trigger 2 compilation iterations, and the following macros can be user by user programs to determine which mode the compiler is in.

| Macro | Meaning |
|----|--------|
| ```__KALMAR_ACCELERATOR__``` | not 0 in case the compiler runs in kernel code compilation mode |
| ```__KALMAR_CPU__``` | not 0 in case the compiler runs in host code compilation mode |

---

# HC-specific features

- relaxed rules in operations allowed in kernels
- new syntax of tiled_extent and tiled_index
- dynamic group segment memory allocation
- true asynchronous kernel launching behavior
- additional HSA-specific APIs
