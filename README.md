HCC : An open source C++ compiler for heterogeneous devices
===========================================================
This repository hosts HCC compiler implementation project. The goal is to implement a compiler that takes a program conforming parallel programming standards such as C++ AMP, HC, C++ 17 ParallelSTL, or OpenMP and transforms it into AMD GCN ISA.

The project is based on LLVM+CLANG.  For more information, please visit the [hcc wiki][1]:

[https://github.com/RadeonOpenCompute/hcc/wiki][1]

Git submodules
==============
The project now employs git submodules to manage external components it depends upon. It it advised to add `--recursive` when you clone the project so all submodules are fetched automatically.

For example:
```
# automatically fetches all submodules
git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git
```

For more information about git submodules, please refer to [git documentation][2].

[//]: # (References)
[1]: https://github.com/RadeonOpenCompute/hcc/wiki
[2]: https://git-scm.com/book/en/v2/Git-Tools-Submodules

Device libraries
================
HCC device library is a part of [ROCm-Device-Libs](https://github.com/RadeonOpenCompute/ROCm-Device-Libs).
When compiling device code with hcc, rocm-device-libs package needs to be
installed.

In case rocm-device-libs package is not present, you are required to build it
from source. Please refer to [ROCm-Device-Libs build procedure](https://github.com/RadeonOpenCompute/ROCm-Device-Libs#building) for more details.

Once it's built, run `make install` and config ToT HCC like:

```
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET=<AMD GPU ISA version string> \
    -DROCM_DEVICE_LIB_DIR=<where bitcodes of ROCm-Device-Libs are> \
    <ToT HCC checkout directory>
```

An example would be:
```
# Use AMD:AMDGPU:8:0:3 AMD GPU ISA
# ROCm-Device-Libs is built at ~/ocml/build , bitcodes are at
# ~/ocml/build/dist/lib
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET=AMD:AMDGPU:8:0:3 \
    -DROCM_DEVICE_LIB_DIR=~/ocml/build/dist/lib \
    ..
```

Multiple ISA
============

HCC now supports having multiple GCN ISAs in one executable file. You can do it in different ways:

use `--amdgpu-target=` command line option
------------------------------------------
It's possible to specify multiple `--amdgpu-target=` option. Example:

```
# ISA for Hawaii(7:0:1), Carrizo(8:0:1), Fiji(8:0:3) would be produced
hcc `hcc-config --cxxflags --ldflags` \
    --amdgpu-target=AMD:AMDGPU:7:0:1 \
    --amdgpu-target=AMD:AMDGPU:8:0:1 \
    --amdgpu-target=AMD:AMDGPU:8:0:3 \
    foo.cpp
```

use `HCC_AMDGPU_TARGET` env var
------------------------------------------
Use `,` to delimit each AMDGPU target in HCC. Example:

```
export HCC_AMDGPU_TARGET=AMD:AMDGPU:7:0:1,AMD:AMDGPU:8:0:1,AMD:AMDGPU:8:0:3
# ISA for Hawaii(7:0:1), Carrizo(8:0:1), Fiji(8:0:3) would be produced
hcc `hcc-config --cxxflags --ldflags` foo.cpp
```

configure HCC use CMake `HSA_AMDGPU_GPU_TARGET` variable
---------------------------------------------------------
If you build HCC from source, it's possible to configure it to automatically
produce multiple ISAs via `HSA_AMDGPU_GPU_TARGET` CMake variable.

Use `;` to delimit each AMDGPU target. Example:

```
# ISA for Hawaii(7:0:1), Carrizo(8:0:1), Fiji(8:0:3) be configured by default
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DROCM_DEVICE_LIB_DIR=~hcc/ROCm-Device-Libs/build/dist/lib \
    -DHSA_AMDGPU_GPU_TARGET="AMD:AMDGPU:7:0:1;AMD:AMDGPU:8:0:1;AMD:AMDGPU:8:0:3" \
    ../hcc
```

Building clang_tot_upgrade branch on Ubuntu 16.04.1
----------------------------------------------------
The following issue is common when building HCC tot branch.

```
In file included from /home/aditya/rocm/hcc.lc.tot/lib/mcwamp.cpp:8:
In file included from /usr/include/c++/v1/iostream:38:
In file included from /usr/include/c++/v1/ios:216:
In file included from /usr/include/c++/v1/__locale:15:
/usr/include/c++/v1/string:1938:44: error: 'basic_string<_CharT, _Traits, _Allocator>' is
      missing exception specification
      'noexcept(is_nothrow_copy_constructible<allocator_type>::value)'
basic_string<_CharT, _Traits, _Allocator>::basic_string(const allocator_type& __a)
                                           ^
/usr/include/c++/v1/string:1326:40: note: previous declaration is here
    _LIBCPP_INLINE_VISIBILITY explicit basic_string(const allocator_type& __a)
                                       ^
1 error generated.
lib/CMakeFiles/mcwamp.dir/build.make:62: recipe for target 'lib/CMakeFiles/mcwamp.dir/mcwamp.cpp.o' failed
make[2]: *** [lib/CMakeFiles/mcwamp.dir/mcwamp.cpp.o] Error 1
CMakeFiles/Makefile2:229: recipe for target 'lib/CMakeFiles/mcwamp.dir/all' failed
make[1]: *** [lib/CMakeFiles/mcwamp.dir/all] Error 2
Makefile:149: recipe for target 'all' failed
make: *** [all] Error 2
```

This is because of broken libc++ package that ships with ubuntu 16.04.1. 
This can be solved by installing libc++1 and libc++-dev from 
https://packages.debian.org/sid/libc++1
https://packages.debian.org/sid/libc++-dev

version at the time of writing this is 3.9.0-3 which is working (replaces 3.7.x).

```
wget http://ftp.us.debian.org/debian/pool/main/libc/libc++/libc++-dev_3.9.0-3_amd64.deb
wget http://ftp.us.debian.org/debian/pool/main/libc/libc++/libc++1_3.9.0-3_amd64.deb

dpkg -i libc++1_3.9.0-3_amd64.deb
dpkg -i libc++-dev_3.9.0-3_amd64.deb
```

This replaces previous version of libc++.
