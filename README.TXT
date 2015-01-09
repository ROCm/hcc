C++AMP Compiler Implementation for OpenCL/HSA

Introduction
--------------------------------------------------------------------------------
This repository hosts C++AMP compiler implementation project. The goal is to
implement a compiler that takes a program conforming C++AMP standard and
transforms it into the following targets:

1. OpenCL SPIR
2. OpenCL C
3. HSA BRIG 

The project is based on LLVM+CLANG.

HSA-specific features are also gradually being added to the project.  For more HSA-related information please visit: https://bitbucket.org/multicoreware/cppamp-driver-ng/wiki/HSA%20Support%20Status


Repository Structure
-------------------------------------------------------------------------------
The current repository is composed of as following:
1. cppamp-driver
  - This repository. It contains all the C++AMP related implementations, tests,
    scripts and build system.
2. cppamp
  - Modified Clang. It will be downloaded as part of the build procedure which
    will be described later. This repository is basically a clone of Clang
    repository. Default branch will be a placeholder for the cloned Clang
    repository. C++AMP specific changes will be made to cppamp branch, which is
    the main branch for the C++AMP development. Note that one should NOT commit
    to the default trunk(admin should reject it even if such case happens).
    Developers can also create their own branches for their own development.


Build Instruction for HSA
--------------------------------------------------------------------------------
* Prerequisite/OpenCL

1. Linux distribution: Ubuntu 14.04 or higher

2. OpenCL Driver: OpenCL 1.1 or above
  - AMD:
    - OpenCL 1.2 : http://support.amd.com/en-us/download
    - (NOT thoroughly tested yet) OpenCL 2.0 : http://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx
  - Intel:
    - https://software.intel.com/en-us/articles/opencl-drivers
  - NVidia:
    - http://www.nvidia.com/Download/index.aspx?lang=en-us

3. OpencL SDK: OpenCL 1.1 or above
  - AMD:
    - http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
  - Intel:
    - https://software.intel.com/en-us/intel-opencl
  - NVidia:
    - https://developer.nvidia.com/cuda-downloads

* Prerequisite/HSA

1. Linux distribution: Ubuntu 14.04 or higher

2. HSA Linux Driver:
  - https://github.com/HSAFoundation/HSA-Drivers-Linux-AMD
  - make sure kfd_check_installation.sh shows "Can run HSA YES"

3. HSA Runtime:
  - https://github.com/HSAFoundation/HSA-Runtime-AMD

4. HSAIL Compiler:
   There are 2 versions available: HSAIL-HLC-Stable and HSAIL-HLC-Development.  It is suggested to use Stable version at this moment.  Debian packages are built to only support HSAIL-HLC-Stable.
  - Stable: https://github.com/HSAFoundation/HSAIL-HLC-Stable
  - Development: https://github.com/HSAFoundation/HSAIL-HLC-Development

* Prerequisite/Repository

For any development work, you should fork from them and create your own
repository for development. This build procedure assumes that you will
work on your own repository. As such, the following repositories should
be available:

  https://${username}@bitbucket.org/${username}/cppamp-driver-ng
  https://${username}@bitbucket.org/${username}/cppamp-ng

, where ${username} is a placeholder for your ID for bitbucket.org.

Also developers need to register their public key to bitbucket. Instructions
are:

1. Create public-private key pair
  # ssh-keygen

2. Upload .ssh/id_rsa.pub to bitbucket, under Admin of your account.


* Now, here goes actual build instructions.

1. Prepare a directory for work space.
   Please make sure there is no special characters like '+' or space in the full path.
  # mkdir cppamp

2. Pull your cppamp-driver-ng repository from bitbucket.
  # cd cppamp
  # git clone git@bitbucket.org:${username}/cppamp-driver-ng.git src

3. Create a build directory and configure using CMake.
  # mkdir build
  # cd build
  # cmake ../src

  The default options can be overridden on the command line:
  # cmake ../src \
      -DCLANG_URL=https://bitbucket.org/multicoreware/cppamp-ng.git \
      -DOPENCL_HEADER_DIR=/opt/AMDAPP/include \
      -DOPENCL_LIBRARY_DIR=/opt/AMDAPP/lib/x86_64 \
      -DHSA_HEADER_DIR=/opt/hsa/include \
      -DHSA_LIBRARY_DIR=/opt/hsa/lib \
      -DHSA_EXT_LIBRARY_DIR=/opt/hsa/lib \
      -DHSA_KMT_LIBRARY_DIR=/opt/hsa/lib \
      -DHSAIL_COMPILER_DIR=/opt/amd/bin \
      -DCMAKE_BUILD_TYPE=Release

   - CLANG_URL : URL of C++AMP clang implementation

   - OPENCL_HEADER_DIR : OpenCL header include path
   - OPENCL_LIBRARY_DIR : OpenCL runtime library path

   - HSA_HEADER_DIR : HSA runtime header include path
   - HSA_LIBRARY_DIR : HSA runtime library path
   - HSA_EXT_LIBRARY_DIR : HSA runtime extension library path

   - HSA_KMT_LIBRARY_DIR : HSA kernel library path

   - HSAIL_COMPILER_DIR : HSAIL compiler binary path

   - CMAKE_BUILD_TYPE : Release or Debug  

4. Build the whole system. This will build clang and other libraries
   that require one time build.
  # make [-j #] world           // # is the number of parallel builds
  # make                        // this builds llvm utilities

   For recurring builds for llvm and clang:
  # make

5. Test. (optional, would be more useful in development cycle)
  # make test

   Please notice HSA-specific tests shall be executed in the following way:
  # CLAMP_RUNTIME=HSA make test

6. Rebuild build system from CMake. When you bring changes to build system by
   modifying CMakeFiles.txt for example, the build system could be messed up.
   When such happens, you can remove cache of CMake and repopulate the build
   system from scratch.
  # cd build
  # rm -f CMakeCache.txt         // or use rm -rf *
  # (repeat No. 3 in this section)


Dynamic Libaries
-------------------------------------------------------------------------------
After building, please change /etc/ld.so.conf to let dynamic libraries be
locatable at runtime:

# If you install deb files:
# C++AMP runtime libraries
# libc++ & C++AMP runtime implementations
/opt/clamp/lib

# If you build from source:
# C++AMP runtime libraries
# libc++
(path_of_your_build_directory)/libc++/libcxx/lib
(path_of_your_build_directory)/libc++/libcxxrt/lib
# C++AMP runtime implementations
(path_of_your_build_directory)/build/Release/lib

Remember to use: sudo ldconfig -v to reload ld.so cache.


Environment varialbes
-------------------------------------------------------------------------------
The following environment variables can change the behavior of Clamp.

1. CLAMP_NOSPIR

export CLAMP_NOSPIR=1 could force Clamp emit OpenCL C instead of SPIR at
compile time.

2. CLAMP_RUNTIME

C++AMP programs will automatically pick C++AMP runtime to use in the following
precedence:

- HSA
- OpenCL 1.2
- OpenCL 1.1

Set CLAMP_RUNTIME to different values to force pick different C++AMP runtime.
Available options are:

- HSA : HSA
- CL12 : OpenCL 1.2 (with zero-copy performance optimizations)
- CL11 : OpenCL 1.1


How to push your changes to the main repository
-------------------------------------------------------------------------------
Create pull request from bitbucket.


How to make your repository synchronized by pulling from main repository
-------------------------------------------------------------------------------
Synchronization can be done via scripts:

  # cppamp/src/scripts/pull_from_trunk.py

It will pull from both cppamp-driver and cppamp. Note that you still need to
update repositories. Again, when build system does not seem to work, you will
need to redo step No. 6 from the build instruction.


Directory Structure
-------------------------------------------------------------------------------
$cppamp                // top-level directory
  build/               // all outputs
  src/                 // llvm
    compiler/
      tools/
        clang/         // modified Clang frontend
  libcxx/              // libc++ to help compile
  tests/               // tests including unit tests, TDD issues
  include/             // C++AMP related headers
  lib/                 // C++AMP related implementions


Synchronization with Clang repository
-------------------------------------------------------------------------------
C++AMP is an extension to C++11. Since support for C++11 is still very new and
mostly in beta status from Clang, we will need to be periodically synchronized
to the Clang development. For this, we create a repository which is a mirror of
Clang repository, but in Mercurial format. Then, by pulling changes from the
repository into the localized development Clang repository, synchronization can
be done.

We first need to be able to use hgsubversion, a tool converting a Subversion
repository into Mercurial repository. Note that the selection of hgsubversion
is not mendatory, but is one of such tools that we find useful so far. If you
know better tool for this work, you may use that as well. Here goes procedure
to make hgsubversion available:

  1. Assuming Ubuntu Linux, we will need a package installed.
    # apt-get install python-subversion

  2. Download hgsubversion package.
    # hg clone http://bitbucket.org/durin42/hgsubversion hgsubversion

  3. Edit ~/.hgrc to use the package which is an extension to Mercurial:

    [extensions]
    hgsubversion = <the location where you downloaded hgsubversion>/hgsubversion

    For instance, if you downloaded hgsubversion at /opt/hgsubversion, then 

    [extensions]
    hgsubversion = /opt/hgsubversion/hgsubversion

  4. Make sure you correctly install everything by:
    # hg help subversion
.

Then, we need to create a Clang repository in Mercurial format using the tool
we just installed:

  # cd cppamp     <- your top level development directory
  # hg clone http://llvm.org/svn/llvm-project/cfe/trunk clangsvn

. Note that this is required for the first time and later you can just reuse it
with hg pull/update command:

  # cd clangsvn
  # hg pull
  # hg update

. Now, you can synchronize by pulling changes from the repository:

  # cd cppamp/src/compiler/tools/clang
  # hg pull ../../../../clangsvn
  # hg merge -r <the most recent changeset number from clangsvn>

. Note that you will need to push to make the change visible to other people.


Test Driven Development
-------------------------------------------------------------------------------
C++AMP development project adapts TDD(Test Driven Development). It helps keep
track of development status by working tests intact. Check out src/tests for
actual implementation.


Tips for Developers
-------------------------------------------------------------------------------
1. Debugging Clang

  Clang itself is not debuggable as it is actually 'driver' that calls actual
  compiler. Debugging becomes available once you give '-v' option to the Clang
  execution:

    # clang -v <other arguments>

  It will print out verbose messages that actually it runs. Using this
  information will enable debugging.
