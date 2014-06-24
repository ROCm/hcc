
                    C++AMP Compiler Implementation for OpenCL/SPIR

Introduction
--------------------------------------------------------------------------------
This repository hosts C++AMP compiler implementation project. The goal is to
implement a compiler that takes a program conforming C++AMP standard and
transforms it into a SPIR binary for OpenCL. It will be based on LLVM+CLANG.


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


Build Instruction
--------------------------------------------------------------------------------
* Prerequisite/OpenCL

Make sure OpenCL SDK is installed in your system. You can test it with the clInfo
utility at http://graphics.stanford.edu/~yoel/notes/clInfo.c

Here is a snippet of the output message from clInfo, running on a Linux w/ ATI GPU
-------------------------------------------------------------------------
Found 1 platform(s).
platform[0x7f7950040140]: profile: FULL_PROFILE
platform[0x7f7950040140]: version: OpenCL 1.2 AMD-APP (923.1)
platform[0x7f7950040140]: name: AMD Accelerated Parallel Processing
platform[0x7f7950040140]: vendor: Advanced Micro Devices, Inc.
platform[0x7f7950040140]: extensions: cl_khr_icd cl_amd_event_callback cl_amd_offline_devices
platform[0x7f7950040140]: Found 2 device(s).
        device[0x27adc90]: NAME: Capeverde
        device[0x27adc90]: VENDOR: Advanced Micro Devices, Inc.
        device[0x27adc90]: PROFILE: FULL_PROFILE
        device[0x27adc90]: VERSION: OpenCL 1.2 AMD-APP (923.1)
        device[0x27adc90]: EXTENSIONS: cl_khr_fp64 cl_amd_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_gl_sharing cl_ext_atomic_counters_32 cl_amd_device_attribute_query cl_amd_vec3 cl_amd_printf cl_amd_media_ops cl_amd_popcnt 
        device[0x27adc90]: DRIVER_VERSION: CAL 1.4.1720 (VM)

        device[0x27adc90]: Type: GPU
--------------------------------------------------------------------------

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
      -DCLANG_URL=git@bitbucket.org:${username}/cppamp-ng.git \
      -DOPENCL_HEADER_DIR=<directory that contains cl.h/> \
      -DOPENCL_LIBRARY_DIR=<directory that contains libOpenCL.so on Linux> \
      -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=CBackend \
      -DGMAC_URL=https://bitbucket.org/multicoreware/gmac \
      -DCMAKE_BUILD_TYPE=Debug
  The default paths for major OpenCL stacks are searched. For example,
  on x86-64 platforms with AMD GPU and AMD APP SDK installed at the
  default path, this becomes
  # cmake ../src \
      -DCLANG_URL=git@bitbucket.org:${username}/cppamp-ng.git \
      -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=CBackend \
      -DGMAC_URL=https://bitbucket.org/multicoreware/gmac \
      -DCMAKE_BUILD_TYPE=Debug

4. Build the whole system. This will build clang and other libraries
   that require one time build.
  # make [-j #] world           // # is the number of parallel builds
  # make                        // this builds llvm utilities

   For recurring builds for llvm and clang:
  # make

5. Test. (optional, would be more useful in development cycle)
  # make test

6. Rebuild build system from CMake. When you bring changes to build system by
   modifying CMakeFiles.txt for example, the build system could be messed up.
   When such happens, you can remove cache of CMake and repopulate the build
   system from scratch.
  # cd build
  # rm -f CMakeCache.txt         // or use rm -rf *
  # (repeat No. 3 in this section)


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


