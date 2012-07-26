
                        C++AMP Implementation for HSAIL

Introduction
--------------------------------------------------------------------------------
This repository hosts C++AMP implementation on top of LLVM+CLANG. It aims to be
able to generate codes for HSAIL.


Build Instruction
--------------------------------------------------------------------------------
1. Prepare a directory
  # mkdir cppamp

2. Pull a repository from bitbucket:
  https://bitbucket.org/multicoreware/cppamp-driver.

  # cd cppamp
  # hg clone https://bitbucket.org/multicoreware/cppamp-driver src

3. Build a build directory and configure using CMake
  # mkdir build
  # cd build
  # cmake ../src

4. Build.
  # make

5. Test. (optional, would be more useful in development cycle)
  # make test

Note that you should also have an access to the repository for C++AMP frontend:
https://bitbucket.org/multicoreware/cppamp. You will be asked id/password while
it is configured.


Repository Structure
-------------------------------------------------------------------------------
The current repository is composed of as following:
1. cppamp-driver
  - This repository. It contains all the C++AMP related implementations, tests,
    scripts and build system.
2. cppamp
  - Modified Clang. It will be downloaded from the build procedure as described
    above. This repository is basically a clone of Clang repository. The
    default branch will be a placeholder for the cloned Clang repository.
    C++AMP specific changes will be made to dev branch, which is the main
    branch for the C++AMP development. Developers can also create their own
    branches for their own development.


Synchronization with Clang repository
-------------------------------------------------------------------------------
C++AMP is an extension to C++11. Since support for C++11 is still very new and
mostly in beta status from Clang, we will need to be periodically synchronized
to the Clang development.

How to synchronize with Clang:
1. Before synchronize from Clang SVN to our local Clang, you need to initialize
   a Clang SVN into a local mecurial repository:

  A. Go to the cppamp directory
    # cd cppamp
  B. Execute init_clangsvn.py
    # ./src/scriptsinit_clangsvn.py

Note that you will need Python SVN bindings to make this work, such as
'python-subversion' package in Ubuntu.

2. Synchronize from Clang SVN to default branch in the local Clang repository.

  A. Go to clang directory
    # cd cppamp/src/compilier/tools/clang
  B. Execute sync_clang.py
    # ../../../scripts/sync_clang.py

3. You will need to run 'hg push' to make the change globally visible.


Test Driven Development
-------------------------------------------------------------------------------
C++AMP development project adapts TDD(Test Driven Development). It helps keep
track of development status by working tests intact. Check out src/tests for
actual implementation.


Tips for developers
-------------------------------------------------------------------------------
1. Debugging Clang

  Clang itself is not debuggable as it is actually 'driver' that calls actual
  compiler. Debugging becomes available once you give '-v' option to the Clang
  execution:

    # clang -v <other arguments>

  It will print out verbose messages that actually it runs. Using this
  information will enable debugging.


