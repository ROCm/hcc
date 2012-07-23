----------------
Build Instruction
-----------------

1. Prepare a directory
# mkdir cppamp

2. Pull a repository from bitbucket: https://bitbucket.org/multicoreware/cppamp-driver.
# cd cppamp
# hg clone https://bitbucket.org/multicoreware/cppamp-driver src

3. Build a build directory and configure using CMake
# mkdir build
# cd build
# cmake ../src

4. Build.
# make



----------------
Before synchronize from Clang SVN to our local Clang:
   initialize a Clang SVN into a local mecurial repository
-----------------
1. Go to the cppamp directory
# cd cppamp
#
2. Execute init_clangsvn.py
# ./src/scriptsinit_clangsvn.py



----------------
Synchronize from Clang SVN to default branch under our local Clang
----------------
1. Go to clang directory
# cd cppamp/compilier/tools/clang

2. Execute sync_clang.py
# ../../../scripts/sync_clang.py
