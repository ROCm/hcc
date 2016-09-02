How to synchronize ToT HCC with upstream
========================================
This section shows the step to synchronize ToT HCC with upstream projects, such
as Clang, LLVM, and LLD. The process is currently carried out manually but it
could and should be automated.

ToT HCC has been configured use git submodules. This document has been revised
to cope with this architectural change.

Upstream Clang, LLVM, and LLD all sit in different git repositories and they
are almost changed daily. Sometimes a change upstream may affect several
projects, but it is usually not easy to figure it out from the commit log.

ToT HCC depends on amd-common LLVM / LLD, which is a fork maintained by AMD,
and may contain patches yet upstreamed. amd-common LLVM / LLD is automatically
synchronized with upstream LLVM / LLD every 4 hours, so they are very close to
the latest codes upstream.

Generally speaking, the process goes like this:

Process A: merge amd-common LLVM / LLD

1. Fetch amd-common LLVM commits
2. Fetch amd-common LLD commits
3. Build amd-common LLVM / LLD
4. Update LLVM / LLD submodules in ToT HCC

Process B: merge upstream Clang

1. Add git remote for upstream Clang
2. Fetch upstream Clang commits
3. Merge upstream Clang with ToT HCC Clang
4. Build merged ToT HCC Clang
   - For failures introduced by changes in LLVM / LLD, re-execute Process A
5. Quick sanity tests on merged ToT HCC Clang
6. Push ToT HCC submodules

Detailed step-by-step instructions are in following sections.

Process A: merge upstream LLVM / LLD
------------------------------------
git locations of amd-common LLVM / LLD are:
- amd-common LLVM
  - URL : git@github.com:RadeonOpenCompute/llvm.git
  - branch : amd-common
- amd-common LLD
  - URL: git@github.com:RadeonOpenCompute/lld.git
  - branch : amd-common

### Fetch amd-common LLVM commits
Assume there is already an amd-common LLVM checkout outside ToT HCC checkout.

- change to amd-common LLVM directory
- `git pull`

### Fetch amd-common LLD commits
Assume there is already an amd-common LLD checkout. Normally it would be in
"tools/lld" in amd-common LLVM checkout.

- change to amd-common LLD directory
- `git pull`

### Build amd-common LLVM / LLD
Assume there is a build directory for amd-common LLVM / LLD. If there is not,
follow Appendix A to configure one.

- change to amd-common LLVM build directory
- `make -j40` , recommended number is your logical processor times 2.

### Update LLVM / LLD submodule configuration

- change to amd-common LLVM directory
- `git rev-parse HEAD`, log the commit #
- change to ToT HCC directory
- `cd compiler`
- `git checkout amd-common`
- In case you built ToT HCC before, remove patches from ToT HCC by:
  - `git checkout -- .`
  - `rm lib/Analysis/TileUniform lib/Transforms/CpuRename lib/Transforms/EraseNonkernel lib/Transforms/HC lib/Transforms/Promote lib/Transforms/RemoveSpecialSection`
- `git pull`
- `git reset --hard <commit # of amd-common LLVM>`

- change to amd-common LLD directory
- `git rev-parse HEAD`, log the commit #
- change to ToT HCC directory
- `cd lld`
- `git checkout amd-common`
- `git pull`
- `git reset --hard <commit # of amd-common LLD>`

Process B: merge upstream Clang
-------------------------------

### Add git remote for upstream Clang
It is assumed there is already a ToT HCC checkout, and also a ToT HCC Clang
submodule, normally found under `clang` diretory under ToT HCC checkout.

- Enter `clang`
- `git remote -v` to check if there is a git remote pointing to:
  `git@github.com:llvm-mirror/clang.git`
- If there is not, add it by:
  `git remote add clang git@github.com:llvm-mirror/clang.git`

### Fetch upstream Clang commits
Assume commands below are carried out in `clang`.

- `git checkout upstream` : change to the branch to keep upstream commits. The branch contains no HCC-specific codes.
- `git fetch clang` : fetch upstream commits
- `git merge --no-ff clang/master` : get upstream commits merged into upstream branch

### Merge upstream Clang with ToT HCC Clang
Assume commands below are carried out in `clang`.

- `git checkout clang_tot_upgrade` : change to the main develop branch for ToT HCC Clang
- `git merge upstream` : merge commits from upstream Clang to ToT HCC Clang

Resolve any merge conflicts encountered here. Commit to clang_tot_upgrade branch.

### Build merged ToT HCC Clang
Assume a ToT HCC build directory is there. If there is not, follow Appendix B to configure one.

- change to ToT HCC build directory
- re-run CMake according to Appendix B
- `make -j40` , recommended job number is the number of your logical processor times 2.

Fix any compilation failures if there is any. For failures introduced by changes
in LLVM / LLD. Stop. Execute Process A and come back here once it is done.
Then repeat this step until ToT HCC can be built.

### Quick sanity tests on merged ToT HCC Clang
Assume commands below are carried out in ToT HCC build directory. And ToT HCC
checkout is at `~/hcc_upstream`.

Test with one C++AMP FP math unit test.
```
bin/hcc `bin/clamp-config --build --cxxflags --ldflags` -lm ~/hcc_upstream/tests/Unit/AmpMath/amp_math_cos.cpp
./a.out ; echo $?
```

Test with one grid_launch unit test with AM library usage.
```
bin/hcc `bin/hcc-config --build --cxxflags --ldflags` -lhc_am ~/hcc_upstream/tests/Unit/GridLaunch/glp_const.cpp
./a.out ; echo $?
```

### Commit and push ToT HCC Clang submodule

- change to ToT HCC Clang directory
- `git checkout clang_tot_upgrade`
- `git push`
- `git checkout upstream`
- `git push`

Following steps are to ensure "develop" and "master" branch are kept the same
as "clang_tot_upgrade" branch.
- `git checkout develop`
- `git merge clang_tot_upgrade`
- `git push`
- `git checkout master`
- `git merge clang_tot_upgrade`
- `git push`

Finally switch back to "clang_tot_upgrade" branch.
- `git checkout clang_tot_upgrade`

### Update submodules configuration

- change to ToT HCC Clang directory
- `git add clang compiler lld`
- `git commit` and provide commit log
- `git push` to push submodules configuration online

Upon reaching here, the merge process is completed.

Appendix A: CMake command for amd-common LLVM / LLD
===================================================

```
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
    <amd-common LLVM checkout directory>
```

Appendix B: CMake command for ToT HCC
=====================================

```
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_LLVM_BIN_DIR=<amd-common LLVM build directory>/bin \
    -DHSA_AMDGPU_GPU_TARGET=AMD:AMDGPU:8:0:3 \
    -DROCM_DEVICE_LIB_DIR=<build directory of ROCm-Device-Libs>/dist/lib \
    <ToT HCC checkout directory>
```
