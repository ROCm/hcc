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

 1. Merge amd-common LLVM commits
 2. Merge amd-common LLD commits
 3. Add git remote for upstream Clang
 4. Fetch upstream Clang commits
 5. Merge upstream Clang with ToT HCC Clang
 6. Build merged ToT HCC
 7. Quick sanity tests on merged ToT HCC
 8. Push ToT HCC Clang submodules
 9. Push amd-hcc LLVM submodule
10. Update submodules configuration

Detailed step-by-step instructions are in following sections.

Useful github repositories
------------------------------------
git locations of repositories used in the merge process are:
- amd-common LLVM
  - URL : git@github.com:RadeonOpenCompute/llvm.git
  - branch : amd-common
- amd-common LLD
  - URL: git@github.com:RadeonOpenCompute/lld.git
  - branch : amd-common
- upstram Clang
  - URL: git@github.com:llvm-mirror/clang.git
  - branch : master

Step-by-step Merge Process
------------------------------------
### Merge amd-common LLVM commits

- change to ToT HCC directory
- `cd compiler`
- `git checkout amd-hcc`
- `git pull`
- `git merge origin/amd-common`

Resolve any merge conflicts encountered here. Commit to amd-hcc branch.

### Merge amd-common LLD commits

- `cd ../lld`
- `git checkout amd-common`
- `git pull`

### Add git remote for upstream Clang

- `cd ../clang`
- `git remote -v` to check if there is a git remote pointing to:
  `git@github.com:llvm-mirror/clang.git`
- If there is not, add it by:
  `git remote add clang git@github.com:llvm-mirror/clang.git`

### Fetch upstream Clang commits

- `git checkout upstream`
  - change to the branch to keep upstream commits.
  - The branch contains no HCC-specific codes.
- `git fetch clang`
- `git merge --no-ff clang/master`

### Merge upstream Clang with ToT HCC Clang

- `git checkout clang_tot_upgrade`
  - change to the main develop branch for ToT HCC Clang
- `git merge upstream`

Resolve merge conflicts encountered here. Commit to clang_tot_upgrade branch.

### Build merged ToT HCC
Assume a ToT HCC build directory is there. If there is not, follow Appendix A
to configure one.

- change to ToT HCC build directory
- re-run CMake according to Appendix A
- `make -j56` , recommended number is your logical processor times 2.

Fix any compilation failures if there is any. Repeat this step until ToT HCC
can be built.

### Quick sanity tests on merged ToT HCC
Assume commands below are carried out in ToT HCC build directory. And ToT HCC
checkout is at `~/hcc_upstream`.

Test with one C++AMP FP math unit test.
```
bin/hcc `bin/clamp-config --build --cxxflags --ldflags` -lm \
  ~/hcc/hcc_upstream/tests/Unit/AmpMath/amp_math_cos.cpp
./a.out ; echo $?
```

Test with one grid_launch unit test with AM library usage.
```
bin/hcc `bin/hcc-config --build --cxxflags --ldflags` -lhc_am \
  ~/hcc/hcc_upstream/tests/Unit/GridLaunch/glp_const.cpp
./a.out ; echo $?
```

### Push ToT HCC Clang submodule

- change to ToT HCC directory
- `cd clang`
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

### Push amd-hcc LLVM submodule

- `cd ../compiler`
- `git checkout amd-hcc`
- `git push`

### Update submodules configuration

- `cd ..`
- `git add clang compiler lld`
- `git commit` and provide commit log
- `git push` to push submodules configuration online

Upon reaching here, the merge process is completed.

Appendix A: CMake command for ToT HCC
=====================================

```
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_AMDGPU_GPU_TARGET=AMD:AMDGPU:8:0:3 \
    -DROCM_DEVICE_LIB_DIR=<build directory of ROCm-Device-Libs>/dist/lib \
    <ToT HCC checkout directory>
```
