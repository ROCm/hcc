How to synchronize HCC with upstream
====================================
This section shows the step to synchronize HCC with upstream projects, such
as Clang, LLVM, and LLD. The process is currently carried out manually but it
could and should be automated.

HCC has been configured use git submodules. This document has been revised
to cope with this architectural change.

Upstream Clang, LLVM, and LLD all sit in different git repositories and they
are almost changed daily. Sometimes a change upstream may affect several
projects, but it is usually not easy to figure it out from the commit log.

HCC depends on amd-common LLVM / LLD / Clang, which is a fork maintained by AMD,
and may contain patches yet upstreamed. amd-common LLVM / LLD / Clang is automatically
synchronized with upstream LLVM / LLD / Clang every 4 hours, so they are very close to
the latest codes upstream.

Generally speaking, the process goes like this:

 1. Merge master ROCm-Device-Libs commits
 2. Merge amd-common LLVM commits
 3. Merge amd-common LLD commits
 4. Merge master COMPILER-RT commits from llvm-mirror
 5. Add git remote for amd-common Clang
 6. Fetch amd-common Clang commits
 7. Merge amd-common Clang with HCC Clang
 8. Build merged HCC
 9. Quick sanity tests on merged HCC
10. Push HCC Clang submodules
11. Push amd-hcc LLVM submodule
12. Push amd-hcc LLD submodule
13. Push amd-hcc COMPILER-RT submodule
14. Push remove-promote-change-addr-space ROCm-Device-Libs submodule
14. Update submodules configuration

Detailed step-by-step instructions are in following sections.

Useful github repositories
--------------------------
git locations of repositories used in the merge process are:
- master ROCm-Device-Libs
  - URL: git@github.com:RadeonOpenCompute/ROCm-Device-Libs.git
  - branch : master
- amd-common LLVM
  - URL : git@github.com:RadeonOpenCompute/llvm.git
  - branch : amd-common
- amd-common LLD
  - URL: git@github.com:RadeonOpenCompute/lld.git
  - branch : amd-common
- amd-common COMPILER-RT
  - URL: git@github.com:llvm-mirror/compiler-rt.git
  - branch : master
- amd-common Clang
  - URL: git@github.com:RadeonOpenCompute/clang.git
  - branch : amd-common

Set SSH URL for git push
------------------------
HCC has been configured to use HTTPS URL by default. It is easy for users
to clone it anonymously. But it would be hard for committing changes. Use the
following commands to setup SSH URL.

- change to HCC directory
- `git remote set-url --push origin git@github.com:RadeonOpenCompute/hcc.git`
- `cd clang`
- `git remote set-url --push origin git@github.com:RadeonOpenCompute/hcc-clang-upgrade.git`
- `cd ../compiler`
- `git remote set-url --push origin git@github.com:RadeonOpenCompute/llvm.git`
- `cd ../lld`
- `git remote set-url --push origin git@github.com:RadeonOpenCompute/lld.git`
- `cd ../compiler-rt`
- `git remote set-url --push origin git@github.com:RadeonOpenCompute/compiler-rt.git`
- `cd ../rocdl`
- `git remote set-url --push origin git@github.com:RadeonOpenComoute/ROCm-Device-Libs.git`

This only needs to be done once.

Step-by-step Merge Process
--------------------------
### Merge master ROCm-Device-Libs commits

- `cd ../rocdl`
- `git checkout remove-promote-change-addr-space`
- `git pull`
- `git merge origin/master --no-edit`

Resolve any merge conflicts encountered here. Commit to remove-promote-change-addr-space branch.

### Merge amd-common LLVM commits

- change to HCC directory
- `cd compiler`
- `git checkout amd-hcc`
- `git pull`
- `git merge origin/amd-common --no-edit`

Resolve any merge conflicts encountered here. Commit to amd-hcc branch.

### Merge amd-common LLD commits

- `cd ../lld`
- `git checkout amd-hcc`
- `git pull`
- `git merge origin/amd-common --no-edit`

Resolve any merge conflicts encountered here. Commit to amd-hcc branch.

### Add git remote for master COMPILER-RT

- `cd ../compiler-rt`
- `git remote -v` to check if there is a git remote pointing to:
  `git@github.com:llvm-mirror/compiler-rt.git`
- If there is not, add it by:
  `git remote add compiler-rt https://github.com/llvm-mirror/compiler-rt`

### Merge amd-common COMPILER-RT commits

- `git checkout amd-hcc`
- `git fetch compiler-rt`
- `git merge --no-ff compiler-rt/master --no-edit`

Resolve any merge conflicts encountered here. Commit to amd-hcc branch.

### Add git remote for amd-common Clang

- `cd ../clang`
- `git remote -v` to check if there is a git remote pointing to:
  `git@github.com:RadeonOpenCompute/clang.git`
- If there is not, add it by:
  `git remote add clang https://github.com/RadeonOpenCompute/clang`

### Fetch amd-common Clang commits

- `git checkout clang_tot_upgrade`
- `git fetch clang`
- `git merge --no-ff clang/amd-common --no-edit`

Resolve merge conflicts encountered here. Commit to clang_tot_upgrade branch.

### Build merged HCC
Assume a HCC build directory is there. If there is not, follow Appendix A
to configure one.

- change to HCC build directory
- re-run CMake according to Appendix A
- `make`

Fix any compilation failures if there is any. Repeat this step until HCC
can be built.

### Quick sanity tests on merged HCC
Assume commands below are carried out in HCC build directory. And HCC
checkout is at `~/hcc`.

Test with one C++AMP FP math unit test.
```bash
bin/hcc `bin/clamp-config --build --cxxflags --ldflags` -lm \
  ~/hcc/tests/Unit/AmpMath/amp_math_cos.cpp
./a.out ; echo $?
```

Test with one grid_launch unit test with AM library usage.
```bash
bin/hcc `bin/hcc-config --build --cxxflags --ldflags` -lhc_am \
  ~/hcc/tests/Unit/GridLaunch/glp_const.cpp
./a.out ; echo $?
```

Test with one HC unit test with atomic function and 64-bit arithmetic.
```bash
bin/hcc `bin/hcc-config --build --cxxflags --ldflags` \
  ~/hcc/tests/Unit/HC/hc_atomic_add_global.cpp
./a.out ; echo $?
```

### Push HCC Clang submodule

- change to HCC directory
- `cd clang`
- `git checkout clang_tot_upgrade`
- `git push`

Following steps are to ensure "master" branch are kept the same
as "clang_tot_upgrade" branch.
- `git checkout master`
- `git merge clang_tot_upgrade --no-edit`
- `git push`

Finally switch back to "clang_tot_upgrade" branch.
- `git checkout clang_tot_upgrade`

### Push amd-hcc LLD submodule

- `cd ../lld`
- `git checkout amd-hcc`
- `git push`

### Push amd-hcc LLVM submodule

- `cd ../compiler`
- `git checkout amd-hcc`
- `git push`

### Push amd-hcc COMPILER-RT submodule

- `cd ../compiler-rt`
- `git checkout amd-hcc`
- `git push`

### Push remove-promote-change-addr-space ROCm-Device-Libs submodule

- `cd ../rocdl`
- `git checkout remove-promote-change-addr-space`
- `git push`

### Update submodules configuration

- `cd ..`
- `git add clang compiler lld compiler-rt rocdl`
- `git commit -m "[Config] revise submodule configuration"`, or provide custom
  commit log
- `git push` to push submodules configuration online

Upon reaching here, the merge process is completed.

Appendix A: CMake command for HCC
=================================

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
    <HCC checkout directory>
```
