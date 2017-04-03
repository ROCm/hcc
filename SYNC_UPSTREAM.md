How to synchronize HCC with upstream
====================================
This section shows the step to synchronize HCC with upstream projects, such
as Clang, LLVM, and LLD. The process is currently carried out manually but it
could and should be automated.

Upstream Clang, LLVM, and LLD all sit in different git repositories and they
are almost changed daily. Sometimes a change upstream may affect several
projects, but it is usually not easy to figure it out from the commit log.

HCC has been configured use git submodules to track these dependencies, but
there are users who prefer using repo tool. This document has been revised to
cope with both git submodules and repo.

HCC depends on amd-common LLVM / LLD / Clang, which is a fork maintained by AMD,
and may contain patches yet upstreamed. amd-common LLVM / LLD / Clang is
automatically synchronized with upstream LLVM / LLD / Clang every 4 hours, so
they are very close to the latest codes upstream.

Generally speaking, the process goes like this:

 1. Clone HCC repo manifest
 2. Clone HCC with repo
 3. Initialize git submodules
 4. Merge master ROCm-Device-Libs commits
 5. Merge amd-common LLVM commits
 6. Merge amd-common LLD commits
 7. Merge master COMPILER-RT commits from llvm-mirror
 8. Add git remote for amd-common Clang
 9. Fetch amd-common Clang commits
10. Merge amd-common Clang with HCC Clang
11. Build merged HCC
12. Quick sanity tests on merged HCC
13. Push HCC Clang submodules
14. Push amd-hcc LLVM submodule
15. Push amd-hcc LLD submodule
16. Push amd-hcc COMPILER-RT submodule
17. Push remove-promote-change-addr-space ROCm-Device-Libs submodule
18. Update HCC git submodules configuration
19. Update HCC repo manifest

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
- `git remote set-url --push github git@github.com:RadeonOpenCompute/hcc.git`
- `cd clang`
- `git remote set-url --push github git@github.com:RadeonOpenCompute/hcc-clang-upgrade.git`
- `cd ../compiler`
- `git remote set-url --push github git@github.com:RadeonOpenCompute/llvm.git`
- `cd ../lld`
- `git remote set-url --push github git@github.com:RadeonOpenCompute/lld.git`
- `cd ../compiler-rt`
- `git remote set-url --push github git@github.com:RadeonOpenCompute/compiler-rt.git`
- `cd ../rocdl`
- `git remote set-url --push github git@github.com:RadeonOpenComoute/ROCm-Device-Libs.git`

This only needs to be done once.

Step-by-step Merge Process
--------------------------

### Clone HCC with repo

- `repo init -u https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA.git`
- `repo sync`

There will be an `hcc` directory inside the repo. It would be referred as
*HCC directory* hereafter.

### Clone HCC repo manifest

A manifest directory is created to track working commits for all components.

- `git clone git@github.com:RadeonOpenCompute/HCC-Native-GCN-ISA.git manifest`

This directory would be referred as *HCC manifest directory* hereafter.

### Initialize git submodules

- change to HCC directory
- `git checkout clang_tot_upgrade`
- `git submodule update --init`

### Merge master ROCm-Device-Libs commits

- change to HCC directory
- `cd rocdl`
- `git checkout remove-promote-change-addr-space`
- `git pull`
- `git merge github/master --no-edit`

Resolve any merge conflicts encountered here. Commit to remove-promote-change-addr-space branch.

### Merge amd-common LLVM commits

- change to HCC directory
- `cd compiler`
- `git checkout amd-hcc`
- `git pull`
- `git merge github/amd-common --no-edit`

Resolve any merge conflicts encountered here. Commit to amd-hcc branch.

### Merge amd-common LLD commits

- change to HCC directory
- `cd lld`
- `git checkout amd-hcc`
- `git pull`
- `git merge github/amd-common --no-edit`

Resolve any merge conflicts encountered here. Commit to amd-hcc branch.

### Add git remote for master COMPILER-RT

- change to HCC directory
- `cd compiler-rt`
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

- change to HCC directory
- `cd clang`
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

- change to HCC directory
- `mkdir -p build; cd build`
- `cmake -DCMAKE_BUILD_TYPE=Release ..`
- `make`

Fix any compilation failures if there is any. Repeat this step until HCC
can be built.

### Quick sanity tests on merged HCC
Assume commands below are carried out in HCC build directory. And HCC
directory is at `~/hcc`.

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

- change to HCC directory
- `cd lld`
- `git checkout amd-hcc`
- `git push`

### Push amd-hcc LLVM submodule

- change to HCC directory
- `cd compiler`
- `git checkout amd-hcc`
- `git push`

### Push amd-hcc COMPILER-RT submodule

- change to HCC directory
- `cd compiler-rt`
- `git checkout amd-hcc`
- `git push`

### Push remove-promote-change-addr-space ROCm-Device-Libs submodule

- change to HCC directory
- `cd rocdl`
- `git checkout remove-promote-change-addr-space`
- `git push`

### Update HCC git submodules configuration

- change to HCC directory
- `git checkout clang_tot_upgrade`
- `git add clang compiler lld compiler-rt rocdl`
- `git commit -m "[Config] revise submodule configuration"`, or provide custom
  commit log
- `git push` to push HCC git submodules configuration online

### Update HCC repo manifest

- change to HCC directory
- `repo manifest -r > default.xml`
- copy `default.xml` to HCC manifest directory
- change too HCC manifest directory
- `git add default.xml`
- `git commit -m "[Config] revise manifest"`, or provide custom commit log
- `git push` to push HCC repo manifest online

Upon reaching here, the merge process is completed.
