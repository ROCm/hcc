How to synchronize HCC with upstream
====================================
This section shows the step to synchronize HCC with upstream projects, such
as Clang, LLVM, and LLD. The process is currently carried out manually but it
could and should be automated.

Upstream Clang, LLVM, and LLD all sit in different git repositories and they
are almost changed daily. Sometimes a change upstream may affect several
projects, but it is usually not easy to figure it out from the commit log.

HCC has been configured to use git submodules to track these dependencies, but
there are users who prefer using repo tool. This document has been revised to
cope with both git submodules and repo. Maintaining repo manifest is an
optional step for now.

HCC depends on amd-common LLVM / LLD / Clang, which is a fork maintained by AMD,
and may contain patches not yet upstreamed. amd-common LLVM / LLD / Clang is
automatically synchronized with upstream LLVM / LLD / Clang every 4 hours, so
they are very close to the latest codes upstream.

There are two roles in the HCC merge process. The one who conducts the upstream
sync process through getting all the amd-common code, merging it into amd-hcc, 
and filing the pull requests. And the one who reviews all of your sync pull
requests. The HCC merge maintainer will take on the first role, and is required
to interact, modify, and satisfy the changes requested by the reviewer before
the pull requests can be accepted.

Detailed step-by-step instructions are in following sections.

Useful github remote repositories (For reference)
-------------------------------------------------
git locations of repositories used in the merge process are:
- master ROCm-Device-Libs
  - URL: https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git
  - branch : master
- amd-common LLVM
  - URL : https://github.com/RadeonOpenCompute/llvm.git
  - branch : amd-common
- amd-common LLD
  - URL: https://github.com/RadeonOpenCompute/lld.git
  - branch : amd-common
- amd-common Clang
  - URL: https://github.com/RadeonOpenCompute/clang.git
  - branch : amd-common
- clang_tot_upgrade HCC Clang
  - URL: https://github.com/RadeonOpenCompute/hcc-clang-upgrade.git
  - branch : clang_tot_upgrade
- master COMPILER-RT
  - URL: https://github.com/llvm-mirror/compiler-rt.git
  - branch : master
- master CLANG-TOOLS-EXTRA
  - URL: https://github.com/llvm-mirror/clang-tools-extra.git
  - branch : master

Set own forks for HCC modules
-----------------------------
Go to these links and fork onto your personal account. If you have old forks,
please remove them, or create another fork which tracks the tip of
corresponding repositories.

1) https://github.com/RadeonOpenCompute/hcc
2) https://github.com/RadeonOpenCompute/hcc-clang-upgrade
3) https://github.com/RadeonOpenCompute/llvm
4) https://github.com/RadeonOpenCompute/lld
5) https://github.com/llvm-mirror/compiler-rt
6) https://github.com/llvm-mirror/clang-tools-extra

Note, you'd won't need to do this for rocdl, since we will use its master branch.

Step-by-step Merge Process
--------------------------
Assuming that we have created forks for all the repositories above, follow this:

### From here on, replace *your_name* with your own github id

- `git clone git@github.com:your_name/hcc.git hcc_fork`
- `cd hcc_fork`
- `git checkout clang_tot_upgrade`
- `git submodule update --init`

This directory would be referred as *HCC directory* hereafter.

### Add all remotes and forks for submodules

- change to HCC directory
- `cd lld`
- `git remote add lld_fork git@github.com:your_name/lld.git`
- `cd ../compiler`
- `git remote add llvm_fork git@github.com:your_name/llvm.git`
- `cd ../compiler-rt`
- `git remote add compiler-rt_fork git@github.com:your_name/compiler-rt.git`
- `git remote add compiler-rt https://github.com/llvm-mirror/compiler-rt`
- `cd ../clang-tools-extra`
- `git remote add clang-tools-extra_fork git@github.com:your_name/clang-tools-extra.git`
- `git remote add clang-tools-extra https://github.com/llvm-mirror/clang-tools-extra`
- `cd ../clang`
- `git remote add clang_fork git@github.com:your_name/hcc-clang-upgrade.git`
- `git remote add clang https://github.com/RadeonOpenCompute/clang`

**Use `git remote -v` to verify the new remotes are added.**

### Checkout latest master ROCm-Device-Libs

- change to HCC directory
- `cd rocdl`
- `git fetch --all`
- `git checkout origin/master`

### Merge amd-common LLD commits
### From here, replace YYYYMMDD with your date.

- change to HCC directory
- `cd lld`
- `git fetch --all`
- `git checkout -b merge_YYYYMMDD origin/amd-hcc`
- `git merge origin/amd-common --no-edit`
- Resolve any merge conflicts encountered here

### Merge amd-common LLVM commits

- change to HCC directory
- `cd compiler`
- `git fetch --all`
- `git checkout -b merge_YYYYMMDD origin/amd-hcc`
- `git merge origin/amd-common --no-edit`
- Resolve any merge conflicts encountered here

### Merge amd-common COMPILER-RT commits from llvm-mirror

- change to HCC directory
- `cd compiler-rt`
- `git fetch --all`
- `git checkout -b merge_YYYYMMDD origin/amd-hcc`
- `git merge --no-ff compiler-rt/master --no-edit`
- Resolve any merge conflicts encountered here

### Merge amd-common CLANG-TOOLS-EXTRA commits from llvm-mirror

- change to HCC directory
- `cd clang-tools-extra`
- `git fetch --all`
- `git checkout -b merge_YYYYMMDD origin/amd-hcc`
- `git merge clang-tools-extra/master --no-edit`
- Resolve any merge conflicts encountered here

### Merge amd-common Clang commits

- change to HCC directory
- `cd clang`
- `git fetch --all`
- `git checkout -b merge_YYYYMMDD origin/clang_tot_upgrade`
- `git merge --no-ff clang/amd-common --no-edit`
- Resolve any merge conflicts encountered here

### Build merged HCC

- change to HCC directory
- `cd ..; mkdir -p build; cd build`
- `cmake -DCMAKE_BUILD_TYPE=Release ../hcc_fork`
- `make -j$(nproc)`

Fix any compilation failures if there is any. Repeat this step until HCC
can be built.

### Quick sanity tests on merged HCC
Assume commands below are carried out in HCC build directory. And HCC
directory is at `~/hcc_fork`.

Test with one C++AMP FP math unit test.
```bash
compiler/bin/hcc `bin/clamp-config --build --cxxflags --ldflags` -lm \
  ~/hcc_fork/tests/Unit/AmpMath/amp_math_cos.cpp
./a.out ; echo $?
```

Test with one grid_launch unit test with AM library usage.
```bash
compiler/bin/hcc `bin/hcc-config --build --cxxflags --ldflags` -lhc_am \
  ~/hcc_fork/tests/Unit/GridLaunch/glp_const.cpp
./a.out ; echo $?
```

Test with one HC unit test with atomic function and 64-bit arithmetic.
```bash
compiler/bin/hcc `bin/hcc-config --build --cxxflags --ldflags` \
  ~/hcc_fork/tests/Unit/HC/hc_atomic_add_global.cpp
./a.out ; echo $?
```

### Full sanity tests on merged HCC
Test all unit tests.
- change to HCC build directory
- `make test`

Resolve any issues encountered here.

### Push and Create Pull Request for HCC Clang submodule

- change to HCC directory
- `cd clang`
- `git push clang_fork merge_YYYYMMDD:merge_YYYYMMDD`
Create a pull request to merge `merge_YYYYMMDD` from your clang fork
into https://github.com/RadeonOpenCompute/hcc-clang-upgrade
on branch `clang_tot_upgrade`.

### Push and Create Pull Request for amd-hcc LLD submodule

- change to HCC directory
- `cd lld`
- `git push lld_fork merge_YYYYMMDD:merge_YYYYMMDD`
Create a pull request to merge `merge_YYYYMMDD` from your lld fork
into https://github.com/RadeonOpenCompute/lld.git
on branch `amd-hcc`.

### Push and Create Pull Request for amd-hcc LLVM submodule

- change to HCC directory
- `cd compiler`
- `git push llvm_fork merge_YYYYMMDD:merge_YYYYMMDD`
Create a pull request to merge `merge_YYYYMMDD` from your llvm fork
into https://github.com/RadeonOpenCompute/llvm.git
on branch `amd-hcc`.

### Push and Create Pull Request for amd-hcc COMPILER-RT submodule

- change to HCC directory
- `cd compiler-rt`
- `git push compiler-rt_fork merge_YYYYMMDD:merge_YYYYMMDD`
Create a pull request to merge `merge_YYYYMMDD` from your compiler-rt fork
into https://github.com/RadeonOpenCompute/compiler-rt
on branch `amd-hcc`.

### Push and Create Pull Request for amd-hcc CLANG-TOOLS-EXTRA submodule

- change to HCC directory
- `cd clang-tools-extra`
- `git push clang-tools-extra_fork merge_YYYYMMDD:merge_YYYYMMDD`
Create a pull request to merge `merge_YYYYMMDD` from your clang-tools-extra fork
into https://github.com/RadeonOpenCompute/clang-tools-extra
on branch `amd-hcc`.

*** Wait until all Pull Requests are approved and merged. ***
On github when making a pull request, the repository you are trying to update
will have a list of reviewers who have the authority to approve pull requests
and merge them into the repositories. During the review period, you are
required to interact with the reviewers.

For the pull requests to be approved, you must wait for a reviewer to accept
the changes you are making. The reviewer will request changes if the code is
not ready for merge. It is your responsibility to interact and resolve all
conflicts so that the PR is ready for merge.

### Update HCC git submodules configuration

- change to HCC directory
- `git checkout origin/clang_tot_upgrade`
- `cd clang`
- `git fetch --all`
- `git checkout origin/clang_tot_upgrade`
- `cd ../compiler`
- `git fetch --all`
- `git checkout origin/amd-hcc`
- `cd ../compiler-rt`
- `git fetch --all`
- `git checkout origin/amd-hcc`
- `cd ../clang-tools-extra`
- `git fetch --all`
- `git checkout origin/amd-hcc`
- `cd ../lld`
- `git fetch --all`
- `git checkout origin/amd-hcc`
- `cd ../rocdl`
- `git fetch --all`
- `git checkout origin/master`

- change to HCC directory
- `git checkout -b merge_YYYYMMDD clang_tot_upgrade`
- `git add clang clang-tools-extra compiler compiler-rt lld rocdl`
- `git commit -m "[Config] revise submodule configuration"`, or provide custom
  commit log
- `git push origin merge_YYYYMMDD:merge_YYYYMMDD`
Create a pull request to merge `merge_YYYYMMDD` from your hcc fork
into https://github.com/RadeonOpenCompute/hcc on branch `clang_tot_upgrade`.

Upon reaching here, the merge process is completed.


### (Optional) Update HCC repo manifest
### Clone HCC with repo

- Create a directory called 'repo'. It would be called as *HCC Repo directory*
  hereafter.
- `repo init -u https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA.git`
- `repo sync`

### Clone HCC repo manifest

A manifest directory is created to track working commits for all components.

- `git clone https://github.com/RadeonOpenCompute/HCC-Native-GCN-ISA.git manifest`

This directory would be referred as *HCC manifest directory* hereafter.

### Initialize git submodules

- change to HCC Repo directory
- `cd hcc`
- `git checkout clang_tot_upgrade`
- `git pull`
- `git submodule update --init`

### Update HCC repo manifest
- `repo manifest -r > default.xml`
- copy `default.xml` to HCC manifest directory
- change too HCC manifest directory
- `git add default.xml`
- `git commit -m "[Config] revise manifest"`, or provide custom commit log
- `git push` to push HCC repo manifest online

