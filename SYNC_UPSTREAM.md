How to synchronize ToT HCC with upstream
========================================
This section shows the step to synchronize ToT HCC with upstream projects, such
as Clang, LLVM, and LLD. The process is currently carried out manually but it
could and should be automated.

ToT HCC has been configured use git submodules. This document has been revised
to cope with this architectural change.

Upstream Clang, LLVM, and LLD all sit in different git repositories and they
are almost changed daily. Sometimes a change upstream may affect several
projects, but it's usually not easy to figure it out from the commit log.

Generally speaking, the process goes like this:

Process A: merge upstream Clang

1. Add git remote for upstream Clang
2. Fetch upstream Clang commits
3. Merge upstream Clang with ToT HCC Clang
4. Build merged ToT HCC Clang
   - For failures introduced by changes in LLVM / LLD, execute Process B
5. Quick sanity tests on merged ToT HCC Clang
6. Push ToT HCC submodule
7. 

Process B: merge upstream LLVM / LLD

1. Fetch upstream LLVM commits
2. Fetch upstream LLD commits 
3. Build upstream LLVM / LLD
4. Update LLVM / LLD submodules in ToT HCC
   - Resume step 4 of Process A

Detailed step-by-step instructions are in following sections.

Process A: merge upstream Clang
-------------------------------

### Add git remote for upstream Clang
It's assumed there's already a ToT HCC checkout, and also a ToT HCC Clang
submodule, normally found under `clang` diretory under ToT HCC checkout.

- Enter `clang`
- `git remote -v` to check if there's a git remote pointing to:
  `git@github.com:llvm-mirror/clang.git`
- If there's not, add it by:
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
Assume a ToT HCC build directory is there. If there's not, follow Appendix B to configure one.

- change to ToT HCC build directory
- `make -j40` , recommended job number is the number of your logical processor times 2.

Fix any compilation failures if there's any. For failures introduced by changes
in LLVM / LLD. Stop. Execute Process B and come back here once it's done.
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
- If only Process A is executed:
  `git add clang`
  if Process B is also executed, then
  `git add clang compiler lld`
- `git commit` and provide commit log
- `git push` to push submodules configuration online

Upon reaching here, the merge process is completed.


Process B: merge upstream LLVM / LLD
------------------------------------
Sometimes it's not possible to synchronize with upstream Clang without also
synchronizing with upstream LLVM / LLD. This section explains steps to do that.

### Fetch upstream LLVM commits
Assume there is already an upstream LLVM checkout for AMDGPU(Lightning) backend
outside ToT HCC checkout.

- change to upstream LLVM directory
- `git pull`

### Fetch upstream LLD commits
Assume there is already an upstream LLD checkout for AMDGPU(Lightning) backend. Normally it would be in "tools/lld" in LLVM checkout.

- change to upstream LLD directory
- `git pull`

### Build upstream LLVM / LLD
Assume there's a build directory for upstream LLVM / LLD. If there's not, follow Appendix A to configure one.

- change to LLVM build directory
- `make -j40` , recommended job number is the number of your logical processor times 2.

### Update LLVM / LLD submodule configuration

- change to upstream LLVM directory
- `git rev-parse HEAD`, log the commit #
- change to ToT HCC directory
- `cd compiler`
- `git checkout master`
- `git pull`
- `git reset --hard <commit # of upstream LLVM>`

- change to upstream LLD directory
- `git rev-parse HEAD`, log the commit #
- change to ToT HCC directory
- `cd lld`
- `git checkout master`
- `git pull`
- `git reset --hard <commit # of upstream LLD>`


Appendix A: CMake command for upstream LLVM / LLD
=================================================

```
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
    <upstream LLVM checkout directory>
```

Appendix B: CMake command for ToT HCC
=====================================

```
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DHSA_LLVM_BIN_DIR=<upstream LLVM build directory>/bin \
    -DHSA_AMDGPU_GPU_TARGET=AMD:AMDGPU:8:0:3 \
    -DHSA_USE_AMDGPU_BACKEND=ON \
    <ToT HCC checkout directory>
```
