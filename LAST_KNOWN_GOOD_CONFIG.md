Overview
========
This document shows the last known good configuration for the following
projects:
- ToT HCC
- ToT HCC Clang
- upstream Clang (where ToT HCC Clang is synchronized with)
- upstream LLVM
- upstream LLD

ToT HCC
-------
- clone: git@github.com:RadeonOpenCompute/hcc.git
- branch: clang_tot_upgrade
- commit: 8b31719a2afef6855c1bd67087453cfe79f00dee

ToT HCC Clang
-------------
- clone: git@github.com:RadeonOpenCompute/hcc-clang-upgrade.git
- branch: clang_tot_upgrade
- commit: de547dc3f213aee7274557227aa1e370bc6b5d1f

upstream Clang
--------------
- clone: git@github.com:RadeonOpenCompute/hcc-clang-upgrade.git
- branch: upstream
- commit: 3c55f1d0231063428f4c11082d5337a55cd2ed38

upstream LLVM
-------------
- clone: https://github.com/llvm-mirror/llvm
- branch: master
- commit: 2970c2210ea61f80f9ff77ccfa1039508b813d64

upstream LLD
------------
- clone: https://github.com/llvm-mirror/lld 
- branch: master
- commit: e49e72b3bf9afb9c29ee27bbd3b3c0c040828118

How to synchronize ToT HCC with upstream
========================================
This section shows the step to synchronize ToT HCC with upstream projects, such
as Clang, LLVM, and LLD. The process is currently carried out manually but it
could and should be automated.

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
6. Update LAST_KNOWN_GOOD_CONFIG.txt (this document)
7. Push everything

Process B: merge upstream LLVM / LLD

1. Fetch upstream LLVM commits
2. Fetch upstream LLD commits
3. Build upstream LLVM / LLD
4. Remove ToT HCC checkout and restart Process A

Detailed step-by-step instructions are in following sections.

Process A: merge upstream Clang
-------------------------------

### Add git remote for upstream Clang
It's assumed there's already a ToT HCC checkout, and also a ToT HCC Clang
checkout, normally found under `compiler/tools/clang` diretory under ToT HCC
checkout.

- Enter `compiler/tools/clang`
- `git remote -v` to check if there's a git remote pointing to:
  `git@github.com:llvm-mirror/clang.git`
- If there's not, add it by:
  `git add remote clang git@github.com:llvm-mirror/clang.git`

### Fetch upstream Clang commits
Assume commands below are carried out in `compiler/tools/clang`.

- `git checkout upstream` : change to the branch to keep upstream commits. The branch contains no HCC-specific codes.
- `git fetch clang` : fetch upstream commits
- `git merge --no-ff clang/master` : get upstream commits merged into upstream branch

### Merge upstream Clang with ToT HCC Clang
Assume commands below are carried out in `compiler/tools/clang`.

- `git checkout clang_tot_upgrade` : change to the main develop branch for ToT HCC Clang
- `git merge upstream` : merge commits from upstream Clang to ToT HCC Clang

Resolve any merge conflicts encountered here.

### Build merged ToT HCC Clang
Assume a ToT HCC build directory is there.

- change to ToT HCC build directory
- `make -j16`

Fix any compilation failures if there's any. For failures introduced by changes
in LLVM / LLD. Stop. Execute Process B and restart Process A.

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

### Update LAST_KNOWN_GOOD_CONFIG.txt (this document)

- change to ToT HCC Clang directory
- `git checkout upstream`
- `git rev-parse HEAD` : log the result in "upstream Clang" in the beginning of this document
- `git checkout clang_tot_upgrade`
- `git rev-parse HEAD` : log the result in "ToT HCC Clang" in the beginning of this document
- change to ToT HCC directory
- `git rev-parse HEAD` : log the result in "ToT HCC" in the beginning of this document

### Push everything

- change to ToT HCC directory
- `git add LAST_KNOWN_GOOD_CONFIG.txt`
- `git commit`
- `git push`
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

Upon reaching here, the merge process is completed.


Process B: merge upstream LLVM / LLD
------------------------------------

### Fetch upstream LLVM commits

### Fetch upstream LLD commits

### Build upstream LLVM / LLD

### Remove ToT HCC checkout and restart Process A

