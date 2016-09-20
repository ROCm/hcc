
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "hc.hpp"
#include "hc_am.hpp"
#include "grid_launch.hpp"

#include <algorithm>
#include <iostream>

struct Foo {
  int x;
  int y;
  double z;
};


// https://bitbucket.org/multicoreware/namd_hip/src/5ff9d6f11eda7cf5dbe97090c0170d250657754a/src/ComputeNonbondedCUDAKernel.h?at=master&fileviewer=file-view-default#ComputeNonbondedCUDAKernel.h-22
struct patch_pair {
  //int patch1_start;
  union {
    bool patch_done[2];
    struct{
      int plist_star;
      int plist_size;
    };
  };
};

const int N = 1024;

__attribute__ ((hc_grid_launch))
void kernel (grid_launch_parm lp, Foo *foo) {
  const int t = amp_get_global_id(0);

  // https://bitbucket.org/multicoreware/namd_hip/src/5ff9d6f11eda7cf5dbe97090c0170d250657754a/src/ComputeNonbondedCUDAKernelBase.h?at=master&fileviewer=file-view-default#ComputeNonbondedCUDAKernelBase.h-182
  tile_static patch_pair sh_pp;

  // https://bitbucket.org/multicoreware/namd_hip/src/5ff9d6f11eda7cf5dbe97090c0170d250657754a/src/ComputeNonbondedCUDAKernelBase.h?at=master&fileviewer=file-view-default#ComputeNonbondedCUDAKernelBase.h-715
  if (t == 0) {
    sh_pp.patch_done[1] = false;
  }

  // https://bitbucket.org/multicoreware/namd_hip/src/5ff9d6f11eda7cf5dbe97090c0170d250657754a/src/ComputeNonbondedCUDAKernelBase.h?at=master&fileviewer=file-view-default#ComputeNonbondedCUDAKernelBase.h-789
  if (sh_pp.patch_done[1]) {
    foo[amp_get_global_id(0)].x = 1;
  }
}

int main () {
  struct Foo foo[N];

  std::for_each(std::begin(foo), std::end(foo), [](struct Foo &F){ F.x = 10; });

  using namespace hc;

  auto acc = hc::accelerator();
  struct Foo* foo_d = (struct Foo*)am_alloc(N*sizeof(Foo), acc, 0);
  am_copy(foo_d, foo, N*sizeof(Foo));


  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim = gl_dim3(N, 1);
  lp.group_dim = gl_dim3(1, 1);

  completion_future cf;
  lp.cf = &cf;

  unsigned int global_counters = 1;
  kernel(lp, foo_d);
  lp.cf->wait();

  return 0;
}

