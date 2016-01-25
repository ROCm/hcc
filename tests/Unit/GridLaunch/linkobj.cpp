// XFAIL: *
// RUN: %hc -lhc_am %s -o %t.out && %t.out

// FIXME: GridLaunch tests would hang HSA dGPU if executed in multi-thread
// environment. Need further invetigation

#error FIXME

#include "grid_launch.h"
#include "hc_am.hpp"

__attribute__((hc_grid_launch)) void foo(grid_launch_parm lp, int* a)
{
  int x = lp.threadId.x + lp.groupDim.x*lp.groupId.x;
  a[x] = x;
}

int main()
{
  int size = 1000;

  int* a = (int*)malloc(sizeof(int)*size);

  int* a_d = (int*)hc::am_alloc(size*sizeof(int), hc::accelerator(), 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.groupDim = gl_dim3(size);

  hc::completion_future cf;
  lp.cf = &cf;
  foo(lp, a_d);
  lp.cf->wait();

  hc::am_copy(a, a_d, size*sizeof(int));

  int ret = 0;
  for(int i = 0; i < size; ++i)
  {
    if(a[i] != i)
    {
      ret = 1;
      if(i < 64)
        printf("%d %d\n", a[i], i);
      break;
    }
  }

  return ret;
}
