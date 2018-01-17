
// RUN: %hc %s -DFILE_1 -c -o %t1.o && %hc %s -DFILE_2 -c -o %t2.o && %hc %t1.o %t2.o -lhc_am -o %t1.out && %t1.out
// RUN: %hc -lhc_am %s -DFILE_1 -DFILE_2 -o %t2.out && %t2.out


#include "grid_launch.hpp"
#include "hc_am.hpp"
#include "hc.hpp"

#ifdef FILE_1

__attribute__((hc_grid_launch)) void foo(grid_launch_parm lp, int* a)
{
  int x = hc_get_workitem_id(0) + hc_get_group_id(0)*lp.group_dim.x;
  a[x] = x;
}

#endif

#ifdef FILE_2

__attribute__((hc_grid_launch)) void foo(grid_launch_parm lp, int* a);

int main()
{
  int size = 1000;

  int* a = (int*)malloc(sizeof(int)*size);

  auto acc = hc::accelerator();
  int* a_d = (int*)hc::am_alloc(size*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.group_dim = gl_dim3(size);

  hc::completion_future cf;
  lp.cf = &cf;
  foo(lp, a_d);
  lp.cf->wait();


  static hc::accelerator_view av = acc.get_default_view();
  av.copy(a_d, a, size*sizeof(int));

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

  hc::am_free(a_d);
  free(a);

  return ret;
}

#endif

