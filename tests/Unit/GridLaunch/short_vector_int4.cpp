// RUN: %hc -lhc_am %s -o %t.out && %t.out
// RUN: %t.out

#include "grid_launch.hpp"
#include "hc_am.hpp"
#include "hc.hpp"
#include "hc_short_vector.hpp"
#include <iostream>

#define GRID_SIZE 256
#define TILE_SIZE 16

typedef hc::short_vector::int4::vector_value_type v_type;

const int SIZE = GRID_SIZE*TILE_SIZE;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, v_type *x) {
  int i = hc_get_workitem_id(0) + hc_get_group_id(0)*lp.group_dim.x;
  x[i] = {i, i * 10, i * 20, i * 30 };
}

int main(void) {
  auto acc = hc::accelerator();
  v_type* data1_d = (v_type*)hc::am_alloc(SIZE*sizeof(v_type), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim.x = GRID_SIZE;
  lp.group_dim.x = TILE_SIZE;

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1_d);
  lp.cf->wait();

  int *data1 = (int *)malloc(SIZE*sizeof(v_type));
  static hc::accelerator_view av = acc.get_default_view();
  av.copy(data1_d, data1, SIZE*sizeof(v_type));

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if(data1[i*4] != i
       || data1[i*4+1] != i*10
       || data1[i*4+2] != i*20
       || data1[i*4+3] != i*30){
      ret = 1;
      break;
    }
  }

  hc::am_free(data1_d);
  free(data1);

  return ret;
}
