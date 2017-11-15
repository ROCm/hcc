
// RUN: %hc %s -lhc_am -o %t.out && %t.out

#include <cstdlib>
#include <cstdio>
#include <hc.hpp>
#include <hc_am.hpp>

class Point {
public:
  int x;
  int y;
  int z;

  /* hcc requires an explicit dtor?? */
  ~Point() [[hc,cpu]] {}
};

int main() {

  hc::extent<1> grid(64);
  hc::accelerator acc;
  static hc::accelerator_view av = acc.get_default_view();
  char* gpu_buffer = hc::am_alloc(sizeof(Point)*grid[0], acc, 0);

  hc::parallel_for_each(grid,[=](hc::index<1> idx) [[hc]] {
     auto p = new(gpu_buffer + sizeof(Point) * idx[0]) Point;
     p->x = idx[0];
     p->y = 0;
     p->z = 0;
     p->~Point();
  }).wait();

  char* buffer = (char*)malloc(sizeof(Point)*grid[0]);
  av.copy(gpu_buffer, buffer, sizeof(Point)*grid[0]);
  hc::am_free(gpu_buffer);

  int errors = 0;
  for (int i = 0; i < grid[0]; i++) {
    auto p = new(buffer+sizeof(Point)*i) Point;
    if (p->x != i
       || p->y != 0
       || p->z != 0) {
      errors++;
    }
#ifdef DEBUG
    printf("Point[%d]: x=%d, y=%d, z=%d\n",i,p->x,p->y,p->z);
#endif
    p->~Point();
  }
  free(buffer);
  return errors;
}
