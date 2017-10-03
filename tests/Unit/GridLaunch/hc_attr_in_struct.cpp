
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "hc.hpp"
#include "grid_launch.hpp"
#include "hc_am.hpp"

struct Foo
{
  int m_bar = 5;

  __attribute__((hc)) int bar(int a)
  {
    return a+m_bar;
  }

  __attribute__((hc,cpu)) Foo() = default;
};

__attribute__((hc_grid_launch)) void kernel(const grid_launch_parm lp, const int a, int* data_d)
{
  Foo foo;
  data_d[0] = foo.bar(a);
}

int main() {

  auto acc = hc::accelerator();
  int* data1_d = (int*)hc::am_alloc(sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);
  lp.grid_dim = gl_dim3(1);
  lp.group_dim = gl_dim3(1);
  static hc::accelerator_view av = hc::accelerator().get_default_view();

  lp.cf = NULL;
  lp.av = &av;

  const int a = 1;
  kernel(lp, a, data1_d);

  int data1 = 0;
  av.copy(data1_d, &data1, sizeof(int));

  // Verify results
  Foo foo;
  return !(data1 == (a+foo.m_bar));

}

