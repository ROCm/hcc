// XFAIL: *
// RUN: %hc -lhc_am %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_am.hpp>

struct Foo {
  int table[3];
};

int main() {
  using namespace hc;

  Foo f;
  f.table[0] = 0;
  f.table[1] = 1;
  f.table[2] = 2;

  auto acc = accelerator();
  accelerator_view av = acc.get_default_view();

  int data[3] { 0 };
  int* data_d = (int*) am_alloc(3 * sizeof(int), acc, 0);

  av.copy(data, data_d, 3 * sizeof(int));

  parallel_for_each(extent<1>(3), [=](index<1> idx) [[hc]] {
    data_d[idx[0]] = f.table[idx[0]] + 999;
  });

  av.copy(data_d, data, 3 * sizeof(int));

  bool ret = true;
  for (int i = 0; i < 3; ++i) {
    ret &= (data[i] == (i + 999));
  }

  am_free(data_d);

  return !(ret == true);
}
