// RUN: %not %hc %s -o %t.out 2>&1 | %not grep 'Segmentation fault'
#include <hc.hpp>
#include <vector>

#define GRID_SIZE (1024)
void func(void) [[hc]];

int main() {
  using namespace hc;
  array<unsigned int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  auto k = [&](index<1>& idx) [[hc]] {
    table(idx) = idx[0];
    func();
  };
  parallel_for_each(ex, k).wait();
  return 0;
}

