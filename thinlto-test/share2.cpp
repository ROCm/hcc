#include <iostream>
#include <hc.hpp>
extern "C" int foo(int) [[hc]];
extern "C" int bar(int) [[hc]];

int main() {
  using namespace hc;
  int ret = 0;
  int grid_size = 4;
  extent<1> ex(grid_size);
  array_view<int, 1> av(grid_size);

  parallel_for_each(ex, [=](index<1>& idx) [[hc]] {
    av(idx) = foo(grid_size);
    av(idx) = bar(av(idx));
  }).wait();

  for (int i = 0; i < grid_size; ++i) {
    ret += av[i];
  }

  std::cout << "end: " << ret << std::endl;
  return ret;
}
