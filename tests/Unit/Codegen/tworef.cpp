// RUN: %amp_device -D__KALMAR_ACCELERATOR__ -c -S -emit-llvm %s
#include <hc.hpp>

using namespace hc;

int main()
{
  unsigned length = 16;
  array<int, 1> temp(length);
  array<int, 1> data(length);
  extent<1> cdomain_transpose(16);
  parallel_for_each (cdomain_transpose, [=, &data, &temp] (hc::index<1> tidx)  [[hc]] {});
  return 0;
}
