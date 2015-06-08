// RUN: %amp_device -D__KALMAR_ACCELERATOR__ -c -S -emit-llvm %s
#include <amp.h>

using namespace concurrency;

int main()
{
  unsigned length = 16;
  array<int, 1> temp(length);
  array<int, 1> data(length);
  extent<1> cdomain_transpose(16);
  parallel_for_each (cdomain_transpose, [=, &data, &temp] (index<1> tidx)  restrict(amp) {});
  return 0;
}
