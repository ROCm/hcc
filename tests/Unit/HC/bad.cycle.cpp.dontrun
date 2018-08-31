
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#define GRID_SIZE (1024)

int main() {
  using namespace hc;
  array<uint64_t, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

  // launch a kernel, log current hardware cycle count
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table(idx) = __cycle_u64();
  }).wait();

  // we can't do any verification here because the hardware cycle count may be
  // affected by DPVS, plus on HSAIL this would always return 0
  // so for now the best verification is this program compiles and runs

  bool ret = true;

  return !(ret == true);
}

