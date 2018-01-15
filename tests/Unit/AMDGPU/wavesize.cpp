
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>

#define ANSWER (64) // for now all HSA agents have wavefront of size 64

#define GRID_SIZE (1024)

int main() {
  using namespace hc;
  array<unsigned int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table(idx) = __wavesize();
  }).wait();

  // verify result
  bool ret = true;
  std::vector<unsigned int> result = table;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (result[i] == ANSWER);
  }

  return !(ret == true);
}

