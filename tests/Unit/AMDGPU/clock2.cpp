
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

#define GRID_SIZE (1024)

int main() {
  using namespace hc;
  array<uint64_t, 1> table(GRID_SIZE);
  array<uint64_t, 1> table2(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

  // launch a kernel, log current timestamp
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table(idx) = __clock_u64();
    table2(idx) = __clock_u64();
  }).wait();

  // verify result
  bool ret = true;
  std::vector<uint64_t> result = table;
  std::vector<uint64_t> result2 = table2;
  
  // The 2nd timestamp must be larger than the 1st timestamp
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (result2[i] > result[i]);
  }

  return !(ret == true);
}

