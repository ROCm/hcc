
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
  extent<1> ex(GRID_SIZE);

  // launch a kernel, log current timestamp
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table(idx) = __clock_u64();
  }).wait();

  // sleep for 1 second
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // launch a kernel again, log current timestamp
  array<uint64_t, 1> table2(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table2(idx) = __clock_u64();
  }).wait();

  // verify result
  bool ret = true;
  std::vector<uint64_t> result = table;
  std::vector<uint64_t> result2 = table2;
  
  // timestamp in the 2nd kernel must be larger than timestamp in the 1st kernel
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (result2[i] > result[i]);
  }

  return !(ret == true);
}

