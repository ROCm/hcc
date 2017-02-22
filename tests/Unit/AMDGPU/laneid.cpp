
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>
#include <vector>

#define WAVEFRONT_SIZE (64) // as of now, all HSA agents have wavefront size of 64

#define GRID_SIZE (1024)

#define TEST_DEBUG (0)

// A test case to verify AMDGPU builtin function
// - __lane_id()

// test __lane_id()
bool test() {
  using namespace hc;
  bool ret = true;

  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    output_GPU(idx) = __lane_id();
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE / WAVEFRONT_SIZE; ++i) {
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
      // each work-item in each wavefront must have unique active lane id
      ret &= (output[i * WAVEFRONT_SIZE + j] == j);
#if TEST_DEBUG
      std::cout << output[i * WAVEFRONT_SIZE + j] << " ";
#endif
    }
#if TEST_DEBUG
    std::cout << "\n";
#endif
  }

  return ret;
}


int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

