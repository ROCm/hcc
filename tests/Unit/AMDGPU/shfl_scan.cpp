
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>
#include <random>

#define WAVEFRONT_SIZE (64) // as of now, all HSA agents have wavefront size of 64

#define TEST_DEBUG (0)

// A test case to verify HSAIL builtin function
// - __shfl_up

bool test_scan(int grid_size, int sub_wavefront_width) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<int, 1> table(grid_size);

  parallel_for_each(ex, [&, sub_wavefront_width](index<1>& idx) [[hc]] {
    int laneId = __lane_id();
    int logicalLaneId = laneId % sub_wavefront_width;
    int value = (WAVEFRONT_SIZE - 1) - laneId;

    for (int i = 1; i <= (sub_wavefront_width / 2); i *= 2) {
      int n = __shfl_up(value, i, sub_wavefront_width);
      if (logicalLaneId >= i)
        value += n;
    }
    table(idx) = value;
  }).wait();

  std::vector<int> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % WAVEFRONT_SIZE;
    int logicalLaneId = laneId % sub_wavefront_width;
    int subWavefrontId = laneId / sub_wavefront_width;
    int expected = 0;
    for (int j = 0; j <= logicalLaneId; ++j) {
      expected += (WAVEFRONT_SIZE - 1) - (sub_wavefront_width * subWavefrontId) - j;
    }
    ret &= (output[i] == expected);
#if TEST_DEBUG
    //std::cout << expected << " ";
    std::cout << output[i] << " ";
#endif
  }
#if TEST_DEBUG
    std::cout << "\n";
#endif

  return ret;
}

int main() {
  bool ret = true;

  // test scan algorithm using __shfl_up
  ret &= test_scan(8, 2);
  ret &= test_scan(8, 4);
  ret &= test_scan(32, 8);
  ret &= test_scan(64, 8);
  ret &= test_scan(64, 32);
  ret &= test_scan(64, 64);
  ret &= test_scan(128, 8);
  ret &= test_scan(128, 16);
  ret &= test_scan(128, 32);
  ret &= test_scan(128, 64);

  return !(ret == true);
}

