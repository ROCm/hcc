// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>
#include <random>

#define WAVEFRONT_SIZE (64) // as of now, all HSA agents have wavefront size of 64

#define TEST_DEBUG (0)

// A test case to verify HSAIL builtin function
// - __shfl_up

// test __shfl_up
bool test__shfl_up(int grid_size, int arg) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<int, 1> table(grid_size);

  // shift values up in a wavefront
  parallel_for_each(ex, [&, arg](index<1>& idx) [[hc]] {
    int value = hsail_activelaneid_u32();
    value = __shfl_up(value, arg);
    table(idx) = value;
  }).wait();

  std::vector<int> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % WAVEFRONT_SIZE;
    ret &= (output[i] == ((laneId < arg) ? laneId : laneId - arg));
#if TEST_DEBUG
    std::cout << output[i] << " ";
#endif
  }
#if TEST_DEBUG
    std::cout << "\n";
#endif

  return ret;
}

// test __shfl_up with different sub-wavefront widths
bool test__shfl_up2(int grid_size, int sub_wavefront_width, int arg) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<int, 1> table(grid_size);

  // shift values up in a wavefront, divided into subsections
  parallel_for_each(ex, [&, arg, sub_wavefront_width](index<1>& idx) [[hc]] {
    int value = hsail_activelaneid_u32() % sub_wavefront_width;
    value = __shfl_up(value, arg, sub_wavefront_width);
    table(idx) = value;
  }).wait();

  std::vector<int> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % sub_wavefront_width;
    ret &= (output[i] == ((laneId < arg) ? laneId : laneId - arg));
#if TEST_DEBUG
    std::cout << output[i] << " ";
#endif
  }
#if TEST_DEBUG
    std::cout << "\n";
#endif

  return ret;
}

bool test_scan(int grid_size, int sub_wavefront_width) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<int, 1> table(grid_size);

  parallel_for_each(ex, [&, sub_wavefront_width](index<1>& idx) [[hc]] {
    int laneId = hsail_activelaneid_u32();
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

  // test __shfl_up for different grid sizes and offsets
  ret &= test__shfl_up(2, 0);
  ret &= test__shfl_up(2, 1);
  ret &= test__shfl_up(3, 0);
  ret &= test__shfl_up(3, 1);
  ret &= test__shfl_up(3, 2);
  ret &= test__shfl_up(16, 1);
  ret &= test__shfl_up(16, 2);
  ret &= test__shfl_up(16, 3);
  ret &= test__shfl_up(16, 7);
  ret &= test__shfl_up(31, 1);
  ret &= test__shfl_up(31, 2);
  ret &= test__shfl_up(31, 4);
  ret &= test__shfl_up(31, 8);
  ret &= test__shfl_up(31, 15);
  ret &= test__shfl_up(64, 1);
  ret &= test__shfl_up(64, 2);
  ret &= test__shfl_up(64, 4);
  ret &= test__shfl_up(64, 8);
  ret &= test__shfl_up(64, 15);
  ret &= test__shfl_up(127, 1);
  ret &= test__shfl_up(127, 2);
  ret &= test__shfl_up(127, 4);
  ret &= test__shfl_up(127, 8);
  ret &= test__shfl_up(127, 16);
  ret &= test__shfl_up(127, 32);
  ret &= test__shfl_up(1023, 1);
  ret &= test__shfl_up(1023, 2);
  ret &= test__shfl_up(1023, 4);
  ret &= test__shfl_up(1023, 8);
  ret &= test__shfl_up(1023, 16);
  ret &= test__shfl_up(1023, 32);

  // test __shfl_up for different grid sizes, different subsection sizes, and different offsets
  ret &= test__shfl_up2(3, 2, 0);
  ret &= test__shfl_up2(3, 2, 1);
  ret &= test__shfl_up2(16, 2, 1);
  ret &= test__shfl_up2(16, 4, 1);
  ret &= test__shfl_up2(16, 4, 2);
  ret &= test__shfl_up2(16, 8, 1);
  ret &= test__shfl_up2(16, 8, 2);
  ret &= test__shfl_up2(16, 8, 4);
  ret &= test__shfl_up2(31, 2, 1);
  ret &= test__shfl_up2(31, 4, 1);
  ret &= test__shfl_up2(31, 4, 2);
  ret &= test__shfl_up2(31, 8, 1);
  ret &= test__shfl_up2(31, 8, 2);
  ret &= test__shfl_up2(31, 8, 4);
  ret &= test__shfl_up2(31, 16, 1);
  ret &= test__shfl_up2(31, 16, 2);
  ret &= test__shfl_up2(31, 16, 4);
  ret &= test__shfl_up2(31, 16, 8);
  ret &= test__shfl_up2(64, 2, 1);
  ret &= test__shfl_up2(64, 4, 1);
  ret &= test__shfl_up2(64, 4, 2);
  ret &= test__shfl_up2(64, 8, 1);
  ret &= test__shfl_up2(64, 8, 3);
  ret &= test__shfl_up2(64, 8, 5);
  ret &= test__shfl_up2(64, 16, 1);
  ret &= test__shfl_up2(64, 16, 2);
  ret &= test__shfl_up2(64, 16, 4);
  ret &= test__shfl_up2(64, 16, 8);
  ret &= test__shfl_up2(64, 32, 1);
  ret &= test__shfl_up2(64, 32, 2);
  ret &= test__shfl_up2(64, 32, 3);
  ret &= test__shfl_up2(64, 32, 4);
  ret &= test__shfl_up2(64, 32, 7);
  ret &= test__shfl_up2(64, 32, 15);
  ret &= test__shfl_up2(127, 2, 1);
  ret &= test__shfl_up2(127, 4, 1);
  ret &= test__shfl_up2(127, 4, 2);
  ret &= test__shfl_up2(127, 8, 1);
  ret &= test__shfl_up2(127, 8, 2);
  ret &= test__shfl_up2(127, 8, 4);
  ret &= test__shfl_up2(127, 16, 1);
  ret &= test__shfl_up2(127, 16, 2);
  ret &= test__shfl_up2(127, 16, 4);
  ret &= test__shfl_up2(127, 16, 8);
  ret &= test__shfl_up2(127, 32, 1);
  ret &= test__shfl_up2(127, 32, 2);
  ret &= test__shfl_up2(127, 32, 4);
  ret &= test__shfl_up2(127, 32, 8);
  ret &= test__shfl_up2(127, 32, 16);
  ret &= test__shfl_up2(1023, 2, 1);
  ret &= test__shfl_up2(1023, 4, 1);
  ret &= test__shfl_up2(1023, 4, 2);
  ret &= test__shfl_up2(1023, 8, 1);
  ret &= test__shfl_up2(1023, 8, 2);
  ret &= test__shfl_up2(1023, 8, 4);
  ret &= test__shfl_up2(1023, 16, 1);
  ret &= test__shfl_up2(1023, 16, 2);
  ret &= test__shfl_up2(1023, 16, 4);
  ret &= test__shfl_up2(1023, 16, 8);
  ret &= test__shfl_up2(1023, 32, 1);
  ret &= test__shfl_up2(1023, 32, 2);
  ret &= test__shfl_up2(1023, 32, 4);
  ret &= test__shfl_up2(1023, 32, 8);
  ret &= test__shfl_up2(1023, 32, 16);

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

