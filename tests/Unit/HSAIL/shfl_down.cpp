// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>
#include <random>

#define WAVEFRONT_SIZE (64) // as of now, all HSA agents have wavefront size of 64

#define TEST_DEBUG (0)

// A test case to verify HSAIL builtin function
// - __shfl_down

// test __shfl_down
bool test__shfl_down(int grid_size, int arg) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<int, 1> table(grid_size);

  // shift values down in a wavefront
  parallel_for_each(ex, [&, arg](index<1>& idx) [[hc]] {
    int value = __activelaneid_u32();
    value = __shfl_down(value, arg);
    table(idx) = value;
  }).wait();

  std::vector<int> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % WAVEFRONT_SIZE;
    ret &= (output[i] == (((laneId + arg) >= WAVEFRONT_SIZE) ? laneId : laneId + arg));
#if TEST_DEBUG
    std::cout << output[i] << " ";
    if ((i + 1) % WAVEFRONT_SIZE == 0)
      std::cout << "\n";
#endif
  }
#if TEST_DEBUG
    std::cout << "\n";
#endif

  return ret;
}

// test __shfl_down with different sub-wavefront widths
bool test__shfl_down2(int grid_size, int sub_wavefront_width, int arg) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<int, 1> table(grid_size);

  // shift values down in a wavefront, divided into subsections
  parallel_for_each(ex, [&, arg, sub_wavefront_width](index<1>& idx) [[hc]] {
    int value = __activelaneid_u32() % sub_wavefront_width;
    value = __shfl_down(value, arg, sub_wavefront_width);
    table(idx) = value;
  }).wait();

  std::vector<int> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % sub_wavefront_width;
    ret &= (output[i] == (((laneId + arg) >= sub_wavefront_width) ? laneId : laneId + arg));
#if TEST_DEBUG
    std::cout << output[i] << " ";
    if ((i + 1) % WAVEFRONT_SIZE == 0)
      std::cout << "\n";
#endif
  }
#if TEST_DEBUG
    std::cout << "\n";
#endif

  return ret;
}

int main() {
  bool ret = true;

  // NOTICE: the test case is designed so grid size is always a multiple of wavefront size (64),
  // so there will not be any inactive work-items

  // test __shfl_down for different grid sizes and offsets
  ret &= test__shfl_down(64, 1);
  ret &= test__shfl_down(64, 2);
  ret &= test__shfl_down(64, 4);
  ret &= test__shfl_down(64, 8);
  ret &= test__shfl_down(64, 15);
  ret &= test__shfl_down(128, 1);
  ret &= test__shfl_down(128, 2);
  ret &= test__shfl_down(128, 4);
  ret &= test__shfl_down(128, 8);
  ret &= test__shfl_down(128, 16);
  ret &= test__shfl_down(128, 32);
  ret &= test__shfl_down(1024, 1);
  ret &= test__shfl_down(1024, 2);
  ret &= test__shfl_down(1024, 4);
  ret &= test__shfl_down(1024, 8);
  ret &= test__shfl_down(1024, 16);
  ret &= test__shfl_down(1024, 32);

  // test __shfl_down for different grid sizes, different subsection sizes, and different offsets
  ret &= test__shfl_down2(64, 2, 1);
  ret &= test__shfl_down2(64, 4, 1);
  ret &= test__shfl_down2(64, 4, 2);
  ret &= test__shfl_down2(64, 8, 1);
  ret &= test__shfl_down2(64, 8, 3);
  ret &= test__shfl_down2(64, 8, 5);
  ret &= test__shfl_down2(64, 16, 1);
  ret &= test__shfl_down2(64, 16, 2);
  ret &= test__shfl_down2(64, 16, 4);
  ret &= test__shfl_down2(64, 16, 8);
  ret &= test__shfl_down2(64, 32, 1);
  ret &= test__shfl_down2(64, 32, 2);
  ret &= test__shfl_down2(64, 32, 3);
  ret &= test__shfl_down2(64, 32, 4);
  ret &= test__shfl_down2(64, 32, 7);
  ret &= test__shfl_down2(64, 32, 15);
  ret &= test__shfl_down2(128, 2, 1);
  ret &= test__shfl_down2(128, 4, 1);
  ret &= test__shfl_down2(128, 4, 2);
  ret &= test__shfl_down2(128, 8, 1);
  ret &= test__shfl_down2(128, 8, 2);
  ret &= test__shfl_down2(128, 8, 4);
  ret &= test__shfl_down2(128, 16, 1);
  ret &= test__shfl_down2(128, 16, 2);
  ret &= test__shfl_down2(128, 16, 4);
  ret &= test__shfl_down2(128, 16, 8);
  ret &= test__shfl_down2(128, 32, 1);
  ret &= test__shfl_down2(128, 32, 2);
  ret &= test__shfl_down2(128, 32, 4);
  ret &= test__shfl_down2(128, 32, 8);
  ret &= test__shfl_down2(128, 32, 16);
  ret &= test__shfl_down2(1024, 2, 1);
  ret &= test__shfl_down2(1024, 4, 1);
  ret &= test__shfl_down2(1024, 4, 2);
  ret &= test__shfl_down2(1024, 8, 1);
  ret &= test__shfl_down2(1024, 8, 2);
  ret &= test__shfl_down2(1024, 8, 4);
  ret &= test__shfl_down2(1024, 16, 1);
  ret &= test__shfl_down2(1024, 16, 2);
  ret &= test__shfl_down2(1024, 16, 4);
  ret &= test__shfl_down2(1024, 16, 8);
  ret &= test__shfl_down2(1024, 32, 1);
  ret &= test__shfl_down2(1024, 32, 2);
  ret &= test__shfl_down2(1024, 32, 4);
  ret &= test__shfl_down2(1024, 32, 8);
  ret &= test__shfl_down2(1024, 32, 16);

  return !(ret == true);
}

