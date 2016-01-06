// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>
#include <random>

#define WAVEFRONT_SIZE (64) // as of now, all HSA agents have wavefront size of 64

#define TEST_DEBUG (0)

// A test case to verify HSAIL builtin function
// - __shfl

// test __shfl
bool test__shfl(int grid_size, int arg) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<int, 1> table(grid_size);

  // broadcast of a single value across a wavefront
  parallel_for_each(ex, [&, arg](index<1>& idx) [[hc]] {
    int value = 0;
    if (__activelaneid_u32() == 0)
      value = arg;
    value = __shfl(value, 0);
    table(idx) = value;
  }).wait();

  std::vector<int> output = table;
  for (int i = 0; i < grid_size; ++i) {
    ret &= (output[i] == arg);
#if TEST_DEBUG
    std::cout << output[i] << " ";
#endif
  }
#if TEST_DEBUG
    std::cout << "\n";
#endif

  return ret;
}

// test __shfl with different sub-wavefront widths
bool test__shfl2(int grid_size, int sub_wavefront_width, int arg) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<int, 1> table(grid_size);

  // broadcast of a single value across a sub-wavefront
  parallel_for_each(ex, [&, arg, sub_wavefront_width](index<1>& idx) [[hc]] {
    int value = 0;
    unsigned int laneId = __activelaneid_u32();
    // each subsection of a wavefront would have a different test value
    if (laneId % sub_wavefront_width == 0)
      value = (arg + laneId / sub_wavefront_width);
    // broadcast the value within the subsection of a wavefront
    value = __shfl(value, 0, sub_wavefront_width);
    table(idx) = value;
  }).wait();

  std::vector<int> output = table;
  for (int i = 0; i < grid_size; ++i) {
    // each subsection of a wavefront would have a different value
    // the value will be restored to arg after one wavefront
    ret &= (output[i] == (arg + ((i % WAVEFRONT_SIZE) / sub_wavefront_width)));
#if TEST_DEBUG
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

  std::random_device rd;
  std::uniform_int_distribution<int> int_dist(0, 1023);

  // test __shfl for different grid sizes
  ret &= test__shfl(2, int_dist(rd));
  ret &= test__shfl(3, int_dist(rd));
  ret &= test__shfl(16, int_dist(rd));
  ret &= test__shfl(31, int_dist(rd));
  ret &= test__shfl(64, int_dist(rd));
  ret &= test__shfl(127, int_dist(rd));
  ret &= test__shfl(1023, int_dist(rd));

  // test __shfl for different grid sizes and different subsection sizes
  ret &= test__shfl2(3, 2, int_dist(rd));
  ret &= test__shfl2(16, 2, int_dist(rd));
  ret &= test__shfl2(16, 4, int_dist(rd));
  ret &= test__shfl2(16, 8, int_dist(rd));
  ret &= test__shfl2(31, 2, int_dist(rd));
  ret &= test__shfl2(31, 4, int_dist(rd));
  ret &= test__shfl2(31, 8, int_dist(rd));
  ret &= test__shfl2(31, 16, int_dist(rd));
  ret &= test__shfl2(64, 2, int_dist(rd));
  ret &= test__shfl2(64, 4, int_dist(rd));
  ret &= test__shfl2(64, 8, int_dist(rd));
  ret &= test__shfl2(64, 16, int_dist(rd));
  ret &= test__shfl2(64, 32, int_dist(rd));
  ret &= test__shfl2(127, 2, int_dist(rd));
  ret &= test__shfl2(127, 4, int_dist(rd));
  ret &= test__shfl2(127, 8, int_dist(rd));
  ret &= test__shfl2(127, 16, int_dist(rd));
  ret &= test__shfl2(127, 32, int_dist(rd));
  ret &= test__shfl2(1023, 2, int_dist(rd));
  ret &= test__shfl2(1023, 4, int_dist(rd));
  ret &= test__shfl2(1023, 8, int_dist(rd));
  ret &= test__shfl2(1023, 16, int_dist(rd));
  ret &= test__shfl2(1023, 32, int_dist(rd));

  return !(ret == true);
}

