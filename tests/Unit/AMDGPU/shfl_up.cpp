
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
template<typename T>
bool test__shfl_up(int grid_size, int offset, T init_value) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<T, 1> table(grid_size);

  // shift values up in a wavefront
  parallel_for_each(ex, [&, offset, init_value](index<1>& idx) [[hc]] {
    T value = init_value + __lane_id();
    value = __shfl_up(value, offset);
    table(idx) = value;
  }).wait();

  std::vector<T> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % WAVEFRONT_SIZE;
    ret &= (output[i] == (init_value + ((laneId < offset) ? laneId : laneId - offset)));
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
template<typename T>
bool test__shfl_up2(int grid_size, int sub_wavefront_width, int offset, T init_value) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<T, 1> table(grid_size);

  // shift values up in a wavefront, divided into subsections
  parallel_for_each(ex, [&, offset, sub_wavefront_width, init_value](index<1>& idx) [[hc]] {
    T value = init_value + (__lane_id() % sub_wavefront_width);
    value = __shfl_up(value, offset, sub_wavefront_width);
    table(idx) = value;
  }).wait();

  std::vector<T> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % sub_wavefront_width;
    ret &= (output[i] == (init_value + ((laneId < offset) ? laneId : laneId - offset)));
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
  std::uniform_real_distribution<float> float_dist(0.0, 1.0);

  // test __shfl_up for different grid sizes and offsets
  ret &= test__shfl_up<int>(2, 0, int_dist(rd));
  ret &= test__shfl_up<int>(2, 1, int_dist(rd));
  ret &= test__shfl_up<int>(3, 0, int_dist(rd));
  ret &= test__shfl_up<int>(3, 1, int_dist(rd));
  ret &= test__shfl_up<int>(3, 2, int_dist(rd));
  ret &= test__shfl_up<int>(16, 1, int_dist(rd));
  ret &= test__shfl_up<int>(16, 2, int_dist(rd));
  ret &= test__shfl_up<int>(16, 3, int_dist(rd));
  ret &= test__shfl_up<int>(16, 7, int_dist(rd));
  ret &= test__shfl_up<int>(31, 1, int_dist(rd));
  ret &= test__shfl_up<int>(31, 2, int_dist(rd));
  ret &= test__shfl_up<int>(31, 4, int_dist(rd));
  ret &= test__shfl_up<int>(31, 8, int_dist(rd));
  ret &= test__shfl_up<int>(31, 15, int_dist(rd));
  ret &= test__shfl_up<int>(64, 1, int_dist(rd));
  ret &= test__shfl_up<int>(64, 2, int_dist(rd));
  ret &= test__shfl_up<int>(64, 4, int_dist(rd));
  ret &= test__shfl_up<int>(64, 8, int_dist(rd));
  ret &= test__shfl_up<int>(64, 15, int_dist(rd));
  ret &= test__shfl_up<int>(127, 1, int_dist(rd));
  ret &= test__shfl_up<int>(127, 2, int_dist(rd));
  ret &= test__shfl_up<int>(127, 4, int_dist(rd));
  ret &= test__shfl_up<int>(127, 8, int_dist(rd));
  ret &= test__shfl_up<int>(127, 16, int_dist(rd));
  ret &= test__shfl_up<int>(127, 32, int_dist(rd));
  ret &= test__shfl_up<int>(1023, 1, int_dist(rd));
  ret &= test__shfl_up<int>(1023, 2, int_dist(rd));
  ret &= test__shfl_up<int>(1023, 4, int_dist(rd));
  ret &= test__shfl_up<int>(1023, 8, int_dist(rd));
  ret &= test__shfl_up<int>(1023, 16, int_dist(rd));
  ret &= test__shfl_up<int>(1023, 32, int_dist(rd));

  ret &= test__shfl_up<float>(2, 0, float_dist(rd));
  ret &= test__shfl_up<float>(2, 1, float_dist(rd));
  ret &= test__shfl_up<float>(3, 0, float_dist(rd));
  ret &= test__shfl_up<float>(3, 1, float_dist(rd));
  ret &= test__shfl_up<float>(3, 2, float_dist(rd));
  ret &= test__shfl_up<float>(16, 1, float_dist(rd));
  ret &= test__shfl_up<float>(16, 2, float_dist(rd));
  ret &= test__shfl_up<float>(16, 3, float_dist(rd));
  ret &= test__shfl_up<float>(16, 7, float_dist(rd));
  ret &= test__shfl_up<float>(31, 1, float_dist(rd));
  ret &= test__shfl_up<float>(31, 2, float_dist(rd));
  ret &= test__shfl_up<float>(31, 4, float_dist(rd));
  ret &= test__shfl_up<float>(31, 8, float_dist(rd));
  ret &= test__shfl_up<float>(31, 15, float_dist(rd));
  ret &= test__shfl_up<float>(64, 1, float_dist(rd));
  ret &= test__shfl_up<float>(64, 2, float_dist(rd));
  ret &= test__shfl_up<float>(64, 4, float_dist(rd));
  ret &= test__shfl_up<float>(64, 8, float_dist(rd));
  ret &= test__shfl_up<float>(64, 15, float_dist(rd));
  ret &= test__shfl_up<float>(127, 1, float_dist(rd));
  ret &= test__shfl_up<float>(127, 2, float_dist(rd));
  ret &= test__shfl_up<float>(127, 4, float_dist(rd));
  ret &= test__shfl_up<float>(127, 8, float_dist(rd));
  ret &= test__shfl_up<float>(127, 16, float_dist(rd));
  ret &= test__shfl_up<float>(127, 32, float_dist(rd));
  ret &= test__shfl_up<float>(1023, 1, float_dist(rd));
  ret &= test__shfl_up<float>(1023, 2, float_dist(rd));
  ret &= test__shfl_up<float>(1023, 4, float_dist(rd));
  ret &= test__shfl_up<float>(1023, 8, float_dist(rd));
  ret &= test__shfl_up<float>(1023, 16, float_dist(rd));
  ret &= test__shfl_up<float>(1023, 32, float_dist(rd));

  // test __shfl_up for different grid sizes, different subsection sizes, and different offsets
  ret &= test__shfl_up2<int>(3, 2, 0, int_dist(rd));
  ret &= test__shfl_up2<int>(3, 2, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(16, 2, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(16, 4, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(16, 4, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(16, 8, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(16, 8, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(16, 8, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 2, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 4, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 4, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 8, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 8, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 8, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 16, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 16, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 16, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(31, 16, 8, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 2, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 4, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 4, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 8, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 8, 3, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 8, 5, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 16, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 16, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 16, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 16, 8, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 32, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 32, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 32, 3, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 32, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 32, 7, int_dist(rd));
  ret &= test__shfl_up2<int>(64, 32, 15, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 2, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 4, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 4, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 8, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 8, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 8, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 16, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 16, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 16, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 16, 8, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 32, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 32, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 32, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 32, 8, int_dist(rd));
  ret &= test__shfl_up2<int>(127, 32, 16, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 2, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 4, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 4, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 8, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 8, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 8, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 16, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 16, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 16, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 16, 8, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 32, 1, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 32, 2, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 32, 4, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 32, 8, int_dist(rd));
  ret &= test__shfl_up2<int>(1023, 32, 16, int_dist(rd));

  ret &= test__shfl_up2<float>(3, 2, 0, float_dist(rd));
  ret &= test__shfl_up2<float>(3, 2, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(16, 2, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(16, 4, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(16, 4, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(16, 8, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(16, 8, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(16, 8, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 2, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 4, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 4, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 8, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 8, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 8, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 16, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 16, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 16, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(31, 16, 8, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 2, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 4, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 4, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 8, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 8, 3, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 8, 5, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 16, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 16, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 16, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 16, 8, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 32, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 32, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 32, 3, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 32, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 32, 7, float_dist(rd));
  ret &= test__shfl_up2<float>(64, 32, 15, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 2, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 4, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 4, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 8, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 8, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 8, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 16, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 16, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 16, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 16, 8, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 32, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 32, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 32, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 32, 8, float_dist(rd));
  ret &= test__shfl_up2<float>(127, 32, 16, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 2, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 4, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 4, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 8, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 8, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 8, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 16, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 16, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 16, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 16, 8, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 32, 1, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 32, 2, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 32, 4, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 32, 8, float_dist(rd));
  ret &= test__shfl_up2<float>(1023, 32, 16, float_dist(rd));

  return !(ret == true);
}

