
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
template<typename T>
bool test__shfl_down(int grid_size, int offset, T init_value) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<T, 1> table(grid_size);

  // shift values down in a wavefront
  parallel_for_each(ex, [&, offset, init_value](index<1>& idx) [[hc]] {
    T value = init_value + __lane_id();
    value = __shfl_down(value, offset);
    table(idx) = value;
  }).wait();

  std::vector<T> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % WAVEFRONT_SIZE;
    ret &= (output[i] == (init_value + (((laneId + offset) >= WAVEFRONT_SIZE) ? laneId : laneId + offset)));
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
template<typename T>
bool test__shfl_down2(int grid_size, int sub_wavefront_width, int offset, T init_value) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<T, 1> table(grid_size);

  // shift values down in a wavefront, divided into subsections
  parallel_for_each(ex, [&, offset, sub_wavefront_width, init_value](index<1>& idx) [[hc]] {
    T value = init_value + (__lane_id() % sub_wavefront_width);
    value = __shfl_down(value, offset, sub_wavefront_width);
    table(idx) = value;
  }).wait();

  std::vector<T> output = table;
  for (int i = 0; i < grid_size; ++i) {
    int laneId = i % sub_wavefront_width;
    ret &= (output[i] == (init_value + (((laneId + offset) >= sub_wavefront_width) ? laneId : laneId + offset)));
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

  std::random_device rd;
  std::uniform_int_distribution<int> int_dist(0, 1023);
  std::uniform_real_distribution<float> float_dist(0.0, 1.0);

  // test __shfl_down for different grid sizes and offsets
  ret &= test__shfl_down<int>(64, 1, int_dist(rd));
  ret &= test__shfl_down<int>(64, 2, int_dist(rd));
  ret &= test__shfl_down<int>(64, 4, int_dist(rd));
  ret &= test__shfl_down<int>(64, 8, int_dist(rd));
  ret &= test__shfl_down<int>(64, 15, int_dist(rd));
  ret &= test__shfl_down<int>(128, 1, int_dist(rd));
  ret &= test__shfl_down<int>(128, 2, int_dist(rd));
  ret &= test__shfl_down<int>(128, 4, int_dist(rd));
  ret &= test__shfl_down<int>(128, 8, int_dist(rd));
  ret &= test__shfl_down<int>(128, 16, int_dist(rd));
  ret &= test__shfl_down<int>(128, 32, int_dist(rd));
  ret &= test__shfl_down<int>(1024, 1, int_dist(rd));
  ret &= test__shfl_down<int>(1024, 2, int_dist(rd));
  ret &= test__shfl_down<int>(1024, 4, int_dist(rd));
  ret &= test__shfl_down<int>(1024, 8, int_dist(rd));
  ret &= test__shfl_down<int>(1024, 16, int_dist(rd));
  ret &= test__shfl_down<int>(1024, 32, int_dist(rd));

  ret &= test__shfl_down<float>(64, 1, float_dist(rd));
  ret &= test__shfl_down<float>(64, 2, float_dist(rd));
  ret &= test__shfl_down<float>(64, 4, float_dist(rd));
  ret &= test__shfl_down<float>(64, 8, float_dist(rd));
  ret &= test__shfl_down<float>(64, 15, float_dist(rd));
  ret &= test__shfl_down<float>(128, 1, float_dist(rd));
  ret &= test__shfl_down<float>(128, 2, float_dist(rd));
  ret &= test__shfl_down<float>(128, 4, float_dist(rd));
  ret &= test__shfl_down<float>(128, 8, float_dist(rd));
  ret &= test__shfl_down<float>(128, 16, float_dist(rd));
  ret &= test__shfl_down<float>(128, 32, float_dist(rd));
  ret &= test__shfl_down<float>(1024, 1, float_dist(rd));
  ret &= test__shfl_down<float>(1024, 2, float_dist(rd));
  ret &= test__shfl_down<float>(1024, 4, float_dist(rd));
  ret &= test__shfl_down<float>(1024, 8, float_dist(rd));
  ret &= test__shfl_down<float>(1024, 16, float_dist(rd));
  ret &= test__shfl_down<float>(1024, 32, float_dist(rd));

  // test __shfl_down for different grid sizes, different subsection sizes, and different offsets
  ret &= test__shfl_down2<int>(64, 2, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 4, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 4, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 8, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 8, 3, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 8, 5, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 16, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 16, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 16, 4, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 16, 8, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 32, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 32, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 32, 3, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 32, 4, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 32, 7, int_dist(rd));
  ret &= test__shfl_down2<int>(64, 32, 15, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 2, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 4, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 4, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 8, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 8, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 8, 4, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 16, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 16, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 16, 4, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 16, 8, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 32, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 32, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 32, 4, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 32, 8, int_dist(rd));
  ret &= test__shfl_down2<int>(128, 32, 16, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 2, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 4, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 4, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 8, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 8, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 8, 4, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 16, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 16, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 16, 4, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 16, 8, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 32, 1, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 32, 2, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 32, 4, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 32, 8, int_dist(rd));
  ret &= test__shfl_down2<int>(1024, 32, 16, int_dist(rd));

  ret &= test__shfl_down2<float>(64, 2, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 4, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 4, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 8, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 8, 3, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 8, 5, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 16, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 16, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 16, 4, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 16, 8, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 32, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 32, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 32, 3, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 32, 4, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 32, 7, float_dist(rd));
  ret &= test__shfl_down2<float>(64, 32, 15, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 2, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 4, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 4, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 8, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 8, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 8, 4, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 16, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 16, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 16, 4, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 16, 8, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 32, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 32, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 32, 4, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 32, 8, float_dist(rd));
  ret &= test__shfl_down2<float>(128, 32, 16, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 2, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 4, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 4, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 8, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 8, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 8, 4, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 16, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 16, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 16, 4, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 16, 8, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 32, 1, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 32, 2, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 32, 4, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 32, 8, float_dist(rd));
  ret &= test__shfl_down2<float>(1024, 32, 16, float_dist(rd));

  return !(ret == true);
}

