
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
template<typename T>
bool test__shfl(int grid_size, T arg) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<T, 1> table(grid_size);

  // broadcast of a single value across a wavefront
  parallel_for_each(ex, [&, arg](index<1>& idx) [[hc]] {
    T value = T();
    if (__lane_id() == 0)
      value = arg;
    value = __shfl(value, 0);
    table(idx) = value;
  }).wait();

  std::vector<T> output = table;
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
template<typename T>
bool test__shfl2(int grid_size, int sub_wavefront_width, T arg) {
  bool ret = true;

  using namespace hc;
  extent<1> ex(grid_size);
  array<T, 1> table(grid_size);

  // broadcast of a single value across a sub-wavefront
  parallel_for_each(ex, [&, arg, sub_wavefront_width](index<1>& idx) [[hc]] {
    T value = T();
    unsigned int laneId = __lane_id();
    // each subsection of a wavefront would have a different test value
    if (laneId % sub_wavefront_width == 0)
      value = (arg + laneId / sub_wavefront_width);
    // broadcast the value within the subsection of a wavefront
    value = __shfl(value, 0, sub_wavefront_width);
    table(idx) = value;
  }).wait();

  std::vector<T> output = table;
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
  std::uniform_real_distribution<float> float_dist(0.0, 1.0);

  // test __shfl for different grid sizes
  ret &= test__shfl<int>(2, int_dist(rd));
  ret &= test__shfl<int>(3, int_dist(rd));
  ret &= test__shfl<int>(16, int_dist(rd));
  ret &= test__shfl<int>(31, int_dist(rd));
  ret &= test__shfl<int>(64, int_dist(rd));
  ret &= test__shfl<int>(127, int_dist(rd));
  ret &= test__shfl<int>(1023, int_dist(rd));

  ret &= test__shfl<float>(2, float_dist(rd));
  ret &= test__shfl<float>(3, float_dist(rd));
  ret &= test__shfl<float>(16, float_dist(rd));
  ret &= test__shfl<float>(31, float_dist(rd));
  ret &= test__shfl<float>(64, float_dist(rd));
  ret &= test__shfl<float>(127, float_dist(rd));
  ret &= test__shfl<float>(1023, float_dist(rd));

  // test __shfl for different grid sizes and different subsection sizes
  ret &= test__shfl2<int>(3, 2, int_dist(rd));
  ret &= test__shfl2<int>(16, 2, int_dist(rd));
  ret &= test__shfl2<int>(16, 4, int_dist(rd));
  ret &= test__shfl2<int>(16, 8, int_dist(rd));
  ret &= test__shfl2<int>(31, 2, int_dist(rd));
  ret &= test__shfl2<int>(31, 4, int_dist(rd));
  ret &= test__shfl2<int>(31, 8, int_dist(rd));
  ret &= test__shfl2<int>(31, 16, int_dist(rd));
  ret &= test__shfl2<int>(64, 2, int_dist(rd));
  ret &= test__shfl2<int>(64, 4, int_dist(rd));
  ret &= test__shfl2<int>(64, 8, int_dist(rd));
  ret &= test__shfl2<int>(64, 16, int_dist(rd));
  ret &= test__shfl2<int>(64, 32, int_dist(rd));
  ret &= test__shfl2<int>(127, 2, int_dist(rd));
  ret &= test__shfl2<int>(127, 4, int_dist(rd));
  ret &= test__shfl2<int>(127, 8, int_dist(rd));
  ret &= test__shfl2<int>(127, 16, int_dist(rd));
  ret &= test__shfl2<int>(127, 32, int_dist(rd));
  ret &= test__shfl2<int>(1023, 2, int_dist(rd));
  ret &= test__shfl2<int>(1023, 4, int_dist(rd));
  ret &= test__shfl2<int>(1023, 8, int_dist(rd));
  ret &= test__shfl2<int>(1023, 16, int_dist(rd));
  ret &= test__shfl2<int>(1023, 32, int_dist(rd));

  ret &= test__shfl2<float>(3, 2, float_dist(rd));
  ret &= test__shfl2<float>(16, 2, float_dist(rd));
  ret &= test__shfl2<float>(16, 4, float_dist(rd));
  ret &= test__shfl2<float>(16, 8, float_dist(rd));
  ret &= test__shfl2<float>(31, 2, float_dist(rd));
  ret &= test__shfl2<float>(31, 4, float_dist(rd));
  ret &= test__shfl2<float>(31, 8, float_dist(rd));
  ret &= test__shfl2<float>(31, 16, float_dist(rd));
  ret &= test__shfl2<float>(64, 2, float_dist(rd));
  ret &= test__shfl2<float>(64, 4, float_dist(rd));
  ret &= test__shfl2<float>(64, 8, float_dist(rd));
  ret &= test__shfl2<float>(64, 16, float_dist(rd));
  ret &= test__shfl2<float>(64, 32, float_dist(rd));
  ret &= test__shfl2<float>(127, 2, float_dist(rd));
  ret &= test__shfl2<float>(127, 4, float_dist(rd));
  ret &= test__shfl2<float>(127, 8, float_dist(rd));
  ret &= test__shfl2<float>(127, 16, float_dist(rd));
  ret &= test__shfl2<float>(127, 32, float_dist(rd));
  ret &= test__shfl2<float>(1023, 2, float_dist(rd));
  ret &= test__shfl2<float>(1023, 4, float_dist(rd));
  ret &= test__shfl2<float>(1023, 8, float_dist(rd));
  ret &= test__shfl2<float>(1023, 16, float_dist(rd));
  ret &= test__shfl2<float>(1023, 32, float_dist(rd));

  return !(ret == true);
}

