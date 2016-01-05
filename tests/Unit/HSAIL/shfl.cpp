// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>
#include <random>

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
    if (hsail_activelaneid_u32() == 0)
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

  return !(ret == true);
}

