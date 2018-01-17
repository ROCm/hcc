
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>
#include <vector>

#define GRID_SIZE (1024)

#define TEST_DEBUG (0)

// A test case to verify HSAIL builtin function
// - __popcount_u32_b32
// - __popcount_u32_b64

// CPU implementation of popcount
template<typename T>
uint32_t popcount_cpu(T value) {
  uint32_t ret = 0;
  while (value) {
    if (value & 0x1) ++ret;
    value >>= 1;
  }
  return ret;
}

// test __popcount_u32_b32
bool test32() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<uint32_t> uint32_t_dist;
  std::vector<uint32_t> test(GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    test[i] = uint32_t_dist(rd);
  }
  array<uint32_t, 1> test_GPU(GRID_SIZE);
  copy(test.begin(), test_GPU);

  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    output_GPU(idx) = __popcount_u32_b32(test_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == popcount_cpu(test[i])); 
#if TEST_DEBUG
    std::cout << test[i] << " " << popcount_cpu(test[i]) << " " << output[i] << "\n";
#endif
  }

  return ret;
}

// test __popcount_u32_b64
bool test64() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<uint64_t> uint64_t_dist;
  std::vector<uint64_t> test(GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    test[i] = uint64_t_dist(rd);
  }
  array<uint64_t, 1> test_GPU(GRID_SIZE);
  copy(test.begin(), test_GPU);
  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    output_GPU(idx) = __popcount_u32_b64(test_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == popcount_cpu(test[i])); 
#if TEST_DEBUG
    std::cout << test[i] << " " << popcount_cpu(test[i]) << " " << output[i] << "\n";
#endif
  }

  return ret;
}

int main() {
  bool ret = true;

  // test 32-bit version
  ret &= test32();

  // test 64-bit version
  ret &= test64();

  return !(ret == true);
}

