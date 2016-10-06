
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>
#include <vector>

#define WAVEFRONT_SIZE (64) // as of now, all HSA agents have wavefront size of 64

#define TEST_DEBUG (0)

// A test case to verify HSAIL builtin function
// - __activelanepermute_b32

// test __activelanepermute_b32
template<size_t GRID_SIZE>
bool test() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::vector<uint32_t> test(GRID_SIZE); // vector of input data
  std::vector<uint32_t> laneID(GRID_SIZE); // vector of laneID where the value would be transfered

  std::random_device rd;
  std::uniform_int_distribution<int> value_dist(0, GRID_SIZE - 1);
  std::uniform_int_distribution<int> index_dist(0, WAVEFRONT_SIZE - 1);

  for (int i = 0; i < (GRID_SIZE / WAVEFRONT_SIZE); ++i) {
    // for each block of WAVEFRONT_SIZE, we randomly set values inside the test vector
    // each item in the laneID will be initialized as 0..WAVEFRONT_SIZE-1 for each wavefront
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
      test[i * WAVEFRONT_SIZE + j] = value_dist(rd);
      laneID[i * WAVEFRONT_SIZE + j] = j;
    }

    // randomly shuffle items in the landID table
    for (int j = 0; j < WAVEFRONT_SIZE * 10; ++j) {
      int k1 = index_dist(rd);
      int k2 = index_dist(rd); 
      if (k1 != k2) {
        laneID[i * WAVEFRONT_SIZE + k1] ^= laneID[i * WAVEFRONT_SIZE + k2] ^= laneID[i * WAVEFRONT_SIZE + k1] ^= laneID[i * WAVEFRONT_SIZE + k2];
      }
    }
  }

  for (int i = 0; i < (GRID_SIZE / WAVEFRONT_SIZE); ++i) {
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
#if TEST_DEBUG
      std::cout << "(" << test[i * WAVEFRONT_SIZE + j] << ", " << laneID[i * WAVEFRONT_SIZE + j] << ") ";
#endif
    }
#if TEST_DEBUG
    std::cout << "\n";
#endif
  }

  array<uint32_t, 1> test_GPU(GRID_SIZE);
  copy(test.begin(), test_GPU);
  array<uint32_t, 1> laneID_GPU(GRID_SIZE);
  copy(laneID.begin(), laneID_GPU);

  array<uint32_t, 1> output_GPU(GRID_SIZE);
  array<uint32_t, 1> output2_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    // test __activelanepermute_b32 without useIdentity
    output_GPU(idx) = __activelanepermute_b32(test_GPU(idx), laneID_GPU(idx), 0, 0);
    // test __activelanepermute_b32 with useIdentity
    output2_GPU(idx) = __activelanepermute_b32(test_GPU(idx), laneID_GPU(idx), 1, 1);
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  std::vector<uint32_t> output2 = output2_GPU;
  for (int i = 0; i < (GRID_SIZE / WAVEFRONT_SIZE); ++i) {
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
      ret &= (output[i * WAVEFRONT_SIZE + j] == test[i * WAVEFRONT_SIZE + laneID[i * WAVEFRONT_SIZE + j]]);
      ret &= (output2[i * WAVEFRONT_SIZE + j] == 1);
#if TEST_DEBUG
      std::cout << "(" << output[i * WAVEFRONT_SIZE + j] << ", " << laneID[i * WAVEFRONT_SIZE + j] << ") ";
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

#if __hcc_backend__ == HCC_BACKEND_AMDGPU
  // XXX activelanepermute is not yet implemented on LC backend. let this case fail directly.
  ret = false;
#else
  ret &= test<64>();
  ret &= test<256>();
  ret &= test<1024>();
#endif

  return !(ret == true);
}

