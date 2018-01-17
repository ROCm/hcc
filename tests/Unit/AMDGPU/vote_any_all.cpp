
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>
#include <vector>

#define WAVEFRONT_SIZE (64) // as of now, all HSA agents have wavefront size of 64

#define GRID_SIZE (WAVEFRONT_SIZE * (WAVEFRONT_SIZE + 1))

#define TEST_DEBUG (0)

// A test case to verify HSAIL builtin function
// - __any
// - __all

// test __any
bool test__any() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  // test is a table of size WAVEFRONT_SIZE * WAVEFRONT_SIZE
  std::vector<uint32_t> test(GRID_SIZE);

  std::random_device rd;
  std::uniform_int_distribution<int> int_dist(0, WAVEFRONT_SIZE - 1);

  // for each block of WAVEFRONT_SIZE, we randomly set 1s inside the block
  // the number of 1s in the block equals to the index of the block
  // (the 1st block of WAVEFRONT_SIZE has 0 1s, the 2nd block of WAVEFRONT_SIZE has 1 1, and so on)
  for (int i = 0; i <= WAVEFRONT_SIZE; ++i) {
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
      if (j < i) {
        test[i * WAVEFRONT_SIZE + j] = 1;
      } else {
        test[i * WAVEFRONT_SIZE + j] = 0;
      }
    }

    // randomly shuffle items in the block
    for (int j = 0; j < WAVEFRONT_SIZE * 10; ++j) {
      int k1 = int_dist(rd);
      int k2 = int_dist(rd); 
      if (k1 != k2) {
        test[i * WAVEFRONT_SIZE + k1] ^= test[i * WAVEFRONT_SIZE + k2] ^= test[i * WAVEFRONT_SIZE + k1] ^= test[i * WAVEFRONT_SIZE + k2];
      }
    }
  }

  for (int i = 0; i <= WAVEFRONT_SIZE; ++i) {
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
#if TEST_DEBUG
      std::cout << test[i * WAVEFRONT_SIZE + j] << " ";
#endif
    }
#if TEST_DEBUG
    std::cout << "\n";
#endif
  }

  array<uint32_t, 1> test_GPU(GRID_SIZE);
  copy(test.begin(), test_GPU);

  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    output_GPU(idx) = __any(test_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i <= WAVEFRONT_SIZE; ++i) {
    // in test__any(), only the first wavefront would have result 0
    // all other wavefronts would have non-zero result because at least
    // one thread have non-zero predicate, so __any would let all threads
    // have non-zero results
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
      ret &= ((i == 0) ? (output[i * WAVEFRONT_SIZE + j] == 0) : (output[i * WAVEFRONT_SIZE + j] != 0));
#if TEST_DEBUG
      std::cout << output[i * WAVEFRONT_SIZE + j] << " ";
#endif
    }
#if TEST_DEBUG
    std::cout << "\n";
#endif
  }

  return ret;
}

// test __all
bool test__all() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  // test is a table of size WAVEFRONT_SIZE * WAVEFRONT_SIZE
  std::vector<uint32_t> test(GRID_SIZE);

  std::random_device rd;
  std::uniform_int_distribution<int> int_dist(0, WAVEFRONT_SIZE - 1);

  // for each block of WAVEFRONT_SIZE, we randomly set 1s inside the block
  // the number of 1s in the block equals to the index of the block
  // (the 1st block of WAVEFRONT_SIZE has 0 1s, the 2nd block of WAVEFRONT_SIZE has 1 1, and so on)
  for (int i = 0; i <= WAVEFRONT_SIZE; ++i) {
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
      if (j < i) {
        test[i * WAVEFRONT_SIZE + j] = 1;
      } else {
        test[i * WAVEFRONT_SIZE + j] = 0;
      }
    }

    // randomly shuffle items in the block
    for (int j = 0; j < WAVEFRONT_SIZE * 10; ++j) {
      int k1 = int_dist(rd);
      int k2 = int_dist(rd); 
      if (k1 != k2) {
        test[i * WAVEFRONT_SIZE + k1] ^= test[i * WAVEFRONT_SIZE + k2] ^= test[i * WAVEFRONT_SIZE + k1] ^= test[i * WAVEFRONT_SIZE + k2];
      }
    }
  }

  for (int i = 0; i <= WAVEFRONT_SIZE; ++i) {
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
#if TEST_DEBUG
      std::cout << test[i * WAVEFRONT_SIZE + j] << " ";
#endif
    }
#if TEST_DEBUG
    std::cout << "\n";
#endif
  }

  array<uint32_t, 1> test_GPU(GRID_SIZE);
  copy(test.begin(), test_GPU);

  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    output_GPU(idx) = __all(test_GPU(idx));
  });

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i <= WAVEFRONT_SIZE; ++i) {
    // in test__all(), only the last wavefront would have result 1
    // all other wavefronts would have zero result because at least
    // one thread have non-zero predicate, so __all would let all threads
    // have results 0
    for (int j = 0; j < WAVEFRONT_SIZE; ++j) {
      ret &= ((i == WAVEFRONT_SIZE) ? (output[i * WAVEFRONT_SIZE + j] != 0) : (output[i * WAVEFRONT_SIZE + j] == 0));
#if TEST_DEBUG
      std::cout << output[i * WAVEFRONT_SIZE + j] << " ";
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

  ret &= test__any();
  ret &= test__all();

  return !(ret == true);
}

