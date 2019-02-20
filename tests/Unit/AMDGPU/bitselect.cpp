
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>
#include <vector>

#define GRID_SIZE (256)

#define TEST_DEBUG (0)


// A test case to verify bit functions
// - __bitselect_b32
// - __bitselect_b64

// CPU implementation of bitselect
uint32_t bitselect_b32(uint32_t src0, uint32_t src1, uint32_t src2)
{
    return (src1 & src0) | (src2 & ~src0);
}
uint64_t bitselect_b64(uint64_t src0, uint64_t src1, uint64_t src2)
{
    return (src1 & src0) | (src2 & ~src0);
}

// test __bitselect_b32
bool test_bitselect_b32() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<uint32_t> uint32_src_dist;
  std::vector<uint32_t> test0(GRID_SIZE);
  std::vector<uint32_t> test1(GRID_SIZE);
  std::vector<uint32_t> test2(GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    test0[i] = uint32_src_dist(rd);
    test1[i] = uint32_src_dist(rd);
    test2[i] = uint32_src_dist(rd);
  }
  array<uint32_t, 1> test0_GPU(GRID_SIZE);
  copy(test0.begin(), test0_GPU);
  array<uint32_t, 1> test1_GPU(GRID_SIZE);
  copy(test1.begin(), test1_GPU);
  array<uint32_t, 1> test2_GPU(GRID_SIZE);
  copy(test2.begin(), test2_GPU);

  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](hc::index<1>& idx) [[hc]] {
    output_GPU(idx) = __bitselect_b32(test0_GPU(idx), test1_GPU(idx), test2_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == bitselect_b32(test0[i], test1[i], test2[i]));
#if TEST_DEBUG
    std::cout << test0[i] << " " << test1[i] << " " << test2[i] << " "
              << bitselect_b32(test0[i], test1[i], test2[i])
              << " " << output[i] << "\n";
#endif
  }

  return ret;
}

// test __bitselect_b64
bool test_bitselect_b64() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<uint64_t> uint64_src_dist;
  std::vector<uint64_t> test0(GRID_SIZE);
  std::vector<uint64_t> test1(GRID_SIZE);
  std::vector<uint64_t> test2(GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    test0[i] = uint64_src_dist(rd);
    test1[i] = uint64_src_dist(rd);
    test2[i] = uint64_src_dist(rd);
  }
  array<uint64_t, 1> test0_GPU(GRID_SIZE);
  copy(test0.begin(), test0_GPU);
  array<uint64_t, 1> test1_GPU(GRID_SIZE);
  copy(test1.begin(), test1_GPU);
  array<uint64_t, 1> test2_GPU(GRID_SIZE);
  copy(test2.begin(), test2_GPU);

  array<uint64_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](hc::index<1>& idx) [[hc]] {
    output_GPU(idx) = __bitselect_b64(test0_GPU(idx), test1_GPU(idx), test2_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint64_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == bitselect_b64(test0[i], test1[i], test2[i]));
#if TEST_DEBUG
    std::cout << test0[i] << " " << test1[i] << " " << test2[i] << " "
              << bitselect_b64(test0[i], test1[i], test2[i])
              << " " << output[i] << "\n";
#endif
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test_bitselect_b32();
  ret &= test_bitselect_b64();

#if TEST_DEBUG
  std::cout << "ret: " << ret << std::endl;
#endif
  return !(ret == true);
}

