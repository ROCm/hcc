
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>
#include <vector>

#define GRID_SIZE (256)

#define TEST_DEBUG (0)


// A test case to verify bit functions
// - __bitextract_u32
// - __bitextract_u64

// CPU implementation of bitextract
uint32_t bitextract_u32(uint32_t src0, uint32_t src1, uint32_t src2)
{
  uint32_t offset = src1 & 31;
  uint32_t width = src2 & 31;
  if (width == 0) {
    return 0;
  } else {
    return (src0 << (32 - width - offset)) >> (32 - width);
  }
}
uint64_t bitextract_u64(uint64_t src0, uint32_t src1, uint32_t src2)
{
  uint32_t offset = src1 & 63;
  uint32_t width = src2 & 63;
  if (width == 0) {
    return 0;
  } else {
    return (src0 << (64 - width - offset)) >> (64 - width);
  }
}

// test __bitextract_u32
bool test_bitextract_u32() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<uint32_t> uint32_src0_dist;
  std::uniform_int_distribution<uint32_t> uint32_src12_dist(0,31);
  std::vector<uint32_t> test0(GRID_SIZE);
  std::vector<uint32_t> test1(GRID_SIZE);
  std::vector<uint32_t> test2(GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    test0[i] = uint32_src0_dist(rd);
    test1[i] = uint32_src12_dist(rd);
    test2[i] = uint32_src12_dist(rd);
  }
  array<uint32_t, 1> test0_GPU(GRID_SIZE);
  array<uint32_t, 1> test1_GPU(GRID_SIZE);
  array<uint32_t, 1> test2_GPU(GRID_SIZE);
  copy(test0.begin(), test0_GPU);
  copy(test1.begin(), test1_GPU);
  copy(test2.begin(), test2_GPU);

  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](hc::index<1>& idx) [[hc]] {
    output_GPU(idx) = __bitextract_u32(test0_GPU(idx), test1_GPU(idx), test2_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == bitextract_u32(test0[i], test1[i], test2[i]));
#if TEST_DEBUG
    std::cout << test0[i] << " " << test1[i] << " " << test2[i] << " "
              << bitextract_u32(test0[i], test1[i], test2[i])
              << " " << output[i] << "\n";
#endif
  }

  return ret;
}

// test __bitextract_u64
bool test_bitextract_u64() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<uint64_t> uint64_src0_dist;
  std::uniform_int_distribution<uint32_t> uint32_src12_dist(0,63);
  std::vector<uint64_t> test0(GRID_SIZE);
  std::vector<uint64_t> test1(GRID_SIZE);
  std::vector<uint64_t> test2(GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    test0[i] = uint64_src0_dist(rd);
    test1[i] = uint32_src12_dist(rd);
    test2[i] = uint32_src12_dist(rd);
  }
  array<uint64_t, 1> test0_GPU(GRID_SIZE);
  array<uint64_t, 1> test1_GPU(GRID_SIZE);
  array<uint64_t, 1> test2_GPU(GRID_SIZE);
  copy(test0.begin(), test0_GPU);
  copy(test1.begin(), test1_GPU);
  copy(test2.begin(), test2_GPU);

  array<uint64_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](hc::index<1>& idx) [[hc]] {
    output_GPU(idx) = __bitextract_u64(test0_GPU(idx), test1_GPU(idx), test2_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint64_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == bitextract_u64(test0[i], test1[i], test2[i]));
#if TEST_DEBUG
    std::cout << test0[i] << " " << test1[i] << " " << test2[i] << " "
              << bitextract_u64(test0[i], test1[i], test2[i])
              << " " << output[i] << "\n";
#endif
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test_bitextract_u32();
  ret &= test_bitextract_u64();

#if TEST_DEBUG
  std::cout << "ret: " << ret << std::endl;
#endif
  return !(ret == true);
}

