
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>
#include <vector>

#define GRID_SIZE (1024)

#define TEST_DEBUG (0)

// A test case to verify HSAIL builtin function
// - __firstbit_u32_u32
// - __firstbit_u32_u64
// - __firstbit_u32_s32
// - __firstbit_u32_s64

// CPU implementation of firstbit
// adopted from HSA PRM 5.9
uint32_t firstbit_u32(uint32_t a)
{  
   if (a == 0)
      return -1;  
   uint32_t pos = 0;  
   while ((int32_t)a > 0) { 
      a <<= 1; pos++;
   }
   return pos;
}
uint32_t firstbit_s32(int32_t a)
{  
   uint32_t u = a >= 0? a: ~a; // complement negative numbers  
   return firstbit_u32(u);
} 

uint32_t firstbit_u64(uint64_t a)
{  
   if (a == 0)
      return -1;  
   uint32_t pos = 0;  
   while ((int64_t)a > 0) { 
      a <<= 1; pos++;
   }
   return pos;
}
uint32_t firstbit_s64(int64_t a)
{  
   uint64_t u = a >= 0? a: ~a; // complement negative numbers  
   return firstbit_u64(u);
} 

// check if firstbit_u32 works
bool test_cpu_u32() {
  bool ret = true;

  ret &= (firstbit_u32(0) == -1);

  std::vector<uint32_t> test(32);
  for (int i = 0; i < 32; ++i) {
    test[i] = ((uint32_t)(-1) >> i);
    ret &= (firstbit_u32(test[i]) == i);
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_u32(test[i]) << "\n";
#endif
  }

  return ret;
}

// check if firstbit_u64 works
bool test_cpu_u64() {
  bool ret = true;

  ret &= (firstbit_u64(0) == -1);

  std::vector<uint64_t> test(64);
  for (int i = 0; i < 64; ++i) {
    test[i] = ((uint64_t)(-1) >> i);
    ret &= (firstbit_u64(test[i]) == i);
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_u64(test[i]) << "\n";
#endif
  }

  return ret;
}

// check if firstbit_s32 works
bool test_cpu_s32() {
  bool ret = true;

  ret &= (firstbit_s32(0) == -1);

  std::vector<int32_t> test(32);
  // positive integers
  for (int i = 1; i < 32; ++i) {
    test[i] = 1 << (31 - i); 
    ret &= (firstbit_s32(test[i]) == i);
#if TEST_DEBUG 
    std::cout << test[i] << " " << firstbit_s32(test[i]) << "\n";
#endif
  }
  // negative integers
  for (int i = 1; i < 32; ++i) {
    test[i] = -1 << (32 - i);  // notice the difference from positive cases
    ret &= (firstbit_s32(test[i]) == i);
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_s32(test[i]) << "\n";
#endif
  }

  return ret;
}

// check if firstbit_s64 works
bool test_cpu_s64() {
  bool ret = true;

  ret &= (firstbit_s64(0) == -1);

  std::vector<int64_t> test(64);
  // positive integers
  for (int i = 1; i < 64; ++i) {
    test[i] = (int64_t)1 << (63 - i); 
    ret &= (firstbit_s64(test[i]) == i);
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_s64(test[i]) << "\n";
#endif
  }
  // negative integers
  for (int i = 1; i < 64; ++i) {
    test[i] = (int64_t)-1 << (64 - i);  // notice the difference from positive cases
    ret &= (firstbit_s64(test[i]) == i);
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_s64(test[i]) << "\n";
#endif
  }

  return ret;
}

// test __firstbit_u32_u32
bool test_gpu_u32() {
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
    output_GPU(idx) = __firstbit_u32_u32(test_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == firstbit_u32(test[i])); 
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_u32(test[i]) << " " << output[i] << "\n";
#endif
  }

  return ret;
}

// test __firstbit_u32_u64
bool test_gpu_u64() {
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
    output_GPU(idx) = __firstbit_u32_u64(test_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == firstbit_u64(test[i])); 
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_u64(test[i]) << " " << output[i] << "\n";
#endif
  }

  return ret;
}

// test __firstbit_u32_s32
bool test_gpu_s32() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int32_t_dist;
  std::vector<int32_t> test(GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    test[i] = int32_t_dist(rd);
  }
  array<int32_t, 1> test_GPU(GRID_SIZE);
  copy(test.begin(), test_GPU);

  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    output_GPU(idx) = __firstbit_u32_s32(test_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == firstbit_s32(test[i])); 
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_s32(test[i]) << " " << output[i] << "\n";
#endif
  }

  return ret;
}

// test __firstbit_u32_s64
bool test_gpu_s64() {
  using namespace hc;
  bool ret = true;

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int64_t> int64_t_dist;
  std::vector<int64_t> test(GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    test[i] = int64_t_dist(rd);
  }
  array<int64_t, 1> test_GPU(GRID_SIZE);
  copy(test.begin(), test_GPU);
  array<uint32_t, 1> output_GPU(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    output_GPU(idx) = __firstbit_u32_s64(test_GPU(idx));
  }).wait();

  // verify result
  std::vector<uint32_t> output = output_GPU;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (output[i] == firstbit_s64(test[i])); 
#if TEST_DEBUG
    std::cout << test[i] << " " << firstbit_s64(test[i]) << " " << output[i] << "\n";
#endif
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test_cpu_u32();
  ret &= test_cpu_u64();
  ret &= test_cpu_s32();
  ret &= test_cpu_s64();

  ret &= test_gpu_u32();
  ret &= test_gpu_u64();
  ret &= test_gpu_s32();
  ret &= test_gpu_s64();

  return !(ret == true);
}

