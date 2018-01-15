
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>

// this test case checks:
// - hc::array::accelerator_pointer() : obtain device memory pointer from an array instance
// - hc::array(int, void*) : construct another hc::array from a given device memory pointer
// - hc::array(int, int, void*) : construct another hc::array from a given device memory pointer
// - hc::array(int, int, int, void*) : construct another hc::array from a given device memory pointer
// - hc::array(extent&, void*) : construct another hc::array from a given device memory pointer

// this is the 1D case
template<int N>
bool test1D() {
  bool ret = true;

  hc::array<int, 1> array1(N);

  // fetch the device pointer of array1
  void* array1_devptr = array1.accelerator_pointer();

  // construct another array based on the pointer
  hc::array<int, 1> array2(N, array1_devptr);

  // execute a kernel, and use the second array
  hc::completion_future fut = parallel_for_each(hc::extent<1>(N), [&](hc::index<1>& idx) [[hc]] {
    array2[idx] = idx[0];
  });

  fut.wait();

  // construct yet another array based on the pointer
  hc::array<int, 1> array3(array1.get_extent(), array1_devptr);

  // execute a kernel, and use the third array
  hc::completion_future fut2 = parallel_for_each(hc::extent<1>(N), [&](hc::index<1>& idx) [[hc]] {
    array3[idx] = -array3[idx];
  });

  // read out the value from the first array
  std::vector<int> result1 = array1;

  // read out the value from the second array
  std::vector<int> result2 = array2;

  // read out the value from the third array
  std::vector<int> result3 = array3;
   
  // verify all three versions are the same
  ret &= (result1.size() == result2.size());
  ret &= (result1.size() == result3.size());
  for (int i = 0; i < result1.size(); ++i) {
    ret &= (result1[i] == -i);
    ret &= (result1[i] == result2[i]);
    ret &= (result1[i] == result3[i]);
  }

  return ret;
}

// this is the 2D case
template<int N, int M>
bool test2D() {
  bool ret = true;

  hc::array<int, 2> array1(N, M);

  // fetch the device pointer of array1
  void* array1_devptr = array1.accelerator_pointer();

  // construct another array based on the pointer
  hc::array<int, 2> array2(N, M, array1_devptr);

  // execute a kernel, and use the second array
  hc::completion_future fut = parallel_for_each(hc::extent<2>(N, M), [&](hc::index<2>& idx) [[hc]] {
    array2[idx] = idx[0] * M + idx[1];
  });

  fut.wait();

  // construct yet another array based on the pointer
  hc::array<int, 2> array3(array1.get_extent(), array1_devptr);

  // execute a kernel, and use the third array
  hc::completion_future fut2 = parallel_for_each(hc::extent<2>(N, M), [&](hc::index<2>& idx) [[hc]] {
    array3[idx] = -array3[idx];
  });

  // read out the value from the first array
  std::vector<int> result1 = array1;

  // read out the value from the second array
  std::vector<int> result2 = array2;

  // read out the value from the third array
  std::vector<int> result3 = array3;

  // verify all three versions are the same
  ret &= (result1.size() == result2.size());
  ret &= (result1.size() == result3.size());
  for (int i = 0; i < result1.size(); ++i) {
    ret &= (result1[i] == -i);
    ret &= (result1[i] == result2[i]);
    ret &= (result1[i] == result3[i]);
  }

  return ret;
}

// this is the 3D case
template<int N, int M, int O>
bool test3D() {
  bool ret = true;

  hc::array<int, 3> array1(N, M, O);

  // fetch the device pointer of array1
  void* array1_devptr = array1.accelerator_pointer();

  // construct another array based on the pointer
  hc::array<int, 3> array2(N, M, O, array1_devptr);

  // execute a kernel, and use the second array
  hc::completion_future fut = parallel_for_each(hc::extent<3>(N, M, O), [&](hc::index<3>& idx) [[hc]] {
    array2[idx] = idx[0] * M * O + idx[1] * O + idx[2];
  });

  fut.wait();

  // construct yet another array based on the pointer
  hc::array<int, 3> array3(array1.get_extent(), array1_devptr);

  // execute a kernel, and use the third array
  hc::completion_future fut2 = parallel_for_each(hc::extent<3>(N, M, O), [&](hc::index<3>& idx) [[hc]] {
    array3[idx] = -array3[idx];
  });

  // read out the value from the first array
  std::vector<int> result1 = array1;

  // read out the value from the second array
  std::vector<int> result2 = array2;

  // read out the value from the third array
  std::vector<int> result3 = array3;

  // verify all three versions are the same
  ret &= (result1.size() == result2.size());
  ret &= (result1.size() == result3.size());
  for (int i = 0; i < result1.size(); ++i) {
    ret &= (result1[i] == -i);
    ret &= (result1[i] == result2[i]);
    ret &= (result1[i] == result3[i]);
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<16>();
  ret &= test1D<1024>();
  ret &= test1D<256 * 1024>();

  ret &= test2D<2, 8>();
  ret &= test2D<16, 64>();
  ret &= test2D<256, 1024>();

  ret &= test3D<2, 4, 8>();
  ret &= test3D<4, 8, 32>();
  ret &= test3D<16, 64, 1024>();

  return !(ret == true);
}

