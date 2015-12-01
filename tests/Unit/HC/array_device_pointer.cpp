// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <vector>

// this test case checks:
// - hc::array::accelerator_pointer() : obtain device memory pointer from an array instance
// - hc::array(int, void*) : construct another hc::array from a given device memory pointer
// this is the 1D case, further commits should provide 2D and 3D test cases
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
  hc::array<int, 1> array3(N, array1_devptr);

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
  ret &= (result1.size() == result2.size() == result3.size());
  for (int i = 0; i < result1.size(); ++i) {
    ret &= (result1[i] == -i);
    ret &= (result1[i] == result2[i]);
    ret &= (result1[i] == result3[i]);
  }

  return !(ret == true);
}

int main() {
  bool ret = true;

  ret &= test1D<16>();
  ret &= test1D<1024>();
  ret &= test1D<256 * 1024>();

  return !(ret == true);
}

