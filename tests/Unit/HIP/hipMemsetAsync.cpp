// XFAIL: Linux
// RUN: %hc %s -lhip_runtime -o %t.out && %t.out

#include "hip_runtime.h"


template<typename T, int SIZE>
bool test() {

  hipStream_t stream1;
  hipStreamCreate(&stream1);

  T* data1;

  hipMallocHost((void**)&data1, SIZE*sizeof(T));

  for(int i = 0; i < SIZE; ++i) {
    data1[i] = i;
  }

  hipMemsetAsync(data1, 0, SIZE*sizeof(T), stream1);
  hipStreamSynchronize(stream1);

  bool ret = true;
  for(int i = 0; i < SIZE; ++i) {
    ret &= data1[i] == 0;
  }

  hipStreamDestroy(stream1);


  // test default stream
  for(int i = 0; i < SIZE; ++i) {
    data1[i] = i;
  }

  hipMemsetAsync(data1, 0, SIZE*sizeof(T));
  hipStreamSynchronize();

  for(int i = 0; i < SIZE; ++i) {
    ret &= data1[i] == 0;
  }

  return ret;
}

int main(void) {
  bool ret = true;

  const int SIZE = 1024;
  ret &= test<int, SIZE>();
  ret &= test<float, SIZE>();
  ret &= test<unsigned, SIZE>();
  ret &= test<double, SIZE>();

  return !ret;
}
