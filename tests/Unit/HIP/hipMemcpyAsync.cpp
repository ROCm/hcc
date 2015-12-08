// XFAIL:
// RUN: %hc %s -lhip_runtime -o %t.out && %t.out

#include "hip_runtime.h"

struct Foo {
  double y;
  int x;

  Foo &operator=(const int& n) {
    this->x = n;
    this->y = n;
    return *this;
  }

  bool operator==(const Foo& other) {
    return this->x == other.x &&
           this->y == other.y;
  }
};



template<typename T, int SIZE>
bool test() {

  hipStream_t stream1;
  hipStreamCreate(&stream1);

  T* data1;
  T* data2;

  hipMallocHost((void**)&data1, SIZE*sizeof(T));
  hipMallocHost((void**)&data2, SIZE*sizeof(T));

  for(int i = 0; i < SIZE; ++i) {
    data1[i] = i;
    data2[i] = 0;
  }

  hipMemcpyAsync(data2, data1, SIZE*sizeof(T), hipMemcpyHostToDevice, stream1);
  hipStreamSynchronize(stream1);

  bool ret = true;
  for(int i = 0; i < SIZE; ++i) {
    ret &= data1[i] == data2[i];
  }

  hipStreamDestroy(stream1);


  // test default stream
  for(int i = 0; i < SIZE; ++i) {
    data1[i] = i;
    data2[i] = 0;
  }

  hipMemcpyAsync(data2, data1, SIZE*sizeof(T), hipMemcpyHostToDevice);
  hipStreamSynchronize();

  for(int i = 0; i < SIZE; ++i) {
    ret &= data1[i] == data2[i];
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
  ret &= test<struct Foo, SIZE>();

  return !ret;
}
