
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include <iostream>
#include <random>
#include <hc_am.hpp>
#include <hc.hpp>

[[hc]] void setC(int a, int b, int* c, int idx) {
  c[idx] = a + b;
}

[[hc]] int getB(int* a, int* b, int* c, int idx) {
  return b[idx];
}

[[hc]] int getA(int* a, int* b, int* c, int idx) {
  return a[idx];
}

bool test() {
  // define inputs and output
  const int vecSize = 2048;

  int* data1 = (int*) malloc(vecSize * sizeof(int));
  int* data2 = (int*) malloc(vecSize * sizeof(int));
  int* data3 = (int*) malloc(vecSize * sizeof(int));

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist(0, 255);
  for (int i = 0; i < vecSize; ++i) {
    data1[i] = int_dist(rd);
    data2[i] = int_dist(rd);
    data3[i] = 0;
  }

  auto acc = hc::accelerator();
  int* data1_d = (int*) hc::am_alloc(vecSize * sizeof(int), acc, 0);
  int* data2_d = (int*) hc::am_alloc(vecSize * sizeof(int), acc, 0);
  int* data3_d = (int*) hc::am_alloc(vecSize * sizeof(int), acc, 0);

  hc::accelerator_view av = acc.get_default_view();
  av.copy(data1, data1_d, vecSize * sizeof(int));
  av.copy(data2, data2_d, vecSize * sizeof(int));
  av.copy(data3, data3_d, vecSize * sizeof(int));

  auto k = [=](hc::index<1> idx) [[hc]] {
    int a = getA(data1_d, data2_d, data3_d, idx[0]);
    int b = getB(data1_d, data2_d, data3_d, idx[0]);
    setC(a, b, data3_d, idx[0]);
  };

  // launch kernel
  hc::extent<1> e(vecSize);
  hc::parallel_for_each(e, k);

  av.copy(data1_d, data1, vecSize * sizeof(int));
  av.copy(data2_d, data2, vecSize * sizeof(int));
  av.copy(data3_d, data3, vecSize * sizeof(int));

  // verify
  bool ret = true;
  for(int i = 0; i < vecSize; ++i) {
    //std::cout << data1[i] << " " << data2[i] << " " << data3[i] << "\n";
    ret &= (data3[i] == (data1[i] + data2[i]));
  }

  if (ret) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  hc::am_free(data1_d);
  hc::am_free(data2_d);
  hc::am_free(data3_d);
  free(data1);
  free(data2);
  free(data3);

  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}


