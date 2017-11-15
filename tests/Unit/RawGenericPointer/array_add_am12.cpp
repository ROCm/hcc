// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include <iostream>
#include <random>
#include <type_traits>

#include <hc_am.hpp>
#include <hc.hpp>

template<typename T>
[[hc]] void kfunc(T* a, T* b, T* c, int idx) {
  c[idx] = a[idx] + b[idx];
}

template<typename T>
bool test() {
  // define inputs and output
  const int vecSize = 2048;

  T* data1 = (T*) malloc(vecSize * sizeof(T));
  T* data2 = (T*) malloc(vecSize * sizeof(T));
  T* data3 = (T*) malloc(vecSize * sizeof(T));

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist(0, 255);
  std::uniform_real_distribution<float>  real_dist(0, 255);
  for (int i = 0; i < vecSize; ++i) {
    if (std::is_integral<T>::value) {
      data1[i] = int_dist(rd);
      data2[i] = int_dist(rd);
      data3[i] = T();
    } else if (std::is_floating_point<T>::value) {
      data1[i] = real_dist(rd);
      data2[i] = real_dist(rd);
      data3[i] = T();
    }
  }

  auto acc = hc::accelerator();
  T* data1_d = (T*) hc::am_alloc(vecSize * sizeof(T), acc, 0);
  T* data2_d = (T*) hc::am_alloc(vecSize * sizeof(T), acc, 0);
  T* data3_d = (T*) hc::am_alloc(vecSize * sizeof(T), acc, 0);

  hc::accelerator_view av = acc.get_default_view();
  av.copy(data1, data1_d, vecSize * sizeof(T));
  av.copy(data2, data2_d, vecSize * sizeof(T));
  av.copy(data3, data3_d, vecSize * sizeof(T));

  auto k = [=](hc::index<1> idx) [[hc]] {
    kfunc(data1_d, data2_d, data3_d, idx[0]);
  };

  // launch kernel
  hc::extent<1> e(vecSize);
  hc::parallel_for_each(e, k);

  av.copy(data1_d, data1, vecSize * sizeof(T));
  av.copy(data2_d, data2, vecSize * sizeof(T));
  av.copy(data3_d, data3, vecSize * sizeof(T));

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

  ret &= test<int>();
  ret &= test<unsigned>();
  ret &= test<long>();
  ret &= test<short>();

  ret &= test<float>();
  ret &= test<double>();

  return !(ret == true);
}


