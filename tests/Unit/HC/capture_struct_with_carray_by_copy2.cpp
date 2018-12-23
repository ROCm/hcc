// RUN: %hc  %s -o %t.out && %t.out
#include <hc/hc.hpp>
#include <hc/hc_am.hpp>

template<typename T>
struct Foo {
  T table[3];
};

template<typename T>
bool test() {
  using namespace hc;

  Foo<T> f;
  f.table[0] = T(0);
  f.table[1] = T(1);
  f.table[2] = T(2);

  auto acc = accelerator();
  accelerator_view av = acc.get_default_view();

  T data[3] { 0 };
  T* data_d = (T*) am_alloc(3 * sizeof(T), acc, 0);

  av.copy(data, data_d, 3 * sizeof(T));

  parallel_for_each(extent<1>(3), [=](index<1> idx) [[hc]] {
                      data_d[idx[0]] = f.table[idx[0]] + T(999);
                    });

  av.copy(data_d, data, 3 * sizeof(T));

  bool ret = true;
  for (int i = 0; i < 3; ++i) {
    ret &= (data[i] == (i + 999));
  }

  am_free(data_d);

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<int>();
  ret &= test<unsigned>();
  ret &= test<float>();
  ret &= test<double>();

  return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}
