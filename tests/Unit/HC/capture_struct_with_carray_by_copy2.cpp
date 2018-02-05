// XFAIL: *
// RUN: %hc -lhc_am %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_am.hpp>

template<typename T>
struct Foo {
  T table[3];

  Foo() = default;

  __attribute__((annotate("user_deserialize")))
  Foo(T x0, T x1, T x2) [[cpu]][[hc]] {
    table[0] = x0;
    table[1] = x1;
    table[2] = x2;
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    s.Append(sizeof(T), &table[0]);
    s.Append(sizeof(T), &table[1]);
    s.Append(sizeof(T), &table[2]);
  }
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

  // XXX the test would cause soft hang now
  // explicitly disable the test for now
#if 0
  ret &= test<int>();
  ret &= test<unsigned>();
  ret &= test<float>();
  ret &= test<double>();

  return !(ret == true);
#else
  return !(false == true);
#endif
}
