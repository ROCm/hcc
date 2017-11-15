// XFAIL: *
// RUN: %hc -lhc_am %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_am.hpp>

#include <iostream>

template<typename T, size_t N>
struct Foo {
  T table[N];

  Foo() = default;
};

// partial specialization of Foo<T, 1>
template<typename T>
struct Foo<T, 1> {
  T table[1];

  Foo() = default;

  __attribute__((annotate("user_deserialize")))
  Foo(T x0) [[cpu]][[hc]] {
    table[0] = x0;
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    s.Append(sizeof(T), &table[0]);
  }
};

// partial specialization of Foo<T, 2>
template<typename T>
struct Foo<T, 2> {
  T table[2];

  Foo() = default;

  __attribute__((annotate("user_deserialize")))
  Foo(T x0, T x1) [[cpu]][[hc]] {
    table[0] = x0;
    table[1] = x1;
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    s.Append(sizeof(T), &table[0]);
    s.Append(sizeof(T), &table[1]);
  }
};

// partial specialization of Foo<T, 3>
template<typename T>
struct Foo<T, 3> {
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

// partial specialization of Foo<T, 4>
template<typename T>
struct Foo<T, 4> {
  T table[4];

  Foo() = default;

  __attribute__((annotate("user_deserialize")))
  Foo(T x0, T x1, T x2, T x3) [[cpu]][[hc]] {
    table[0] = x0;
    table[1] = x1;
    table[2] = x2;
    table[3] = x3;
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    s.Append(sizeof(T), &table[0]);
    s.Append(sizeof(T), &table[1]);
    s.Append(sizeof(T), &table[2]);
    s.Append(sizeof(T), &table[3]);
  }
};

// partial specialization of Foo<T, 5>
template<typename T>
struct Foo<T, 5> {
  T table[5];

  Foo() = default;

  __attribute__((annotate("user_deserialize")))
  Foo(T x0, T x1, T x2, T x3, T x4) [[cpu]][[hc]] {
    table[0] = x0;
    table[1] = x1;
    table[2] = x2;
    table[3] = x3;
    table[4] = x4;
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    s.Append(sizeof(T), &table[0]);
    s.Append(sizeof(T), &table[1]);
    s.Append(sizeof(T), &table[2]);
    s.Append(sizeof(T), &table[3]);
    s.Append(sizeof(T), &table[4]);
  }
};

template<typename T, size_t N>
bool test() {
  using namespace hc;

  Foo<T, N> f;
  for (size_t i = 0; i < N; ++i) {
    f.table[i] = T(i);
  }

  auto acc = accelerator();
  accelerator_view av = acc.get_default_view();

  T data[N] { 0 };
  T* data_d = (T*) am_alloc(N * sizeof(T), acc, 0);

  av.copy(data, data_d, N * sizeof(T));

  parallel_for_each(extent<1>(N), [=](index<1> idx) [[hc]] {
                      data_d[idx[0]] = f.table[idx[0]] + T(999);
                    });

  av.copy(data_d, data, N * sizeof(T));

  bool ret = true;
  for (int i = 0; i < N; ++i) {
    ret &= (data[i] == (i + T(999)));
  }

  am_free(data_d);

  return ret;
}

int main() {
  bool ret = true;

  // XXX the test would cause soft hang now
  // explicitly disable the test for now
#if 0
  ret &= test<int, 1>();
  ret &= test<int, 2>();
  ret &= test<int, 3>();
  ret &= test<int, 4>();
  ret &= test<int, 5>();

  ret &= test<unsigned, 1>();
  ret &= test<unsigned, 2>();
  ret &= test<unsigned, 3>();
  ret &= test<unsigned, 4>();
  ret &= test<unsigned, 5>();

  ret &= test<float, 1>();
  ret &= test<float, 2>();
  ret &= test<float, 3>();
  ret &= test<float, 4>();
  ret &= test<float, 5>();

  ret &= test<double, 1>();
  ret &= test<double, 2>();
  ret &= test<double, 3>();
  ret &= test<double, 4>();
  ret &= test<double, 5>();

  return !(ret == true);
#else
  return !(false == true);
#endif
}
