// XFAIL: *
// RUN: %hc -lhc_am %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_am.hpp>

#include <iostream>
#include <type_traits>

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

// Bar extends Foo
// since it does not add any new member variables
// we don't supply any user-defined deserializer / serailizer
// and just reuse ones from Foo
template<typename T, size_t N>
struct Bar : Foo<T, N> {

  Bar() = default;

  // We want none of these custom ctor be invoked when we pass instances of Bar to GPU
  template<size_t M = N, typename std::enable_if<M == 1>::type>
  Bar(T x0) {
    this->table[0] = x0 + 123;
  }

  template<size_t M = N, typename std::enable_if<M == 2>::type>
  Bar(T x0, T x1) {
    this->table[0] = x0 + 123;
    this->table[1] = x1 + 123;
  }

  template<size_t M = N, typename std::enable_if<M == 3>::type>
  Bar(T x0, T x1, T x2) {
    this->table[0] = x0 + 123;
    this->table[1] = x1 + 123;
    this->table[2] = x2 + 123;
  }

  template<size_t M = N, typename std::enable_if<M == 4>::type>
  Bar(T x0, T x1, T x2, T x3) {
    this->table[0] = x0 + 123;
    this->table[1] = x1 + 123;
    this->table[2] = x2 + 123;
    this->table[3] = x3 + 123;
  }

  template<size_t M = N, typename std::enable_if<M == 5>::type>
  Bar(T x0, T x1, T x2, T x3, T x4) {
    this->table[0] = x0 + 123;
    this->table[1] = x1 + 123;
    this->table[2] = x2 + 123;
    this->table[3] = x3 + 123;
    this->table[4] = x4 + 123;
  }
};

template<typename T, size_t N, typename U>
bool test() {
  using namespace hc;

  U f;
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
 ret &= test<int, 1, Foo<int, 1> >();
  ret &= test<int, 2, Foo<int, 2> >();
  ret &= test<int, 3, Foo<int, 3> >();
  ret &= test<int, 4, Foo<int, 4> >();
  ret &= test<int, 5, Foo<int, 5> >();

  ret &= test<int, 1, Bar<int, 1> >();
  ret &= test<int, 2, Bar<int, 2> >();
  ret &= test<int, 3, Bar<int, 3> >();
  ret &= test<int, 4, Bar<int, 4> >();
  ret &= test<int, 5, Bar<int, 5> >();

  ret &= test<unsigned, 1, Foo<unsigned, 1> >();
  ret &= test<unsigned, 2, Foo<unsigned, 2> >();
  ret &= test<unsigned, 3, Foo<unsigned, 3> >();
  ret &= test<unsigned, 4, Foo<unsigned, 4> >();
  ret &= test<unsigned, 5, Foo<unsigned, 5> >();

  ret &= test<unsigned, 1, Bar<unsigned, 1> >();
  ret &= test<unsigned, 2, Bar<unsigned, 2> >();
  ret &= test<unsigned, 3, Bar<unsigned, 3> >();
  ret &= test<unsigned, 4, Bar<unsigned, 4> >();
  ret &= test<unsigned, 5, Bar<unsigned, 5> >();

  ret &= test<float, 1, Foo<float, 1> >();
  ret &= test<float, 2, Foo<float, 2> >();
  ret &= test<float, 3, Foo<float, 3> >();
  ret &= test<float, 4, Foo<float, 4> >();
  ret &= test<float, 5, Foo<float, 5> >();

  ret &= test<float, 1, Bar<float, 1> >();
  ret &= test<float, 2, Bar<float, 2> >();
  ret &= test<float, 3, Bar<float, 3> >();
  ret &= test<float, 4, Bar<float, 4> >();
  ret &= test<float, 5, Bar<float, 5> >();

  ret &= test<double, 1, Foo<double, 1> >();
  ret &= test<double, 2, Foo<double, 2> >();
  ret &= test<double, 3, Foo<double, 3> >();
  ret &= test<double, 4, Foo<double, 4> >();
  ret &= test<double, 5, Foo<double, 5> >();

  ret &= test<double, 1, Bar<double, 1> >();
  ret &= test<double, 2, Bar<double, 2> >();
  ret &= test<double, 3, Bar<double, 3> >();
  ret &= test<double, 4, Bar<double, 4> >();
  ret &= test<double, 5, Bar<double, 5> >();

  return !(ret == true);
#else
  return !(false == true);
#endif
}
