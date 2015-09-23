// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

#include <iostream>

#define SIZE (16)

using namespace concurrency;

// test supply a template class with operator() to parallel_for_each
template<typename _Tp, size_t N>
class prog {
  _Tp (&input)[N];

public:
  prog(_Tp (&t)[N]) : input(t) {
  }

  void operator() (index<1>& idx) restrict(amp) {
    input[idx[0]] = idx[0];
  }

  void run() {
    parallel_for_each(extent<1>(N), *this);
  }

  // verify output
  bool test() {
    bool ret = true;
    for (int i = 0; i < N; ++i) {
      if (input[i] != i) {
        ret = false;
        break;
      }
    }
    return true;
  }
};

int main() {
  bool ret = true;
 
  // prepare test data
  int input_int[SIZE] { 0 };
  unsigned input_unsigned[SIZE] { 0 };
  float input_float[SIZE] { 0 };
  double input_double[SIZE] { 0 };

  // launch kernel
  prog<int, SIZE>      p1(input_int);
  p1.run();
  prog<unsigned, SIZE> p2(input_unsigned);
  p2.run();
  prog<float, SIZE>    p3(input_float);
  p3.run();
  prog<double, SIZE>   p4(input_double);
  p4.run();

  // check result
  ret &= p1.test();
  ret &= p2.test();
  ret &= p3.test();
  ret &= p4.test();

  return !(ret == true);
}

