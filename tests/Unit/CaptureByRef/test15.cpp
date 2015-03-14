// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>
#include <cstdlib>

#define VECTOR_SIZE (1024)

class POD {
public:
  int foo;
  int bar;
};

class POD2 : public POD {
public:
  int baz;
};

class POD3 : public POD2 {
public:
  int qux;
};

int main() {
  using namespace Concurrency;

  int table[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    table[i] = i;
  }

  POD3 p;
  p.foo = rand() % 15 + 1;
  p.bar = rand() % 15 + 1;
  p.baz = rand() % 15 + 1;
  p.qux = rand() % 15 + 1;

  extent<1> ex(VECTOR_SIZE);
  parallel_for_each(ex, [&](index<1> idx) restrict(amp) {
    // capture array type, and an inherited type by reference
    table[idx[0]] = (p.foo * p.bar * p.baz * p.qux);
  });

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (table[i] != (p.foo * p.bar * p.baz * p.qux)) {
      std::cout << "Failed at " << i << std::endl;
      return 1;
    }
  }

  std::cout << "Passed" << std::endl;
  return 0;
}

