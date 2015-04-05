// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>
#include <cstdlib>

#define VECTOR_SIZE (1024)

struct POD {
  int foo;
  int bar;
};

struct POD2 {
  int foo;
  int bar;
  int baz;
};

class POD3 {
public:
  int foo;
  int bar;
  int baz;
  int qux;
};

int main() {
  using namespace Concurrency;

  int table[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    table[i] = i;
  }

  POD p;
  p.foo = rand() % 15 + 1;
  p.bar = rand() % 15 + 1;

  POD2 p2;
  p2.foo = rand() % 15 + 1;
  p2.bar = rand() % 15 + 1;
  p2.baz = rand() % 15 + 1;

  POD3 p3;
  p3.foo = rand() % 15 + 1;
  p3.bar = rand() % 15 + 1;
  p3.baz = rand() % 15 + 1;
  p3.qux = rand() % 15 + 1;

  extent<1> ex(VECTOR_SIZE);
  array_view<int, 1> av(ex, table);
  parallel_for_each(av.get_extent(), [&, av](index<1> idx) restrict(amp) {
    // capture multitple POD types by reference
    av[idx] *= ((p.foo + p.bar) + (p2.foo + p2.bar + p2.baz) + (p3.foo + p3.bar + p3.baz + p3.qux));
  });

  av.synchronize();

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (table[i] != i * ((p.foo + p.bar) + (p2.foo + p2.bar + p2.baz) + (p3.foo + p3.bar + p3.baz + p3.qux)) ) {
      std::cout << "Failed at " << i << std::endl;
      return 1;
    }
  }

  std::cout << "Passed" << std::endl;
  return 0;
}

