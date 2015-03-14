// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>
#include <cstdlib>

#define VECTOR_SIZE (1024)

int main() {
  using namespace Concurrency;

  int table[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    table[i] = i;
  }

  int val = rand() % 15 + 1;
  int val2 = rand() % 15 + 1;

  extent<1> ex(VECTOR_SIZE);
  array_view<int, 1> av(ex, table);
  parallel_for_each(av.get_extent(), [&, av](index<1> idx) restrict(amp) {
    // capture multiple scalar types by reference
    av[idx] *= (val + val2);
  });

  av.synchronize();

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (table[i] != i * (val + val2)) {
      std::cout << "Failed at " << i << std::endl;
      return 1;
    }
  }

  std::cout << "Passed" << std::endl;
  return 0;
}

