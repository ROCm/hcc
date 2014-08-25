// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace concurrency;

int main(void) {
  const int vecSize = 100;

  // Alloc & init input data
  array<int, 1> count(vecSize);
  for(unsigned i = 0; i < vecSize; i++) {
    count[i] = 0;
  }

  parallel_for_each(count.extent, [=, &count](index<1> idx) restrict(amp) {
    for(unsigned i = 0; i < vecSize; i++) {
      atomic_fetch_add(&count[i], 1);
    }
  });

  std::vector<accelerator> accs = accelerator::get_all();
  for(unsigned i = 0; i < vecSize && accs.size(); i++) {
      if(count[i] != vecSize) {
        return 1;
      }
  }

  return 0;
}
