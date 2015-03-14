// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>
#include <cstdlib>

#define VECTOR_SIZE (1024)

class POD {
public:
  int getFoo() restrict(cpu,amp) { return foo; }
  int getBar() restrict(cpu,amp) { return bar; }
  int getFooCrossBar() restrict(cpu,amp) { return foo * bar; }
  void setFoo(int f) restrict(cpu) { foo = f; }
  void setBar(int b) restrict(cpu) { bar = b; }
private:
  int foo;
  int bar;
};

int main() {
  using namespace Concurrency;

  int table[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    table[i] = i;
  }

  POD p;
  p.setFoo(rand() % 15 + 1);
  p.setBar(rand() % 15 + 1);

  extent<1> ex(VECTOR_SIZE);
  parallel_for_each(ex, [&](index<1> idx) restrict(amp) {
    // capture array type, and POD type by reference
    // use member function to access POD type
    table[idx[0]] *= (p.getFoo() * p.getBar());
  });

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (table[i] != i * (p.getFooCrossBar())) {
      std::cout << "Failed at " << i << std::endl;
      return 1;
    }
  }

  std::cout << "Passed" << std::endl;
  return 0;
}

