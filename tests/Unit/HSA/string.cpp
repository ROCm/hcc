// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <amp.h>

using namespace concurrency;

class List {
public:
  List() {
    strings[0] = "0";
    strings[1] = "1";
    strings[2] = "2";
    strings[3] = "3";
  }
  char* strings[4];
};

int main() {

  List l;
  int sum_gpu = 0;
  int sum_cpu = 0;

  // test on GPU
  parallel_for_each(extent<1>(1),[=,&l,&sum_gpu](index<1> i) restrict(amp) {
    for (int j = 0; j < 4; j++) {
      sum_gpu+=l.strings[j][0];
    }
  });

  // test on CPU
  {
    for (int j = 0; j < 4; j++) {
      sum_cpu+=l.strings[j][0];
    }
  }

  // verify
  int error = sum_cpu - sum_gpu;
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return error != 0;
}

