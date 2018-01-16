// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <string>
#include <iostream>


template <unsigned int x, unsigned int y>
bool test2D() {

  hc::array_view<unsigned int, 1> data(x * y);
  hc::extent<2> e(y,x);
  hc::parallel_for_each(e, [=](hc::index<2> i) [[hc]] {
    unsigned int flat_id = i[1] + i[0] * x; 
    data[flat_id] = flat_id;
  });

  int errors = 0;
  for (int i = 0; i < x*y; ++i) {
    if (data[i] != i) {
      ++errors;
      std::cerr << "Error data[" << i << "], expected " << i << ", actual " << data[i] << std::endl;
    }
  }

  return errors == 0;
}


int main() {
  bool pass = true;
  pass &= test2D<7, 64>();
  pass &= test2D<63, 64>();
  pass &= test2D<100, 4>();
  pass &= test2D<1024, 1>();
  pass &= test2D<1, 1024>();
  return pass==true?0:1;
}
