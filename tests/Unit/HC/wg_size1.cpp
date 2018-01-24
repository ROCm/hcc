// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <string>
#include <iostream>


template <unsigned int x>
bool test1D() {

  hc::array_view<unsigned int, 1> data(x);
  hc::extent<1> e(x);
  hc::parallel_for_each(e, [=](hc::index<1> i) [[hc]] {
    unsigned int flat_id = i[0]; 
    data[flat_id] = flat_id;
  });

  int errors = 0;
  for (int i = 0; i < x; ++i) {
    if (data[i] != i) {
      ++errors;
      std::cerr << "Error data[" << i << "], expected " << i << ", actual " << data[i] << std::endl;
    }
  }

  return errors == 0;
}


int main() {
  bool pass = true;
  pass &= test1D<7>();
  pass &= test1D<63>();
  pass &= test1D<1024>();
  return pass==true?0:1;
}
