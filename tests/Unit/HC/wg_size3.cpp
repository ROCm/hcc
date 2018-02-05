// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <string>
#include <iostream>


template <unsigned int x, unsigned int y, unsigned int z>
bool test3D() {

  hc::array_view<unsigned int, 1> data(x * y * z);
  hc::extent<3> e(z,y,x);
  hc::parallel_for_each(e, [=](hc::index<3> i) [[hc]] {
    unsigned int flat_id = i[2] + i[1] * x + i[0] * x * y; 
    data[flat_id] = flat_id;
  });

  int errors = 0;
  for (int i = 0; i < x*y*z; ++i) {
    if (data[i] != i) {
      ++errors;
      std::cerr << "Error data[" << i << "], expected " << i << ", actual " << data[i] << std::endl;
    }
  }

  return errors == 0;
}


int main() {
  bool pass = true;
  pass &= test3D<7, 3, 512>();
  pass &= test3D<63, 4, 4>();
  pass &= test3D<100, 4, 4>();
  return pass==true?0:1;
}
