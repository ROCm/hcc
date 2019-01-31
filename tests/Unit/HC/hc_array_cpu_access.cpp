// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <cstdlib>
#include <hc.hpp>

int main(int argc, char* argv[]) {
  hc::array<int, 1> a(1);
  auto at = a.get_cpu_access_type();
  int num_catches = 0;
  bool pass = false;
  switch(at) {
    default:
    case hc::access_type_none:
      // e.g. dGPU
      // expect all access to hc::array to
      // generate an exception
      try {
        a[hc::index<1>(0)]++;
      } catch (hc::runtime_exception e) { num_catches++; }

      try {
        a[0]++;
      } catch (hc::runtime_exception e) { num_catches++; }

      try {
        a(0);
      } catch (hc::runtime_exception e) { num_catches++; }
      pass = (num_catches == 3);
      break;
    case hc::access_type_read_write:
      // e.g. HSA compliant APU
      a[hc::index<1>(0)]++;
      a[0]++;
      a(0);
      pass = (num_catches == 0);
      break;
  }
  return pass?0:1;
}
