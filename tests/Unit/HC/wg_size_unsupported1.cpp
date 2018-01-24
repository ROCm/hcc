// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <string>
#include <iostream>

int main() {
  bool pass = false;
  try  {
    // We expect the runtime will fire an exception due to a large work group size
    hc::parallel_for_each(hc::extent<1>(8192).tile(8192), [](hc::tiled_index<1> i) [[hc]] {});
  } catch (Kalmar::runtime_exception e) {
    std::string err_str = e.what();
    pass = err_str.find("The extent of the tile") != std::string::npos &&
    err_str.find("exceeds the device limit") != std::string::npos;
  }
  return pass==true?0:1;
}
