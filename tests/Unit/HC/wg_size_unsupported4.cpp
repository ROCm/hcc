// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <string>
#include <iostream>

int main() {
  bool pass = false;

  try  {
    hc::parallel_for_each(hc::extent<3>(16,16,16).tile(32,1,1), [](hc::tiled_index<3> i) [[hc]] {});
  } catch (Kalmar::runtime_exception e) {
    std::string err_str = e.what();
    pass = err_str.find("The extent of the tile") != std::string::npos &&
    err_str.find("exceeds the compute grid extent") != std::string::npos;
  }

  try  {
    hc::parallel_for_each(hc::extent<3>(16,16,16).tile(1,32,1), [](hc::tiled_index<3> i) [[hc]] {});
  } catch (Kalmar::runtime_exception e) {
    std::string err_str = e.what();
    pass = err_str.find("The extent of the tile") != std::string::npos &&
    err_str.find("exceeds the compute grid extent") != std::string::npos;
  }

  try  {
    hc::parallel_for_each(hc::extent<3>(16,16,16).tile(1,1,32), [](hc::tiled_index<3> i) [[hc]] {});
  } catch (Kalmar::runtime_exception e) {
    std::string err_str = e.what();
    pass = err_str.find("The extent of the tile") != std::string::npos &&
    err_str.find("exceeds the compute grid extent") != std::string::npos;
  }

  return pass==true?0:1;
}
