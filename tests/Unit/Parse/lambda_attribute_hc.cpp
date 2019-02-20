// RUN: %cxxamp %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

#define SIZE (16)

int main() {

  bool ret = true;
  using namespace hc;

  array_view<int, 1> av(SIZE);

  {
    // case 1: placed between parameter list and left bracket
    auto k1 = [=] (const hc::index<1>& idx) __attribute__((hc)) {
      av[idx] = idx[0];
    };
  
    parallel_for_each(extent<1>(SIZE), k1);
  
    for (int i = 0; i < SIZE; ++i) {
      if (av[i] != i) {
        ret = false;
        break;
      }
    }
  }

  {
    // case 2: placed between lambda introducer and parameter list
    auto k2 = [=] __attribute__((hc)) (const hc::index<1>& idx) {
      av[idx] = idx[0];
    };
  
    parallel_for_each(extent<1>(SIZE), k2);
  
    for (int i = 0; i < SIZE; ++i) {
      if (av[i] != i) {
        ret = false;
        break;
      }
    }
  }

  {
    // case 3: placed in front of lambda introducer
    auto k3 = __attribute__((hc)) [=] (const hc::index<1>& idx) {
      av[idx] = idx[0];
    };
  
    parallel_for_each(extent<1>(SIZE), k3);
  
    for (int i = 0; i < SIZE; ++i) {
      if (av[i] != i) {
        ret = false;
        break;
      }
    }
  }

  return !(ret == true);
}

