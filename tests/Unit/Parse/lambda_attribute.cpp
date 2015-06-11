// RUN: %cxxamp %s -o %t.out && %t.out

#include <amp.h>

#include <iostream>

#define SIZE (16)

int main() {

  bool ret = true;
  using namespace concurrency;

  array_view<int, 1> av(SIZE);

  {
    // case 1: placed between parameter list and left bracket
    auto k1 = [=] (const index<1>& idx) __attribute__((amp)) {
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
    auto k2 = [=] __attribute__((amp)) (const index<1>& idx) {
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
    auto k3 = __attribute__((amp)) [=] (const index<1>& idx) {
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

