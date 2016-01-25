// RUN: %hc -Xclang -fauto-auto %s -o %t.out && %t.out

#include <vector>
#include <amp.h>
#include <iostream>

using namespace concurrency;

bool CopyArray() {
    std::vector<int> va{1, 2, 3, 4, 5};
    std::vector<int> vb{6, 7, 8, 9, 10};

    array<int, 1> a(va.size(), va.data());
    array<int, 1> b(vb.size(), vb.data());
    array<int, 1> c(va.size());

    parallel_for_each(
        extent<1>(va.size()), 
        [&](index<1> idx) 
        {
            c(idx) = a(idx) + b(idx);
        }
    );

    std::vector<int> vsum = c;

    // verify
    for (int i = 0; i < 5; i++) {
        if (vsum[i] != va[i] + vb[i]) {
          return false;
        }
    }
    return true;
}

int main() {
  bool ret = true;

  ret &= CopyArray();

  return !(ret == true);
}

