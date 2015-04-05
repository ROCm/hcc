// RUN: %cxxamp %s -o %t.out && %t.out
#include <vector>
#include <amp.h>
#include <iostream>
using namespace concurrency;

int CopyArray() {
    std::vector<int> va{1, 2, 3, 4, 5};
    std::vector<int> vb{6, 7, 8, 9, 10};
    std::vector<int> vsum(5);
    int* a = (int*)va.data();
    int* b = (int*)vb.data();
    int* sum = (int*)vsum.data();

    parallel_for_each(
        extent<1>(va.size()), 
        [=](index<1> idx) restrict(auto)
        {
            sum[idx[0]] = a[idx[0]] + b[idx[0]];
        }
    );

    // verify
    for (int i = 0; i < 5; i++) {
        if (vsum[i] != va[i] + vb[i]) {
          return 1;
        }
    }
    return 0;
}

int main(){ 
    return CopyArray();
}

