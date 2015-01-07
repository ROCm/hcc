// RUN: %cxxamp -shared -fPIC -Wl,-Bsymbolic -DSHARED_LIBRARY %s -o %t.so
// RUN: %cxxamp %t.so %s -o %t.out && %t.out
#ifdef SHARED_LIBRARY
#include <amp.h>
#include <iostream>
using namespace concurrency;

const int size = 5;

void CppAmpMethod() {
    int aCPP[] = {1, 2, 3, 4, 5};
    int bCPP[] = {6, 7, 8, 9, 10};
    int sumCPP[size];
    
    // Create C++ AMP objects.
    array_view<const int, 1> a(size, aCPP);
    array_view<const int, 1> b(size, bCPP);
    array_view<int, 1> sum(size, sumCPP);
    sum.discard_data();

    parallel_for_each( 
        // Define the compute domain, which is the set of threads that are created.
        sum.get_extent(), 
        // Define the code to run on each thread on the accelerator.
        [=](index<1> idx) restrict(amp)
    {
        sum[idx] = a[idx] + b[idx];
    }
    );

    // Print the results. The expected output is "7, 9, 11, 13, 15".
    for (int i = 0; i < size; i++) {
      if (sum[i] != aCPP[i]+bCPP[i])
        exit(1);
      std::cout << sum[i] << "\n";
    }
}
#else
extern void CppAmpMethod();

int main()
{
    CppAmpMethod();
    return 0;
}
#endif
