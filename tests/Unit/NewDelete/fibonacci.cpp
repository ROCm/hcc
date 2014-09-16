// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <iomanip>
#include <amp.h>
#include <ctime>
#include "hsa_new.h"

#define DEBUG 1

// An HSA version of C++AMP program
int main ()
{
  // Removed until linking/alloc qualifier issue is solved
  auto ptr_a = newInit.ptr_a;
  auto ptr_b = newInit.ptr_b;
  auto ptr_c = newInit.ptr_c;

  // define inputs and output
  const int vecSize = 16;
  const int tileSize = 4;
  const int tileCount = vecSize / tileSize;

  // launch kernel
  unsigned long int sumCPP[vecSize];
  Concurrency::array_view<unsigned long int, 1> sum(vecSize, sumCPP);

  clock_t m_start = clock();
  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    // Removed until linking/alloc qualifier issue is solved
    put_ptr_a(ptr_a);
    put_ptr_b(ptr_b);
    put_ptr_c(ptr_c);

    int tid = idx[0];
    int *fib = new int[(tid < 2) ? 2 : tid + 1];

    fib[0] = 0;
    fib[1] = 1;

    for (int i = 2; i <= tid; ++i)
    {
      fib[i] = fib[i-1] + fib[i-2];
    }

    sum[idx[0]] = fib[tid];
    delete[] fib;
  });
  clock_t m_stop = clock();

#if DEBUG
  for (int i = 0; i < vecSize; i++)
  {
    printf("Fib[n] is %lu\n", sum[i]);
  }
#endif

  // Verify
  int *fibh = new int[vecSize + 1];

  fibh[0] = 0;
  fibh[1] = 1;

  for (int i = 2; i < vecSize; i++)
  {
    fibh[i] = fibh[i-1] + fibh[i-2];
  }

  for (int i = 0; i < vecSize; i++)
  {
    if (fibh[i] != sum[i]) {
      std::cout << "Verify failed!\n";
      return 1;
    }
  }

  delete[] fibh;

  std::cout << "Verify success!\n";
  return 0;
}

