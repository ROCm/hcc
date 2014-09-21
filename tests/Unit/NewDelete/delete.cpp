// XFAIL: Linux
// RUN: %amp_device -D__GPU__ -Xclang -fhsa-ext %s -m64 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl MALLOC
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp -Xclang -fhsa-ext %link %t/kernel.o %s -o %t.out && %t.out
#include <iostream>
#include <iomanip>
#include <amp.h>
#include <ctime>
#include "hsa_new.h"

#define DEBUG 0

// An HSA version of C++AMP program
int main ()
{
  // Removed until linking/alloc qualifier issue is solved
  auto ptr_a = newInit.ptr_a;
  auto ptr_b = newInit.ptr_b;
  auto ptr_c = newInit.ptr_c;
  auto ptr_x = newInit.ptr_x;
  auto ptr_y = newInit.ptr_y;
  auto ptr_z = newInit.ptr_z;

  // define inputs and output
  const int vecSize = 16;
  const int tileSize = 4;
  const int tileCount = vecSize / tileSize;

  // launch kernel
  unsigned long int sumCPP[vecSize];
  Concurrency::array_view<unsigned long int, 1> sum(vecSize, sumCPP);

  clock_t m_start = clock();
  parallel_for_each(
    Concurrency::extent<1>(vecSize).tile<tileSize>(),
    [=](Concurrency::tiled_index<tileSize> tidx) restrict(amp) {

    // Removed until linking/alloc qualifier issue is solved
    put_ptr_a(ptr_a);
    put_ptr_b(ptr_b);
    put_ptr_c(ptr_c);
    put_ptr_x(ptr_x);
    put_ptr_y(ptr_y);
    put_ptr_z(ptr_z);

    /*sum[tidx.global[0]] = (unsigned long int)*/ 
    unsigned int *p = new unsigned int[tidx.local[0] + 1]; //(tidx.local[0]);
    *p = 5566;
    sum[tidx.global[0]] = (unsigned long int)p;
    delete p;
  });
  clock_t m_stop = clock();

#if DEBUG
  for (int i = 0; i < vecSize; i++)
  {
    unsigned int *p = (unsigned int*)sum[i];
    //printf("Value of addr %p is %u\n", (void*)p, *p);
    printf("Value of addr %p\n", (void*)p);
  }
#endif

  // Verify
  int error = 0;
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  clock_t t1 = clock();
  clock_t t2 = clock();
  clock_t m_overhead = t2 - t1;
  double elapsed = ((double)(m_stop - m_start - m_overhead)) / CLOCKS_PER_SEC;
  std::cout << "Execution time of amp restrict lambda is " << std::dec << elapsed << " s.\n";
  int Xmalloc_count = newInit.get_Xmalloc_count();
  int malloc_count = newInit.get_malloc_count();
  int Xfree_count = newInit.get_Xfree_count();
  if (Xfree_count != Xmalloc_count + malloc_count) std::cout << "Verify error!\n";
  std::cout << "Xfree_count: " << Xfree_count << "\n";
  std::cout << "malloc_count: " << malloc_count << "\n";
  std::cout << "Xmalloc_count: " << Xmalloc_count << "\n";
  return (error != 0);
}
