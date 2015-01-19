// XFAIL: Linux
// RUN: %cxxamp -I/opt/hsa/include/ -Wl,--rpath=/opt/hsa/lib -lhsa-runtime64 -L/opt/hsa/lib -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <iomanip>
#include <amp.h>
#include "hsa_new.h"

#define DEBUG 1

// An HSA version of C++AMP program
int main ()
{
  // Removed until linking/alloc qualifier issue is solved
  auto XmallocFlag = newInit.XmallocFlag;
  auto mallocFlag = newInit.mallocFlag;
  auto ptr_a = newInit.ptr_a; // pointer to Xmalloc syscall numbers
  auto ptr_b = newInit.ptr_b; // pointer to Xmalloc syscall parameters
  auto ptr_c = newInit.ptr_c; // pointer to Xmalloc test results
  auto ptr_x = newInit.ptr_x; // pointer to Xfree/free/malloc syscall numbers
  auto ptr_y = newInit.ptr_y; // pointer to Xfree/free/malloc syscall parameters
  auto ptr_z = newInit.ptr_z; // pointer to Xfree/free/malloc test results

  // define inputs and output
  const int vecSize = 16;
  const int tileSize = 4;
  const int tileCount = vecSize / tileSize;

  // launch kernel
  unsigned long int sumCPP[vecSize];
  Concurrency::array_view<unsigned long int, 1> sum(vecSize, sumCPP);

  parallel_for_each(
    Concurrency::extent<1>(vecSize).tile<tileSize>(),
    [=](Concurrency::tiled_index<tileSize> tidx) restrict(amp) {

    // Removed until linking/alloc qualifier issue is solved
    putXmallocFlag(XmallocFlag);
    putMallocFlag(mallocFlag);
    put_ptr_a(ptr_a);
    put_ptr_b(ptr_b);
    put_ptr_c(ptr_c);
    put_ptr_x(ptr_x);
    put_ptr_y(ptr_y);
    put_ptr_z(ptr_z);

    int local = tidx.local[0];

    unsigned int *p = NULL; 
    p = new unsigned int(local * 2);
    sum[tidx.global[0]] = (unsigned long int)p;
    delete p;
  });

#if DEBUG
  for (int i = 0; i < vecSize; i++)
  {
    unsigned int *p = (unsigned int*)sum[i];
    printf("Value of addr %p\n", (void*)p);
  }
#endif

  // Verify
  int error = 0;
  if ((newInit.get_Xmalloc_count() == 0) || (newInit.get_malloc_count() != 0)) {
    error = 1;
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error != 0);
  //return 1; // FIXME tempoary make this test case fail no matter what
}
