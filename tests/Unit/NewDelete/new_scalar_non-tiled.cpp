// XFAIL: Linux, hsa
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <iomanip>
#include <amp.h>
#include <hsa_new.h>

#define DEBUG 1

// An HSA version of C++AMP program
int main ()
{
  // Removed until linking/alloc qualifier issue is solved
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
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    // Removed until linking/alloc qualifier issue is solved
    put_ptr_a(ptr_a);
    put_ptr_b(ptr_b);
    put_ptr_c(ptr_c);
    put_ptr_x(ptr_x);
    put_ptr_y(ptr_y);
    put_ptr_z(ptr_z);

    sum[idx[0]] = (unsigned long int)new unsigned int(idx[0]);
  });

#if DEBUG
  for (int i = 0; i < vecSize; i++)
  {
    unsigned int *p = (unsigned int*)sum[i];
    printf("Value of addr %p is %u\n", (void*)p, *p);
  }
#endif

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    unsigned int *p = (unsigned int*)sum[i];
    error += abs(*p - i);
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  //return (error != 0);
  return 1; // FIXME tempoary make this test case fail no matter what
}

