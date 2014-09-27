// XFAIL: Linux, hsa
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <iomanip>
#include <amp.h>
#include <ctime>
#include <hsa_new.h>

#define DEBUG 0

#define INNER_LOOP 1
#define OUTER_LOOP 0
#define WITH_DELETE 1

#define inner_size 1
#define outer_size 1

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
  const int vecSize = 256;
  const int tileSize = 256;
  const int tileCount = vecSize / tileSize;

  // launch kernel
  unsigned long int sumCPP[vecSize];
  Concurrency::array_view<unsigned long int, 1> sum(vecSize, sumCPP);

  clock_t m_start = clock();
#if OUTER_LOOP
  for (int i = 0; i < outer_size; i++) {
#endif
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

#if INNER_LOOP
      for (int j = 0; j < inner_size; j++) {
#endif
        unsigned long int *p =  new unsigned long int;
        *p = 5566;
        sum[tidx.global[0]] = (unsigned long int)p;
#if WITH_DELETE
        delete p;
#endif
#if INNER_LOOP
      }
#endif
  });
#if OUTER_LOOP
}
#endif
  clock_t m_stop = clock();

#if DEBUG
  for (int i = 0; i < vecSize; i++)
  {
    unsigned int *p = (unsigned int*)sum[i];
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
  std::cout << "Xfree count: " << newInit.get_Xfree_count() 
              << ", Xmalloc count: " << newInit.get_Xmalloc_count()
              << ", malloc count: " << newInit.get_malloc_count() << "\n";
  //return (error != 0);
  return 1; // FIXME tempoary make this test case fail no matter what
}
