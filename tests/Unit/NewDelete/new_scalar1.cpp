// XFAIL: Linux
// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>

#define TEST_DEBUG 1

using namespace concurrency;

const int size = 5;

int main()
{
  unsigned long int sumCPP[size];

  // Create C++ AMP objects.
  array_view<unsigned long int, 1> sum(size, sumCPP);

  parallel_for_each(
    // Define the compute domain, which is the set of threads that are created.
    sum.get_extent(),
    // Define the code to run on each thread on the accelerator.
    [=](index<1> idx) restrict(amp)
  {
    sum[idx] = (unsigned long int)new unsigned int();
  }
  );

#if TEST_DEBUG
  for (int i = 0; i < size; i++)
  {
    unsigned int *p = (unsigned int*)sum[i];
    printf("Value of addr %p is %u\n", (void*)p, *p);
  }
#endif

  // Verify
  int error = 0;
  for(int i = 0; i < size; i++) {
    unsigned int *p = (unsigned int*)sum[i];
    error += abs(*p);
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return (error != 0);
}
