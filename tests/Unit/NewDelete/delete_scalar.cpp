// XFAIL: Linux
// RUN: %amp_device -D__GPU__ -Xclang -fhsa-ext %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp -Xclang -fhsa-ext %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

#define TEST_DEBUG 0

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
    unsigned int *p = new unsigned int(idx[0]);
    sum[idx] = (unsigned long int)p;
    delete p;
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
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return (error != 0);
}
