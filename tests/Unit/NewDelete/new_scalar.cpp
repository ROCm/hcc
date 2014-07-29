// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

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
       sum[idx] = (unsigned long int)new unsigned int;
    }
    );

   for (int i = 0; i < size; i++)
   {
     unsigned int *p = (unsigned int*)sum[i];
     printf("Value of addr %p is %u\n", (void*)p, *p);
   }

    parallel_for_each(
        // Define the compute domain, which is the set of threads that are created.
        sum.get_extent(),
        // Define the code to run on each thread on the accelerator.
        [=](index<1> idx) restrict(amp)
    {
       sum[idx] = (unsigned long int)new unsigned int(7);
    }
    );

   for (int i = 0; i < size; i++)
   {
     unsigned int *p1 = (unsigned int*)sum[i];
     printf("Value of addr %p is %u\n", (void*)p1, *p1);
   }

  return 0;
}
