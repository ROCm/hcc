// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include "point.h"

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
     sum[idx] = (unsigned long int)new Point[2]();
  }
  );

#if TEST_DEBUG
  for (int i = 0; i < size; i++)
  {
    Point *p = (Point *)sum[i];
    printf("Value of addr %p is %d & %d, addr %p is %d & %d\n", 
      (void*)p, p->get_x(), p->get_y(), 
      (void*)(p + 1), (p + 1)->get_x(), (p + 1)->get_y());
  }
#endif

  // Verify
  int error = 0;
  for(int i = 0; i < size; i++) {
    Point *p = (Point*)sum[i];
    Point pt;
    error += (abs(p->get_x() - pt.get_x()) + abs(p->get_y() - pt.get_y())
               + abs((p + 1)->get_x() - pt.get_x()) + abs((p + 1)->get_y() - pt.get_y()));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return (error != 0);
}
