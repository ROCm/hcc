// XFAIL: Linux
// RUN: %cxxamp %s -o %t.out && %t.out
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
    sum[idx] = (unsigned long int)new Point(idx[0], idx[0] * 2);
  }
  );

#if TEST_DEBUG
  for (int i = 0; i < size; i++)
  {
    Point *p = (Point *)sum[i];
    printf("Value of addr %p is %d & %d\n", (void*)p, p->get_x(), p->get_y());
  }
#endif

  // Verify
  int error = 0;
  for(int i = 0; i < size; i++) {
    Point *p = (Point*)sum[i];
    Point pt(i, i * 2);
    error += (abs(p->get_x() - pt.get_x()) + abs(p->get_y() - pt.get_y()));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return (error != 0);
}
