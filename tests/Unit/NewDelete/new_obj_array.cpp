// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

using namespace concurrency;

const int size = 5;

class Point
{
  int _x;
  int _y;
public:
  Point() restrict(amp) {}
  Point(int x, int y) restrict(amp) : _x(x), _y(y) {}
  int get_x() { return _x; }
  int get_y() { return _y; }
};

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
       sum[idx] = (unsigned long int)new Point[2];
    }
    );

   for (int i = 0; i < size; i++)
   {
     Point *p = (Point *)sum[i];
     printf("Value of addr %p is %d & %d\n", (void*)p, p->get_x(), p->get_y());
   }

  return 0;
}
