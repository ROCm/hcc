// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

using namespace concurrency;

const int size = 3;

class Point
{
  int _x;
  int _y;
public:
  Point(int x, int y) restrict(amp/*, cpu*/) : _x(x), _y(y) {}
  int get_x() { return _x; }
  int get_y() { return _y; }
  //Point() restrict(amp) {}
};

int main()
{
    unsigned long int sumCPP[size];
    unsigned long int sumCPP1[size];

    // Create C++ AMP objects.
    array_view<unsigned long int, 1> sum(size, sumCPP);
    array_view<unsigned long int, 1> sum1(size, sumCPP1);

    //Point p(3, 4);

#if 1
    parallel_for_each(
        // Define the compute domain, which is the set of threads that are created.
        sum.get_extent(),
        // Define the code to run on each thread on the accelerator.
        [=](index<1> idx) restrict(amp)
    {
       sum[idx] = (unsigned long int)new unsigned int[2];
       sum1[idx] = (unsigned long int)new unsigned int;

    }
    );
#endif

   for (int i = 0; i < size; i++)
   {
     //Point *p = (Point *)sum[i];
     unsigned int *p = (unsigned int*)sum[i];
     printf("Value of addr %p is %u\n", (void*)p, *p);
   }

   printf("====\n");

   for (int i = 0; i < size; i++)
   {
     //Point *p = (Point *)sum[i];
     unsigned int *p1 = (unsigned int*)sum1[i];
     printf("Value of addr %p is %u\n", (void*)p1, *p1);
   }

  return 0;
}
