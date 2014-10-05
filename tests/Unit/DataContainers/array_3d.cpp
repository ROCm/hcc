// RUN: %cxxamp %s -o %t.out && %t.out

#include <iostream> 
#include <amp.h>
#include <vector>
using namespace concurrency; 
int main() 
{
  array<int, 3> a(5, 2, 3);
  int i = 0;
  for(unsigned int i = 0; i < a.extent[0]; i++)
    for(unsigned int j = 0; j < a.extent[1]; j++)
      for(unsigned int k = 0; k < a.extent[2]; k++) {
        a(i, j, k) = i*2*3+j*3+k+1; 
      }

  extent<3> e (5, 2, 3);
  {
    parallel_for_each(e, [&a](index<3> idx) restrict(amp) { 
	a(idx) -= 2; 
	a(idx[0], idx[1], idx[2]) += 1; 
    });
    assert(a.extent == e);
    for(unsigned int i = 0; i < a.extent[0]; i++)
      for(unsigned int j = 0; j < a.extent[1]; j++)
        for(unsigned int k = 0; k < a.extent[2]; k++)
          assert(i*2*3+j*3+k == static_cast<int>(a(i, j, k)));
  }
  return 0;
}
