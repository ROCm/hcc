// RUN: %amp_device -D__GPU__=1 %s -m32 -emit-llvm -c -S -O3 -o %t.ll 
// RUN: mkdir -p %t
// RUN: %llc -march=c -o %t/kernel.cl < %t.ll
// RUN: pushd %t 
// RUN: objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o
// RUN: popd
// RUN: %cxxamp %t/kernel.o %s %link -o %t.out && %t.out

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
