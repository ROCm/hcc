// RUN: %amp_device -D__GPU__=1 %s -m32 -emit-llvm -c -S -O3 -o %t.ll 
// RUN: mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t 
// RUN: objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o
// RUN: popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include <iostream> 
#include <amp.h>
#include <vector>
using namespace concurrency; 
int main() 
{
  std::vector<int> vv(10);
  for (int i = 0; i<10; i++)
    vv[i] = i+1;

  extent<2> e(5, 2);
  {
    array_view<int, 2> av(5, 2, vv); 
    parallel_for_each(av.get_extent(), [=](index<2> idx) restrict(amp) { 
	av(idx) -= 1; 
	});
    assert(av.get_extent() == e);
    for(unsigned int i = 0; i < av.get_extent()[0]; i++)
      for(unsigned int j = 0; j < av.get_extent()[1]; j++)
	assert(i*2+j == static_cast<char>(av(i, j)));
  }
  return 0;
}
