// RUN: %amp_device -D__GPU__=1 %s -m32 -emit-llvm -c -S -O3 -o %t.ll 
// RUN: mkdir -p %t
// RUN: %llc -march=c -o %t/kernel.cl < %t.ll
// RUN: pushd %t 
// RUN: objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o
// RUN: popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include <iostream> 
#include <amp.h> 
using namespace concurrency; 
int main() 
{
  int v[11] = {'G', 'd', 'k', 'k', 'n', 31, 'v', 'n', 'q', 'k', 'c'};

  array_view<int> av(11, v); 
  parallel_for_each(av.get_extent(), [=](index<1> idx) restrict(amp) { 
    av[idx] += 1; 
  });

  std::string expected("Hello world");
  for(unsigned int i = 0; i < av.get_extent().size(); i++) {
    assert(expected[i] == static_cast<char>(av(i)));
    std::cout << static_cast<char>(av(i));
  }
  return 0;
}
