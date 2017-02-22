
// RUN: %hc -DSTATIC_LIB %s -c -o %T/static_library4.o
// RUN: ar rcs %T/libstatic_library4.a %T/static_library4.o
// RUN: %hc --amdgpu-target=gfx701 --amdgpu-target=gfx801 --amdgpu-target=gfx802 --amdgpu-target=gfx803 %s -L%T -lstatic_library4 -o %t.out && %t.out

#include <cstdio>
#include <hc.hpp>

extern "C" int sum(hc::array_view<int,1>& input);

#ifdef STATIC_LIB

int sum(hc::array_view<int,1>& input) {

  hc::array_view<int,1> s(1);
  s[0]=0;

  hc::parallel_for_each(input.get_extent(), [=](hc::index<1> idx) [[hc]] {
    if (idx[0]==0) {
      int num = input.get_extent()[0];
      for (int i = 0; i < num; i++) {
        s[0]+=input[i];
      }
    }
  }).wait();

  return s[0];
}

#else

int main()
{
  hc::array_view<int,1> av(64);
  for (int i = 0;i < 64; i++)
    av[i] = i;

  int s = sum(av);

 // printf("sum: %d\n",s);

  return !(s==2016);
}

#endif
