// RUN: %cxxamp %s -o %t.out && %t.out
#include <iostream> 
#include <amp.h> 
using namespace concurrency;
int main() {
  int v[11] = {0,1,2,3,4,5,6,7,8,9,10};
  int expexted_v[11] = {11,12,13,14,15,16,17,18,19,20,21};
  array_view<int> av(11, v);
  parallel_for_each(av.get_extent(), [=](index<1> idx) restrict(amp) {
    av[idx] +=1 ;
  });

  parallel_for_each(av.get_extent(), [=](index<1> idx) restrict(amp) {
    av[idx] += 10;
  });

  for(int i = 0; i < 11; i++) {
    assert(expexted_v[i] == av(i));
    std::cout<<av[i]<<" ";
  }
  return 0;
}


