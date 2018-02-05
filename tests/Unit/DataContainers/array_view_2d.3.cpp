// RUN: %cxxamp %s -o %t.out && %t.out

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
