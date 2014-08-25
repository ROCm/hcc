// RUN: %cxxamp %s -o %t.out && %t.out

#include <iostream> 
#include <amp.h> 
using namespace concurrency; 
int main() 
{
  int v[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  extent<2> e(5, 2);
  {
    array_view<int, 2> av(e, v); 
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
