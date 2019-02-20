// RUN: %cxxamp %s -o %t.out && %t.out
#include <iostream> 
#include <hc.hpp>
#include <vector>
using namespace hc; 
int main() 
{
  std::vector<int> vv(10);
  for (int i = 0; i<10; i++)
    vv[i] = i+2;

  extent<2> e(5, 2);
  {
    array_view<int, 2> av(5, 2, vv.data()); 
    parallel_for_each(av.get_extent(), [=](hc::index<2> idx) [[hc]] { 
	av(idx) -= av.get_extent()[1]; 
    });
    assert(av.get_extent() == e);
    for(unsigned int i = 0; i < av.get_extent()[0]; i++)
      for(unsigned int j = 0; j < av.get_extent()[1]; j++)
	assert(i*2+j == static_cast<char>(av(i, j)));
  }
  return 0;
}
