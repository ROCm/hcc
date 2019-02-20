// RUN: %cxxamp %s -o %t.out && %t.out
#include <iostream> 
#include <hc.hpp> 
using namespace hc; 
int main() 
{
  int v[11] = {'G', 'd', 'k', 'k', 'n', 31, 'v', 'n', 'q', 'k', 'c'};

  array_view<int> av(11, v); 
  parallel_for_each(av.get_extent(), [=](hc::index<1> idx) [[hc]] { 
    av[idx] += 1; 
  });

  std::string expected("Hello world");
  for(unsigned int i = 0; i < av.get_extent().size(); i++) {
    assert(expected[i] == static_cast<char>(av(i)));
    std::cout << static_cast<char>(av(i));
  }
  return 0;
}
