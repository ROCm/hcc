
#include <hc.hpp>
#include <hc_am.hpp>
#include <iostream>
#include <string>
#include <cmath>

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

int main() {

  auto acc = hc::accelerator();
  int* data1_d = (int*)hc::am_alloc(256*sizeof(int), acc, 0);

  hc::array_view<int,1> av(64);
  for (int i = 0;i < 64; i++)
    av[i] = i;

  int s = std::sqrt(sum(av));

  std::string ss = std::to_string(s);
  std::cout << "sum: " << ss << std::endl;

 // printf("sum: %d\n",s);

  hc::am_free(data1_d);

  return !(s==44);
}
