// RUN: %hc %s -g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined -fno-sanitize=vptr -o %t.out && %t.out

#include <hc.hpp>
#include <cstdlib>

void fill(hc::array_view<int,1>& input, int x) {

  hc::parallel_for_each(input.get_extent(), [=](const hc::index<1>& idx) [[hc]] {
    input[idx[0]] = x;
  }).wait();

}

int main() {

  hc::array_view<int,1> av(64);
  for (int i = 0;i < 64; i++)
    av[i] = 88;

  fill(av, 21);

  for (int i = 0;i < 64; i++)
  {
    if (av[i] != 21)
    {
      printf("Fill failed\n");
      std::abort();
    }
  }
}

