// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>

using namespace concurrency;

#define T int
#define INIT 50

int main(void) {
  const int vecSize = 100;
  std::vector<accelerator> accs = accelerator::get_all();
  accelerator gpu_acc;
  for (auto& it: accs)
    if (it != accelerator(accelerator::cpu_accelerator)) {
      gpu_acc = it;
      break;
    }
  accelerator_view gpu_av = gpu_acc.get_default_view();

  std::vector<T> source(vecSize, INIT + 1);
  array<T, 1> src(vecSize, source.begin());

  std::vector<T> destination(vecSize, INIT);
  array<T, 1> dest(vecSize, destination.begin());

  // array that holds original value of dest
  std::vector<T> target(vecSize, 0);
  array<T, 1> tgt(vecSize, target.begin());

  // Run in a separate thread
  std::thread t([&]() {
     parallel_for_each(gpu_av, dest.get_extent(), [=, &dest, &tgt](index<1> idx) restrict(amp) {
     for(unsigned i = 0; i < vecSize; i++)
       for (unsigned j = 0; j < vecSize; j++)
         tgt[idx] = dest[i];
     });
    });
  t.join();

  // At this point, the copying needs to wait for availability of dest in thread t
  // otherwise, undefined behavior happens in PFE since dest[i] is not deterministic
  copy(src, dest);
  
  // Verify tgt on CPU
  array_view<T> av(tgt);
  bool ret = true;
  for(unsigned i = 0; i < vecSize; ++i) {
      if(av[i] != INIT) {
        printf("tgt = %d\n", av[i]);
        ret = false;
      }
  }
  return !(ret == true);
}
