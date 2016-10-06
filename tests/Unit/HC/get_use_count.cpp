// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <assert.h>

void checkPassByValue (hc::completion_future cf, int expectedCount)
{
    assert(cf.get_use_count() == expectedCount);
}


void checkPassByRef (const hc::completion_future &cf, int expectedCount)
{
    assert(cf.get_use_count() == expectedCount);
}

#define GRID_SIZE 1000

int main() {
  using namespace hc;
  array<uint64_t, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

  // launch a kernel, log current hardware cycle count
  completion_future cf = parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table(idx) = __cycle_u64();
  });


  // At this point the HCC queue structure has a reference the CF, and we have one here in CF:

  assert(cf.get_use_count() == 2);

  checkPassByValue(cf, 3);
  checkPassByRef(cf, 2);

  cf.wait();

  // waiting removes the reference from the queue so we just have the one here in CF.
  assert(cf.get_use_count() == 1);


  return 0;
}

