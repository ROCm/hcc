// XFAIL: Linux
// RUN: %cxxamp %s -Xclang -fhsa-ext -o %t.out && %t.out

#include <cassert>
#include <algorithm>
#include <iostream>
#include <hc.hpp>

#define GRID_SIZE (1024)

/// test fetching the number of async operations associated with one accelerator_view
int main() {

  hc::am_status_t mem_status;
  int* v1 = hc::am_alloc(GRID_SIZE * sizeof(int), AM_DEFAULT_FLAGS, &mem_status); 
  assert(mem_status == AM_SUCCESS);

  int n(0);
  std::generate(v1, v1+GRID_SIZE, [&n]{ return n++; });
 
  hc::completion_future fut = hc::parallel_for_each(
    hc::extent<1>(GRID_SIZE), 
    [=](hc::index<1>& idx) __attribute((hc)) {
      v1[idx[0]]++;
  });
  fut.wait();

  n = 1;
  int errors = std::count_if(v1, v1+GRID_SIZE, [=,&n](int i) { return (i!=n++); });

  mem_status = hc::am_free(v1);
  assert(mem_status == AM_SUCCESS);

  return errors;
}

