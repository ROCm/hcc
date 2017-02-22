
// RUN: %hc %s -o %t.out && %t.out

#include <cstdio>
#include <hc.hpp>
#include <cassert>

// test HC with fine-grained SVM
// requires HSA Full Profile to operate successfully

struct base {
  int dummy;
  virtual ~base() {};
};

struct derived : base {
  int x;
  int y;
  derived(int x, int y) : x(x), y(y) {}
};

using base_ptr = std::unique_ptr<base>;


bool test1() {
  base* ptr_value = new derived(0,0);
  int* ptr_i = nullptr;

  hc::parallel_for_each(hc::extent<1>(1), [&](hc::index<1> i) [[hc]] {
    static_cast<derived*>(ptr_value)->x = 1000 + i[0];
    ptr_i = &(static_cast<derived*>(ptr_value)->x);
  }).wait();

  //printf("test1, x: expected 1000, actual %d\n", static_cast<derived*>(ptr_value)->x);
  return (ptr_i == &(static_cast<derived*>(ptr_value)->x));
}


bool test2() {

  base_ptr bp(new derived(0,0));
  base* ptr_value = nullptr;
  int* ptr_i = nullptr;

  hc::parallel_for_each(hc::extent<1>(1), [&](hc::index<1> i) [[hc]] {
    static_cast<derived*>(bp.get())->x = 1000 + i[0];
#if 0
    // crashes the backend
    ptr_value = bp.get();
#endif 
    ptr_i = &(static_cast<derived*>(bp.get())->x);
  }).wait();

  //printf("test2, x: expected 1000, actual %d\n", static_cast<derived*>(bp.get())->x);
  return (ptr_i == &(static_cast<derived*>(bp.get())->x));
}


int main() {
  bool ret = true;

  // only conduct the test in case we are running on a HSA full profile stack
  hc::accelerator acc;
  if (acc.is_hsa_accelerator() &&
      acc.get_profile() == hc::hcAgentProfileFull) {

    ret &= test1();
    ret &= test2();
  }

  return !(ret == true);
}
