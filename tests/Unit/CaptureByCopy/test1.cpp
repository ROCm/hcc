
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

#include <atomic>
#include <iostream>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully
// test capture a user functor by copy

#define SIZE (128)

using namespace concurrency;

class user_functor {
public:
  user_functor() restrict(amp,cpu) {}

  long value(const int& i) const restrict(amp,cpu) { return i + 1; }
};

// test get the result from the functor, store the value on stack and use it
bool test1(const user_functor& functor) {
  bool ret = true;

  // prepare test data
  long* const terms = new long[SIZE];
  for (int i = 0; i < SIZE; ++i) {
    terms[i] = 0;
  }
  std::atomic<long>* accumulator = new std::atomic<long>;
  *accumulator = 0;

  extent<1> ex(SIZE);
  parallel_for_each(ex, [=] (index<1>& idx) restrict(amp) {
    long t = functor.value(idx[0]);
    terms[idx[0]] = t;
    accumulator->fetch_add(t);
  });

  // verify result
  long expected_accumulator = 0;
  for (int i = 0; i < SIZE; ++i) {
    if (terms[i] != i + 1) {
      ret = false;
    }
    expected_accumulator += (i + 1);
  }

  if (*accumulator != expected_accumulator) {
    ret = false;
  }

  // release memory allocated
  delete[] terms;
  delete accumulator;

  return ret;
}

// test get the result from the functor, store the value to memory and use it
bool test2(const user_functor& functor) {
  bool ret = true;

  // prepare test data
  long* const terms = new long[SIZE];
  for (int i = 0; i < SIZE; ++i) {
    terms[i] = 0;
  }
  std::atomic<long>* accumulator = new std::atomic<long>;
  *accumulator = 0;

  extent<1> ex(SIZE);
  parallel_for_each(ex, [=] (index<1>& idx) restrict(amp) {
    terms[idx[0]] = functor.value(idx[0]);
    accumulator->fetch_add(terms[idx[0]]);
  });

  // verify result
  long expected_accumulator = 0;
  for (int i = 0; i < SIZE; ++i) {
    if (terms[i] != i + 1) {
      ret = false;
    }
    expected_accumulator += (i + 1);
  }

  if (*accumulator != expected_accumulator) {
    ret = false;
  }

  // release memory allocated
  delete[] terms;
  delete accumulator;

  return ret;
}

// dummy test, functor is called but value is not used
bool test3(const user_functor& functor) {
  bool ret = true;

  // prepare test data
  long* const terms = new long[SIZE];
  for (int i = 0; i < SIZE; ++i) {
    terms[i] = 0;
  }
  std::atomic<long>* accumulator = new std::atomic<long>;
  *accumulator = 0;

  extent<1> ex(SIZE);
  parallel_for_each(ex, [=] (index<1>& idx) restrict(amp) {
    long t = idx[0] + 1;
    terms[idx[0]] = t;
    accumulator->fetch_add(t);
    functor.value(idx[0]);
  });

  // verify result
  long expected_accumulator = 0;
  for (int i = 0; i < SIZE; ++i) {
    if (terms[i] != i + 1) {
      ret = false;
    }
    expected_accumulator += (i + 1);
  }

  if (*accumulator != expected_accumulator) {
    ret = false;
  }

  // release memory allocated
  delete[] terms;
  delete accumulator;

  return ret;
}

int main() {
  bool ret = true;

  // only conduct the test in case we are running on a HSA full profile stack
  hc::accelerator acc;
  if (acc.is_hsa_accelerator() &&
      acc.get_profile() == hc::hcAgentProfileFull) {

    ret &= test1(user_functor());
    ret &= test2(user_functor());
    ret &= test3(user_functor());

  }

  return !(ret == true);
}

