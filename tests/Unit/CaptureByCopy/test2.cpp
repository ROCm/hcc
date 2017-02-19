
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

#include <atomic>
#include <iostream>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully
// test capture a user functor by copy
// test funtions and user functor are now constructed from templates

#define SIZE (128)

using namespace concurrency;

template<typename _Tp>
class user_functor {
public:
  user_functor() restrict(amp,cpu) {}

  _Tp value(const _Tp& i) const restrict(amp,cpu) { return i + 1; }
};

// test get the result from the functor, store the value on stack and use it
template<typename _Tp, size_t N>
bool test1(const user_functor<_Tp>& functor) {
  bool ret = true;

  // prepare test data
  _Tp* const terms = new _Tp[N];
  for (size_t i = 0; i < N; ++i) {
    terms[i] = _Tp{};
  }
  std::atomic<_Tp>* accumulator = new std::atomic<_Tp>;
  *accumulator = _Tp{};

  extent<1> ex(N);
  parallel_for_each(ex, [=] (index<1>& idx) restrict(amp) {
    _Tp t = functor.value(idx[0]);
    terms[idx[0]] = t;
    accumulator->fetch_add(t);
  });

  // verify result
  _Tp expected_accumulator = _Tp{};
  for (size_t i = 0; i < N; ++i) {
    if (static_cast<size_t>(terms[i]) != i + 1) {
      ret = false;
    }
    expected_accumulator += static_cast<_Tp>(i + 1);
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
template<typename _Tp, size_t N>
bool test2(const user_functor<_Tp>& functor) {
  bool ret = true;

  // prepare test data
  _Tp* const terms = new _Tp[N];
  for (size_t i = 0; i < N; ++i) {
    terms[i] = 0;
  }
  std::atomic<_Tp>* accumulator = new std::atomic<_Tp>;
  *accumulator = _Tp{};

  extent<1> ex(N);
  parallel_for_each(ex, [=] (index<1>& idx) restrict(amp) {
    terms[idx[0]] = functor.value(idx[0]);
    accumulator->fetch_add(terms[idx[0]]);
  });

  // verify result
  _Tp expected_accumulator = _Tp{};
  for (size_t i = 0; i < N; ++i) {
    if (static_cast<size_t>(terms[i]) != i + 1) {
      ret = false;
    }
    expected_accumulator += static_cast<_Tp>(i + 1);
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
template<typename _Tp, size_t N>
bool test3(const user_functor<_Tp>& functor) {
  bool ret = true;

  // prepare test data
  _Tp* const terms = new _Tp[SIZE];
  for (size_t i = 0; i < N; ++i) {
    terms[i] = 0;
  }
  std::atomic<_Tp>* accumulator = new std::atomic<_Tp>;
  *accumulator = _Tp{};

  extent<1> ex(SIZE);
  parallel_for_each(ex, [=] (index<1>& idx) restrict(amp) {
    _Tp t = idx[0] + 1;
    terms[idx[0]] = t;
    accumulator->fetch_add(t);
    functor.value(idx[0]);
  });

  // verify result
  _Tp expected_accumulator = _Tp{};
  for (size_t i = 0; i < N; ++i) {
    if (static_cast<size_t>(terms[i]) != i + 1) {
      ret = false;
    }
    expected_accumulator += static_cast<_Tp>(i + 1);
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

    ret &= test1<int, SIZE>(user_functor<int>());
    ret &= test1<unsigned, SIZE>(user_functor<unsigned>());
    ret &= test1<long, SIZE>(user_functor<long>());
    ret &= test1<unsigned long, SIZE>(user_functor<unsigned long>());
  
    ret &= test2<int, SIZE>(user_functor<int>());
    ret &= test2<unsigned, SIZE>(user_functor<unsigned>());
    ret &= test2<long, SIZE>(user_functor<long>());
    ret &= test2<unsigned long, SIZE>(user_functor<unsigned long>());
  
    ret &= test3<int, SIZE>(user_functor<int>());
    ret &= test3<unsigned, SIZE>(user_functor<unsigned>());
    ret &= test3<long, SIZE>(user_functor<long>());
    ret &= test3<unsigned long, SIZE>(user_functor<unsigned long>());

  }

  return !(ret == true);
}

