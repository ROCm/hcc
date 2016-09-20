
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

#include <atomic>
#include <iostream>
#include <random>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully
// test capture a user functor with customized ctor by copy
// test funtions and user functor are now constructed from templates

#define SIZE (128)

using namespace concurrency;

template<typename _Tp>
class user_functor {
  _Tp val;
public:
  user_functor(const user_functor& other) restrict(amp,cpu) : val(other.val) {}

  user_functor(_Tp v) restrict(amp,cpu) : val(v) {}

  _Tp value(const _Tp& i) const restrict(amp,cpu) { return i + val; }
};

// test get the result from the functor, store the value on stack and use it
template<typename _Tp, size_t N>
bool test1(const user_functor<_Tp>& functor, _Tp val) {
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
    if (terms[i] != static_cast<_Tp>(i + val)) {
      ret = false;
    }
    expected_accumulator += static_cast<_Tp>(i + val);
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
bool test2(const user_functor<_Tp>& functor, _Tp val) {
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
    if (terms[i] != static_cast<_Tp>(i + val)) {
      ret = false;
    }
    expected_accumulator += static_cast<_Tp>(i + val);
  }

  if (*accumulator != expected_accumulator) {
    ret = false;
  }

  // release memory allocated
  delete[] terms;
  delete accumulator;

  return ret;
}

// dummy test, functor is called but the value is not used
template<typename _Tp, size_t N>
bool test3(const user_functor<_Tp>& functor, _Tp val) {
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
    _Tp t = idx[0] + val;
    terms[idx[0]] = t;
    accumulator->fetch_add(t);
    functor.value(idx[0]);
  });

  // verify result
  _Tp expected_accumulator = _Tp{};
  for (size_t i = 0; i < N; ++i) {
    if (terms[i] != static_cast<_Tp>(i + val)) {
      ret = false;
    }
    expected_accumulator += static_cast<_Tp>(i + val);
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

    // setup RNG
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<int> dis_int(1, 16);
    std::uniform_int_distribution<unsigned> dis_unsigned(1, 16);
    std::uniform_int_distribution<long> dis_long(1, 16);
    std::uniform_int_distribution<unsigned long> dis_ulong(1, 16);
  
    int val_int = dis_int(gen);
    unsigned val_unsigned = dis_unsigned(gen);
    long val_long = dis_long(gen);
    unsigned long val_ulong = dis_ulong(gen);
  
    ret &= test1<int, SIZE>(user_functor<int>(val_int), val_int);
    ret &= test1<unsigned, SIZE>(user_functor<unsigned>(val_unsigned), val_unsigned);
    ret &= test1<long, SIZE>(user_functor<long>(val_long), val_long);
    ret &= test1<unsigned long, SIZE>(user_functor<unsigned long>(val_ulong), val_ulong);
  
    ret &= test2<int, SIZE>(user_functor<int>(val_int), val_int);
    ret &= test2<unsigned, SIZE>(user_functor<unsigned>(val_unsigned), val_unsigned);
    ret &= test2<long, SIZE>(user_functor<long>(val_long), val_long);
    ret &= test2<unsigned long, SIZE>(user_functor<unsigned long>(val_ulong), val_ulong);
  
    ret &= test3<int, SIZE>(user_functor<int>(val_int), val_int);
    ret &= test3<unsigned, SIZE>(user_functor<unsigned>(val_unsigned), val_unsigned);
    ret &= test3<long, SIZE>(user_functor<long>(val_long), val_long);
    ret &= test3<unsigned long, SIZE>(user_functor<unsigned long>(val_ulong), val_ulong);

  }

  return !(ret == true);
}

