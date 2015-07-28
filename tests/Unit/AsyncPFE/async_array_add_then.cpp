// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <random>
#include <future>
#include <hc.hpp>

// An example which shows how to launch a kernel asynchronously
int main ()
{
  // define inputs and output
  const int vecSize = 2048;

  int table_a[vecSize];
  int table_b[vecSize];
  int table_c[vecSize];
  int *p_a = &table_a[0];
  int *p_b = &table_b[0];
  int *p_c = &table_c[0];

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;
  for (int i = 0; i < vecSize; ++i) {
    table_a[i] = int_dist(rd);
    table_b[i] = int_dist(rd);
  }

  // launch kernel
  std::cout << "async launch the 1st kernel\n";
  hc::extent<1> e(vecSize);
  hc::completion_future fut = hc::async_parallel_for_each(
    e,
    [=](hc::index<1> idx) restrict(amp) {
      p_c[idx[0]] = p_a[idx[0]] + p_b[idx[0]];

  });

  // use completion_future::then() to launch another kernel
  std::promise<void> done_promise;
  fut.then([=, &done_promise] {
    std::cout << "async launch the 2nd kernel\n";
    hc::completion_future fut2 = hc::async_parallel_for_each(
      e,
      [=](hc::index<1> idx) restrict(amp) {
        p_c[idx[0]] += p_a[idx[0]] + p_b[idx[0]];
      });

    // use completion_future::then() yet again
    fut2.then([=, &done_promise] {
      std::cout << "sync launch the 3rd kernel\n";
      // FIXME: we need a pfe interface which doesn't need ts_allocator in hc namespace
      hc::ts_allocator tsa;
      parallel_for_each(
        e,
        tsa,
        [=](hc::index<1> idx) restrict(amp) {
          p_c[idx[0]] += p_a[idx[0]] + p_b[idx[0]];
        });
      done_promise.set_value();
    });

  });
  done_promise.get_future().wait();

  // verify
  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += table_c[i] - (table_a[i] + table_b[i]) * 3;
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return error != 0;
}

