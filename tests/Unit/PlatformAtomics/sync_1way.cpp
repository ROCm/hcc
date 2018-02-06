
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <iomanip>
#include <random>
#include <atomic>
#include <thread>
#include <chrono>
#include <amp.h>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with platform atomics functions
// requires HSA Full Profile to operate successfully

bool test() {
  // define inputs and output
  const int vecSize = 8;
  const int loopCount = 1024 * 1024;
  const int cpuSleepMsec = 50;

  std::atomic_int table_a[vecSize];
  auto ptr_a = &table_a[0];

  std::atomic_bool done(false);
  auto ptr_done = &done;

  // initialize test data
  for (int i = 0; i < vecSize; ++i) {
    table_a[i].store(0);
  }

  // fire CPU thread
  std::thread cpu_thread([=]() {
    std::cout << "Enter CPU monitor thread..." << std::endl;
    std::chrono::milliseconds dura( cpuSleepMsec );
    while (!*ptr_done) {
      for (int i = 0; i < vecSize; ++i) {
        std::cout << std::setw(8) << (ptr_a + i)->load(std::memory_order_acquire);
        if (i < vecSize - 1) {
          std::cout << ", ";
        } else {
          std::cout << std::endl;
        }
      }

      std::this_thread::sleep_for( dura );
    }
    std::cout << "Leave CPU monitor thread." << std::endl;
  });

  // launch kernel
  Concurrency::extent<1> e(vecSize);
  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp) {

    int tid = idx[0];
    int counter = 0;

    while (++counter <= loopCount) {
      (ptr_a + tid)->fetch_add(1, std::memory_order_release);
    }
  });

  // stop CPU thread
  done.store(true);
  cpu_thread.join();

  // Verify
  std::cout << "Final value:\n";
  int error = 0;
  int val = 0;
  for (int i = 0; i < vecSize; ++i) {
    val = (ptr_a + i)->load();

    std::cout << std::setw(8) << val;
    if (i < vecSize - 1) {
      std::cout << ", ";
    } else {
      std::cout << std::endl;
    }

    if (val != loopCount)
      ++error;
  }

  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error == 0);
}

int main() {
  bool ret = true;

  // only conduct the test in case we are running on a HSA full profile stack
  hc::accelerator acc;
  if (acc.is_hsa_accelerator() &&
      acc.get_profile() == hc::hcAgentProfileFull) {
    ret &= test();
  }

  return !(ret == true);
}


