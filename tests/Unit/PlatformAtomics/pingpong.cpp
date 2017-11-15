
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
  const int cpuSleepMsec = 25;
  const int pingPongShots = 4;

  std::atomic_int table_a[vecSize];
  auto ptr_a = &table_a[0];

  std::atomic_int thread_flag;
  auto ptr_flag = &thread_flag;

  std::atomic_bool done(false);
  auto ptr_done = &done;

  // initialize test data
  for (int i = 0; i < vecSize; ++i) {
    table_a[i].store(0);
  }
  thread_flag.store(0);

  // fire CPU thread
  std::thread cpu_thread([=]() {
    std::cout << "Enter CPU monitor thread..." << std::endl;
    std::chrono::milliseconds dura( cpuSleepMsec );
    int val;
    int direction = 0;
    int shots = 0;
    while (!*ptr_done) {
      for (int i = 0; i < vecSize; ++i) {
        val = (ptr_a + i)->load(std::memory_order_acquire);
        std::cout << std::setw(2) << val;
        if (i < vecSize - 1) {
          std::cout << ", ";
        } else {
          std::cout << std::endl;
        }

        if (val == 1) {
          if (direction == 0) {
            if ((ptr_flag->load(std::memory_order_acquire) == vecSize - 1) && (shots < pingPongShots)) {
              direction = 1;
              ++shots;
            } else {
              ptr_flag->fetch_add(1, std::memory_order_acq_rel);
            }
          } else {
            if ((ptr_flag->load(std::memory_order_acquire) == 0) && (shots < pingPongShots)) {
              direction = 0;
              ++shots;
            } else {
              ptr_flag->fetch_sub(1, std::memory_order_acq_rel);
            }
          }
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
    int flag;
    while (true) {
      flag = ptr_flag->load(std::memory_order_acquire);
      if (flag >= vecSize || flag < 0) {
        break;
      }

      if (flag == tid) {
        (ptr_a + tid)->store(1, std::memory_order_release);
      } else {
        (ptr_a + tid)->store(0, std::memory_order_release);
      }
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

    std::cout << std::setw(2) << val;
    if (i < vecSize - 1) {
      std::cout << ", ";
    } else {
      std::cout << std::endl;
    }
  }

  if (pingPongShots % 2 == 0) {
    for (int i = 0; i < vecSize; ++i) {
      val = (ptr_a + i)->load();
      if (i < vecSize - 1) {
        if (val != 0) {
          ++error;
        }
      } else {
        if (val != 1) {
          ++error;
        }
      }
    }
  } else {
    for (int i = 0; i < vecSize; ++i) {
      val = (ptr_a + i)->load();
      if (i > 0) {
        if (val != 0) {
          ++error;
        }
      } else {
        if (val != 1) {
          ++error;
        }
      }
    }
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

