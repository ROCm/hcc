
// RUN: %hc %s -o %t.out && %t.out

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <atomic>
#include <thread>
#include <amp.h>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with platform atomics functions
// requires HSA Full Profile to operate successfully

bool test() {
  // define inputs and output
  const int vecSize = 16;
  const int cpuSleepMsec = 25;

  // specify syscall number
  std::atomic_int table_a[vecSize];
  auto ptr_a = &table_a[0];

  // syscall parameter
  std::atomic_long table_b[vecSize];
  auto ptr_b = &table_b[0];

  // test result
  std::atomic_long table_c[vecSize];
  auto ptr_c = &table_c[0];

  // CPU syscall service thread control
  std::atomic_bool done(false);
  auto ptr_done = &done;

  // initialize test data
  for (int i = 0; i < vecSize; ++i) {
    table_a[i].store(0);
    table_b[i].store(0);
    table_c[i].store(0);
  }

  // fire CPU thread
  std::thread cpu_thread([=]() {
    std::cout << "Enter CPU syscall service thread..." << std::endl;
    std::chrono::milliseconds dura( cpuSleepMsec );
    int syscall;
    while (!*ptr_done) {
      for (int i = 0; i < vecSize; ++i) {
        syscall = (ptr_a + i)->load(std::memory_order_acquire);

        if (syscall) {
          // load parameter
          long param = (ptr_b + i)->load(std::memory_order_acquire);
          
          // do actual stuff
          long result;
          switch (syscall) {
            case 1: // malloc
              std::cout << "tid: " << i << ", malloc(" << param << ")\n";
              result = (long)malloc(param);
            break;
            case 2: // free
              std::cout << "tid: " << i << ", free(" << param << ")\n";
              free((void*)param);
              result = 0;
            break;
          }

          // store result
          (ptr_b + i)->store(result, std::memory_order_release); 

          // reset flag
          (ptr_a + i)->store(0, std::memory_order_release);
        }
      }

      std::this_thread::sleep_for( dura );
    }
    std::cout << "Leave CPU syscall service thread." << std::endl;
  });

  // launch kernel
  Concurrency::extent<1> e(vecSize);
  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp) {

    int tid = idx[0];
    int flag;

    int amount = (tid + 1) * sizeof(int);

    // part 1
    // test call malloc (func 1) with argument amount

    // store the parameter
    (ptr_b + tid)->store(amount, std::memory_order_release);

    // store the signal value
    (ptr_a + tid)->store(1, std::memory_order_release);

    // wait until syscall returns
    while ((ptr_a + tid)->load(std::memory_order_acquire));

    // load result from CPU
    long result = (ptr_b + tid)->load();

    // store result
    (ptr_c + tid)->store(result, std::memory_order_release);

    // test access the memory allocated
    int *alloc_array = (int*)result;
    alloc_array[tid] = tid;
    // store result
    (ptr_c + tid)->store(alloc_array[tid], std::memory_order_release);
   
    // part 2
    // test call free (func 2) with argument result

    // store the parameter
    (ptr_b + tid)->store(result, std::memory_order_release);

    // store the signal value
    (ptr_a + tid)->store(2, std::memory_order_release);

    // wait until syscall returns
    while ((ptr_a + tid)->load(std::memory_order_acquire));

    // ignore free result

  });

  // stop CPU thread
  done.store(true);
  cpu_thread.join();

  // Verify
  std::cout << "Final value:\n";
  int error = 0;
  long val = 0;
  for (int i = 0; i < vecSize; ++i) {
    val = (ptr_c + i)->load();

    std::cout << std::setw(2) << val;
    if (i < vecSize - 1) {
      std::cout << ", ";
      if (val != i) {
        ++error;
      }
    } else {
      std::cout << std::endl;
      if (val != vecSize - 1) {
        ++error;
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

