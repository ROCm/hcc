// RUN: %hc %s -I/opt/rocm/hsa/include -L/opt/rocm/lib -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>

#include <hsa/hsa.h>

// loop to deliberately slow down kernel execution
#define LOOP_COUNT (1024)

#define TEST_DEBUG (0)

// An example which shows how to use accelerator_view::create_blocking_marker(completion_future&)
///
/// The test case only works on HSA because it directly uses HSA runtime API
/// It would use completion_future::get_native_handle() to retrieve the
/// underlying hsa_signal_t data structure to query if dependent kernels have
/// really finished execution before the new kernel is executed.
bool test() {
  bool ret = true;

  // define inputs and output
  const int vecSize = 2048;

  hc::array_view<int, 1> table_a(vecSize);
  hc::array_view<int, 1> table_b(vecSize);
  hc::array_view<int, 1> table_c(vecSize);

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;
  for (int i = 0; i < vecSize; ++i) {
    table_a[i] = int_dist(rd);
    table_b[i] = int_dist(rd);
  }

  // launch kernel
  hc::extent<1> e(vecSize);
  hc::completion_future fut = hc::parallel_for_each(
    e,
    [=](hc::index<1> idx) __HC__ {
      for (int i = 0; i < LOOP_COUNT; ++i)
        table_c(idx) = table_a(idx) + table_b(idx);
  });

  // create a barrier packet
  hc::accelerator_view av  = hc::accelerator().get_default_view();
  hc::accelerator_view av2 = hc::accelerator().create_view();
  hc::accelerator_view av3 = hc::accelerator().create_view();
  hc::completion_future fut2 = av.create_blocking_marker(fut);

  void* nativeHandle = fut.get_native_handle();
  void* nativeHandle2 = fut2.get_native_handle();

#if TEST_DEBUG
  std::cout << nativeHandle << "\n";
  std::cout << nativeHandle2 << "\n";
#endif

  hsa_signal_value_t signal_value;
  hsa_signal_value_t signal_value2;

  signal_value = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle));
#if TEST_DEBUG
  std::cout << "kernel signal value: " << signal_value << "\n";
#endif

  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle2));
#if TEST_DEBUG
  std::cout << "blocking barrier signal value: " << signal_value << "\n";
#endif

  // wait on the barrier packet
  fut2.wait();

  // the barrier packet would ensure all previous packets were processed

  signal_value = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle));
#if TEST_DEBUG
  std::cout << "kernel signal value: " << signal_value << "\n";
#endif
  ret &= (signal_value == 0);

  signal_value2 = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(nativeHandle2));
#if TEST_DEBUG
  std::cout << "barrier signal value: " << signal_value << "\n";
#endif
  ret &= (signal_value2 == 0);

  // verify
  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += table_c[i] - (table_a[i] + table_b[i]);
  }
  if (error == 0) {
    std::cout << "Verify first part success!\n";
  } else {
    std::cout << "Verify first part failed!\n";
  }
  ret &= (error == 0);


  // Try another case with events that are not dispatched.
  // These will have null asyncOps so tests corner case in HCC:
  {
      hc::completion_future nullcf[7];
      hc::completion_future cfA;
      
      cfA = av.create_blocking_marker(nullcf[0]);
      cfA.wait();

      cfA= av.create_blocking_marker({nullcf[0], nullcf[1], nullcf[2]});
      cfA.wait();

      // Test case with > 6 completion futures:
      cfA = av.create_blocking_marker({nullcf[0], nullcf[1], nullcf[2], nullcf[3], nullcf[4], nullcf[5], nullcf[6]});
      cfA.wait();


      std::cout << "null signals verify OK\n";


      hc::completion_future cf_pfe, cf_pfe2;
     
      cf_pfe  = hc::parallel_for_each(av,
        e,
        [=](hc::index<1> idx) __HC__ {
          for (int i = 0; i < LOOP_COUNT; ++i)
            table_c(idx) = table_a(idx) + table_b(idx);
      });

      cfA = av2.create_blocking_marker({nullcf[0], cf_pfe});
      cfA.wait();

      hsa_signal_value_t signal_value_pfe = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(cf_pfe.get_native_handle()));
      hsa_signal_value_t signal_value_cbm = hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(cfA.get_native_handle()));

      std::cout << "create_blocking_marker on single PFE verify OK\n";
      

      // Both signals should have completed.
      assert(signal_value_pfe == 0);
      assert(signal_value_cbm == 0);



      // Try a 3-way:
      // Two kernels sent to different PFE, then wait on all three
      cf_pfe  = hc::parallel_for_each(av,
        e,
        [=](hc::index<1> idx) __HC__ {
          for (int i = 0; i < LOOP_COUNT; ++i)
            table_c(idx) = table_a(idx) + table_b(idx);
      });
      cf_pfe2  = hc::parallel_for_each(av2,
        e,
        [=](hc::index<1> idx) __HC__ {
          for (int i = 0; i < LOOP_COUNT; ++i)
            table_c(idx) = table_a(idx) + table_b(idx);
      });


      cfA = av2.create_blocking_marker({nullcf[0], cf_pfe, nullcf[1], cf_pfe2});
      cfA.wait();

      assert (hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(cf_pfe.get_native_handle())) == 0);
      assert (hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(cf_pfe2.get_native_handle())) == 0);
      assert (hsa_signal_load_acquire(*static_cast<hsa_signal_t*>(cfA.get_native_handle())) == 0);
      std::cout << "create_blocking_marker on dual PFE verify OK\n";
  }



  return ret;
}

int main() {
  bool ret = true;

  ret &= test();

  return !(ret == true);
}

