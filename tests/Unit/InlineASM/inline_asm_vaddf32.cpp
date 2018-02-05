// RUN: %hc %s -o %t.out && %t.out

#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

// header file for the hc API
#include <hc.hpp>

#define N  (1024 * 500)

int main() {

  float host_x[N];
  float host_y[N];

  // initialize the input data
  std::default_random_engine random_gen;
  std::uniform_real_distribution<float> distribution(-N, N);
  std::generate_n(host_x, N, [&]() { return distribution(random_gen); });
  std::generate_n(host_y, N, [&]() { return distribution(random_gen); });

  // CPU implementation of saxpy
  float host_result_y[N];
  for (int i = 0; i < N; i++) {
    host_result_y[i] = host_x[i] + host_y[i];
  }

  // wrap the data buffer around with an array_view
  // to let the hcc runtime to manage the data transfer
  hc::array_view<float, 1> x(N, host_x);
  hc::array_view<float, 1> y(N, host_y);

  // launch a GPU kernel to compute the saxpy in parallel 
  hc::parallel_for_each(hc::extent<1>(N)
                      , [=](hc::index<1> i) [[hc]] {
    // y[i] = x[i] + y[i]
    asm volatile ("v_add_f32_e32 %0, %1, %2" : "=v" (y[i]) : "v" (x[i]),"v" (y[i]));
  });
   
  // verify the results
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (fabs(y[i] - host_result_y[i]) > fabs(host_result_y[i] * 0.0001f)) {
      errors++;
    }
  }
  std::cout << errors << " errors" << std::endl;

  return errors;
}
