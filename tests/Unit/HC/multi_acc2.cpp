// RUN: %hc --amdgpu-target=gfx701 --amdgpu-target=gfx801 --amdgpu-target=gfx802 --amdgpu-target=gfx803 %s -o %t.out && %t.out
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>

// header file for the hc API
#include <hc.hpp>

int main() {

  constexpr int N = 1024 * 1024 * 64;
  constexpr float a = 100.0f;

  std::vector<float> host_x(N);
  std::vector<float> host_y(N);

  // initialize the input data
  std::default_random_engine random_gen;
  std::uniform_real_distribution<float> distribution(-N, N);
  std::generate(host_x.begin(), host_x.end(), [&]() { return distribution(random_gen); });
  std::generate(host_y.begin(), host_y.end(), [&]() { return distribution(random_gen); });

  // CPU implementation of saxpy
  std::vector<float> host_result_y(N);
  for (int i = 0; i < N; i++) {
    host_result_y[i] = a * host_x[i] + host_y[i];
  }

  std::vector<hc::accelerator> all_accelerators = hc::accelerator::get_all();
  std::vector<hc::accelerator> accelerators;
  for (auto a = all_accelerators.begin(); a != all_accelerators.end(); a++) {

    // only pick accelerators supported by the HSA runtime
    if (a->is_hsa_accelerator()) {
      accelerators.push_back(*a);
    }
  }

  constexpr int numViewPerAcc = 2;
  int numSaxpyPerView = N/(accelerators.size() * numViewPerAcc);

  std::vector<hc::accelerator_view> acc_views;
  std::vector<hc::array_view<float,1>> x_views;
  std::vector<hc::array_view<float,1>> y_views;
  std::vector<hc::completion_future> futures;

  int dataCursor = 0;
  for (auto acc = accelerators.begin(); acc != accelerators.end(); acc++) {
    for (int i = 0; i < numViewPerAcc; i++) {

      // create a new accelerator_view
      acc_views.push_back(acc->create_view());

      // create array_views that only covers the data portion needed by this accelerator_view
      x_views.push_back(hc::array_view<float,1>(numSaxpyPerView, host_x.data() + dataCursor));
      y_views.push_back(hc::array_view<float,1>(numSaxpyPerView, host_y.data() + dataCursor));
      dataCursor+=numSaxpyPerView;


      auto& x_av = x_views.back();
      auto& y_av = y_views.back();
      hc::completion_future f;
      f = hc::parallel_for_each(acc_views.back(), x_av.get_extent()
                            , [=](hc::index<1> i) [[hc]] {
        y_av[i] = a * x_av[i] + y_av[i];
      });
      futures.push_back(f);


      //printf("dataCursor: %d\n",dataCursor);
    }
  }

  // If N is not a multiple of the number of acc_views,
  // calculate the remaining saxpy on the host
  for (; dataCursor!=N; dataCursor++) {
    host_y[dataCursor] = a * host_x[dataCursor] + host_y[dataCursor];
  }

  // synchronize all the results back to the host
  for(auto v = y_views.begin(); v != y_views.end(); v++) {
    v->synchronize();
  }

  // verify the results
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (fabs(host_y[i] - host_result_y[i]) > fabs(host_result_y[i] * 0.0001f))
      errors++;
  }
  //std::cout << errors << " errors" << std::endl;

  return errors;
}
