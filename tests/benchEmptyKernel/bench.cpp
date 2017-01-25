// RUN: %hc %s %S/statutils.CPP -O3  %S/hsacodelib.CPP  -o %t.out -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 
// RUN: %t.out 10000 %T
// // This runs burst of 100 kernels to measure kernel-to-kernel overhead:
// RUN: %t.out 1000 %T 100
// RUN: test -e pfe.dat && mv pfe.dat %T/pfe.dat
// RUN: test -e grid_launch.dat && mv grid_launch.dat %T/grid_launch.dat

// benchmark for empty PFE/grid_launch kernel
//
// Authors: Kevin Wu, Yan-Ming Li
//
// For best results set GPU performance level to high, where N in cardN is a number
// echo high | sudo tee /sys/class/drm/cardX/device/power_dpm_force_performance_level

// hcc `hcc-config --cxxflags --ldflags` bench.cpp -o bench
// ./bench 10000 && gnuplot plot.plt
//
//

#define BENCH_HSA 1


#include "hc.hpp"
#include "hc_am.hpp"
#include "grid_launch.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <iomanip>


#include "statutils.h"

#define GRID_SIZE 16
#define TILE_SIZE 16

#define DISPATCH_COUNT 10000
#define TOL_HI 1e-4

// Text width for labels.
#define TW 48

#define PFE_ACTIVE  0x1
#define PFE_BLOCKED 0x2
#define GL_ACTIVE   0x4
#define GL_BLOCKED  0x8
#define DISPATCH_HSA_KERNEL_CF    0x10
#define DISPATCH_HSA_KERNEL_NOCF  0x20

int p_tests = 0xff;
//int p_tests = DISPATCH_HSA_KERNEL_CF+DISPATCH_HSA_KERNEL_NOCF;
//
int p_useSystemScope = false;

__attribute__((hc_grid_launch)) 
void nullkernel(const grid_launch_parm lp, float* A) {
    if (A) {
        A[0] = 0x13;
    }
}


#if BENCH_HSA

#include "hsacodelib.h"

void explicit_launch_null_kernel(const grid_launch_parm *lp, const Kernel &k)
{
    struct NullKernelArgs {
        uint32_t hidden[6];
        void *Ad;
    } args;

    args.hidden[0] = GRID_SIZE;
    args.hidden[1] = 1;
    args.hidden[2] = 1;
    args.hidden[3] = TILE_SIZE;
    args.hidden[4] = 1;
    args.hidden[5] = 1;
    args.Ad        = nullptr;


    dispatch_glp_kernel(lp, k, &args, sizeof(NullKernelArgs), p_useSystemScope);
}

#define KERNEL_NAME "_ZN12_GLOBAL__N_142_Z10nullkernel16grid_launch_parmPf_functor19__cxxamp_trampolineEiiiiiiPf"

void time_dispatch_hsa_kernel(std::string testName, int dispatch_count, int burst_count, const grid_launch_parm *lp, const char *nullkernel_hsaco_dir)
{
  std::string nullkernel_hsaco(nullkernel_hsaco_dir);
  nullkernel_hsaco += "/nullkernel-fiji.hsaco";
  Kernel k = load_hsaco(lp->av, nullkernel_hsaco.c_str(), KERNEL_NAME);
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

  std::vector<std::chrono::duration<double>> elapsed_timer;

  // Timing null grid_launch call, active wait
  for(int i = 0; i < dispatch_count; ++i) {
    start = std::chrono::high_resolution_clock::now();

    for (int j=0; j<burst_count ;j++) {
        explicit_launch_null_kernel(lp, k);
    };

    //std::cout << "CF get_use_count=" << cf.get_use_count() << "is_ready=" << cf.is_ready()<< "\n";
    //
    lp->av->wait(hc::hcWaitModeActive);
    //cf.wait(hc::hcWaitModeActive);

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;
    elapsed_timer.push_back(dur);
  }
  std::vector<std::chrono::duration<double>> outliers;
  remove_outliers(elapsed_timer, outliers);
  plot(testName, elapsed_timer);
  std::cout << std::setw(TW-2) << std::left << testName << ": " 
            << std::setw(8) << std::setprecision(8) << average(elapsed_timer)*1000000.0 << "\n";
};

#endif

int main(int argc, char* argv[]) {

  const char *nullkernel_hsaco_dir = NULL;

  int dispatch_count = DISPATCH_COUNT;
  int burst_count = 1;
  if(argc > 1)
    dispatch_count = std::stoi(argv[1]);
  if(argc > 2) {
    nullkernel_hsaco_dir = argv[2];
  }
  if(argc > 3) {
    burst_count = std::stoi(argv[3]);
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::vector<std::chrono::duration<double>> elapsed_pfe;
  std::vector<std::chrono::duration<double>> elapsed_grid_launch;
  std::vector<std::chrono::duration<double>> elapsed_exception;
  std::chrono::duration<double> tol_hi(TOL_HI);
  std::vector<std::chrono::duration<double>> outliers_pfe;
  std::vector<std::chrono::duration<double>> outliers_gl;
  std::vector<std::chrono::duration<double>> outliers_gl_ex;


  hc::accelerator acc = hc::accelerator();
  // Set up extra stuff
  static hc::accelerator_view av = acc.create_view(hc::execute_any_order);
  //static hc::accelerator_view av = acc.create_view(hc::execute_in_order);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim = gl_dim3(GRID_SIZE);
  lp.group_dim = gl_dim3(TILE_SIZE);
  lp.av = &av;

  std::cout << "Iterations per test:              " << dispatch_count << "\n";
  std::cout << "Bursts (#dispatches before sync): " << burst_count  << "\n";
  std::cout << "\n";


  // launch empty kernel to initialize everything first
  // timing for null kernel launch appears later

  hc::parallel_for_each(av, hc::extent<3>(lp.grid_dim.x*lp.group_dim.x,1,1).tile(lp.group_dim.x,1,1),
  [=](hc::index<3>& idx) __HC__ {
  }).wait();

  // Setting lp.cf to completion_future so we can track completion: (NULL ignores all synchronization)

  auto wait_time_us = std::chrono::milliseconds(10);


  // Timing null pfe, active wait
  if (p_tests & PFE_ACTIVE) {
      for(int i = 0; i < dispatch_count; ++i) {
        start = std::chrono::high_resolution_clock::now();

        hc::completion_future cf;
        for (int j=0; j<burst_count ;j++) {
            cf = hc::parallel_for_each(av, hc::extent<3>(lp.grid_dim.x*lp.group_dim.x,1,1).tile(lp.group_dim.x,1,1),
            [=](hc::index<3>& idx) __HC__ {
            });
        };
        cf.wait(hc::hcWaitModeActive);

        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        elapsed_pfe.push_back(dur);
      }
      remove_outliers(elapsed_pfe, outliers_pfe);
      plot("pfe", elapsed_pfe);
      std::cout << std::setw(TW) << "pfe time, active (us):                  " 
                << std::setprecision(8) << average(elapsed_pfe)*1000000.0 << "\n";
  }


  if (p_tests & PFE_BLOCKED) {
      // Timing null pfe, blocking wait
      for(int i = 0; i < dispatch_count; ++i) {
        start = std::chrono::high_resolution_clock::now();
        hc::completion_future cf;
        for (int j=0; j<burst_count ;j++) {
            cf = hc::parallel_for_each(av, hc::extent<3>(lp.grid_dim.x*lp.group_dim.x,1,1).tile(lp.group_dim.x,1,1),
            [=](hc::index<3>& idx) __HC__ {
            });
        };
        cf.wait(hc::hcWaitModeBlocked);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        elapsed_pfe.push_back(dur);
      }
      remove_outliers(elapsed_pfe, outliers_pfe);
      plot("pfe", elapsed_pfe);
      std::cout << std::setw(TW) << "pfe time, blocked (us):                 " 
                << std::setprecision(8) << average(elapsed_pfe)*1000000.0 << "\n";
  }


  if (p_tests & GL_ACTIVE) {
      // Timing null grid_launch call, active wait
      for(int i = 0; i < dispatch_count; ++i) {
        start = std::chrono::high_resolution_clock::now();
        hc::completion_future cf; // create new completion-future 
        lp.cf = &cf;

        for (int j=0; j<burst_count ;j++) {
            nullkernel(lp, 0x0);
        }
        //std::cout << "CF get_use_count=" << cf.get_use_count() << "is_ready=" << cf.is_ready()<< "\n";
        cf.wait(hc::hcWaitModeActive);

        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        elapsed_grid_launch.push_back(dur);
      }
      remove_outliers(elapsed_grid_launch, outliers_gl);
      plot("grid_launch", elapsed_grid_launch);
      std::cout << std::setw(TW) << "grid_launch time, active (us):          " 
                << std::setprecision(8) << average(elapsed_grid_launch)*1000000.0 << "\n";
  }


  if (p_tests & GL_BLOCKED) {
      // Timing null grid_launch call, blocked wait
      for(int i = 0; i < dispatch_count; ++i) {
        start = std::chrono::high_resolution_clock::now();
        hc::completion_future cf; // create new completion-future 
        lp.cf = &cf;

        for (int j=0; j<burst_count ;j++) {
            nullkernel(lp, 0x0);
        };
        //std::cout << "CF get_use_count=" << cf.get_use_count() << "is_ready=" << cf.is_ready()<< "\n";
        cf.wait(hc::hcWaitModeBlocked);

        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        elapsed_grid_launch.push_back(dur);
      }
      remove_outliers(elapsed_grid_launch, outliers_gl);
      plot("grid_launch", elapsed_grid_launch);
      std::cout << std::setw(TW) << "grid_launch time, blocked (us):         " 
                << std::setprecision(8) << average(elapsed_grid_launch)*1000000.0 << "\n";
  }


  if (nullkernel_hsaco_dir) {
      if (p_tests & DISPATCH_HSA_KERNEL_CF) {
          hc::completion_future cf; // create new completion-future 
          lp.cf = &cf;
          time_dispatch_hsa_kernel("dispatch_hsa_kernel, withcompletion, active", dispatch_count, burst_count, &lp, nullkernel_hsaco_dir);
      }

      if (p_tests & DISPATCH_HSA_KERNEL_NOCF) {
          lp.cf=0x0;
          time_dispatch_hsa_kernel("dispatch_hsa_kernel, nocompletion, active", dispatch_count, burst_count, &lp, nullkernel_hsaco_dir);
      }
  } else {
      std::cout << "skipping dispatch_hsa_kernel - must specify directory with hsaco on commandline.  (ie: ./bench 10000 Inputs/)\n";
  }

  return 0;
}
