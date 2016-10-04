// RUN: %hc %s -o %t.out
// RUN: %t.out 10000
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


#include "hc.hpp"
#include "hc_am.hpp"
#include "grid_launch.hpp"
#include <iostream>
#include <fstream>

#include <chrono>
#include <vector>
#include <thread>
#include <iomanip>

#define GRID_SIZE 16
#define TILE_SIZE 16

#define DISPATCH_COUNT 100000
#define TOL_HI 1e-4

__attribute__((hc_grid_launch)) 
void nullkernel(const grid_launch_parm lp, float* A) {
    if (A) {
        A[0] = 0x13;
    }
}


template <typename T>
T average(const std::vector<std::chrono::duration<T>> &data) {
  T avg_duration = 0;

  for(auto &i : data)
    avg_duration += i.count();

  return avg_duration/data.size();
}

void plot(const std::string &filename, const std::vector<std::chrono::duration<double>> &data) {
  std::ofstream file(filename + ".dat", std::ios_base::out | std::ios_base::trunc);
  file << "#x y\n";
  for(auto i = data.begin(); i != data.end(); i++)
    file << i - data.begin() << ' ' << i->count() << '\n';
  file << "A_mean = " << average(data) << "\n";
  file.close();
}

void remove_outliers(std::vector<std::chrono::duration<double>> &data,
                     std::vector<std::chrono::duration<double>> &outliers) {

  auto tdata = data;
  std::sort(tdata.begin(), tdata.end());

  const int size = tdata.size();
  const bool parity = size % 2;
  /*
         ---------.---------
    |----|   .    |    .   | ----|
         ---------'---------
         Q1       Q2       Q3

    Q2: median
    Q1: first quartile, median of the lower half, ~25% lie below Q1, ~75% lie above Q1
    Q3: third quartile, median of the upper half ~ 75% lie below Q3, ~25% lie above Q3
    IQR: interquartile range, distance between Q3 and Q1
    outliers: any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
  */

  const double Q2 = parity ? tdata[size/2].count() : (tdata[size/2].count() + tdata[size/2 - 1].count())/2;
  const double Q1 = (tdata[size/4].count() + tdata[size/4 - 1].count())/2;
  const double Q3 = (tdata[size - size/4].count() + tdata[size - size/4 - 1].count())/2;

  const double IQR = Q3 - Q1;
  const double lwrB = Q1 - 1.5*IQR;
  const double uppB = Q3 + 1.5*IQR;

  std::copy_if(data.begin(), data.end(), std::back_inserter(outliers),
    [&](std::chrono::duration<double> dur) { return dur.count() < lwrB || dur.count() > uppB;} );

  data.erase(std::remove_if(data.begin(), data.end(),
        [&](std::chrono::duration<double> dur) { return dur.count() < lwrB || dur.count() > uppB;} ),
      data.end());
}

void printVecInfo(const std::string &name, const std::vector<std::chrono::duration<double>> &data) {
  std::cout << name << "count: " << data.size() << "\n";
  for(auto &i : data)
    std::cout << "  " << i.count() << "\n";
}

int main(int argc, char* argv[]) {

  int dispatch_count = DISPATCH_COUNT;
  if(argc > 1)
    dispatch_count = std::stoi(argv[1]);

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::vector<std::chrono::duration<double>> elapsed_pfe;
  std::vector<std::chrono::duration<double>> elapsed_grid_launch;
  std::vector<std::chrono::duration<double>> elapsed_exception;
  std::chrono::duration<double> tol_hi(TOL_HI);
  std::vector<std::chrono::duration<double>> outliers_pfe;
  std::vector<std::chrono::duration<double>> outliers_gl;
  std::vector<std::chrono::duration<double>> outliers_gl_ex;

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim = gl_dim3(GRID_SIZE);
  lp.group_dim = gl_dim3(TILE_SIZE);

  // Set up extra stuff
  static hc::accelerator_view av = hc::accelerator().get_default_view();


  // launch empty kernel to initialize everything first
  // timing for null kernel launch appears later

  hc::parallel_for_each(av, hc::extent<3>(lp.grid_dim.x*lp.group_dim.x,1,1).tile(lp.group_dim.x,1,1),
  [=](hc::index<3>& idx) __HC__ {
  }).wait();

  // Setting lp.cf to completion_future so we can track completion: (NULL ignores all synchronization)

  std::cout << "Iterations per test:           " << dispatch_count << "\n";
  auto wait_time_us = std::chrono::milliseconds(10);

  // Timing null pfe
  for(int i = 0; i < dispatch_count; ++i) {
    start = std::chrono::high_resolution_clock::now();
    auto cf = hc::parallel_for_each(av, hc::extent<3>(lp.grid_dim.x*lp.group_dim.x,1,1).tile(lp.group_dim.x,1,1),
    [=](hc::index<3>& idx) __HC__ {
    });
    cf.wait();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;
    elapsed_pfe.push_back(dur);
  }
  remove_outliers(elapsed_pfe, outliers_pfe);
  plot("pfe", elapsed_pfe);
  std::cout << "pfe time (us):                  " 
            << std::setprecision(8) << average(elapsed_pfe)*1000000.0 << "\n";

  // Timing null grid_launch call
  for(int i = 0; i < dispatch_count; ++i) {
    start = std::chrono::high_resolution_clock::now();
    hc::completion_future cf; // create new completion-future 
    lp.cf = &cf;
    lp.av = &av;

    nullkernel(lp, 0x0);
    //std::cout << "CF get_use_count=" << cf.get_use_count() << "is_ready=" << cf.is_ready()<< "\n";
    cf.wait();

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;
    elapsed_grid_launch.push_back(dur);
  }
  remove_outliers(elapsed_grid_launch, outliers_gl);
  plot("grid_launch", elapsed_grid_launch);
  std::cout << "grid_launch time (us):          " 
            << std::setprecision(8) << average(elapsed_grid_launch)*1000000.0 << "\n";

  return 0;
}
