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
//
//

#define BENCH_HSA 1


#include "hc.hpp"
#include "hc_am.hpp"
#include "grid_launch.hpp"
#include <iostream>
#include <fstream>

#include <chrono>
#include <vector>
#include <thread>
#include <iomanip>

#include <elf.h>
#include <hsa/hsa.h>

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


#if BENCH_HSA
static uint64_t ElfSize(const void *emi){
    const Elf64_Ehdr *ehdr = (const Elf64_Ehdr*)emi;
    const Elf64_Shdr *shdr = (const Elf64_Shdr*)((char*)emi + ehdr->e_shoff);

    uint64_t max_offset = ehdr->e_shoff;
    uint64_t total_size = max_offset + ehdr->e_shentsize * ehdr->e_shnum;

    for(uint16_t i=0;i < ehdr->e_shnum;++i){
        uint64_t cur_offset = static_cast<uint64_t>(shdr[i].sh_offset);
        if(max_offset < cur_offset){
            max_offset = cur_offset;
            total_size = max_offset;
            if(SHT_NOBITS != shdr[i].sh_type){
                total_size += static_cast<uint64_t>(shdr[i].sh_size);
            }
        }
    }
    return total_size;
}


// Load HSACO 
uint64_t load_hsaco(hc::accelerator_view *av, const char * fileName, const char *kernelName)
{
    hsa_region_t systemRegion = *(hsa_region_t*)av->get_hsa_am_system_region();
    hsa_agent_t hsaAgent = *(hsa_agent_t*) av->get_hsa_agent();

    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    std::streamsize fsize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fsize);
    if (file.read(buffer.data(), fsize))
    {
        uint64_t elfSize = ElfSize(&buffer[0]);
        assert(fsize == elfSize);

        hsa_status_t status;

        hsa_code_object_t code_object = {0};
        status = hsa_code_object_deserialize(&buffer[0], fsize, NULL, &code_object);
        assert(status == HSA_STATUS_SUCCESS);

        hsa_executable_t hsaExecutable;
        status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN,
                                       NULL, &hsaExecutable);
        assert(status == HSA_STATUS_SUCCESS);

        // Load the code object.
        status = hsa_executable_load_code_object(hsaExecutable, hsaAgent, code_object, NULL);
        assert(status == HSA_STATUS_SUCCESS);

        // Freeze the executable.
        status = hsa_executable_freeze(hsaExecutable, NULL);
        assert(status == HSA_STATUS_SUCCESS);


        // Get symbol handle.
        hsa_executable_symbol_t kernelSymbol;
        status = hsa_executable_get_symbol(hsaExecutable, NULL, kernelName, hsaAgent, 0, &kernelSymbol);


        uint64_t kernelCodeHandle;
        status = hsa_executable_symbol_get_info(kernelSymbol,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                   &kernelCodeHandle);
        assert(status == HSA_STATUS_SUCCESS);


        return kernelCodeHandle;


    } else {
        printf("could not open code object '%s'\n", fileName);
        assert(0);
    }
}

void explicit_launch_null_kernel(hc::accelerator_view *av, uint64_t kernelCodeHandle)
{
    hsa_kernel_dispatch_packet_t aql;
    memset(&aql, 0, sizeof(aql));

    aql.completion_signal.handle = 0; // signal;
    aql.grid_size_x = GRID_SIZE;
    aql.grid_size_y = 1;
    aql.grid_size_z = 1;
    aql.workgroup_size_x = TILE_SIZE;
    aql.workgroup_size_y = 1;
    aql.workgroup_size_z = 1;

    aql.group_segment_size = 0;
    aql.private_segment_size = 0;
    aql.kernarg_address = 0;
    aql.kernel_object = kernelCodeHandle;

    aql.header =   (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                        (1 << HSA_PACKET_HEADER_BARRIER) |
                        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    aql.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

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

    av->dispatch_hsa_kernel(&aql, &args, sizeof(NullKernelArgs), nullptr/*completionSignal*/);
}

#endif

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


  hc::accelerator acc = hc::accelerator();
  // Set up extra stuff
  static hc::accelerator_view av = acc.get_default_view();


  // launch empty kernel to initialize everything first
  // timing for null kernel launch appears later

  hc::parallel_for_each(av, hc::extent<3>(lp.grid_dim.x*lp.group_dim.x,1,1).tile(lp.group_dim.x,1,1),
  [=](hc::index<3>& idx) __HC__ {
  }).wait();

  // Setting lp.cf to completion_future so we can track completion: (NULL ignores all synchronization)

  std::cout << "Iterations per test:           " << dispatch_count << "\n";
  auto wait_time_us = std::chrono::milliseconds(10);

  // Timing null pfe, active wait
  for(int i = 0; i < dispatch_count; ++i) {
    start = std::chrono::high_resolution_clock::now();
    auto cf = hc::parallel_for_each(av, hc::extent<3>(lp.grid_dim.x*lp.group_dim.x,1,1).tile(lp.group_dim.x,1,1),
    [=](hc::index<3>& idx) __HC__ {
    });
    cf.wait(hc::hcWaitModeActive);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;
    elapsed_pfe.push_back(dur);
  }
  remove_outliers(elapsed_pfe, outliers_pfe);
  plot("pfe", elapsed_pfe);
  std::cout << "pfe time, active (us):                  " 
            << std::setprecision(8) << average(elapsed_pfe)*1000000.0 << "\n";


  // Timing null pfe, blocking wait
  for(int i = 0; i < dispatch_count; ++i) {
    start = std::chrono::high_resolution_clock::now();
    auto cf = hc::parallel_for_each(av, hc::extent<3>(lp.grid_dim.x*lp.group_dim.x,1,1).tile(lp.group_dim.x,1,1),
    [=](hc::index<3>& idx) __HC__ {
    });
    cf.wait(hc::hcWaitModeBlocked);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;
    elapsed_pfe.push_back(dur);
  }
  remove_outliers(elapsed_pfe, outliers_pfe);
  plot("pfe", elapsed_pfe);
  std::cout << "pfe time, blocked (us):                  " 
            << std::setprecision(8) << average(elapsed_pfe)*1000000.0 << "\n";


  // Timing null grid_launch call, active wait
  for(int i = 0; i < dispatch_count; ++i) {
    start = std::chrono::high_resolution_clock::now();
    hc::completion_future cf; // create new completion-future 
    lp.cf = &cf;
    lp.av = &av;

    nullkernel(lp, 0x0);
    //std::cout << "CF get_use_count=" << cf.get_use_count() << "is_ready=" << cf.is_ready()<< "\n";
    cf.wait(hc::hcWaitModeActive);

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;
    elapsed_grid_launch.push_back(dur);
  }
  remove_outliers(elapsed_grid_launch, outliers_gl);
  plot("grid_launch", elapsed_grid_launch);
  std::cout << "grid_launch time, active (us):          " 
            << std::setprecision(8) << average(elapsed_grid_launch)*1000000.0 << "\n";


  // Timing null grid_launch call, blocked wait
  for(int i = 0; i < dispatch_count; ++i) {
    start = std::chrono::high_resolution_clock::now();
    hc::completion_future cf; // create new completion-future 
    lp.cf = &cf;
    lp.av = &av;

    nullkernel(lp, 0x0);
    //std::cout << "CF get_use_count=" << cf.get_use_count() << "is_ready=" << cf.is_ready()<< "\n";
    cf.wait(hc::hcWaitModeBlocked);

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;
    elapsed_grid_launch.push_back(dur);
  }
  remove_outliers(elapsed_grid_launch, outliers_gl);
  plot("grid_launch", elapsed_grid_launch);
  std::cout << "grid_launch time, blocked (us):          " 
            << std::setprecision(8) << average(elapsed_grid_launch)*1000000.0 << "\n";


#define FILENAME "nullkernel.hsaco"
#define KERNEL_NAME "NullKernel"
  uint64_t kernelCodeHandle = load_hsaco(&av, FILENAME, KERNEL_NAME);

  // TODO - create and init AQL packet here:
  // TODO - move AQL code to another file?

  // Timing null grid_launch call, active wait
  for(int i = 0; i < dispatch_count; ++i) {
    start = std::chrono::high_resolution_clock::now();
    hc::completion_future cf; // create new completion-future 
    lp.cf = &cf;
    lp.av = &av;

    explicit_launch_null_kernel(&av, kernelCodeHandle);
    
    //std::cout << "CF get_use_count=" << cf.get_use_count() << "is_ready=" << cf.is_ready()<< "\n";
    cf.wait(hc::hcWaitModeActive);

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;
    elapsed_grid_launch.push_back(dur);
  }
  remove_outliers(elapsed_grid_launch, outliers_gl);
  plot("grid_launch", elapsed_grid_launch);
  std::cout << "grid_launch time, active (us):          " 
            << std::setprecision(8) << average(elapsed_grid_launch)*1000000.0 << "\n";

  return 0;
}
