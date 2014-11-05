//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <amp.h>

#include <dlfcn.h>

namespace Concurrency {

// initialize static class members
const wchar_t accelerator::gpu_accelerator[] = L"gpu";
const wchar_t accelerator::cpu_accelerator[] = L"cpu";
const wchar_t accelerator::default_accelerator[] = L"default";

std::shared_ptr<accelerator> accelerator::_gpu_accelerator = std::make_shared<accelerator>(accelerator::gpu_accelerator);
std::shared_ptr<accelerator> accelerator::_cpu_accelerator = std::make_shared<accelerator>(accelerator::cpu_accelerator);
std::shared_ptr<accelerator> accelerator::_default_accelerator = nullptr;

} // namespace Concurrency

std::vector<std::string> __mcw_kernel_names;

extern "C" char * cl_kernel_source[] asm ("_binary_kernel_cl_start") __attribute__((weak));
extern "C" char * cl_kernel_size[] asm ("_binary_kernel_cl_size") __attribute__((weak));
extern "C" char * hsa_kernel_source[] asm ("_binary_kernel_brig_start") __attribute__((weak));
extern "C" char * hsa_kernel_size[] asm ("_binary_kernel_brig_size") __attribute__((weak));


namespace Concurrency {
namespace CLAMP {

// forward declaration
// FIXME remove in the near future
void* CLCreateKernel(const char*, void*, void*) __attribute__((weak));
void* HSACreateKernel(const char*, void*, void*) __attribute__((weak));

////////////////////////////////////////////////////////////
// Class declaration
////////////////////////////////////////////////////////////
/**
 * \brief Base class of platform detection
 */
class PlatformDetect {
public:
  PlatformDetect(const std::string& name, 
                 const std::string& ampRuntimeLibrary, 
                 const std::string& systemRuntimeLibrary,
                 void* const kernel_source)
    : m_name(name), 
      m_ampRuntimeLibrary(ampRuntimeLibrary), 
      m_systemRuntimeLibrary(systemRuntimeLibrary), 
      m_kernel_source(kernel_source) {}

  bool detect() {
    //std::cout << "Detecting " << m_name << "...";
    // detect if kernel is available
    if (!m_kernel_source) {
      //std::cout << " kernel not found" << std::endl;
      return false;
    }
    //std::cout << " kernel found...";

    void* handle = nullptr;

    // detect if system runtime is available
    handle = dlopen(m_systemRuntimeLibrary.c_str(), RTLD_LAZY);
    if (!handle) {
      //std::cout << " system runtime not found" << std::endl;
      return false;
    }
    dlerror();  // clear any existing error
    //std::cout << " system runtime found...";
    dlclose(handle);

    // detect if C++AMP runtime is available
    handle = dlopen(m_ampRuntimeLibrary.c_str(), RTLD_LAZY);
    if (!handle) {
      //std::cout << " C++AMP runtime not found" << std::endl;
      return false;
    }
    dlerror();  // clear any existing error
    //std::cout << " C++AMP runtime found" << std::endl;
    dlclose(handle);

    return true;
  }

private:
  std::string m_systemRuntimeLibrary;
  std::string m_ampRuntimeLibrary;
  std::string m_name;
  void* m_kernel_source;
};

/**
 * \brief OpenCL runtime detection
 */
class OpenCLPlatformDetect : public PlatformDetect {
public:
  OpenCLPlatformDetect() : PlatformDetect("OpenCL", "libmcwamp_opencl.so", "libOpenCL.so", cl_kernel_source) {}
};

/**
 * \brief HSA runtime detection
 */
class HSAPlatformDetect : public PlatformDetect {
public:
  HSAPlatformDetect() : PlatformDetect("HSA", "libmcwamp_hsa.so", "libhsa-runtime64.so", hsa_kernel_source) {}
};

// Levenshtein Distance to measure the difference of two sequences
// The shortest distance it returns the more likely the two sequences are equal
static inline int ldistance(const std::string source, const std::string target)
{
  int n = source.length();
  int m = target.length();
  if (m == 0)
    return n;
  if (n == 0)
    return m;

  //Construct a matrix
  typedef std::vector < std::vector < int >>Tmatrix;
  Tmatrix matrix(n + 1);

  for (int i = 0; i <= n; i++)
    matrix[i].resize(m + 1);
  for (int i = 1; i <= n; i++)
    matrix[i][0] = i;
  for (int i = 1; i <= m; i++)
    matrix[0][i] = i;

  for (int i = 1; i <= n; i++) {
    const char si = source[i - 1];
    for (int j = 1; j <= m; j++) {
      const char dj = target[j - 1];
      int cost;
      if (si == dj)
        cost = 0;
      else
        cost = 1;
      const int above = matrix[i - 1][j] + 1;
      const int left = matrix[i][j - 1] + 1;
      const int diag = matrix[i - 1][j - 1] + cost;
      matrix[i][j] = std::min(above, std::min(left, diag));
    }
  }
  return matrix[n][m];
}

// transformed_kernel_name (mangled) might differ if usages of 'm32' flag in CPU/GPU
// paths are mutually exclusive. We can scan all kernel names and replace
// transformed_kernel_name with the one that has the shortest distance from it by using 
// Levenshtein Distance measurement
void MatchKernelNames(std::string& fixed_name) {
  if (__mcw_kernel_names.size()) {
    // Must start from a big value > 10
    int distance = 1024;
    int hit = -1;
    std::string shortest;
    for (std::vector < std::string >::iterator it = __mcw_kernel_names.begin();
         it != __mcw_kernel_names.end(); ++it) {
      if ((*it) == fixed_name) {
        // Perfect match. Mark no need to replace and skip the loop
        hit = -1;
        break;
      }
      int n = ldistance(fixed_name, (*it));
      if (n <= distance) {
        distance = n;
        hit = 1;
        shortest = (*it);
      }
    }
    /* Replacement. Skip if not hit or the distance is too far (>5)*/
    if (hit >= 0 && distance < 5)
      fixed_name = shortest;
  }
  return;
}

void DetectRuntime() {
  HSAPlatformDetect hsa_rt;
  OpenCLPlatformDetect opencl_rt;
  if (!hsa_rt.detect()) {
    if (!opencl_rt.detect()) {
      std::cerr << "Can't load any C++AMP platform!" << std::endl;
      exit(-1);
    } else {
      // load OpenCL C++AMP runtime
      std::cout << "Use OpenCL runtime" << std::endl;
    }
  } else {
    // load HSA C++AMP runtime
    std::cout << "Use HSA runtime" << std::endl;
  }
}

void *CreateKernel(std::string s) {
  // FIXME use runtime detection in the future
  OpenCLPlatformDetect opencl_rt;
  if (opencl_rt.detect()) {
    // OpenCL path
    return CLAMP::CLCreateKernel(s.c_str(), cl_kernel_size, cl_kernel_source);
  } else {
    // HSA path
    return CLAMP::HSACreateKernel(s.c_str(), hsa_kernel_size, hsa_kernel_source);
  }
}

} // namespace CLAMP
} // namespace Concurrency

