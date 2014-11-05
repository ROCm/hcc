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
extern "C" void CLEnumerateDevicesImpl(int*, int*) __attribute__((weak));
extern "C" void HSAEnumerateDevicesImpl(int*, int*) __attribute__((weak));
extern "C" void CLQueryDeviceInfoImpl(const wchar_t*, bool*, size_t*, bool*, wchar_t*) __attribute__((weak));
extern "C" void HSAQueryDeviceInfoImpl(const wchar_t*, bool*, size_t*, bool*, wchar_t*) __attribute__((weak));
extern "C" void* CLCreateKernelImpl(const char*, void*, void*) __attribute__((weak));
extern "C" void* HSACreateKernelImpl(const char*, void*, void*) __attribute__((weak));
extern "C" void CLLaunchKernelImpl(void *, size_t, size_t*, size_t*) __attribute__((weak));
extern "C" void HSALaunchKernelImpl(void *, size_t, size_t*, size_t*) __attribute__((weak));
extern "C" void* CLLaunchKernelAsyncImpl(void *, size_t, size_t*, size_t*) __attribute__((weak));
extern "C" void* HSALaunchKernelAsyncImpl(void *, size_t, size_t*, size_t*) __attribute__((weak));
extern "C" void* CLMatchKernelNamesImpl(char *) __attribute__((weak));
extern "C" void* HSAMatchKernelNamesImpl(char *) __attribute__((weak));
extern "C" void* CLPushArgImpl(void *, int, size_t, const void *) __attribute__((weak));
extern "C" void* HSAPushArgImpl(void *, int, size_t, const void *) __attribute__((weak));

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
      //std::cout << dlerror() << std::endl;
      return false;
    }
    dlerror();  // clear any existing error
    //std::cout << " system runtime found...";
    dlclose(handle);

    // detect if C++AMP runtime is available
    handle = dlopen(m_ampRuntimeLibrary.c_str(), RTLD_LAZY);
    if (!handle) {
      //std::cout << " C++AMP runtime not found" << std::endl;
      //std::cout << dlerror() << std::endl;
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

// used in amp.h
std::vector<int> EnumerateDevices() {
  // FIXME use runtime detection in the future
  OpenCLPlatformDetect opencl_rt;
  int num = 0;
  std::vector<int> ret;
  int* devices = nullptr;
  if (opencl_rt.detect() && CLEnumerateDevicesImpl) {
    // OpenCL path
    CLEnumerateDevicesImpl(NULL, &num);
    assert(num > 0);
    devices = new int[num];
    CLEnumerateDevicesImpl(devices, NULL);
  } else {
    // HSA path
    HSAEnumerateDevicesImpl(NULL, &num);
    assert(num > 0);
    devices = new int[num];
    HSAEnumerateDevicesImpl(devices, NULL);
  }
  
  for (int i = 0; i < num; ++i) {
    ret.push_back(devices[i]);
  }
  delete[] devices;
  return ret;
}

// used in amp_impl.h
void QueryDeviceInfo(const std::wstring& device_path, 
  bool& supports_cpu_shared_memory,
  size_t& dedicated_memory, 
  bool& supports_limited_double_precision, 
  std::wstring& description) {
  wchar_t des[128];
  // FIXME use runtime detection in the future
  OpenCLPlatformDetect opencl_rt;
  if (opencl_rt.detect() && CLQueryDeviceInfoImpl) {
    // OpenCL path
    CLQueryDeviceInfoImpl(device_path.c_str(), &supports_cpu_shared_memory, &dedicated_memory, &supports_limited_double_precision, des);
  } else {
    // HSA path
    HSAQueryDeviceInfoImpl(device_path.c_str(), &supports_cpu_shared_memory, &dedicated_memory, &supports_limited_double_precision, des);
  }
  description = std::wstring(des);
}

// used in parallel_for_each.h
void *CreateKernel(std::string s) {
  // FIXME use runtime detection in the future
  OpenCLPlatformDetect opencl_rt;
  if (opencl_rt.detect() && CLCreateKernelImpl) {
    // OpenCL path
    return CLCreateKernelImpl(s.c_str(), cl_kernel_size, cl_kernel_source);
  } else {
    // HSA path
    return HSACreateKernelImpl(s.c_str(), hsa_kernel_size, hsa_kernel_source);
  }
}

void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {
  // FIXME use runtime detection in the future
  OpenCLPlatformDetect opencl_rt;
  if (opencl_rt.detect() && CLLaunchKernelImpl) {
    // OpenCL path
    CLLaunchKernelImpl(kernel, dim_ext, ext, local_size);
  } else {
    // HSA path
    HSALaunchKernelImpl(kernel, dim_ext, ext, local_size);
  }
}

std::future<void>* LaunchKernelAsync(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {
  // FIXME use runtime detection in the future
  OpenCLPlatformDetect opencl_rt;
  void *ret = nullptr;
  if (opencl_rt.detect() && CLLaunchKernelAsyncImpl) {
    // OpenCL path
    ret = CLLaunchKernelAsyncImpl(kernel, dim_ext, ext, local_size);
  } else {
    // HSA path
    ret = HSALaunchKernelAsyncImpl(kernel, dim_ext, ext, local_size);
  }
  return static_cast<std::future<void>*>(ret);
}


void MatchKernelNames(std::string& fixed_name) {
  // FIXME use runtime detection in the future
  OpenCLPlatformDetect opencl_rt;
  char* ret = new char[fixed_name.length() * 2];
  assert(ret);
  memset(ret, 0, fixed_name.length() * 2);
  memcpy(ret, fixed_name.c_str(), fixed_name.length());
  if (opencl_rt.detect() && CLMatchKernelNamesImpl) {
    // OpenCL path
    CLMatchKernelNamesImpl(ret);
  } else {
    // HSA path
    HSAMatchKernelNamesImpl(ret);
  }
  fixed_name = ret;
  delete[] ret;
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

void PushArg(void *k_, int idx, size_t sz, const void *s) {
  // FIXME use runtime detection in the future
  OpenCLPlatformDetect opencl_rt;
  if (opencl_rt.detect() && CLPushArgImpl) {
    // OpenCL path
    CLPushArgImpl(k_, idx, sz, s);
  } else {
    // HSA path
    HSAPushArgImpl(k_, idx, sz, s);
  }
}

} // namespace CLAMP
} // namespace Concurrency

