//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>
#include <cassert>

#include <amp.h>
#include <mutex>

#include "mcwamp_impl.hpp"

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

// weak symbols of kernel codes

// OpenCL kernel codes
extern "C" char * cl_kernel_source[] asm ("_binary_kernel_cl_start") __attribute__((weak));
extern "C" char * cl_kernel_end[] asm ("_binary_kernel_cl_end") __attribute__((weak));

// SPIR kernel codes
extern "C" char * spir_kernel_source[] asm ("_binary_kernel_spir_start") __attribute__((weak));
extern "C" char * spir_kernel_end[] asm ("_binary_kernel_spir_end") __attribute__((weak));

// HSA kernel codes
extern "C" char * hsa_kernel_source[] asm ("_binary_kernel_brig_start") __attribute__((weak));
extern "C" char * hsa_kernel_end[] asm ("_binary_kernel_brig_end") __attribute__((weak));


// interface of C++AMP runtime implementation
struct RuntimeImpl {
  RuntimeImpl(const char* libraryName) :
    m_ImplName(libraryName),
    m_RuntimeHandle(nullptr),
    m_EnumerateDevicesImpl(nullptr),
    m_QueryDeviceInfoImpl(nullptr),
    m_CreateKernelImpl(nullptr),
    m_LaunchKernelImpl(nullptr),
    m_LaunchKernelAsyncImpl(nullptr),
    m_MatchKernelNamesImpl(nullptr),
    m_PushArgImpl(nullptr),
    m_PushArgPtrImpl(nullptr),
    m_GetAllocatorImpl(nullptr),
    isCPU(false) {
    //std::cout << "dlopen(" << libraryName << ")\n";
    m_RuntimeHandle = dlopen(libraryName, RTLD_LAZY);
    if (!m_RuntimeHandle) {
      std::cerr << "C++AMP runtime load error: " << dlerror() << std::endl;
      return;
    }
    LoadSymbols();
  }

  ~RuntimeImpl() {
    if (m_RuntimeHandle) {
      dlclose(m_RuntimeHandle);
    }
  }

  // load symbols from C++AMP runtime implementation
  void LoadSymbols() {

    m_EnumerateDevicesImpl = (EnumerateDevicesImpl_t) dlsym(m_RuntimeHandle, "EnumerateDevicesImpl");
    m_QueryDeviceInfoImpl = (QueryDeviceInfoImpl_t) dlsym(m_RuntimeHandle, "QueryDeviceInfoImpl");
    m_CreateKernelImpl = (CreateKernelImpl_t) dlsym(m_RuntimeHandle, "CreateKernelImpl");
    m_LaunchKernelImpl = (LaunchKernelImpl_t) dlsym(m_RuntimeHandle, "LaunchKernelImpl");
    m_LaunchKernelAsyncImpl = (LaunchKernelAsyncImpl_t) dlsym(m_RuntimeHandle, "LaunchKernelAsyncImpl");
    m_MatchKernelNamesImpl = (MatchKernelNamesImpl_t) dlsym(m_RuntimeHandle, "MatchKernelNamesImpl");
    m_PushArgImpl = (PushArgImpl_t) dlsym(m_RuntimeHandle, "PushArgImpl");
    m_PushArgPtrImpl = (PushArgPtrImpl_t) dlsym(m_RuntimeHandle, "PushArgPtrImpl");
    m_GetAllocatorImpl = (GetAllocatorImpl_t) dlsym(m_RuntimeHandle, "GetAllocatorImpl");

  }

  void set_cpu() { isCPU = true; }
  bool is_cpu() const { return isCPU; }

  std::string m_ImplName;
  void* m_RuntimeHandle;
  EnumerateDevicesImpl_t m_EnumerateDevicesImpl;
  QueryDeviceInfoImpl_t m_QueryDeviceInfoImpl;
  CreateKernelImpl_t m_CreateKernelImpl;
  LaunchKernelImpl_t m_LaunchKernelImpl;
  LaunchKernelAsyncImpl_t m_LaunchKernelAsyncImpl;
  MatchKernelNamesImpl_t m_MatchKernelNamesImpl;
  PushArgImpl_t m_PushArgImpl;
  PushArgPtrImpl_t m_PushArgPtrImpl;
  GetAllocatorImpl_t m_GetAllocatorImpl;
  bool isCPU;
};

namespace Concurrency {
namespace CLAMP {

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

  virtual bool detect() {
    //std::cout << "Detecting " << m_name << "...";
    // detect if kernel is available
    if (!m_kernel_source) {
      //std::cout << " kernel not found" << std::endl;
      return false;
    }
    //std::cout << " kernel found...";

    void* handle = nullptr;

    // detect if system runtime is available
    //std::cout << "dlopen(" << m_systemRuntimeLibrary << ")\n";
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
    //std::cout << "dlopen(" << m_ampRuntimeLibrary << ")\n";
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

class OpenCLPlatformDetect : public PlatformDetect {
public:
    OpenCLPlatformDetect()
      : PlatformDetect("OpenCL", "libmcwamp_opencl.so", "libOpenCL.so", cl_kernel_source) {}

  bool hasSPIR() {
    void* ocl_version_test_handle = nullptr;
    typedef int (*spir_test_t) ();
    spir_test_t test_func = nullptr;
    bool result = false;

    ocl_version_test_handle = dlopen("libmcwamp_opencl_version.so", RTLD_LAZY);
    if (!ocl_version_test_handle) {
      result = false;
    } else {
      test_func = (spir_test_t) dlsym(ocl_version_test_handle, "IsSPIRAvailable");
      if (!test_func) {
        result = false;
      } else {
        result = (test_func() > 0);
      }
    }
    if (ocl_version_test_handle)
      dlclose(ocl_version_test_handle);
    return result;
  }
};

/**
 * \brief HSA runtime detection
 */
class HSAPlatformDetect : public PlatformDetect {
public:
  HSAPlatformDetect() : PlatformDetect("HSA", "libmcwamp_hsa.so", "libhsa-runtime64.so", hsa_kernel_source) {}
};

static RuntimeImpl* LoadOpenCLRuntime() {
  RuntimeImpl* runtimeImpl = nullptr;
  // load OpenCL C++AMP runtime
  std::cout << "Use OpenCL C++AMP runtime" << std::endl;
  runtimeImpl = new RuntimeImpl("libmcwamp_opencl.so");
  if (!runtimeImpl->m_RuntimeHandle) {
    std::cerr << "Can't load OpenCL C++AMP runtime!" << std::endl;
    delete runtimeImpl;
    exit(-1);
  } else {
    //std::cout << "OpenCL C++AMP runtime loaded" << std::endl;
  }
  return runtimeImpl;
}

static RuntimeImpl* LoadHSARuntime() {
  RuntimeImpl* runtimeImpl = nullptr;
  // load HSA C++AMP runtime
  std::cout << "Use HSA C++AMP runtime" << std::endl;
  runtimeImpl = new RuntimeImpl("libmcwamp_hsa.so");
  if (!runtimeImpl->m_RuntimeHandle) {
    std::cerr << "Can't load HSA C++AMP runtime!" << std::endl;
    delete runtimeImpl;
    exit(-1);
  } else {
    //std::cout << "HSA C++AMP runtime loaded" << std::endl;
  }
  return runtimeImpl;
}

static RuntimeImpl* LoadCPURuntime() {
  RuntimeImpl* runtimeImpl = nullptr;
  // load CPU runtime
  std::cout << "Use CPU runtime" << std::endl;
  runtimeImpl = new RuntimeImpl("libmcwamp_cpu.so");
  if (!runtimeImpl->m_RuntimeHandle) {
    std::cerr << "Can't load CPU runtime!" << std::endl;
    delete runtimeImpl;
    exit(-1);
  }
  return runtimeImpl;
}

RuntimeImpl* GetOrInitRuntime() {
  static RuntimeImpl* runtimeImpl = nullptr;
  if (runtimeImpl == nullptr) {
    HSAPlatformDetect hsa_rt;
    OpenCLPlatformDetect opencl_rt;

    // force use certain C++AMP runtime from CLAMP_RUNTIME environment variable
    char* runtime_env = getenv("CLAMP_RUNTIME");
    if (runtime_env != nullptr) {
      if (std::string("HSA") == runtime_env) {
        if (hsa_rt.detect()) {
          runtimeImpl = LoadHSARuntime();
        } else {
          std::cerr << "Ignore unsupported CLAMP_RUNTIME environment variable: " << runtime_env << std::endl;
        }
      } else if (runtime_env[0] == 'C' && runtime_env[1] == 'L') {
          if (opencl_rt.detect()) {
              runtimeImpl = LoadOpenCLRuntime();
          } else {
              std::cerr << "Ignore unsupported CLAMP_RUNTIME environment variable: " << runtime_env << std::endl;
          }
      } else if(std::string("CPU") == runtime_env) {
          // CPU runtime should be available
          runtimeImpl = LoadCPURuntime();
          runtimeImpl->set_cpu();
      } else {
        std::cerr << "Ignore unknown CLAMP_RUNTIME environment variable:" << runtime_env << std::endl;
      }
    }

    // If can't determined by environment variable, try detect what can be used
    if (runtimeImpl == nullptr) {
      if (hsa_rt.detect()) {
        runtimeImpl = LoadHSARuntime();
      } else if (opencl_rt.detect()) {
        runtimeImpl = LoadOpenCLRuntime();
      } else {
          runtimeImpl = LoadCPURuntime();
          runtimeImpl->set_cpu();
          std::cerr << "No suitable runtime detected. Fall back to CPU!" << std::endl;
      }
    }
  }
  return runtimeImpl;
}

//
// implementation of C++AMP runtime interfaces
// declared in amp_runtime.h and amp_allocator.h
//

// used in amp.h
std::vector<int> EnumerateDevices() {
  int num = 0;
  std::vector<int> ret;
  int* devices = nullptr;
  GetOrInitRuntime()->m_EnumerateDevicesImpl(NULL, &num);
  assert(num > 0);
  devices = new int[num];
  GetOrInitRuntime()->m_EnumerateDevicesImpl(devices, NULL);
  for (int i = 0; i < num; ++i) {
    ret.push_back(devices[i]);
  }
  delete[] devices;
  return ret;
}

bool is_cpu()
{
    return GetOrInitRuntime()->is_cpu();
}

static bool in_kernel = false;
bool in_cpu_kernel() { return in_kernel; }
void enter_kernel() { in_kernel = true; }
void leave_kernel() { in_kernel = false; }

// used in amp_impl.h
void QueryDeviceInfo(const std::wstring& device_path,
  bool& supports_cpu_shared_memory,
  size_t& dedicated_memory,
  bool& supports_limited_double_precision,
  std::wstring& description) {
  wchar_t des[128];
  GetOrInitRuntime()->m_QueryDeviceInfoImpl(device_path.c_str(), &supports_cpu_shared_memory, &dedicated_memory, &supports_limited_double_precision, des);
  description = std::wstring(des);
}

// used in parallel_for_each.h
void *CreateKernel(std::string s) {
  // FIXME need a more elegant way
  if (GetOrInitRuntime()->m_ImplName.find("libmcwamp_opencl") != std::string::npos) {
    static bool firstTime = true;
    static bool hasSPIR = false;
    if (firstTime) {
      // force use OpenCL C kernel from CLAMP_NOSPIR environment variable
      char* kernel_env = getenv("CLAMP_NOSPIR");
      if (kernel_env == nullptr) {
          OpenCLPlatformDetect opencl_rt;
        if (opencl_rt.hasSPIR()) {
          std::cout << "Use OpenCL SPIR kernel\n";
          hasSPIR = true;
        } else {
          std::cout << "Use OpenCL C kernel\n";
        }
      } else {
        std::cout << "Use OpenCL C kernel\n";
      }
      firstTime = false;
    }
    if (hasSPIR) {
      // SPIR path
        size_t kernel_size =
        (ptrdiff_t)((void *)spir_kernel_end) -
        (ptrdiff_t)((void *)spir_kernel_source);
      return GetOrInitRuntime()->m_CreateKernelImpl(s.c_str(), (void *)kernel_size, spir_kernel_source);
    } else {
      // OpenCL path
        size_t kernel_size =
        (ptrdiff_t)((void *)cl_kernel_end) -
        (ptrdiff_t)((void *)cl_kernel_source);
      return GetOrInitRuntime()->m_CreateKernelImpl(s.c_str(), (void *)kernel_size, cl_kernel_source);
    }
  } else {
    // HSA path
       size_t kernel_size =
        (ptrdiff_t)((void *)hsa_kernel_end) -
        (ptrdiff_t)((void *)hsa_kernel_source);
     return GetOrInitRuntime()->m_CreateKernelImpl(s.c_str(), (void *)kernel_size, hsa_kernel_source);
   }
}

void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {
  GetOrInitRuntime()->m_LaunchKernelImpl(kernel, dim_ext, ext, local_size);
}

std::shared_future<void>* LaunchKernelAsync(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {
  void *ret = nullptr;
  ret = GetOrInitRuntime()->m_LaunchKernelAsyncImpl(kernel, dim_ext, ext, local_size);
  return static_cast<std::shared_future<void>*>(ret);
}


void MatchKernelNames(std::string& fixed_name) {
  char* ret = new char[fixed_name.length() * 2];
  assert(ret);
  memset(ret, 0, fixed_name.length() * 2);
  memcpy(ret, fixed_name.c_str(), fixed_name.length());
  GetOrInitRuntime()->m_MatchKernelNamesImpl(ret);
  fixed_name = ret;
  delete[] ret;
}

void PushArg(void *k_, int idx, size_t sz, const void *s) {
  GetOrInitRuntime()->m_PushArgImpl(k_, idx, sz, s);
}
void PushArgPtr(void *k_, int idx, size_t sz, const void *s) {
  GetOrInitRuntime()->m_PushArgPtrImpl(k_, idx, sz, s);
}

} // namespace CLAMP

AMPAllocator *getAllocator() {
  return static_cast<AMPAllocator*>(CLAMP::GetOrInitRuntime()->m_GetAllocatorImpl());
}
} // namespace Concurrency

