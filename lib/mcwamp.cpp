//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>
#include <cassert>
#include <tuple>

#include <amp.h>
#include <mutex>

#include "mcwamp_impl.hpp"

#include <dlfcn.h>

namespace Concurrency {

const wchar_t accelerator::cpu_accelerator[] = L"cpu";
const wchar_t accelerator::default_accelerator[] = L"default";

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

// HSA offline finalized kernel codes
extern "C" char * hsa_offline_finalized_kernel_source[] asm ("_binary_kernel_isa_start") __attribute__((weak));
extern "C" char * hsa_offline_finalized_kernel_end[] asm ("_binary_kernel_isa_end") __attribute__((weak));

// tame TLS objects
extern "C" int __cxa_thread_atexit(void (*func)(), void *obj,
                                   void *dso_symbol) {
  int __cxa_thread_atexit_impl(void (*)(), void *, void *);
  return __cxa_thread_atexit_impl(func, obj, dso_symbol);
}

// interface of C++AMP runtime implementation
struct RuntimeImpl {
  RuntimeImpl(const char* libraryName) :
    m_ImplName(libraryName),
    m_RuntimeHandle(nullptr),
    m_PushArgImpl(nullptr),
    m_PushArgPtrImpl(nullptr),
    m_GetContextImpl(nullptr),
    isCPU(false) {
    //std::cout << "dlopen(" << libraryName << ")\n";
    m_RuntimeHandle = dlopen(libraryName, RTLD_LAZY|RTLD_NODELETE);
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
    m_PushArgImpl = (PushArgImpl_t) dlsym(m_RuntimeHandle, "PushArgImpl");
    m_PushArgPtrImpl = (PushArgPtrImpl_t) dlsym(m_RuntimeHandle, "PushArgPtrImpl");
    m_GetContextImpl= (GetContextImpl_t) dlsym(m_RuntimeHandle, "GetContextImpl");
  }

  void set_cpu() { isCPU = true; }
  bool is_cpu() const { return isCPU; }

  std::string m_ImplName;
  void* m_RuntimeHandle;
  PushArgImpl_t m_PushArgImpl;
  PushArgPtrImpl_t m_PushArgPtrImpl;
  GetContextImpl_t m_GetContextImpl;
  bool isCPU;
};

namespace Kalmar {
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
    handle = dlopen(m_systemRuntimeLibrary.c_str(), RTLD_LAZY|RTLD_NODELETE);
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
    handle = dlopen(m_ampRuntimeLibrary.c_str(), RTLD_LAZY|RTLD_NODELETE);
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

    ocl_version_test_handle = dlopen("libmcwamp_opencl_version.so", RTLD_LAZY|RTLD_NODELETE);
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


/**
 * \brief Flag to turn on/off platform-dependent runtime messages
 */
static bool mcwamp_verbose = false;

static RuntimeImpl* LoadOpenCLRuntime() {
  RuntimeImpl* runtimeImpl = nullptr;
  // load OpenCL C++AMP runtime
  if (mcwamp_verbose)
    std::cout << "Use OpenCL runtime" << std::endl;
  runtimeImpl = new RuntimeImpl("libmcwamp_opencl.so");
  if (!runtimeImpl->m_RuntimeHandle) {
    std::cerr << "Can't load OpenCL runtime!" << std::endl;
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
  if (mcwamp_verbose)
    std::cout << "Use HSA runtime" << std::endl;
  runtimeImpl = new RuntimeImpl("libmcwamp_hsa.so");
  if (!runtimeImpl->m_RuntimeHandle) {
    std::cerr << "Can't load HSA runtime!" << std::endl;
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
  if (mcwamp_verbose)
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

    char* verbose_env = getenv("HCC_VERBOSE");
    if (verbose_env != nullptr) {
      if (std::string("ON") == verbose_env) {
        mcwamp_verbose = true;
      }
    }

    // force use certain C++AMP runtime from HCC_RUNTIME environment variable
    char* runtime_env = getenv("HCC_RUNTIME");
    if (runtime_env != nullptr) {
      if (std::string("HSA") == runtime_env) {
        if (hsa_rt.detect()) {
          runtimeImpl = LoadHSARuntime();
        } else {
          std::cerr << "Ignore unsupported HCC_RUNTIME environment variable: " << runtime_env << std::endl;
        }
      } else if (runtime_env[0] == 'C' && runtime_env[1] == 'L') {
          if (opencl_rt.detect()) {
              runtimeImpl = LoadOpenCLRuntime();
          } else {
              std::cerr << "Ignore unsupported HCC_RUNTIME environment variable: " << runtime_env << std::endl;
          }
      } else if(std::string("CPU") == runtime_env) {
          // CPU runtime should be available
          runtimeImpl = LoadCPURuntime();
          runtimeImpl->set_cpu();
      } else {
        std::cerr << "Ignore unknown HCC_RUNTIME environment variable:" << runtime_env << std::endl;
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

bool is_cpu()
{
    return GetOrInitRuntime()->is_cpu();
}

static bool in_kernel = false;
bool in_cpu_kernel() { return in_kernel; }
void enter_kernel() { in_kernel = true; }
void leave_kernel() { in_kernel = false; }

// used in parallel_for_each.h
void *CreateKernel(std::string s, KalmarQueue* pQueue) {
  static bool firstTime = true;
  static bool hasSPIR = false;
  static bool hasFinalized = false;

  char* kernel_env = nullptr;
  size_t kernel_size = 0;

  // FIXME need a more elegant way
  if (GetOrInitRuntime()->m_ImplName.find("libmcwamp_opencl") != std::string::npos) {
    if (firstTime) {
      // force use OpenCL C kernel from HCC_NOSPIR environment variable
      kernel_env = getenv("HCC_NOSPIR");
      if (kernel_env == nullptr) {
          OpenCLPlatformDetect opencl_rt;
        if (opencl_rt.hasSPIR()) {
          if (mcwamp_verbose)
            std::cout << "Use OpenCL SPIR kernel\n";
          hasSPIR = true;
        } else {
          if (mcwamp_verbose)
            std::cout << "Use OpenCL C kernel\n";
        }
      } else {
        if (mcwamp_verbose)
          std::cout << "Use OpenCL C kernel\n";
      }
      firstTime = false;
    }
    if (hasSPIR) {
      // SPIR path
      kernel_size =
        (ptrdiff_t)((void *)spir_kernel_end) -
        (ptrdiff_t)((void *)spir_kernel_source);
      return pQueue->getDev()->CreateKernel(s.c_str(), (void *)kernel_size, spir_kernel_source, true);
    } else {
      // OpenCL path
      kernel_size =
        (ptrdiff_t)((void *)cl_kernel_end) -
        (ptrdiff_t)((void *)cl_kernel_source);
      return pQueue->getDev()->CreateKernel(s.c_str(), (void *)kernel_size, cl_kernel_source, true);
    }
  } else {
    // HSA path

    if (firstTime) {
      // force use HSA BRIG kernel from HCC_NOISA environment variable
      kernel_env = getenv("HCC_NOISA");
      if (kernel_env == nullptr) {
        // check if offline finalized kernels are available
        size_t kernel_finalized_size = 
          (ptrdiff_t)((void *)hsa_offline_finalized_kernel_end) -
          (ptrdiff_t)((void *)hsa_offline_finalized_kernel_source);
        if (kernel_finalized_size > 0) {
          if (mcwamp_verbose)
            std::cout << "Use offline finalized HSA kernels\n";
          hasFinalized = true;
        } else {
          if (mcwamp_verbose)
            std::cout << "Use HSA BRIG kernel\n";
        }
      } else {
        // force use BRIG kernel
        if (mcwamp_verbose)
          std::cout << "Use HSA BRIG kernel\n";
      }
      firstTime = false;
    }
    if (hasFinalized) {
      kernel_size =
        (ptrdiff_t)((void *)hsa_offline_finalized_kernel_end) -
        (ptrdiff_t)((void *)hsa_offline_finalized_kernel_source);
      return pQueue->getDev()->CreateKernel(s.c_str(), (void *)kernel_size, hsa_offline_finalized_kernel_source, false);
    } else {
      kernel_size = 
        (ptrdiff_t)((void *)hsa_kernel_end) -
        (ptrdiff_t)((void *)hsa_kernel_source);
      return pQueue->getDev()->CreateKernel(s.c_str(), (void *)kernel_size, hsa_kernel_source, true);
    }
  }
}

void PushArg(void *k_, int idx, size_t sz, const void *s) {
  GetOrInitRuntime()->m_PushArgImpl(k_, idx, sz, s);
}
void PushArgPtr(void *k_, int idx, size_t sz, const void *s) {
  GetOrInitRuntime()->m_PushArgPtrImpl(k_, idx, sz, s);
}

} // namespace CLAMP

KalmarContext *getContext() {
  return static_cast<KalmarContext*>(CLAMP::GetOrInitRuntime()->m_GetContextImpl());
}

} // namespace Kalmar
