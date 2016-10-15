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
                 void* const kernel_source)
    : m_name(name),
      m_ampRuntimeLibrary(ampRuntimeLibrary),
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

    // detect if C++AMP runtime is available and 
    // whether all platform library dependencies are satisfied
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
  std::string m_ampRuntimeLibrary;
  std::string m_name;
  void* m_kernel_source;
};

class OpenCLPlatformDetect : public PlatformDetect {
public:
    OpenCLPlatformDetect()
      : PlatformDetect("OpenCL", "libmcwamp_opencl.so",  cl_kernel_source) {}

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
  HSAPlatformDetect() : PlatformDetect("HSA", "libmcwamp_hsa.so",  hsa_kernel_source) {}
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

void DetermineAndGetProgram(KalmarQueue* pQueue, size_t* kernel_size, void** kernel_source, bool* needs_compilation) {
  static bool firstTime = true;
  static bool hasSPIR = false;
  static bool hasFinalized = false;

  char* kernel_env = nullptr;

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
      *kernel_size =
        (ptrdiff_t)((void *)spir_kernel_end) -
        (ptrdiff_t)((void *)spir_kernel_source);
      *kernel_source = spir_kernel_source;
      *needs_compilation = true;
    } else {
      // OpenCL path
      *kernel_size =
        (ptrdiff_t)((void *)cl_kernel_end) -
        (ptrdiff_t)((void *)cl_kernel_source);
      *kernel_source = cl_kernel_source;
      *needs_compilation = true;
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
        // check if offline finalized kernel is compatible with ISA of the HSA agent
        if ((kernel_finalized_size > 0) &&
            (pQueue->getDev()->IsCompatibleKernel((void*)kernel_finalized_size, hsa_offline_finalized_kernel_source))) {
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
      *kernel_size =
        (ptrdiff_t)((void *)hsa_offline_finalized_kernel_end) -
        (ptrdiff_t)((void *)hsa_offline_finalized_kernel_source);
      *kernel_source = hsa_offline_finalized_kernel_source;
      *needs_compilation = false;
    } else {
      *kernel_size = 
        (ptrdiff_t)((void *)hsa_kernel_end) -
        (ptrdiff_t)((void *)hsa_kernel_source);
      *kernel_source = hsa_kernel_source;
      *needs_compilation = true;
    }
  }
}

void BuildProgram(KalmarQueue* pQueue) {
  size_t kernel_size = 0;
  void* kernel_source = nullptr;
  bool needs_compilation = true;

  DetermineAndGetProgram(pQueue, &kernel_size, &kernel_source, &needs_compilation);
  pQueue->getDev()->BuildProgram((void*)kernel_size, kernel_source, needs_compilation);
}

// used in parallel_for_each.h
void *CreateKernel(std::string s, KalmarQueue* pQueue) {
  size_t kernel_size = 0;
  void* kernel_source = nullptr;
  bool needs_compilation = true;

  DetermineAndGetProgram(pQueue, &kernel_size, &kernel_source, &needs_compilation);

  return pQueue->getDev()->CreateKernel(s.c_str(), (void *)kernel_size, kernel_source, needs_compilation);
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

// Kalmar runtime bootstrap logic
class KalmarBootstrap {
private:
  RuntimeImpl* runtime;
public:
  KalmarBootstrap() : runtime(nullptr) {
    bool to_init = true;
    char* lazyinit_env = getenv("HCC_LAZYINIT");
    if (lazyinit_env != nullptr) {
      if (std::string("ON") == lazyinit_env) {
        to_init = false;
      }
    }

    if (to_init) {
      // initialize runtime
      runtime = CLAMP::GetOrInitRuntime();
  
      // get context
      KalmarContext* context = static_cast<KalmarContext*>(runtime->m_GetContextImpl());
    
      const std::vector<KalmarDevice*> devices = context->getDevices();

      for (auto dev = devices.begin(); dev != devices.end(); dev++) {

        // get default queue on the default device
        std::shared_ptr<KalmarQueue> queue = (*dev)->get_default_queue();
  
        // build kernels on the default queue on the default device
        CLAMP::BuildProgram(queue.get());
      }
    }
  }
};

} // namespace Kalmar

extern "C" void __attribute__((constructor)) __hcc_shared_library_init() {
  // this would initialize kernels when the shared library get loaded
  static Kalmar::KalmarBootstrap boot;
}

extern "C" void __attribute__((destructor)) __hcc_shared_library_fini() {
}

// conversion routines between float and half precision
static inline std::uint32_t f32_as_u32(float f) { union { float f; std::uint32_t u; } v; v.f = f; return v.u; }
static inline float u32_as_f32(std::uint32_t u) { union { float f; std::uint32_t u; } v; v.u = u; return v.f; }
static inline int clamp_int(int i, int l, int h) { return std::min(std::max(i, l), h); }

// half à float, the f16 is in the low 16 bits of the input argument ¿a¿
static inline float __convert_half_to_float(std::uint32_t a) noexcept {
  std::uint32_t u = ((a << 13) + 0x70000000U) & 0x8fffe000U;
  std::uint32_t v = f32_as_u32(u32_as_f32(u) * 0x1.0p+112f) + 0x38000000U;
  u = (a & 0x7fff) != 0 ? v : u;
  return u32_as_f32(u) * 0x1.0p-112f;
}

// float à half with nearest even rounding
// The lower 16 bits of the result is the bit pattern for the f16
static inline std::uint32_t __convert_float_to_half(float a) noexcept {
  std::uint32_t u = f32_as_u32(a);
  int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
  std::uint32_t m = ((u >> 11) & 0xffe) | ((u & 0xfff) != 0);
  std::uint32_t i = 0x7c00 | (m != 0 ? 0x0200 : 0);
  std::uint32_t n = ((std::uint32_t)e << 12) | m;
  std::uint32_t s = (u >> 16) & 0x8000;
  int b = clamp_int(1-e, 0, 13);
  std::uint32_t d = (0x1000 | m) >> b;
  d |= (d << b) != (0x1000 | m);
  std::uint32_t v = e < 1 ? d : n;
  v = (v >> 2) + (((v & 0x7) == 3) | ((v & 0x7) > 5));
  v = e > 30 ? 0x7c00 : v;
  v = e == 143 ? i : v;
  return s | v;
}

extern "C" float __gnu_h2f_ieee(unsigned short h){
  return __convert_half_to_float((std::uint32_t) h);
}

extern "C" unsigned short __gnu_f2h_ieee(float f){
  return (unsigned short)__convert_float_to_half(f);
}
