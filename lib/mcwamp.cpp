//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "hc_rt_debug.h"
#include "mcwamp_impl.hpp"

#include <kalmar_runtime.h>

#include <iostream>
#include <string>
#include <cassert>
#include <cstddef>
#include <tuple>

#include <mutex>

#include <dlfcn.h>

// Kernel bundle
extern "C" char * kernel_bundle_source[] asm ("_binary_kernel_bundle_start") __attribute__((visibility("default")));

// interface of HCC runtime implementation
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

/**
 * \brief HSA runtime detection
 */
class HSAPlatformDetect : public PlatformDetect {
public:
  HSAPlatformDetect() : PlatformDetect("HSA", "libmcwamp_hsa.so",  kernel_bundle_source) {}
};


/**
 * \brief Flag to turn on/off platform-dependent runtime messages
 */
static bool mcwamp_verbose = false;

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


/// Handler for binary files. The bundled file will have the following format
/// (all integers are stored in little-endian format):
///
/// "OFFLOAD_BUNDLER_MAGIC_STR" (ASCII encoding of the string)
///
/// NumberOfOffloadBundles (8-byte integer)
///
/// OffsetOfBundle1 (8-byte integer)
/// SizeOfBundle1 (8-byte integer)
/// NumberOfBytesInTripleOfBundle1 (8-byte integer)
/// TripleOfBundle1 (byte length defined before)
///
/// ...
///
/// OffsetOfBundleN (8-byte integer)
/// SizeOfBundleN (8-byte integer)
/// NumberOfBytesInTripleOfBundleN (8-byte integer)
/// TripleOfBundleN (byte length defined before)
///
/// Bundle1
/// ...
/// BundleN

static inline uint64_t Read8byteIntegerFromBuffer(const char *data, size_t pos) {
  uint64_t Res = 0;
  for (unsigned i = 0; i < 8; ++i) {
    Res <<= 8;
    uint64_t Char = (uint64_t)data[pos + 7 - i];
    Res |= 0xffu & Char;
  }
  return Res;
}

#define RUNTIME_ERROR(val, error_string, line) { \
  hc::print_backtrace(); \
  printf("### HCC RUNTIME ERROR: %s at file:%s line:%d\n", error_string, __FILENAME__, line); \
  exit(val); \
}

struct _code_bundle {
  uint64_t offset;
  uint64_t size;
  uint64_t triple_size;
  const char* triple;
  const char* device_binary;
};

static void read_code_bundles(std::vector<_code_bundle>& bundles) {

  const char* bundles_data_start = (const char *)kernel_bundle_source;

  while (true) {

    static const std::string OFFLOAD_BUNDLER_MAGIC_STR("__CLANG_OFFLOAD_BUNDLE__");
    std::string bundle_magic(bundles_data_start, OFFLOAD_BUNDLER_MAGIC_STR.length());
    if (!std::equal(OFFLOAD_BUNDLER_MAGIC_STR.begin(),
                    OFFLOAD_BUNDLER_MAGIC_STR.end(),
                    bundle_magic.begin())) return;

    // skip the magic string
    const char* bundles_data_ptr = bundles_data_start + 
                                   OFFLOAD_BUNDLER_MAGIC_STR.length();
    
    // where this bundle ends
    size_t bundle_end = 0;

    // get number of bundles
    uint64_t num_bundles;
    std::memcpy(&num_bundles, bundles_data_ptr, sizeof(num_bundles));
    bundles_data_ptr += sizeof(num_bundles);
    for (uint64_t i = 0; i < num_bundles; ++i) {

      _code_bundle b = {};
 
      std::memcpy(&b.offset, bundles_data_ptr, sizeof(b.offset));
      bundles_data_ptr += sizeof(b.offset);

      std::memcpy(&b.size, bundles_data_ptr, sizeof(b.size));
      bundles_data_ptr += sizeof(b.size);

      std::memcpy(&b.triple_size, bundles_data_ptr, sizeof(b.triple_size));
      bundles_data_ptr += sizeof(b.triple_size);

      b.triple = bundles_data_ptr;
      bundles_data_ptr += b.triple_size;

      b.device_binary = bundles_data_start + b.offset;
      bundle_end = std::max(bundle_end, b.offset + b.size);

      static const std::string hcc_triple_prefix("hcc-amdgcn-amd-amdhsa--");
      std::string triple(b.triple, b.triple_size);
      if (std::equal(hcc_triple_prefix.begin(),
                     hcc_triple_prefix.end(),
                     triple.begin())) {
                     bundles.push_back(std::move(b));
      }
    }
    // bump to read the next group of bundles
    bundles_data_start += bundle_end;
  }
}

void LoadInMemoryProgram(KalmarQueue* pQueue) {

  static std::vector<_code_bundle> bundles;
  static std::once_flag f;
  std::call_once(f, [&](){ read_code_bundles(bundles); });

  for (auto&& b : bundles) {
    if (pQueue->getDev()->IsCompatibleKernel((void*) b.size, (void*) b.device_binary)) {
      pQueue->getDev()->BuildProgram((void*) b.size, (void*) b.device_binary);
    }
  }
}

// used in parallel_for_each.h
void *CreateKernel(std::string s, KalmarQueue* pQueue) {
  // TODO - should create a HSAQueue:: CreateKernel member function that creates and returns a dispatch.
  return pQueue->getDev()->CreateKernel(s.c_str(), pQueue);
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
      } else if (strtol(lazyinit_env, nullptr, 0)) {
        to_init = false;
      }
    }

    if (to_init) {
      // initialize runtime
      runtime = CLAMP::GetOrInitRuntime();

      // get context
      KalmarContext* context = static_cast<KalmarContext*>(runtime->m_GetContextImpl());

      const std::vector<KalmarDevice*> devices = context->getDevices();

      // load kernels on the default queue for each device
      for (auto dev = devices.begin(); dev != devices.end(); dev++) {

        // get default queue on the device
        std::shared_ptr<KalmarQueue> queue = (*dev)->get_default_queue();

        // load kernels on the default queue for the device
        CLAMP::LoadInMemoryProgram(queue.get());
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

// half � float, the f16 is in the low 16 bits of the input argument �a�
static inline float __convert_half_to_float(std::uint32_t a) noexcept {
  std::uint32_t u = ((a << 13) + 0x70000000U) & 0x8fffe000U;
  std::uint32_t v = f32_as_u32(u32_as_f32(u) * 0x1.0p+112f) + 0x38000000U;
  u = (a & 0x7fff) != 0 ? v : u;
  return u32_as_f32(u) * 0x1.0p-112f;
}

// float � half with nearest even rounding
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
