//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "hc_rt_debug.h"
#include "mcwamp_impl.hpp"
#include "../hc2/external/elfio/elfio.hpp"

#include <kalmar_runtime.h>

#include <iostream>
#include <string>
#include <cassert>
#include <cstddef>
#include <tuple>

#include <mutex>

#include <link.h>
#include <dlfcn.h>

#define XSTRINGIFY(X) STRINGIFY(X)
#define STRINGIFY(X) #X
#define LIB_NAME_WITH_VERSION(library) library "." XSTRINGIFY(HCC_MAJOR_VERSION) "."  XSTRINGIFY(HCC_MINOR_VERSION)

// interface of HCC runtime implementation
struct RuntimeImpl {
  RuntimeImpl(const char* libraryName) :
    m_ImplName(libraryName),
    m_RuntimeHandle(nullptr),
    m_PushArgImpl(nullptr),
    m_PushArgPtrImpl(nullptr),
    m_GetContextImpl(nullptr),
    m_ShutdownImpl(nullptr),
    m_InitActivityCallbackImpl(nullptr),
    m_EnableActivityCallbackImpl(nullptr),
    m_GetCmdNameImpl(nullptr),
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
      m_ShutdownImpl();
      // shutdown call above has already cleaned up all resources,
      // dlclose not needed and was causing seg fault for some applications
      //dlclose(m_RuntimeHandle);
    }
  }

  // load symbols from C++AMP runtime implementation
  void LoadSymbols() {
    m_PushArgImpl = (PushArgImpl_t) dlsym(m_RuntimeHandle, "PushArgImpl");
    m_PushArgPtrImpl = (PushArgPtrImpl_t) dlsym(m_RuntimeHandle, "PushArgPtrImpl");
    m_GetContextImpl= (GetContextImpl_t) dlsym(m_RuntimeHandle, "GetContextImpl");
    m_ShutdownImpl= (ShutdownImpl_t) dlsym(m_RuntimeHandle, "ShutdownImpl");
    m_InitActivityCallbackImpl = (InitActivityCallbackImpl_t) dlsym(m_RuntimeHandle, "InitActivityCallbackImpl");
    m_EnableActivityCallbackImpl = (EnableActivityCallbackImpl_t) dlsym(m_RuntimeHandle, "EnableActivityCallbackImpl");
    m_GetCmdNameImpl = (GetCmdNameImpl_t) dlsym(m_RuntimeHandle, "GetCmdNameImpl");
  }

  void set_cpu() { isCPU = true; }
  bool is_cpu() const { return isCPU; }

  std::string m_ImplName;
  void* m_RuntimeHandle;
  PushArgImpl_t m_PushArgImpl;
  PushArgPtrImpl_t m_PushArgPtrImpl;
  GetContextImpl_t m_GetContextImpl;
  ShutdownImpl_t m_ShutdownImpl;

  // Activity profiling routines
  InitActivityCallbackImpl_t m_InitActivityCallbackImpl;
  EnableActivityCallbackImpl_t m_EnableActivityCallbackImpl;
  GetCmdNameImpl_t m_GetCmdNameImpl;

  bool isCPU;
};

namespace Kalmar {
namespace CLAMP {

////////////////////////////////////////////////////////////
// Class declaration
////////////////////////////////////////////////////////////

static std::string& get_library_path()
{
    static std::string library_path;
    static std::once_flag once;
    std::call_once(once, [] () {
        // determine the search path for libmcwamp_hsa based on the
        // path of this library
        dl_iterate_phdr([](struct dl_phdr_info* info, size_t size, void* data) {
          if (info->dlpi_name) {
            std::string p(info->dlpi_name);
            auto pos = p.find("libmcwamp.so");
            if (pos != std::string::npos) {
              p.erase(p.begin()+pos, p.end());
              library_path = std::move(p);
              return 1;
            }
          }
          return 0;
        } , nullptr);
    });

    return library_path;
}

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
    void* handle = nullptr;

    // detect if C++AMP runtime is available and
    // whether all platform library dependencies are satisfied
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
  std::string m_ampRuntimeLibrary;
  std::string m_name;
  void* m_kernel_source;
};

/**
 * \brief HSA runtime detection
 */
class HSAPlatformDetect : public PlatformDetect {
public:
  HSAPlatformDetect() : PlatformDetect("HSA", get_library_path() + LIB_NAME_WITH_VERSION("libmcwamp_hsa.so"), nullptr) {}
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
  std::string lib = get_library_path() + LIB_NAME_WITH_VERSION("libmcwamp_hsa.so");
  runtimeImpl = new RuntimeImpl(lib.c_str());
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
  std::string lib = get_library_path() + LIB_NAME_WITH_VERSION("libmcwamp_cpu.so"); 
  runtimeImpl = new RuntimeImpl(lib.c_str());
  if (!runtimeImpl->m_RuntimeHandle) {
    std::cerr << "Can't load CPU runtime!" << std::endl;
    delete runtimeImpl;
    exit(-1);
  }
  return runtimeImpl;
}

static RuntimeImpl* GetOrInitRuntime_impl() {
  RuntimeImpl* runtimeImpl = nullptr;
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
  return runtimeImpl;
}

static std::unique_ptr<RuntimeImpl> runtime;
RuntimeImpl* GetOrInitRuntime() {
  static std::once_flag f;
  std::call_once(f, []() {
    runtime = std::move(std::unique_ptr<RuntimeImpl>(GetOrInitRuntime_impl()));
  });
  return runtime.get();
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


template<typename P>
inline
ELFIO::section* find_section_if(ELFIO::elfio& reader, P p) {
    const auto it = std::find_if(
        reader.sections.begin(), reader.sections.end(), std::move(p));

    return it != reader.sections.end() ? *it : nullptr;
}


static std::vector<std::vector<char>>& get_code_blobs() {

  static std::once_flag f;
  static std::vector<std::vector<char>> blobs{};

  std::call_once(f, []() {

    dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void*) {
      ELFIO::elfio tmp;

      const auto elf =
        (info->dlpi_addr && *info->dlpi_name != '\0') ? info->dlpi_name : "/proc/self/exe";

      if (!tmp.load(elf)) return 0;

      const auto it = find_section_if(tmp, [](const ELFIO::section* x) {
        return x->get_name() == ".kernel";
      });

      if (!it) return 0;

      blobs.emplace_back(it->get_data(), it->get_data() + it->get_size());

      return 0;
    }, nullptr);
  });

  return blobs;
}

static void read_code_bundles(std::vector<_code_bundle>& bundles) {

  std::vector<std::vector<char>>& blobs = get_code_blobs();
  static char* bundles_data_start = nullptr;

  for (auto &b : blobs) {

    const char* bundles_data_start = b.data();

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

// Activity profiling routines
void InitActivityCallback(void* id_callback, void* op_callback, void* arg) {
  GetOrInitRuntime()->m_InitActivityCallbackImpl(id_callback, op_callback, arg);
}
bool EnableActivityCallback(uint32_t op, bool enable) {
  return GetOrInitRuntime()->m_EnableActivityCallbackImpl(op, enable);
}
const char* GetCmdName(uint32_t id) {
  return GetOrInitRuntime()->m_GetCmdNameImpl(id);
}

} // namespace CLAMP

KalmarContext *getContext() {
  return static_cast<KalmarContext*>(CLAMP::GetOrInitRuntime()->m_GetContextImpl());
}

// Kalmar runtime bootstrap logic
class KalmarBootstrap {
public:
  KalmarBootstrap() {

    bool to_init = false;
    char* lazyinit_env = getenv("HCC_LAZYINIT");
    if (lazyinit_env != nullptr) {
      if (std::string("OFF") == lazyinit_env) {
        to_init = true;
      } else if (std::string("0") == lazyinit_env) {
        to_init = true;
      }
    }

    if (to_init) {
      const std::vector<KalmarDevice*> devices = getContext()->getDevices();

      // load kernels on the default queue for each device
      for (auto dev = devices.begin(); dev != devices.end(); dev++) {

        // get default queue on the device
        std::shared_ptr<KalmarQueue> queue = (*dev)->get_default_queue();

        // load kernels on the default queue for the device
        CLAMP::LoadInMemoryProgram(queue.get());
      }
    }
    else {
      // Instead of loading kernels, we still load code blobs eagerly.
      // They are used later when the first kernel is requested.
      // Return value is discarded here, but used by read_code_bundles().
      (void)CLAMP::get_code_blobs();
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
