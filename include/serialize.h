#pragma once

namespace Concurrency {
#if defined(CXXAMP_ENABLE_HSA_OKRA)
namespace CLAMP {
extern void OkraPushArg(void *, size_t, const void *);
extern void OkraPushPointer(void *, void *);
}
#elif defined(CXXAMP_ENABLE_HSA)
namespace CLAMP {
extern void HSAPushArg(void *, size_t, const void *);
extern void HSAPushPointer(void *, void *);
}
#endif
class Serialize {
 public:
#if defined(CXXAMP_ENABLE_HSA_OKRA)
  typedef void *okra_kernel;
  Serialize(okra_kernel k): k_(k) {}
#elif defined(CXXAMP_ENABLE_HSA)
  typedef void *hsa_kernel;
  Serialize(hsa_kernel k): k_(k) {}
#else
  Serialize(ecl_kernel k): k_(k), current_idx_(0) {}
#endif
  void Append(size_t sz, const void *s) {
#if defined(CXXAMP_ENABLE_HSA_OKRA)
    CLAMP::OkraPushArg(k_, sz, s);
#elif defined(CXXAMP_ENABLE_HSA)
    CLAMP::HSAPushArg(k_, sz, s);
#else
    ecl_error err;
    err = eclSetKernelArg(k_, current_idx_++, sz, s);
    assert(err == eclSuccess);
#endif
  }
  void AppendPtr(const void *ptr) {
#if defined(CXXAMP_ENABLE_HSA_OKRA)
    CLAMP::OkraPushPointer(k_, const_cast<void*>(ptr));
#elif defined(CXXAMP_ENABLE_HSA)
    CLAMP::HSAPushPointer(k_, const_cast<void*>(ptr));
#else
    ecl_error err;
    err = eclSetKernelArgPtr(k_, current_idx_++, ptr);
    assert(err == eclSuccess);
#endif
  }
 private:
#if defined(CXXAMP_ENABLE_HSA_OKRA)
  okra_kernel k_;
#elif defined(CXXAMP_ENABLE_HSA)
  hsa_kernel k_;
#else
  ecl_kernel k_;
#endif
  int current_idx_;
};
}
