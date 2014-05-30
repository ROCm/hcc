//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace Concurrency {
#ifdef CXXAMP_ENABLE_HSA_OKRA
namespace CLAMP {
extern void OkraPushArg(void *, size_t, const void *);
extern void OkraPushPointer(void *, void *);
}
#endif
class Serialize {
 public:
#ifdef CXXAMP_ENABLE_HSA_OKRA
  typedef void *okra_kernel;
  Serialize(okra_kernel k): k_(k) {}
#else
  Serialize(ecl_kernel k): k_(k), current_idx_(0) {}
#endif
  void Append(size_t sz, const void *s) {
#ifdef CXXAMP_ENABLE_HSA_OKRA
    CLAMP::OkraPushArg(k_, sz, s);
#else
    ecl_error err;
    err = eclSetKernelArg(k_, current_idx_++, sz, s);
    assert(err == eclSuccess);
#endif
  }
  void AppendPtr(const void *ptr) {
#ifdef CXXAMP_ENABLE_HSA_OKRA
    CLAMP::OkraPushPointer(k_, const_cast<void*>(ptr));
#else
    ecl_error err;
    err = eclSetKernelArgPtr(k_, current_idx_++, ptr);
    assert(err == eclSuccess);
#endif
  }
 private:
#ifdef CXXAMP_ENABLE_HSA_OKRA
  okra_kernel k_;
#else
  ecl_kernel k_;
#endif
  int current_idx_;
};
}
