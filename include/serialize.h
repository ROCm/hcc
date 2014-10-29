//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#if !defined(CXXAMP_ENABLE_HSA)
#include <CL/opencl.h>
#endif

namespace Concurrency {
#if defined(CXXAMP_ENABLE_HSA)
namespace CLAMP {
extern void HSAPushArg(void *, size_t, const void *);
extern void HSAPushPointer(void *, void *);
}
#endif
class Serialize {
 public:
#if defined(CXXAMP_ENABLE_HSA)
  typedef void *hsa_kernel;
  Serialize(hsa_kernel k): k_(k) {}
  void AppendPtr(const void *ptr) {
    CLAMP::HSAPushPointer(k_, const_cast<void*>(ptr));
  }
  void Append(size_t sz, const void *s) {
    CLAMP::HSAPushArg(k_, sz, s);
  }
#else
  Serialize(cl_kernel k): k_(k), current_idx_(0) {}
  void Append(size_t sz, const void *s) {
    cl_int err;
    err = clSetKernelArg(k_, current_idx_++, sz, s);
    assert(err == CL_SUCCESS);
  }
#endif
 private:
#if defined(CXXAMP_ENABLE_HSA)
  hsa_kernel k_;
#else
  cl_kernel k_;
#endif
  int current_idx_;
};
}
