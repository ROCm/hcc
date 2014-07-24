//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/opencl.h>

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
  void AppendPtr(const void *ptr) {
    CLAMP::OkraPushPointer(k_, const_cast<void*>(ptr));
  }
  void Append(size_t sz, const void *s) {
    CLAMP::OkraPushArg(k_, sz, s);
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
#ifdef CXXAMP_ENABLE_HSA_OKRA
  okra_kernel k_;
#else
  cl_kernel k_;
#endif
  int current_idx_;
};
}
