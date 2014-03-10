#pragma once
namespace Concurrency {
class Serialize {
 public:
#ifndef CXXAMP_ENABLE_HSA_OKRA
  Serialize(ecl_kernel k): k_(k), current_idx_(0) {}
#endif
  void Append(size_t sz, const void *s) {
#ifdef CXXAMP_ENABLE_HSA_OKRA
#else
    ecl_error err;
    err = eclSetKernelArg(k_, current_idx_++, sz, s);
    assert(err == eclSuccess);
#endif
  }
  void AppendPtr(const void *ptr) {
#ifdef CXXAMP_ENABLE_HSA_OKRA
#else
    ecl_error err;
    err = eclSetKernelArgPtr(k_, current_idx_++, ptr);
    assert(err == eclSuccess);
#endif
  }
 private:
#ifndef CXXAMP_ENABLE_HSA_OKRA
  ecl_kernel k_;
#endif
  int current_idx_;
};
}
