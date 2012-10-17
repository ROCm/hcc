#pragma once
namespace Concurrency {
class Serialize {
 public:
  Serialize(ecl_kernel k): k_(k), current_idx_(0) {}
  void Append(size_t sz, const void *s) {
    ecl_error err;
    err = eclSetKernelArg(k_, current_idx_++, sz, s);
    assert(err == eclSuccess);
  }
  void AppendPtr(const void *ptr) {
    ecl_error err;
    err = eclSetKernelArgPtr(k_, current_idx_++, ptr);
    assert(err == eclSuccess);
  }
 private:
  ecl_kernel k_;
  int current_idx_;
};
}
