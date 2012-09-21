#pragma once
namespace Concurrency {
class Serialize {
 public:
  Serialize(cl_context c, cl_kernel k): c_(c), k_(k), current_idx_(0) {}
  void Append(size_t sz, const void *s) {
    cl_int err;
    err = clSetKernelArg(k_, current_idx_++, sz, s);
    assert(err == CL_SUCCESS);
  }
  cl_context getContext(void) const {return c_;}
 private:
  cl_kernel k_;
  cl_context c_;
  int current_idx_;
};
}
