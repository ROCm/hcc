//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <amp_runtime.h>

namespace Concurrency {
class Serialize {
public:
  typedef void *kernel;
  Serialize(kernel k): k_(k), current_idx_(0) {}
  void Append(size_t sz, const void *s) {
    CLAMP::PushArg(k_, current_idx_++, sz, s);
  }
  void* getKernel() { 
    return k_; 
  }
  int getAndIncCurrentIndex() { 
    int ret = current_idx_; 
    current_idx_++; 
    return ret; 
  }
private:
  kernel k_;
  int current_idx_;
};
}
