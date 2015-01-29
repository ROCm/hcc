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
  Serialize(kernel k, int sync = 1): k_(k), current_idx_(0), sync(sync) {}
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
  int get_sync() const { return sync; }
private:
  kernel k_;
  int current_idx_;
  int sync;
};
}
