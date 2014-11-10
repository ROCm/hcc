// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

#define LLVM_OVERRIDE override

int64_t current_pos() LLVM_OVERRIDE { return 1; }

  /// preferred_buffer_size - Determine an efficient buffer size.
size_t preferred_buffer_size() LLVM_OVERRIDE;

int main(void)
{
  return 0;
}
