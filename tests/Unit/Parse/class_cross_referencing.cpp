// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

class unorm;
class norm {
  float f;
public:
  norm(const unorm& other) restrict(cpu, amp);
};

class unorm {
  float f;

public:
  unorm() restrict(cpu, amp) {}
  unorm(const norm& other) restrict(cpu, amp) {}
};

int main(void)
{
  return 0;  // expected: success
}

