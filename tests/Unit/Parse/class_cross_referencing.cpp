// RUN: %cxxamp %s -o %t.out && %t.out
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

