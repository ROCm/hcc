// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>

class unorm;
class norm {
  float f;
public:
  norm(const unorm& other) [[cpu, hc]];
};

class unorm {
  float f;

public:
  unorm() [[cpu, hc]] {}
  unorm(const norm& other) [[cpu, hc]] {}
};

int main(void)
{
  return 0;  // expected: success
}

