// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_short_vectors.h>

using namespace concurrency;
using namespace concurrency::graphics;

int main(void) {
  
  // float_2 operator=(const float_2& other) restrict(cpu, amp);
  {
    float_2 a(1.0f);
    float_2 b = a;
    assert(a == b);
  }

  // Unary Negation

  // float_2 operator-() const restrict(cpu, amp);
  {
    float a = 2.0f;
    float b = -a;
    float_2 c(a), d(b);
    float_2 e(-c);
    assert(d == e);
  }

  return 0;
}
