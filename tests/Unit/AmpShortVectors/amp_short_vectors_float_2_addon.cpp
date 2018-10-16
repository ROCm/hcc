// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_short_vector.hpp>

using namespace hc;
using namespace hc::short_vector;

int main(void) {
  
  // float_2 operator=(const float_2& other) [[cpu, hc]];
  {
    float_2 a(1.0f);
    float_2 b = a;
    assert(a == b);
  }

  // Unary Negation

  // float_2 operator-() const [[cpu, hc]];
  {
    float a = 2.0f;
    float b = -a;
    float_2 c(a), d(b);
    float_2 e(-c);
    assert(d == e);
  }

  return 0;
}
