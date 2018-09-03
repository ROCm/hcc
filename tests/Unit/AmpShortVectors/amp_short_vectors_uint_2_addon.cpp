// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_short_vector.hpp>

using namespace hc;
using namespace hc::short_vector;

int main(void) {
  // More Interger Operators

  // uint_2 operator~() const [[cpu, hc]];
  {
    unsigned int a = 5u;
    unsigned int b = ~a;
    uint_2 c(a), d(b);
    uint_2 e(~c);
    assert(d == e);
  }

  // uint_2& operator%=(const uint_2& rhs) [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    a %= b;
    uint_2 e(a);
    c %= d;
    assert(c == e);
  }

  // uint_2& operator^=(const uint_2& rhs) [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    a ^= b;
    uint_2 e(a);
    c ^= d;
    assert(c == e);
  }

  // uint_2& operator|=(const uint_2& rhs) [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    a |= b;
    uint_2 e(a);
    c |= d;
    assert(c == e);
  }

  // uint_2& operator&=(const uint_2& rhs) [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    a &= b;
    uint_2 e(a);
    c &= d;
    assert(c == e);
  }

  // uint_2& operator>>=(const uint_2& rhs) [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    a >>= b;
    uint_2 e(a);
    c >>= d;
    assert(c == e);
  }

  // uint_2& operator<<=(const uint_2& rhs) [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    a <<= b;
    uint_2 e(a);
    c <<= d;
    assert(c == e);
  }

  // uint_2 operator%(const uint_2& lhs, const uint_2& rhs) 
  //   [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    uint_2 e(a % b);
    uint_2 f = c % d;
    assert(e == f); 
  }

  // uint_2 operator^(const uint_2& lhs, const uint_2& rhs) 
  //   [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    uint_2 e(a ^ b);
    uint_2 f = c ^ d;
    assert(e == f); 
  }

  // uint_2 operator|(const uint_2& lhs, const uint_2& rhs) 
  //   [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    uint_2 e(a | b);
    uint_2 f = c | d;
    assert(e == f); 
  }

  // uint_2 operator&(const uint_2& lhs, const uint_2& rhs) 
  //   [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    uint_2 e(a & b);
    uint_2 f = c & d;
    assert(e == f); 
  }

  // uint_2 operator<<(const uint_2& lhs, const uint_2& rhs) 
  //   [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    uint_2 e(a << b);
    uint_2 f = c << d;
    assert(e == f); 
  }

  // uint_2 operator>>(const uint_2& lhs, const uint_2& rhs) 
  //   [[cpu, hc]];
  {
    unsigned int a = 5u, b = 10u;
    uint_2 c(a), d(b);
    uint_2 e(a >> b);
    uint_2 f = c >> d;
    assert(e == f); 
  }

  return 0;
}
