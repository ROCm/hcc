// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>

using namespace concurrency;
using namespace concurrency::graphics;

int main(void) {
  // Constructor

  // unorm() restrict(cpu, amp);
  {
    unorm a;
  }

  // explicit unorm(float v) restrict(cpu, amp);
  {
    unorm a(-0.5f), b(0.0f), c(0.5f), d(1.0f), e(2.0f);
    assert(a == b);
    assert(d == e);
  }

  // explicit unorm(unsigned int v) restrict(cpu, amp);
  {
    unorm a(0u), b(1u), c(2u);
    assert(b == c);
  }

  // explicit unorm(int v) restrict(cpu, amp);
  {
    unorm a(-1), b(0), c(1), d(2);
    assert(a == b);
    assert(c == d);
  }

  // explicit unorm(double v) restrict(cpu, amp);
  {
    double a = -0.5f, b = 0.0f, c = 0.5f, d = 1.0f, e = 2.0f;
    unorm f(a), g(b), h(c), i(d), j(e);
    assert(f == g);
    assert(i == j);
  }

  // unorm(const unorm& other) restrict(cpu, amp);
  {
    unorm a(0.3f);
    unorm b(a);
    assert(a == b);
  }

  // explicit unorm(const norm& other) restrict(cpu, amp);
  {
    norm a(0.4f), b(-0.3f);
    unorm c(a), d(b);
    assert(a == c);
    assert(b != d);
  }

  // unorm& operator=(const unorm& other) restrict(cpu, amp);
  {
    unorm a(0.8f), b;
    b = a;
    assert(a == b);
  }

  // operator float(void) const restrict(cpu, amp);
  {
    unorm a(0.8f);
    float b = static_cast<float>(a);
    assert(b == 0.8f);
  }

  // unorm& operator+=(const unorm& other) restrict(cpu, amp);
  {
    unorm a(0.8f), b(0.4f);
    a += b;
    float c = static_cast<float>(a);
    assert(c == 1.0f);
  }

  // unorm& operator-=(const unorm& other) restrict(cpu, amp);
  {
    unorm a(0.8f);
    a -= a;
    float b = static_cast<float>(a);
    assert(b == 0.0f);
  }

  // unorm& operator*=(const unorm& other) restrict(cpu, amp);
  {
    unorm a(1.0f), b(2.0f);
    a *= b;
    float c = static_cast<float>(a);
    assert(c == 1.0f);
  }

  // unorm& operator/=(const unorm& other) restrict(cpu, amp);
  {
    unorm a(1.0f), b(2.0f);
    a /= b;
    float c = static_cast<float>(a);
    assert(c == 1.0f);
  }

  // unorm& operator++() restrict(cpu, amp);
  {
    unorm a(0.5f);
    ++a;
    float b = static_cast<float>(a);
    assert(b == 1.0f);
  }

  // unorm& operator++(int) restrict(cpu, amp);
  {
    unorm a(0.5f);
    a++;
    float b = static_cast<float>(a);
    assert(b == 1.0f);
  }

  // unorm& operator--() restrict(cpu, amp);
  {
    unorm a(0.5f);
    --a;
    float b = static_cast<float>(a);
    assert(b == 0.0f);
  }

  // unorm& operator--(int) restrict(cpu, amp);
  {
    unorm a(0.5f);
    a--;
    float b = static_cast<float>(a);
    assert(b == 0.0f);
  }

  // unorm operator+(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(0.5f), b(0.6f);
    unorm c = a + b;
    float d  = static_cast<float>(c);
    assert(d == 1.0f);
  }

  // unorm operator-(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(0.5f), b(0.5f);
    unorm c = a - b;
    float d  = static_cast<float>(c);
    assert(d == 0.0f);
  }

  // unorm operator*(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(1.0f), b(-1.0f);
    unorm c = a * b;
    float d  = static_cast<float>(c);
    assert(d == 0.0f);
  }

  // unorm operator/(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(1.0f), b(0.5f);
    unorm c = a / b;
    float d  = static_cast<float>(c);
    assert(d == 1.0f);
  }

  // bool operator==(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(0.5f), b(0.5f);
    assert(a == b);
  }

  // bool operator!=(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(0.5f), b(0.6f);
    assert(a != b);
  }

  // bool operator>(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(0.6f), b(0.3f);
    assert(a > b);
  }

  // bool operator<(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(-0.6f), b(2.0f);
    assert(a < b);
  }

  // bool operator>=(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(0.6f), b(-0.4f), c(-0.4f);
    assert(a >= b);
    assert(b >= c);
  }

  // bool operator<=(const unorm& lhs, const unorm& rhs) restrict(cpu, amp);
  {
    unorm a(0.6f), b(1.5f), c(2.0f);
    assert(a <= b);
    assert(b <= c);
  }

  // Constants

  // #define UNORM_ZERO ((norm)0.0f)
  {
    unorm a(0.5f), b(0.5f);
    unorm c = a - b;
    assert (c == UNORM_ZERO);
  }

  // #define UNORM_MIN ((unorm)0.0f)
  {
    unorm a(0.5f), b(0.7f);
    unorm c = a - b;
    assert (c >= UNORM_MIN);
  }

  // #define UNORM_MAX ((unorm)1.0f)
  {
    unorm a(0.9f), b(1.2f);
    unorm c = a + b;
    assert (c <= UNORM_MAX);
  }

  // Sequential v.s. Parallel
  {
    const int vecSize = 1000;

    // Alloc & init input data
    extent<1> e(vecSize);
    array<unorm, 1> a(vecSize);
    array<unorm, 1> b(vecSize);
    array<unorm, 1> c(vecSize); // Parallel results
    array<unorm, 1> d(vecSize); // Sequential results
    array_view<unorm> ga(a);
    array_view<unorm> gb(b);
    array_view<unorm> gc(c);
    array_view<unorm> gd(d);
    for (index<1> i(0); i[0] < vecSize; i++) {
      unorm tmp1(rand() / 1000.0f);
      ga[i] = tmp1;
      unorm tmp2(rand() / 1000.0f);
      gb[i] = tmp2;
    }

    parallel_for_each(
      e,
      [=](index<1> idx) restrict(amp) {
      gc[idx] = ga[idx];
      gc[idx] += (ga[idx] + gb[idx]);
      gc[idx] -= (ga[idx] - gb[idx]);
      gc[idx] *= (ga[idx] * gb[idx]);
    });

    for(unsigned i = 0; i < vecSize; i++) {
      gd[i] = ga[i];
      gd[i] += (ga[i] + gb[i]);
      gd[i] -= (ga[i] - gb[i]);
      gd[i] *= (ga[i] * gb[i]);
    }

    float sum = 0;
    for(unsigned i = 0; i < vecSize; i++) {
      sum += fast_math::fabs(fast_math::fabs(gd[i]) - fast_math::fabs(gc[i]));
    }
    return (sum > 0.1f);
  }
}
