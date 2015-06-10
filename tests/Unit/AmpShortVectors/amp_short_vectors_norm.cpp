// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>

using namespace concurrency;
using namespace concurrency::graphics;

int main(void) {
  // Constructor

  // norm() restrict(cpu, amp);
  {
    norm a;
  }

  // explicit norm(float v) restrict(cpu, amp);
  {
    norm a(-2.0f), b(-1.0f), c(-0.5f), d(0.0f), e(0.5f), f(1.0f), g(2.0f);
    assert(a == b);
    assert(f == g);
  }

  // explicit norm(unsigned int v) restrict(cpu, amp);
  {
    norm a(0u), b(1u), c(2u);
    assert(b == c);
  }

  // explicit norm(int v) restrict(cpu, amp);
  {
    norm a(-2), b(-1), c(0), d(1), e(2);
    assert(a == b);
    assert(d == e);
  }

  // explicit norm(double v) restrict(cpu, amp);
  {
    double a = -2.0f, b = -1.0f, c = -0.5f, d = 0.0f, e = 0.5f, f = 1.0f, g = 2.0f;
    norm h(a), i(b), j(c), k(d), l(e), m(f), n(g);
    assert(h == i);
    assert(m == n);
  }

  // norm(const norm& other) restrict(cpu, amp);
  {
    norm a(-0.3f);
    norm b(a);
    assert(a == b);
  }

  // norm(const unorm& other) restrict(cpu, amp);
  {
    unorm a(0.4f);
    norm b(a);
    assert(a == b);
  }

  // norm& operator=(const norm& other) restrict(cpu, amp);
  {
    norm a(0.8f), b;
    b = a;
    assert(a == b);
  }

  // operator float(void) const restrict(cpu, amp);
  {
    norm a(0.8f);
    float b = static_cast<float>(a);
    assert(b == 0.8f);
  }

  // norm& operator+=(const norm& other) restrict(cpu, amp);
  {
    norm a(0.8f), b(0.4f);
    a += b;
    float c = static_cast<float>(a);
    assert(c == 1.0f);
  }

  // norm& operator-=(const norm& other) restrict(cpu, amp);
  {
    norm a(0.8f);
    a -= a;
    float b = static_cast<float>(a);
    assert(b == 0.0f);
  }

  // norm& operator*=(const norm& other) restrict(cpu, amp);
  {
    norm a(1.0f), b(2.0f);
    a *= b;
    float c = static_cast<float>(a);
    assert(c == 1.0f);
  }

  // norm& operator/=(const norm& other) restrict(cpu, amp);
  {
    norm a(1.0f), b(-1.0f);
    a /= b;
    float c = static_cast<float>(a);
    assert(c == -1.0f);
  }

  // norm& operator++() restrict(cpu, amp);
  {
    norm a(0.5f);
    ++a;
    float b = static_cast<float>(a);
    assert(b == 1.0f);
  }

  // norm& operator++(int) restrict(cpu, amp);
  {
    norm a(0.5f);
    a++;
    float b = static_cast<float>(a);
    assert(b == 1.0f);
  }

  // norm& operator--() restrict(cpu, amp);
  {
    norm a(-0.5f);
    --a;
    float b = static_cast<float>(a);
    assert(b == -1.0f);
  }

  // norm& operator--(int) restrict(cpu, amp);
  {
    norm a(-0.5f);
    a--;
    float b = static_cast<float>(a);
    assert(b == -1.0f);
  }

  // norm operator-() restrict(cpu, amp);
  {
    norm a(-2.0f);
    float b  = static_cast<float>(-a);
    assert(b == 1.0f);
  }

  // norm operator+(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(0.5f), b(0.6f);
    norm c = a + b;
    float d  = static_cast<float>(c);
    assert(d == 1.0f);
  }

  // norm operator-(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(0.5f), b(0.5f);
    norm c = a - b;
    float d  = static_cast<float>(c);
    assert(d == 0.0f);
  }

  // norm operator*(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(1.0f), b(-1.0f);
    norm c = a * b;
    float d  = static_cast<float>(c);
    assert(d == -1.0f);
  }

  // norm operator/(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(1.0f), b(-1.0f);
    norm c = a / b;
    float d  = static_cast<float>(c);
    assert(d == -1.0f);
  }

  // bool operator==(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(0.5f), b(0.5f);
    assert(a == b);
  }

  // bool operator!=(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(0.5f), b(0.6f);
    assert(a != b);
  }

  // bool operator>(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(0.6f), b(-0.7f);
    assert(a > b);
  }

  // bool operator<(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(-0.6f), b(2.0f);
    assert(a < b);
  }

  // bool operator>=(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(0.6f), b(-0.4f), c(-0.4f);
    assert(a >= b);
    assert(b >= c);
  }

  // bool operator<=(const norm& lhs, const norm& rhs) restrict(cpu, amp);
  {
    norm a(0.6f), b(1.5f), c(2.0f);
    assert(a <= b);
    assert(b <= c);
  }

  // Constants

  // #define NORM_ZERO ((norm)0.0f)
  {
    norm a(0.5f), b(0.5f);
    norm c = a - b;
    assert (c == NORM_ZERO);
  }

  // #define NORM_MIN  ((norm)-1.0f)
  {
    norm a(-0.5f), b(0.7f);
    norm c = a - b;
    assert (c >= NORM_MIN);
  }

  // #define NORM_MAX  ((norm)1.0f)
  {
    norm a(0.9f), b(1.2f);
    norm c = a + b;
    assert (c <= NORM_MAX);
  }

  // Sequential v.s. Parallel
  {
    const int vecSize = 1000;

    // Alloc & init input data
    extent<1> e(vecSize);
    array<norm, 1> a(vecSize);
    array<norm, 1> b(vecSize);
    array<norm, 1> c(vecSize); // Parallel results
    array<norm, 1> d(vecSize); // Sequential results


    array_view<norm> ga(a);
    array_view<norm> gb(b);
    array_view<norm> gc(c);
    array_view<norm> gd(d);
    for (index<1> i(0); i[0] < vecSize; i++) {
      norm tmp1(rand() / 1000.0f);
      ga[i] = tmp1;
      norm tmp2(rand() / 1000.0f);
      gb[i] = tmp2;
    }
    parallel_for_each(
      e,
      [=](index<1> idx) restrict(amp) {
      gc[idx] = -ga[idx];
      gc[idx] += (ga[idx] + gb[idx]);
      gc[idx] -= (ga[idx] - gb[idx]);
      gc[idx] *= (ga[idx] * gb[idx]);
    });

    for(unsigned i = 0; i < vecSize; i++) {
      gd[i] = -ga[i];
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
