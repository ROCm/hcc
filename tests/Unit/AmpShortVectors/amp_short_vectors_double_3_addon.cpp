// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_short_vectors.h>

using namespace concurrency;
using namespace concurrency::graphics;

int main(void) {
  // Two-component Access

  // double_2 get_Sxz() const restrict(cpu, amp);
  {
    double a = 1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c);
    double_2 e(a, c), f;
    f = d.get_xz();
    assert(e == f);
  }

  {
    double a = 1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c);
    double_2 e(c, a), f;
    f = d.get_zx();
    assert(e == f);
  }

  // void set_Sxz(double_2 v) restrict(cpu, amp);
  {
    double a = 1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(a, b, b);
    double_2 f(a, b);
    d.set_xz(f);
    assert(d == e);
  }

  {
    double a = 1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(b, b, a);
    double_2 f(a, b);
    d.set_zx(f);
    assert(d == e);
  }

  // Three-component Access

  // double_3 get_Sxyz() const restrict(cpu, amp);
  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(a, b, c), f;
    f = d.get_xyz();
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(a, c, b), f;
    f = d.get_xzy();
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(b, a, c), f;
    f = d.get_yxz();
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(b, c, a), f;
    f = d.get_yzx();
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(c, a, b), f;
    f = d.get_zxy();
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(c, b, a), f;
    f = d.get_zyx();
    assert(e == f);
  }

  // void set_Sxyz(double_3 v) restrict(cpu, amp);
  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(a, b, c), f;
    f.set_xyz(d);
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(a, c, b), f;
    f.set_xzy(d);
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(b, a, c), f;
    f.set_yxz(d);
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(c, a, b), f;
    f.set_yzx(d);
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(b, c, a), f;
    f.set_zxy(d);
    assert(e == f);
  }

  {
    double a = -1.2f, b = 3.4f, c = -5.6f;
    double_3 d(a, b, c), e(c, b, a), f;
    f.set_zyx(d);
    assert(e == f);
  }

  return 0;
}
