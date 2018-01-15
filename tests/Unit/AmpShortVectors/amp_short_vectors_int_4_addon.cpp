// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_short_vectors.h>

using namespace concurrency;
using namespace concurrency::graphics;

int main(void) {
  // Three-component Access

  // int_3 get_Sxyw() const restrict(cpu, amp);
  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d);
    int_3 f(a, b, d), g;
    g = e.get_xyw();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d);
    int_3 f(a, d, b), g;
    g = e.get_xwy();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d);
    int_3 f(b, a, d), g;
    g = e.get_yxw();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d);
    int_3 f(b, d, a), g;
    g = e.get_ywx();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d);
    int_3 f(d, a, b), g;
    g = e.get_wxy();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d);
    int_3 f(d, b, a), g;
    g = e.get_wyx();
    assert(f == g);
  }

  // void set_Sxyw() restrict(cpu, amp);
  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, b, c, c);
    int_3 g(a, b, c);
    e.set_xyw(g);
    assert(e == f);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, c, c, b);
    int_3 g(a, b, c);
    e.set_xwy(g);
    assert(e == f);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, a, c, c);
    int_3 g(a, b, c);
    e.set_yxw(g);
    assert(e == f);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(c, a, c, b);
    int_3 g(a, b, c);
    e.set_ywx(g);
    assert(e == f);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, c, c, a);
    int_3 g(a, b, c);
    e.set_wxy(g);
    assert(e == f);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(c, b, c, a);
    int_3 g(a, b, c);
    e.set_wyx(g);
    assert(e == f);
  }

  // Four-component Access

  // int_4 get_Sxyzw() const restrict(cpu, amp);
  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, b, c, d), g;
    g = e.get_xyzw();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, b, d, c), g;
    g = e.get_xywz();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, c, b, d), g;
    g = e.get_xzyw();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, c, d, b), g;
    g = e.get_xzwy();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, d, b, c), g;
    g = e.get_xwyz();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, d, c, b), g;
    g = e.get_xwzy();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, a, c, d), g;
    g = e.get_yxzw();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, a, d, c), g;
    g = e.get_yxwz();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, c, a, d), g;
    g = e.get_yzxw();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, c, d, a), g;
    g = e.get_yzwx();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, d, a, c), g;
    g = e.get_ywxz();
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, d, c, a), g;
    g = e.get_ywzx();
    assert(f == g);
  }

  // void set_Sxyzw(int_4 v) restrict(cpu, amp);
  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, b, c, d), g;
    g.set_xyzw(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, b, d, c), g;
    g.set_xywz(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, c, b, d), g;
    g.set_xzyw(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, d, b, c), g;
    g.set_xzwy(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, c, d, b), g;
    g.set_xwyz(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(a, d, c, b), g;
    g.set_xwzy(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, c, a, d), g;
    g.set_zxyw(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(b, d, a, c), g;
    g.set_zxwy(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(c, b, a, d), g;
    g.set_zyxw(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(d, b, a, c), g;
    g.set_zywx(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(c, d, a, b), g;
    g.set_zwxy(e);
    assert(f == g);
  }

  {
    int a = -1, b = 2, c = -3, d = 4;
    int_4 e(a, b, c, d), f(d, c, a, b), g;
    g.set_zwyx(e);
    assert(f == g);
  }

  return 0;
}
