// RUN: %clang_cc1 -std=c++amp %s -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++amp %s -fsyntax-only -verify -famp-is-device

namespace Test1 {
struct float2 {
  float x,y;
  __attribute__((hc)) float2(){}
  __attribute__((hc, cpu)) float2(float _x, float _y) : x(_x), y(_y) {}
};

typedef float2 float2_t;

__attribute__((hc)) float2 g1;
__attribute__((hc)) float2 g2(1, 2);
__attribute__((hc)) float2 g_array[] = { float2(1, 2), float2(3, 4) };
__attribute__((hc)) float2 g_array2[2]; // expected-error {{call from CPU-restricted function to AMP-restricted function}}

__attribute__((hc)) float2_t gt1;
__attribute__((hc)) float2_t gt2(1, 2);
__attribute__((hc)) float2_t gt_array[] = { float2_t(1, 2), float2_t(3, 4) };
__attribute__((hc)) float2_t gt_array2[2]; // expected-error {{call from CPU-restricted function to AMP-restricted function}}
}

namespace Test2 {
struct float2 {
  float x,y;
  __attribute__((hc, cpu)) float2(){}
  __attribute__((hc)) float2(float _x, float _y) : x(_x), y(_y) {}
};

typedef float2 float2_t;

__attribute__((hc)) float2 g1;
__attribute__((hc)) float2 g2(1, 2);
__attribute__((hc)) float2 g_array[] = { float2(1, 2), float2(3, 4) }; // expected-error {{call from CPU-restricted function to AMP-restricted function}}
__attribute__((hc)) float2 g_array2[2];

__attribute__((hc)) float2_t gt1;
__attribute__((hc)) float2_t gt2(1, 2);
__attribute__((hc)) float2_t gt_array[] = { float2_t(1, 2), float2_t(3, 4) }; // expected-error {{call from CPU-restricted function to AMP-restricted function}}
__attribute__((hc)) float2_t gt_array2[2];
}

namespace Test3 {
template<typename T> struct float2 {
  T x,y;
  __attribute__((hc, cpu)) float2(){}
  __attribute__((hc)) float2(T _x, T _y) : x(_x), y(_y) {}
};

__attribute__((hc)) float2<float> g1;
__attribute__((hc)) float2<float> g2(1, 2);
__attribute__((hc)) float2<float> g_array[] = { float2<float>(1, 2), float2<float>(3, 4) }; // expected-error {{call from CPU-restricted function to AMP-restricted function}}
__attribute__((hc)) float2<float> g_array2[2];
}

namespace Test4 {
struct float2 {
  float x,y;
  __attribute__((hc, cpu)) float2(){}
  __attribute__((hc)) float2(float _x) : x(_x), y(0) {}
  __attribute__((hc, cpu)) float2(float _x, float _y) : x(_x), y(_y) {}
};

struct S {
  float2 f;
  __attribute__((hc)) S() {}
  __attribute__((hc, cpu)) S(float x) : f(x)  {}
  __attribute__((hc, cpu)) S(float x, float y) : f(x, y)  {}
};

__attribute__((hc)) S g1;
__attribute__((hc)) S g2(1);
__attribute__((hc)) S g3(1, 2);
__attribute__((hc)) S g_array1[2]; // expected-error {{call from CPU-restricted function to AMP-restricted function}}
__attribute__((hc)) S g_array2[] = { S(1) }; // expected-error {{call from CPU-restricted function to AMP-restricted function}}
__attribute__((hc)) S g_array3[] = { S(1, 2), S(3, 4) };
__attribute__((hc)) S g_array4[] = { S(1), S(3, 4) }; // expected-error {{call from CPU-restricted function to AMP-restricted function}}
}
