
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>

// test hc::accelerator::get_all_views() with 4 threads

void test(std::vector<hc::accelerator_view>& v, int default_views_to_push, int views_to_create) {

  hc::accelerator acc = hc::accelerator();

  for (int i = 0; i < default_views_to_push; ++i)
    v.push_back(acc.get_default_view());

  for (int i = 0; i < views_to_create; ++i) 
    v.push_back(acc.create_view());
}

int main() {
  bool ret = true;

  // HCC_LAZYINIT default was changed from OFF to ON.
  // If old behavior is requested, it changes the total number of views.
  int maybe_additional_view = 0;
  char* lazyinit_env = std::getenv("HCC_LAZYINIT");
  if (lazyinit_env != nullptr) {
    if (std::string("OFF") == lazyinit_env) {
      maybe_additional_view = 1;
    } else if (std::string("0") == lazyinit_env) {
      maybe_additional_view = 1;
    }
  }

  std::vector<hc::accelerator_view> v1;
  std::vector<hc::accelerator_view> v2;
  std::vector<hc::accelerator_view> v3;
  std::vector<hc::accelerator_view> v4;

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(1, 16);

  int d1 = dis(gen); int d2 = dis(gen); int d3 = dis(gen); int d4 = dis(gen);
  int c1 = dis(gen); int c2 = dis(gen); int c3 = dis(gen); int c4 = dis(gen);

  // fire 4 threads to create accelerator_view instances
  std::thread t1(test, std::ref(v1), d1, c1);
  std::thread t2(test, std::ref(v2), d2, c2);
  std::thread t3(test, std::ref(v3), d3, c3);
  std::thread t4(test, std::ref(v4), d4, c4);

  t1.join(); t2.join(); t3.join(); t4.join();

  hc::accelerator acc = hc::accelerator();

  std::vector<hc::accelerator_view> v5 = acc.get_all_views();

  ret &= (v1.size() == (d1 + c1));
  ret &= (v2.size() == (d2 + c2));
  ret &= (v3.size() == (d3 + c3));
  ret &= (v4.size() == (d4 + c4));

#if !TLS_QUEUE
  ret &= (v5.size() == (c1 + c2 + c3 + c4 + maybe_additional_view));
#else
  // now default accelerator view is thread local, so the total number of views
  // would the the number of threads (4), plus additional views created in each
  // thread, plus another one inside HCC runtime if lazy init is disabled
  ret &= (v5.size() == (4 + c1 + c2 + c3 + c4 + maybe_additional_view));
#endif

  return !(ret == true);
}

