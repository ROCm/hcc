
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <thread>

void test(std::vector<hc::accelerator_view>& v, int default_views_to_push, int views_to_create) {
  hc::accelerator acc = hc::accelerator();
  for (int i = 0; i < default_views_to_push; ++i)
    v.push_back(std::move(acc.get_default_view()));

  for (int i = 0; i < views_to_create; ++i) 
    v.push_back(std::move(acc.create_view()));
}

int main() {
  bool ret = true;

  constexpr int num_threads = 4;

  std::vector<std::thread> workers;
  std::vector<std::vector<hc::accelerator_view>> workers_views(num_threads);
  int expected_unique_views = 0;
  for (int i = 0; i < num_threads; ++i) {
    workers.push_back(std::move(std::thread(test, std::ref(workers_views[i]), i+1, i+1)));
    expected_unique_views++;      // one default view per thread
    expected_unique_views+=(i+1); // num of non-default views created by this thread
  }

  for (int i = 0; i < num_threads; ++i) {
    workers[i].join();
  }
  std::vector<hc::accelerator_view> all_views(std::move(hc::accelerator().get_all_views()));

  bool failed = all_views.size() != expected_unique_views;
  return failed;
}

