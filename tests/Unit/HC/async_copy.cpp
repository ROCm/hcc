// RUN: %hc %s -std=c++14 -o %t.out && %t.out

#include <hc.hpp>
#include <hc_am.hpp>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <cstdlib>

template<typename T>
class hc_am_buffer {
  public:
    hc_am_buffer() = delete;
    hc_am_buffer(hc::accelerator& acc, uint32_t n, unsigned flags = 0) : _n(n), _flags(flags) {
      _p = (T*)hc::am_alloc(sizeof(T) * _n, acc, _flags);
      assert(_p!=nullptr);
    }
    ~hc_am_buffer() {
      if (_p!=nullptr) {
        hc::am_free(_p);
      }
    }
    T* operator()() { return _p; }
    size_t size() { return _n * sizeof(T); }
    size_t num()  { return _n; }
    hc::extent<1> extent() { return hc::extent<1>(_n); }
  private:
    T* _p;
    const uint32_t _n;
    const unsigned _flags;
};


int main(int argc, char* argv[]) {

  constexpr int n = 1024 * 4;
  hc::accelerator acc;

  hc_am_buffer<int> pinned_host_a(acc, n, amHostPinned);
  hc_am_buffer<int> device_buffer_a(acc, n);
  hc_am_buffer<int> device_buffer_b(acc, n);
  hc_am_buffer<int> device_buffer_c(acc, n);
  hc_am_buffer<int> pinned_host_b(acc, n, amHostPinned);

  std::generate_n(pinned_host_a(), n, []() {
    static int n = 0;
    return n++;
  });
  std::memset(pinned_host_b(), 0, pinned_host_b.size());

  auto acc_view = acc.get_default_view();

  acc_view.copy_async(pinned_host_a(), device_buffer_a(), pinned_host_a.size()).wait();
  hc::parallel_for_each(acc_view, device_buffer_a.extent(), 
                       [p_a = device_buffer_a(), p_b = device_buffer_b()](hc::index<1> i) [[hc]] {
    p_b[i[0]] = p_a[i[0]];
  }).wait();

  acc_view.copy_async(device_buffer_b(), device_buffer_c(), device_buffer_c.size()).wait();
  acc_view.copy_async(device_buffer_c(), pinned_host_b(), pinned_host_b.size()).wait();

  int errors = 0;
  auto a = pinned_host_a();
  auto b = pinned_host_b();
  for (int i = 0; i < pinned_host_a.num(); ++i) {
    if (*(a++) != *(b++)) ++errors;
  }

  return !(errors==0);
}
