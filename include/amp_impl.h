#ifndef INCLUDE_AMP_IMPL_H
#define INCLUDE_AMP_IMPL_H

#include <iostream>
#if __APPLE__
#include <OpenCL/cl.h>
#elif !defined(CXXAMP_ENABLE_HSA_OKRA)
#include <CL/cl.h>
#endif
#define CHECK_ERROR(error_code, message) \
  if (error_code != CL_SUCCESS) { \
    std::cout << "Error: " << message << "\n"; \
    std::cout << "Code: " << error_code << "\n"; \
    std::cout << "Line: " << __LINE__ << "\n"; \
    exit(1); \
  }

#define CHECK_ERROR_GMAC(error_code, message) \
  if (error_code != eclSuccess) { \
    std::cout << "Error: " << message << "\n"; \
    std::cout << "Code: " << error_code << "\n"; \
    std::cout << "Line: " << __LINE__ << "\n"; \
    exit(1); \
  }
// Specialization of AMP classes/templates

namespace Concurrency {
// Accelerators
inline accelerator::accelerator(): accelerator(default_accelerator) {
  if (device_path != std::wstring(default_accelerator)) {
    std::wcerr << L"CLAMP: Warning: the given accelerator is not supported: ";
    std::wcerr << device_path << std::endl;
  }

  description = L"Default GMAC+OpenCL";
  if (!default_view_) {
    default_view_ = new accelerator_view(0);
    default_view_->accelerator_ = this;
  }
}
inline accelerator::accelerator(const accelerator& other): device_path(other.device_path), version(other.version),
dedicated_memory(other.dedicated_memory), is_debug(other.is_debug), is_emulated(other.is_emulated), has_display(other.has_display),
supports_double_precision(other.supports_double_precision), supports_limited_double_precision(other.supports_limited_double_precision),
supports_cpu_shared_memory(other.supports_cpu_shared_memory) {
  if (device_path != std::wstring(default_accelerator)) {
    std::wcerr << L"CLAMP: Warning: the given accelerator is not supported: ";
    std::wcerr << device_path << std::endl;
  }

  description = L"Default GMAC+OpenCL";

  default_view_ = new accelerator_view(0);
  default_view_->accelerator_ = this;
}

// TODO(I-Jui Sung): perform real OpenCL queries here..
inline accelerator::accelerator(const std::wstring& path): device_path(path),
  version(0), is_debug(false), is_emulated(false),
  has_display(false), supports_double_precision(true),
  supports_limited_double_precision(false), supports_cpu_shared_memory(false) {
  if (path != std::wstring(default_accelerator)) {
    std::wcerr << L"CLAMP: Warning: the given accelerator is not supported: ";
    std::wcerr << path << std::endl;
  }
#ifndef CXXAMP_ENABLE_HSA_OKRA
  AcceleratorInfo accInfo;
  for (unsigned i = 0; i < eclGetNumberOfAccelerators(); i++) {
    assert(eclGetAcceleratorInfo(i, &accInfo) == eclSuccess);
    if ( (accInfo.acceleratorType == GMAC_ACCELERATOR_TYPE_GPU)
      && (path ==std::wstring(gpu_accelerator)))
      this->accInfo = accInfo;

    if ( (accInfo.acceleratorType == GMAC_ACCELERATOR_TYPE_CPU)
      && (path ==std::wstring(cpu_accelerator)))
      this->accInfo = accInfo;
  }
  dedicated_memory=accInfo.memAllocSize/(size_t)1024;

  if(accInfo.singleFPConfig & GMAC_ACCELERATOR_FP_FMA
     & GMAC_ACCELERATOR_FP_ROUND_TO_NEAREST
     & GMAC_ACCELERATOR_FP_ROUND_TO_ZERO
     & GMAC_ACCELERATOR_FP_INF_NAN
     & GMAC_ACCELERATOR_FP_DENORM)
    supports_limited_double_precision = true;
#endif
  description = L"Default GMAC+OpenCL";
  if (!default_view_) {
    default_view_ = new accelerator_view(0);
    default_view_->accelerator_ = this;
  }
}
inline accelerator& accelerator::operator=(const accelerator& other) {
  device_path = other.device_path;
  version = other.version;
  dedicated_memory = other.dedicated_memory;
  is_emulated = other.is_emulated;
  is_debug = other.is_debug;
  has_display = other.has_display;
  supports_double_precision = other.supports_double_precision;
  supports_limited_double_precision = other.supports_limited_double_precision;
  supports_cpu_shared_memory = other.supports_cpu_shared_memory;
  return *this;
}
inline bool accelerator::operator==(const accelerator& other) const {
  return device_path == other.device_path &&
         version == other.version &&
         dedicated_memory == other.dedicated_memory &&
         is_debug == other.is_debug &&
         is_emulated == other.is_emulated &&
         has_display == other.has_display &&
         supports_double_precision == other.supports_double_precision &&
         supports_cpu_shared_memory == other.supports_cpu_shared_memory;
}
inline bool accelerator::operator!=(const accelerator& other) const {
  return !(*this == other);
}

inline accelerator_view& accelerator::get_default_view() const {
  return *default_view_;
}

// Accelerator view
inline accelerator_view accelerator::create_view(void) {
  return create_view(queuing_mode_immediate);
}
inline accelerator_view accelerator::create_view(queuing_mode qmode) {
  accelerator_view sa(0);
  sa.queuing_mode = qmode;
  sa.accelerator_ = this;
  return sa;
}

inline completion_future accelerator_view::create_marker(){return completion_future();}

template <int N>
index<N> operator+(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r += rhs;
    return __r;
}
template <int N>
index<N> operator+(const index<N>& lhs, int rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r += rhs;
    return __r;
}
template <int N>
index<N> operator+(int lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r = rhs;
    __r += lhs;
    return __r;
}
template <int N>
index<N> operator-(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r -= rhs;
    return __r;
}
template <int N>
index<N> operator-(const index<N>& lhs, int rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r -= rhs;
    return __r;
}
template <int N>
index<N> operator-(int lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r;
    for (int i = 0; i < N; ++i) __r[i] = lhs;
    __r -= rhs;
    return __r;
}
template<>
inline index<1> operator-(int lhs, const index<1>& rhs) restrict(amp,cpu) {
    index<1> __r(lhs);
    __r -= rhs;
    return __r;
}
template<>
inline index<2> operator-(int lhs, const index<2>& rhs) restrict(amp,cpu) {
    index<2> __r(lhs,lhs);
    __r -= rhs;
    return __r;
}
template<>
inline index<3> operator-(int lhs, const index<3>& rhs) restrict(amp,cpu) {
    index<3> __r(lhs,lhs,lhs);
    __r -= rhs;
    return __r;
}
template <int N>
index<N> operator*(const index<N>& lhs, int rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r *= rhs;
    return __r;
}
template <int N>
index<N> operator*(int lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r = rhs;
    __r *= lhs;
    return __r;
}
template <int N>
index<N> operator/(const index<N>& lhs, int rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r /= rhs;
    return __r;
}
template <int N>
index<N> operator/(int lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r;
    for (int i = 0; i < N; ++i) __r[i] = lhs;
    __r /= rhs;
    return __r;
}
template <>
inline index<1> operator/(int lhs, const index<1>& rhs) restrict(amp,cpu) {
    index<1> __r(lhs);
    __r /= rhs;
    return __r;
}
template <>
inline index<2> operator/(int lhs, const index<2>& rhs) restrict(amp,cpu) {
    index<2> __r(lhs,lhs);
    __r /= rhs;
    return __r;
}
template <>
inline index<3> operator/(int lhs, const index<3>& rhs) restrict(amp,cpu) {
    index<3> __r(lhs,lhs,lhs);
    __r /= rhs;
    return __r;
}
template <int N>
index<N> operator%(const index<N>& lhs, int rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r %= rhs;
    return __r;
}
template <int N>
index<N> operator%(int lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r;
    for (int i = 0; i < N; ++i) __r[i] = lhs;
    __r %= rhs;
    return __r;
}
template <>
inline index<1> operator%(int lhs, const index<1>& rhs) restrict(amp,cpu) {
    index<1> __r(lhs);
    __r %= rhs;
    return __r;
}
template <>
inline index<2> operator%(int lhs, const index<2>& rhs) restrict(amp,cpu) {
    index<2> __r(lhs,lhs);
    __r %= rhs;
    return __r;
}
template <>
inline index<3> operator%(int lhs, const index<3>& rhs) restrict(amp,cpu) {
    index<3> __r(lhs,lhs,lhs);
    __r %= rhs;
    return __r;
}

template <int N>
extent<N> operator+(const extent<N>& lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r += rhs;
    return __r;
}
template <int N>
extent<N> operator+(const extent<N>& lhs, int rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r += rhs;
    return __r;
}
template <int N>
extent<N> operator+(int lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r = rhs;
    __r += lhs;
    return __r;
}
template <int N>
extent<N> operator-(const extent<N>& lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r -= rhs;
    return __r;
}
template <int N>
extent<N> operator-(const extent<N>& lhs, int rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r -= rhs;
    return __r;
}
template <int N>
extent<N> operator-(int lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r;
    for (int i = 0; i < N; ++i) __r[i] = lhs;
    __r -= rhs;
    return __r;
}
template<>
inline extent<1> operator-(int lhs, const extent<1>& rhs) restrict(amp,cpu) {
    extent<1> __r(lhs);
    __r -= rhs;
    return __r;
}
template<>
inline extent<2> operator-(int lhs, const extent<2>& rhs) restrict(amp,cpu) {
    extent<2> __r(lhs,lhs);
    __r -= rhs;
    return __r;
}
template<>
inline extent<3> operator-(int lhs, const extent<3>& rhs) restrict(amp,cpu) {
    extent<3> __r(lhs,lhs,lhs);
    __r -= rhs;
    return __r;
}
template <int N>
extent<N> operator*(const extent<N>& lhs, int rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r *= rhs;
    return __r;
}
template <int N>
extent<N> operator*(int lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r = rhs;
    __r *= lhs;
    return __r;
}
template <int N>
extent<N> operator/(const extent<N>& lhs, int rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r /= rhs;
    return __r;
}
template <int N>
extent<N> operator/(int lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r;
    for (int i = 0; i < N; ++i) __r[i] = lhs;
    __r /= rhs;
    return __r;
}
template <>
inline extent<1> operator/(int lhs, const extent<1>& rhs) restrict(amp,cpu) {
    extent<1> __r(lhs);
    __r /= rhs;
    return __r;
}
template <int N>
inline extent<2> operator/(int lhs, const extent<2>& rhs) restrict(amp,cpu) {
    extent<2> __r(lhs,lhs);
    __r /= rhs;
    return __r;
}
template <>
inline extent<3> operator/(int lhs, const extent<3>& rhs) restrict(amp,cpu) {
    extent<3> __r(lhs,lhs,lhs);
    __r /= rhs;
    return __r;
}
template <int N>
extent<N> operator%(const extent<N>& lhs, int rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r %= rhs;
    return __r;
}
template <int N>
extent<N> operator%(int lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r;
    for (int i = 0; i < N; ++i) __r[i] = lhs;
    __r %= rhs;
    return __r;
}
template <>
inline extent<1> operator%(int lhs, const extent<1>& rhs) restrict(amp,cpu) {
    extent<1> __r(lhs);
    __r %= rhs;
    return __r;
}
template <>
inline extent<2> operator%(int lhs, const extent<2>& rhs) restrict(amp,cpu) {
    extent<2> __r(lhs,lhs);
    __r %= rhs;
    return __r;
}
template <>
inline extent<3> operator%(int lhs, const extent<3>& rhs) restrict(amp,cpu) {
    extent<3> __r(lhs,lhs,lhs);
    __r %= rhs;
    return __r;
}


template<int N> class extent;
template<typename T, int N> array<T, N>::array(const Concurrency::extent<N>& ext)
    : extent(ext), m_device(nullptr), pav(nullptr), paav(nullptr) {
#ifndef __GPU__
        initialize();
#endif
    }
template<typename T, int N> array<T, N>::array(int e0)
    : array(Concurrency::extent<1>(e0)) {}
template<typename T, int N> array<T, N>::array(int e0, int e1)
    : array(Concurrency::extent<2>(e0, e1)) {}
template<typename T, int N> array<T, N>::array(int e0, int e1, int e2)
    : array(Concurrency::extent<3>(e0, e1, e2)) {}


template<typename T, int N>
array<T, N>::array(const Concurrency::extent<N>& ext, accelerator_view av, access_type cpu_access_type) : array(ext) {
  this->cpu_access_type = cpu_access_type;
  pav = new accelerator_view(av);
}
template<typename T, int N>
array<T, N>::array(int e0, accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<1>(e0), av, cpu_access_type) {}
template<typename T, int N>
array<T, N>::array(int e0, int e1, accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<2>(e0, e1), av, cpu_access_type) {}
template<typename T, int N>
array<T, N>::array(int e0, int e1, int e2, accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<3>(e0, e1, e2), av, cpu_access_type) {}


template<typename T, int N>
array<T, N>::array(const Concurrency::extent<N>& extent, accelerator_view av, accelerator_view associated_av) : array(extent) {
  pav = new accelerator_view(av);
  paav = new accelerator_view(associated_av);
}
template<typename T, int N>
array<T, N>::array(int e0, accelerator_view av, accelerator_view associated_av) : array(Concurrency::extent<1>(e0), av, associated_av) {}
template<typename T, int N>
array<T, N>::array(int e0, int e1, accelerator_view av, accelerator_view associated_av) : array(Concurrency::extent<2>(e0, e1), av, associated_av) {}
template<typename T, int N>
array<T, N>::array(int e0, int e1, int e2, accelerator_view av, accelerator_view associated_av) : array(Concurrency::extent<3>(e0, e1, e2), av, associated_av) {}


template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin)
    : extent(ext), m_device(nullptr), pav(nullptr), paav(nullptr) {
#ifndef __GPU__
        InputIterator srcEnd = srcBegin;
        std::advance(srcEnd, extent.size());
        initialize(srcBegin, srcEnd);
#endif
    }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, InputIterator srcEnd)
    : extent(ext), m_device(nullptr), pav(nullptr), paav(nullptr) {
#ifndef __GPU__
        initialize(srcBegin, srcEnd);
#endif
 }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin)
    : array(Concurrency::extent<1>(e0), srcBegin) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, InputIterator srcEnd)
    : array(Concurrency::extent<1>(e0), srcBegin, srcEnd) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin)
    : array(Concurrency::extent<2>(e0, e1), srcBegin) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd)
    : array(Concurrency::extent<2>(e0, e1), srcBegin, srcEnd) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, srcEnd) {}



template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, accelerator_view av,
                   access_type cpu_access_type) : array(ext, srcBegin) {
  this->cpu_access_type = cpu_access_type;
  pav = new accelerator_view(av);
}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, access_type cpu_access_type) : array(ext, srcBegin, srcEnd) {
  this->cpu_access_type = cpu_access_type;
  pav = new accelerator_view(av);
}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, accelerator_view av, access_type cpu_access_type)
    : array(Concurrency::extent<1>(e0), srcBegin, av, cpu_access_type) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, InputIterator srcEnd, accelerator_view av, access_type cpu_access_type)
    : array(Concurrency::extent<1>(e0), srcBegin, srcEnd, av, cpu_access_type) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, accelerator_view av, access_type cpu_access_type)
    : array(Concurrency::extent<2>(e0, e1), srcBegin, av, cpu_access_type) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<2>(e0, e1), srcBegin, srcEnd, av, cpu_access_type) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, accelerator_view av, access_type cpu_access_type)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, av, cpu_access_type) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, srcEnd, av, cpu_access_type) {}





template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, accelerator_view av,
                   accelerator_view associated_av) : array(ext, srcBegin, av, access_type_none) {
  paav = new accelerator_view(associated_av);
}
template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, accelerator_view associated_av)
    : array(ext, srcBegin, srcEnd) {
  pav = new accelerator_view(av);
  paav = new accelerator_view(associated_av);
}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, accelerator_view av,
                   accelerator_view associated_av)
    : array(Concurrency::extent<1>(e0), srcBegin, av, associated_av) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, accelerator_view associated_av)
    : array(Concurrency::extent<1>(e0), srcBegin, srcEnd, av, associated_av) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, accelerator_view av,
                   accelerator_view associated_av)
    : array(Concurrency::extent<2>(e0, e1), srcBegin, av, associated_av) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, accelerator_view associated_av)
    : array(Concurrency::extent<2>(e0, e1), srcBegin, srcEnd, av, associated_av) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, accelerator_view av,
                   accelerator_view associated_av)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, av, associated_av) {}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, accelerator_view associated_av)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, srcEnd, av, associated_av) {}


template<typename T, int N> array<T, N>::array(const array& other)
    : extent(other.extent), m_device(other.m_device), pav(other.pav), paav(other.paav) {
  if(pav) pav = new accelerator_view(*(other.pav));
  if(paav) paav = new accelerator_view(*(other.paav));
}
template<typename T, int N> array<T, N>::array(array&& other)
    : extent(other.extent), m_device(other.m_device),pav(other.pav), paav(other.paav) {
  if(pav) pav = new accelerator_view(*(other.pav));
  if(paav) paav = new accelerator_view(*(other.paav));
}
template<typename T, int N>
array<T, N>::array(const array_view<const T, N>& src, accelerator_view av,
                   access_type cpu_access_type)
    : array(src) {
  this->cpu_access_type = cpu_access_type;
  pav = new accelerator_view(av);
}
template<typename T, int N>
array<T, N>::array(const array_view<const T, N>& src, accelerator_view av,
                   accelerator_view associated_av)
    : array(src) {
  pav = new accelerator_view(av);
  paav = new accelerator_view(associated_av);
}

#define __global
//array_view<T, N>
#ifndef __GPU__

template <typename T, int N>
void array_view<T, N>::synchronize() const {
  if(p_ && cache.get())
    cache.synchronize();
}

template <typename T, int N>
completion_future array_view<T, N>::synchronize_async() const {
  assert(cache.get());
  assert(p_);
  if (extent_base == extent && offset == 0) {
      std::future<void> fut = std::async([&]() mutable {
          memmove(const_cast<void*>(reinterpret_cast<const void*>(p_)),
              reinterpret_cast<const void*>(cache.get()), extent.size() * sizeof(T));
          });
    return completion_future(fut.share());

  } else {
    std::future<void> fut = std::async([&]() mutable {
      for (int i = 0; i < extent_base[0]; ++i){
          int off = extent_base.size() / extent_base[0];
          memmove(const_cast<void*>(reinterpret_cast<const void*>(&p_[offset + i * off])),
                  reinterpret_cast<const void*>(&(cache.get()[offset + i * off])),
                  extent.size() / extent[0] * sizeof(T));
          }
      });
    return completion_future(fut.share());
  }
}

template <typename T, int N>
array_view<T, N>::array_view(const Concurrency::extent<N>& ext,
                             value_type* src) restrict(amp,cpu)
    : extent(ext), p_(src),
      cache(GMACAllocator<T>().allocate(ext.size()), GMACDeleter<T>(), src, ext.size() * sizeof(T)),
      offset(0), extent_base(ext) {}

template <typename T, int N>
array_view<T, N>::array_view(const Concurrency::extent<N>& ext) restrict(amp,cpu)
    : extent(ext), p_(nullptr),
    cache(GMACAllocator<T>().allocate(ext.size()), GMACDeleter<T>()),
    offset(0), extent_base(ext) {}

template <typename T, int N>
void array_view<T, N>::refresh() const {
    assert(cache.get());
    assert(extent == extent_base && "Only support non-sectioned view");
    assert(offset == 0 && "Only support non-sectioned view");
    cache.refresh();
}

#else // GPU implementations

template <typename T, int N>
array_view<T,N>::array_view(const Concurrency::extent<N>& ext,
                            value_type* src) restrict(amp,cpu)
    : extent(ext), p_(nullptr), cache((__global T *)(src)),
    offset(0), extent_base(ext) {}

#endif

//array_view<const T, N>
#ifndef __GPU__

template <typename T, int N>
void array_view<const T, N>::synchronize() const {
  if(p_ && cache.get())
    cache.synchronize();
}

template <typename T, int N>
completion_future array_view<const T, N>::synchronize_async() const {
  assert(cache.get());
  assert(p_);
  if (extent_base == extent && offset == 0) {
      std::future<void> fut = std::async([&]() mutable {
          memmove(const_cast<void*>(reinterpret_cast<const void*>(p_)),
              reinterpret_cast<const void*>(cache.get()), extent.size() * sizeof(T));
          });
    return completion_future(fut.share());

  } else {
    std::future<void> fut = std::async([&]() mutable {
      for (int i = 0; i < extent_base[0]; ++i){
          int off = extent_base.size() / extent_base[0];
          memmove(const_cast<void*>(reinterpret_cast<const void*>(&p_[offset + i * off])),
                  reinterpret_cast<const void*>(&(cache.get()[offset + i * off])),
                  extent.size() / extent[0] * sizeof(T));
          }
      });
    return completion_future(fut.share());
  }
}

template <typename T, int N>
array_view<const T, N>::array_view(const Concurrency::extent<N>& ext,
                             value_type* src) restrict(amp,cpu)
    : extent(ext), p_(src),
      cache(GMACAllocator<nc_T>().allocate(ext.size()), GMACDeleter<nc_T>(), const_cast<nc_T*>(src), ext.size() * sizeof(T)),
      offset(0), extent_base(ext) {}

template <typename T, int N>
void array_view<const T, N>::refresh() const {
    assert(cache.get());
    assert(extent == extent_base && "Only support non-sectioned view");
    assert(offset == 0 && "Only support non-sectioned view");
    cache.refresh();
}

#else // GPU implementations

template <typename T, int N>
array_view<const T,N>::array_view(const Concurrency::extent<N>& ext,
                            value_type* src) restrict(amp,cpu)
    : extent(ext), p_(nullptr), cache((__global nc_T *)(src)),
    offset(0), extent_base(ext) {}

#endif
#undef __global

} //namespace Concurrency
#endif //INCLUDE_AMP_IMPL_H
