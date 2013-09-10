#ifndef INCLUDE_AMP_IMPL_H
#define INCLUDE_AMP_IMPL_H

#include <iostream>
#include <CL/cl.h>

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
inline accelerator::accelerator(): accelerator(default_accelerator) {}

inline accelerator::accelerator(const std::wstring& path): device_path(path), version(0), dedicated_memory(1<<20),
is_emulated(0), has_display(0), supports_double_precision(1), supports_limited_double_precision(0) {
  if (path != std::wstring(default_accelerator)) {
    std::wcerr << L"CLAMP: Warning: the given accelerator is not supported: ";
    std::wcerr << path << std::endl;
    return;
  }
  std::string s_desc("Default GMAC+OpenCL");
  std::copy(s_desc.begin(), s_desc.end(), std::back_inserter(description));
  if (!default_view_)
    default_view_ = new accelerator_view(0);
}

inline bool accelerator::operator==(const accelerator& other) const {
  return device_path == other.device_path;
}
inline bool accelerator::operator!=(const accelerator& other) const {
  return device_path != other.device_path;
}

inline accelerator_view& accelerator::get_default_view() const {
  return *default_view_;
}

// Accelerator view
inline accelerator_view accelerator::create_view(void) {
  accelerator_view sa(0);
  return sa;
}

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
    index<N> __r = rhs;
    __r -= lhs;
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
    index<N> __r = rhs;
    __r /= lhs;
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
    index<N> __r = rhs;
    __r %= lhs;
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
    extent<N> __r = rhs;
    __r -= lhs;
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
    extent<N> __r = rhs;
    __r /= lhs;
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
    extent<N> __r = rhs;
    __r %= lhs;
    return __r;
}

/// Concurrency::array
#define __global __attribute__((address_space(1)))
#ifndef __GPU__
/// Concurrency::array constructors that are only defined for the host
template <typename T, int N>
array<T, N>::array(int e0, int e1, accelerator_view av):
  extent(Concurrency::extent<N>(e0, e1)), accelerator_view_(av) {
  m_device.reset(GMACAllocator<T>().allocate(e0), GMACDeleter<T>());
}

template <typename T, int N>
array<T, N>::array(const Concurrency::extent<N>& ext): extent(ext),
  m_device(nullptr), accelerator_view_(accelerator().get_default_view()) {
  if (ext.size()) {
    m_device.reset(GMACAllocator<T>().allocate(ext.size()),
	GMACDeleter<T>());
  }
}

template <typename T, int N>
array<T,N>:: array(const array& other): extent(other.extent),
    accelerator_view_(other.accelerator_view_), m_device(other.m_device) {}

// 1D array constructors

template <typename T>
array<T,1>:: array(const array& other): extent(other.extent),
    accelerator_view_(other.accelerator_view_), m_device(other.m_device) {}

template <typename T>
array<T, 1>::array(int e0, accelerator_view av):
  extent(e0), accelerator_view_(av) {
  m_device.reset(GMACAllocator<T>().allocate(e0), GMACDeleter<T>());
}

template <typename T>
array<T, 1>::array(const Concurrency::extent<1>& ext): extent(ext),
  m_device(nullptr), accelerator_view_(accelerator().get_default_view()) {
  if (extent[0])
    m_device.reset(GMACAllocator<T>().allocate(ext.size()), GMACDeleter<T>());
}
#endif
#undef __global

} //namespace Concurrency
#endif //INCLUDE_AMP_IMPL_H
