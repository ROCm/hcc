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
accelerator::accelerator(): accelerator(default_accelerator) {}

accelerator::accelerator(const std::wstring& path): device_path(path) {
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

bool accelerator::operator==(const accelerator& other) const {
  return device_path == other.device_path;
}
bool accelerator::operator!=(const accelerator& other) const {
  return device_path != other.device_path;
}

accelerator_view& accelerator::get_default_view() const {
  return *default_view_;
}

accelerator_view *accelerator::default_view_ = NULL;
const wchar_t accelerator::direct3d_ref[] = L"direct3d\\ref";
const wchar_t accelerator::default_accelerator[] = L"default";

// Accelerator view
accelerator_view accelerator::create_view(void) {
  accelerator_view sa(0);
  return sa;
}
// Concurrency::index
template <int N>
index<N> operator+(const index<N> &lhs, const index<N> &rhs)
  restrict(amp, cpu) {
  index<N> x(lhs);
  x[0] += rhs[0];
  if (N > 1) x[1] += rhs[1];
  if (N > 2) x[2] += rhs[2];
  return  x;
}

/// Concurrency::extent
template<int N>
extent<N> operator-(const extent<N> &lhs, const extent<N> &rhs)
  restrict(amp, cpu) {
    extent<N> i;
    i[0] = lhs[0] - rhs[0];
    if (N>1)
      i[1] = lhs[1] - rhs[1];
    if (N>2)
      i[2] = lhs[2] - rhs[2];
    return i;
  }


inline bool operator==(const extent<1>& lhs, const extent<1>& rhs) restrict(amp,cpu) {
  return (lhs[0] == rhs[0]);
}
inline bool operator==(const extent<2>& lhs, const extent<2>& rhs) restrict(amp,cpu) {
  return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]);
}
inline bool operator==(const extent<3>& lhs, const extent<3>& rhs) restrict(amp,cpu) {
  return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]);
}

inline bool operator!=(const extent<1>& lhs, const extent<1>& rhs) restrict(amp,cpu) {
  return !(lhs == rhs);
}

inline bool operator!=(const extent<2>& lhs, const extent<2>& rhs) restrict(amp,cpu) {
  return !(lhs == rhs);
}

inline bool operator!=(const extent<3>& lhs, const extent<3>& rhs) restrict(amp,cpu) {
  return !(lhs == rhs);
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

template <typename T, int N>
__attribute__((annotate("serialize")))
void array<T, N>::__cxxamp_serialize(Serialize& s) const {
  m_device.__cxxamp_serialize(s);
  extent.__cxxamp_serialize(s);
}

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

template <typename T>
__attribute__((annotate("serialize")))
void array<T, 1>::__cxxamp_serialize(Serialize& s) const {
  m_device.__cxxamp_serialize(s);
  cl_int e0 = extent[0];
  s.Append(sizeof(cl_int), &e0);
}

#else
/// Concurrency::array implementations that are only visible when compiling
/// for the GPU
template <typename T, int N>
__attribute__((annotate("deserialize"))) 
array<T, N>::array(__global T *p, cl_int e0, cl_int e1, cl_int e2)
  restrict(amp): extent(e0, e1, e2), m_device(p) {}

template <typename T>
__attribute__((annotate("deserialize"))) 
array<T, 1>::array(__global T *p, cl_int e) restrict(amp):
  extent(e), m_device(p) {}
#endif
#undef __global

} //namespace Concurrency
#endif //INCLUDE_AMP_IMPL_H
