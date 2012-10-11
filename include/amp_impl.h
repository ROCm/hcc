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

// Specialization of AMP classes/templates

namespace Concurrency {
//Accelerators
accelerator::accelerator(void) {
  cl_int error_code;
  error_code = clGetPlatformIDs(1, &platform_, NULL);
  CHECK_ERROR(error_code, "clGetPlatformIDs");
  error_code = clGetDeviceIDs(platform_,
    CL_DEVICE_TYPE_GPU, 1, &device_, NULL);
  CHECK_ERROR(error_code, "clGetDeviceIDs");
  if (!default_view_)
    default_view_ = new accelerator_view(device_);
}
accelerator_view accelerator::create_view(void) {
  accelerator_view sa(device_);
  return sa;
}

accelerator_view::accelerator_view(const accelerator_view &v):
  device_(v.device_), context_(v.context_),
  command_queue_(v.command_queue_) {
  //std::cerr << "SAV: copy constructor calld. This = " << this;
  //std::cerr << " v = " << &v << std::endl;
  cl_int err = clRetainCommandQueue(command_queue_);
  CHECK_ERROR(err, "clRetainCommandQueue");
  err = clRetainContext(context_);
  CHECK_ERROR(err, "clRetainContext");
}

accelerator_view::~accelerator_view() {
  clReleaseCommandQueue(command_queue_);
  clReleaseContext(context_);
}

accelerator_view::accelerator_view(cl_device_id d):
    device_(d) {
  cl_int error_code;
  context_ = clCreateContext(0, 1, &d, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, "clCreateContext");
  command_queue_ = clCreateCommandQueue(context_, d, 0, &error_code);
  CHECK_ERROR(error_code, "clCreateCommandQueue");
}

accelerator_view& accelerator::get_default_view() const { return *default_view_; }
accelerator_view *accelerator::default_view_ = NULL;

/// Concurrency::extent

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
array<T, N>::array(int e0, accelerator_view av):
  extent(e0), accelerator_view_(av) {
  m_device.reset(GMACAllocator<T>().allocate(e0), GMACDeleter<T>());
}

template <typename T, int N>
array<T, N>::array(const Concurrency::extent<N>& ext): extent(ext),
  m_device(nullptr), accelerator_view_(accelerator().get_default_view()) {
  size_t sz = ext[0];
  if (rank == 2) {
    e1_ = ext[1];
    sz *= e1_;
  }
  if (sz) {
    m_device.reset(GMACAllocator<T>().allocate(sz),
	GMACDeleter<T>());
  }
}

template <typename T, int N>
array<T,N>:: array(const array& other): extent(other.m_extent),
    accelerator_view_(other.accelerator_view_), m_device(other.m_device) {}

template <typename T, int N>
__attribute__((annotate("serialize")))
void array<T, N>::__cxxamp_serialize(Serialize& s) const {
  m_device.__cxxamp_serialize(s);
  s.Append(sizeof(cl_int), &e1_);
}

// 1D array constructors

template <typename T>
array<T, 1>::array(int e0, accelerator_view av):
  e0_(e0), accelerator_view_(av) {
  m_device.reset(GMACAllocator<T>().allocate(e0), GMACDeleter<T>());
}

template <typename T>
array<T, 1>::array(const extent<1>& ext): e0_(ext[0]),
  m_device(nullptr), accelerator_view_(accelerator().get_default_view()) {
  if (ext[0])
    m_device.reset(GMACAllocator<T>().allocate(e0_), GMACDeleter<T>());
}

template <typename T>
__attribute__((annotate("serialize")))
void array<T, 1>::__cxxamp_serialize(Serialize& s) const {
  m_device.__cxxamp_serialize(s);
  s.Append(sizeof(cl_int), &e0_);
}

#else
/// Concurrency::array implementations that are only visible when compiling
/// for the GPU
template <typename T, int N>
__attribute__((annotate("deserialize"))) 
array<T, N>::array(__global T *p, cl_int e) restrict(amp): m_device(p), e1_(e) {}

template <typename T>
__attribute__((annotate("deserialize"))) 
array<T, 1>::array(__global T *p, cl_int e) restrict(amp): m_device(p),
  e0_(e) {}
#endif
#undef __global

} //namespace Concurrency
#endif //INCLUDE_AMP_IMPL_H
