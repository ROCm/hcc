//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <amp_runtime.h>

// Specialization of AMP classes/templates
namespace Concurrency {
// Accelerators
inline accelerator::accelerator() : 
  accelerator(default_accelerator) {}

inline accelerator::accelerator(const accelerator& other) : 
  device_path(other.device_path), 
  version(other.version), 
  description(other.description),
  is_debug(other.is_debug),
  is_emulated(other.is_emulated),
  has_display(other.has_display),
  supports_double_precision(other.supports_double_precision),
  supports_limited_double_precision(other.supports_limited_double_precision),
  supports_cpu_shared_memory(other.supports_cpu_shared_memory),
  dedicated_memory(other.dedicated_memory),
  default_access_type(other.default_access_type),
  default_view(other.default_view)
  {}

inline accelerator::accelerator(const std::wstring& path) :
  device_path( (path != std::wstring(default_accelerator)) ? 
					path : 
					(_default_accelerator != nullptr) ?
						accelerator::_default_accelerator->get_device_path() :
                        std::wstring(gpu_accelerator) ),
  version(0), 
  description(L""),
  is_debug(false),
  is_emulated(false),
  has_display(false),
  supports_double_precision(true),
  supports_limited_double_precision(false), // constructor will set it
  supports_cpu_shared_memory(false), // constructor will set it
  dedicated_memory(0), // constructor will set it
  default_access_type(access_type_none),
  default_view( (device_path == std::wstring(gpu_accelerator)) ?
					( (_gpu_accelerator != nullptr) ?
						new accelerator_view(_gpu_accelerator.get()) : new accelerator_view(this)) :
					( (_cpu_accelerator != nullptr) ?
						new accelerator_view(_cpu_accelerator.get()) : new accelerator_view(this)) )
{
    if (_gpu_accelerator != nullptr && device_path == std::wstring(gpu_accelerator)) {
      supports_cpu_shared_memory = _gpu_accelerator->supports_cpu_shared_memory;
      dedicated_memory = _gpu_accelerator->dedicated_memory;
      supports_limited_double_precision = _gpu_accelerator->supports_limited_double_precision;
      description = _gpu_accelerator->description;
      default_view->is_auto_selection = !supports_cpu_shared_memory;
      return;
    } else if (_cpu_accelerator != nullptr && device_path == std::wstring(cpu_accelerator)) {
      supports_cpu_shared_memory = _cpu_accelerator->supports_cpu_shared_memory;
      dedicated_memory = _cpu_accelerator->dedicated_memory;
      supports_limited_double_precision = _cpu_accelerator->supports_limited_double_precision;
      description = _cpu_accelerator->description;
      default_view->is_auto_selection = !supports_cpu_shared_memory;
      is_emulated = true;
      return;
    }
    CLAMP::QueryDeviceInfo(device_path, supports_cpu_shared_memory, dedicated_memory, supports_limited_double_precision, description);
    if (device_path == std::wstring(cpu_accelerator))
      is_emulated = true;
}

inline accelerator& accelerator::operator=(const accelerator& other) {
  device_path = other.device_path;
  version = other.version;
  description = other.description;
  is_debug = other.is_debug;
  is_emulated = other.is_emulated;
  has_display = other.has_display;
  supports_double_precision = other.supports_double_precision;
  supports_limited_double_precision = other.supports_limited_double_precision;
  supports_cpu_shared_memory = other.supports_cpu_shared_memory;
  dedicated_memory = other.dedicated_memory;
  default_access_type = other.default_access_type;
  default_view = other.default_view;
  return *this;
}
inline bool accelerator::operator==(const accelerator& other) const {
  return device_path == other.device_path &&
         version == other.version &&
         description == other.description &&
         is_debug == other.is_debug &&
         is_emulated == other.is_emulated &&
         has_display == other.has_display &&
         supports_double_precision == other.supports_double_precision &&
         supports_limited_double_precision == other.supports_limited_double_precision &&
         supports_cpu_shared_memory == other.supports_cpu_shared_memory &&
         dedicated_memory == other.dedicated_memory &&
         default_access_type == other.default_access_type;
         // do not check default_view
}
inline bool accelerator::operator!=(const accelerator& other) const {
  return !(*this == other);
}

inline bool accelerator_view::operator==(const accelerator_view& other) const {
  return is_debug == other.is_debug &&
         is_auto_selection == other.is_auto_selection &&
         version == other.version &&
         queuing_mode == other.queuing_mode &&
         *_accelerator == *other._accelerator;
}

inline accelerator_view& accelerator::get_default_view() const {
  return *default_view;
}

// Accelerator view
inline accelerator_view accelerator::create_view(void) {
  return create_view(queuing_mode_automatic);
}
inline accelerator_view accelerator::create_view(queuing_mode qmode) {
  // make accelarator_view's accelerator pointer always points to static class members
  accelerator_view sa( (device_path == std::wstring(gpu_accelerator)) ?
						_gpu_accelerator.get() : _cpu_accelerator.get() );
  sa.queuing_mode = qmode;
  return sa;
}

inline completion_future accelerator_view::create_marker(){return completion_future();}

// Default access type
inline bool accelerator::set_default_cpu_access_type(access_type type) {
  if(get_supports_cpu_shared_memory() == false &&
        ((type&access_type_read) || (type&access_type_write))) {
    // TODO: Throw runtime exception here 
  }
#ifdef __AMP_CPU__
  if(type == access_type_auto)
      default_access_type = access_type_none;
  else
      default_access_type = type;
#else
  if (&default_view->get_accelerator() == this) {
      if(type == access_type_auto)
          default_access_type = access_type_none;
      else
          default_access_type = type;
  } else
      default_view->get_accelerator().set_default_cpu_access_type(type);
#endif
  return true;
}
inline access_type accelerator::get_default_cpu_access_type() const {return default_access_type;}

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
    index<N> __r(lhs);
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
    index<N> __r(lhs);
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
    index<N> __r(lhs);
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
    extent<N> __r(lhs);
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
    extent<N> __r(lhs);
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
    extent<N> __r(lhs);
    __r %= rhs;
    return __r;
}


template<int N> class extent;
template <int N>
const Concurrency::extent<N>& check(const Concurrency::extent<N>& ext)
{
#ifndef __GPU__
    for (int i = 0; i < N; i++)
    {
        if(ext[i] <=0)
            throw runtime_exception("errorMsg_throw", 0);
    }
#endif
    return ext;
}
    template<typename T, int N> array<T, N>::array(const Concurrency::extent<N>& ext)
: extent(check(ext)), m_device(ext.size()), pav(nullptr), paav(nullptr)
{
    this->cpu_access_type = Concurrency::accelerator(accelerator::default_accelerator).get_default_view().get_accelerator().get_default_cpu_access_type();
#ifndef __GPU__
    initialize();
#endif
}
template<typename T, int N> array<T, N>::array(int e0)
    : array(Concurrency::extent<1>(e0))
    { static_assert(N == 1, "array(int) is only permissible on array<T, 1>"); }
template<typename T, int N> array<T, N>::array(int e0, int e1)
    : array(Concurrency::extent<2>(e0, e1))
    { static_assert(N == 2, "array(int, int) is only permissible on array<T, 2>"); }
template<typename T, int N> array<T, N>::array(int e0, int e1, int e2)
    : array(Concurrency::extent<3>(e0, e1, e2))
    {  static_assert(N == 3, "array(int, int, int) is only permissible on array<T, 3>"); }


template<typename T, int N>
array<T, N>::array(const Concurrency::extent<N>& ext, accelerator_view av, access_type cpu_access_type) : array(ext) {
  if(cpu_access_type == access_type_auto)
    this->cpu_access_type = av.get_accelerator().get_default_cpu_access_type();
  else {
    this->cpu_access_type = cpu_access_type;
    av.get_accelerator().set_default_cpu_access_type(cpu_access_type);
  }
  pav = new accelerator_view(av);
}
template<typename T, int N>
array<T, N>::array(int e0, accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<1>(e0), av, cpu_access_type)
  { static_assert(N == 1, "array(int, accelerator_view) is only permissible on array<T, 1>"); }
template<typename T, int N>
array<T, N>::array(int e0, int e1, accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<2>(e0, e1), av, cpu_access_type)
  { static_assert(N == 2, "array(int, int, accelerator_view) is only permissible on array<T, 2>"); }
template<typename T, int N>
array<T, N>::array(int e0, int e1, int e2, accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<3>(e0, e1, e2), av, cpu_access_type)
{ static_assert(N == 3, "array(int, int, int, accelerator_view) is only permissible on array<T, 3>"); }

template<typename T, int N>
array<T, N>::array(const Concurrency::extent<N>& extent, accelerator_view av, accelerator_view associated_av) : array(extent) {
  pav = new accelerator_view(av);
  if (av.get_accelerator() == accelerator(accelerator::gpu_accelerator) &&
      associated_av.get_accelerator() == accelerator(accelerator::cpu_accelerator)) {
    paav = new accelerator_view(av);
  } else {
    paav = new accelerator_view(associated_av);
  }
}
template<typename T, int N>
array<T, N>::array(int e0, accelerator_view av, accelerator_view associated_av) : array(Concurrency::extent<1>(e0), av, associated_av)
  { static_assert(N == 1, "array(int, accelerator_view, accelerator_view) is only permissible on array<T, 1>"); }
template<typename T, int N>
array<T, N>::array(int e0, int e1, accelerator_view av, accelerator_view associated_av) : array(Concurrency::extent<2>(e0, e1), av, associated_av)
  { static_assert(N == 2, "array(int, int, accelerator_view, accelerator_view) is only permissible on array<T, 2>"); }
template<typename T, int N>
array<T, N>::array(int e0, int e1, int e2, accelerator_view av, accelerator_view associated_av) : array(Concurrency::extent<3>(e0, e1, e2), av, associated_av)
  { static_assert(N == 3, "array(int, int, int, accelerator_view, accelerator_view) is only permissible on array<T, 3>"); }



template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin)
    : extent(ext), m_device(ext.size()), pav(nullptr), paav(nullptr) {
  this->cpu_access_type = Concurrency::accelerator(accelerator::default_accelerator).get_default_view().get_accelerator().get_default_cpu_access_type();
#ifndef __GPU__
        InputIterator srcEnd = srcBegin;
        std::advance(srcEnd, extent.size());
        initialize(srcBegin, srcEnd);
#endif
    }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, InputIterator srcEnd)
    : extent(ext), m_device(ext.size()), pav(nullptr), paav(nullptr) {
#ifndef __GPU__
    if(ext.size() < std::distance(srcBegin,srcEnd) )
      throw runtime_exception("errorMsg_throw", 0);
#endif
  this->cpu_access_type = Concurrency::accelerator(accelerator::default_accelerator).get_default_view().get_accelerator().get_default_cpu_access_type();
#ifndef __GPU__
        initialize(srcBegin, srcEnd);
#endif
 }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin)
    : array(Concurrency::extent<1>(e0), srcBegin)
      { static_assert(N == 1, "array(int, iterator) is only permissible on array<T, 1>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, InputIterator srcEnd)
    : array(Concurrency::extent<1>(e0), srcBegin, srcEnd)
    { static_assert(N == 1, "array(int, iterator, iterator) is only permissible on array<T, 1>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin)
    : array(Concurrency::extent<2>(e0, e1), srcBegin)
    { static_assert(N == 2, "array(int, int, iterator) is only permissible on array<T, 2>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd)
    : array(Concurrency::extent<2>(e0, e1), srcBegin, srcEnd)
    { static_assert(N == 2, "array(int, int, iterator, iterator) is only permissible on array<T, 2>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin)
    { static_assert(N == 3, "array(int, int, int, iterator) is only permissible on array<T, 3>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, srcEnd)
    { static_assert(N == 3, "array(int, int, int, iterator, iterator) is only permissible on array<T, 3>"); }



template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, accelerator_view av,
                   access_type cpu_access_type) : array(ext, av, cpu_access_type) {
#ifndef __GPU__
  InputIterator srcEnd = srcBegin;
  std::advance(srcEnd, extent.size());
  initialize(srcBegin, srcEnd);
#endif
}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, access_type cpu_access_type) : array(ext, srcBegin, av, cpu_access_type) {
#ifndef __GPU__
  initialize(srcBegin, srcEnd);
#endif
}

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, accelerator_view av, access_type cpu_access_type)
    : array(Concurrency::extent<1>(e0), srcBegin, av, cpu_access_type)
    { static_assert(N == 1, "array(int, iterator, accelerator_view) is only permissible on array<T, 1>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, InputIterator srcEnd, accelerator_view av, access_type cpu_access_type)
    : array(Concurrency::extent<1>(e0), srcBegin, srcEnd, av, cpu_access_type)
    { static_assert(N == 1, "array(int, iterator, iterator, accelerator_view) is only permissible on array<T, 1>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, accelerator_view av, access_type cpu_access_type)
    : array(Concurrency::extent<2>(e0, e1), srcBegin, av, cpu_access_type)
    { static_assert(N == 2, "array(int, int, iterator, accelerator_view) is only permissible on array<T, 2>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<2>(e0, e1), srcBegin, srcEnd, av, cpu_access_type)
                   { static_assert(N == 2, "array(int, int, iterator, iterator, accelerator_view) is only permissible on array<T, 2>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, accelerator_view av, access_type cpu_access_type)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, av, cpu_access_type)
    { static_assert(N == 3, "array(int, int, int, iterator, accelerator_view) is only permissible on array<T, 3>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, access_type cpu_access_type) : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, srcEnd, av, cpu_access_type)
    { static_assert(N == 3, "array(int, int, int, iterator, iterator, accelerator_view) is only permissible on array<T, 3>"); }





template<typename T, int N> template <typename InputIterator>
array<T, N>::array(const Concurrency::extent<N>& ext, InputIterator srcBegin, accelerator_view av,
                   accelerator_view associated_av) : array(ext, av, associated_av) {
#ifndef __GPU__
  InputIterator srcEnd = srcBegin;
  std::advance(srcEnd, extent.size());
  initialize(srcBegin, srcEnd);
#endif
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
    : array(Concurrency::extent<1>(e0), srcBegin, av, associated_av)
    { static_assert(N == 1, "array(int, iterator, iterator, accelerator_view, acclerator_view) is only permissible on array<T, 1>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, accelerator_view associated_av)
    : array(Concurrency::extent<1>(e0), srcBegin, srcEnd, av, associated_av)
    { static_assert(N == 1, "array(int, iterator, iterator, accelerator_view, accelerator_view) is only permissible on array<T, 1>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, accelerator_view av,
                   accelerator_view associated_av)
    : array(Concurrency::extent<2>(e0, e1), srcBegin, av, associated_av)
    { static_assert(N == 2, "array(int, int, iterator, accelerator_view, acclerator_view) is only permissible on array<T, 2>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, accelerator_view associated_av)
    : array(Concurrency::extent<2>(e0, e1), srcBegin, srcEnd, av, associated_av)
    { static_assert(N == 2, "array(int, int, iterator, iterator, accelerator_view, acclerator_view) is only permissible on array<T, 2>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, accelerator_view av,
                   accelerator_view associated_av)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, av, associated_av)
    { static_assert(N == 3, "array(int, int, int, iterator, accelerator_view, acclerator_view) is only permissible on array<T, 3>"); }

template<typename T, int N> template <typename InputIterator>
array<T, N>::array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd,
                   accelerator_view av, accelerator_view associated_av)
    : array(Concurrency::extent<3>(e0, e1, e2), srcBegin, srcEnd, av, associated_av)
    { static_assert(N == 3, "array(int, int, int, iterator, iterator, accelerator_view, acclerator_view) is only permissible on array<T, 3>"); }


template<typename T, int N> array<T, N>::array(const array& other)
    : extent(other.extent), m_device(other.m_device), pav(other.pav), paav(other.paav) {
  if(pav) pav = new accelerator_view(*(other.pav));
  if(paav) paav = new accelerator_view(*(other.paav));
}
template<typename T, int N> array<T, N>::array(array&& other)
    : extent(other.extent), m_device(other.m_device),pav(other.pav), paav(other.paav) {
  if(pav) pav = new accelerator_view(*(other.pav));
  if(paav) paav = new accelerator_view(*(other.paav));
  this->cpu_access_type = other.cpu_access_type;
}
template<typename T, int N>
array<T, N>::array(const array_view<const T, N>& src, accelerator_view av,
                   access_type cpu_access_type)
    : array(src) {
  if(cpu_access_type == access_type_auto)
    this->cpu_access_type = av.get_accelerator().get_default_cpu_access_type();
  else
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
    if(cache.get())
        cache.synchronize();
}

template <typename T, int N>
completion_future array_view<T, N>::synchronize_async() const {
    assert(cache.get());
    std::future<void> fut = std::async([&]() mutable { synchronize(); });
    return completion_future(fut.share());
}

template <typename T, int N>
array_view<T, N>::array_view(const Concurrency::extent<N>& ext,
                             value_type* src) restrict(amp,cpu)
    : extent(ext), extent_base(ext), cache(ext.size(), src), offset(0) {}

template <typename T, int N>
array_view<T, N>::array_view(const Concurrency::extent<N>& ext) restrict(amp,cpu)
    : extent(ext), extent_base(ext), cache(ext.size()), offset(0) {}

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
    : extent(ext), extent_base(ext), cache((__global T *)(src)), offset(0) {}

#endif

//array_view<const T, N>
#ifndef __GPU__

template <typename T, int N>
void array_view<const T, N>::synchronize() const {
  if(cache.get())
      cache.synchronize();
}

template <typename T, int N>
completion_future array_view<const T, N>::synchronize_async() const {
    assert(cache.get());
    std::future<void> fut = std::async([&]() mutable { synchronize(); });
    return completion_future(fut.share());
}

template <typename T, int N>
array_view<const T, N>::array_view(const Concurrency::extent<N>& ext,
                             value_type* src) restrict(amp,cpu)
    : extent(ext), extent_base(ext), cache(ext.size(), const_cast<nc_T*>(src)),
    offset(0) {}

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
    : extent(ext), extent_base(ext), cache((__global nc_T *)(src)), offset(0) {}

#endif
#undef __global

} //namespace Concurrency
