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

inline accelerator accelerator_view::get_accelerator() const { return pMan; }
inline completion_future accelerator_view::create_marker(){ return completion_future(); }

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
    : cache(accelerator::get_auto_selection_view().pAloc, ext.size(), src),
    extent(ext), extent_base(ext), offset(0) {}

template <typename T, int N>
array_view<T, N>::array_view(const Concurrency::extent<N>& ext) restrict(amp,cpu)
    : cache(accelerator::get_auto_selection_view().pAloc, ext.size()),
    extent(ext), extent_base(ext), offset(0) {}

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
    : cache(accelerator::get_auto_selection_view().pAloc, ext.size(), const_cast<nc_T*>(src)),
    extent(ext), extent_base(ext), offset(0) {}

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
