#pragma once

#ifndef _PINNED_VECTOR_H
#define _PINNED_VECTOR_H

#include <new>
#include "hc.hpp"
#include "hc_am.hpp"

namespace hc
{

// minimal allocator that uses am_alloc to allocate pinned memory on the host,
// with comparison functions used by the C++ standard library
  
template <class T>
struct am_allocator {
  typedef T value_type;

  am_allocator() = default;

  template <class U> am_allocator(const am_allocator<U>&) {}

  T* allocate(std::size_t n) {
    hc::accelerator acc;
    auto p = static_cast<T*>(hc::am_alloc(n*sizeof(T), acc, amHostPinned));
    if(p == nullptr){ throw std::bad_alloc(); }
    return p;
  }

  void deallocate(T* p, std::size_t) {
    // am_free returns an am_status_t; we can't return that, since
    // allocate is a void function, and we can't throw an exception either,
    // since deallocate is used in destructors. Hmmm.
    hc::am_free(p);
  }
};

template <class T, class U>
bool operator==(const am_allocator<T>&, const am_allocator<U>&) { return true; }

template <class T, class U>
bool operator!=(const am_allocator<T>&, const am_allocator<U>&) { return false; }


// convenience alias 
template<typename T>
using pinned_vector = std::vector<T, am_allocator<T>>;

} // namespace hc

#endif // _PINNED_VECTOR_H
