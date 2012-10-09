#pragma once
#include "declspec_get.h"
#define __global __attribute__((address_space(1)))
template <int> class extent; 

template <typename T>
class array_view<T, 1>
{
public:
#ifdef __GPU__
  typedef _data<T> gmac_buffer_t;
#else
  typedef _data_host<T> gmac_buffer_t;
#endif
  static const int rank = 1;
  typedef T value_type;

  array_view() = delete;

  array_view(array<T,1>& src) restrict(amp,cpu):
    extent(this), p_(NULL), cache_(src.internal()) {}

  template <typename Container>
    array_view(const extent<1>& extent, Container& src):
      array_view(extent, src.data()) {}

  template <typename Container>
    array_view(int e0, Container& src):array_view(Concurrency::extent<1>(e0),
      src) {}

  ~array_view() restrict(amp, cpu) {
#ifndef __GPU__
    if (p_) {
      synchronize();
      cache_.reset();
    }
#endif
  }

  array_view(const extent<1>& extent, value_type* src) restrict(amp,cpu);

  array_view(int e0, value_type* src) restrict(amp,cpu):
    array_view(Concurrency::extent<1>(e0), src) {}

  array_view(const array_view& other) restrict(amp,cpu): extent(this),
    p_(other.p_), size_(other.size_) , cache_(other.cache_) {}

  array_view& operator=(const array_view& other) restrict(amp,cpu);

  void copy_to(array<T,1>& dest) const;

  void copy_to(const array_view& dest) const;

  // __declspec(property(get)) extent<1> extent;
  DeclSpecGetExtent<array_view<T, 1>, Concurrency::extent<1>> extent;
  Concurrency::extent<1> __get_extent(void) restrict(cpu, amp) {
    return Concurrency::extent<1>(size_);
  }

  // These are restrict(amp,cpu)
  __global T& operator[](const index<1>& idx) const restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(cache_.get())[idx[0]];
  }

  __global T& operator[](int i) const restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(cache_.get())[i];
  }

  __global T& operator()(const index<1>& idx) const restrict(amp,cpu) {
    return this->operator[](idx);
  }

  __global T& operator()(int i) const restrict(amp,cpu) {
    return this->operator[](i);
  }

  array_view<T,1> section(const index<1>& idx,
    const Concurrency::extent<1>& ext) restrict(amp,cpu);
  array_view<T,1> section(const index<1>& idx) const restrict(amp,cpu);
  array_view<T,1> section(const Concurrency::extent<1>& ext)
    const restrict(amp,cpu);
  array_view<T,1> section(int i0, int e0) const restrict(amp,cpu);

  void synchronize() const;

  completion_future synchronize_async() const;

  void refresh() const;

  void discard_data() const;

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const;

  __attribute__((annotate("deserialize")))
  array_view(__global T *p, cl_int size) restrict(amp);
  // End CLAMP

 private:
#ifndef __GPU__
  // Stop defining implicit deserialization for this class
  __attribute__((cpu)) int dummy_;
#endif
  // Holding user pointer in CPU mode; null if in device mode
  __global T *p_;
  cl_uint size_;
  // Cached value if initialized with a user ptr;
  // GMAC array pointer if initialized with a Concurrency::array
  gmac_buffer_t cache_;
};

/// 2D array view
template <typename T>
class array_view<T, 2>
{
public:
#ifdef __GPU__
  typedef _data<T> gmac_buffer_t;
#else
  typedef _data_host<T> gmac_buffer_t;
#endif
  static const int rank = 2;

  typedef T value_type;

  array_view() = delete;
#if 0 //disabled for now
  array_view(array<T,1>& src) restrict(amp,cpu): p_(NULL)
#ifndef __GPU__
  , cache_(src.internal())
#endif
  {}
#endif
  template <typename Container>
    array_view(const Concurrency::extent<2>& ext, Container& src):
      array_view(ext, src.data()) {}

  template <typename Container>
    array_view(int e0, int e1, Container& src):
    array_view(Concurrency::extent<2>(e0, e1), src) {}

  ~array_view() restrict(amp, cpu) {
#ifndef __GPU__
    if (p_) {
      synchronize();
      cache_.reset();
    }
#endif
  }

  array_view(const extent<2>& ext, value_type* src) restrict(amp,cpu);

  array_view(int e0, int e1, value_type* src) restrict(amp,cpu):
    array_view(Concurrency::extent<2>(e0, e1), src) {}

#if 0 //disabled for now
  array_view(const array_view& other) restrict(amp,cpu):
    p_(other.p_),
    size_(other.size_)
#ifndef __GPU__
    , cache_(other.cache_)
#endif
    {}

  array_view& operator=(const array_view& other) restrict(amp,cpu);

  void copy_to(array<T,1>& dest) const;
  void copy_to(const array_view& dest) const;
#endif
  // __declspec(property(get)) extent<N> extent;
  DeclSpecGetExtent<array_view<T, 2>, Concurrency::extent<2>> extent;
  Concurrency::extent<2> __get_extent(void) restrict(cpu, amp) {
    return Concurrency::extent<2>(e0_, e1_);
  }

  // These are restrict(amp,cpu)
  __global T& operator[](const index<2>& idx) const restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(cache_.get())[idx[0]*e1_+idx[1]];
  }

  __global T& operator()(const index<2>& idx) const restrict(amp,cpu) {
    return operator[](idx);
  }

  __global T& operator()(int i0, int i1) const restrict(amp,cpu) {
    index<2> idx(i0, i1);
    return (*this)[idx];
  }
#if 0 // disabled for now
  array_view<T,1> section(const index<1>& idx, const extent<1>& ext) restrict(amp,cpu);
  array_view<T,1> section(const index<1>& idx) const restrict(amp,cpu);
  array_view<T,1> section(const extent<1>& ext) const restrict(amp,cpu);
  array_view<T,1> section(int i0, int e0) const restrict(amp,cpu);
#endif
  void synchronize() const {
#ifndef __GPU__
    assert(cache_);
    assert(p_);
    memmove(reinterpret_cast<void*>(p_),
            reinterpret_cast<void*>(cache_.get()), e0_*e1_ * sizeof(T));
#endif
  }
#if 0
  completion_future synchronize_async() const;
#endif //disable

  void refresh() const;

  void discard_data() const;

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const;

  __attribute__((annotate("deserialize")))
  array_view(cl_int e0, cl_int e1, __global T *p) restrict(amp);
  // End CLAMP
 private:
#ifndef __GPU__
  // Stop defining implicit deserialization for this class
  __attribute__((cpu)) int dummy_;
#endif
  cl_int e0_, e1_;
  // Holding user pointer in CPU mode; holding device pointer in GPU mode
  __global T *p_;

  // Cached value if initialized with a user ptr;
  // GMAC array pointer if initialized with a Concurrency::array
  // Note: does not count for deserialization due to the attribute
  gmac_buffer_t cache_;
};

// Out-of-line implementations
// 1D array_view
#ifndef __GPU__

template <typename T>
void array_view<T, 1>::synchronize() const {
  assert(cache_);
  assert(p_);
  memmove(reinterpret_cast<void*>(p_),
      reinterpret_cast<void*>(cache_.get()), size_ * sizeof(T));
}

template <typename T>
array_view<T, 1>::array_view(const Concurrency::extent<1>& extent,
  value_type* src) 
  restrict(amp,cpu): extent(this),
    p_(reinterpret_cast<__global T*>(src)),
    size_(extent[0]), cache_(nullptr) {
    cache_.reset(GMACAllocator<T>().allocate(size_),
      GMACDeleter<T>());
    refresh();
}

template <typename T>
void array_view<T, 1>::refresh() const {
  assert(cache_);
  assert(p_);
  memmove(reinterpret_cast<void*>(cache_.get()),
      reinterpret_cast<void*>(p_), size_ * sizeof(T));
}

template <typename T>
__attribute__((annotate("serialize")))
void array_view<T, 1>::__cxxamp_serialize(Serialize& s) const {
  cache_.__cxxamp_serialize(s);
  s.Append(sizeof(cl_uint), &size_);
}

template <typename T>
array_view<T, 2>::array_view(const Concurrency::extent<2>& ext,
  value_type* src) restrict(amp,cpu): extent(this),
    e0_(ext[0]), e1_(ext[1]),
    p_(reinterpret_cast<__global T*>(src)) {
    cache_.reset(GMACAllocator<T>().allocate(e0_*e1_),
      GMACDeleter<T>());
    refresh();
}

template <typename T>
void array_view<T, 2>::refresh() const {
  assert(cache_);
  assert(p_);
  memmove(reinterpret_cast<void*>(cache_.get()),
      reinterpret_cast<void*>(p_), e0_ * e1_ * sizeof(T));
}

template <typename T>
__attribute__((annotate("serialize")))
void array_view<T, 2>::__cxxamp_serialize(Serialize& s) const {
  s.Append(sizeof(cl_int), &e0_);
  s.Append(sizeof(cl_int), &e1_);
  cache_.__cxxamp_serialize(s);
}

#else // GPU implementations

template <typename T>
array_view<T,1>::array_view(const Concurrency::extent<1>& extent,
  value_type* src) restrict(amp,cpu): extent(this),
    p_(nullptr), size_(extent[0]),
    cache_(reinterpret_cast<__global value_type *>(src)) {}

template <typename T>
__attribute__((annotate("deserialize")))
array_view<T, 1>::array_view(__global T *p, cl_int size) restrict(amp):
  extent(this), p_(nullptr), size_(size), cache_(p) {}


template <typename T>
array_view<T, 2>::array_view(const Concurrency::extent<2>& ext, value_type* src)
  restrict(amp,cpu): extent(this), p_(nullptr), e0_(ext[0]), e1_(ext[1]),
  cache_(reinterpret_cast<__global value_type *>(src)) {}

template <typename T>
__attribute__((annotate("deserialize")))
array_view<T, 2>::array_view(cl_int e0, cl_int e1, __global T *p) restrict(amp):
  extent(this), p_(nullptr), e0_(e0), e1_(e1), cache_(p) {}
#endif
#undef __global  

