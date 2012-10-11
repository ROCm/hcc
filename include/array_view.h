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
    extent(this), p_(NULL), size_(src.get_extent()[0]),
    cache_(src.internal()), offset_(0) {}

  template <typename Container>
    array_view(const extent<1>& extent, Container& src):
      array_view(extent, src.data()) {}

  template <typename Container>
    array_view(int e0, Container& src):
      array_view(Concurrency::extent<1>(e0), src) {}

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
    p_(other.p_), size_(other.size_) , cache_(other.cache_),
    offset_(other.offset_) {}

  array_view& operator=(const array_view& other) restrict(amp,cpu) {
    p_ = other.p_; size_ = other.size_; cache_ = other.cache_;
    offset_ = other.offset_;
    return *this;
  }

  void copy_to(array<T,1>& dest) const { assert(0 && "Unimplemented"); }

  void copy_to(const array_view& dest) const {
    assert(0 && "Unimplemented");
  }

  // __declspec(property(get)) extent<1> extent;
  DeclSpecGetExtent<array_view<T, 1>, Concurrency::extent<1>> extent;
  Concurrency::extent<1> get_extent(void) const restrict(cpu, amp) {
    return Concurrency::extent<1>(size_);
  }

  // These are restrict(amp,cpu)
  __global T& operator[](const index<1>& idx) const restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(cache_.get()+offset_)[idx[0]];
  }

  __global T& operator[](int i) const restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(cache_.get()+offset_)[i];
  }

  __global T& operator()(const index<1>& idx) const restrict(amp,cpu) {
    return this->operator[](idx);
  }

  __global T& operator()(int i) const restrict(amp,cpu) {
    return this->operator[](i);
  }

  array_view<T,1> section(const index<1>& idx,
    const Concurrency::extent<1>& ext) restrict(amp,cpu) {
    return section(idx[0], ext[0]);
  }
  array_view<T,1> section(const index<1>& idx) const restrict(amp,cpu) {
    return section(idx[0], size_-idx[0]);
  }
  array_view<T,1> section(const Concurrency::extent<1>& ext)
    const restrict(amp,cpu) {
    return section(0, ext[0]);
  }

  array_view<T,1> section(int i0, int e0) const restrict(amp,cpu) {
    array_view<T, 1> av(p_, Concurrency::extent<1>(e0), cache_, i0);
    return av;
  }

  template <int K>
    array_view<T, K> view_as(Concurrency::extent<K> viewExtent) const
    restrict(amp, cpu) {
    array_view<T, K> av(p_, viewExtent, cache_, offset_);
    return av;
  }

  T *data() const restrict(amp, cpu) {
    return cache_.get()+offset_;
  }

  void synchronize() const;

  completion_future synchronize_async() const {
    assert(0 && "Not implemented yet");
  }

  void refresh() const;

  void discard_data() const {/*No operation */}

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const;

  __attribute__((annotate("deserialize")))
  array_view(__global T *p, cl_int size, cl_int offset) restrict(amp);
  // End CLAMP
  
  //TODO: move to private; used only by projection
  array_view(__global T *p, Concurrency::extent<1> ext, 
    const gmac_buffer_t &cache, cl_uint offset) restrict(amp,cpu):
    extent(this), p_(p), size_(ext[0]), cache_(cache), offset_(offset) {}

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
  // Offset of this view regarding to the base of cache.
  cl_uint offset_;
  
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

  array_view(array<T,2>& src) restrict(amp,cpu): 
    extent(this),
    e0_(src.get_extent()[0]), e1_(src.get_extent()[1]),
    s1_(src.get_extent()[1]), offset_(0), p_(NULL), //FIXME: copy these from array
    cache_(src.internal()) {}

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

  array_view(const array_view& other) restrict(amp,cpu):
    extent(this), p_(other.p_), e0_(other.e0_), e1_(other.e1_), s1_(other.s1_),
    offset_(other.offset_), cache_(other.cache_) {}

  array_view& operator=(const array_view& other) restrict(amp,cpu) {
    e0_=other.e0_; e1_=other.e1_; s1_=other.s1_; offset_=other.offset_;
    cache_ = other.cache_;
    return *this;
  }

  void copy_to(array<T,1>& dest) const { assert(0&&"Unimplemented"); }
  void copy_to(const array_view& dest) const { assert(0&&"Unimplemented"); }

  
  // __declspec(property(get)) extent<N> extent;
  DeclSpecGetExtent<array_view<T, 2>, Concurrency::extent<2> > extent;
  Concurrency::extent<2> get_extent(void) const restrict(cpu, amp) {
    return Concurrency::extent<2>(e0_, e1_);
  }

  // These are restrict(amp,cpu)
  __global T& operator[](const index<2>& idx) const restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(cache_.get())[idx[0]*s1_+idx[1]+offset_];
  }

  // Projection to 1D
  array_view<T, 1> operator[](int i) const restrict(amp, cpu) {
    array_view<T, 1> av(p_, Concurrency::extent<1>(e1_), cache_, e1_*i);
    return av;
  }

  __global T& operator()(const index<2>& idx) const restrict(amp,cpu) {
    return operator[](idx);
  }

  __global T& operator()(int i0, int i1) const restrict(amp,cpu) {
    index<2> idx(i0, i1);
    return (*this)[idx];
  }

  array_view<T, 2> section(const index<2>& idx, const Concurrency::extent<2>& ext)
    restrict(amp,cpu) {
    return section(idx[0], idx[1], ext[0], ext[1]);
  }

  array_view<T, 2> section(const index<2>& idx) const restrict(amp,cpu) {
    return section(idx[0], idx[1], e0_-idx[0], e1_-idx[1]);
  }

  array_view<T, 2> section(const Concurrency::extent<2>& ext) const restrict(amp,cpu) {
    return section(0, 0, ext[0], ext[1]);
  }

  array_view<T, 2> section(int i0, int i1, int e0, int e1) const restrict(amp,cpu) {
    array_view<T, 2> av(e0, e1, /*stride*/ e1_,
      /*offset*/i0*e1_+i1, p_, cache_);
    return av;
  }
  void synchronize() const {
#ifndef __GPU__
    assert(cache_);
    assert(p_);
    assert(e1_ == s1_ && "Only support non-sectioned view");
    assert(offset_ == 0 && "Only support non-sectioned view");
    memmove(reinterpret_cast<void*>(p_),
            reinterpret_cast<void*>(cache_.get()), e0_*e1_ * sizeof(T));
#endif
  }

  completion_future synchronize_async() const {
    assert(0 && "Not implemented yet");
  }


  void refresh() const;

  void discard_data() const;

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const;

  __attribute__((annotate("deserialize")))
  array_view(cl_int e0, cl_int e1, cl_int s1_, cl_uint offset, __global T *p) restrict(amp);
  // End CLAMP
  //Used only by view_as
  array_view(__global T *p, Concurrency::extent<2> ext, 
    const gmac_buffer_t &cache, cl_uint offset) restrict(amp,cpu):
  array_view(ext[0], ext[1], ext[1], offset, p, cache) {}
 private:
  //Used only by projection and section
  array_view(cl_int e0, cl_int e1, cl_int s1, cl_uint offset, 
    __global T*p, const gmac_buffer_t &cache) restrict(amp,cpu):
    extent(this), e0_(e0), e1_(e1), s1_(s1), offset_(offset),
    p_(p), cache_(cache) {}
#ifndef __GPU__
  // Stop defining implicit deserialization for this class
  __attribute__((cpu)) int dummy_;
#endif
  // Extent (e0_, e1_)
  cl_int e0_, e1_;
  // Strides for flattening. == e1_ if not constructed by sectioning another view
  cl_int s1_;
  // Offset of this view regarding to the base of cache.
  cl_uint offset_;
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
  assert(offset_ == 0 && "Does not support views created with offsets");
  memmove(reinterpret_cast<void*>(p_),
      reinterpret_cast<void*>(cache_.get()), size_ * sizeof(T));
}

template <typename T>
array_view<T, 1>::array_view(const Concurrency::extent<1>& extent,
  value_type* src) 
  restrict(amp,cpu): extent(this),
    p_(reinterpret_cast<__global T*>(src)),
    size_(extent[0]), cache_(nullptr), offset_(0) {
    cache_.reset(GMACAllocator<T>().allocate(size_),
      GMACDeleter<T>());
    refresh();
}

template <typename T>
void array_view<T, 1>::refresh() const {
  assert(cache_);
  assert(p_);
  assert(offset_ == 0 && "Does not support views created with offsets");
  memmove(reinterpret_cast<void*>(cache_.get()),
      reinterpret_cast<void*>(p_), size_ * sizeof(T));
}

template <typename T>
__attribute__((annotate("serialize")))
void array_view<T, 1>::__cxxamp_serialize(Serialize& s) const {
  cache_.__cxxamp_serialize(s);
  s.Append(sizeof(cl_uint), &size_);
  s.Append(sizeof(cl_uint), &offset_);
}

template <typename T>
array_view<T, 2>::array_view(const Concurrency::extent<2>& ext,
  value_type* src) restrict(amp,cpu): extent(this),
    e0_(ext[0]), e1_(ext[1]), s1_(ext[1]), offset_(0),
    p_(reinterpret_cast<__global T*>(src)) {
    cache_.reset(GMACAllocator<T>().allocate(e0_*e1_),
      GMACDeleter<T>());
    refresh();
}

template <typename T>
void array_view<T, 2>::refresh() const {
  assert(cache_);
  assert(p_);
  assert(e1_ == s1_ && "Only support non-sectioned view");
  assert(offset_ == 0 && "Only support non-sectioned view");
  memmove(reinterpret_cast<void*>(cache_.get()),
      reinterpret_cast<void*>(p_), e0_ * e1_ * sizeof(T));
}

template <typename T>
__attribute__((annotate("serialize")))
void array_view<T, 2>::__cxxamp_serialize(Serialize& s) const {
  s.Append(sizeof(cl_int), &e0_);
  s.Append(sizeof(cl_int), &e1_);
  s.Append(sizeof(cl_int), &s1_);
  s.Append(sizeof(cl_int), &offset_);
  cache_.__cxxamp_serialize(s);
}

#else // GPU implementations

template <typename T>
array_view<T,1>::array_view(const Concurrency::extent<1>& extent,
  value_type* src) restrict(amp,cpu): extent(this),
    p_(nullptr), size_(extent[0]),
    cache_(reinterpret_cast<__global value_type *>(src)), offset_(0) {}

template <typename T>
__attribute__((annotate("deserialize")))
array_view<T, 1>::array_view(__global T *p, cl_int size, cl_int off) restrict(amp):
  extent(this), p_(nullptr), size_(size), cache_(p), offset_(off) {}


template <typename T>
array_view<T, 2>::array_view(const Concurrency::extent<2>& ext, value_type* src)
  restrict(amp,cpu): extent(this), p_(nullptr), e0_(ext[0]), e1_(ext[1]), s1_(ext[1]),
  offset_(0), cache_(reinterpret_cast<__global value_type *>(src)) {}

template <typename T>
__attribute__((annotate("deserialize")))
array_view<T, 2>::array_view(cl_int e0, cl_int e1, cl_int s1, cl_uint offset,
  __global T *p) restrict(amp):
  extent(this), p_(nullptr), e0_(e0), e1_(e1), s1_(s1), offset_(offset), cache_(p) {}
#endif
#undef __global  

