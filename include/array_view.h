#pragma once
template <typename T>
class array_view<T, 1>
{
#define __global __attribute__((address_space(1)))
public:
  static const int rank = 1;
  typedef T value_type;

  array_view() = delete;
  array_view(array<T,1>& src) restrict(amp,cpu): p_(NULL)
#ifndef __GPU__
  , cache_(src.internal())
#endif
  {}
  template <typename Container>
    array_view(const extent<1>& extent, Container& src): array_view(extent, src.data()) {}
  template <typename Container>
    array_view(int e0, Container& src):array_view(extent<1>(e0), src) {}
  ~array_view() restrict(amp, cpu) {
#ifndef __GPU__
    if (p_) {
      synchronize();
      cache_.reset();
    }
#endif
  }
  array_view(const extent<1>& extent, value_type* src) restrict(amp,cpu):
    p_(reinterpret_cast<__global T*>(src)),
    size_(extent[0]) {
#ifndef __GPU__
    cache_.reset(GMACAllocator<T>().allocate(size_),
      GMACDeleter<T>());
    refresh();
#endif
  }
  array_view(int e0, value_type* src) restrict(amp,cpu):array_view(extent<1>(e0), src) {}

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

  // __declspec(property(get)) extent<N> extent;
  extent<1> get_extent() const;

  // These are restrict(amp,cpu)
  __global T& operator[](const index<1>& idx) const restrict(amp,cpu) {
#ifdef __GPU__
    return p_[idx[0]];
#else
    return reinterpret_cast<__global T*>(cache_.get())[idx[0]];
#endif
  }
  __global T& operator[](int i) const restrict(amp,cpu) {
#ifdef __GPU__
    return p_[i];
#else
    return reinterpret_cast<__global T*>(cache_.get())[i];
#endif
  }

  __global T& operator()(const index<1>& idx) const restrict(amp,cpu);
  __global T& operator()(int i) const restrict(amp,cpu);

  array_view<T,1> section(const index<1>& idx, const extent<1>& ext) restrict(amp,cpu);
  array_view<T,1> section(const index<1>& idx) const restrict(amp,cpu);
  array_view<T,1> section(const extent<1>& ext) const restrict(amp,cpu);
  array_view<T,1> section(int i0, int e0) const restrict(amp,cpu);

  void synchronize() const {
#ifndef __GPU__
    assert(cache_);
    assert(p_);
    memmove(reinterpret_cast<void*>(p_),
            reinterpret_cast<void*>(cache_.get()), size_ * sizeof(T));
#endif
  }
  completion_future synchronize_async() const;

  void refresh() const {
#ifndef __GPU__
    assert(cache_);
    assert(p_);
    memmove(reinterpret_cast<void*>(cache_.get()),
            reinterpret_cast<void*>(p_), size_ * sizeof(T));
#endif
  }
  void discard_data() const;

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const {
#ifndef __GPU__
    cl_int err;
    cl_context context = s.getContext();
    cl_mem t = clGetBuffer(context, (const void *)cache_.get());
    s.Append(sizeof(cl_mem), &t);
    s.Append(sizeof(cl_uint), &size_);
#endif
  }
  // End CLAMP

 private:
  // Holding user pointer in CPU mode; holding device pointer in GPU mode
  __global T *p_;
  cl_uint size_;
#ifndef __GPU__
  // Cached value if initialized with a user ptr;
  // GMAC array pointer if initialized with a Concurrency::array
  // Note: does not count for deserialization due to the attribute
  __attribute__((cpu)) std::shared_ptr<T> cache_;
#endif
#undef __global  
};

