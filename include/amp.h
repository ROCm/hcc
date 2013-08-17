// CAVEATS: There could be walkarounds for quick evaluation purposes. Here we
// list such features that are used in the code with description.
//
// ACCELERATOR
//  According to specification, each array should have its binding accelerator
//  instance. For now, we haven't implemented such binding nor actual
//  implementation of accelerator. For a quick and dirty walkaround for
//  OpenCL based prototype, we allow key OpenCL objects visible globally so
//  that we don't have to be bothered with such implementaion effort.

#pragma once

#include <cassert>
#include <string>
#include <vector>
#include <chrono>
#include <future>
#include <string.h> //memcpy
#include <gmac/opencl.h>
#include <memory>
#include <algorithm>
#include <set>
// CLAMP
#include <serialize.h>
// End CLAMP

/* COMPATIBILITY LAYER */
#define STD__FUTURE_STATUS__FUTURE_STATUS std::future_status

#ifndef WIN32
#define __declspec(ignored) /* */
#endif

namespace Concurrency {
/*
  This is not part of C++AMP standard, but borrowed from Parallel Patterns
  Library. 
*/
  template <typename _Type> class task;
  template <> class task<void>;

enum queuing_mode {
  queuing_mode_immediate,
  queuing_mode_automatic
};

class accelerator_view;
class accelerator;

class accelerator {
public:
  static const wchar_t default_accelerator[];   // = L"default"
  static const wchar_t direct3d_warp[];         // = L"direct3d\\warp"
  static const wchar_t direct3d_ref[];          // = L"direct3d\\ref"
  static const wchar_t cpu_accelerator[];       // = L"cpu"
  
  accelerator();
  explicit accelerator(const std::wstring& path);
  accelerator(const accelerator& other);
  static std::vector<accelerator> get_all();
  static bool set_default(const std::wstring& path);
  accelerator& operator=(const accelerator& other);
  //__declspec(property(get)) std::wstring device_path;
  const std::wstring device_path;
  const std::wstring &get_device_path() { return device_path; }
  __declspec(property(get)) unsigned int version; // hiword=major, loword=minor
  //__declspec(property(get)) std::wstring description;
  std::wstring description;
  const std::wstring &get_description() { return description; }
  __declspec(property(get)) bool is_debug;
  __declspec(property(get)) bool is_emulated;
  __declspec(property(get)) bool has_display;
  __declspec(property(get)) bool supports_double_precision;
  __declspec(property(get)) bool supports_limited_double_precision;
  __declspec(property(get)) size_t dedicated_memory;
  accelerator_view& get_default_view() const;
  // __declspec(property(get=get_default_view)) accelerator_view default_view;

  accelerator_view create_view();
  accelerator_view create_view(queuing_mode qmode);
  
  bool operator==(const accelerator& other) const;
  bool operator!=(const accelerator& other) const;
 private:
  static accelerator_view *default_view_;
};

class completion_future;
class accelerator_view {
public:
  accelerator_view() = delete;
  accelerator_view(const accelerator_view& other) {}
  accelerator_view& operator=(const accelerator_view& other) {
    return *this;
  }

  accelerator get_accelerator() const;
  // __declspec(property(get=get_accelerator)) Concurrency::accelerator accelerator;

  __declspec(property(get)) bool is_debug;
  __declspec(property(get)) unsigned int version;
  __declspec(property(get)) queuing_mode queuing_mode;
  void flush();
  void wait();
  completion_future create_marker();
  bool operator==(const accelerator_view& other) const;
  bool operator!=(const accelerator_view& other) const;
  ~accelerator_view() {}
 private:
  //CLAMP-specific
  friend class accelerator;
  explicit accelerator_view(int) {}
  //End CLAMP-specific
};
//CLAMP
extern "C" __attribute__((pure)) int get_global_id(int n) restrict(amp);
extern "C" __attribute__((pure)) int get_local_id(int n) restrict(amp);
extern "C" __attribute__((pure)) int get_group_id(int n) restrict(amp);
#define tile_static static __attribute__((address_space(3)))
extern "C" void barrier(int n) restrict(amp);
//End CLAMP
class completion_future {
public:
  completion_future();
  completion_future(const completion_future& _Other);
  completion_future(completion_future&& _Other);
  ~completion_future();
  completion_future& operator=(const completion_future& _Other);
  completion_future& operator=(completion_future&& _Other);

  void get() const;
  bool valid() const;
  void wait() const;

  template <class _Rep, class _Period>
  STD__FUTURE_STATUS__FUTURE_STATUS wait_for(
    const std::chrono::duration<_Rep, _Period>& _Rel_time) const;

  template <class _Clock, class _Duration>
  STD__FUTURE_STATUS__FUTURE_STATUS wait_until(
    const std::chrono::time_point<_Clock, _Duration>& _Abs_time) const;

  operator std::shared_future<void>() const;

  template <typename _Functor>
  void then(const _Functor &_Func) const;

  Concurrency::task<void> to_task() const;
};

template <int N> class extent;
template <int N>
class index {
public:
  static const int rank = N;

  typedef int value_type;

  index() restrict(amp,cpu):index(0, 0, 0) {}

  index(const index& other) restrict(amp,cpu):
    i0_(other.i0_), i1_(other.i1_), i2_(other.i2_) {}

  explicit index(int i0) restrict(amp,cpu):index(i0, 0, 0) {} // N==1

  index(int i0, int i1) restrict(amp,cpu):index(i0, i1, 0) {} // N==2

  __attribute__((annotate("deserialize")))
  index(int i0, int i1, int i2) restrict(amp,cpu):
    i0_(i0), i1_(i1), i2_(i2) {}

  explicit index(const int components[]) restrict(amp,cpu);

  index& operator=(const index& other) restrict(amp,cpu) {
    if (rank > 0)
      i0_ = other.i0_;
    if (rank > 1)
      i1_ = other.i1_;
    if (rank > 2)
      i2_ = other.i2_;
    return *this;
  }

  int operator[](unsigned int c) const restrict(amp,cpu) {
    if (c==0)
      return i0_;
    else if (c==1)
      return i1_;
    else
      return i2_;
  }

  int& operator[](unsigned int c) restrict(amp,cpu) {
    if (c==0)
      return i0_;
    else if (c==1)
      return i1_;
    else
      return i2_;
  }
  
  template <int M>
    friend index<M> operator+(const index<N> &lhs,
      const index<N> &rhs) restrict(amp, cpu);
  index& operator+=(const index& rhs) restrict(amp,cpu) {
    i0_ += rhs[0];
    if (N > 1) i1_ += rhs[1];
    if (N > 2) i2_ += rhs[2];
    return *this;
  }
  index& operator-=(const index& rhs) restrict(amp,cpu) {
    i0_ -= rhs[0];
    if (N > 1) i1_ -= rhs[1];
    if (N > 2) i2_ -= rhs[2];
    return *this;
  }

  index& operator+=(int rhs) restrict(amp,cpu);
  index& operator-=(int rhs) restrict(amp,cpu);
  index& operator*=(int rhs) restrict(amp,cpu);
  index& operator/=(int rhs) restrict(amp,cpu);
  index& operator%=(int rhs) restrict(amp,cpu);
  
  index& operator++() restrict(amp,cpu) {
    //FIXME extent?
    i0_++;
    return *this;
  }
  index operator++(int) restrict(amp,cpu) {
    //FIXME extent?
    index ret = *this;
    i0_++;
    return ret;
  }
  index& operator--() restrict(amp,cpu);
  index operator--(int) restrict(amp,cpu);
 private:
  //CLAMP
  template<class Y>
  friend void parallel_for_each(extent<1>, const Y&);
  __attribute__((annotate("__cxxamp_opencl_index")))
  void __cxxamp_opencl_index() restrict(amp,cpu)
#ifdef __GPU__
  {
    if (rank == 1) {
      i0_ = get_global_id(0);
    } else if (rank == 2) {
      i0_ = get_global_id(1);
      i1_ = get_global_id(0);
    } else {
      i0_ = get_global_id(2);
      i1_ = get_global_id(1);
      i2_ = get_global_id(0);
    }
  }
#else
  ;
#endif // __GPU__
  //End CLAMP
  int i0_, i1_, i2_;
};

// C++AMP LPM 4.5
class tile_barrier {
 public:
  void wait() const restrict(amp) {
#ifdef __GPU__
    barrier(0);
#endif
  }
 private:
  tile_barrier() restrict(amp) {}
  template<int D0, int D1, int D2>
  friend class tiled_index;
};

template <typename T, int N> class array;

template <typename T, int N> class array_view;

// forward decls
template <int D0, int D1=0, int D2=0> class tiled_extent;

template <int N>
class extent {
public:
  static const int rank = N;

  typedef int value_type;

  extent() restrict(amp,cpu):
    extent(0, 0, 0) {}

  extent(const extent& other) restrict(amp,cpu):
    extent(other.e0_, other.e1_, other.e2_) {
    static_assert(N<=3, "Does not support N>3");
  }

  explicit extent(int e0) restrict(amp,cpu):
    extent(e0, 0, 0) {} // N==1

  extent(int e0, int e1) restrict(amp,cpu):
    extent(e0, e1, 0) {} // N==2

  __attribute__((annotate("deserialize"))) 
  extent(int e0, int e1, int e2) restrict(amp,cpu):
    e0_(e0), e1_(e1), e2_(e2) {} // N==3

  explicit extent(const int components[]) restrict(amp,cpu){
    assert(0 && "Not supported yet");
  }

  extent& operator=(const extent& other) restrict(amp,cpu);

  int operator[](unsigned int c) const restrict(amp,cpu) {
    static_assert(N<=3, "Does not support N>3");
    int r[3]={e0_, e1_, e2_};
    return r[c];
  }

  int &operator[](unsigned int c) restrict(amp,cpu) {
    static_assert(N<=3, "Does not support N>3");
    if (c==0) return e0_;
    if (c==1) return e1_;
    return e2_;
  }

  int size() const restrict(amp,cpu) {
    int retSize = e0_;
    if (rank > 1) retSize *= e1_;
    if (rank > 2) retSize *= e2_;
    return retSize;
  }

  bool contains(const index<N>& idx) const restrict(amp,cpu) {
    bool c = idx[0] < e0_;
    if (rank > 1) c = ( c && (idx[1] < e1_ ));
    if (rank > 2) c = ( c && (idx[2] < e2_ ));
    return c;
  }

  template <int D0> tiled_extent<D0> tile() const;
  template <int D0, int D1> tiled_extent<D0,D1> tile() const;
  template <int D0, int D1, int D2> tiled_extent<D0,D1,D2> tile() const;

  friend inline bool operator==(const extent<N>& lhs, const extent<N>& rhs)
    restrict(amp,cpu);

  friend inline bool operator!=(const extent<N>& lhs, const extent<N>& rhs)
    restrict(amp,cpu);

  template <int M>
    friend extent<M> operator-(const extent<M> &lhs, const extent<M> &rhs)
    restrict(amp, cpu);

  extent operator+(const index<N>& idx) restrict(amp,cpu);

  extent operator-(const index<N>& idx) restrict(amp,cpu);

  extent& operator+=(int rhs) restrict(amp,cpu);
  extent& operator-=(int rhs) restrict(amp,cpu);
  extent& operator*=(int rhs) restrict(amp,cpu);
  extent& operator/=(int rhs) restrict(amp,cpu);
  extent& operator%=(int rhs) restrict(amp,cpu);

  extent& operator++() restrict(amp,cpu);
  extent operator++(int) restrict(amp,cpu);
  extent& operator--() restrict(amp,cpu);
  extent operator--(int) restrict(amp,cpu);

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const {
    s.Append(sizeof(cl_int), &e0_);
    s.Append(sizeof(cl_int), &e1_);
    s.Append(sizeof(cl_int), &e2_);
  }

private:
  cl_int e0_, e1_, e2_; // Store the data
};

// C++AMP LPM 4.4.1

template <int D0, int D1=0, int D2=0>
class tiled_index {
 public:
  static const int rank = 3;
};
template <int N> class extent;
template <int D0>
class tiled_index<D0, 0, 0> {
 public:
  const index<1> global;
  const index<1> local;
  const index<1> tile;
  const index<1> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<1>& g) restrict(amp, cpu):global(g){}
  tiled_index(const tiled_index<D0>& o) restrict(amp, cpu):
    global(o.global), local(o.local) {}
  operator const index<1>() const restrict(amp,cpu) {
    return global;
  }
  const Concurrency::extent<1> tile_extent;
 private:
  //CLAMP
  __attribute__((annotate("__cxxamp_opencl_index")))
  __attribute__((always_inline)) tiled_index() restrict(amp)
#ifdef __GPU__
  : global(index<1>(get_global_id(0))),
    local(index<1>(get_local_id(0))),
    tile(index<1>(get_group_id(0))),
    tile_origin(index<1>(get_global_id(0)-get_local_id(0))),
    tile_extent(D0)
#endif // __GPU__
  {}
  template<int D, typename K>
  friend void parallel_for_each(tiled_extent<D>, const K&);
};

template <int D0, int D1>
class tiled_index<D0, D1, 0> {
 public:
  const index<2> global;
  const index<2> local;
  const index<2> tile;
  const index<2> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<2>& g) restrict(amp, cpu):global(g){}
  tiled_index(const tiled_index<D0, D1>& o) restrict(amp, cpu):
    global(o.global), local(o.local) {}
  operator const index<2>() const restrict(amp,cpu) {
    return global;
  }
  const Concurrency::extent<2> tile_extent;
 private:
  //CLAMP
  __attribute__((annotate("__cxxamp_opencl_index")))
  __attribute__((always_inline)) tiled_index() restrict(amp)
#ifdef __GPU__
  : global(index<2>(get_global_id(1), get_global_id(0))),
    local(index<2>(get_local_id(1), get_local_id(0))),
    tile(index<2>(get_group_id(1), get_group_id(0))),
    tile_origin(index<2>(get_global_id(1)-get_local_id(1),
                         get_global_id(0)-get_local_id(0))),
    tile_extent(D0, D1)
#endif // __GPU__
  {}
  template<int D0_, int D1_, typename K>
  friend void parallel_for_each(tiled_extent<D0_, D1_>, const K&);
};



template <int D0, int D1/*=0*/, int D2/*=0*/>
class tiled_extent : public extent<3>
{
public:
  static const int rank = 3;
  tiled_extent() restrict(amp,cpu);
  tiled_extent(const tiled_extent& other) restrict(amp,cpu): extent(other){}
  tiled_extent(const extent<3>& ext) restrict(amp,cpu): extent(ext) {}
  tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent pad() const restrict(amp,cpu) {
    tiled_extent padded(*this);
    padded[0] = ((padded[0] + D0 - 1)/D0) * D0;
    padded[1] = ((padded[1] + D1 - 1)/D1) * D1;
    padded[2] = ((padded[2] + D2 - 1)/D2) * D2;
    return padded;
  }
  tiled_extent truncate() const restrict(amp,cpu) {
    tiled_extent trunc(*this);
    trunc[0] = (trunc[0]/D0) * D0;
    trunc[1] = (trunc[1]/D1) * D1;
    trunc[2] = (trunc[2]/D2) * D2;
    return trunc;
  }

  // __declspec(property(get)) extent<3> tile_extent;
  extent<3> get_tile_extent() const;
  static const int tile_dim0 = D0;
  static const int tile_dim1 = D1;
  static const int tile_dim2 = D2;
  friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
  friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
};

template <int D0, int D1>
class tiled_extent<D0,D1,0> : public extent<2>
{
public:
  static const int rank = 2;
  tiled_extent() restrict(amp,cpu);
  tiled_extent(const tiled_extent& other) restrict(amp,cpu):extent(other) {}
  tiled_extent(const extent<2>& ext) restrict(amp,cpu):extent(ext) {}
  tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent pad() const restrict(amp,cpu) {
    tiled_extent padded(*this);
    padded[0] = ((padded[0] + D0 - 1)/D0) * D0;
    padded[1] = ((padded[1] + D1 - 1)/D1) * D1;
    return padded;
  }
  tiled_extent truncate() const restrict(amp,cpu) {
    tiled_extent trunc(*this);
    trunc[0] = (trunc[0]/D0) * D0;
    trunc[1] = (trunc[1]/D1) * D1;
    return trunc;
  }
  // __declspec(property(get)) extent<2> tile_extent;
  extent<2> get_tile_extent() const;
  static const int tile_dim0 = D0;
  static const int tile_dim1 = D1;
  friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
  friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
};

template<>
template <int D0, int D1>
tiled_extent<D0, D1> extent<2>::tile() const {
  return tiled_extent<D0, D1>(*this);
}

template <int D0>
class tiled_extent<D0,0,0> : public extent<1>
{
public:
  static const int rank = 1;
  tiled_extent() restrict(amp,cpu);
  tiled_extent(const tiled_extent& other) restrict(amp,cpu):
    extent(static_cast<extent>(other)) {}
  tiled_extent(const extent<1>& ext) restrict(amp,cpu):extent(ext) {}
  tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent pad() const restrict(amp,cpu) {
    tiled_extent padded(*this);
    padded[0] = ((padded[0] + D0 - 1)/D0) * D0;
    return padded;
  }
  tiled_extent truncate() const restrict(amp,cpu) {
    tiled_extent trunc(*this);
    trunc[0] = (trunc[0]/D0) * D0;
    return trunc;
  }
  // __declspec(property(get)) extent<1> tile_extent;
  extent<1> get_tile_extent() const;
  static const int tile_dim0 = D0;
  friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
  friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
};

template<>
template <int D0>
tiled_extent<D0> extent<1>::tile() const {
  return tiled_extent<D0>(*this);
}
// ------------------------------------------------------------------------
// For array's operator[](int i). This is a temporally workaround.
// N must be greater or equal to 2
template <typename T, int N>
class array_projection_helper
{
public:
  static array_view<T, N-1> project(array<T, N>* Array, int i) restrict(amp,cpu);
  static array_view<T, N-1> project(const array<T, N>* Array, int i) restrict(amp,cpu);
};

template <typename T>
class array_projection_helper<T, 1>
{
public:
  static T& project(array<T, 1>* Array, int i) restrict(amp,cpu);
  static const T& project(const array<T,1>* Array, int i) restrict(amp,cpu);
};
// ------------------------------------------------------------------------

#define __global __attribute__((address_space(1)))
#include "gmac_manage.h"

template <typename T, int N=1>
class array {
private:
  // Data members that do not show up at GPU side
  __attribute__((cpu)) int dummy_; //Don't define deserialization for this class
public:
#ifdef __GPU__
  typedef _data<T> gmac_buffer_t;
#else
  typedef _data_host<T> gmac_buffer_t;
#endif
  static const int rank = N;
  typedef T value_type;
  array() = delete;

  explicit array(const extent<N>& ext);

  explicit array(int e0): array(extent<1>(e0)) {}

  explicit array(int e0, int e1): 
    array(Concurrency::extent<N>(e0, e1)) {
      assert(N == 2 && "For constructing array<T, 2> only");
    }

  explicit array(int e0, int e1, int e2):
    array(Concurrency::extent<N>(e0, e1, e2)) {
      assert(N == 3 && "For constructing array<T, 3> only");
    }

  array(int e0, accelerator_view av) {
    assert(0 && "Only applicable to array<T, 1>");
  }

  array(int e0, int e1, accelerator_view av);

  array(int e0, int e1, int e2, accelerator_view av) {
    assert(0 && "Not Implemented Yet.");
  }

  array(const extent<N>& extent, accelerator_view av, accelerator_view associated_av); //staging

  array(int e0, accelerator_view av, accelerator_view associated_av);

  array(int e0, int e1, accelerator_view av, accelerator_view associated_av); //staging

  array(int e0, int e1, int e2, accelerator_view av, accelerator_view associated_av); //staging

  template <typename InputIterator>
    array(const extent<N>& ext, InputIterator srcBegin):
	array(ext, srcBegin, accelerator().get_default_view()) {}

  template <typename InputIterator>
  array(const extent<N>& extent, InputIterator srcBegin, InputIterator srcEnd);
#ifndef __GPU__
  // CAVEAT: ACCELERATOR
  template <typename InputIterator>
  array(int e0, InputIterator srcBegin) : extent(e0),
    accelerator_view_(accelerator().get_default_view()) {
    if (e0) {
      m_device.reset(GMACAllocator<T>().allocate(e0),
        GMACDeleter<T>());
      InputIterator srcEnd = srcBegin;
      std::advance(srcEnd, e0);
      std::copy(srcBegin, srcEnd, m_device.get());
    }
  }
#endif
  template <typename InputIterator>
  array(int e0, InputIterator srcBegin, InputIterator srcEnd);

  template <typename InputIterator>
  array(int e0, int e1, InputIterator srcBegin):
    array(Concurrency::extent<N>(e0, e1), srcBegin) {
      assert(N == 2 && "For constructing array<T, 2> only");
    }

  template <typename InputIterator>
  array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd);

  template <typename InputIterator>
  array(int e0, int e1, int e2, InputIterator srcBegin):
    array(Concurrency::extent<N>(e0, e1, e2), srcBegin) {
      assert(N == 3 && "For constructing array<T, 3> only");
    }



  template <typename InputIterator>
  array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd);

  template <typename InputIterator>
  array(const extent<N>& extent, InputIterator srcBegin,
    accelerator_view av, accelerator_view associated_av); // staging

  template <typename InputIterator>
  array(const extent<N>& extent, InputIterator srcBegin, InputIterator srcEnd,
    accelerator_view av, accelerator_view associated_av); // staging

  template <typename InputIterator>
  array(int e0, InputIterator srcBegin,
  accelerator_view av, accelerator_view associated_av); // staging

  template <typename InputIterator>
  array(int e0, InputIterator srcBegin, InputIterator srcEnd,
  accelerator_view av, accelerator_view associated_av); // staging

  template <typename InputIterator>
  array(int e0, int e1, InputIterator srcBegin,
  accelerator_view av, accelerator_view associated_av); // staging

  template <typename InputIterator>
  array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd,
  accelerator_view av, accelerator_view associated_av); // staging

  template <typename InputIterator>
  array(int e0, int e1, int e2, InputIterator srcBegin,
  accelerator_view av, accelerator_view associated_av); // staging

  template <typename InputIterator>
  array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd,
  accelerator_view av, accelerator_view associated_av); // staging

  template <typename InputIterator>
  array(const extent<N>& ext, InputIterator srcBegin, accelerator_view av)
    : extent(ext), m_device(nullptr)
#ifdef __GPU__
    { assert(0 && "Unrechable"); }
#else
    , accelerator_view_(av) {
    if (ext.size()) {
      m_device.reset(GMACAllocator<T>().allocate(ext.size()),
        GMACDeleter<T>());
      InputIterator srcEnd = srcBegin;
      std::advance(srcEnd, extent.size());
      std::copy(srcBegin, srcEnd, m_device.get());
    }
  }
#endif

  template <typename InputIterator>
  array(int e0, InputIterator SrcBegin, accelerator_view av);

  template <typename InputIterator>
  array(int e0, int e1, InputIterator SrcBegin, accelerator_view av);

  template <typename InputIterator>
  array(int e0, int e1, int e2, InputIterator SrcBegin, accelerator_view av);

  template <typename InputIterator>
  array(const extent<N>& extent, InputIterator srcBegin, InputIterator srcEnd,
  accelerator_view av);

  template <typename InputIterator>
  array(int e0, InputIterator srcBegin, InputIterator srcEnd, accelerator_view av);

  template <typename InputIterator>
  array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd, accelerator_view av);

  template <typename InputIterator>
  array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd, accelerator_view av) {
    assert(0 && "Not Implemented Yet.");
  }


  explicit array(const array_view<const T,N>& src):
    array(src.extent) {
      memmove(const_cast<void*>(reinterpret_cast<const void*>(m_device.get())),
	  reinterpret_cast<const void*>(src.cache_.get()),
	  extent.size() * sizeof(T));
    }

  // Not really in C++AMP spec 1.0, but required by samples
  explicit array(const array_view<T,N>& src):
    array(src.extent) {
      memmove(const_cast<void*>(reinterpret_cast<const void*>(m_device.get())),
	  reinterpret_cast<const void*>(src.cache_.get()),
	  extent.size() * sizeof(T));
    }


  array(const array_view<const T,N>& src,

  accelerator_view av, accelerator_view associated_av); // staging

  array(const array_view<const T,N>& src, accelerator_view av);

  array(const array& other);

  array(array&& other);

  array& operator=(const array& other);

  array& operator=(array&& other);

  array& operator=(const array_view<const T,N>& src);

  void copy_to(array& dest) const;

  void copy_to(const array_view<T,N>& dest) const;

  //__declspec(property(get)) Concurrency::extent<N> extent;

  const Concurrency::extent<N> extent;
  Concurrency::extent<N> get_extent() const {
    return extent;
  }


  // __declspec(property(get)) accelerator_view accelerator_view;
  accelerator_view get_accelerator_view() const;

  // __declspec(property(get)) accelerator_view associated_accelerator_view;
  accelerator_view get_associated_accelerator_view() const;

  __global T& operator[](const index<N>& idx) restrict(amp,cpu) {
    if (rank == 1)
      return reinterpret_cast<__global T*>(m_device.get())[idx[0]];
    else if (rank == 2)
      return reinterpret_cast<__global T*>(m_device.get())
	[idx[0] * extent[1] + idx[1]];
    else if (rank == 3)
      return reinterpret_cast<__global T*>(m_device.get())
	[idx[0] * extent[1] * extent[2] + idx[1]*extent[2] + idx[2]];
  }
  __global const T& operator[](const index<N>& idx) const restrict(amp,cpu) {
    if (rank == 1)
      return reinterpret_cast<__global T*>(m_device.get())[idx[0]];
    else if (rank == 2)
      return reinterpret_cast<__global T*>(m_device.get())
	[idx[0] * extent[1] + idx[1]];
    else if (rank == 3)
      return reinterpret_cast<__global T*>(m_device.get())
	[idx[0] * extent[1] * extent[2] + idx[1]*extent[2] + idx[2]];
  }

  auto operator[](int i) restrict(amp,cpu) -> decltype(array_projection_helper<T, N>::project((array<T,N> *)NULL, i));

  auto operator[](int i) const restrict(amp,cpu) -> decltype(array_projection_helper<T, N>::project((const array<T, N>* )NULL, i));

  const T& operator()(const index<N>& idx) const restrict(amp,cpu);

  __global T& operator()(const index<N>& idx) restrict(amp,cpu) {
    return (*this)[idx];
  }

  __global T& operator()(int i0, int i1) restrict(amp,cpu) {
    return (*this)[index<2>(i0, i1)];
  }
  __global const T& operator()(int i0, int i1) const restrict(amp,cpu) {
    return (*this)[index<2>(i0, i1)];
  }

  __global T& operator()(int i0, int i1, int i2) restrict(amp,cpu) {
    return (*this)[index<3>(i0, i1, i2)];
  }

  __global const T& operator()(int i0, int i1, int i2) 
    const restrict(amp,cpu) {
    return (*this)[index<3>(i0, i1, i2)];
  }


  auto operator()(int i) restrict(amp,cpu) -> decltype(array_projection_helper<T, N>::project((array<T,N> *)NULL, i));

  auto operator()(int i) const restrict(amp,cpu) -> decltype(array_projection_helper<T, N>::project((const array<T, N>* )NULL, i));

  array_view<T,N> section(const index<N>& idx,
      const Concurrency::extent<N>& ext) restrict(amp,cpu);

  array_view<const T,N> section(const index<N>& idx,
      const Concurrency::extent<N>& ext) const restrict(amp,cpu);

  array_view<T,N> section(const index<N>& idx) restrict(amp,cpu);

  array_view<const T,N> section(const index<N>& idx) const restrict(amp,cpu);

  array_view<T,1> section(int i0, int e0) restrict(amp,cpu);

  array_view<const T,1> section(int i0, int e0) const restrict(amp,cpu);

  array_view<T,2> section(int i0, int i1, int e0, int e1) restrict(amp,cpu);

  array_view<const T,2> section(int i0, int i1, int e0, int e1) const restrict(amp,cpu);

  array_view<T,3> section(int i0, int i1, int i2,
  int e0, int e1, int e2) restrict(amp,cpu);

  array_view<const T,3> section(int i0, int i1, int i2,
  int e0, int e1, int e2) const restrict(amp,cpu);

  template <typename ElementType>
  array_view<ElementType,1> reinterpret_as() restrict(amp,cpu);

  template <typename ElementType>
  array_view<const ElementType,1> reinterpret_as() const restrict(amp,cpu);

  template <int K>
    array_view<T,K> view_as(const Concurrency::extent<K>& viewExtent)
    restrict(amp,cpu);

  template <int K>
    array_view<const T,K> view_as(const Concurrency::extent<K>& viewExtent)
    const restrict(amp,cpu);

  operator std::vector<T>() const {
    T *begin = reinterpret_cast<T*>(m_device.get()),
      *end = reinterpret_cast<T*>(m_device.get()+extent.size());
    return std::vector<T>(begin, end);
  }

  T* data() restrict(amp,cpu) {
    return m_device.get();
  }

  const T* data() const restrict(amp,cpu) {
    return m_device.get();
  }

  ~array() { // For GMAC
    m_device.reset();
  }

  const gmac_buffer_t& internal() const { return m_device; }

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const;

  __attribute__((annotate("deserialize"))) 
  array(__global T *p, cl_int e0, cl_int e1, cl_int e2) restrict(amp);
  // End CLAMP
private:
#ifndef __GPU__
  accelerator_view accelerator_view_;
#endif
  gmac_buffer_t m_device;
};

template <typename T>
class array<T, 1> {
 private:
#ifndef __GPU__
  //Don't define deserialization for this class
  __attribute__((cpu)) int dummy_;
#endif
 public:
#ifdef __GPU__
  typedef _data<T> gmac_buffer_t;
#else
  typedef _data_host<T> gmac_buffer_t;
#endif
  static const int rank = 1;
  typedef T value_type;

  array() = delete;

  explicit array(const Concurrency::extent<1>& ext);

  explicit array(int e0): array(Concurrency::extent<1>(e0)) {}

  array(const Concurrency::extent<1>& extent,
    accelerator_view av, accelerator_view associated_av) {
    //staging
    assert(0 && "Staging array is not supported");
  }

  array(int e0, accelerator_view av, accelerator_view associated_av) {
    //staging
    assert(0 && "Staging array is not supported");
  }

  array(const Concurrency::extent<1>& extent, accelerator_view av);

  array(int e0, accelerator_view av);

  template <typename InputIterator>
    array(const Concurrency::extent<1>& extent, InputIterator srcBegin):
      array(extent[0], srcBegin) {}

  template <typename InputIterator>
    array(const Concurrency::extent<1>& extent, InputIterator srcBegin,
      InputIterator srcEnd)
    : array(extent, srcBegin, srcEnd, accelerator().get_default_view()) {}

  template <typename InputIterator>
    array(int e0, InputIterator srcBegin)
    : array(Concurrency::extent<1>(e0), srcBegin,
        accelerator().get_default_view()) {}

  template <typename InputIterator>
    array(int e0, InputIterator srcBegin, InputIterator srcEnd):
      array(Concurrency::extent<1>(e0), srcBegin, srcEnd) {}

  template <typename InputIterator>
    array(const Concurrency::extent<1>& extent, InputIterator srcBegin,
        accelerator_view av, accelerator_view associated_av) { // staging
      assert(0 && "Staging array is not supported");
    }

  template <typename InputIterator>
    array(const Concurrency::extent<1>& extent, InputIterator srcBegin,
        InputIterator srcEnd,
        accelerator_view av, accelerator_view associated_av) { // staging
      assert(0 && "Staging array is not supported");
    }

  template <typename InputIterator>
    array(int e0, InputIterator srcBegin,
        accelerator_view av, accelerator_view associated_av) { // staging
      assert(0 && "Staging array is not supported");
    }

  template <typename InputIterator>
    array(int e0, InputIterator srcBegin, InputIterator srcEnd,
        accelerator_view av, accelerator_view associated_av) { // staging
      assert(0 && "Staging array is not supported");
    }

  template <typename InputIterator>
  array(const Concurrency::extent<1>& ext, InputIterator srcBegin,
    accelerator_view av): extent(ext), m_device(nullptr)
#ifdef __GPU__
    { assert(0 && "Unrechable"); }
#else
    , accelerator_view_(av) {
    if (ext[0]) {
      m_device.reset(GMACAllocator<T>().allocate(ext[0]),
        GMACDeleter<T>());
      InputIterator srcEnd = srcBegin;
      std::advance(srcEnd, extent[0]);
      std::copy(srcBegin, srcEnd, m_device.get());
    }
  }
#endif

  template <typename InputIterator>
    array(int e0, InputIterator SrcBegin, accelerator_view av):
      array(Concurrency::extent<1>(e0), SrcBegin, av) {}

  template <typename InputIterator>
    array(const Concurrency::extent<1>& ext, InputIterator srcBegin,
        InputIterator srcEnd, accelerator_view av): extent(ext),
#ifdef __GPU__
    m_device(nullptr)  { assert(0 && "Unrechable"); }
#else
    accelerator_view_(av) {
    if (ext[0]) {
      m_device.reset(GMACAllocator<T>().allocate(ext.size()),
        GMACDeleter<T>());
      InputIterator srcCopyEnd = srcBegin;
      std::advance(srcCopyEnd,
        std::min(std::distance(srcBegin, srcEnd),
          decltype(std::distance(srcBegin, srcEnd))(ext.size())));
      std::copy(srcBegin, srcCopyEnd, m_device.get());
    }
  }
#endif

  template <typename InputIterator>
    array(int e0, InputIterator srcBegin, InputIterator srcEnd,
        accelerator_view av): array(Concurrency::extent<1>(e0), srcBegin,
          srcEnd, av) {}

  explicit array(const array_view<const T,1>& src);

  array(const array_view<const T,1>& src,
    accelerator_view av, accelerator_view associated_av) { // staging
      assert(0 && "Staging array is not supported");
    }

  array(const array_view<const T,1>& src, accelerator_view av);

  array(const array& other);

  array(array&& other);

  array& operator=(const array& other);

  array& operator=(array&& other);

  array& operator=(const array_view<const T,1>& src);

  void copy_to(array& dest) const;

  void copy_to(const array_view<T,1>& dest) const;

  //__declspec(property(get)) Concurrency::extent<N> extent;
  const Concurrency::extent<1> extent;
  const Concurrency::extent<1> &get_extent() const restrict(amp,cpu) {
    return extent;
  }

  // __declspec(property(get)) accelerator_view accelerator_view;
  accelerator_view get_accelerator_view() const;

  // __declspec(property(get)) accelerator_view associated_accelerator_view;
  accelerator_view get_associated_accelerator_view() const;

  __global T& operator[](const index<1>& idx) restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(m_device.get())[idx[0]];
  }
  __global const T& operator[](const index<1>& idx) const restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(m_device.get())[idx[0]];
  }

  __global T& operator[](int i0) restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(m_device.get())[i0];
  }

  const __global T& operator[](int i0) const restrict(amp,cpu) {
    return reinterpret_cast<__global T*>(m_device.get())[i0];
  }

  __global T& operator()(const index<1>& idx) restrict(amp,cpu) {
    return this->operator[](idx);
  }

  const __global T& operator()(const index<1>& idx) const restrict(amp,cpu) {
    return this->operator[](idx);
  }

  __global T& operator()(int i0) restrict(amp,cpu) {
    return this->operator[](i0);
  }

  const __global T& operator()(int i0) const restrict(amp,cpu) {
    return this->operator[](i0);
  }

  array_view<T,1> section(const index<1>& idx,
    const Concurrency::extent<1>& ext) restrict(amp,cpu);

  array_view<const T,1> section(const index<1>& idx,
    const Concurrency::extent<1>& ext) const restrict(amp,cpu);

  array_view<T,1> section(const index<1>& idx) restrict(amp,cpu);

  array_view<const T,1> section(const index<1>& idx) const restrict(amp,cpu);

  template <typename ElementType>
    array_view<ElementType, 1> reinterpret_as() restrict(amp,cpu);

  template <typename ElementType>
    array_view<const ElementType, 1> reinterpret_as() const restrict(amp,cpu);

  template <int K>
    array_view<T,K> view_as(const Concurrency::extent<K>& viewExtent)
    restrict(amp,cpu) {
      array_view<T, 1> av(*this);
      return av.view_as(viewExtent);
    }

  template <int K>
    array_view<const T,K> view_as(const Concurrency::extent<K>& viewExtent)
    const restrict(amp,cpu);

  operator std::vector<T>() const;

  T* data() restrict(amp,cpu) {
    return m_device.get();
  }

  const T* data() const restrict(amp,cpu) {
    return m_device.get();
  }

  ~array() { // For GMAC
    m_device.reset();
  }

  const gmac_buffer_t& internal() const restrict(amp, cpu) { return m_device; }

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const;

  __attribute__((annotate("deserialize"))) 
  array(__global T *p, cl_int e) restrict(amp);
  // End CLAMP
private:
#ifndef __GPU__
  accelerator_view accelerator_view_;
#endif
  gmac_buffer_t m_device;
#undef __global
};

template <typename T, int N = 1>
class array_view
{
public:
  static const int rank = N;
  typedef T value_type;

  array_view() = delete;
  array_view(array<T,N>& src) restrict(amp,cpu) {}
  template <typename Container>
    array_view(const extent<N>& extent, Container& src);
  array_view(const extent<N>& extent, value_type* src) restrict(amp,cpu);

  array_view(const array_view& other) restrict(amp,cpu);

  array_view& operator=(const array_view& other) restrict(amp,cpu);

  void copy_to(array<T,N>& dest) const;
  void copy_to(const array_view& dest) const;

  // __declspec(property(get)) extent<N> extent;
  extent<N> get_extent() const;

  // These are restrict(amp,cpu)
  T& operator[](const index<N>& idx) const restrict(amp,cpu);

  T& operator()(const index<N>& idx) const restrict(amp,cpu);
  array_view<T,N-1> operator()(int i) const restrict(amp,cpu);

  array_view<T,N> section(const index<N>& idx, const extent<N>& ext) restrict(amp,cpu);
  array_view<T,N> section(const index<N>& idx) const restrict(amp,cpu);

  void synchronize() const;
  completion_future synchronize_async() const;

  void refresh() const;
  void discard_data() const;
};
#include <array_view.h>

#if 0 // Cause ambiguity on clang 
template <typename T, int N>
class array_view<const T,N>
{
public:
  static const int rank = N;
  typedef const T value_type;

  array_view() = delete;
  array_view(const array<T,N>& src) restrict(amp,cpu);
  template <typename Container>
    array_view(const extent<N>& extent, const Container& src);
  array_view(const extent<N>& extent, const value_type* src) restrict(amp,cpu);
  array_view(const array_view<T,N>& other) restrict(amp,cpu);

  array_view(const array_view<const T,N>& other) restrict(amp,cpu);

  array_view& operator=(const array_view& other) restrict(amp,cpu);

  void copy_to(array<T,N>& dest) const;
  void copy_to(const array_view<T,N>& dest) const;

  // __declspec(property(get)) extent<N> extent;
  extent<N> get_extent() const;

  const T& operator[](const index<N>& idx) const restrict(amp,cpu);
  array_view<const T,N-1> operator[](int i) const restrict(amp,cpu);

  const T& operator()(const index<N>& idx) const restrict(amp,cpu);
  array_view<const T,N-1> operator()(int i) const restrict(amp,cpu);

  array_view<const T,N> section(const index<N>& idx, const extent<N>& ext) const restrict(amp,cpu);
  array_view<const T,N> section(const index<N>& idx) const restrict(amp,cpu);

  void refresh() const;
};
#endif
// class index operators
template <int N>
bool operator==(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu);
template <int N>
bool operator!=(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu);

template <int N>
index<N>& index<N>::operator+=(int rhs) restrict(amp,cpu) {
  if (rank > 0)
      i0_ += rhs;
  if (rank > 1)
      i1_ += rhs;
  if (rank > 2)
      i2_ += rhs;
  return *this;
}
template <int N>
index<N> operator+(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu);
template <int N>
index<N> operator+(const index<N>& lhs, int rhs) restrict(amp,cpu) {
  index<N> __r = lhs;
  __r += rhs;
  return __r;
}
template <int N>
index<N> operator+(int lhs, const index<N>& rhs) restrict(amp,cpu);
template <int N>
index<N> operator-(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu);
template <int N>
index<N> operator-(const index<N>& lhs, int rhs) restrict(amp,cpu);
template <int N>
index<N> operator-(int lhs, const index<N>& rhs) restrict(amp,cpu);
template <int N>
index<N> operator*(const index<N>& lhs, int rhs) restrict(amp,cpu);
template <int N>
index<N> operator*(int lhs, const index<N>& rhs) restrict(amp,cpu);
template <int N>
index<N> operator/(const index<N>& lhs, int rhs) restrict(amp,cpu);
template <int N>
index<N> operator/(int lhs, const index<N>& rhs) restrict(amp,cpu);
template <int N>
index<N> operator%(const index<N>& lhs, int rhs) restrict(amp,cpu);
template <int N>
index<N> operator%(int lhs, const index<N>& rhs) restrict(amp,cpu);

// class extent operators
template <int N>
bool operator==(const extent<N>& lhs, const extent<N>& rhs) restrict(amp,cpu);
template <int N>
bool operator!=(const extent<N>& lhs, const extent<N>& rhs) restrict(amp,cpu);

template <int N>
extent<N> operator+(const extent<N>& lhs, int rhs) restrict(amp,cpu);
template <int N> 
extent<N> operator+(int lhs, const extent<N>& rhs) restrict(amp,cpu);
template <int N>
extent<N> operator-(const extent<N>& lhs, int rhs) restrict(amp,cpu);
template <int N>
extent<N> operator-(int lhs, const extent<N>& rhs) restrict(amp,cpu);
template <int N>
extent<N> operator*(const extent<N>& lhs, int rhs) restrict(amp,cpu);
template <int N>
extent<N> operator*(int lhs, const extent<N>& rhs) restrict(amp,cpu);
template <int N>
extent<N> operator/(const extent<N>& lhs, int rhs) restrict(amp,cpu);
template <int N>
extent<N> operator/(int lhs, const extent<N>& rhs) restrict(amp,cpu);
template <int N>
extent<N> operator%(const extent<N>& lhs, int rhs) restrict(amp,cpu);
template <int N>
extent<N> operator%(int lhs, const extent<N>& rhs) restrict(amp,cpu);


template <int N, typename Kernel>
void parallel_for_each(extent<N> compute_domain, const Kernel& f);

template <int D0, int D1, int D2, typename Kernel>
void parallel_for_each(tiled_extent<D0,D1,D2> compute_domain, const Kernel& f);

template <int D0, int D1, typename Kernel>
void parallel_for_each(tiled_extent<D0,D1> compute_domain, const Kernel& f);

template <int D0, typename Kernel>
void parallel_for_each(tiled_extent<D0> compute_domain, const Kernel& f);

template <int N, typename Kernel>
void parallel_for_each(const accelerator_view& accl_view, extent<N> compute_domain, const Kernel& f);

template <int D0, int D1, int D2, typename Kernel>
void parallel_for_each(const accelerator_view& accl_view, tiled_extent<D0,D1,D2> compute_domain, const Kernel& f);

template <int D0, int D1, typename Kernel>
void parallel_for_each(const accelerator_view& accl_view, tiled_extent<D0,D1> compute_domain, const Kernel& f);

template <int D0, typename Kernel>
void parallel_for_each(const accelerator_view& accl_view, tiled_extent<D0> compute_domain, const Kernel& f);

} // namespace Concurrency
namespace concurrency = Concurrency;
// Specialization and inlined implementation of C++AMP classes/templates
#include "amp_impl.h"
#include "parallel_for_each.h"


namespace Concurrency {
//std::vector====array_view
//1D
template <typename _Value_type>
void copy(typename std::vector<_Value_type>::iterator _SrcFirst,
          const array_view<_Value_type, 1> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i) {
    _Dest(i) = *_SrcFirst;
    _SrcFirst++;
  }
}

template <typename _Value_type>
void copy(const array_view<_Value_type, 1> &_Src,
          typename std::vector<_Value_type>::iterator _DestIter) {
  for(int i = 0; i < _Src.extent[0]; ++i) {
    *_DestIter = _Src(i);
    _DestIter++;
  }
}

//2D
template <typename _Value_type>
void copy(typename std::vector<_Value_type>::iterator _SrcFirst,
          const array_view<_Value_type, 2> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i) {
    for(int j = 0; j < _Dest.extent[1]; ++j) {
      _Dest(i, j) = *_SrcFirst;
      _SrcFirst++;
    }
  }
}

template <typename _Value_type>
void copy(const array_view<_Value_type, 2> &_Src,
          typename std::vector<_Value_type>::iterator _DestIter) {
  for(int i = 0; i < _Src.extent[0]; ++i) {
    for(int j = 0; j < _Src.extent[1]; ++j) {
      *_DestIter = _Src(i, j);
     _DestIter++;
    }
  }
}

//array=====std::vector
//1D
template <typename _Value_type>
void copy(const array<_Value_type, 1> &_Src,
          typename std::vector<_Value_type>::iterator _DestIter) {
  for(int i = 0; i < _Src.extent[0]; ++i) {
    *_DestIter = _Src(i);
    _DestIter++;
  }
}

template <typename _Value_type>
void copy(typename std::vector<_Value_type>::iterator _SrcFirst,
          array<_Value_type, 1> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i) {
    _Dest(i) = *_SrcFirst;
    _SrcFirst++;
  }
}
//2D
template <typename _Value_type>
void copy(const array<_Value_type, 2> &_Src,
          typename std::vector<_Value_type>::iterator _DestIter) {
  for(int i = 0; i < _Src.extent[0]; ++i) {
    for(int j = 0; j < _Src.extent[1]; ++j) {
      *_DestIter = _Src(i, j);
      _DestIter++;
    }
  } 
}

template <typename _Value_type>
void copy(typename std::vector<_Value_type>::iterator _SrcFirst,
          array<_Value_type, 2> &_Dest) {   
  for(int i = 0; i < _Dest.extent[0]; ++i) {
    for(int j = 0; j < _Dest.extent[1]; ++j) {
      _Dest(i, j) = *_SrcFirst;
      _SrcFirst++;
    }
  }
}
//array====array_view
//1D
template <typename _Value_type>
void copy(const array<_Value_type, 1> &_Src,
          array_view<_Value_type, 1> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i)
    _Dest(i) = _Src(i);
}

template <typename _Value_type>
void copy(const array_view<_Value_type, 1> &_Src,
          array<_Value_type, 1> &_Dest) {
  for(int i = 0; i < _Src.extent[0]; ++i)
    _Dest(i) = _Src(i);
}

//2D
template <typename _Value_type>
void copy(const array<_Value_type, 2> &_Src,
          array_view<_Value_type, 2> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i) {
    for(int j = 0; j < _Dest.extent[1]; ++j) {
      _Dest(i, j) = _Src(i, j);
    }
  }
}

template <typename _Value_type>
void copy(const array_view<_Value_type, 2> &_Src,
          array<_Value_type, 2> &_Dest) {
  for(int i = 0; i < _Src.extent[0]; ++i)
    for(int j = 0; j < _Src.extent[1]; ++j)
      _Dest(i, j) = _Src(i, j);
}

//array====array
//1D
template <typename _Value_type>
void copy(const array<_Value_type, 1> &_Src, array<_Value_type, 1> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i)
    _Dest(i) = _Src(i);
}

//2D
template <typename _Value_type>
void copy(const array<_Value_type, 2> &_Src, array<_Value_type, 2> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i)
    for(int j = 0; j < _Dest.extent[1]; ++j)
      _Dest(i, j) = _Src(i, j);  
}

//array_view====array_view
//1D
template <typename _Value_type>
void copy(const array_view<_Value_type, 1> &_Src,
          array_view<_Value_type, 1> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i)
    _Dest(i) = _Src(i);
}

//2D
template <typename _Value_type>
void copy(const array_view<_Value_type, 2> &_Src,
          array_view<_Value_type, 2> &_Dest) {
  for(int i = 0; i < _Dest.extent[0]; ++i)
    for(int j = 0; j < _Dest.extent[1]; ++j)
      _Dest(i, j) = _Src(i, j);
}

#ifdef __GPU__
extern "C" unsigned atomic_add_local(volatile __attribute__((address_space(3))) unsigned *p, unsigned val) restrict(amp);
static inline unsigned atomic_fetch_add(__attribute__((address_space(3)))unsigned *x, unsigned y) restrict(amp) { 
  return atomic_add_local(reinterpret_cast<volatile __attribute__((address_space(3))) unsigned *>(x), y);
}
#else
extern unsigned atomic_fetch_add(__attribute__((address_space(3)))unsigned *x, unsigned y) restrict(amp);
#endif

#ifdef __GPU__
extern "C" int atomic_add_global(volatile __attribute__((address_space(1))) int *p, int val) restrict(amp, cpu);
static inline int atomic_fetch_add(__attribute__((address_space(1))) int *x, int y) restrict(amp) { 
  return atomic_add_global(reinterpret_cast<volatile __attribute__((address_space(1))) int *>(x), y);
}
#else
extern int atomic_fetch_add(__attribute__((address_space(1)))int *x, int y) restrict(amp, cpu);
#endif

}//namespace Concurrency

