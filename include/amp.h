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
#include <gmac/cl.h>
#include <memory>
#include <algorithm>
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
  __declspec(property(get)) std::wstring device_path;
  __declspec(property(get)) unsigned int version; // hiword=major, loword=minor
  __declspec(property(get)) std::wstring description;
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
  //CLAMP
  cl_device_id clamp_get_device_id() const { return device_; }
  //End CLAMP
 private:
  cl_platform_id platform_;
  cl_device_id device_;
  static accelerator_view *default_view_;
};

class completion_future;
class accelerator_view {
public:
  accelerator_view() = delete;
  accelerator_view(const accelerator_view& other);
  accelerator_view& operator=(const accelerator_view& other) {
    clRetainCommandQueue(other.command_queue_);
    clReleaseCommandQueue(command_queue_);
    command_queue_ = other.command_queue_;
    clRetainContext(other.context_);
    clReleaseContext(context_);
    context_ = other.context_;
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
  //CLAMP-specific
  ~accelerator_view();
  const cl_device_id& clamp_get_device(void) const { return device_; }
  cl_context clamp_get_context(void) const { return context_; }
  cl_command_queue clamp_get_command_queue(void) const { return command_queue_; }
  //End CLAMP-specific
 private:
  //CLAMP-specific
  friend class accelerator;
  accelerator_view(cl_device_id d);
  cl_device_id device_;     
  cl_context context_;
  cl_command_queue command_queue_;
  //End CLAMP-specific
};
//CLAMP
extern "C" __attribute__((pure)) int get_global_id(int n) restrict(amp);
extern "C" __attribute__((pure)) int get_local_id(int n) restrict(amp);
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

  index() restrict(amp,cpu) {
    for (int i = 0; i < N; i++)
      m_internal[i] = 0;
  }

  index(const index& other) restrict(amp,cpu) {
    for (int i = 0; i < N; i++)
      m_internal[i] = other.m_internal[i];
  }

  explicit index(int i0) restrict(amp,cpu) { // N==1
    m_internal[0] = i0;
  }

  index(int i0, int i1) restrict(amp,cpu) { // N==2
    m_internal[0] = i0;
    if (N==2)
      m_internal[1] = i1;
  }

  index(int i0, int i1, int i2) restrict(amp,cpu); // N==3

  explicit index(const int components[]) restrict(amp,cpu);

  index& operator=(const index& other) restrict(amp,cpu);

  int operator[](unsigned int c) const restrict(amp,cpu) {
    return m_internal[c];
  }

  int& operator[](unsigned int c) restrict(amp,cpu) {
    return m_internal[c];
  }
  
  index& operator+=(const index& rhs) restrict(amp,cpu);
  index& operator-=(const index& rhs) restrict(amp,cpu);
  
  index& operator+=(int rhs) restrict(amp,cpu);
  index& operator-=(int rhs) restrict(amp,cpu);
  index& operator*=(int rhs) restrict(amp,cpu);
  index& operator/=(int rhs) restrict(amp,cpu);
  index& operator%=(int rhs) restrict(amp,cpu);
  
  index& operator++() restrict(amp,cpu) {
    //FIXME extent?
    m_internal[0]++;
    return *this;
  }
  index operator++(int) restrict(amp,cpu) {
    //FIXME extent?
    index ret = *this;
    m_internal[0]++;
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
      m_internal[0] = get_global_id(0);
    } else if (rank == 2) {
      m_internal[0] = get_global_id(1);
      m_internal[1] = get_global_id(0);
    }
  }
#else
  ;
#endif // __GPU__
  //End CLAMP
  int m_internal[N]; // Store the data 
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

// C++AMP LPM 4.4.1
// forward decls
template <int D0, int D1=0, int D2=0> class tiled_extent;

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
  const tile_barrier barrier;
  tiled_index(const index<1>& g) restrict(amp, cpu):global(g){}
  tiled_index(const tiled_index<D0>& o) restrict(amp, cpu):
    global(o.global), local(o.local) {}
  operator const index<1>() const restrict(amp,cpu) {
    return global;
  }
 private:
  //CLAMP
  __attribute__((annotate("__cxxamp_opencl_index")))
  tiled_index() restrict(amp)
#ifdef __GPU__
  : global(index<1>(get_global_id(0))),
    local(index<1>(get_local_id(0)))
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
  const tile_barrier barrier;
  tiled_index(const index<2>& g) restrict(amp, cpu):global(g){}
  tiled_index(const tiled_index<D0, D1>& o) restrict(amp, cpu):
    global(o.global), local(o.local) {}
  operator const index<2>() const restrict(amp,cpu) {
    return global;
  }
 private:
  //CLAMP
  __attribute__((annotate("__cxxamp_opencl_index")))
  tiled_index() restrict(amp)
#ifdef __GPU__
  : global(index<2>(get_global_id(1), get_global_id(0))),
    local(index<2>(get_local_id(1), get_local_id(0)))
#endif // __GPU__
  {}
  template<int D0_, int D1_, typename K>
  friend void parallel_for_each(tiled_extent<D0_, D1_>, const K&);
};



template <typename T, int N> class array;

template <typename T, int N> class array_view;

template <int N>
class extent {
public:
  static const int rank = N;

  typedef int value_type;

  extent() restrict(amp,cpu);

  extent(const extent& other) restrict(amp,cpu) {
    for (int i = 0; i < N; ++i)
      m_internal[i] = other[i];
  }

  explicit extent(int e0) restrict(amp,cpu) { // N==1
    m_internal[0] = e0;
  }

  extent(int e0, int e1) restrict(amp,cpu) { // N==2
    m_internal[0] = e0;
    m_internal[1] = e1;
  }

  extent(int e0, int e1, int e2) restrict(amp,cpu) { // N==3
    m_internal[0] = e0;
    m_internal[1] = e1;
    m_internal[2] = e2;
  }

  explicit extent(const int components[]) restrict(amp,cpu);

  extent& operator=(const extent& other) restrict(amp,cpu);

  int operator[](unsigned int c) const restrict(amp,cpu) {
    return m_internal[c];
  }

  int& operator[](unsigned int c) restrict(amp,cpu) {
    return m_internal[c];
  }

  int size() const restrict(amp,cpu) {
    int retSize = m_internal[0];
    for (int i = 1; i < N; ++i){
      retSize *= m_internal[i];
    }
    return retSize;
  }

  bool contains(const index<N>& idx) const restrict(amp,cpu);

  template <int D0> tiled_extent<D0> tile() const;
  template <int D0, int D1> tiled_extent<D0,D1> tile() const;
  template <int D0, int D1, int D2> tiled_extent<D0,D1,D2> tile() const;

  friend inline bool operator==(const extent<N>& lhs, const extent<N>& rhs)
    restrict(amp,cpu);

  friend inline bool operator!=(const extent<N>& lhs, const extent<N>& rhs)
    restrict(amp,cpu);

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

private:
  int m_internal[N]; // Store the data
};

template <int D0, int D1/*=0*/, int D2/*=0*/>
class tiled_extent : public extent<3>
{
public:
  static const int rank = 3;
  tiled_extent() restrict(amp,cpu);
  tiled_extent(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent(const extent<3>& extent) restrict(amp,cpu);
  tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent pad() const restrict(amp,cpu);
  tiled_extent truncate() const restrict(amp,cpu);
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
  tiled_extent(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent(const extent<2>& ext) restrict(amp,cpu):extent(ext) {}
  tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent pad() const restrict(amp,cpu);
  tiled_extent truncate() const restrict(amp,cpu);
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
  tiled_extent pad() const restrict(amp,cpu);
  tiled_extent truncate() const restrict(amp,cpu);
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

  explicit array(int e0, int e1);

  explicit array(int e0, int e1, int e2);

  array(int e0, accelerator_view av);

  array(int e0, int e1, accelerator_view av);

  array(int e0, int e1, int e2, accelerator_view av);

  array(const extent<N>& extent, accelerator_view av, accelerator_view associated_av); //staging

  array(int e0, accelerator_view av, accelerator_view associated_av);

  array(int e0, int e1, accelerator_view av, accelerator_view associated_av); //staging

  array(int e0, int e1, int e2, accelerator_view av, accelerator_view associated_av); //staging

  template <typename InputIterator>
  array(const extent<N>& extent, InputIterator srcBegin);

  template <typename InputIterator>
  array(const extent<N>& extent, InputIterator srcBegin, InputIterator srcEnd);
#ifndef __GPU__
  // CAVEAT: ACCELERATOR
  template <typename InputIterator>
  array(int e0, InputIterator srcBegin) : m_extent(e0),
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
  array(int e0, int e1, InputIterator srcBegin);

  template <typename InputIterator>
  array(int e0, int e1, InputIterator srcBegin, InputIterator srcEnd);

  template <typename InputIterator>
  array(int e0, int e1, int e2, InputIterator srcBegin);

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
  array(const extent<N>& extent, InputIterator srcBegin, accelerator_view av);

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
  array(int e0, int e1, int e2, InputIterator srcBegin, InputIterator srcEnd, accelerator_view av);

  explicit array(const array_view<const T,N>& src);

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
#ifndef __GPU__
  extent<N> get_extent() const {
    return m_extent;
  }
#endif

  // __declspec(property(get)) accelerator_view accelerator_view;
  accelerator_view get_accelerator_view() const;

  // __declspec(property(get)) accelerator_view associated_accelerator_view;
  accelerator_view get_associated_accelerator_view() const;

  __global T& operator[](const index<N>& idx) restrict(amp,cpu) {
    if (rank == 1)
      return reinterpret_cast<__global T*>(m_device.get())[idx[0]];
    else if (rank == 2)
      return reinterpret_cast<__global T*>(m_device.get())
	[idx[0] * e1_ + idx[1]];
  }
  __global const T& operator[](const index<N>& idx) const restrict(amp,cpu) {
    if (rank == 1)
      return reinterpret_cast<__global T*>(m_device.get())[idx[0]];
    else if (rank == 2)
      return reinterpret_cast<__global T*>(m_device.get())
	[idx[0] * e1_ + idx[1]];
  }

  auto operator[](int i) restrict(amp,cpu) -> decltype(array_projection_helper<T, N>::project((array<T,N> *)NULL, i));

  auto operator[](int i) const restrict(amp,cpu) -> decltype(array_projection_helper<T, N>::project((const array<T, N>* )NULL, i));

  const T& operator()(const index<N>& idx) const restrict(amp,cpu);

  T& operator()(const index<N>& idx) restrict(amp,cpu);

  __global T& operator()(int i0, int i1) restrict(amp,cpu) {
    return (*this)[index<2>(i0, i1)];
  }
  __global const T& operator()(int i0, int i1) const restrict(amp,cpu) {
    return (*this)[index<2>(i0, i1)];
  }

  T& operator()(int i0, int i1, int i2) restrict(amp,cpu);

  const T& operator()(int i0, int i1, int i2) const restrict(amp,cpu);

  auto operator()(int i) restrict(amp,cpu) -> decltype(array_projection_helper<T, N>::project((array<T,N> *)NULL, i));

  auto operator()(int i) const restrict(amp,cpu) -> decltype(array_projection_helper<T, N>::project((const array<T, N>* )NULL, i));

  array_view<T,N> section(const index<N>& idx, const extent<N>& ext) restrict(amp,cpu);

  array_view<const T,N> section(const index<N>& idx, const extent<N>& ext) const restrict(amp,cpu);

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
  array_view<T,K> view_as(const extent<K>& viewExtent) restrict(amp,cpu);

  template <int K>
  array_view<const T,K> view_as(const extent<K>& viewExtent) const restrict(amp,cpu);

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

  const gmac_buffer_t& internal() const { return m_device; }

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const;

  __attribute__((annotate("deserialize"))) 
  array(__global T *p, cl_int e) restrict(amp);
  // End CLAMP
private:
#ifndef __GPU__
  // Data members that do not show up at GPU side
  __attribute__((cpu)) int dummy_; //Don't define deserialization for this class
  Concurrency::extent<N> m_extent;
  accelerator_view accelerator_view_;
#endif
  gmac_buffer_t m_device;
  cl_int e1_;
};

template <typename T>
class array<T, 1> {
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

  explicit array(int e0): array(extent<1>(e0)) {}

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
    array(const extent<1>& extent, InputIterator srcBegin,
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
    array(const extent<1>& extent, InputIterator srcBegin,
        accelerator_view av, accelerator_view associated_av) { // staging
      assert(0 && "Staging array is not supported");
    }

  template <typename InputIterator>
    array(const extent<1>& extent, InputIterator srcBegin,
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
  array(const Concurrency::extent<1>& extent, InputIterator srcBegin,
    accelerator_view av): e0_(extent[0]), m_device(nullptr)
#ifdef __GPU__
    { assert(0 && "Unrechable"); }
#else
    , accelerator_view_(av) {
    if (e0_) {
      m_device.reset(GMACAllocator<T>().allocate(e0_),
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
    array(const extent<1>& extent, InputIterator srcBegin,
        InputIterator srcEnd, accelerator_view av): e0_(extent[0]),
#ifdef __GPU__
    m_device(nullptr)  { assert(0 && "Unrechable"); }
#else
    accelerator_view_(av) {
    if (e0_) {
      m_device.reset(GMACAllocator<T>().allocate(e0_),
        GMACDeleter<T>());
      InputIterator srcCopyEnd = srcBegin;
      std::advance(srcCopyEnd,
        std::min(std::distance(srcBegin, srcEnd),
          decltype(std::distance(srcBegin, srcEnd))(e0_)));
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
  extent<1> get_extent() const {
    return Concurrency::extent<1>(e0_);
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

  __global T& operator()(const index<1>& idx) restrict(amp,cpu) {
    return this->operator[](idx);
  }

  const __global T& operator()(const index<1>& idx) const restrict(amp,cpu) {
    return this->operator[](idx);
  }

  array_view<T,1> section(const index<1>& idx, const extent<1>& ext) restrict(amp,cpu);

  array_view<const T,1> section(const index<1>& idx, const extent<1>& ext) const restrict(amp,cpu);

  array_view<T,1> section(const index<1>& idx) restrict(amp,cpu);

  array_view<const T,1> section(const index<1>& idx) const restrict(amp,cpu);

  template <typename ElementType>
    array_view<ElementType, 1> reinterpret_as() restrict(amp,cpu);

  template <typename ElementType>
    array_view<const ElementType, 1> reinterpret_as() const restrict(amp,cpu);

  template <int K>
    array_view<T,K> view_as(const extent<K>& viewExtent) restrict(amp,cpu) {
      array_view<T, 1> av(*this);
      return av.view_as(viewExtent);
    }

  template <int K>
    array_view<const T,K> view_as(const extent<K>& viewExtent) const restrict(amp,cpu);

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

  const gmac_buffer_t& internal() const { return m_device; }

  // CLAMP: The serialization interface
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const;

  __attribute__((annotate("deserialize"))) 
  array(__global T *p, cl_int e) restrict(amp);
  // End CLAMP
private:
#ifndef __GPU__
  // Data members that do not show up at GPU side
  __attribute__((cpu)) int dummy_; //Don't define deserialization for this class
  accelerator_view accelerator_view_;
#endif
  gmac_buffer_t m_device;
  cl_int e0_;
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
index<N> operator+(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu);
template <int N>
index<N> operator+(const index<N>& lhs, int rhs) restrict(amp,cpu);
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

