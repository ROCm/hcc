#pragma once

#include <kalmar_defines.h>
#include <kalmar_exception.h>
#include <kalmar_index.h>
#include <kalmar_runtime.h>
#include <kalmar_serialize.h>
#include <kalmar_launch.h>

#include <hsa_atomic.h>

namespace hc {

using namespace Kalmar::enums;

// forward declaration
class accelerator;
class accelerator_view;
class completion_future;
template <int N> class extent;
template <int N> class tiled_extent;
class ts_allocator;

// type alias
// hc::index is just an alias of Kalmar::index
template <int N>
using index = Kalmar::index<N>;

using runtime_exception = Kalmar::runtime_exception;
using invalid_compute_domain = Kalmar::invalid_compute_domain;
using accelerator_view_removed = Kalmar::accelerator_view_removed;


class accelerator_view {
    accelerator_view(std::shared_ptr<Kalmar::KalmarQueue> pQueue)
        : pQueue(pQueue) {}
public:
  accelerator_view(const accelerator_view& other) :
      pQueue(other.pQueue) {}
  accelerator_view& operator=(const accelerator_view& other) {
      pQueue = other.pQueue;
      return *this;
  }

  accelerator get_accelerator() const;
  bool get_is_debug() const { return 0; } 
  unsigned int get_version() const { return 0; } 
  queuing_mode get_queuing_mode() const { return pQueue->get_mode(); }
  bool get_is_auto_selection() { return false; }

  void flush() { pQueue->flush(); }
  void wait() { pQueue->wait(); }

  bool operator==(const accelerator_view& other) const {
      return pQueue == other.pQueue;
  }
  bool operator!=(const accelerator_view& other) const { return !(*this == other); }

  // returns the size of tile static area
  size_t get_max_tile_static_size() {
    return pQueue.get()->getDev()->GetMaxTileStaticSize();
  }

private:
  std::shared_ptr<Kalmar::KalmarQueue> pQueue;
  friend class accelerator;

  template<typename Kernel> friend
      void* Kalmar::mcw_cxxamp_get_kernel(const std::shared_ptr<Kalmar::KalmarQueue>&, const Kernel&);
  template<typename Kernel, int dim_ext> friend
      void Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory(const std::shared_ptr<Kalmar::KalmarQueue>&, size_t *, size_t *, const Kernel&, void*, size_t);
  template<typename Kernel, int dim_ext> friend
      std::shared_future<void>* Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async(const std::shared_ptr<Kalmar::KalmarQueue>&, size_t *, size_t *, const Kernel&, void*, size_t);
  template<typename Kernel, int dim_ext> friend
      void Kalmar::mcw_cxxamp_launch_kernel(const std::shared_ptr<Kalmar::KalmarQueue>&, size_t *, size_t *, const Kernel&);
  template<typename Kernel, int dim_ext> friend
      std::shared_future<void>* Kalmar::mcw_cxxamp_launch_kernel_async(const std::shared_ptr<Kalmar::KalmarQueue>&, size_t *, size_t *, const Kernel&);

  // FIXME: enable CPU execution path for HC
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  template <typename Kernel, int N> friend
      void Kalmar::launch_cpu_task(const std::shared_ptr<Kalmar::KalmarQueue>&, Kernel const&, extent<N> const&);
#endif

  // non-tiled parallel_for_each with dynamic group segment
  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const extent<1>&, ts_allocator&, const Kernel&);
  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const extent<2>&, ts_allocator&, const Kernel&);
  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const extent<3>&, ts_allocator&, const Kernel&);

  // non-tiled parallel_for_each with dynamic group segment
  template <typename Kernel> friend
      completion_future parallel_for_each(const extent<1>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
      completion_future parallel_for_each(const extent<2>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
      completion_future parallel_for_each(const extent<3>&, ts_allocator&, const Kernel&);

  // tiled parallel_for_each with dynamic group segment
  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, ts_allocator&, const Kernel&);
  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, ts_allocator&, const Kernel&);
  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, ts_allocator&, const Kernel&);

  // tiled parallel_for_each with dynamic group segment
  template <typename Kernel> friend
      completion_future parallel_for_each(const tiled_extent<1>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
      completion_future parallel_for_each(const tiled_extent<2>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
      completion_future parallel_for_each(const tiled_extent<3>&, ts_allocator&, const Kernel&);

  // non-tiled parallel_for_each
  // generic version
  template <int N, typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const extent<N>&, const Kernel&);

  // 1D specialization
  template <typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const extent<1>&, const Kernel&);

  // 2D specialization
  template <typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const extent<2>&, const Kernel&);

  // 3D specialization
  template <typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const extent<3>&, const Kernel&);

  // tiled parallel_for_each, 3D version
  template <typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, const Kernel&);

  // tiled parallel_for_each, 2D version
  template <typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, const Kernel&);

  // tiled parallel_for_each, 1D version
  template <typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, const Kernel&);


#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
public:
#endif
  __attribute__((annotate("user_deserialize")))
      accelerator_view() restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
          throw runtime_exception("errorMsg_throw", 0);
#endif
      }
};

class accelerator
{
  accelerator(Kalmar::KalmarDevice* pDev) : pDev(pDev) {}
public:
  static const wchar_t default_accelerator[];
  static const wchar_t cpu_accelerator[];

  accelerator() : accelerator(default_accelerator) {}
  explicit accelerator(const std::wstring& path)
      : pDev(Kalmar::getContext()->getDevice(path)) {}
  accelerator(const accelerator& other) : pDev(other.pDev) {}

  static std::vector<accelerator> get_all() {
      auto Devices = Kalmar::getContext()->getDevices();
      std::vector<accelerator> ret(Devices.size());
      for (int i = 0; i < ret.size(); ++i)
          ret[i] = Devices[i];
      return std::move(ret);
  }
  static bool set_default(const std::wstring& path) {
      return Kalmar::getContext()->set_default(path);
  }
  static accelerator_view get_auto_selection_view() { return Kalmar::getContext()->auto_select(); }

  accelerator& operator=(const accelerator& other) {
      pDev = other.pDev;
      return *this;
  }

  std::wstring get_device_path() const { return pDev->get_path(); }
  unsigned int get_version() const { return 0; }
  std::wstring get_description() const { return pDev->get_description(); }
  bool get_is_debug() const { return false; }
  bool get_is_emulated() const { return pDev->is_emulated(); }
  bool get_has_display() const { return false; }
  bool get_supports_double_precision() const { return pDev->is_double(); }
  bool get_supports_limited_double_precision() const { return pDev->is_lim_double(); }
  size_t get_dedicated_memory() const { return pDev->get_mem(); }
  accelerator_view get_default_view() const { return pDev->get_default_queue(); }
  access_type get_default_cpu_access_type() const { return pDev->get_access(); }
  bool get_supports_cpu_shared_memory() const { return pDev->is_unified(); }

  bool set_default_cpu_access_type(access_type type) {
      pDev->set_access(type);
      return true;
  }
  accelerator_view create_view(queuing_mode mode = queuing_mode_automatic) {
      auto pQueue = pDev->createQueue();
      pQueue->set_mode(mode);
      return pQueue;
  }

  bool operator==(const accelerator& other) const { return pDev == other.pDev; }
  bool operator!=(const accelerator& other) const { return !(*this == other); }

  // returns the size of tile static area
  size_t get_max_tile_static_size() {
    return get_default_view().get_max_tile_static_size();
  }

private:
  friend class accelerator_view;
  Kalmar::KalmarDevice* pDev;
};

inline accelerator accelerator_view::get_accelerator() const { return pQueue->getDev(); }

// FIXME: this will cause troubles later in separated compilation
const wchar_t accelerator::cpu_accelerator[] = L"cpu";
const wchar_t accelerator::default_accelerator[] = L"default";

// FIXME: needs to think about what its new semantic should be
// FIXME: get create_marker() implemented
class completion_future {
public:

    completion_future() {};

    completion_future(const completion_future& _Other)
        : __amp_future(_Other.__amp_future), __thread_then(_Other.__thread_then) {}

    completion_future(completion_future&& _Other)
        : __amp_future(std::move(_Other.__amp_future)), __thread_then(_Other.__thread_then) {}

    ~completion_future() {
      if (__thread_then != nullptr) {
        __thread_then->join();
      }
      delete __thread_then;
      __thread_then = nullptr;
    }

    completion_future& operator=(const completion_future& _Other) {
        if (this != &_Other) {
           __amp_future = _Other.__amp_future;
           __thread_then = _Other.__thread_then;
        }
        return (*this);
    }

    completion_future& operator=(completion_future&& _Other) {
        if (this != &_Other) {
            __amp_future = std::move(_Other.__amp_future);
            __thread_then = _Other.__thread_then;
        }
        return (*this);
    }

    void get() const {
        __amp_future.get();
    }

    bool valid() const {
        return __amp_future.valid();
    }
    void wait() const {
        if(this->valid())
          __amp_future.wait();
    }

    template <class _Rep, class _Period>
    std::future_status wait_for(const std::chrono::duration<_Rep, _Period>& _Rel_time) const {
        return __amp_future.wait_for(_Rel_time);
    }

    template <class _Clock, class _Duration>
    std::future_status wait_until(const std::chrono::time_point<_Clock, _Duration>& _Abs_time) const {
        return __amp_future.wait_until(_Abs_time);
    }

    operator std::shared_future<void>() const {
        return __amp_future;
    }

    // notice we removed const from the signature here
    template<typename functor>
    void then(const functor & func) {
#if __KALMAR_ACCELERATOR__ != 1
      // could only assign once
      if (__thread_then == nullptr) {
        // spawn a new thread to wait on the future and then execute the callback functor
        __thread_then = new std::thread([&]() restrict(cpu) {
          this->wait();
          if(this->valid())
            func();
        });
      }
#endif
    }
private:
    std::shared_future<void> __amp_future;
    std::thread* __thread_then = nullptr;

    // __future is dynamically allocated in C++AMP runtime implementation
    // after we copy its content in __amp_future, we need to delete it
    completion_future(std::shared_future<void>* __future)
        : __amp_future(*__future) {
      delete __future;
    }

    completion_future(const std::shared_future<void> &__future)
        : __amp_future(__future) {}

    // non-tiled parallel_for_each with dynamic group segment
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<1>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<2>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<3>&, ts_allocator&, const Kernel&);
  
    // non-tiled parallel_for_each with dynamic group segment
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<1>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<2>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<3>&, ts_allocator&, const Kernel&);
  
    // tiled parallel_for_each with dynamic group segment
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, ts_allocator&, const Kernel&);
  
    // tiled parallel_for_each with dynamic group segment
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<1>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<2>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<3>&, ts_allocator&, const Kernel&);

    // non-tiled parallel_for_each
    // generic version
    template <int N, typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<N>&, const Kernel&);

    // 1D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<1>&, const Kernel&);

    // 2D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<2>&, const Kernel&);

    // 3D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<3>&, const Kernel&);

    // tiled parallel_for_each, 3D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, const Kernel&);

    // tiled parallel_for_each, 2D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, const Kernel&);

    // tiled parallel_for_each, 1D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, const Kernel&);
};


template <int N>
class extent {
public:
    static const int rank = N;
    typedef int value_type;

    extent() restrict(amp,cpu) : base_() {
      static_assert(N > 0, "Dimensionality must be positive");
    };
    extent(const extent& other) restrict(amp,cpu)
        : base_(other.base_) {}
    template <typename ..._Tp>
        explicit extent(_Tp ... __t) restrict(amp,cpu)
        : base_(__t...) {
      static_assert(sizeof...(__t) <= 3, "Can only supply at most 3 individual coordinates in the constructor");
      static_assert(sizeof...(__t) == N, "rank should be consistency");
    }
    explicit extent(int component) restrict(amp,cpu)
        : base_(component) {}
    explicit extent(int components[]) restrict(amp,cpu)
        : base_(components) {}
    explicit extent(const int components[]) restrict(amp,cpu)
        : base_(components) {}

    extent& operator=(const extent& other) restrict(amp,cpu) {
        base_.operator=(other.base_);
        return *this;
    }

    int operator[] (unsigned int c) const restrict(amp,cpu) {
        return base_[c];
    }
    int& operator[] (unsigned int c) restrict(amp,cpu) {
        return base_[c];
    }

    bool operator==(const extent& other) const restrict(amp,cpu) {
        return Kalmar::index_helper<N, extent<N> >::equal(*this, other);
    }
    bool operator!=(const extent& other) const restrict(amp,cpu) {
        return !(*this == other);
    }

    unsigned int size() const restrict(amp,cpu) {
        return Kalmar::index_helper<N, extent<N>>::count_size(*this);
    }
    bool contains(const index<N>& idx) const restrict(amp,cpu) {
        return Kalmar::amp_helper<N, index<N>, extent<N>>::contains(idx, *this);
    }
    tiled_extent<1> tile(int t0) const;
    tiled_extent<2> tile(int t0, int t1) const;
    tiled_extent<3> tile(int t0, int t1, int t2) const;

    extent operator+(const index<N>& idx) restrict(amp,cpu) {
        extent __r = *this;
        __r += idx;
        return __r;
    }
    extent operator-(const index<N>& idx) restrict(amp,cpu) {
        extent __r = *this;
        __r -= idx;
        return __r;
    }
    extent& operator+=(const index<N>& idx) restrict(amp,cpu) {
        base_.operator+=(idx.base_);
        return *this;
    }
    extent& operator-=(const index<N>& idx) restrict(amp,cpu) {
        base_.operator-=(idx.base_);
        return *this;
    }
    extent& operator+=(const extent& __r) restrict(amp,cpu) {
        base_.operator+=(__r.base_);
        return *this;
    }
    extent& operator-=(const extent& __r) restrict(amp,cpu) {
        base_.operator-=(__r.base_);
        return *this;
    }
    extent& operator*=(const extent& __r) restrict(amp,cpu) {
        base_.operator*=(__r.base_);
        return *this;
    }
    extent& operator/=(const extent& __r) restrict(amp,cpu) {
        base_.operator/=(__r.base_);
        return *this;
    }
    extent& operator%=(const extent& __r) restrict(amp,cpu) {
        base_.operator%=(__r.base_);
        return *this;
    }
    extent& operator+=(int __r) restrict(amp,cpu) {
        base_.operator+=(__r);
        return *this;
    }
    extent& operator-=(int __r) restrict(amp,cpu) {
        base_.operator-=(__r);
        return *this;
    }
    extent& operator*=(int __r) restrict(amp,cpu) {
        base_.operator*=(__r);
        return *this;
    }
    extent& operator/=(int __r) restrict(amp,cpu) {
        base_.operator/=(__r);
        return *this;
    }
    extent& operator%=(int __r) restrict(amp,cpu) {
        base_.operator%=(__r);
        return *this;
    }
    extent& operator++() restrict(amp,cpu) {
        base_.operator+=(1);
        return *this;
    }
    extent operator++(int) restrict(amp,cpu) {
        extent ret = *this;
        base_.operator+=(1);
        return ret;
    }
    extent& operator--() restrict(amp,cpu) {
        base_.operator-=(1);
        return *this;
    }
    extent operator--(int) restrict(amp,cpu) {
        extent ret = *this;
        base_.operator-=(1);
        return ret;
    }
private:
    typedef Kalmar::index_impl<typename Kalmar::__make_indices<N>::type> base;
    base base_;
    template <int K, typename Q> friend struct Kalmar::index_helper;
    template <int K, typename Q1, typename Q2> friend struct Kalmar::amp_helper;
};

// tile extent supporting dynamic tile size
template <int N>
class tiled_extent : public extent<N> {
public:
  static const int rank = N;
  int tile_dim[N];
  tiled_extent() restrict(amp,cpu) : extent<N>(), tile_dim{0} {}
  tiled_extent(const tiled_extent& other) restrict(amp,cpu) : extent<N>(other) {
    for (int i = 0; i < N; ++i) {
      tile_dim[i] = other.tile_dim[i];
    }
  }
};

// tile extent supporting dynamic tile size
// 1D specialization
template <>
class tiled_extent<1> : public extent<1> {
public:
  static const int rank = 1;
  int tile_dim[1];
  tiled_extent() restrict(amp,cpu) : extent(0), tile_dim{0} {}
  tiled_extent(int e0, int t0) restrict(amp,cpu) : extent(e0), tile_dim{t0} {}
  tiled_extent(const tiled_extent<1>& other) restrict(amp,cpu) : extent(other[0]), tile_dim{other.tile_dim[0]} {}
  tiled_extent(const extent<1>& ext, int t0) restrict(amp,cpu) : extent(ext), tile_dim{t0} {} 
};

// tile extent supporting dynamic tile size
// 2D specialization
template <>
class tiled_extent<2> : public extent<2> {
public:
  static const int rank = 2;
  int tile_dim[2];
  tiled_extent() restrict(amp,cpu) : extent(0, 0), tile_dim{0, 0} {}
  tiled_extent(int e0, int e1, int t0, int t1) restrict(amp,cpu) : extent(e0, e1), tile_dim{t0, t1} {}
  tiled_extent(const tiled_extent<2>& other) restrict(amp,cpu) : extent(other[0], other[1]), tile_dim{other.tile_dim[0], other.tile_dim[1]} {}
  tiled_extent(const extent<2>& ext, int t0, int t1) restrict(amp,cpu) : extent(ext), tile_dim{t0, t1} {}
};

// tile extent supporting dynamic tile size
// 3D specialization
template <>
class tiled_extent<3> : public extent<3> {
public:
  static const int rank = 3;
  int tile_dim[3];
  tiled_extent() restrict(amp,cpu) : extent(0, 0, 0), tile_dim{0, 0, 0} {}
  tiled_extent(int e0, int e1, int e2, int t0, int t1, int t2) restrict(amp,cpu) : extent(e0, e1, e2), tile_dim{t0, t1, t2} {}
  tiled_extent(const tiled_extent<3>& other) restrict(amp,cpu) : extent(other[0], other[1], other[2]), tile_dim{other.tile_dim[0], other.tile_dim[1], other.tile_dim[2]} {}
  tiled_extent(const extent<3>& ext, int t0, int t1, int t2) restrict(amp,cpu) : extent(ext), tile_dim{t0, t1, t2} {}
};

template <int N>
inline
tiled_extent<1> extent<N>::tile(int t0) const restrict(amp,cpu) {
  static_assert(N == 1, "One-dimensional tile() method only available on extent<1>");
  return tiled_extent<1>(*this, t0);
}

template <int N>
inline
tiled_extent<2> extent<N>::tile(int t0, int t1) const restrict(amp,cpu) {
  static_assert(N == 2, "Two-dimensional tile() method only available on extent<2>");
  return tiled_extent<2>(*this, t0, t1);
}

template <int N>
inline
tiled_extent<3> extent<N>::tile(int t0, int t1, int t2) const restrict(amp,cpu) {
  static_assert(N == 3, "Three-dimensional tile() method only available on extent<3>");
  return tiled_extent<3>(*this, t0, t1, t2);
}


/// getLDS : C interface of HSA builtin function to fetch an address within group segment
extern "C" __attribute__((address_space(3))) void* getLDS(unsigned int offset) restrict(amp);

class ts_allocator {
private:
  unsigned int static_group_segment_size;
  unsigned int dynamic_group_segment_size;
  int cursor;

  void setStaticGroupSegmentSize(unsigned int size) restrict(cpu) {
    static_group_segment_size = size;
  } 

  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const extent<1>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const extent<2>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const extent<3>&, ts_allocator&, const Kernel&);

  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, ts_allocator&, const Kernel&);

public:
  ts_allocator() :
    static_group_segment_size(0), 
    dynamic_group_segment_size(0),
    cursor(0) {}

  ~ts_allocator() {}

  unsigned int getStaticGroupSegmentSize() restrict(amp,cpu) {
    return static_group_segment_size;
  }

  void setDynamicGroupSegmentSize(unsigned int size) restrict(cpu) {
    dynamic_group_segment_size = size;
  }

  unsigned int getDynamicGroupSegmentSize() restrict(amp,cpu) {
    return dynamic_group_segment_size;
  }

  void reset() restrict(amp,cpu) {
    cursor = 0;
  }

  // Allocate the requested size in tile static memory and return its pointer
  // returns NULL if the requested size can't be allocated
  // It requires all threads in a tile to hit the same ts_alloc call site at the
  // same time.
  // Only one instance of the tile static memory will be allocated per call site
  // and all threads within a tile will get the same tile static memory address.
  __attribute__((address_space(3))) void* alloc(unsigned int size) restrict(amp) {
    int offset = cursor;

    // only the first workitem in the workgroup moves the cursor
    if (amp_get_local_id(0) == 0 && amp_get_local_id(1) == 0 && amp_get_local_id(2) == 0) {
      cursor += size;
    }

    // fetch the beginning address of dynamic group segment
    __attribute__((address_space(3))) unsigned char* lds = (__attribute__((address_space(3))) unsigned char*) getLDS(static_group_segment_size);

    // return the address
    return lds + offset;
  }   
};  

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
template <typename Ker, typename Ti>
void bar_wrapper(Ker *f, Ti *t)
{
    (*f)(*t);
}
struct barrier_t {
    std::unique_ptr<ucontext_t[]> ctx;
    int idx;
    barrier_t (int a) :
        ctx(new ucontext_t[a + 1]) {}
    template <typename Ti, typename Ker>
    void setctx(int x, char *stack, Ker& f, Ti* tidx, int S) {
        getcontext(&ctx[x]);
        ctx[x].uc_stack.ss_sp = stack;
        ctx[x].uc_stack.ss_size = S;
        ctx[x].uc_link = &ctx[x - 1];
        makecontext(&ctx[x], (void (*)(void))bar_wrapper<Ker, Ti>, 2, &f, tidx);
    }
    void swap(int a, int b) {
        swapcontext(&ctx[a], &ctx[b]);
    }
    void wait() {
        --idx;
        swapcontext(&ctx[idx + 1], &ctx[idx]);
    }
};
#endif

#ifndef CLK_LOCAL_MEM_FENCE
#define CLK_LOCAL_MEM_FENCE (1)
#endif

#ifndef CLK_GLOBAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE (2)
#endif

class tile_barrier {
 public:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  using pb_t = std::shared_ptr<barrier_t>;
  tile_barrier(pb_t pb) : pbar(pb) {}
  tile_barrier(const tile_barrier& other) restrict(amp,cpu) : pbar(other.pbar) {}
#else
  tile_barrier(const tile_barrier& other) restrict(amp,cpu) {}
#endif
  void wait() const restrict(amp) {
#if __KALMAR_ACCELERATOR__ == 1
    wait_with_all_memory_fence();
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      pbar->wait();
#endif
  }
  void wait_with_all_memory_fence() const restrict(amp) {
#if __KALMAR_ACCELERATOR__ == 1
    amp_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      pbar->wait();
#endif
  }
  void wait_with_global_memory_fence() const restrict(amp) {
#if __KALMAR_ACCELERATOR__ == 1
    amp_barrier(CLK_GLOBAL_MEM_FENCE);
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      pbar->wait();
#endif
  }
  void wait_with_tile_static_memory_fence() const restrict(amp) {
#if __KALMAR_ACCELERATOR__ == 1
    amp_barrier(CLK_LOCAL_MEM_FENCE);
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      pbar->wait();
#endif
  }
 private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  tile_barrier() restrict(amp,cpu) = default;
  pb_t pbar;
#else
  tile_barrier() restrict(amp) {}
#endif
  template <int N> friend
    class tiled_index;

  friend class tiled_index_1D;
  friend class tiled_index_2D;
  friend class tiled_index_3D;
};

template <int N=3>
class tiled_index {
public:
  const index<3> global;
  const index<3> local;
  const index<3> tile;
  const index<3> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<3>& g) restrict(amp,cpu) : global(g) {}
  tiled_index(const tiled_index& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<3>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index(int a0, int a1, int a2, int b0, int b1, int b2, int c0, int c1, int c2, tile_barrier& pb) restrict(amp,cpu) :
    global(a2, a1, a0), local(b2, b1, b0), tile(c2, c1, c0), tile_origin(a2 - b2, a1 - b1, a0 - b0), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index() restrict(amp) :
    global(index<3>(amp_get_global_id(2), amp_get_global_id(1), amp_get_global_id(0))),
    local(index<3>(amp_get_local_id(2), amp_get_local_id(1), amp_get_local_id(0))),
    tile(index<3>(amp_get_group_id(2), amp_get_group_id(1), amp_get_group_id(0))),
    tile_origin(index<3>(amp_get_global_id(2) - amp_get_local_id(2),
                         amp_get_global_id(1) - amp_get_local_id(1),
                         amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<N>&, ts_allocator&, const Kernel&);

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<N>&, const Kernel&);
};

template<>
class tiled_index<1> {
public:
  const index<1> global;
  const index<1> local;
  const index<1> tile;
  const index<1> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<1>& g) restrict(amp,cpu) : global(g) {}
  tiled_index(const tiled_index& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<1>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index(int a, int b, int c, tile_barrier& pb) restrict(amp,cpu) :
    global(a), local(b), tile(c), tile_origin(a - b), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index() restrict(amp) :
    global(index<1>(amp_get_global_id(0))),
    local(index<1>(amp_get_local_id(0))),
    tile(index<1>(amp_get_group_id(0))),
    tile_origin(index<1>(amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, ts_allocator&, const Kernel&);

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, const Kernel&);
};

template<>
class tiled_index<2> {
public:
  const index<2> global;
  const index<2> local;
  const index<2> tile;
  const index<2> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<2>& g) restrict(amp,cpu) : global(g) {}
  tiled_index(const tiled_index& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<2>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index(int a0, int a1, int b0, int b1, int c0, int c1, tile_barrier& pb) restrict(amp,cpu) :
    global(a1, a0), local(b1, b0), tile(c1, c0), tile_origin(a1 - b1, a0 - b0), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index() restrict(amp) :
    global(index<2>(amp_get_global_id(1), amp_get_global_id(0))),
    local(index<2>(amp_get_local_id(1), amp_get_local_id(0))),
    tile(index<2>(amp_get_group_id(1), amp_get_group_id(0))),
    tile_origin(index<2>(amp_get_global_id(1) - amp_get_local_id(1),
                         amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, ts_allocator&, const Kernel&);

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, const Kernel&);
};

// async pfe
template <int N, typename Kernel>
completion_future parallel_for_each(const accelerator_view&, const extent<N>&, const Kernel&);

template <typename Kernel>
completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, const Kernel&);

template <typename Kernel>
completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, const Kernel&);

template <typename Kernel>
completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, const Kernel&);

template <int N, typename Kernel>
completion_future parallel_for_each(const extent<N>& compute_domain, const Kernel& f) {
    return parallel_for_each(accelerator::get_auto_selection_view(), compute_domain, f);
}

template <typename Kernel>
completion_future parallel_for_each(const tiled_extent<3>& compute_domain, const Kernel& f) {
    return parallel_for_each(accelerator::get_auto_selection_view(), compute_domain, f);
}

template <typename Kernel>
completion_future parallel_for_each(const tiled_extent<2>& compute_domain, const Kernel& f) {
    return parallel_for_each(accelerator::get_auto_selection_view(), compute_domain, f);
}

template <typename Kernel>
completion_future parallel_for_each(const tiled_extent<1>& compute_domain, const Kernel& f) {
    return parallel_for_each(accelerator::get_auto_selection_view(), compute_domain, f);
}

template <int N, typename Kernel, typename _Tp>
struct pfe_helper
{
    static inline void call(Kernel& k, _Tp& idx) restrict(amp,cpu) {
        int i;
        for (i = 0; i < k.ext[N - 1]; ++i) {
            idx[N - 1] = i;
            pfe_helper<N - 1, Kernel, _Tp>::call(k, idx);
        }
    }
};
template <typename Kernel, typename _Tp>
struct pfe_helper<0, Kernel, _Tp>
{
    static inline void call(Kernel& k, _Tp& idx) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ == 1
        k.k(idx);
#endif
    }
};

template <int N, typename Kernel>
class pfe_wrapper
{
public:
    explicit pfe_wrapper(const extent<N>& other, const Kernel& f) restrict(amp,cpu)
        : ext(other), k(f) {}
    void operator() (index<N> idx) restrict(amp,cpu) {
        pfe_helper<N - 3, pfe_wrapper<N, Kernel>, index<N>>::call(*this, idx);
    }
private:
    const extent<N> ext;
    const Kernel k;
    template <int K, typename Ker, typename _Tp>
        friend struct pfe_helper;
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma clang diagnostic ignored "-Wunused-variable"
//ND parallel_for_each, nontiled
template <int N, typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av,
    const extent<N>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
    size_t compute_domain_size = 1;
    for(int i = 0 ; i < N ; i++)
    {
      if(compute_domain[i]<=0)
        throw invalid_compute_domain("Extent is less or equal than 0.");
      if (static_cast<size_t>(compute_domain[i]) > 4294967295L)
        throw invalid_compute_domain("Extent size too large.");
      compute_domain_size *= static_cast<size_t>(compute_domain[i]);
      if (compute_domain_size > 4294967295L)
        throw invalid_compute_domain("Extent size too large.");
    }
    size_t ext[3] = {static_cast<size_t>(compute_domain[N - 1]),
        static_cast<size_t>(compute_domain[N - 2]),
        static_cast<size_t>(compute_domain[N - 3])};
    if (av.get_accelerator().get_device_path() == L"cpu") {
      throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
    }
    const pfe_wrapper<N, Kernel> _pf(compute_domain, f);
    return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<pfe_wrapper<N, Kernel>, 3>(av.pQueue, ext, NULL, _pf));
#else
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  int* foo1 = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
    auto bar = &pfe_wrapper<N, Kernel>::operator();
    auto qq = &index<N>::__cxxamp_opencl_index;
    int* foo = reinterpret_cast<int*>(&pfe_wrapper<N, Kernel>::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//1D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const extent<1>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext = compute_domain[0];
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 1>(av.pQueue, &ext, NULL, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//2D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const extent<2>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = {static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 2>(av.pQueue, ext, NULL, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//3D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const extent<3>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[3] = {static_cast<size_t>(compute_domain[2]),
                   static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 3>(av.pQueue, ext, NULL, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//1D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const tiled_extent<1>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext = compute_domain[0];
  size_t tile = compute_domain.tile_dim[0];
  if (static_cast<size_t>(compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if(ext % tile != 0) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 1>(av.pQueue, &ext, &tile, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<1> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//2D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const tiled_extent<2>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = { static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[2] = { static_cast<size_t>(compute_domain.tile_dim[1]),
                     static_cast<size_t>(compute_domain.tile_dim[0]) };
  if (static_cast<size_t>(compute_domain.tile_dim[1] * compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 2>(av.pQueue, ext, tile, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<2> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//3D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const tiled_extent<3>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[3] = { static_cast<size_t>(compute_domain[2]),
                    static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[3] = { static_cast<size_t>(compute_domain.tile_dim[2]),
                     static_cast<size_t>(compute_domain.tile_dim[1]),
                     static_cast<size_t>(compute_domain.tile_dim[0]) };
  if (static_cast<size_t>(compute_domain.tile_dim[2] * compute_domain.tile_dim[1]* compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0) || (ext[2] % tile[2] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 3>(av.pQueue, ext, tile, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<3> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//1D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const extent<1>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
      return;
  }
#endif
  size_t ext = compute_domain[0];
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 1>(av.pQueue, &ext, NULL, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//2D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const extent<2>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
      return;
  }
#endif
  size_t ext[2] = {static_cast<size_t>(compute_domain[1]),
      static_cast<size_t>(compute_domain[0])};
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 2>(av.pQueue, ext, NULL, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//3D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const extent<3>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
      return;
  }
#endif
  size_t ext[3] = {static_cast<size_t>(compute_domain[2]),
      static_cast<size_t>(compute_domain[1]),
      static_cast<size_t>(compute_domain[0])};
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 3>(av.pQueue, ext, NULL, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//1D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const tiled_extent<1>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L) {
    throw invalid_compute_domain("Extent size too large.");
  }
  size_t ext = compute_domain[0];
  size_t tile = compute_domain.tile_dim[0];
  if (static_cast<size_t>(compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if (ext % tile != 0) {
    throw invalid_compute_domain("Extent can't be evenly divisible by tile size.");
  }
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
      return;
  }
#endif
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 1>(av.pQueue, &ext, &tile, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<1> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//2D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const tiled_extent<2>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = { static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[2] = { static_cast<size_t>(compute_domain.tile_dim[1]),
                     static_cast<size_t>(compute_domain.tile_dim[0]) };
  if (static_cast<size_t>(compute_domain.tile_dim[1] * compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
  } else
#endif
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 2>(av.pQueue, ext, tile, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<2> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//3D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const tiled_extent<3>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[3] = { static_cast<size_t>(compute_domain[2]),
                    static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[3] = { static_cast<size_t>(compute_domain.tile_dim[2]),
                     static_cast<size_t>(compute_domain.tile_dim[1]),
                     static_cast<size_t>(compute_domain.tile_dim[0]) };
  if (static_cast<size_t>(compute_domain.tile_dim[2] * compute_domain.tile_dim[1]* compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0) || (ext[2] % tile[2] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
  } else
#endif
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 3>(av.pQueue, ext, tile, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<3> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

template <typename Kernel>
completion_future parallel_for_each(const extent<1>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template <typename Kernel>
completion_future parallel_for_each(const extent<2>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template <typename Kernel>
completion_future parallel_for_each(const extent<3>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template <typename Kernel>
completion_future parallel_for_each(const tiled_extent<1>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template<typename Kernel>
completion_future parallel_for_each(const tiled_extent<2>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template<typename Kernel>
completion_future parallel_for_each(const tiled_extent<3>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

} // namespace hc
