#pragma once

// FIXME: remove C++AMP header dependency
#include <amp.h>

#include <hsa_atomic.h>

namespace hc {

// FIXME: remove Concurrency dependency
using namespace Concurrency;

// forward declaration
class accelerator;
class accelerator_view;
class tiled_extent_1D;
class tiled_extent_2D;
class tiled_extent_3D;
class tiled_index_1D;
class tiled_index_2D;
class tiled_index_3D;
class ts_allocator;

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
  completion_future create_marker();

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
      void* Kalmar::mcw_cxxamp_get_kernel(const accelerator_view&, const Kernel&);
  template<typename Kernel, int dim_ext> friend
      void Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory(const accelerator_view&, size_t *, size_t *, const Kernel&, void*, size_t);
  template<typename Kernel, int dim_ext> friend
      void Kalmar::mcw_cxxamp_launch_kernel(const accelerator_view&, size_t *, size_t *, const Kernel&);
  template<typename Kernel, int dim_ext> friend
      std::shared_future<void>* Kalmar::mcw_cxxamp_launch_kernel_async(const accelerator_view&, size_t *, size_t *, const Kernel&);
  template <typename Kernel, int N> friend
      void Kalmar::launch_cpu_task(const accelerator_view&, Kernel const&, extent<N> const&);

  template<typename Kernel> friend
      void parallel_for_each(const accelerator_view&, const tiled_extent_1D&, ts_allocator&, const Kernel&);
  template<typename Kernel> friend
      void parallel_for_each(const accelerator_view&, const tiled_extent_2D&, ts_allocator&, const Kernel&);
  template<typename Kernel> friend
      void parallel_for_each(const accelerator_view&, const tiled_extent_3D&, ts_allocator&, const Kernel&);

  // FIXME: remove C++AMP dependencies
  template <typename Q, int K> friend class array;
  template <typename Q, int K> friend class array_view;
  template <typename T, int N> friend class array_helper;

  // FIXME: C++AMP parallel_for_each interfaces
#if 0
  template <int N, typename Kernel>
      friend void parallel_for_each(extent<N> compute_domain, const Kernel& f);
  template <int D0, int D1, int D2, typename Kernel>
      friend void parallel_for_each(tiled_extent<D0,D1,D2> compute_domain, const Kernel& f);
  template <int D0, int D1, typename Kernel>
      friend void parallel_for_each(tiled_extent<D0,D1> compute_domain, const Kernel& f);
  template <int D0, typename Kernel>
      friend void parallel_for_each(tiled_extent<D0> compute_domain, const Kernel& f);
#endif

  // FIXME: C++AMP parallel_for_each interfaces
#if 0
  template <int D0, typename Kernel>
      friend void parallel_for_each(const accelerator_view&, tiled_extent<D0>, const Kernel&) restrict(cpu,amp);
  template <int D0, int D1, typename Kernel>
      friend void parallel_for_each(const accelerator_view&, tiled_extent<D0, D1>, const Kernel&) restrict(cpu,amp);
  template <int D0, int D1, int D2, typename Kernel>
      friend void parallel_for_each(const accelerator_view&, tiled_extent<D0, D1, D2>, const Kernel&) restrict(cpu,amp);
#endif

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

// tile extent supporting dynamic tile size
// FIXME: disable dependency to Concurrency::extent
class tiled_extent_1D : public extent<1> {
public:
  static const int rank = 1;
  int tile_dim0;
  tiled_extent_1D() restrict(amp,cpu) : extent(0), tile_dim0(0) {}
  tiled_extent_1D(int e0, int t0) restrict(amp,cpu) : extent(e0), tile_dim0(t0) {}
  tiled_extent_1D(const tiled_extent_1D& other) restrict(amp,cpu) : extent(other[0]), tile_dim0(other.tile_dim0) {}
  tiled_extent_1D(const extent<1>& ext, int t0) restrict(amp,cpu) : extent(ext), tile_dim0(t0) {} 
};

// tile extent supporting dynamic tile size
// FIXME: disable dependency to Concurrency::extent
class tiled_extent_2D : public extent<2> {
public:
  static const int rank = 2;
  int tile_dim0;
  int tile_dim1;
  tiled_extent_2D() restrict(amp,cpu) : extent(0, 0), tile_dim0(0), tile_dim1(0) {}
  tiled_extent_2D(int e0, int e1, int t0, int t1) restrict(amp,cpu) : extent(e0, e1), tile_dim0(t0), tile_dim1(t1) {}
  tiled_extent_2D(const tiled_extent_2D& other) restrict(amp,cpu) : extent(other[0], other[1]), tile_dim0(other.tile_dim0), tile_dim1(other.tile_dim1) {}
  tiled_extent_2D(const extent<2>& ext, int t0, int t1) restrict(amp,cpu) : extent(ext), tile_dim0(t0), tile_dim1(t1) {}
};

// tile extent supporting dynamic tile size
// FIXME: disable dependency to Concurrency::extent
class tiled_extent_3D : public extent<3> {
public:
  static const int rank = 3;
  int tile_dim0;
  int tile_dim1;
  int tile_dim2;
  tiled_extent_3D() restrict(amp,cpu) : extent(0, 0, 0), tile_dim0(0), tile_dim1(0), tile_dim2(0) {}
  tiled_extent_3D(int e0, int e1, int e2, int t0, int t1, int t2) restrict(amp,cpu) : extent(e0, e1, e2), tile_dim0(t0), tile_dim1(t1), tile_dim2(t2) {}
  tiled_extent_3D(const tiled_extent_3D& other) restrict(amp,cpu) : extent(other[0], other[1], other[2]), tile_dim0(other.tile_dim0), tile_dim1(other.tile_dim1), tile_dim2(other.tile_dim2) {}
  tiled_extent_3D(const extent<3>& ext, int t0, int t1, int t2) restrict(amp,cpu) : extent(ext), tile_dim0(t0), tile_dim1(t1), tile_dim2(t2) {}
};

/// getLDS : C interface of HSA builtin function to fetch an address within group segment
extern "C" __attribute__((address_space(3))) void* getLDS(unsigned int offset) restrict(amp);

class ts_allocator {
private:
  unsigned int static_group_segment_size;
  unsigned int dynamic_group_segment_size;

  void setStaticGroupSegmentSize(unsigned int size) restrict(cpu) {
    static_group_segment_size = size;
  } 

  template <typename Kernel> friend
    void parallel_for_each(const accelerator_view&, const tiled_extent_1D&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    void parallel_for_each(const accelerator_view&, const tiled_extent_2D&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    void parallel_for_each(const accelerator_view&, const tiled_extent_3D&, ts_allocator&, const Kernel&);

public:
  ts_allocator() :
    static_group_segment_size(0), 
    dynamic_group_segment_size(0) {}

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

  // Allocate the requested size in tile static memory and return its pointer
  // returns NULL if the requested size can't be allocated
  // It requires all threads in a tile to hit the same ts_alloc call site at the
  // same time.
  // Only one instance of the tile static memory will be allocated per call site
  // and all threads within a tile will get the same tile static memory address.
  __attribute__((address_space(3))) void* alloc(unsigned int size) restrict(amp) {
    tile_static int cursor;
    cursor = 0;

    // fetch the beginning address of dynamic group segment
    __attribute__((address_space(3))) unsigned char* lds = (__attribute__((address_space(3))) unsigned char*) getLDS(static_group_segment_size);

    // use atomic fetch_add to allocate
    return lds + __hsail_atomic_fetch_add_int(&cursor, size);
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
  friend class tiled_index_1D;
  friend class tiled_index_2D;
  friend class tiled_index_3D;
};


// FIXME: disable dependency to Concurrency::index
class tiled_index_1D {
public:
  const index<1> global;
  const index<1> local;
  const index<1> tile;
  const index<1> tile_origin;
  const tile_barrier barrier;
  tiled_index_1D(const index<1>& g) restrict(amp,cpu) : global(g) {}
  tiled_index_1D(const tiled_index_1D& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<1>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index_1D(int a, int b, int c, tile_barrier& pb) restrict(amp,cpu) :
    global(a), local(b), tile(c), tile_origin(a - b), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index_1D() restrict(amp) :
    global(index<1>(amp_get_global_id(0))),
    local(index<1>(amp_get_local_id(0))),
    tile(index<1>(amp_get_group_id(0))),
    tile_origin(index<1>(amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index_1D() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index_1D() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel>
  friend void parallel_for_each(const accelerator_view&, const tiled_extent_1D&, ts_allocator&, const Kernel&);
};

// FIXME: disable dependency to Concurrency::index
class tiled_index_2D {
public:
  const index<2> global;
  const index<2> local;
  const index<2> tile;
  const index<2> tile_origin;
  const tile_barrier barrier;
  tiled_index_2D(const index<2>& g) restrict(amp,cpu) : global(g) {}
  tiled_index_2D(const tiled_index_2D& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<2>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index_2D(int a0, int a1, int b0, int b1, int c0, int c1, tile_barrier& pb) restrict(amp,cpu) :
    global(a1, a0), local(b1, b0), tile(c1, c0), tile_origin(a1 - b1, a0 - b0), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index_2D() restrict(amp) :
    global(index<2>(amp_get_global_id(1), amp_get_global_id(0))),
    local(index<2>(amp_get_local_id(1), amp_get_local_id(0))),
    tile(index<2>(amp_get_group_id(1), amp_get_group_id(0))),
    tile_origin(index<2>(amp_get_global_id(1) - amp_get_local_id(1),
                         amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index_2D() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index_2D() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel>
  friend void parallel_for_each(const accelerator_view&, const tiled_extent_2D&, ts_allocator&, const Kernel&);
};

// FIXME: disable dependency to Concurrency::index
class tiled_index_3D {
public:
  const index<3> global;
  const index<3> local;
  const index<3> tile;
  const index<3> tile_origin;
  const tile_barrier barrier;
  tiled_index_3D(const index<3>& g) restrict(amp,cpu) : global(g) {}
  tiled_index_3D(const tiled_index_3D& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<3>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index_3D(int a0, int a1, int a2, int b0, int b1, int b2, int c0, int c1, int c2, tile_barrier& pb) restrict(amp,cpu) :
    global(a2, a1, a0), local(b2, b1, b0), tile(c2, c1, c0), tile_origin(a2 - b2, a1 - b1, a0 - b0), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index_3D() restrict(amp) :
    global(index<3>(amp_get_global_id(2), amp_get_global_id(1), amp_get_global_id(0))),
    local(index<3>(amp_get_local_id(2), amp_get_local_id(1), amp_get_local_id(0))),
    tile(index<3>(amp_get_group_id(2), amp_get_group_id(1), amp_get_group_id(0))),
    tile_origin(index<3>(amp_get_global_id(2) - amp_get_local_id(2),
                         amp_get_global_id(1) - amp_get_local_id(1),
                         amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index_3D() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index_3D() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel>
  friend void parallel_for_each(const accelerator_view&, const tiled_extent_3D&, ts_allocator&, const Kernel&);
};

// variants of parallel_for_each that supports runtime allocation of tile static
template <typename Kernel>
__attribute__((noinline,used))
void parallel_for_each(const accelerator_view& av,
                       const tiled_extent_1D& compute_domain,
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
  size_t tile = compute_domain.tile_dim0;
  if (static_cast<size_t>(compute_domain.tile_dim0) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if (ext % tile != 0) {
    throw invalid_compute_domain("Extent can't be evenly divisible by tile size.");
  }
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av, f, compute_domain);
      return;
  }
#endif
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory<Kernel, 1>(av.pQueue, &ext, &tile, f, kernel, allocator.getDynamicGroupSegmentSize());
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index_1D this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}

// variants of parallel_for_each that supports runtime allocation of tile static
template <typename Kernel>
__attribute__((noinline,used))
void parallel_for_each(const accelerator_view& av,
                       const tiled_extent_2D& compute_domain,
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
  size_t tile[2] = { static_cast<size_t>(compute_domain.tile_dim1),
                     static_cast<size_t>(compute_domain.tile_dim0) };
  if (static_cast<size_t>(compute_domain.tile_dim1 * compute_domain.tile_dim0) > 1024) {
    throw invalid_compute_domain("The maximum nuimber of threads in a tile is 1024");
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
  Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory<Kernel, 2>(av.pQueue, ext, tile, f, kernel, allocator.getDynamicGroupSegmentSize());
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index_2D this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}

// variants of parallel_for_each that supports runtime allocation of tile static
template <typename Kernel>
__attribute__((noinline,used))
void parallel_for_each(const accelerator_view& av,
                       const tiled_extent_3D& compute_domain,
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
  size_t tile[3] = { static_cast<size_t>(compute_domain.tile_dim2),
                     static_cast<size_t>(compute_domain.tile_dim1),
                     static_cast<size_t>(compute_domain.tile_dim0) };
  if (static_cast<size_t>(compute_domain.tile_dim2 * compute_domain.tile_dim1* compute_domain.tile_dim0) > 1024) {
    throw invalid_compute_domain("The maximum nuimber of threads in a tile is 1024");
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
  Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory<Kernel, 3>(av.pQueue, ext, tile, f, kernel, allocator.getDynamicGroupSegmentSize());
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index_3D this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}

template <typename Kernel>
void parallel_for_each(const tiled_extent_1D& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, allocator, f);
}

template<typename Kernel>
void parallel_for_each(const tiled_extent_2D& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, allocator, f);
}

template<typename Kernel>
void parallel_for_each(const tiled_extent_3D& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, allocator, f);
}


} // namespace hc
