#pragma once

#include <amp.h>

#include <hsa_atomic.h>

// FIXME: use hc namespace
namespace Concurrency {

// returns the size of tile static area
// FIXME: make it a member function inside hc::acclerator_view
size_t get_max_tile_static_size(const accelerator_view& av) {
  return av.pQueue.get()->getDev()->GetMaxTileStaticSize();
}

// returns the size of tile static area
// FIXME: make it a member function inside hc::accelerator
size_t get_max_tile_static_size(const accelerator& acc) {
  return get_max_tile_static_size(acc.get_default_view());
}


// tile extent supporting dynamic tile size
// FIXME: move to hc namespace
class tiled_extent_1D : public extent<1> {
public:
  static const int rank = 1;
  int tile_dim0;
  tiled_extent_1D() restrict(amp,cpu) : extent(0), tile_dim0(0) {}
  tiled_extent_1D(int e0, int t0) restrict(amp,cpu) : extent(e0), tile_dim0(t0) {}
  tiled_extent_1D(const tiled_extent_1D& other) restrict(amp,cpu) : extent(other[0]), tile_dim0(other.tile_dim0) {}
  tiled_extent_1D(const extent<1>& ext, int t0) restrict(amp,cpu) : extent(ext), tile_dim0(t0) {} 
};

// FIXME: move to hc namespace
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

// FIXME: move to hc namespace
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

class tiled_index_2D {
public:
  const index<2> global;
  const index<2> local;
  const index<2> tile;
  const index<2> tile_origin;
  // FIXME: add tile_barrier
  // const tile_barrier barrier;
  tiled_index_2D(const index<2>& g) restrict(amp,cpu) : global(g) {}
  tiled_index_2D(const tiled_index_2D& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin) /*, barrier(o.barrier)*/ {}
  operator const index<2>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index_2D(int a0, int a1, int b0, int b1, int c0, int c1/*, tile_barrier& pb*/) restrict(amp,cpu) :
    global(a1, a0), local(b1, b0), tile(c1, c0), tile_origin(a1 - b1, a0 - b0)/*, barrier(pb) */ {}
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

class tiled_index_3D {
public:
  const index<3> global;
  const index<3> local;
  const index<3> tile;
  const index<3> tile_origin;
  // FIXME: add tile_barrier
  // const tile_barrier barrier;
  tiled_index_3D(const index<3>& g) restrict(amp,cpu) : global(g) {}
  tiled_index_3D(const tiled_index_3D& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin) /*, barrier(o.barrier)*/ {}
  operator const index<3>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index_3D(int a0, int a1, int a2, int b0, int b1, int b2, int c0, int c1, int c2/*, tile_barrier& pb*/) restrict(amp,cpu) :
    global(a2, a1, a0), local(b2, b1, b0), tile(c2, c1, c0), tile_origin(a2 - b2, a1 - b1, a0 - b0)/*, barrier(pb) */ {}
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
// FIXME: move from Concurrency namespace to hc
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(const accelerator_view& av, const tiled_extent_1D& compute_domain,
                                                      ts_allocator& allocator, const Kernel& f) restrict(amp,cpu) {
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
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory<Kernel, 1>(av, &ext, &tile, f, kernel, allocator.getDynamicGroupSegmentSize());
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index_1D this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}

// variants of parallel_for_each that supports runtime allocation of tile static
// FIXME: move from Concurrency namespace to hc
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(const accelerator_view& av, const tiled_extent_2D& compute_domain,
                                                      ts_allocator& allocator, const Kernel& f) restrict(cpu,amp) {
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
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory<Kernel, 2>(av, ext, tile, f, kernel, allocator.getDynamicGroupSegmentSize());
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index_2D this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}

// variants of parallel_for_each that supports runtime allocation of tile static
// FIXME: move from Concurrency namespace to hc
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(const accelerator_view& av, const tiled_extent_3D& compute_domain,
                                                      ts_allocator& allocator, const Kernel& f) restrict(cpu,amp) {
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
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory<Kernel, 3>(av, ext, tile, f, kernel, allocator.getDynamicGroupSegmentSize());
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index_3D this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}

template <typename Kernel>
void parallel_for_each(const tiled_extent_1D& compute_domain, ts_allocator& allocator, const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, allocator, f);
}

template<typename Kernel>
void parallel_for_each(const tiled_extent_2D& compute_domain, ts_allocator& allocator, const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, allocator, f);
}

template<typename Kernel>
void parallel_for_each(const tiled_extent_3D& compute_domain, ts_allocator& allocator, const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, allocator, f);
}


} // namespace Concurrency
// FIXME: use hc namespace
