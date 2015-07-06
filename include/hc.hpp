#pragma once

#include <amp.h>

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

#if 0
// FIXME: move to hc namespace
class tiled_extent_3D : public extent<3> {
public:
  static const int rank = 3;
  int tile_dim0;
  int tile_dim1;
  int tile_dim2;
  tiled_extent_3D() restrict(amp,cpu) {}
  tiled_extent_3D(int e0, int e1, int e2, int t0, int t1, int t2) restrict(amp,cpu) : extent(e0, e1, e2), tile_dim0(t0), tile_dim1(t1), tile_dim2(t2) {}
  tiled_extent_3D(const tiled_extent_3D& other) restrict(amp,cpu) : extent(other[0], other[1], other[2]), tile_dim0(other.tile_dim0), tile_dim1(other.tile_dim1), tile_dim2(other.tile_dim2) {}
  tiled_extent_3D(const extent<3>& ext, int t0, int t1, int t2) restrict(amp,cpu) : extent(ext), tile_dim0(t0), tile_dim1(t1), tile_dim2(t2) {}
};
#endif

#if 0
// variants of parallel_for_each that supports runtime allocation of tile static
// FIXME: move from Concurrency::accelerator_view to hc::accelerator_view
// FIXME: take tile_static_allocatable_size into consideration
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(const accelerator_view& av, tiled_extent_1D compute_domain, size_t tile_static_allocatable_size, const Kernel& f) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av, f, compute_domain);
      return;
  }
#endif
  size_t ext = compute_domain[0];
  mcw_cxxamp_launch_kernel<Kernel, 1>(av, &ext, NULL, f);
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#endif


// FIXME: enable it after 1D case work
#if 0
template<typename Kernel>
void parallel_for_each(const accelerator_view& av, tiled_extent_2D compute_domain, size_t tile_static_allocatable_size, const Kernel& f);

template<typename Kernel>
void parallel_for_each(const accelerator_view& av, tiled_extent_3D compute_domain, size_t tile_static_allocatable_size, const Kernel& f);
#endif

#if 0
template <typename Kernel>
void parallel_for_each(tiled_extent_1D compute_domain, size_t tile_static_allocatable_size, const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, tile_static_allocatable_size, f);
}
#endif

// FIXME: enable it after 1D case work
#if 0
template<typename Kernel>
void parallel_for_each(tiled_extent_2D compute_domain, size_t tile_static_allocatable_size, const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, tile_static_allocatable_size, f);
}

template<typename Kernel>
void parallel_for_each(tiled_extent_3D compute_domain, size_t tile_static_allocatable_size, const Kernel& f) {
  parallel_for_each(accelerator().get_default_view(), compute_domain, tile_static_allocatable_size, f);
}
#endif

#if 0
// Allocate the requested size in tile static memory and return its pointer
// returns NULL if the requested size can't be allocated
// It requires all threads in a tile to hit the same ts_alloc call site at the
// same time.
// Only one instance of the tile static memory will be allocated per call site
// and all threads within a tile will get the same tile static memory address.
__attribute__((amp))
void* ts_alloc(size_t size);
#endif

} // namespace Concurrency
// FIXME: use hc namespace
