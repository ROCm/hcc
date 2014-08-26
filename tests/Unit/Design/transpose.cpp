// RUN: %cxxamp %s -o %t.out && %t.out
//----------------------------------------------------------------------------
// File: transpose.cpp
//
// Implement C++ AMP version of matrix transpose
//----------------------------------------------------------------------------

#include <amp.h>
#include <cmath>
#include <assert.h>
#include <iostream>
#include <sstream>


using namespace concurrency;


//-----------------------------------------------------------------------------
// Common utility functions and definitions
//-----------------------------------------------------------------------------
template <typename _2d_index_type>
_2d_index_type transpose(const _2d_index_type& idx) restrict(cpu, amp) {
  return _2d_index_type(idx[1], idx[0]);
}

//-----------------------------------------------------------------------------
// Using simple model transpose
//-----------------------------------------------------------------------------
template <typename _value_type>
void transpose_simple(const array_view<const _value_type, 2>& data,
                      const array_view<_value_type, 2>& data_transpose) {
  assert(data.get_extent() == transpose(data_transpose.get_extent()));

  data_transpose.discard_data();
  parallel_for_each(data.get_extent(), [=] (index<2> idx) restrict(amp) {
    data_transpose[transpose(idx)] = data[idx];
  });
}

//-----------------------------------------------------------------------------
// Using tiled model transpose, assumes input size is evenly divided
// by the tile size
//-----------------------------------------------------------------------------
template <typename _value_type, int _tile_size>
void transpose_tiled_even(const array_view<const _value_type, 2>& data,
                          const array_view<_value_type, 2>& data_transpose) {
  assert(data.get_extent() == transpose(data_transpose.get_extent()));
  assert(data.get_extent()[0] % _tile_size == 0);
  assert(data.get_extent()[1] % _tile_size == 0);

  data_transpose.discard_data();
  extent<2> e = data.get_extent();

  parallel_for_each(e.tile<_tile_size, _tile_size>(),
      [=] (tiled_index<_tile_size, _tile_size> tidx) restrict(amp) {
    tile_static _value_type t1[_tile_size][_tile_size];
    t1[tidx.local[1]][tidx.local[0]] = data[tidx.global];

    tidx.barrier.wait();

    index<2> idxdst(transpose(tidx.tile_origin) + tidx.local);
    data_transpose[idxdst] = t1[tidx.local[0]][tidx.local[1]];
  });
}

//-----------------------------------------------------------------------------
// Tiled transpose -- padding-based solution.
//
// This solution doesn't assume input size is evenly divided by the tile size.
// To handle the unevenness, extra threads are padded to the extent. The extra
// threads are filtered out when they need to execute global memory operations.
//-----------------------------------------------------------------------------
template <typename _value_type>
_value_type guarded_read(const array_view<const _value_type, 2>& data,
                         const index<2>& idx) restrict(amp) {
  auto e = data.get_extent();
  return e.contains(idx) ? data[idx] : _value_type();
}

template <typename _value_type>
void guarded_write(const array_view<_value_type, 2>& data, const index<2>& idx,
                   const _value_type& val) restrict(amp) {
  auto e = data.get_extent();
  if(e.contains(idx))
    data[idx] = val;
}

template <typename _value_type, int _tile_size>
void transpose_tiled_pad(const array_view<const _value_type, 2>& data,
                         const array_view<_value_type, 2>& data_transpose) {
  assert(data.get_extent() == transpose(data_transpose.get_extent()));

  data_transpose.discard_data();
  extent<2> e = data.get_extent();
  parallel_for_each(e.tile<_tile_size, _tile_size>().pad(),
      [=] (tiled_index<_tile_size, _tile_size> tidx) restrict(amp) {
    tile_static _value_type t1[_tile_size][_tile_size];
    t1[tidx.local[1]][tidx.local[0]] = guarded_read(data, tidx.global);

    tidx.barrier.wait();

    index<2> idxdst(transpose(tidx.tile_origin) + tidx.local);
    guarded_write(data_transpose, idxdst, t1[tidx.local[0]][tidx.local[1]]);
  });
}

//-----------------------------------------------------------------------------
// Tiled transpose -- truncation-based solution (first take)
//
// This solution doesn't assume input size is evenly divided by the tile size.
// To handle the unevenness, the extent is truncated to even tile boundaries.
// The extra work is carried by threads which reside on the lower and right
// boundaries of the compute domain. The thread at the bottom-right corner
// takes care of the little corner-right corner which isn't handled by anybody
// else.
//
// This is illustrated by the following ascii graphic
//
//       01234567890123456789
//       --------------------
//     0|MMMMMMMMMMMMMMMMRRRR
//     1|MMMMMMMMMMMMMMMMRRRR
//     2|MMMMMMMMMMMMMMMMRRRR
//     3|MMMMMMMMMMMMMMMMRRRR
//     4|MMMMMMMMMMMMMMMMRRRR
//     5|MMMMMMMMMMMMMMMMRRRR
//     6|MMMMMMMMMMMMMMMMRRRR
//     7|MMMMMMMMMMMMMMMMRRRR
//     8|MMMMMMMMMMMMMMMMRRRR
//     9|MMMMMMMMMMMMMMMMRRRR
//     0|MMMMMMMMMMMMMMMMRRRR
//     1|MMMMMMMMMMMMMMMMRRRR
//     2|MMMMMMMMMMMMMMMMRRRR
//     3|MMMMMMMMMMMMMMMMRRRR
//     4|MMMMMMMMMMMMMMMMRRRR
//     5|MMMMMMMMMMMMMMMMRRRR
//     6|BBBBBBBBBBBBBBBBCCCC
//     7|BBBBBBBBBBBBBBBBCCCC
//     8|BBBBBBBBBBBBBBBBCCCC
//     9|BBBBBBBBBBBBBBBBCCCC
//
// The graphic shows that if we transpose a (20,20) matrix, then the threads in
// the region (0,0) to (15,15) will just transpose their index. This takes care
// of the indices marked with 'M'.
//
// Then, for any row index r in the range (0,15), i.e., the threads in the 15th
// column,  will transpose in addition to their ID also the cells (r,16),
// (r,17), (r,18) and (r,19). This takes care of the cells marked with 'R'.
//
// Similarly, the 15th row of threads takes care of the cells marked with 'B'.
//
// Finally, the thread with ID (15,15) takes care of transposing the cells
// marked with 'C'.
//-----------------------------------------------------------------------------
template <typename _value_type, int _tile_size>
void transpose_tiled_truncate_option_a(
          const array_view<const _value_type, 2>& data,
          const array_view<_value_type, 2>& data_transpose) {
  extent<2> e = data.get_extent();
  tiled_extent<_tile_size, _tile_size> e_truncated(e.tile<_tile_size,
                                                   _tile_size>().truncate());

  data_transpose.discard_data();
  parallel_for_each(e_truncated,
      [=] (tiled_index<_tile_size, _tile_size> tidx) restrict(amp) {
    // Normal processing
    tile_static _value_type t1[_tile_size][_tile_size];
    t1[tidx.local[1]][tidx.local[0]] = data[tidx.global];

    tidx.barrier.wait();

    index<2> idxdst(transpose(tidx.tile_origin) + tidx.local);
    data_transpose[idxdst] = t1[tidx.local[0]][tidx.local[1]];

    // Leftover processing
    int idx0, idx1;
    bool is_bottom_thread = tidx.global[0] == e_truncated[0]-1;
    bool is_rightmost_thread = tidx.global[1] == e_truncated[1]-1;
    if(is_rightmost_thread | is_bottom_thread) {
      // Right leftover band
      if(is_rightmost_thread) {
        idx0 = tidx.global[0];
        for(idx1 = e_truncated[1]; idx1 < data.get_extent()[1]; idx1++)
          data_transpose(idx1, idx0) = data(idx0, idx1);
      }
      // Bottom leftover band
      if(is_bottom_thread) {
        idx1 = tidx.global[1];
        for(idx0 = e_truncated[0]; idx0 < data.get_extent()[0]; idx0++)
          data_transpose(idx1, idx0) = data(idx0, idx1);
      }
      // Bottom right leftover corner
      if(is_bottom_thread & is_rightmost_thread) {
        for(idx0 = e_truncated[0]; idx0 < data.get_extent()[0]; idx0++)
          for(idx1 = e_truncated[1]; idx1 < data.get_extent()[1]; idx1++)
            data_transpose(idx1, idx0) = data(idx0, idx1);
      }
    }
  });
}

//-----------------------------------------------------------------------------
// Tiled transpose -- truncation-based solution (second take)
//
// This solution doesn't assume input size is evenly divided by the tile size.
// To handle the unevenness, three kernel invocations are used:
//
// 1) The evenly divided part of the matrix is handled by transpose_tiled_even
// 2) The leftover lower band is handled by the simple algorithm.
// 3) The leftover right bad is handled by the simple algorithm too.
//
//-----------------------------------------------------------------------------
template <typename _value_type, int _tile_size>
void transpose_tiled_truncate_option_b(
         const array_view<const _value_type, 2>& data,
         const array_view<_value_type, 2>& data_transpose) {
  extent<2> e = data.get_extent();
  tiled_extent<_tile_size, _tile_size> e_tiled(e.tile<_tile_size,
                                               _tile_size>());
  tiled_extent<_tile_size, _tile_size> e_truncated(e_tiled.truncate());

  // Transform matrix to be multiple of 16*16 and transpose.
  auto b  = data.section(index<2>(0,0), e_truncated);
  auto b_t = data_transpose.section(index<2>(0,0),
                 transpose(static_cast<extent<2>>(e_truncated)));
  transpose_tiled_even<_value_type, _tile_size>(b, b_t);

  // leftover processing
  if(e_truncated[0] < e_tiled[0]) {
    index<2> offset(e_truncated[0],0);
    extent<2> e(data.get_extent()[0]-e_truncated[0], e_truncated[1]);

    auto b  = data.section(offset, e);
    auto b_t = data_transpose.section(transpose(offset), transpose(e));
    transpose_simple(b, b_t);
  }
  if(e_truncated[1] < e_tiled[1]) {
    index<2> offset(0, e_truncated[1]);
    auto b  = data.section(offset);
    auto b_t = data_transpose.section(transpose(offset));
    transpose_simple(b, b_t);
  }
}

//-----------------------------------------------------------------------------
// Test driver
//-----------------------------------------------------------------------------
typedef void traspose_func(
    const array_view<const float, 2>& data,
    const array_view<float, 2>& data_transpose);

void test_transpose_func(int m, int n, traspose_func *user_func,
                         std::string func_name) {
  std::cout << "Testing implementation " << func_name << std::endl;
  std::vector<float> v_data(m * n);
  std::vector<float> v_data_transpose(n * m);

  bool passed = true;

  for(int ir = 0; ir < m; ir++) {
    for(int ic = 0; ic < n; ic++) {
      v_data[ir * n + ic] = ir * 37.0f + ic * 7.0f;
      v_data_transpose[ir * n + ic] = -1.0f;
    }
  }

  array_view<const float, 2> data_av(m, n, v_data);
  array_view<float, 2> data_transpose_av(n, m, v_data_transpose);

  user_func(data_av, data_transpose_av);

  data_transpose_av.synchronize();

  for(int ir = 0; ir < m; ir++) {
    for(int ic = 0; ic < n; ic++) {
      if(v_data[ir * n + ic] != v_data_transpose[ic * m + ir]) {
        std::cout << "Mismatch at (" << ir << "," << ic << ") data="
                  << v_data[ir * n + ic] << " transpose="
                  << v_data_transpose[ic * m + ir] << std::endl;
        passed = false;
      }
    }
  }
  std::cout << "Test "
            << static_cast<const char *>(passed ? "passed" : "failed")
            << std::endl;
}

int main() {
  accelerator default_device;
  std::wcout << L"Using device : " << default_device.get_description()
             << std::endl;

  std::cout << "Running test transpose_simple" << std::endl;
  test_transpose_func(999, 666, transpose_simple<float>, "transpose_simple");

  std::cout << "Running test transpose_tiled_even" << std::endl;
  test_transpose_func(992, 656, transpose_tiled_even<float, 16>,
                      "transpose_tiled_even");

  std::cout << "Running test transpose_tiled_pad" << std::endl;
  test_transpose_func(999, 666, transpose_tiled_pad<float, 16>,
                      "transpose_tiled_pad");

  std::cout << "Running test transpose_tiled_truncate_option_a" << std::endl;
  test_transpose_func(999, 666, transpose_tiled_truncate_option_a<float, 16>,
                      "transpose_tiled_truncate_option_a");
#if 1
  std::cout << "Running test transpose_tiled_truncate_option_b" << std::endl;
  test_transpose_func(999, 666, transpose_tiled_truncate_option_b<float, 16>,
                      "transpose_tiled_truncate_option_b");
#endif
  return 0;
}

