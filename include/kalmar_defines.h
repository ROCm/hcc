#pragma once

// C++ headers
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// CPU execution path
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
#include <ucontext.h>
#endif

//
// work-item related builtin functions
//
extern "C" __attribute__((const,hc)) int64_t hc_get_grid_size(unsigned int n);
extern "C" __attribute__((const,hc)) int64_t hc_get_workitem_absolute_id(unsigned int n);
extern "C" __attribute__((const,hc)) int64_t hc_get_group_size(unsigned int n);
extern "C" __attribute__((const,hc)) int64_t hc_get_workitem_id(unsigned int n);
extern "C" __attribute__((const,hc)) int64_t hc_get_num_groups(unsigned int n);
extern "C" __attribute__((const,hc)) int64_t hc_get_group_id(unsigned int n);

extern "C" __attribute__((const)) int64_t amp_get_global_size(unsigned int n) restrict(amp);
extern "C" __attribute__((const)) int64_t amp_get_global_id(unsigned int n) restrict(amp);
extern "C" __attribute__((const)) int64_t amp_get_local_size(unsigned int n) restrict(amp);
extern "C" __attribute__((const)) int64_t amp_get_local_id(unsigned int n) restrict(amp);
extern "C" __attribute__((const)) int64_t amp_get_num_groups(unsigned int n) restrict(amp);
extern "C" __attribute__((const)) int64_t amp_get_group_id(unsigned int n) restrict(amp);


#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
#define tile_static thread_local
#else
#define tile_static static __attribute__((section("clamp_opencl_local")))
#endif

extern "C" __attribute__((noduplicate,hc)) void hc_barrier(unsigned int n);

extern "C" __attribute__((noduplicate)) void amp_barrier(unsigned int n) restrict(amp);
extern "C" __attribute__((noduplicate)) void hc_barrier(unsigned int n);

/// macro to set if we want default queue be thread-local or not
#define TLS_QUEUE (1)

/**
 * @namespace Kalmar
 * namespace for internal classes of Kalmar compiler / runtime
 */
namespace Kalmar {
} // namespace Kalmar
