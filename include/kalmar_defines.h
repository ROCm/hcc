#pragma once

// C++ headers
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// CPU execution path
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
#include <thread>
#include <ucontext.h>
#endif

//
// work-item related builtin functions
//
extern "C" __attribute__((pure)) int64_t amp_get_global_id(unsigned int n) restrict(amp);
extern "C" __attribute__((pure)) int64_t amp_get_local_id(unsigned int n) restrict(amp);
extern "C" __attribute__((pure)) int64_t amp_get_group_id(unsigned int n) restrict(amp);
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
#define tile_static thread_local
#else
#define tile_static static __attribute__((section("clamp_opencl_local")))
#endif
extern "C" __attribute__((noduplicate)) void amp_barrier(unsigned int n) restrict(amp);


/**
 * @namespace Kalmar
 * namespace for internal classes of Kalmar compiler / runtime
 */
namespace Kalmar {
} // namespace Kalmar
