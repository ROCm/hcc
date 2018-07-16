#pragma once

// C++ headers
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// CPU execution path
#if __HCC_ACCELERATOR__ == 2 || __HCC_CPU__ == 2
#include <ucontext.h>
#endif

namespace hc {
  typedef _Float16 half;
}

//
// work-item related builtin functions
//
extern "C" __attribute__((const,hc)) uint32_t hc_get_grid_size(unsigned int n);
extern "C" __attribute__((const,hc)) uint32_t hc_get_workitem_absolute_id(unsigned int n);
extern "C" __attribute__((const,hc)) uint32_t hc_get_group_size(unsigned int n);
extern "C" __attribute__((const,hc)) uint32_t hc_get_workitem_id(unsigned int n);
extern "C" __attribute__((const,hc)) uint32_t hc_get_num_groups(unsigned int n);
extern "C" __attribute__((const,hc)) uint32_t hc_get_group_id(unsigned int n);

#define tile_static __attribute__((tile_static))

extern "C" __attribute__((noduplicate, hc)) void hc_barrier(unsigned int n);

/// macro to set if we want default queue be thread-local or not
#define TLS_QUEUE (1)


#ifndef CLK_LOCAL_MEM_FENCE
#define CLK_LOCAL_MEM_FENCE (1)
#endif

#ifndef CLK_GLOBAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE (2)
#endif

/**
 * @namespace Kalmar
 * namespace for internal classes of Kalmar compiler / runtime
 */
namespace Kalmar {
} // namespace Kalmar

// Provide automatic type conversion for void*.
class auto_voidp {
    void *_ptr;
    public:
        auto_voidp (void *ptr) : _ptr (ptr) {}
        template<class T> operator T *() { return (T *) _ptr; }
};

// Valid values for__hcc_backend__ to indicate the
// compiler backend
#define HCC_BACKEND_AMDGPU (1)
