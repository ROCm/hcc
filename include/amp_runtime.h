#pragma once

#include <future>
#include <amp_allocator.h>

namespace Concurrency { namespace CLAMP {

// used in parallel_for_each.h
#ifdef __AMP_CPU__
extern bool is_cpu();
extern bool in_cpu_kernel();
extern void enter_kernel();
extern void leave_kernel();
#endif

extern void *CreateKernel(std::string, AMPAllocator*);
extern std::shared_future<void>* LaunchKernelAsync(void *, size_t, size_t *, size_t *);
extern void MatchKernelNames(std::string &);

// used in serialize.h
extern void PushArg(void *, int, size_t, const void *);
extern void PushArgPtr(void *, int, size_t, const void *);

} // namespace CLAMP
} // namespace Concurrency
