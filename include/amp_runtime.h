#pragma once

#include <future>

#define AMP_DEVICE_TYPE_CPU (1)
#define AMP_DEVICE_TYPE_GPU (2)

namespace Concurrency {
namespace CLAMP {
// used in amp.h
extern std::vector<int> EnumerateDevices();

// used in amp_impl.h
extern void QueryDeviceInfo(const std::wstring&, bool&, size_t&, bool&, std::wstring&);

// used in parallel_for_each.h
extern void *CreateKernel(std::string);
extern void LaunchKernel(void *, size_t, size_t *, size_t *);
extern std::future<void>* LaunchKernelAsync(void *, size_t, size_t *, size_t *);
extern void MatchKernelNames(std::string &);

// used in serialize.h
extern void PushArg(void *, int, size_t, const void *);

} // namespace CLAMP
} // namespace Concurrency

