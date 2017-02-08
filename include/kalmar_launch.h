//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hc_defines.h"
#include "kalmar_runtime.h"
#include "kalmar_serialize.h"

/** \cond HIDDEN_SYMBOLS */
namespace Kalmar {

template <typename Kernel>
static void append_kernel(const std::shared_ptr<KalmarQueue>& pQueue, const Kernel& f, void* kernel)
{
  Kalmar::BufferArgumentsAppender vis(pQueue, kernel);
  Kalmar::Serialize s(&vis);
  f.__cxxamp_serialize(s);
}

template <typename Kernel>
static inline std::shared_ptr<KalmarQueue> get_availabe_que(const Kernel& f)
{
    Kalmar::QueueSearcher ser;
    Kalmar::Serialize s(&ser);
    f.__cxxamp_serialize(s);
    if (ser.get_que())
        return ser.get_que();
    else
        return getContext()->auto_select();
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
template<typename Kernel, int dim_ext>
inline std::shared_ptr<KalmarAsyncOp>
mcw_cxxamp_launch_kernel_async(const std::shared_ptr<KalmarQueue>& pQueue, size_t *ext,
  size_t *local_size, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  //Invoke Kernel::__cxxamp_trampoline as an kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  {
      std::string kernel_name(f.__cxxamp_trampoline_name());
      kernel = CLAMP::CreateKernel(kernel_name, pQueue.get());
  }
  append_kernel(pQueue, f, kernel);
  return pQueue->LaunchKernelAsync(kernel, dim_ext, ext, local_size);
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
template<typename Kernel, int dim_ext>
inline
void mcw_cxxamp_launch_kernel(const std::shared_ptr<KalmarQueue>& pQueue, size_t *ext,
                              size_t *local_size, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  //Invoke Kernel::__cxxamp_trampoline as an kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  {
      std::string kernel_name(f.__cxxamp_trampoline_name());
      kernel = CLAMP::CreateKernel(kernel_name, pQueue.get());
  }
  append_kernel(pQueue, f, kernel);
  pQueue->LaunchKernel(kernel, dim_ext, ext, local_size);
#endif // __KALMAR_ACCELERATOR__
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
template<typename Kernel>
inline void* mcw_cxxamp_get_kernel(const std::shared_ptr<KalmarQueue>& pQueue, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  //Invoke Kernel::__cxxamp_trampoline as an kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  std::string kernel_name (f.__cxxamp_trampoline_name());
  kernel = CLAMP::CreateKernel(kernel_name, pQueue.get());
  return kernel;
#else
  return NULL;
#endif
}
#pragma clang diagnostic pop

template<typename Kernel, int dim_ext>
inline
void mcw_cxxamp_execute_kernel_with_dynamic_group_memory(
  const std::shared_ptr<KalmarQueue>& pQueue, size_t *ext, size_t *local_size,
  const Kernel& f, void *kernel, size_t dynamic_group_memory_size) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  append_kernel(pQueue, f, kernel);
  pQueue->LaunchKernelWithDynamicGroupMemory(kernel, dim_ext, ext, local_size, dynamic_group_memory_size);
#endif // __KALMAR_ACCELERATOR__
}

template<typename Kernel, int dim_ext>
inline std::shared_ptr<KalmarAsyncOp>
mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async(
  const std::shared_ptr<KalmarQueue>& pQueue, size_t *ext, size_t *local_size,
  const Kernel& f, void *kernel, size_t dynamic_group_memory_size) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  append_kernel(pQueue, f, kernel);
  return pQueue->LaunchKernelWithDynamicGroupMemoryAsync(kernel, dim_ext, ext, local_size, dynamic_group_memory_size);
#endif // __KALMAR_ACCELERATOR__
}

} // namespace Kalmar
/** \endcond */
