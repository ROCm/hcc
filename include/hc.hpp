//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

/**
 * @file hc.hpp
 * Heterogeneous C++ (HC) API.
 */

#pragma once

#include "hc_defines.h"
#include "kalmar_exception.h"
#include "kalmar_runtime.h"

#include "hcc_features.hpp"

#ifndef __HC__
#   define __HC__ [[hc]]
#endif

#ifndef __CPU__
#   define __CPU__ [[cpu]]
#endif

typedef struct hsa_kernel_dispatch_packet_s hsa_kernel_dispatch_packet_t;

/**
 * @namespace hc
 * Heterogeneous  C++ (HC) namespace
 */
namespace Kalmar {
    class HSAQueue;
};

namespace hc {

class AmPointerInfo;

using namespace Kalmar::enums;
using namespace Kalmar::CLAMP;

// forward declaration
class accelerator;
class accelerator_view;
class completion_future;

using runtime_exception = Kalmar::runtime_exception;
using invalid_compute_domain = Kalmar::invalid_compute_domain;
using accelerator_view_removed = Kalmar::accelerator_view_removed;

// ------------------------------------------------------------------------
// global functions
// ------------------------------------------------------------------------

/**
 * Get the current tick count for the GPU platform.
 *
 * @return An implementation-defined tick count
 */
inline uint64_t get_system_ticks() {
    return Kalmar::getContext()->getSystemTicks();
}

/**
 * Get the frequency of ticks per second for the underlying asynchrnous operation.
 *
 * @return An implementation-defined frequency in Hz in case the instance is
 *         created by a kernel dispatch or a barrier packet. 0 otherwise.
 */
inline uint64_t get_tick_frequency() {
    return Kalmar::getContext()->getSystemTickFrequency();
}

#define GET_SYMBOL_ADDRESS(acc, symbol) \
    acc.get_symbol_address( #symbol );


// ------------------------------------------------------------------------
// accelerator_view
// ------------------------------------------------------------------------

/**
 * Represents a logical (isolated) accelerator view of a compute accelerator.
 * An object of this type can be obtained by calling the default_view property
 * or create_view member functions on an accelerator object.
 */
class accelerator_view {
public:
    /**
     * Copy-constructs an accelerator_view object. This function does a shallow
     * copy with the newly created accelerator_view object pointing to the same
     * underlying view as the "other" parameter.
     *
     * @param[in] other The accelerator_view object to be copied.
     */
    accelerator_view(const accelerator_view& other) :
        pQueue(other.pQueue) {}

    /**
     * Assigns an accelerator_view object to "this" accelerator_view object and
     * returns a reference to "this" object. This function does a shallow
     * assignment with the newly created accelerator_view object pointing to
     * the same underlying view as the passed accelerator_view parameter.
     *
     * @param[in] other The accelerator_view object to be assigned from.
     * @return A reference to "this" accelerator_view object.
     */
    accelerator_view& operator=(const accelerator_view& other) {
        pQueue = other.pQueue;
        return *this;
    }

    /**
     * Returns the queuing mode that this accelerator_view was created with.
     * See "Queuing Mode".
     *
     * @return The queuing mode.
     */
    queuing_mode get_queuing_mode() const { return pQueue->get_mode(); }

    /**
     * Returns the execution order of this accelerator_view.
     */
    execute_order get_execute_order() const { return pQueue->get_execute_order(); }

    /**
     * Returns a boolean value indicating whether the accelerator view when
     * passed to a parallel_for_each would result in automatic selection of an
     * appropriate execution target by the runtime. In other words, this is the
     * accelerator view that will be automatically selected if
     * parallel_for_each is invoked without explicitly specifying an
     * accelerator view.
     *
     * @return A boolean value indicating if the accelerator_view is the auto
     *         selection accelerator_view.
     */
    // FIXME: dummy implementation now
    bool get_is_auto_selection() { return false; }

    /**
     * Returns a 32-bit unsigned integer representing the version number of
     * this accelerator view. The format of the integer is major.minor, where
     * the major version number is in the high-order 16 bits, and the minor
     * version number is in the low-order bits.
     *
     * The version of the accelerator view is usually the same as that of the
     * parent accelerator.
     */
    unsigned int get_version() const;

    /**
     * Returns the accelerator that this accelerator_view has been created on.
     */
    accelerator get_accelerator() const;

    /**
     * Returns a boolean value indicating whether the accelerator_view supports
     * debugging through extensive error reporting.
     *
     * The is_debug property of the accelerator view is usually same as that of
     * the parent accelerator.
     */
    // FIXME: dummy implementation now
    bool get_is_debug() const { return 0; } 

    /**
     * Performs a blocking wait for completion of all commands submitted to the
     * accelerator view prior to calling wait().
     *
     * @param waitMode[in] An optional parameter to specify the wait mode. By
     *                     default it would be hcWaitModeBlocked.
     *                     hcWaitModeActive would be used to reduce latency with
     *                     the expense of using one CPU core for active waiting.
     */
    void wait(hcWaitMode waitMode = hcWaitModeBlocked) { 
      pQueue->wait(waitMode); 
      Kalmar::getContext()->flushPrintfBuffer();
    }

    /**
     * Sends the queued up commands in the accelerator_view to the device for
     * execution.
     *
     * An accelerator_view internally maintains a buffer of commands such as
     * data transfers between the host memory and device buffers, and kernel
     * invocations (parallel_for_each calls). This member function sends the
     * commands to the device for processing. Normally, these commands 
     * to the GPU automatically whenever the runtime determines that they need
     * to be, such as when the command buffer is full or when waiting for 
     * transfer of data from the device buffers to host memory. The flush 
     * member function will send the commands manually to the device.
     *
     * Calling this member function incurs an overhead and must be used with
     * discretion. A typical use of this member function would be when the CPU
     * waits for an arbitrary amount of time and would like to force the
     * execution of queued device commands in the meantime. It can also be used
     * to ensure that resources on the accelerator are reclaimed after all
     * references to them have been removed.
     *
     * Because flush operates asynchronously, it can return either before or
     * after the device finishes executing the buffered commandser, the
     * commands will eventually always complete.
     *
     * If the queuing_mode is queuing_mode_immediate, this function has no effect.
     *
     * @return None
     */
    void flush() { pQueue->flush(); }

    /**
     * This command inserts a marker event into the accelerator_view's command
     * queue. This marker is returned as a completion_future object. When all
     * commands that were submitted prior to the marker event creation have
     * completed, the future is ready.
     *
     * Regardless of the accelerator_view's execute_order (execute_any_order, execute_in_order), 
     * the marker always ensures older commands complete before the returned completion_future
     * is marked ready.   Thus, markers provide a mechanism to enforce order between
     * commands in an execute_any_order accelerator_view.
     *
     * fence_scope controls the scope of the acquire and release fences applied after the marker executes.  Options are:
     *   - no_scope : No fence operation is performed.
     *   - accelerator_scope: Memory is acquired from and released to the accelerator scope where the marker executes.
     *   - system_scope: Memory is acquired from and released to system scope (all accelerators including CPUs)
     *
     * @return A future which can be waited on, and will block until the
     *         current batch of commands has completed.
     */
    completion_future create_marker(memory_scope fence_scope=system_scope) const;

    /**
     * This command inserts a marker event into the accelerator_view's command
     * queue with a prior dependent asynchronous event.
     *
     * This marker is returned as a completion_future object. When its
     * dependent event and all commands submitted prior to the marker event
     * creation have been completed, the future is ready.
     *
     * Regardless of the accelerator_view's execute_order (execute_any_order, execute_in_order), 
     * the marker always ensures older commands complete before the returned completion_future
     * is marked ready.   Thus, markers provide a mechanism to enforce order between
     * commands in an execute_any_order accelerator_view.
     *
     * fence_scope controls the scope of the acquire and release fences applied after the marker executes.  Options are:
     *   - no_scope : No fence operation is performed.
     *   - accelerator_scope: Memory is acquired from and released to the accelerator scope where the marker executes.
     *   - system_scope: Memory is acquired from and released to system scope (all accelerators including CPUs)
     *
     * dependent_futures may be recorded in another queue or another accelerator.  If in another accelerator,
     * the runtime performs cross-accelerator sychronization.  
     *
     * @return A future which can be waited on, and will block until the
     *         current batch of commands, plus the dependent event have
     *         been completed.
     */
    completion_future create_blocking_marker(completion_future& dependent_future, memory_scope fence_scope=system_scope) const;

    /**
     * This command inserts a marker event into the accelerator_view's command
     * queue with arbitrary number of dependent asynchronous events.
     *
     * This marker is returned as a completion_future object. When its
     * dependent events and all commands submitted prior to the marker event
     * creation have been completed, the completion_future is ready.
     *
     * Regardless of the accelerator_view's execute_order (execute_any_order, execute_in_order), 
     * the marker always ensures older commands complete before the returned completion_future
     * is marked ready.   Thus, markers provide a mechanism to enforce order between
     * commands in an execute_any_order accelerator_view.
     *
     * fence_scope controls the scope of the acquire and release fences applied after the marker executes.  Options are:
     *   - no_scope : No fence operation is performed.
     *   - accelerator_scope: Memory is acquired from and released to the accelerator scope where the marker executes.
     *   - system_scope: Memory is acquired from and released to system scope (all accelerators including CPUs)
     *
     * @return A future which can be waited on, and will block until the
     *         current batch of commands, plus the dependent event have
     *         been completed.
     */
    completion_future create_blocking_marker(std::initializer_list<completion_future> dependent_future_list, memory_scope fence_scope=system_scope) const;


    /**
     * This command inserts a marker event into the accelerator_view's command
     * queue with arbitrary number of dependent asynchronous events.
     *
     * This marker is returned as a completion_future object. When its
     * dependent events and all commands submitted prior to the marker event
     * creation have been completed, the completion_future is ready.
     *
     * Regardless of the accelerator_view's execute_order (execute_any_order, execute_in_order), 
     * the marker always ensures older commands complete before the returned completion_future
     * is marked ready.   Thus, markers provide a mechanism to enforce order between
     * commands in an execute_any_order accelerator_view.
     *
     * @return A future which can be waited on, and will block until the
     *         current batch of commands, plus the dependent event have
     *         been completed.
     */
    template<typename InputIterator>
    completion_future create_blocking_marker(InputIterator first, InputIterator last, memory_scope scope) const;

    /**
     * Copies size_bytes bytes from src to dst.  
     * Src and dst must not overlap.  
     * Note the src is the first parameter and dst is second, following C++ convention.
     * The copy command will execute after any commands already inserted into the accelerator_view finish.
     * This is a synchronous copy command, and the copy operation complete before this call returns.
     */
    void copy(const void *src, void *dst, size_t size_bytes) {
        pQueue->copy(src, dst, size_bytes);
    }


    /**
     * Copies size_bytes bytes from src to dst.  
     * Src and dst must not overlap.  
     * Note the src is the first parameter and dst is second, following C++ convention.
     * The copy command will execute after any commands already inserted into the accelerator_view finish.
     * This is a synchronous copy command, and the copy operation complete before this call returns.
     * The copy_ext flavor allows caller to provide additional information about each pointer, which can improve performance by eliminating replicated lookups.
     * This interface is intended for language runtimes such as HIP.
    
     @p copyDir : Specify direction of copy.  Must be hcMemcpyHostToHost, hcMemcpyHostToDevice, hcMemcpyDeviceToHost, or hcMemcpyDeviceToDevice. 
     @p forceUnpinnedCopy : Force copy to be performed with host involvement rather than with accelerator copy engines.
     */
    void copy_ext(const void *src, void *dst, size_t size_bytes, hcCommandKind copyDir, const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, const hc::accelerator *copyAcc, bool forceUnpinnedCopy);


    // TODO - this form is deprecated, provided for use with older HIP runtimes.
    void copy_ext(const void *src, void *dst, size_t size_bytes, hcCommandKind copyDir, const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, bool forceUnpinnedCopy) ;

    /**
     * Copies size_bytes bytes from src to dst.  
     * Src and dst must not overlap.  
     * Note the src is the first parameter and dst is second, following C++ convention.  
     * This is an asynchronous copy command, and this call may return before the copy operation completes.
     * If the source or dest is host memory, the memory must be pinned or a runtime exception will be thrown.
     * Pinned memory can be created with am_alloc with flag=amHostPinned flag.
     *
     * The copy command will be implicitly ordered with respect to commands previously equeued to this accelerator_view:
     * - If the accelerator_view execute_order is execute_in_order (the default), then the copy will execute after all previously sent commands finish execution.
     * - If the accelerator_view execute_order is execute_any_order, then the copy will start after all previously send commands start but can execute in any order.
     *
     *
     */
    completion_future copy_async(const void *src, void *dst, size_t size_bytes);


    /**
     * Copies size_bytes bytes from src to dst.  
     * Src and dst must not overlap.  
     * Note the src is the first parameter and dst is second, following C++ convention.  
     * This is an asynchronous copy command, and this call may return before the copy operation completes.
     * If the source or dest is host memory, the memory must be pinned or a runtime exception will be thrown.
     * Pinned memory can be created with am_alloc with flag=amHostPinned flag.
     *
     * The copy command will be implicitly ordered with respect to commands previously enqueued to this accelerator_view:
     * - If the accelerator_view execute_order is execute_in_order (the default), then the copy will execute after all previously sent commands finish execution.
     * - If the accelerator_view execute_order is execute_any_order, then the copy will start after all previously send commands start but can execute in any order.
     *   The copyAcc determines where the copy is executed and does not affect the ordering.
     *
     * The copy_async_ext flavor allows caller to provide additional information about each pointer, which can improve performance by eliminating replicated lookups,
     * and also allow control over which device performs the copy.  
     * This interface is intended for language runtimes such as HIP.
     *
     *  @p copyDir : Specify direction of copy.  Must be hcMemcpyHostToHost, hcMemcpyHostToDevice, hcMemcpyDeviceToHost, or hcMemcpyDeviceToDevice. 
     *  @p copyAcc : Specify which accelerator performs the copy operation.  The specified accelerator must have access to the source and dest pointers - either
     *               because the memory is allocated on those devices or because the accelerator has peer access to the memory.
     *               If copyAcc is nullptr, then the copy will be performed by the host.  In this case, the host accelerator must have access to both pointers.
     *               The copy operation will be performed by the specified engine but is not synchronized with respect to any operations on that device.  
     *
     */
    completion_future copy_async_ext(const void *src, void *dst, size_t size_bytes, 
                                     hcCommandKind copyDir, const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, 
                                     const hc::accelerator *copyAcc);

    /**
     * Compares "this" accelerator_view with the passed accelerator_view object
     * to determine if they represent the same underlying object.
     *
     * @param[in] other The accelerator_view object to be compared against.
     * @return A boolean value indicating whether the passed accelerator_view
     *         object is same as "this" accelerator_view.
     */
    bool operator==(const accelerator_view& other) const {
        return pQueue == other.pQueue;
    }

    /**
     * Compares "this" accelerator_view with the passed accelerator_view object
     * to determine if they represent different underlying objects.
     *
     * @param[in] other The accelerator_view object to be compared against.
     * @return A boolean value indicating whether the passed accelerator_view
     *         object is different from "this" accelerator_view.
     */
    bool operator!=(const accelerator_view& other) const { return !(*this == other); }

    /**
     * Returns the maximum size of tile static area available on this
     * accelerator view.
     */
    size_t get_max_tile_static_size() {
        return pQueue.get()->getDev()->GetMaxTileStaticSize();
    }

    /**
     * Returns the number of pending asynchronous operations on this
     * accelerator view.
     *
     * Care must be taken to use this API in a thread-safe manner,
     */
    int get_pending_async_ops() {
        return pQueue->getPendingAsyncOps();
    }

    /**
     * Returns true if the accelerator_view is currently empty.
     *
     * Care must be taken to use this API in a thread-safe manner.
     * As the accelerator completes work, the queue may become empty
     * after this function returns false;
     */
    bool get_is_empty() {
        return pQueue->isEmpty();
    }

    /**
     * Returns an opaque handle which points to the underlying HSA queue.
     *
     * @return An opaque handle of the underlying HSA queue, if the accelerator
     *         view is based on HSA.  NULL if otherwise.
     */
    void* get_hsa_queue() {
        return pQueue->getHSAQueue();
    }

    /**
     * Returns an opaque handle which points to the underlying HSA agent.
     *
     * @return An opaque handle of the underlying HSA agent, if the accelerator
     *         view is based on HSA.  NULL otherwise.
     */
    void* get_hsa_agent() {
        return pQueue->getHSAAgent();
    }

    /**
     * Returns an opaque handle which points to the AM region on the HSA agent.
     * This region can be used to allocate accelerator memory which is accessible from the 
     * specified accelerator.
     *
     * @return An opaque handle of the region, if the accelerator is based
     *         on HSA.  NULL otherwise.
     */
    void* get_hsa_am_region() {
        return pQueue->getHSAAMRegion();
    }


    /**
     * Returns an opaque handle which points to the AM system region on the HSA agent.
     * This region can be used to allocate system memory which is accessible from the 
     * specified accelerator.
     *
     * @return An opaque handle of the region, if the accelerator is based
     *         on HSA.  NULL otherwise.
     */
    void* get_hsa_am_system_region() {
        return pQueue->getHSAAMHostRegion();
    }

    /**
     * Returns an opaque handle which points to the AM system region on the HSA agent.
     * This region can be used to allocate finegrained system memory which is accessible from the 
     * specified accelerator.
     *
     * @return An opaque handle of the region, if the accelerator is based
     *         on HSA.  NULL otherwise.
     */
    void* get_hsa_am_finegrained_system_region() {
        return pQueue->getHSACoherentAMHostRegion();
    }

    /**
     * Returns an opaque handle which points to the Kernarg region on the HSA
     * agent.
     *
     * @return An opaque handle of the region, if the accelerator view is based
     *         on HSA.  NULL otherwise.
     */
    void* get_hsa_kernarg_region() {
        return pQueue->getHSAKernargRegion();
    }

    /**
     * Returns if the accelerator view is based on HSA.
     */
    bool is_hsa_accelerator() {
        return pQueue->hasHSAInterOp();
    }

    /**
     * Dispatch a kernel into the accelerator_view.
     *
     * This function is intended to provide a gateway to dispatch code objects, with 
     * some assistance from HCC.  Kernels are specified in the standard code object
     * format, and can be created from a varety of compiler tools including the 
     * assembler, offline cl compilers, or other tools.    The caller also
     * specifies the execution configuration and kernel arguments.    HCC 
     * will copy the kernel arguments into an appropriate segment and insert
     * the packet into the queue.   HCC will also automatically handle signal 
     * and kernarg allocation and deallocation for the command.
     *
     *  The kernel is dispatched asynchronously, and thus this API may return before the 
     *  kernel finishes executing.
     
     *  Kernels dispatched with this API may be interleaved with other copy and kernel
     *  commands generated from copy or parallel_for_each commands.  
     *  The kernel honors the execute_order associated with the accelerator_view.  
     *  Specifically, if execute_order is execute_in_order, then the kernel
     *  will wait for older data and kernel commands in the same queue before
     *  beginning execution.  If execute_order is execute_any_order, then the 
     *  kernel may begin executing without regards to the state of older kernels.  
     *  This call honors the packer barrier bit (1 << HSA_PACKET_HEADER_BARRIER) 
     *  if set in the aql.header field.  If set, this provides the same synchronization
     *  behaviora as execute_in_order for the command generated by this API.
     *
     * @p aql is an HSA-format "AQL" packet. The following fields must 
     * be set by the caller:
     *  aql.kernel_object 
     *  aql.group_segment_size : includes static + dynamic group size
     *  aql.private_segment_size 
     *  aql.grid_size_x, aql.grid_size_y, aql.grid_size_z
     *  aql.group_size_x, aql.group_size_y, aql.group_size_z
     *  aql.setup :  The 2 bits at HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS.
     *  aql.header :  Must specify the desired memory fence operations, and barrier bit (if desired.).  A typical conservative setting would be:
    aql.header = (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                 (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE) |
                 (1 << HSA_PACKET_HEADER_BARRIER);

     * The following fields are ignored.  The API will will set up these fields before dispatching the AQL packet:
     *  aql.completion_signal 
     *  aql.kernarg 
     * 
     * @p args : Pointer to kernel arguments with the size and aligment expected by the kernel.  The args are copied and then passed directly to the kernel.   After this function returns, the args memory may be deallocated.
     * @p argSz : Size of the arguments.
     * @p cf : Written with a completion_future that can be used to track the status
     *          of the dispatch.  May be NULL, in which case no completion_future is 
     *          returned and the caller must use other synchronization techniqueues 
     *          such as calling accelerator_view::wait() or waiting on a younger command
     *          in the same queue.
     * @p kernel_name : Optionally specify the name of the kernel for debug and profiling.  
     * May be null.  If specified, the caller is responsible for ensuring the memory for the name remains allocated until the kernel completes.
     *        
     *
     * The dispatch_hsa_kernel call will perform the following operations:
     *    - Efficiently allocate a kernarg region and copy the arguments.
     *    - Efficiently allocate a signal, if required.
     *    - Dispatch the command into the queue and flush it to the GPU.
     *    - Kernargs and signals are automatically reclaimed by the HCC runtime.
     */
    void dispatch_hsa_kernel(const hsa_kernel_dispatch_packet_t *aql, 
                           const void * args, size_t argsize,
                           hc::completion_future *cf=nullptr, const char *kernel_name = nullptr) 
    {
        pQueue->dispatch_hsa_kernel(aql, args, argsize, cf, kernel_name);
    }

    /**
     * Set a CU affinity to specific command queues. 
     * The setting is permanent until the queue is destroyed or CU affinity is
     * set again. This setting is "atomic", it won't affect the dispatch in flight. 
     *
     * @param cu_mask a bool vector to indicate what CUs you want to use. True
     *        represents using the cu. The first 32 elements represents the first
     *        32 CUs, and so on. If its size is greater than physical CU number,
     *        the extra elements are ignored.
     *        It is user's responsibility to make sure the input is meaningful.
     *
     * @return true if operations succeeds or false if not.
     *
     */
     bool set_cu_mask(const std::vector<bool>& cu_mask) {
        // If it is HSA based accelerator view, set cu mask, otherwise, return;
        if(is_hsa_accelerator()) {
            return pQueue->set_cu_mask(cu_mask);
        }
        return false;
     }

private:
    accelerator_view(std::shared_ptr<Kalmar::KalmarQueue> pQueue) : pQueue(pQueue) {}
    std::shared_ptr<Kalmar::KalmarQueue> pQueue;

    friend class accelerator;
};

// ------------------------------------------------------------------------
// accelerator
// ------------------------------------------------------------------------

/**
 * Represents a physical accelerated computing device. An object of
 * this type can be created by enumerating the available devices, or
 * getting the default device.
 */
class accelerator
{
public:
    /**
     * Constructs a new accelerator object that represents the default
     * accelerator. This is equivalent to calling the constructor 
     * @code{.cpp}
     * accelerator(accelerator::default_accelerator)
     * @endcode
     *
     * The actual accelerator chosen as the default can be affected by calling
     * accelerator::set_default().
     */
    accelerator() : accelerator(L"default") {}

    /**
     * Constructs a new accelerator object that represents the physical device
     * named by the "path" argument. If the path represents an unknown or
     * unsupported device, an exception will be thrown.
     *
     * The path can be one of the following:
     * 1. accelerator::default_accelerator (or L"default"), which represents the
     *    path of the fastest accelerator available, as chosen by the runtime.
     * 2. accelerator::cpu_accelerator (or L"cpu"), which represents the CPU.
     *    Note that parallel_for_each shall not be invoked over this accelerator.
     * 3. A valid device path that uniquely identifies a hardware accelerator
     *    available on the host system.
     *
     * @param[in] path The device path of this accelerator.
     */
    explicit accelerator(const std::wstring& path)
        : pDev(Kalmar::getContext()->getDevice(path)) {}

    /**
     * Copy constructs an accelerator object. This function does a shallow copy
     * with the newly created accelerator object pointing to the same underlying
     * device as the passed accelerator parameter.
     *
     * @param[in] other The accelerator object to be copied.
     */
    accelerator(const accelerator& other) : pDev(other.pDev) {}

    /**
     * Returns a std::vector of accelerator objects (in no specific
     * order) representing all accelerators that are available, including
     * reference accelerators and WARP accelerators if available.
     *
     * @return A vector of accelerators.
     */
    static std::vector<accelerator> get_all() {
        auto Devices = Kalmar::getContext()->getDevices();
        std::vector<accelerator> ret;
        for(auto&& i : Devices)
          ret.push_back(i);
        return ret;
    }

    /**
     * Sets the default accelerator to the device path identified by the "path"
     * argument. See the constructor accelerator(const std::wstring& path)
     * for a description of the allowable path strings.
     *
     * This establishes a process-wide default accelerator and influences all
     * subsequent operations that might use a default accelerator.
     *
     * @param[in] path The device path of the default accelerator.
     * @return A Boolean flag indicating whether the default was set. If the
     *         default has already been set for this process, this value will be
     *         false, and the function will have no effect.
     */
    static bool set_default(const std::wstring& path) {
        return Kalmar::getContext()->set_default(path);
    }

    /**
     * Returns an accelerator_view which when passed as the first argument to a
     * parallel_for_each call causes the runtime to automatically select the
     * target accelerator_view for executing the parallel_for_each kernel. In
     * other words, a parallel_for_each invocation with the accelerator_view
     * returned by get_auto_selection_view() is the same as a parallel_for_each
     * invocation without an accelerator_view argument.
     *
     * For all other purposes, the accelerator_view returned by
     * get_auto_selection_view() behaves the same as the default accelerator_view
     * of the default accelerator (aka accelerator().get_default_view() ).
     *
     * @return An accelerator_view than can be used to indicate auto selection
     *         of the target for a parallel_for_each execution.
     */
    static accelerator_view get_auto_selection_view() {
        return Kalmar::getContext()->auto_select();
    }

    /**
     * Assigns an accelerator object to "this" accelerator object and returns a
     * reference to "this" object. This function does a shallow assignment with
     * the newly created accelerator object pointing to the same underlying
     * device as the passed accelerator parameter.
     *
     * @param other The accelerator object to be assigned from.
     * @return A reference to "this" accelerator object.
     */
    accelerator& operator=(const accelerator& other) {
        pDev = other.pDev;
        return *this;
    }

    /**
     * Returns the default accelerator_view associated with the accelerator.
     * The queuing_mode of the default accelerator_view is queuing_mode_automatic.
     *
     * @return The default accelerator_view object associated with the accelerator.
     */
    accelerator_view get_default_view() const { return pDev->get_default_queue(); }

    /**
     * Creates and returns a new accelerator view on the accelerator with the
     * supplied queuing mode.
     *
     * @param[in] qmode The queuing mode of the accelerator_view to be created.
     *                  See "Queuing Mode". The default value would be
     *                  queueing_mdoe_automatic if not specified.
     */
    accelerator_view create_view(execute_order order = execute_in_order, queuing_mode mode = queuing_mode_automatic) {
        auto pQueue = pDev->createQueue(order);
        pQueue->set_mode(mode);
        return pQueue;
    }
  
    /**
     * Compares "this" accelerator with the passed accelerator object to
     * determine if they represent the same underlying device.
     *
     * @param[in] other The accelerator object to be compared against.
     * @return A boolean value indicating whether the passed accelerator
     *         object is same as "this" accelerator.
     */
    bool operator==(const accelerator& other) const { return pDev == other.pDev; }

    /**
     * Compares "this" accelerator with the passed accelerator object to
     * determine if they represent different devices.
     *
     * @param[in] other The accelerator object to be compared against.
     * @return A boolean value indicating whether the passed accelerator
     *         object is different from "this" accelerator.
     */
    bool operator!=(const accelerator& other) const { return !(*this == other); }

    /**
     * Sets the default_cpu_access_type for this accelerator.
     *
     * The default_cpu_access_type is used for arrays created on this
     * accelerator or for implicit array_view memory allocations accessed on
     * this this accelerator.
     *
     * This method only succeeds if the default_cpu_access_type for the
     * accelerator has not already been overriden by a previous call to this 
     * method and the runtime selected default_cpu_access_type for this 
     * accelerator has not yet been used for allocating an array or for an 
     * implicit array_view memory allocation on this accelerator.
     *
     * @param[in] default_cpu_access_type The default cpu access_type to be used
     *            for array/array_view memory allocations on this accelerator.
     * @return A boolean value indicating if the default cpu access_type for the
     *         accelerator was successfully set.
     */
    bool set_default_cpu_access_type(access_type type) {
        pDev->set_access(type);
        return true;
    }

    /**
     * Returns a system-wide unique device instance path that matches the
     * "Device Instance Path" property for the device in Device Manager, or one
     * of the predefined path constants cpu_accelerator.
     */
    std::wstring get_device_path() const { return pDev->get_path(); }

    /**
     * Returns a short textual description of the accelerator device.
     */
    std::wstring get_description() const { return pDev->get_description(); }

    /**
     * Returns a 32-bit unsigned integer representing the version number of this
     * accelerator. The format of the integer is major.minor, where the major
     * version number is in the high-order 16 bits, and the minor version number
     * is in the low-order bits.
     */
    unsigned int get_version() const { return pDev->get_version(); }

    /**
     * This property indicates that the accelerator may be shared by (and thus
     * have interference from) the operating system or other system software
     * components for rendering purposes. A C++ AMP implementation may set this
     * property to false should such interference not be applicable for a
     * particular accelerator.
     */
    // FIXME: dummy implementation now
    bool get_has_display() const { return false; }

    /**
     * Returns the amount of dedicated memory (in KB) on an accelerator device.
     * There is no guarantee that this amount of memory is actually available to
     * use.
     */
    size_t get_dedicated_memory() const { return pDev->get_mem(); }

    /**
     * Returns a Boolean value indicating whether this accelerator supports
     * double-precision (double) computations. When this returns true,
     * supports_limited_double_precision also returns true.
     */
    bool get_supports_double_precision() const { return pDev->is_double(); }

    /**
     * Returns a boolean value indicating whether the accelerator has limited
     * double precision support (excludes double division, precise_math
     * functions, int to double, double to int conversions) for a
     * parallel_for_each kernel.
     */
    bool get_supports_limited_double_precision() const { return pDev->is_lim_double(); }

    /**
     * Returns a boolean value indicating whether the accelerator supports
     * debugging.
     */
    // FIXME: dummy implementation now
    bool get_is_debug() const { return false; }

    /**
     * Returns a boolean value indicating whether the accelerator is emulated.
     * This is true, for example, with the reference, WARP, and CPU accelerators.
     */
    bool get_is_emulated() const { return pDev->is_emulated(); }

    /**
     * Returns a boolean value indicating whether the accelerator supports memory
     * accessible both by the accelerator and the CPU.
     */
    bool get_supports_cpu_shared_memory() const { return pDev->is_unified(); }

    /**
     * Get the default cpu access_type for buffers created on this accelerator
     */
    access_type get_default_cpu_access_type() const { return pDev->get_access(); }
  
  
    /**
     * Returns the maximum size of tile static area available on this
     * accelerator.
     */
    size_t get_max_tile_static_size() {
      return get_default_view().get_max_tile_static_size();
    }
  
    /**
     * Returns a vector of all accelerator_view associated with this accelerator.
     */
    std::vector<accelerator_view> get_all_views() {
        std::vector<accelerator_view> result;
        std::vector< std::shared_ptr<Kalmar::KalmarQueue> > queues = pDev->get_all_queues();
        for (auto q : queues) {
            result.push_back(q);
        }
        return result;
    }

    /**
     * Returns an opaque handle which points to the AM region on the HSA agent.
     * This region can be used to allocate accelerator memory which is accessible from the 
     * specified accelerator.
     *
     * @return An opaque handle of the region, if the accelerator is based
     *         on HSA.  NULL otherwise.
     */
    void* get_hsa_am_region() const {
        return get_default_view().get_hsa_am_region();
    }

    /**
     * Returns an opaque handle which points to the AM system region on the HSA agent.
     * This region can be used to allocate system memory which is accessible from the 
     * specified accelerator.
     *
     * @return An opaque handle of the region, if the accelerator is based
     *         on HSA.  NULL otherwise.
     */
    void* get_hsa_am_system_region() const {
        return get_default_view().get_hsa_am_system_region();
    }

    /**
     * Returns an opaque handle which points to the AM system region on the HSA agent.
     * This region can be used to allocate finegrained system memory which is accessible from the 
     * specified accelerator.
     *
     * @return An opaque handle of the region, if the accelerator is based
     *         on HSA.  NULL otherwise.
     */
    void* get_hsa_am_finegrained_system_region() const {
        return get_default_view().get_hsa_am_finegrained_system_region();
    }

    /**
     * Returns an opaque handle which points to the Kernarg region on the HSA
     * agent.
     *
     * @return An opaque handle of the region, if the accelerator is based
     *         on HSA.  NULL otherwise.
     */
    void* get_hsa_kernarg_region() const {
        return get_default_view().get_hsa_kernarg_region();
    }

    /**
     * Returns if the accelerator is based on HSA.
     */
    bool is_hsa_accelerator() const {
        return get_default_view().is_hsa_accelerator();
    }

    /**
     * Returns the profile the accelerator.
     * - hcAgentProfileNone in case the accelerator is not based on HSA.
     * - hcAgentProfileBase in case the accelerator is of HSA Base Profile.
     * - hcAgentProfileFull in case the accelerator is of HSA Full Profile.
     */
    hcAgentProfile get_profile() const {
        return pDev->getProfile();
    }

    void memcpy_symbol(const char* symbolName, void* hostptr, size_t count, size_t offset = 0, hcCommandKind kind = hcMemcpyHostToDevice) {
        pDev->memcpySymbol(symbolName, hostptr, count, offset, kind);
    }

    void memcpy_symbol(void* symbolAddr, void* hostptr, size_t count, size_t offset = 0, hcCommandKind kind = hcMemcpyHostToDevice) {
        pDev->memcpySymbol(symbolAddr, hostptr, count, offset, kind);
    }

    void* get_symbol_address(const char* symbolName) {
        return pDev->getSymbolAddress(symbolName);
    }

    /**
     * Returns an opaque handle which points to the underlying HSA agent.
     *
     * @return An opaque handle of the underlying HSA agent, if the accelerator
     *         is based on HSA.  NULL otherwise.
     */
    void* get_hsa_agent() const {
        return pDev->getHSAAgent();
    }

    /**
     * Check if @p other is peer of this accelerator.
     *
     * @return true if other can access this accelerator's device memory pool or false if not.
     * The acceleratos is not its own peer.
     */
    bool get_is_peer(const accelerator& other) const {
        return pDev->is_peer(other.pDev);
    }
      
    /**
     * Return a std::vector of this accelerator's peers. peer is other accelerator which can access this 
     * accelerator's device memory using map_to_peer family of APIs.
     *
     */
    std::vector<accelerator> get_peers() const {
        std::vector<accelerator> peers;

        const auto &accs = get_all();

        for(auto iter = accs.begin(); iter != accs.end(); iter++)
        {
            if(this->get_is_peer(*iter))
                peers.push_back(*iter);
        }
        return peers;
    }

    /**
     * Return the compute unit count of the accelerator.
     *
     */
    unsigned int get_cu_count() const {
        return pDev->get_compute_unit_count();
    }

    /**
     * Return the unique integer sequence-number for the accelerator.
     * Sequence-numbers are assigned in monotonically increasing order starting with 0.
     */
    int get_seqnum() const {
        return pDev->get_seqnum();
    }


    /**
     * Return true if the accelerator's memory can be mapped into the CPU's address space,
     * and the CPU is allowed to access the memory directly with CPU memory operations.
     * Typically this is enabled with "large BAR" or "resizeable BAR" address mapping.
     *
     */
    bool has_cpu_accessible_am() {
        return pDev->has_cpu_accessible_am();
    };

    Kalmar::KalmarDevice *get_dev_ptr() const { return pDev; }; 

private:
    accelerator(Kalmar::KalmarDevice* pDev) : pDev(pDev) {}
    friend class accelerator_view;
    Kalmar::KalmarDevice* pDev;
};

// ------------------------------------------------------------------------
// completion_future
// ------------------------------------------------------------------------

/**
 * This class is the return type of all asynchronous APIs and has an interface
 * analogous to std::shared_future<void>. Similar to std::shared_future, this
 * type provides member methods such as wait and get to wait for asynchronous
 * operations to finish, and the type additionally provides a member method
 * then(), to specify a completion callback functor to be executed upon
 * completion of an asynchronous operation.
 */
class completion_future {
public:

    /**
     * Default constructor. Constructs an empty uninitialized completion_future
     * object which does not refer to any asynchronous operation. Default
     * constructed completion_future objects have valid() == false
     */
    completion_future() : __amp_future(), __thread_then(nullptr), __asyncOp(nullptr) {};

    /**
     * Copy constructor. Constructs a new completion_future object that referes
     * to the same asynchronous operation as the other completion_future object.
     *
     * @param[in] other An object of type completion_future from which to
     *                  initialize this.
     */
    completion_future(const completion_future& other)
        : __amp_future(other.__amp_future), __thread_then(other.__thread_then), __asyncOp(other.__asyncOp) {}

    /**
     * Move constructor. Move constructs a new completion_future object that
     * referes to the same asynchronous operation as originally refered by the
     * other completion_future object. After this constructor returns,
     * other.valid() == false
     *
     * @param[in] other An object of type completion_future which the new
     *                  completion_future
     */
    completion_future(completion_future&& other)
        : __amp_future(std::move(other.__amp_future)), __thread_then(other.__thread_then), __asyncOp(other.__asyncOp) {}

    /**
     * Copy assignment. Copy assigns the contents of other to this. This method
     * causes this to stop referring its current asynchronous operation and
     * start referring the same asynchronous operation as other.
     *
     * @param[in] other An object of type completion_future which is copy
     *                  assigned to this.
     */
    completion_future& operator=(const completion_future& _Other) {
        if (this != &_Other) {
           __amp_future = _Other.__amp_future;
           __thread_then = _Other.__thread_then;
           __asyncOp = _Other.__asyncOp;
        }
        return (*this);
    }

    /**
     * Move assignment. Move assigns the contents of other to this. This method
     * causes this to stop referring its current asynchronous operation and
     * start referring the same asynchronous operation as other. After this
     * method returns, other.valid() == false
     *
     * @param[in] other An object of type completion_future which is move
     *                  assigned to this.
     */
    completion_future& operator=(completion_future&& _Other) {
        if (this != &_Other) {
            __amp_future = std::move(_Other.__amp_future);
            __thread_then = _Other.__thread_then;
           __asyncOp = _Other.__asyncOp;
        }
        return (*this);
    }

    /**
     * This method is functionally identical to std::shared_future<void>::get.
     * This method waits for the associated asynchronous operation to finish
     * and returns only upon the completion of the asynchronous operation. If
     * an exception was encountered during the execution of the asynchronous
     * operation, this method throws that stored exception.
     */
    void get() const {
        __amp_future.get();
    }

    /**
     * This method is functionally identical to
     * std::shared_future<void>::valid. This returns true if this
     * completion_future is associated with an asynchronous operation.
     */
    bool valid() const {
        return __amp_future.valid();
    }

    /** @{ */
    /**
     * These methods are functionally identical to the corresponding
     * std::shared_future<void> methods.
     *
     * The wait method waits for the associated asynchronous operation to
     * finish and returns only upon completion of the associated asynchronous
     * operation or if an exception was encountered when executing the
     * asynchronous operation.
     *
     * The other variants are functionally identical to the
     * std::shared_future<void> member methods with same names.
     *
     * @param waitMode[in] An optional parameter to specify the wait mode. By
     *                     default it would be hcWaitModeBlocked.
     *                     hcWaitModeActive would be used to reduce latency with
     *                     the expense of using one CPU core for active waiting.
     */
    void wait(hcWaitMode mode = hcWaitModeBlocked) const {
        if (this->valid()) {
            if (__asyncOp != nullptr) {
                __asyncOp->setWaitMode(mode);
            }   
            //TODO-ASYNC - need to reclaim older AsyncOps here.
            __amp_future.wait();
        }

        Kalmar::getContext()->flushPrintfBuffer();
    }

    template <class _Rep, class _Period>
    std::future_status wait_for(const std::chrono::duration<_Rep, _Period>& _Rel_time) const {
        return __amp_future.wait_for(_Rel_time);
    }

    template <class _Clock, class _Duration>
    std::future_status wait_until(const std::chrono::time_point<_Clock, _Duration>& _Abs_time) const {
        return __amp_future.wait_until(_Abs_time);
    }

    /** @} */

    /**
     * Conversion operator to std::shared_future<void>. This method returns a
     * shared_future<void> object corresponding to this completion_future
     * object and refers to the same asynchronous operation.
     */
    operator std::shared_future<void>() const {
        return __amp_future;
    }

    /**
     * This method enables specification of a completion callback func which is
     * executed upon completion of the asynchronous operation associated with
     * this completion_future object. The completion callback func should have
     * an operator() that is valid when invoked with non arguments, i.e., "func()".
     */
    // FIXME: notice we removed const from the signature here
    //        the original signature in the specification should be
    //        template<typename functor>
    //        void then(const functor& func) const;
    template<typename functor>
    void then(const functor & func) {
#if __HCC_ACCELERATOR__ != 1
      // could only assign once
      if (__thread_then == nullptr) {
        // spawn a new thread to wait on the future and then execute the callback functor
        __thread_then = new std::thread([&]() __CPU__ {
          this->wait();
          if(this->valid())
            func();
        });
      }
#endif
    }

    /**
     * Get the native handle for the asynchronous operation encapsulated in
     * this completion_future object. The method is mostly used for debugging
     * purpose.
     * Applications should retain the parent completion_future to ensure
     * the native handle is not deallocated by the HCC runtime.  The completion_future
     * pointer to the native handle is reference counted, so a copy of 
     * the completion_future is sufficient to retain the native_handle.
     */
    void* get_native_handle() const {
      if (__asyncOp != nullptr) {
        return __asyncOp->getNativeHandle();
      } else {
        return nullptr;
      }
    }

    /**
     * Get the tick number when the underlying asynchronous operation begins.
     *
     * @return An implementation-defined tick number in case the instance is
     *         created by a kernel dispatch or a barrier packet. 0 otherwise.
     */
    uint64_t get_begin_tick() {
      if (__asyncOp != nullptr) {
        return __asyncOp->getBeginTimestamp();
      } else {
        return 0L;
      }
    }

    /**
     * Get the tick number when the underlying asynchronous operation ends.
     *
     * @return An implementation-defined tick number in case the instance is
     *         created by a kernel dispatch or a barrier packet. 0 otherwise.
     */
    uint64_t get_end_tick() {
      if (__asyncOp != nullptr) {
        return __asyncOp->getEndTimestamp();
      } else {
        return 0L;
      }
    }

    /**
     * Get the frequency of ticks per second for the underlying asynchrnous operation.
     *
     * @return An implementation-defined frequency in Hz in case the instance is
     *         created by a kernel dispatch or a barrier packet. 0 otherwise.
     */
    uint64_t get_tick_frequency() {
      if (__asyncOp != nullptr) {
        return __asyncOp->getTimestampFrequency();
      } else {
        return 0L;
      }
    }

    /**
     * Get if the async operations has been completed.
     *
     * @return True if the async operation has been completed, false if not.
     */
    bool is_ready() {
      if (__asyncOp != nullptr) {
        return __asyncOp->isReady();
      } else {
        return false;
      }
    }

    ~completion_future() {
      if (__thread_then != nullptr) {
        __thread_then->join();
      }
      delete __thread_then;
      __thread_then = nullptr;
      
      if (__asyncOp != nullptr) {
        __asyncOp = nullptr;
      }
    }


    /**
     * @return reference count for the completion future.  Primarily used for debug purposes.
     */
    int get_use_count() const { return __asyncOp.use_count(); };

private:
    std::shared_future<void> __amp_future;
    std::thread* __thread_then = nullptr;
    std::shared_ptr<Kalmar::KalmarAsyncOp> __asyncOp;

    completion_future(std::shared_ptr<Kalmar::KalmarAsyncOp> event) : __amp_future(*(event->getFuture())), __asyncOp(event) {}

    completion_future(const std::shared_future<void> &__future)
        : __amp_future(__future), __thread_then(nullptr), __asyncOp(nullptr) {}

    friend class Kalmar::HSAQueue;

    // accelerator_view
    friend class accelerator_view;
};

// ------------------------------------------------------------------------
// member function implementations
// ------------------------------------------------------------------------

inline accelerator
accelerator_view::get_accelerator() const { return pQueue->getDev(); }

inline completion_future
accelerator_view::create_marker(memory_scope scope) const {
    std::shared_ptr<Kalmar::KalmarAsyncOp> deps[1]; 
    // If necessary create an explicit dependency on previous command
    // This is necessary for example if copy command is followed by marker - we need the marker to wait for the copy to complete.
    std::shared_ptr<Kalmar::KalmarAsyncOp> depOp = pQueue->detectStreamDeps(hcCommandMarker, nullptr);

    int cnt = 0;
    if (depOp) {
        deps[cnt++] = depOp; // retrieve async op associated with completion_future
    }

    return completion_future(pQueue->EnqueueMarkerWithDependency(cnt, deps, scope));
}

inline unsigned int accelerator_view::get_version() const { return get_accelerator().get_version(); }

inline completion_future accelerator_view::create_blocking_marker(completion_future& dependent_future, memory_scope scope) const {
    std::shared_ptr<Kalmar::KalmarAsyncOp> deps[2]; 

    // If necessary create an explicit dependency on previous command
    // This is necessary for example if copy command is followed by marker - we need the marker to wait for the copy to complete.
    std::shared_ptr<Kalmar::KalmarAsyncOp> depOp = pQueue->detectStreamDeps(hcCommandMarker, nullptr);

    int cnt = 0;
    if (depOp) {
        deps[cnt++] = depOp; // retrieve async op associated with completion_future
    }

    if (dependent_future.__asyncOp) {
        deps[cnt++] = dependent_future.__asyncOp; // retrieve async op associated with completion_future
    } 
    
    return completion_future(pQueue->EnqueueMarkerWithDependency(cnt, deps, scope));
}

template<typename InputIterator>
inline completion_future
accelerator_view::create_blocking_marker(InputIterator first, InputIterator last, memory_scope scope) const {
    std::shared_ptr<Kalmar::KalmarAsyncOp> deps[5]; // array of 5 pointers to the native handle of async ops. 5 is the max supported by barrier packet
    hc::completion_future lastMarker;


    // If necessary create an explicit dependency on previous command
    // This is necessary for example if copy command is followed by marker - we need the marker to wait for the copy to complete.
    std::shared_ptr<Kalmar::KalmarAsyncOp> depOp = pQueue->detectStreamDeps(hcCommandMarker, nullptr);

    int cnt = 0;
    if (depOp) {
        deps[cnt++] = depOp; // retrieve async op associated with completion_future
    }


    // loop through signals and group into sections of 5
    // every 5 signals goes into one barrier packet
    // since HC sets the barrier bit in each AND barrier packet, we know
    // the barriers will execute in-order
    for (auto iter = first; iter != last; ++iter) {
        if (iter->__asyncOp) {
            deps[cnt++] = iter->__asyncOp; // retrieve async op associated with completion_future
            if (cnt == 5) {
                lastMarker = completion_future(pQueue->EnqueueMarkerWithDependency(cnt, deps, hc::no_scope));
                cnt = 0;
            }
        }
    }

    if (cnt) {
        lastMarker = completion_future(pQueue->EnqueueMarkerWithDependency(cnt, deps, scope));
    }

    return lastMarker;
}

inline completion_future
accelerator_view::create_blocking_marker(std::initializer_list<completion_future> dependent_future_list, memory_scope scope) const {
    return create_blocking_marker(dependent_future_list.begin(), dependent_future_list.end(), scope);
}


inline void accelerator_view::copy_ext(const void *src, void *dst, size_t size_bytes, hcCommandKind copyDir, const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, const hc::accelerator *copyAcc, bool forceUnpinnedCopy) {
    pQueue->copy_ext(src, dst, size_bytes, copyDir, srcInfo, dstInfo, copyAcc ? copyAcc->pDev : nullptr, forceUnpinnedCopy);
};

inline void accelerator_view::copy_ext(const void *src, void *dst, size_t size_bytes, hcCommandKind copyDir, const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, bool forceHostCopyEngine) {
    pQueue->copy_ext(src, dst, size_bytes, copyDir, srcInfo, dstInfo, forceHostCopyEngine);
};

inline completion_future
accelerator_view::copy_async(const void *src, void *dst, size_t size_bytes) {
    return completion_future(pQueue->EnqueueAsyncCopy(src, dst, size_bytes));
}

inline completion_future
accelerator_view::copy_async_ext(const void *src, void *dst, size_t size_bytes,
                             hcCommandKind copyDir, 
                             const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, 
                             const hc::accelerator *copyAcc)
{
    return completion_future(pQueue->EnqueueAsyncCopyExt(src, dst, size_bytes, copyDir, srcInfo, dstInfo, copyAcc ? copyAcc->pDev : nullptr));
};

// ------------------------------------------------------------------------
// Intrinsic functions for HSAIL instructions
// ------------------------------------------------------------------------

/**
 * Fetch the size of a wavefront
 *
 * @return The size of a wavefront.
 */
#define __HSA_WAVEFRONT_SIZE__ (64)
extern "C" unsigned int __wavesize() __HC__; 


#if __hcc_backend__==HCC_BACKEND_AMDGPU
extern "C" inline unsigned int __wavesize() __HC__ {
  return __HSA_WAVEFRONT_SIZE__;
}
#endif

/**
 * Count number of 1 bits in the input
 *
 * @param[in] input An unsinged 32-bit integer.
 * @return Number of 1 bits in the input.
 */
extern "C" inline unsigned int __popcount_u32_b32(unsigned int input) __HC__ {
  return __builtin_popcount(input);
}

/**
 * Count number of 1 bits in the input
 *
 * @param[in] input An unsinged 64-bit integer.
 * @return Number of 1 bits in the input.
 */
extern "C" inline unsigned int __popcount_u32_b64(unsigned long long int input) __HC__ {
  return __builtin_popcountl(input);
}

/** @{ */
/**
 * Extract a range of bits
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a> for more detailed specification of these functions.
 */
extern "C" inline unsigned int __bitextract_u32(unsigned int src0, unsigned int src1, unsigned int src2) __HC__ {
  uint32_t offset = src1 & 31;
  uint32_t width = src2 & 31;
  return width == 0 ? 0 : (src0 << (32 - offset - width)) >> (32 - width);
}

extern "C" inline uint64_t __bitextract_u64(uint64_t src0, unsigned int src1, unsigned int src2) __HC__ {
  uint64_t offset = src1 & 63;
  uint64_t width = src2 & 63;
  return width == 0 ? 0 : (src0 << (64 - offset - width)) >> (64 - width);
}

extern "C" int __bitextract_s32(int src0, unsigned int src1, unsigned int src2) __HC__;

extern "C" int64_t __bitextract_s64(int64_t src0, unsigned int src1, unsigned int src2) __HC__;
/** @} */

/** @{ */
/**
 * Replace a range of bits
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a> for more detailed specification of these functions.
 */
extern "C" inline unsigned int __bitinsert_u32(unsigned int src0, unsigned int src1, unsigned int src2, unsigned int src3) __HC__ {
  uint32_t offset = src2 & 31;
  uint32_t width = src3 & 31;
  uint32_t mask = (1 << width) - 1;
  return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

extern "C" inline uint64_t __bitinsert_u64(uint64_t src0, uint64_t src1, unsigned int src2, unsigned int src3) __HC__ {
  uint64_t offset = src2 & 63;
  uint64_t width = src3 & 63;
  uint64_t mask = (1 << width) - 1;
  return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

extern "C" int __bitinsert_s32(int src0, int src1, unsigned int src2, unsigned int src3) __HC__;

extern "C" int64_t __bitinsert_s64(int64_t src0, int64_t src1, unsigned int src2, unsigned int src3) __HC__;
/** @} */

/** @{ */
/**
 * Create a bit mask that can be used with bitselect
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a> for more detailed specification of these functions.
 */
extern "C" unsigned int __bitmask_b32(unsigned int src0, unsigned int src1) __HC__;

extern "C" uint64_t __bitmask_b64(unsigned int src0, unsigned int src1) __HC__;
/** @} */

/** @{ */
/**
 * Reverse the bits
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a> for more detailed specification of these functions.
 */

unsigned int __bitrev_b32(unsigned int src0) [[hc]] __asm("llvm.bitreverse.i32");

uint64_t __bitrev_b64(uint64_t src0) [[hc]] __asm("llvm.bitreverse.i64");

/** @} */

/** @{ */
/**
 * Do bit field selection
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a> for more detailed specification of these functions.
 */
extern "C" inline unsigned int __bitselect_b32(unsigned int src0, unsigned int src1, unsigned int src2) __HC__ {
  return (src1 & src0) | (src2 & ~src0);
}

extern "C" inline uint64_t __bitselect_b64(uint64_t src0, uint64_t src1, uint64_t src2) __HC__ {
  return (src1 & src0) | (src2 & ~src0);
}
/** @} */

/**
 * Count leading zero bits in the input
 *
 * @param[in] input An unsigned 32-bit integer.
 * @return Number of 0 bits until a 1 bit is found, counting start from the
 *         most significant bit. -1 if there is no 0 bit.
 */
extern "C" inline unsigned int __firstbit_u32_u32(unsigned int input) __HC__ {
  return input == 0 ? -1 : __builtin_clz(input);
}


/**
 * Count leading zero bits in the input
 *
 * @param[in] input An unsigned 64-bit integer.
 * @return Number of 0 bits until a 1 bit is found, counting start from the
 *         most significant bit. -1 if there is no 0 bit.
 */
extern "C" inline unsigned int __firstbit_u32_u64(unsigned long long int input) __HC__ {
  return input == 0 ? -1 : __builtin_clzl(input);
}

/**
 * Count leading zero bits in the input
 *
 * @param[in] input An signed 32-bit integer.
 * @return Finds the first bit set in a positive integer starting from the
 *         most significant bit, or finds the first bit clear in a negative
 *         integer from the most significant bit.
 *         If no bits in the input are set, then dest is set to -1.
 */
extern "C" inline unsigned int __firstbit_u32_s32(int input) __HC__ {
  if (input == 0) {
    return -1;
  }

  return input > 0 ? __firstbit_u32_u32(input) : __firstbit_u32_u32(~input);
}


/**
 * Count leading zero bits in the input
 *
 * @param[in] input An signed 64-bit integer.
 * @return Finds the first bit set in a positive integer starting from the
 *         most significant bit, or finds the first bit clear in a negative
 *         integer from the most significant bit.
 *         If no bits in the input are set, then dest is set to -1.
 */
extern "C" inline unsigned int __firstbit_u32_s64(long long int input) __HC__ {
  if (input == 0) {
    return -1;
  }

  return input > 0 ? __firstbit_u32_u64(input) : __firstbit_u32_u64(~input);
}

/** @{ */
/**
 * Find the first bit set to 1 in a number starting from the
 * least significant bit
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a> for more detailed specification of these functions.
 */
extern "C" inline unsigned int __lastbit_u32_u32(unsigned int input) __HC__ {
  return input == 0 ? -1 : __builtin_ctz(input);
}

extern "C" inline unsigned int __lastbit_u32_u64(unsigned long long int input) __HC__ {
  return input == 0 ? -1 : __builtin_ctzl(input);
}

extern "C" inline unsigned int __lastbit_u32_s32(int input) __HC__ {
  return __lastbit_u32_u32(input);
}

extern "C" inline unsigned int __lastbit_u32_s64(unsigned long long input) __HC__ {
  return __lastbit_u32_u64(input);
}
/** @} */

/** @{ */
/**
 * Copy and interleave the lower half of the elements from
 * each source into the desitionation
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/packed_data.htm">HSA PRM 5.9</a> for more detailed specification of these functions.
 */
extern "C" unsigned int __unpacklo_u8x4(unsigned int src0, unsigned int src1) __HC__;

extern "C" uint64_t __unpacklo_u8x8(uint64_t src0, uint64_t src1) __HC__;

extern "C" unsigned int __unpacklo_u16x2(unsigned int src0, unsigned int src1) __HC__;

extern "C" uint64_t __unpacklo_u16x4(uint64_t src0, uint64_t src1) __HC__;

extern "C" uint64_t __unpacklo_u32x2(uint64_t src0, uint64_t src1) __HC__;

extern "C" int __unpacklo_s8x4(int src0, int src1) __HC__;

extern "C" int64_t __unpacklo_s8x8(int64_t src0, int64_t src1) __HC__;

extern "C" int __unpacklo_s16x2(int src0, int src1) __HC__;

extern "C" int64_t __unpacklo_s16x4(int64_t src0, int64_t src1) __HC__;

extern "C" int64_t __unpacklo_s32x2(int64_t src0, int64_t src1) __HC__;
/** @} */

/** @{ */
/**
 * Copy and interleave the upper half of the elements from
 * each source into the desitionation
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/packed_data.htm">HSA PRM 5.9</a> for more detailed specification of these functions.
 */
extern "C" unsigned int __unpackhi_u8x4(unsigned int src0, unsigned int src1) __HC__;

extern "C" uint64_t __unpackhi_u8x8(uint64_t src0, uint64_t src1) __HC__;

extern "C" unsigned int __unpackhi_u16x2(unsigned int src0, unsigned int src1) __HC__;

extern "C" uint64_t __unpackhi_u16x4(uint64_t src0, uint64_t src1) __HC__;

extern "C" uint64_t __unpackhi_u32x2(uint64_t src0, uint64_t src1) __HC__;

extern "C" int __unpackhi_s8x4(int src0, int src1) __HC__;

extern "C" int64_t __unpackhi_s8x8(int64_t src0, int64_t src1) __HC__;

extern "C" int __unpackhi_s16x2(int src0, int src1) __HC__;

extern "C" int64_t __unpackhi_s16x4(int64_t src0, int64_t src1) __HC__;

extern "C" int64_t __unpackhi_s32x2(int64_t src0, int64_t src1) __HC__;
/** @} */

/** @{ */
/**
 * Assign the elements of the packed value in src0, replacing
 * the element specified by src2 with the value from src1
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/packed_data.htm">HSA PRM 5.9</a> for more detailed specification of these functions.
 */
extern "C" unsigned int __pack_u8x4_u32(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;

extern "C" uint64_t __pack_u8x8_u32(uint64_t src0, unsigned int src1, unsigned int src2) __HC__;

extern "C" unsigned __pack_u16x2_u32(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;

extern "C" uint64_t __pack_u16x4_u32(uint64_t src0, unsigned int src1, unsigned int src2) __HC__;

extern "C" uint64_t __pack_u32x2_u32(uint64_t src0, unsigned int src1, unsigned int src2) __HC__;

extern "C" int __pack_s8x4_s32(int src0, int src1, unsigned int src2) __HC__;

extern "C" int64_t __pack_s8x8_s32(int64_t src0, int src1, unsigned int src2) __HC__;

extern "C" int __pack_s16x2_s32(int src0, int src1, unsigned int src2) __HC__;

extern "C" int64_t __pack_s16x4_s32(int64_t src0, int src1, unsigned int src2) __HC__;

extern "C" int64_t __pack_s32x2_s32(int64_t src0, int src1, unsigned int src2) __HC__;

extern "C" double __pack_f32x2_f32(double src0, float src1, unsigned int src2) __HC__;
/** @} */

/** @{ */
/**
 * Assign the elements specified by src1 from the packed value in src0
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/packed_data.htm">HSA PRM 5.9</a> for more detailed specification of these functions.
 */
extern "C" unsigned int __unpack_u32_u8x4(unsigned int src0, unsigned int src1) __HC__;

extern "C" unsigned int __unpack_u32_u8x8(uint64_t src0, unsigned int src1) __HC__;

extern "C" unsigned int __unpack_u32_u16x2(unsigned int src0, unsigned int src1) __HC__;

extern "C" unsigned int __unpack_u32_u16x4(uint64_t src0, unsigned int src1) __HC__;

extern "C" unsigned int __unpack_u32_u32x2(uint64_t src0, unsigned int src1) __HC__;

extern "C" int __unpack_s32_s8x4(int src0, unsigned int src1) __HC__;

extern "C" int __unpack_s32_s8x8(int64_t src0, unsigned int src1) __HC__;

extern "C" int __unpack_s32_s16x2(int src0, unsigned int src1) __HC__;

extern "C" int __unpack_s32_s16x4(int64_t src0, unsigned int src1) __HC__;

extern "C" int __unpack_s32_s3x2(int64_t src0, unsigned int src1) __HC__;

extern "C" float __unpack_f32_f32x2(double src0, unsigned int src1) __HC__;
/** @} */

/**
 * Align 32 bits within 64 bits of data on an arbitrary bit boundary
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a> for more detailed specification.
 */
extern "C" unsigned int __bitalign_b32(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;

/**
 * Align 32 bits within 64 bis of data on an arbitrary byte boundary
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a> for more detailed specification.
 */
extern "C" unsigned int __bytealign_b32(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;

/**
 * Do linear interpolation and computes the unsigned 8-bit average of packed
 * data
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a> for more detailed specification.
 */
extern "C" unsigned int __lerp_u8x4(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;

/**
 * Takes four floating-point number, convers them to
 * unsigned integer values, and packs them into a packed u8x4 value
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a> for more detailed specification.
 */
extern "C" unsigned int __packcvt_u8x4_f32(float src0, float src1, float src2, float src3) __HC__;

/**
 * Unpacks a single element from a packed u8x4 value and converts it to an f32.
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a> for more detailed specification.
 */
extern "C" float __unpackcvt_f32_u8x4(unsigned int src0, unsigned int src1) __HC__;

/** @{ */
/**
 * Computes the sum of the absolute differences of src0 and
 * src1 and then adds src2 to the result
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a> for more detailed specification.
 */
extern "C" unsigned int __sad_u32_u32(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;

extern "C" unsigned int __sad_u32_u16x2(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;

extern "C" unsigned int __sad_u32_u8x4(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;
/** @} */

/**
 * This function is mostly the same as sad except the sum of absolute
 * differences is added to the most significant 16 bits of the result
 *
 * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a> for more detailed specification.
 */
extern "C" unsigned int __sadhi_u16x2_u8x4(unsigned int src0, unsigned int src1, unsigned int src2) __HC__;

/**
 * Get system timestamp
 */
extern "C" uint64_t __clock_u64() __HC__;

/**
 * Get hardware cycle count
 *
 * Notice the return value of this function is implementation defined.
 */
extern "C" uint64_t __cycle_u64() __HC__;

/**
 * Get the count of the number of earlier (in flattened
 * work-item order) active work-items within the same wavefront.
 *
 * @return The result will be in the range 0 to WAVESIZE - 1.
 */
extern "C" unsigned int __activelaneid_u32() __HC__;

/**
 * Return a bit mask shows which active work-items in the
 * wavefront have a non-zero input. The affected bit position within the
 * registers of dest corresponds to each work-item's lane ID.
 *
 * The HSAIL instruction would return 4 64-bit registers but the current
 * implementation would only return the 1st one and ignore the other 3 as
 * right now all HSA agents have wavefront of size 64.
 *
 * @param[in] input An unsigned 32-bit integer.
 * @return The bitmask calculated.
 */
extern "C" uint64_t __activelanemask_v4_b64_b1(unsigned int input) __HC__;

/**
 * Count the number of active work-items in the current
 * wavefront that have a non-zero input.
 *
 * @param[in] input An unsigned 32-bit integer.
 * @return The number of active work-items in the current wavefront that have
 *         a non-zero input.
 */
extern "C" inline unsigned int __activelanecount_u32_b1(unsigned int input) __HC__ {
 return  __popcount_u32_b64(__activelanemask_v4_b64_b1(input));
}

// ------------------------------------------------------------------------
// Wavefront Vote Functions
// ------------------------------------------------------------------------

/**
 * Evaluate predicate for all active work-items in the
 * wavefront and return non-zero if and only if predicate evaluates to non-zero
 * for any of them.
 */
extern "C" bool __ockl_wfany_i32(int) __HC__;
extern "C" inline int __any(int predicate) __HC__ {
    return __ockl_wfany_i32(predicate);
}

/**
 * Evaluate predicate for all active work-items in the
 * wavefront and return non-zero if and only if predicate evaluates to non-zero
 * for all of them.
 */
extern "C" bool __ockl_wfall_i32(int) __HC__;
extern "C" inline int __all(int predicate) __HC__ {
    return __ockl_wfall_i32(predicate);
}

/**
 * Evaluate predicate for all active work-items in the
 * wavefront and return an integer whose Nth bit is set if and only if
 * predicate evaluates to non-zero for the Nth work-item of the wavefront and
 * the Nth work-item is active.
 */

// XXX from llvm/include/llvm/IR/InstrTypes.h
#define ICMP_NE 33
__attribute__((convergent))
unsigned long long __llvm_amdgcn_icmp_i32(uint x, uint y, uint z) [[hc]] __asm("llvm.amdgcn.icmp.i32");
extern "C" inline uint64_t __ballot(int predicate) __HC__ {
    return __llvm_amdgcn_icmp_i32(predicate, 0, ICMP_NE);
}

// ------------------------------------------------------------------------
// Wavefront Shuffle Functions
// ------------------------------------------------------------------------

// utility union type
union __u {
    int i;
    unsigned int u;
    float f;
};

/** @{ */
/**
 * Direct copy from indexed active work-item within a wavefront.
 *
 * Work-items may only read data from another work-item which is active in the
 * current wavefront. If the target work-item is inactive, the retrieved value
 * is fixed as 0.
 *
 * The function returns the value of var held by the work-item whose ID is given
 * by srcLane. If width is less than __HSA_WAVEFRONT_SIZE__ then each
 * subsection of the wavefront behaves as a separate entity with a starting
 * logical work-item ID of 0. If srcLane is outside the range [0:width-1], the
 * value returned corresponds to the value of var held by:
 * srcLane modulo width (i.e. within the same subsection).
 *
 * The optional width parameter must have a value which is a power of 2;
 * results are undefined if it is not a power of 2, or is number greater than
 * __HSA_WAVEFRONT_SIZE__.
 */

#if __hcc_backend__==HCC_BACKEND_AMDGPU

/*
 * FIXME: We need to add __builtin_amdgcn_mbcnt_{lo,hi} to clang and call
 * them here instead.
 */

int __amdgcn_mbcnt_lo(int mask, int src) [[hc]] __asm("llvm.amdgcn.mbcnt.lo");
int __amdgcn_mbcnt_hi(int mask, int src) [[hc]] __asm("llvm.amdgcn.mbcnt.hi");

inline int __lane_id(void) [[hc]] {
  int lo = __amdgcn_mbcnt_lo(-1, 0);
  return __amdgcn_mbcnt_hi(-1, lo);
}

#endif

#if __hcc_backend__==HCC_BACKEND_AMDGPU

/**
 * ds_bpermute intrinsic
 * FIXME: We need to add __builtin_amdgcn_ds_bpermute to clang and call it here
 * instead.
 */
int __amdgcn_ds_bpermute(int index, int src) [[hc]] __asm("llvm.amdgcn.ds.bpermute");
inline unsigned int __amdgcn_ds_bpermute(int index, unsigned int src) [[hc]] {
  __u tmp; tmp.u = src;
  tmp.i = __amdgcn_ds_bpermute(index, tmp.i);
  return tmp.u;
}
inline float __amdgcn_ds_bpermute(int index, float src) [[hc]] {
  __u tmp; tmp.f = src;
  tmp.i = __amdgcn_ds_bpermute(index, tmp.i);
  return tmp.f;
}

/**
 * ds_permute intrinsic
 */
extern "C" int __amdgcn_ds_permute(int index, int src) [[hc]];
inline unsigned int __amdgcn_ds_permute(int index, unsigned int src) [[hc]] {
  __u tmp; tmp.u = src;
  tmp.i = __amdgcn_ds_permute(index, tmp.i);
  return tmp.u;
}
inline float __amdgcn_ds_permute(int index, float src) [[hc]] {
  __u tmp; tmp.f = src;
  tmp.i = __amdgcn_ds_permute(index, tmp.i);
  return tmp.f;
}


/**
 * ds_swizzle intrinsic
 */
extern "C" int __amdgcn_ds_swizzle(int src, int pattern) [[hc]];
inline unsigned int __amdgcn_ds_swizzle(unsigned int src, int pattern) [[hc]] {
  __u tmp; tmp.u = src;
  tmp.i = __amdgcn_ds_swizzle(tmp.i, pattern);
  return tmp.u;
}
inline float __amdgcn_ds_swizzle(float src, int pattern) [[hc]] {
  __u tmp; tmp.f = src;
  tmp.i = __amdgcn_ds_swizzle(tmp.i, pattern);
  return tmp.f;
}



/**
 * move DPP intrinsic
 */
extern "C" int __amdgcn_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl) [[hc]]; 

/**
 * Shift the value of src to the right by one thread within a wavefront.  
 * 
 * @param[in] src variable being shifted
 * @param[in] bound_ctrl When set to true, a zero will be shifted into thread 0; otherwise, the original value will be returned for thread 0
 * @return value of src being shifted into from the neighboring lane 
 * 
 */
extern "C" int __amdgcn_wave_sr1(int src, bool bound_ctrl) [[hc]];
inline unsigned int __amdgcn_wave_sr1(unsigned int src, bool bound_ctrl) [[hc]] {
  __u tmp; tmp.u = src;
  tmp.i = __amdgcn_wave_sr1(tmp.i, bound_ctrl);
  return tmp.u;
}
inline float __amdgcn_wave_sr1(float src, bool bound_ctrl) [[hc]] {
  __u tmp; tmp.f = src;
  tmp.i = __amdgcn_wave_sr1(tmp.i, bound_ctrl);
  return tmp.f;
}

/**
 * Shift the value of src to the left by one thread within a wavefront.  
 * 
 * @param[in] src variable being shifted
 * @param[in] bound_ctrl When set to true, a zero will be shifted into thread 63; otherwise, the original value will be returned for thread 63
 * @return value of src being shifted into from the neighboring lane 
 * 
 */
extern "C" int __amdgcn_wave_sl1(int src, bool bound_ctrl) [[hc]];  
inline unsigned int __amdgcn_wave_sl1(unsigned int src, bool bound_ctrl) [[hc]] {
  __u tmp; tmp.u = src;
  tmp.i = __amdgcn_wave_sl1(tmp.i, bound_ctrl);
  return tmp.u;
}
inline float __amdgcn_wave_sl1(float src, bool bound_ctrl) [[hc]] {
  __u tmp; tmp.f = src;
  tmp.i = __amdgcn_wave_sl1(tmp.i, bound_ctrl);
  return tmp.f;
}


/**
 * Rotate the value of src to the right by one thread within a wavefront.  
 * 
 * @param[in] src variable being rotated
 * @return value of src being rotated into from the neighboring lane 
 * 
 */
extern "C" int __amdgcn_wave_rr1(int src) [[hc]];
inline unsigned int __amdgcn_wave_rr1(unsigned int src) [[hc]] {
  __u tmp; tmp.u = src;
  tmp.i = __amdgcn_wave_rr1(tmp.i);
  return tmp.u;
}
inline float __amdgcn_wave_rr1(float src) [[hc]] {
  __u tmp; tmp.f = src;
  tmp.i = __amdgcn_wave_rr1(tmp.i);
  return tmp.f;
}

/**
 * Rotate the value of src to the left by one thread within a wavefront.  
 * 
 * @param[in] src variable being rotated
 * @return value of src being rotated into from the neighboring lane 
 * 
 */
extern "C" int __amdgcn_wave_rl1(int src) [[hc]];
inline unsigned int __amdgcn_wave_rl1(unsigned int src) [[hc]] {
  __u tmp; tmp.u = src;
  tmp.i = __amdgcn_wave_rl1(tmp.i);
  return tmp.u;
}
inline float __amdgcn_wave_rl1(float src) [[hc]] {
  __u tmp; tmp.f = src;
  tmp.i = __amdgcn_wave_rl1(tmp.i);
  return tmp.f;
}

#endif

/* definition to expand macro then apply to pragma message 
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "="  VALUE(var)
#pragma message(VAR_NAME_VALUE(__hcc_backend__))
*/

#if __hcc_backend__==HCC_BACKEND_AMDGPU

inline int __shfl(int var, int srcLane, int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
  int self = __lane_id();
  int index = srcLane + (self & ~(width-1));
  return __amdgcn_ds_bpermute(index<<2, var);
}

inline unsigned int __shfl(unsigned int var, int srcLane, int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
     __u tmp; tmp.u = var;
    tmp.i = __shfl(tmp.i, srcLane, width);
    return tmp.u;
}


inline float __shfl(float var, int srcLane, int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
    __u tmp; tmp.f = var;
    tmp.i = __shfl(tmp.i, srcLane, width);
    return tmp.f;
}

#endif

// FIXME: support half type
/** @} */

/** @{ */
/**
 * Copy from an active work-item with lower ID relative to
 * caller within a wavefront.
 *
 * Work-items may only read data from another work-item which is active in the
 * current wavefront. If the target work-item is inactive, the retrieved value
 * is fixed as 0.
 *
 * The function calculates a source work-item ID by subtracting delta from the
 * caller's work-item ID within the wavefront. The value of var held by the
 * resulting lane ID is returned: in effect, var is shifted up the wavefront by
 * delta work-items. If width is less than __HSA_WAVEFRONT_SIZE__ then each
 * subsection of the wavefront behaves as a separate entity with a starting
 * logical work-item ID of 0. The source work-item index will not wrap around
 * the value of width, so effectively the lower delta work-items will be unchanged.
 *
 * The optional width parameter must have a value which is a power of 2;
 * results are undefined if it is not a power of 2, or is number greater than
 * __HSA_WAVEFRONT_SIZE__.
 */

#if __hcc_backend__==HCC_BACKEND_AMDGPU

inline int __shfl_up(int var, const unsigned int delta, const int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
  int self = __lane_id();
  int index = self - delta;
  index = (index < (self & ~(width-1)))?self:index;
  return __amdgcn_ds_bpermute(index<<2, var);
}

inline unsigned int __shfl_up(unsigned int var, const unsigned int delta, const int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
    __u tmp; tmp.u = var;
    tmp.i = __shfl_up(tmp.i, delta, width);
    return tmp.u;
}

inline float __shfl_up(float var, const unsigned int delta, const int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
    __u tmp; tmp.f = var;
    tmp.i = __shfl_up(tmp.i, delta, width);
    return tmp.f;
}

#endif

// FIXME: support half type
/** @} */

/** @{ */
/**
 * Copy from an active work-item with higher ID relative to
 * caller within a wavefront.
 *
 * Work-items may only read data from another work-item which is active in the
 * current wavefront. If the target work-item is inactive, the retrieved value
 * is fixed as 0.
 *
 * The function calculates a source work-item ID by adding delta from the
 * caller's work-item ID within the wavefront. The value of var held by the
 * resulting lane ID is returned: this has the effect of shifting var up the
 * wavefront by delta work-items. If width is less than __HSA_WAVEFRONT_SIZE__
 * then each subsection of the wavefront behaves as a separate entity with a
 * starting logical work-item ID of 0. The ID number of the source work-item
 * index will not wrap around the value of width, so the upper delta work-items
 * will remain unchanged.
 *
 * The optional width parameter must have a value which is a power of 2;
 * results are undefined if it is not a power of 2, or is number greater than
 * __HSA_WAVEFRONT_SIZE__.
 */

#if __hcc_backend__==HCC_BACKEND_AMDGPU

inline int __shfl_down(int var, const unsigned int delta, const int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
  int self = __lane_id();
  int index = self + delta;
  index = (int)((self&(width-1))+delta) >= width?self:index;
  return __amdgcn_ds_bpermute(index<<2, var);
}

inline unsigned int __shfl_down(unsigned int var, const unsigned int delta, const int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
    __u tmp; tmp.u = var;
    tmp.i = __shfl_down(tmp.i, delta, width);
    return tmp.u;
}

inline float __shfl_down(float var, const unsigned int delta, const int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
    __u tmp; tmp.f = var;
    tmp.i = __shfl_down(tmp.i, delta, width);
    return tmp.f;
}

#endif

// FIXME: support half type
/** @} */

/** @{ */
/**
 * Copy from an active work-item based on bitwise XOR of caller
 * work-item ID within a wavefront.
 *
 * Work-items may only read data from another work-item which is active in the
 * current wavefront. If the target work-item is inactive, the retrieved value
 * is fixed as 0.
 *
 * THe function calculates a source work-item ID by performing a bitwise XOR of
 * the caller's work-item ID with laneMask: the value of var held by the
 * resulting work-item ID is returned.
 *
 * The optional width parameter must have a value which is a power of 2;
 * results are undefined if it is not a power of 2, or is number greater than
 * __HSA_WAVEFRONT_SIZE__.
 */

#if __hcc_backend__==HCC_BACKEND_AMDGPU


inline int __shfl_xor(int var, int laneMask, int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
  int self = __lane_id();
  int index = self^laneMask;
  index = index >= ((self+width)&~(width-1))?self:index;
  return __amdgcn_ds_bpermute(index<<2, var);
}

inline float __shfl_xor(float var, int laneMask, int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
    __u tmp; tmp.f = var;
    tmp.i = __shfl_xor(tmp.i, laneMask, width);
    return tmp.f;
}

// FIXME: support half type
/** @} */

inline unsigned int __shfl_xor(unsigned int var, int laneMask, int width=__HSA_WAVEFRONT_SIZE__) __HC__ {
    __u tmp; tmp.u = var;
    tmp.i = __shfl_xor(tmp.i, laneMask, width);
    return tmp.u;
}

#endif

/**
 * Multiply two unsigned integers (x,y) but only the lower 24 bits will be used in the multiplication.
 *
 * @param[in] x 24-bit unsigned integer multiplier
 * @param[in] y 24-bit unsigned integer multiplicand
 * @return 32-bit unsigned integer product
 */
inline unsigned int __mul24(unsigned int x, unsigned int y) [[hc]] {
  return (x & 0x00FFFFFF) * (y & 0x00FFFFFF);
}

/**
 * Multiply two integers (x,y) but only the lower 24 bits will be used in the multiplication.
 *
 * @param[in] x 24-bit integer multiplier
 * @param[in] y 24-bit integer multiplicand
 * @return 32-bit integer product
 */
inline int __mul24(int x, int y) [[hc]] {
  return  ((x << 8) >> 8) * ((y << 8) >> 8);
}

/**
 * Multiply two unsigned integers (x,y) but only the lower 24 bits will be used in the multiplication and
 * then add the product to a 32-bit unsigned integer
 *
 * @param[in] x 24-bit unsigned integer multiplier
 * @param[in] y 24-bit unsigned integer multiplicand
 * @param[in] z 32-bit unsigned integer to be added to the product
 * @return 32-bit unsigned integer result of mad24
 */
inline unsigned int __mad24(unsigned int x, unsigned int y, unsigned int z) [[hc]] {
  return __mul24(x,y) + z;
}

/**
 * Multiply two integers (x,y) but only the lower 24 bits will be used in the multiplication and
 * then add the product to a 32-bit integer
 *
 * @param[in] x 24-bit integer multiplier
 * @param[in] y 24-bit integer multiplicand
 * @param[in] z 32-bit integer to be added to the product
 * @return 32-bit integer result of mad24
 */
inline int __mad24(int x, int y, int z) [[hc]] {
  return __mul24(x,y) + z;
}

inline void abort() __HC__ {
  __builtin_trap();
}

// ------------------------------------------------------------------------
// group segment
// ------------------------------------------------------------------------

/**
 * Fetch the size of group segment. This includes both static group segment
 * and dynamic group segment.
 *
 * @return The size of group segment used by the kernel in bytes. The value
 *         includes both static group segment and dynamic group segment.
 */
extern "C" unsigned int get_group_segment_size() __HC__;

/**
 * Fetch the size of static group segment
 *
 * @return The size of static group segment used by the kernel in bytes.
 */
extern "C" unsigned int get_static_group_segment_size() __HC__;

/**
 * Fetch the address of the beginning of group segment.
 */
extern "C" void* get_group_segment_base_pointer() __HC__;

/**
 * Fetch the address of the beginning of dynamic group segment.
 */
extern "C" void* get_dynamic_group_segment_base_pointer() __HC__;
} // namespace hc