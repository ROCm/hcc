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

#include <hc/hc_agent_pool.hpp>
#include <hc/hc_atomics.hpp>
#include <hc/hc_callable_attributes.hpp>
#include <hc/hc_defines.hpp>
#include <hc/hc_exception.hpp>
#include <hc/hc_index.hpp>
#include <hc/hc_launch.hpp>
#include <hc/hc_math.hpp>
#include <hc/hc_queue_pool.hpp>
#include <hc/hc_runtime.hpp>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <array>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <future>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

/**
 * @namespace hc
 * Heterogeneous  C++ (HC) namespace
 */

namespace hc
{
    using namespace atomics;
    using namespace detail::enums;

    // forward declaration
    class accelerator;
    class accelerator_view;
    class completion_future;
    template <int> class extent;
    template <int> class tiled_extent;
    template <typename, int> class array_view;
    template <typename, int> class array;

    // namespace alias
    // namespace hc::fast_math is an alias of namespace detail::fast_math
    namespace fast_math = detail::fast_math;

    // namespace hc::precise_math is an alias of namespace detail::precise_math
    namespace precise_math = detail::precise_math;

    // type alias

    /**
     * Represents a unique position in N-dimensional space.
     */
    template <int N>
    using index = detail::index<N>;

    using runtime_exception = detail::runtime_exception;
    using invalid_compute_domain = detail::invalid_compute_domain;
    using accelerator_view_removed = detail::accelerator_view_removed;

    // ------------------------------------------------------------------------
    // global functions
    // ------------------------------------------------------------------------

    /**
     * Get the current tick count for the GPU platform.
     *
     * @return An implementation-defined tick count
     */
    inline
    std::uint64_t get_system_ticks()
    {   // TODO: unify the HSA error checking into a single function.
        std::uint64_t r{};
        detail::throwing_hsa_result_check(
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &r),
            __FILE__, __func__, __LINE__);

        return r;
    }

    /**
     * Get the frequency of ticks per second for the underlying asynchronous
     * operation.
     *
     * @return An implementation-defined frequency in Hz in case the instance is
     *         created by a kernel dispatch or a barrier packet. 0 otherwise.
     */
    inline
    std::uint64_t get_tick_frequency()
    {
        std::uint64_t r{};
        detail::throwing_hsa_result_check(
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &r),
            __FILE__, __func__, __LINE__);

        return r;
    }

    // ------------------------------------------------------------------------
    // completion_future
    // ------------------------------------------------------------------------

    /**
     * This class is the return type of all asynchronous APIs and has an
     * interface analogous to std::shared_future<void>. Similar to
     * std::shared_future, this type provides member methods such as wait and
     * get to wait for asynchronous operations to finish, and the type
     * additionally provides a member method then(), to specify a completion
     * callback functor to be executed upon completion of an asynchronous
     * operation.
     */
    class completion_future {
        std::shared_future<void> future_{};
        std::shared_ptr<std::once_flag> maybe_then_{};

        friend class accelerator_view;
        template<typename, int> friend class array_view;

        // non-tiled parallel_for_each
        // generic version
        template<typename Kernel, int n>
        friend
        completion_future parallel_for_each(
            const accelerator_view&, const extent<n>&, const Kernel&);

        // tiled parallel_for_each
        // generic version
        template<typename Kernel, int n>
        friend
        completion_future parallel_for_each(
            const accelerator_view&, const tiled_extent<n>&, const Kernel&);

        // copy_async
        template<typename T, int N>
        friend
        completion_future copy_async(
            const array_view<const T, N>& src, const array_view<T, N>& dest);
        template<typename T, int N>
        friend
        completion_future copy_async(const array<T, N>& src, array<T, N>& dest);
        template<typename T, int N>
        friend
        completion_future copy_async(
            const array<T, N>& src, const array_view<T, N>& dest);
        template<typename T, int N>
        friend
        completion_future copy_async(
            const array_view<T, N>& src, const array_view<T, N>& dest);
        template<typename T, int N>
        friend
        completion_future copy_async(
            const array_view<const T, N>& src, array<T, N>& dest);

        template<typename InputIter, typename T, int N>
        friend
        completion_future copy_async(
            InputIter srcBegin, InputIter srcEnd, array<T, N>& dest);
        template<typename InputIter, typename T, int N>
        friend
        completion_future copy_async(
            InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest);
        template<typename InputIter, typename T, int N>
        friend
        completion_future copy_async(InputIter srcBegin, array<T, N>& dest);
        template<typename InputIter, typename T, int N>
        friend
        completion_future copy_async(
            InputIter srcBegin, const array_view<T, N>& dest);
        template<typename OutputIter, typename T, int N>
        friend
        completion_future copy_async(
            const array<T, N>& src, OutputIter destBegin);
        template<typename OutputIter, typename T, int N>
        friend
        completion_future copy_async(
            const array_view<T, N>& src, OutputIter destBegin);

        completion_future(std::shared_future<void> future)
            :
            future_{std::move(future)},
            maybe_then_{std::make_shared<std::once_flag>()}
        {}
    public:

        /**
         * Default constructor. Constructs an empty uninitialized
         * completion_future object which does not refer to any asynchronous
         * operation. Default constructed completion_future objects have valid()
         * == false
         */
        completion_future() = default;

        /**
         * Copy constructor. Constructs a new completion_future object that
         * refers to the same asynchronous operation as the other
         * completion_future object.
         *
         * @param[in] other An object of type completion_future from which to
         *                  initialize this.
         */
        completion_future(const completion_future&) = default;

        /**
         * Move constructor. Move constructs a new completion_future object that
         * refers to the same asynchronous operation as originally referred by
         * the other completion_future object. After this constructor returns,
         * other.valid() == false
         *
         * @param[in] other An object of type completion_future which the new
         *                  completion_future
         */
        completion_future(completion_future&&) = default;

        ~completion_future() = default;
        /**
         * Copy assignment. Copy assigns the contents of other to this. This
         * method causes this to stop referring its current asynchronous
         * operation and start referring the same asynchronous operation as
         * other.
         *
         * @param[in] other An object of type completion_future which is copy
         *                  assigned to this.
         */
        completion_future& operator=(const completion_future&) = default;

        /**
         * Move assignment. Move assigns the contents of other to this. This
         * method causes this to stop referring its current asynchronous
         * operation and start referring the same asynchronous operation as
         * other. After this method returns, other.valid() == false
         *
         * @param[in] other An object of type completion_future which is move
         *                  assigned to this.
         */
        completion_future& operator=(completion_future&&) = default;

        /**
         * This method is functionally identical to
         * std::shared_future<void>::get. This method waits for the associated
         * asynchronous operation to finish and returns only upon the completion
         * of the asynchronous operation. If an exception was encountered during
         * the execution of the asynchronous operation, this method throws that
         * stored exception.
         */
        void get() const
        {
            future_.get();
        }

        /**
         * This method is functionally identical to
         * std::shared_future<void>::valid. This returns true if this
         * completion_future is associated with an asynchronous operation.
         */
        bool valid() const
        {
            return future_.valid();
        }

        /** @{ */
        /**
         * These methods are functionally identical to the corresponding
         * std::shared_future<void> methods.
         *
         * The wait method waits for the associated asynchronous operation to
         * finish and returns only upon completion of the associated
         * asynchronous operation or if an exception was encountered when
         * executing the asynchronous operation.
         *
         * The other variants are functionally identical to the
         * std::shared_future<void> member methods with same names.
         *
         * @param waitMode[in] An optional parameter to specify the wait mode.
         *                     By default it would be hcWaitModeBlocked.
         *                     hcWaitModeActive would be used to reduce latency
         *                     with the expense of using one CPU core for active
         *                     waiting.
         */
        void wait() const
        {
            future_.wait();

            // TODO: printf:(
            //detail::getContext()->flushPrintfBuffer();
        }

        template<typename Rep, typename Period>
        std::future_status wait_for(
            const std::chrono::duration<Rep, Period>& rel_time) const
        {
            return future_.wait_for(rel_time);
        }

        template<typename Clock, typename Duration>
        std::future_status wait_until(
            const std::chrono::time_point<Clock, Duration>& abs_time) const
        {
            return future_.wait_until(abs_time);
        }

        /** @} */

        /**
         * Conversion operator to std::shared_future<void>. This method returns
         * a shared_future<void> object corresponding to this completion_future
         * object and refers to the same asynchronous operation.
         */
        operator std::shared_future<void>() const
        {
            return future_;
        }

        /**
         * This method enables specification of a completion callback func which
         * is executed upon completion of the asynchronous operation associated
         * with this completion_future object. The completion callback func
         * should have an operator() that is valid when invoked with non
         * arguments, i.e., "func()".
         */
        template<typename F>
        void then(const F& func) const
        {   // TODO: this is probably incorrect; then() was underspecified in
            //       C++AMP, and subtle to get right; we may want to remove it
            //       or extend it to return a future, otherwise it is
            //       intractable to provide guarantees about when the
            //       continuation executes and, respectively, when it completes.
            std::call_once(
                *maybe_then_, [=](const std::shared_future<void>& fut) {
                std::thread{[=]() { fut.wait(); func(); }}.detach();
            }, std::cref(future_));
        }

        /**
         * Get the native handle for the asynchronous operation encapsulated in
         * this completion_future object. The method is mostly used for
         * debugging purpose.
         * Applications should retain the parent completion_future to ensure the
         * native handle is not deallocated by the HCC runtime. The
         * completion_future pointer to the native handle is reference counted,
         * so a copy of the completion_future is sufficient to retain the
         * native_handle.
         */
        // void* get_native_handle() const
        // {
        //     if (__asyncOp != nullptr) {
        //         return __asyncOp->getNativeHandle();
        //     } else {
        //         return nullptr;
        //     }
        // }

        /**
         * Get the tick number when the underlying asynchronous operation
         * begins.
         *
         * @return An implementation-defined tick number in case the instance is
         *         created by a kernel dispatch or a barrier packet. 0
         *         otherwise.
         */
        // uint64_t get_begin_tick()
        // {
        //     if (__asyncOp != nullptr) {
        //         return __asyncOp->getBeginTimestamp();
        //     } else {
        //         return 0L;
        //     }
        // }

        /**
         * Get the tick number when the underlying asynchronous operation ends.
         *
         * @return An implementation-defined tick number in case the instance is
         *         created by a kernel dispatch or a barrier packet. 0
         *         otherwise.
         */
        // uint64_t get_end_tick()
        // {
        //     if (__asyncOp != nullptr) {
        //         return __asyncOp->getEndTimestamp();
        //     } else {
        //         return 0L;
        //     }
        // }

        /**
         * Get the frequency of ticks per second for the underlying asynchronous
         * operation.
         *
         * @return An implementation-defined frequency in Hz in case the
         *         instance is created by a kernel dispatch or a barrier packet.
         *         0 otherwise.
         */
        // uint64_t get_tick_frequency()
        // {
        //     if (__asyncOp != nullptr) {
        //         return __asyncOp->getTimestampFrequency();
        //     } else {
        //         return 0L;
        //     }
        // }

        /**
         * Get if the async operations has been completed.
         *
         * @return True if the async operation has been completed, false if not.
         */
        bool is_ready()
        {
            return future_.wait_for(std::chrono::nanoseconds{0}) ==
                std::future_status::ready;
        }

        /**
         * @return reference count for the completion future. Primarily used for
         *         debug purposes.
         */
        // int get_use_count() const
        // {
        //     return __asyncOp.use_count();
        // }
    };

    // ------------------------------------------------------------------------
    // accelerator_view
    // ------------------------------------------------------------------------

    /**
     * Represents a logical (isolated) accelerator view of a compute
     * accelerator. An object of this type can be obtained by calling the
     * default_view property or create_view member functions on an accelerator
     * object.
     */
    class accelerator_view {
        mutable std::forward_list<completion_future> pending_tasks_; // TODO: spec fault.
        accelerator const* accelerator_;
        hsa_queue_t* queue_;
        queuing_mode qmode_;

        friend class accelerator;
        template <typename, int> friend class array;
        template <typename, int> friend class array_view;

        template<typename Domain, typename Kernel>
        friend
        void detail::launch_kernel(
            const accelerator_view&,
            const Domain&,
            const Kernel&);
        template<typename Domain, typename Kernel>
        friend
        std::shared_future<void> detail::launch_kernel_async(
            const accelerator_view&,
            const Domain&,
            const Kernel&);

        // non-tiled parallel_for_each
        // generic version
        template <typename Kernel, int n>
        friend
        completion_future parallel_for_each(
            const accelerator_view&, const extent<n>&, const Kernel&);

        // tiled parallel_for_each
        // generic version
        template <typename Kernel, int n>
        friend
        completion_future parallel_for_each(
            const accelerator_view&, const tiled_extent<n>&, const Kernel&);

        // IMPLEMENTATION - MANIPULATORS
        void add_pending_task_(const completion_future& task) const
        {
            pending_tasks_.push_front(task);
        }
        // TODO: reorder completion_future to allow for inline definition or
        //       move to .cpp (the latter may be preferable).
        void wait_for_all_pending_tasks_();

        // IMPLEMENTATION - CREATORS
        accelerator_view(
            const accelerator& accelerator,
            hsa_queue_t* queue,
            queuing_mode qmode = queuing_mode_automatic)
            : accelerator_{&accelerator}, queue_{queue}, qmode_{qmode}
        {}
    public:
        accelerator_view() = delete;
        /**
         * Copy-constructs an accelerator_view object. This function does a
         * shallow copy with the newly created accelerator_view object pointing
         * to the same underlying view as the "other" parameter.
         *
         * @param[in] other The accelerator_view object to be copied.
         */
        accelerator_view(const accelerator_view&) = default;
        accelerator_view(accelerator_view&&) = default;

        ~accelerator_view()
        {
            wait_for_all_pending_tasks_();
        }
        /**
         * Assigns an accelerator_view object to "this" accelerator_view object
         * and returns a reference to "this" object. This function does a
         * shallow assignment with the newly created accelerator_view object
         * pointing to the same underlying view as the passed accelerator_view
         * parameter.
         *
         * @param[in] other The accelerator_view object to be assigned from.
         * @return A reference to "this" accelerator_view object.
         */
        accelerator_view& operator=(const accelerator_view&) = default;
        accelerator_view& operator=(accelerator_view&&) = default;

        /**
         * Returns the queuing mode that this accelerator_view was created with.
         * See "Queuing Mode".
         *
         * @return The queuing mode.
         */
        queuing_mode get_queuing_mode() const noexcept
        {
            return qmode_;
        }

        /**
         * Returns a boolean value indicating whether the accelerator view when
         * passed to a parallel_for_each would result in automatic selection of
         * an appropriate execution target by the runtime. In other words, this
         * is the accelerator view that will be automatically selected if
         * parallel_for_each is invoked without explicitly specifying an
         * accelerator view.
         *
         * @return A boolean value indicating if the accelerator_view is the
         *         auto selection accelerator_view.
         */
        bool get_is_auto_selection() const noexcept;

        /**
         * Returns a 32-bit unsigned integer representing the version number of
         * this accelerator view. The format of the integer is major.minor,
         * where the major version number is in the high-order 16 bits, and the
         * minor version number is in the low-order bits.
         *
         * The version of the accelerator view is usually the same as that of
         * the parent accelerator.
         */
        unsigned int get_version() const;

        /**
         * Returns the accelerator that this accelerator_view has been created
         * on.
         */
        accelerator get_accelerator() const;

        /**
         * Returns a boolean value indicating whether the accelerator_view
         * supports debugging through extensive error reporting.
         *
         * The is_debug property of the accelerator view is usually same as that
         * of the parent accelerator.
         */
        bool get_is_debug() const noexcept
        {   // FIXME: dummy implementation now
            return false;
        }

        /**
         * Performs a blocking wait for completion of all commands submitted to
         * the accelerator view prior to calling wait().
         *
         * @param waitMode[in] An optional parameter to specify the wait mode.
         *                     By default it would be hcWaitModeBlocked.
         *                     hcWaitModeActive would be used to reduce latency
         *                     with the expense of using one CPU core for active
         *                     waiting.
         */
        void wait()
        {
            wait_for_all_pending_tasks_();

            //detail::getContext()->flushPrintfBuffer();
        }

        /**
         * Sends the queued up commands in the accelerator_view to the device
         * for execution.
         *
         * An accelerator_view internally maintains a buffer of commands such as
         * data transfers between the host memory and device buffers, and kernel
         * invocations (parallel_for_each calls). This member function sends the
         * commands to the device for processing. Normally, these commands
         * to the GPU automatically whenever the runtime determines that they
         * need to be, such as when the command buffer is full or when waiting
         * for transfer of data from the device buffers to host memory. The
         * flush member function will send the commands manually to the device.
         *
         * Calling this member function incurs an overhead and must be used with
         * discretion. A typical use of this member function would be when the
         * CPU waits for an arbitrary amount of time and would like to force the
         * execution of queued device commands in the meantime. It can also be
         * used to ensure that resources on the accelerator are reclaimed after
         * all references to them have been removed.
         *
         * Because flush operates asynchronously, it can return either before or
         * after the device finishes executing the buffered commands, the
         * commands will eventually always complete.
         *
         * If the queuing_mode is queuing_mode_immediate, this function has no
         * effect.
         *
         * @return None
         */
        void flush()
        {   // TODO: for now we always submit immediately, so flush is a NOP.
            return;
        }

        /**
         * This command inserts a marker event into the accelerator_view's
         * command queue. This marker is returned as a completion_future object.
         * When all commands that were submitted prior to the marker event
         * creation have completed, the future is ready.
         *
         * Regardless of the accelerator_view's execute_order
         * (execute_any_order, execute_in_order), the marker always ensures
         * older commands complete before the returned completion_future is
         * marked ready. Thus, markers provide a mechanism to enforce order
         * between commands in an execute_any_order accelerator_view.
         *
         * fence_scope controls the scope of the acquire and release fences
         * applied after the marker executes.  Options are:
         *   - no_scope : No fence operation is performed.
         *   - accelerator_scope: Memory is acquired from and released to the
         *     accelerator scope where the marker executes.
         *   - system_scope: Memory is acquired from and released to system
         *     scope (all accelerators including CPUs)
         *
         * @return A future which can be waited on, and will block until the
         *         current batch of commands has completed.
         */
        completion_future create_marker(
            memory_scope fence_scope = system_scope) const;

        /**
         * This command inserts a marker event into the accelerator_view's
         * command queue with a prior dependent asynchronous event.
         *
         * This marker is returned as a completion_future object. When its
         * dependent event and all commands submitted prior to the marker event
         * creation have been completed, the future is ready.
         *
         * Regardless of the accelerator_view's execute_order
         * (execute_any_order, execute_in_order), the marker always ensures
         * older commands complete before the returned completion_future is
         * marked ready. Thus, markers provide a mechanism to enforce order
         * between commands in an execute_any_order accelerator_view.
         *
         * fence_scope controls the scope of the acquire and release fences
         * applied after the marker executes.  Options are:
         *   - no_scope : No fence operation is performed.
         *   - accelerator_scope: Memory is acquired from and released to the
         *     accelerator scope where the marker executes.
         *   - system_scope: Memory is acquired from and released to system
         *     scope (all accelerators including CPUs)
         *
         * dependent_futures may be recorded in another queue or another
         * accelerator.  If in another accelerator, the runtime performs
         * cross-accelerator synchronisation.
         *
         * @return A future which can be waited on, and will block until the
         *         current batch of commands, plus the dependent event have
         *         been completed.
         */
        completion_future create_blocking_marker(
            completion_future& dependent_future,
            memory_scope fence_scope = system_scope) const;

        /**
         * This command inserts a marker event into the accelerator_view's
         * command queue with arbitrary number of dependent asynchronous events.
         *
         * This marker is returned as a completion_future object. When its
         * dependent events and all commands submitted prior to the marker event
         * creation have been completed, the completion_future is ready.
         *
         * Regardless of the accelerator_view's execute_order
         * (execute_any_order, execute_in_order), the marker always ensures
         * older commands complete before the returned completion_future is
         * marked ready. Thus, markers provide a mechanism to enforce order
         * between commands in an execute_any_order accelerator_view.
         *
         * fence_scope controls the scope of the acquire and release fences
         * applied after the marker executes.  Options are:
         *   - no_scope : No fence operation is performed.
         *   - accelerator_scope: Memory is acquired from and released to the
         *     accelerator scope where the marker executes.
         *   - system_scope: Memory is acquired from and released to system
         *     scope (all accelerators including CPUs)
         *
         * @return A future which can be waited on, and will block until the
         *         current batch of commands, plus the dependent event have
         *         been completed.
         */
        completion_future create_blocking_marker(
            std::initializer_list<completion_future> dependent_future_list,
            memory_scope fence_scope = system_scope) const;

        /**
         * This command inserts a marker event into the accelerator_view's
         * command queue with arbitrary number of dependent asynchronous events.
         *
         * This marker is returned as a completion_future object. When its
         * dependent events and all commands submitted prior to the marker event
         * creation have been completed, the completion_future is ready.
         *
         * Regardless of the accelerator_view's execute_order
         * (execute_any_order, execute_in_order), the marker always ensures
         * older commands complete before the returned completion_future is
         * marked ready. Thus, markers provide a mechanism to enforce order
         * between commands in an execute_any_order accelerator_view.
         *
         * @return A future which can be waited on, and will block until the
         *         current batch of commands, plus the dependent event have
         *         been completed.
         */
        template<typename InputIterator>
        completion_future create_blocking_marker(
            InputIterator first,
            InputIterator last,
            memory_scope fence_scope = system_scope) const;

        /**
         * Copies size_bytes bytes from src to dst.
         * Src and dst must not overlap.
         * Note the src is the first parameter and dst is second, following C++
         * convention. The copy command will execute after any commands already
         * inserted into the accelerator_view finish. This is a synchronous copy
         * command, and the copy operation complete before this call returns.
         */
        void copy(const void* src, void* dst, std::size_t size_bytes)
        {
            wait_for_all_pending_tasks_();

            detail::throwing_hsa_result_check(
                hsa_memory_copy(dst, src, size_bytes),
                __FILE__, __func__, __LINE__);
        }

        /**
         * Copies size_bytes bytes from src to dst.
         * Src and dst must not overlap.
         * Note the src is the first parameter and dst is second, following C++
         * convention. This is an asynchronous copy command, and this call may
         * return before the copy operation completes. If the source or dest is
         * host memory, the memory must be pinned or a runtime exception will be
         * thrown. Pinned memory can be created with am_alloc with
         * flag=amHostPinned flag.
         *
         * The copy command will be implicitly ordered with respect to commands
         * previously enqueued to this accelerator_view:
         * - If the accelerator_view execute_order is execute_in_order
         *   (the default), then the copy will execute after all previously sent
         *   commands finish execution.
         * - If the accelerator_view execute_order is execute_any_order, then
         *   the
         *   copy will start after all previously send commands start but can
         *   execute in any order.
         */
        completion_future copy_async(
            const void* src, void* dst, std::size_t size_bytes)
        {
            wait_for_all_pending_tasks_();

            return completion_future{std::async([=]() {
                detail::throwing_hsa_result_check(
                    hsa_memory_copy(dst, src, size_bytes),
                    __FILE__, __func__, __LINE__);
            }).share()};
        }

        /**
         * Compares "this" accelerator_view with the passed accelerator_view
         * object to determine if they represent the same underlying object.
         *
         * @param[in] other The accelerator_view object to be compared against.
         * @return A boolean value indicating whether the passed
         *         accelerator_view object is same as "this" accelerator_view.
         */
        bool operator==(const accelerator_view& other) const noexcept
        {
            return queue_ == other.queue_;
        }

        /**
         * Compares "this" accelerator_view with the passed accelerator_view
         * object to determine if they represent different underlying objects.
         *
         * @param[in] other The accelerator_view object to be compared against.
         * @return A boolean value indicating whether the passed
         *         accelerator_view object is different from "this"
         *         accelerator_view.
         */
        bool operator!=(const accelerator_view& other) const noexcept
        {
            return !(*this == other);
        }

        /**
         * Returns an opaque handle which points to the underlying HSA queue.
         *
         * @return An opaque handle of the underlying HSA queue, if the
         *         accelerator view is based on HSA.  NULL if otherwise.
         */
        void* get_hsa_queue() const
        {
            return queue_;
        }

        /**
         * Dispatch a kernel into the accelerator_view.
         *
         * This function is intended to provide a gateway to dispatch code
         * objects, with some assistance from HCC. Kernels are specified in the
         * standard code object format, and can be created from a variety of
         * compiler tools including the assembler, offline cl compilers, or
         * other tools. The caller also specifies the execution configuration
         * and kernel arguments. HCC will copy the kernel arguments into an
         * appropriate segment and insert the packet into the queue. HCC will
         * also automatically handle signal and kernarg allocation and
         * deallocation for the command.
         *
         * The kernel is dispatched asynchronously, and thus this API may return
         * before the kernel finishes executing.

        * Kernels dispatched with this API may be interleaved with other copy
        * and kernel commands generated from copy or parallel_for_each commands.
        * The kernel honors the execute_order associated with the
        * accelerator_view. Specifically, if execute_order is execute_in_order,
        * then the kernel will wait for older data and kernel commands in the
        * same queue before beginning execution. If execute_order is
        * execute_any_order, then the kernel may begin executing without regards
        * to the state of older kernels. This call honors the packer barrier bit
        * (1 << HSA_PACKET_HEADER_BARRIER) if set in the aql.header field. If
        * set, this provides the same synchronization behavior as
        * execute_in_order for the command generated by this API.
        *
        * @p aql is an HSA-format "AQL" packet. The following fields must
        * be set by the caller:
        *  aql.kernel_object
        *  aql.group_segment_size : includes static + dynamic group size
        *  aql.private_segment_size
        *  aql.grid_size_x, aql.grid_size_y, aql.grid_size_z
        *  aql.group_size_x, aql.group_size_y, aql.group_size_z
        *  aql.setup: The 2 bits at HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS.
        *  aql.header: Must specify the desired memory fence operations, and
        *              barrier bit (if desired.). A typical conservative setting
        *              would be:
        aql.header =
            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE) |
            (1 << HSA_PACKET_HEADER_BARRIER);

        * The following fields are ignored. The API will will set up these
        * fields before dispatching the AQL packet:
        *  aql.completion_signal
        *  aql.kernarg
        *
        * @p args : Pointer to kernel arguments with the size and alignment
        *           expected by the kernel. The args are copied and then passed
        *           directly to the kernel. After this function returns, the
        *           args memory may be deallocated.
        * @p argSz : Size of the arguments.
        * @p cf : Written with a completion_future that can be used to track the
        *         status of the dispatch. May be NULL, in which case no
        *         completion_future is returned and the caller must use other
        *         synchronization techniques such as calling
        *         accelerator_view::wait() or waiting on a younger command in
        *         the same queue.
        * @p kernel_name: Optionally specify the name of the kernel for debug
        *                 and profiling. May be null. If specified, the caller
        *                 is responsible for ensuring the memory for the name
        *                 remains allocated until the kernel completes.
        *
        * The dispatch_hsa_kernel call will perform the following operations:
        *    - Efficiently allocate a kernarg region and copy the arguments.
        *    - Efficiently allocate a signal, if required.
        *    - Dispatch the command into the queue and flush it to the GPU.
        *    - Kernargs and signals are automatically reclaimed by the HCC
        *      runtime.
        */
        // void dispatch_hsa_kernel(
        //     const hsa_kernel_dispatch_packet_t* aql,
        //     void* args,
        //     size_t argsize,
        //     completion_future* cf = nullptr,
        //     const char* kernel_name = nullptr)
        // {
        //     wait_for_all_pending_tasks_(); // TODO: this is conservative.

        //     completion_future tmp{};
        //     queue_->dispatch_hsa_kernel(aql, args, argsize, &tmp, kernel_name);

        //     add_pending_task_(tmp);

        //     if (cf) *cf = std::move(tmp);
        // }

        /**
         * Set a CU affinity to specific command queues.
         * The setting is permanent until the queue is destroyed or CU affinity
         * is set again. This setting is "atomic", it won't affect the dispatch
         * in flight.
         *
         * @param cu_mask a bool vector to indicate what CUs you want to use.
         *                True represents using the cu. The first 32 elements
         *                represents the first 32 CUs, and so on. If its size is
         *                greater than physical CU number, the extra elements
         *                are ignored. It is user's responsibility to make sure
         *                the input is meaningful.
         *
         * @return true if operations succeeds or false if not.
         */
        bool set_cu_mask(const std::vector<bool>& cu_mask);
    };

    // ------------------------------------------------------------------------
    // accelerator
    // ------------------------------------------------------------------------

    /**
     * Represents a physical accelerated computing device. An object of
     * this type can be created by enumerating the available devices, or
     * getting the default device.
     */
    class accelerator {
        // DATA - STATICS
        inline static std::once_flag maybe_set_default_{};

        // DATA
        hsa_agent_t agent_{};

        friend class accelerator_view;

        // IMPLEMENTATION - CREATORS
        explicit
        accelerator(hsa_agent_t agent) : agent_{agent}
        {
            if (detail::Agent_pool::pool().count(agent) != 0) return;

            throw std::logic_error{
                "Tried to create accelerator from unknown HSA agent."};
        }
    public:
        inline static constexpr const wchar_t cpu_accelerator[]{L"cpu"};
        inline static constexpr const wchar_t default_accelerator[]{L"default"};

        /**
         * Constructs a new accelerator object that represents the default
         * accelerator. This is equivalent to calling the constructor
         * @code{.cpp}
         * accelerator(accelerator::default_accelerator)
         * @endcode
         *
         * The actual accelerator chosen as the default can be affected by
         * calling accelerator::set_default().
         */
        accelerator() : accelerator{default_accelerator} {}

        /**
         * Constructs a new accelerator object that represents the physical
         * device named by the "path" argument. If the path represents an
         * unknown or unsupported device, an exception will be thrown.
         *
         * The path can be one of the following:
         * 1. accelerator::default_accelerator (or L"default"), which represents
         *    the path of the fastest accelerator available, as chosen by the
         *    runtime.
         * 2. accelerator::cpu_accelerator (or L"cpu"), which represents the
         *    CPU. Note that parallel_for_each shall not be invoked over this
         *    accelerator.
         * 3. A valid device path that uniquely identifies a hardware
         *    accelerator available on the host system.
         *
         * @param[in] path The device path of this accelerator.
         */
        explicit
        accelerator(const std::wstring& path)
            : accelerator{
                (path == default_accelerator) ?
                    detail::Agent_pool::default_agent() :
                        ((path == cpu_accelerator) ?
                            detail::Agent_pool::cpu_agent() :
                            hsa_agent_t{std::stoull(path)})}
        {}

        /**
         * Copy constructs an accelerator object. This function does a shallow
         * copy with the newly created accelerator object pointing to the same
         * underlying device as the passed accelerator parameter.
         *
         * @param[in] other The accelerator object to be copied.
         */
        accelerator(const accelerator&) = default;
        accelerator(accelerator&&) = default;

        /**
         * Returns a std::vector of accelerator objects (in no specific
         * order) representing all accelerators that are available, including
         * reference accelerators if available.
         *
         * @return A vector of accelerators.
         */
        static
        std::vector<accelerator> get_all()
        {
            static std::vector<accelerator> r;
            static std::once_flag f;

            std::call_once(f, []() {
                for(auto&& agent : detail::Agent_pool::pool()) {
                    r.push_back(accelerator{agent.first});
                }
            });

            return r;
        }

        /**
         * Sets the default accelerator to the device path identified by the
         * "path" argument. See the constructor
         * accelerator(const std::wstring& path) for a description of the
         * allowable path strings.
         *
         * This establishes a process-wide default accelerator and influences
         * all subsequent operations that might use a default accelerator.
         *
         * @param[in] path The device path of the default accelerator.
         * @return A Boolean flag indicating whether the default was set. If the
         *         default has already been set for this process, this value
         *         will be false, and the function will have no effect.
         */
        static
        bool set_default(const std::wstring& path)
        {
            bool r{false};
            std::call_once(maybe_set_default_, [&]() {
                r = true;

                if (path == default_accelerator) return;
                if (path == cpu_accelerator) {
                    detail::Agent_pool::default_agent() =
                        detail::Agent_pool::cpu_agent();

                    return;
                }

                const hsa_agent_t tmp{std::stoull(path)};
                if (detail::Agent_pool::pool().count(tmp) != 0) {
                    detail::Agent_pool::default_agent() = tmp;

                    return;
                }

                throw std::logic_error{
                    "Tried to set unknown HSA agent as default."};
            });

            return r;
        }

        /**
         * Returns an accelerator_view which when passed as the first argument
         * to a parallel_for_each call causes the runtime to automatically
         * select the target accelerator_view for executing the
         * parallel_for_each kernel. In other words, a parallel_for_each
         * invocation with the accelerator_view returned by
         * get_auto_selection_view() is the same as a parallel_for_each
         * invocation without an accelerator_view argument.
         *
         * For all other purposes, the accelerator_view returned by
         * get_auto_selection_view() behaves the same as the default
         * accelerator_view of the default accelerator
         * (aka accelerator().get_default_view()).
         *
         * @return An accelerator_view than can be used to indicate auto
         *         selection of the target for a parallel_for_each execution.
         */
        static
        accelerator_view get_auto_selection_view()
        {
            set_default(default_accelerator);

            static accelerator acc{default_accelerator};

            return acc.get_default_view();
        }

        /**
         * Assigns an accelerator object to "this" accelerator object and
         * returns a reference to "this" object. This function does a shallow
         * assignment with the newly created accelerator object pointing to the
         * same underlying device as the passed accelerator parameter.
         *
         * @param other The accelerator object to be assigned from.
         * @return A reference to "this" accelerator object.
         */
        accelerator& operator=(const accelerator&) = default;
        accelerator& operator=(accelerator&&) = default;

        /**
         * Returns the default accelerator_view associated with the accelerator.
         * The queuing_mode of the default accelerator_view is
         * queuing_mode_automatic.
         *
         * @return The default accelerator_view object associated with the
         * accelerator.
         */
        accelerator_view get_default_view() const
        {
            return accelerator_view{
                *this, detail::Queue_pool::default_queue(agent_)};
        }

        /**
         * Creates and returns a new accelerator view on the accelerator with
         * the supplied queuing mode.
         *
         * @param[in] qmode The queuing mode of the accelerator_view to be
         *                  created. See "Queuing Mode". The default value would
         *                  be queueing_mode_automatic if not specified.
         */
        accelerator_view create_view(
            execute_order = execute_in_order,
            queuing_mode mode = queuing_mode_automatic)
        {
            return accelerator_view{
                *this, detail::Queue_pool::defined_queue(agent_), mode};
        }

        /**
         * Compares "this" accelerator with the passed accelerator object to
         * determine if they represent the same underlying device.
         *
         * @param[in] other The accelerator object to be compared against.
         * @return A boolean value indicating whether the passed accelerator
         *         object is same as "this" accelerator.
         */
        bool operator==(const accelerator& other) const
        {
            return agent_.handle == other.agent_.handle;
        }

        /**
         * Compares "this" accelerator with the passed accelerator object to
         * determine if they represent different devices.
         *
         * @param[in] other The accelerator object to be compared against.
         * @return A boolean value indicating whether the passed accelerator
         *         object is different from "this" accelerator.
         */
        bool operator!=(const accelerator& other) const
        {
            return !(*this == other);
        }

        /**
         * Sets the default_cpu_access_type for this accelerator.
         *
         * The default_cpu_access_type is used for arrays created on this
         * accelerator or for implicit array_view memory allocations accessed on
         * this accelerator.
         *
         * This method only succeeds if the default_cpu_access_type for the
         * accelerator has not already been overriden by a previous call to this
         * method and the runtime selected default_cpu_access_type for this
         * accelerator has not yet been used for allocating an array or for an
         * implicit array_view memory allocation on this accelerator.
         *
         * @param[in] default_cpu_access_type The default cpu access_type to be
         *                                    used for array/array_view memory
         *                                    allocations on this accelerator.
         * @return A boolean value indicating if the default cpu access_type for
         *         the accelerator was successfully set.
         */
        bool set_default_cpu_access_type(access_type type)
        {
            static std::unordered_map<hsa_agent_t, std::once_flag> done;

            bool set{false};
            std::call_once(done[agent_], [&](){
                set = true;

                detail::Agent_pool::pool()[agent_].default_cpu_access = type;
            });

            return set;
        }

        /**
         * Returns a system-wide unique device instance path that matches the
         * "Device Instance Path" property for the device in Device Manager, or
         * one of the predefined path constants cpu_accelerator.
         */
        std::wstring get_device_path() const
        {
            return std::to_wstring(agent_.handle);
        }

        /**
         * Returns a short textual description of the accelerator device.
         */
        std::wstring get_description() const
        {
            return detail::Agent_pool::pool()[agent_].name;
        }

        /**
         * Returns a 32-bit unsigned integer representing the version number of
         * this accelerator. The format of the integer is major.minor, where the
         * major version number is in the high-order 16 bits, and the minor
         * version number is in the low-order bits.
         */
        unsigned int get_version() const
        {
            return detail::Agent_pool::pool()[agent_].version;
        }

        /**
         * This property indicates that the accelerator may be shared by (and
         * thus have interference from) the operating system or other system
         * software components for rendering purposes. A C++ AMP implementation
         * may set this property to false should such interference not be
         * applicable for a particular accelerator.
         */
        bool get_has_display() const
        {   // FIXME: dummy implementation now
            return false;
        }

        /**
         * Returns the amount of dedicated memory (in KB) on an accelerator
         * device. There is no guarantee that this amount of memory is actually
         * available to use.
         */
        size_t get_dedicated_memory() const
        {
            return detail::Agent_pool::pool()[agent_].dedicated_memory;
        }

        /**
         * Returns a Boolean value indicating whether this accelerator supports
         * double-precision (double) computations. When this returns true,
         * supports_limited_double_precision also returns true.
         */
        bool get_supports_double_precision() const
        {   // This is true for all targets we support at the moment.
            return true;
        }

        /**
         * Returns a boolean value indicating whether the accelerator has
         * limited double precision support (excludes double division,
         * precise_math functions, int to double, double to int conversions) for
         * a parallel_for_each kernel.
         */
        bool get_supports_limited_double_precision() const
        {   // This is true for all targets we support at the moment.
            return true;
        }

        /**
         * Returns a boolean value indicating whether the accelerator supports
         * debugging.
         */
        bool get_is_debug() const
        {   // FIXME: dummy implementation now
            return false;
        }

        /**
         * Returns a boolean value indicating whether the accelerator is
         * emulated. This is true, for example, with the reference, and CPU
         * accelerators.
         */
        bool get_is_emulated() const
        {
            return detail::Agent_pool::pool()[agent_].is_cpu;
        }

        /**
         * Returns a boolean value indicating whether the accelerator supports
         * memory accessible both by the accelerator and the CPU.
         */
        bool get_supports_cpu_shared_memory() const
        {
            return detail::Agent_pool::pool()[agent_].has_cpu_shared_memory;
        }

        /**
         * Get the default cpu access_type for buffers created on this
         * accelerator
         */
        access_type get_default_cpu_access_type() const
        {
            return detail::Agent_pool::pool()[agent_].default_cpu_access;
        }


        /**
         * Returns the maximum size of tile static area available on this
         * accelerator.
         */
        std::size_t get_max_tile_static_size() const
        {
            return detail::Agent_pool::pool()[agent_].max_tile_static_size;
        }

        /**
         * Returns an opaque handle which points to the AM region on the HSA
         * agent. This region can be used to allocate accelerator memory which
         * is accessible from the specified accelerator.
         *
         * @return An opaque handle of the region, if the accelerator is based
         *         on HSA.  NULL otherwise.
         */
        void* get_hsa_am_region() const
        {
            auto& acg = detail::Agent_pool::pool()[agent_]
                .agent_allocated_coarse_grained_region;
            if (acg.handle) return &acg;

            return &detail::Agent_pool::pool()[agent_]
                .system_coarse_grained_region;
        }

        /**
         * Returns an opaque handle which points to the AM system region on the
         * HSA agent. This region can be used to allocate system memory which is
         * accessible from the specified accelerator.
         *
         * @return An opaque handle of the region, if the accelerator is based
         *         on HSA.  NULL otherwise.
         */
        void* get_hsa_am_system_region() const
        {
            return
                &detail::Agent_pool::pool()[agent_].system_coarse_grained_region;
        }

        /**
         * Returns an opaque handle which points to the AM system region on the
         * HSA agent. This region can be used to allocate finegrained system
         * memory which is accessible from the specified accelerator.
         *
         * @return An opaque handle of the region, if the accelerator is based
         *         on HSA.  NULL otherwise.
         */
        void* get_hsa_am_finegrained_system_region() const
        {
            return &detail::Agent_pool::pool()[agent_].fine_grained_region;
        }

        /**
         * Returns an opaque handle which points to the Kernarg region on the
         * HSA agent.
         *
         * @return An opaque handle of the region, if the accelerator is based
         *         on HSA.  NULL otherwise.
         */
        void* get_hsa_kernarg_region() const
        {   // TODO: fix
            return nullptr;
        }

        /**
         * Returns if the accelerator is based on HSA.
         */
        bool is_hsa_accelerator() const
        {
            return true;
        }

        /**
         * Returns the profile the accelerator.
         * - accelerator_profile_none in case the accelerator is not based on
         *   HSA.
         * - accelerator_profile_base in case the accelerator implements the HSA
         *   Base Profile.
         * - accelerator_profile_full in case the accelerator implements the HSA
         *   Full Profile.
         */
        accelerator_profile get_profile() const
        {
            return detail::Agent_pool::pool()[agent_].profile;
        }

        /**
         * Returns an opaque handle which points to the underlying HSA agent.
         *
         * @return An opaque handle of the underlying HSA agent, if the
         *         accelerator is based on HSA. NULL otherwise.
         */
        void* get_hsa_agent() const
        {   // TODO: redo, should return the handle directly.
            return const_cast<hsa_agent_t*>(&agent_);
        }

        /**
         * Check if @p other is peer of this accelerator.
         *
         * @return true if other can access this accelerator's device memory
         * pool or false if not. The accelerator is not its own peer.
         */
        bool get_is_peer(const accelerator& other) const
        {
            if (*this == other) return true;

            hsa_amd_memory_pool_access_t r{};
            detail::throwing_hsa_result_check(
                hsa_amd_agent_memory_pool_get_info(
                    *static_cast<hsa_agent_t*>(other.get_hsa_agent()),
                    *static_cast<hsa_amd_memory_pool_t*>(get_hsa_am_region()),
                    HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                    &r),
                __FILE__, __func__, __LINE__);

            return r != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
        }

        /**
         * Return a std::vector of this accelerator's peers. peer is other
         * accelerator which can access this accelerator's device memory using
         * map_to_peer family of APIs.
         */
        std::vector<accelerator> get_peers() const
        {   // TODO: remove / optimise.
            std::vector<accelerator> peers;

            static const auto accs = get_all();
            for (auto&& acc : accs) if (get_is_peer(acc)) peers.push_back(acc);

            return peers;
        }

        /**
         * Return the compute unit count of the accelerator.
         */
        unsigned int get_cu_count() const
        {
            return detail::Agent_pool::pool()[agent_].compute_unit_count;
        }

        /**
         * Return the unique integer sequence-number for the accelerator.
         * Sequence-numbers are assigned in monotonically increasing order
         * starting with 0.
         */
        int get_seqnum() const
        {
            return INT_MAX;
        }


        /**
         * Return true if the accelerator's memory can be mapped into the CPU's
         * address space, and the CPU is allowed to access the memory directly
         * with CPU memory operations. Typically this is enabled with
         * "large BAR" or "resizeable BAR" address mapping.
         */
        bool has_cpu_accessible_am() const
        {   // TODO: fix.
            return detail::Agent_pool::pool()[agent_]
                .has_cpu_accessible_agent_allocated_coarse_grained;
        }
    };


    inline
    accelerator accelerator_view::get_accelerator() const
    {
        if (accelerator_) return *accelerator_;

        throw std::logic_error{
            "Tried to query accelerator from empty accelerator_view."};
    }

    // ------------------------------------------------------------------------
    // member function implementations
    // ------------------------------------------------------------------------

    // TODO: move this into accelerator_view's definition
    inline
    void accelerator_view::wait_for_all_pending_tasks_()
    {   // TODO: this is overly conservative, technically we only need to wait
        //       for the eldest i.e. first in the list, then it should be legal
        //       to clean up.
        for (auto&& task : pending_tasks_) if (task.valid()) task.wait();

        pending_tasks_.clear();
    }

    inline
    completion_future accelerator_view::create_marker(memory_scope) const
    {
        pending_tasks_.push_front(detail::insert_barrier(*this));

        return pending_tasks_.front();
    }

    inline
    completion_future accelerator_view::create_blocking_marker(
        completion_future& dependent_future, memory_scope) const
    {
        pending_tasks_.push_front(completion_future{
            std::async([=]() { dependent_future.wait(); }).share()});

        return pending_tasks_.front();
    }

    // TODO: constrain to take completion_future only.
    template<typename InputIterator>
    inline
    completion_future accelerator_view::create_blocking_marker(
        InputIterator first, InputIterator last, memory_scope) const
    {   // TODO: optimise by nesting the hsa_signal_t inside the
        //       completion_future and then building AND AQL packets.
        std::vector<completion_future> tmp{first, last};
        pending_tasks_.push_front(completion_future{
            std::async([tmp = std::move(tmp)]() {
                for (auto&& x : tmp) if (x.valid()) x.wait();
            }).share()});

        return pending_tasks_.front();
    }

    inline
    completion_future accelerator_view::create_blocking_marker(
        std::initializer_list<completion_future> dependent_future_list,
        memory_scope) const
    {
        return create_blocking_marker(
            dependent_future_list.begin(), dependent_future_list.end());
    }

    inline
    bool accelerator_view::set_cu_mask(const std::vector<bool>& cu_mask)
    {
        const auto agent =
            *static_cast<hsa_agent_t*>(accelerator_->get_hsa_agent());
        const auto cnt = detail::Agent_pool::pool()[agent].compute_unit_count;

        if (cnt == 0) return false;

        static const auto round_up_to_next_multiple_of_32 = [](std::size_t x) {
            x = x + 32 - 1;
            return x - x % 32;
        };

        std::vector<std::uint32_t> mask{cu_mask.cbegin(), cu_mask.cend()};
        mask.resize(round_up_to_next_multiple_of_32(cnt));

        detail::throwing_hsa_result_check(
            hsa_amd_queue_cu_set_mask(queue_, mask.size(), mask.data()),
            __FILE__, __func__, __LINE__);

        return true; // Unclear how this failing could be anything but an error.
    }

    // ------------------------------------------------------------------------
    // extent
    // ------------------------------------------------------------------------

    /**
     * Represents a unique position in N-dimensional space.
     *
     * @tparam N The dimension to this extent applies. Special constructors are
     *           supplied for the cases where @f$N \in \{ 1,2,3 \}@f$, but N can
     *           be any integer greater than or equal to 1.
     */
    template<int N>
    class extent {
        static_assert(N > 0, "Dimensionality must be positive");

        using base =
            detail::index_impl<typename detail::__make_indices<N>::type>;
        base base_;

        template<int, typename> friend struct detail::index_helper;
        template<int, typename, typename> friend struct detail::amp_helper;
    public:
        /**
         * A static member of extent<N> that contains the rank of this extent.
         */
        static constexpr int rank = N;

        /**
         * The element type of extent<N>.
         */
        typedef int value_type;

        /**
         * Default constructor. The value at each dimension is initialized to
         * zero. Thus, "extent<3> ix;" initializes the variable to the position
         * (0,0,0).
         */
        extent() [[cpu, hc]] = default;

        /**
         * Copy constructor. Constructs a new extent<N> from the supplied
         * argument.
         *
         * @param other An object of type extent<N> from which to initialize
         *              this new extent.
         */
        extent(const extent&) [[cpu, hc]] = default;

        /** @{ */
        /**
         * Constructs an extent<N> with the coordinate values provided by
         * @f$e_{0..2}@f$. These are specialized constructors that are only
         * valid when the rank of the extent @f$N \in \{1,2,3\}@f$. Invoking a
         * specialized constructor whose argument @f$count \ne N@f$ will result
         * in a compilation error.
         *
         * @param[in] e0 The component values of the extent vector.
         */
        template<
            typename... Ts,
            typename std::enable_if<sizeof...(Ts) == N>::type* = nullptr>
        explicit
        extent(Ts... i_n) [[cpu, hc]] : base_{i_n...}
        {
            static_assert(
                sizeof...(Ts) <= 3,
                "Can only supply at most 3 individual coordinates in the "
                "constructor.");
        }

        /** @} */

        /**
         * Constructs an extent<N> with the coordinate values provided the array
         * of int component values. If the coordinate array length @f$\ne@f$ N,
         * the behavior is undefined. If the array value is NULL or not a valid
         * pointer, the behavior is undefined.
         *
         * @param[in] components An array of N int values.
         */
        explicit
        extent(const int components[]) [[cpu, hc]] : base_{components} {}

        /**
         * Constructs an extent<N> with the coordinate values provided the array
         * of int component values. If the coordinate array length @f$\ne@f$ N,
         * the behavior is undefined. If the array value is NULL or not a valid
         * pointer, the behavior is undefined.
         *
         * @param[in] components An array of N int values.
         */
        explicit
        extent(int components[]) [[cpu, hc]] : base_{components} {}

        /**
         * Assigns the component values of "other" to this extent<N> object.
         *
         * @param[in] other An object of type extent<N> from which to copy into
         *                  this extent.
         * @return Returns *this.
         */
        extent& operator=(const extent&) [[cpu, hc]] = default;

        /** @{ */
        /**
         * Returns the extent component value at position c.
         *
         * @param[in] c The dimension axis whose coordinate is to be accessed.
         * @return A the component value at position c.
         */
        int operator[](unsigned int c) const [[cpu, hc]]
        {
            return base_[c];
        }
        int& operator[](unsigned int c) [[cpu, hc]]
        {
            return base_[c];
        }

        /** @} */

        /**
         * Tests whether the index "idx" is properly contained within this
         * extent (with an assumed origin of zero).
         *
         * @param[in] idx An object of type index<N>
         * @return Returns true if the "idx" is contained within the space
         *         defined by this extent (with an assumed origin of zero).
         */
        bool contains(const index<N>& idx) const noexcept [[cpu, hc]]
        {
            return detail::amp_helper<N, index<N>, extent<N>>::contains(
                idx, *this);
        }

        /**
         * This member function returns the total linear size of this extent<N>
         * (in units of elements), which is computed as:
         * extent[0] * extent[1] ... * extent[N-1]
         */
        unsigned int size() const noexcept [[cpu, hc]]
        {
            return detail::index_helper<N, extent<N>>::count_size(*this);
        }

        /** @{ */
        /**
         * Produces a tiled_extent object with the tile extents given by t0, t1,
         * and t2.
         *
         * tile(t0, t1, t2) is only supported on extent<1>. It will produce a
         * compile-time error if used on an extent where N @f$\ne@f$ 3.
         * tile(t0, t1) is only supported on extent<2>. It will produce a
         * compile-time error if used on an extent where N @f$\ne@f$ 2.
         * tile(t0) is only supported on extent<1>. It will produce a
         * compile-time error if used on an extent where N @f$\ne@f$ 1.
         */
        tiled_extent<1> tile(int t0) const [[cpu, hc]];
        tiled_extent<2> tile(int t0, int t1) const [[cpu, hc]];
        tiled_extent<3> tile(int t0, int t1, int t2) const [[cpu, hc]];

        /** @} */

        /** @{ */
        /**
         * Produces a tiled_extent object with the tile extents given by t0, t1,
         * and t2, plus a certain amount of dynamic group segment.
         */
        tiled_extent<1> tile_with_dynamic(
            int t0, unsigned int dynamic_size) const;
        tiled_extent<2> tile_with_dynamic(
            int t0, int t1, unsigned int dynamic_size) const;
        tiled_extent<3> tile_with_dynamic(
            int t0, int t1, int t2, unsigned int dynamic_size) const;

        /** @} */

        /** @{ */
        /**
         * Compares two objects of extent<N>.
         *
         * The expression
         * leftExt @f$\oplus@f$ rightExt
         * is true if leftExt[i] @f$\oplus@f$ rightExt[i] for every i from 0 to
         * N-1.
         *
         * @param[in] other The right-hand extent<N> to be compared.
         */
        bool operator==(const extent& other) const [[cpu, hc]]
        {
            return detail::index_helper<N, extent<N> >::equal(*this, other);
        }
        bool operator!=(const extent& other) const [[cpu, hc]]
        {
            return !(*this == other);
        }

        /** @} */

        /** @{ */
        /**
         * Adds (or subtracts) an object of type extent<N> from this extent to
         * form a new extent. The result extent<N> is such that for a given
         * operator @f$\oplus@f$,
         * result[i] = this[i] @f$\oplus@f$ ext[i]
         *
         * @param[in] ext The right-hand extent<N> to be added or subtracted.
         */
        extent& operator+=(const extent& __r) [[cpu, hc]]
        {
            base_.operator+=(__r.base_);
            return *this;
        }
        extent& operator-=(const extent& __r) [[cpu, hc]]
        {
            base_.operator-=(__r.base_);
            return *this;
        }
        extent& operator*=(const extent& __r) [[cpu, hc]]
        {
            base_.operator*=(__r.base_);
            return *this;
        }
        extent& operator/=(const extent& __r) [[cpu, hc]]
        {
            base_.operator/=(__r.base_);
            return *this;
        }
        extent& operator%=(const extent& __r) [[cpu, hc]]
        {
            base_.operator%=(__r.base_);
            return *this;
        }

        /** @} */

        /** @{ */
        /**
         * Adds (or subtracts) an object of type index<N> from this extent to
         * form a new extent. The result extent<N> is such that for a given
         * operator @f$\oplus@f$,
         * result[i] = this[i] @f$\oplus@f$ idx[i]
         *
         * @param[in] idx The right-hand index<N> to be added or subtracted.
         */
        extent operator+(const index<N>& idx) const [[cpu, hc]]
        {
            extent __r = *this;
            __r += idx;
            return __r;
        }
        extent operator-(const index<N>& idx) const [[cpu, hc]]
        {
            extent __r = *this;
            __r -= idx;
            return __r;
        }
        extent& operator+=(const index<N>& idx) [[cpu, hc]]
        {
            base_.operator+=(idx.base_);
            return *this;
        }
        extent& operator-=(const index<N>& idx) [[cpu, hc]]
        {
            base_.operator-=(idx.base_);
            return *this;
        }

        /** @} */

        /** @{ */
        /**
         * For a given operator @f$\oplus@f$, produces the same effect as
         * (*this) = (*this) @f$\oplus@f$ value
         *
         * The return value is "*this".
         *
         * @param[in] value The right-hand int of the arithmetic operation.
         */
        extent& operator+=(int value) [[cpu, hc]]
        {
            base_.operator+=(value);
            return *this;
        }
        extent& operator-=(int value) [[cpu, hc]]
        {
            base_.operator-=(value);
            return *this;
        }
        extent& operator*=(int value) [[cpu, hc]]
        {
            base_.operator*=(value);
            return *this;
        }
        extent& operator/=(int value) [[cpu, hc]]
        {
            base_.operator/=(value);
            return *this;
        }
        extent& operator%=(int value) [[cpu, hc]]
        {
            base_.operator%=(value);
            return *this;
        }

        /** @} */

        /** @{ */
        /**
         * For a given operator @f$\oplus@f$, produces the same effect as
         * (*this) = (*this) @f$\oplus@f$ 1
         *
         * For prefix increment and decrement, the return value is "*this".
         * Otherwise a new extent<N> is returned.
         */
        extent& operator++() [[cpu, hc]]
        {
            base_.operator+=(1);
            return *this;
        }
        extent operator++(int) [[cpu, hc]]
        {
            extent ret = *this;
            base_.operator+=(1);
            return ret;
        }
        extent& operator--() [[cpu, hc]]
        {
            base_.operator-=(1);
            return *this;
        }
        extent operator--(int) [[cpu, hc]]
        {
            extent ret = *this;
            base_.operator-=(1);
            return ret;
        }

        /** @} */
    };

    // ------------------------------------------------------------------------
    // global functions for extent
    // ------------------------------------------------------------------------

    /** @{ */
    /**
     * Adds (or subtracts) two objects of extent<N> to form a new extent. The
     * result extent<N> is such that for a given operator @f$\oplus@f$,
     * result[i] = leftExt[i] @f$\oplus@f$ rightExt[i]
     * for every i from 0 to N-1.
     *
     * @param[in] lhs The left-hand extent<N> to be compared.
     * @param[in] rhs The right-hand extent<N> to be compared.
     */
    // FIXME: the signature is not entirely the same as defined in:
    //        C++AMP spec v1.2 #1253
    template<int N>
    inline
    extent<N> operator+(const extent<N>& lhs, const extent<N>& rhs) [[cpu, hc]]
    {
        extent<N> __r = lhs;
        __r += rhs;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator-(const extent<N>& lhs, const extent<N>& rhs) [[cpu, hc]]
    {
        extent<N> __r = lhs;
        __r -= rhs;
        return __r;
    }

    /** @} */

    /** @{ */
    /**
     * Binary arithmetic operations that produce a new extent<N> that is the
     * result of performing the corresponding binary arithmetic operation on the
     * elements of the extent operands. The result extent<N> is such that for a
     * given operator @f$\oplus@f$,
     * result[i] = ext[i] @f$\oplus@f$ value
     * or
     * result[i] = value @f$\oplus@f$ ext[i]
     * for every i from 0 to N-1.
     *
     * @param[in] ext The extent<N> operand
     * @param[in] value The integer operand
     */
    // FIXME: the signature is not entirely the same as defined in:
    //        C++AMP spec v1.2 #1259
    template<int N>
    inline
    extent<N> operator+(const extent<N>& ext, int value) [[cpu, hc]]
    {
        extent<N> __r = ext;
        __r += value;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator+(int value, const extent<N>& ext) [[cpu, hc]]
    {
        extent<N> __r = ext;
        __r += value;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator-(const extent<N>& ext, int value) [[cpu, hc]]
    {
        extent<N> __r = ext;
        __r -= value;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator-(int value, const extent<N>& ext) [[cpu, hc]]
    {
        extent<N> __r(value);
        __r -= ext;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator*(const extent<N>& ext, int value) [[cpu, hc]]
    {
        extent<N> __r = ext;
        __r *= value;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator*(int value, const extent<N>& ext) [[cpu, hc]]
    {
        extent<N> __r = ext;
        __r *= value;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator/(const extent<N>& ext, int value) [[cpu, hc]]
    {
        extent<N> __r = ext;
        __r /= value;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator/(int value, const extent<N>& ext) [[cpu, hc]]
    {
        extent<N> __r(value);
        __r /= ext;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator%(const extent<N>& ext, int value) [[cpu, hc]]
    {
        extent<N> __r = ext;
        __r %= value;
        return __r;
    }
    template<int N>
    inline
    extent<N> operator%(int value, const extent<N>& ext) [[cpu, hc]]
    {
        extent<N> __r(value);
        __r %= ext;
        return __r;
    }

    /** @} */

    // ------------------------------------------------------------------------
    // tiled_extent
    // ------------------------------------------------------------------------

    /**
     * Represents an extent subdivided into tiles.
     * Tile sizes can be specified at runtime.
     *
     * @tparam N The dimension of the extent and the tile.
     */
    template<int n>
    class tiled_extent : public extent<n> {
        std::uint32_t dynamic_group_segment_size_{};
    public:
        static constexpr int rank{n};

        /**
         * Tile size for each dimension.
         */
        const int tile_dim[n]{};

        // CREATORS
        /**
         * Default constructor. The origin and extent is default-constructed and
         * thus zero.
         */
        tiled_extent() [[cpu, hc]] = default;

        /**
         * Copy constructor. Constructs a new tiled_extent from the supplied
         * argument "other".
         *
         * @param[in] other An object of type tiled_extent from which to
         *                  initialize this new extent.
         */
        tiled_extent(const tiled_extent&) [[cpu, hc]] = default;
        tiled_extent(tiled_extent&&) [[cpu, hc]] = default;

        /**
         * Construct an tiled extent with the size of extent and the size of
         * tile specified.
         *
         * @param[in] e# Size of extent in the #th dimension.
         * @param[in] t# Size of tile in the #th dimension.
         */
        template<int m = n, typename std::enable_if<m == 1>::type* = nullptr>
        tiled_extent(int e0, int t0) [[cpu, hc]] : tiled_extent{e0, t0, 0u}
        {}

        template<int m = n, typename std::enable_if<m == 2>::type* = nullptr>
        tiled_extent(int e0, int e1, int t0, int t1) [[cpu, hc]]
            : tiled_extent{e0, e1, t0, t1, 0u}
        {}

        template<int m = n, typename std::enable_if<m == 3>::type* = nullptr>
        tiled_extent(int e0, int e1, int e2, int t0, int t1, int t2) [[cpu, hc]]
            : tiled_extent{e0, e1, e2, t0, t1, t2, 0u}
        {}

        /**
         * Construct an tiled extent with the size of extent and the size of
         * tile specified.
         *
         * @param[in] e# Size of extent in the #th dimension.
         * @param[in] t# Size of tile in the #th dimension.
         * @param[in] size Size of dynamic group segment.
         */
        template<int m = n, typename std::enable_if<m == 1>::type* = nullptr>
        tiled_extent(int e0, int t0, std::uint32_t size) [[cpu, hc]]
            : tiled_extent{hc::extent<n>{e0}, t0, size}
        {}

        template<int m = n, typename std::enable_if<m == 2>::type* = nullptr>
        tiled_extent(
            int e0, int e1, int t0, int t1, std::uint32_t size) [[cpu, hc]]
            : tiled_extent{hc::extent<n>{e0, e1}, t0, t1, size}
        {}

        template<int m = n, typename std::enable_if<m == 3>::type* = nullptr>
        tiled_extent(
            int e0,
            int e1,
            int e2,
            int t0,
            int t1,
            int t2,
            std::uint32_t size) [[cpu, hc]]
            : tiled_extent{hc::extent<n>{e0, e1, e2}, t0, t1, t2, size}
        {}

        /**
         * Constructs a tiled_extent<N> with the extent "ext".
         *
         * @param[in] ext The extent of this tiled_extent
         * @param[in] ts... Size of tile in dimensions....
         */
        template<   // TODO: tighten constraint.
            typename... Ts,
            typename std::enable_if<sizeof...(Ts) == n>::type* = nullptr>
        tiled_extent(const extent<n>& ext, Ts... ts) [[cpu, hc]]
            : tiled_extent{ext, ts..., 0u}
        {}

        /**
         * Constructs a tiled_extent<N> with the extent "ext".
         *
         * @param[in] ext The extent of this tiled_extent
         * @param[in] t# Size of tile in the #th dimension.
         * @param[in] size Size of dynamic group segment
         */
        template<int m = n, typename std::enable_if<m == 1>::type* = nullptr>
        tiled_extent(
            const hc::extent<n>& ext, int t0, std::uint32_t size) [[cpu, hc]]
            : extent<n>{ext}, dynamic_group_segment_size_{size}, tile_dim{t0}
        {}

        template<int m = n, typename std::enable_if<m == 2>::type* = nullptr>
        tiled_extent(
            const hc::extent<n>& ext,
            int t0,
            int t1,
            std::uint32_t size) [[cpu, hc]]
            :
            extent<n>{ext}, dynamic_group_segment_size_{size}, tile_dim{t0, t1}
        {}

        template<int m = n, typename std::enable_if<m == 3>::type* = nullptr>
        tiled_extent(
            const hc::extent<n>& ext,
            int t0,
            int t1,
            int t2,
            std::uint32_t size) [[cpu, hc]]
            :
            extent<n>{ext},
            dynamic_group_segment_size_{size},
            tile_dim{t0, t1, t2}
        {}

        // MANIPULATORS
        void set_dynamic_group_segment_size(std::uint32_t size) noexcept [[cpu]]
        {
            dynamic_group_segment_size_ = size;
        }

        // ACCESSORS
        /**
         * Return the size of dynamic group segment in bytes.
         */
        std::uint32_t get_dynamic_group_segment_size() const noexcept [[cpu]]
        {
            return dynamic_group_segment_size_;
        }

        tiled_extent pad() const noexcept [[cpu, hc]]
        {
            static const auto round_up_to_next_multiple = [](int x, int y) {
                x = x + y - 1;
                return x - x % y;
            };

            tiled_extent tmp{*this};
            for (auto i = 0; i != n; ++i) {
                tmp[i] = round_up_to_next_multiple(tmp[i], tile_dim[i]);
            }

            return tmp;
        }

        tiled_extent truncate() const noexcept [[cpu, hc]]
        {
            static const auto round_down_to_previous_multiple =
                [](int x, int y) { return x - x % y; };

            tiled_extent tmp{*this};
            for (auto i = 0; i != n; ++i) {
                tmp[i] = round_down_to_previous_multiple(tmp[i], tile_dim[i]);
            }

            return tmp;
        }
    };

    // ------------------------------------------------------------------------
    // implementation of extent<N>::tile()
    // ------------------------------------------------------------------------

    template <int N>
    inline
    tiled_extent<1> extent<N>::tile(int t0) const [[cpu, hc]]
    {
        static_assert(
            N == 1,
            "One-dimensional tile() method only available on extent<1>");
        return tiled_extent<1>{*this, t0};
    }

    template <int N>
    inline
    tiled_extent<2> extent<N>::tile(int t0, int t1) const [[cpu, hc]]
    {
        static_assert(
            N == 2,
            "Two-dimensional tile() method only available on extent<2>");
        return tiled_extent<2>{*this, t0, t1};
    }

    template <int N>
    inline
    tiled_extent<3> extent<N>::tile(int t0, int t1, int t2) const [[cpu, hc]]
    {
        static_assert(
            N == 3,
            "Three-dimensional tile() method only available on extent<3>");
        return tiled_extent<3>{*this, t0, t1, t2};
    }

    // ------------------------------------------------------------------------
    // implementation of extent<N>::tile_with_dynamic()
    // ------------------------------------------------------------------------

    template <int N>
    inline
    tiled_extent<1> extent<N>::tile_with_dynamic(
        int t0, unsigned int dynamic_size) const [[cpu, hc]]
    {
        static_assert(
            N == 1,
            "One-dimensional tile() method only available on extent<1>");
        return tiled_extent<1>{*this, t0, dynamic_size};
    }

    template <int N>
    inline
    tiled_extent<2> extent<N>::tile_with_dynamic(
        int t0, int t1, unsigned int dynamic_size) const [[cpu, hc]]
    {
        static_assert(
            N == 2,
            "Two-dimensional tile() method only available on extent<2>");
        return tiled_extent<2>{*this, t0, t1, dynamic_size};
    }

    template <int N>
    inline
    tiled_extent<3> extent<N>::tile_with_dynamic(
        int t0, int t1, int t2, unsigned int dynamic_size) const [[cpu, hc]]
    {
        static_assert(
            N == 3,
            "Three-dimensional tile() method only available on extent<3>");
        return tiled_extent<3>{*this, t0, t1, t2, dynamic_size};
    }

    // ------------------------------------------------------------------------
    // Intrinsic functions for HSAIL instructions
    // ------------------------------------------------------------------------

    /**
     * Fetch the size of a wavefront
     *
     * @return The size of a wavefront.
     */
    static constexpr auto __HSA_WAVEFRONT_SIZE__ = 64;

    extern "C"
    constexpr
    unsigned int __wavesize() [[hc]];
    #if __hcc_backend__ == HCC_BACKEND_AMDGPU
        extern "C"
        constexpr
        inline
        unsigned int __wavesize() [[hc]]
        {
            return __HSA_WAVEFRONT_SIZE__;
        }
    #endif

    /**
     * Count number of 1 bits in the input
     *
     * @param[in] input An unsigned 32-bit integer.
     * @return Number of 1 bits in the input.
     */
    extern "C"
    inline
    unsigned int __popcount_u32_b32(unsigned int input) [[hc]]
    {
        return __builtin_popcount(input);
    }

    /**
     * Count number of 1 bits in the input
     *
     * @param[in] input An unsigned 64-bit integer.
     * @return Number of 1 bits in the input.
     */
    extern "C"
    inline
    unsigned int __popcount_u32_b64(unsigned long long int input) [[hc]]
    {
        return __builtin_popcountl(input);
    }

    /** @{ */
    /**
     * Extract a range of bits
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    inline
    unsigned int __bitextract_u32(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]]
    {
        uint32_t offset = src1 & 31;
        uint32_t width = src2 & 31;
        return width == 0 ? 0 : (src0 << (32 - offset - width)) >> (32 - width);
    }

    extern "C"
    inline
    std::uint64_t __bitextract_u64(
        std::uint64_t src0, unsigned int src1, unsigned int src2) [[hc]]
    {
        uint64_t offset = src1 & 63;
        uint64_t width = src2 & 63;
        return width == 0 ? 0 : (src0 << (64 - offset - width)) >> (64 - width);
    }

    extern "C"
    int __bitextract_s32(int src0, unsigned int src1, unsigned int src2) [[hc]];

    extern "C"
    std::int64_t __bitextract_s64(
        std::int64_t src0, unsigned int src1, unsigned int src2) [[hc]];
    /** @} */

    /** @{ */
    /**
     * Replace a range of bits
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    inline
    unsigned int __bitinsert_u32(
        unsigned int src0,
        unsigned int src1,
        unsigned int src2,
        unsigned int src3) [[hc]]
    {
        uint32_t offset = src2 & 31;
        uint32_t width = src3 & 31;
        uint32_t mask = (1 << width) - 1;
        return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
    }

    extern "C"
    inline
    std::uint64_t __bitinsert_u64(
        std::uint64_t src0,
        std::uint64_t src1,
        unsigned int src2,
        unsigned int src3) [[hc]]
    {
        uint64_t offset = src2 & 63;
        uint64_t width = src3 & 63;
        uint64_t mask = (1 << width) - 1;
        return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
    }

    extern "C"
    int __bitinsert_s32(
        int src0, int src1, unsigned int src2, unsigned int src3) [[hc]];

    extern "C"
    std::int64_t __bitinsert_s64(
        std::int64_t src0,
        std::int64_t src1,
        unsigned int src2,
        unsigned int src3) [[hc]];
    /** @} */

    /** @{ */
    /**
     * Create a bit mask that can be used with bitselect
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    unsigned int __bitmask_b32(unsigned int src0, unsigned int src1) [[hc]];

    extern "C"
    std::uint64_t __bitmask_b64(unsigned int src0, unsigned int src1) [[hc]];
    /** @} */

    /** @{ */
    /**
     * Reverse the bits
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a>
     * for more detailed specification of these functions.
     */

    unsigned int __bitrev_b32(
        unsigned int src0) [[hc]] __asm("llvm.bitreverse.i32");

    std::uint64_t __bitrev_b64(
        std::uint64_t src0) [[hc]] __asm("llvm.bitreverse.i64");

    /** @} */

    /** @{ */
    /**
     * Do bit field selection
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    inline
    unsigned int __bitselect_b32(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]]
    {
        return (src1 & src0) | (src2 & ~src0);
    }

    extern "C"
    inline
    std::uint64_t __bitselect_b64(
        std::uint64_t src0, std::uint64_t src1, std::uint64_t src2) [[hc]]
    {
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
    extern "C"
    inline
    unsigned int __firstbit_u32_u32(unsigned int input) [[hc]]
    {
        return input == 0 ? -1 : __builtin_clz(input);
    }

    /**
     * Count leading zero bits in the input
     *
     * @param[in] input An unsigned 64-bit integer.
     * @return Number of 0 bits until a 1 bit is found, counting start from the
     *         most significant bit. -1 if there is no 0 bit.
     */
    extern "C"
    inline
    unsigned int __firstbit_u32_u64(unsigned long long int input) [[hc]]
    {
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
    extern "C"
    inline
    unsigned int __firstbit_u32_s32(int input) [[hc]]
    {
        if (input == 0) {
            return -1;
        }

        return input > 0 ?
            __firstbit_u32_u32(input) : __firstbit_u32_u32(~input);
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
    extern "C"
    inline
    unsigned int __firstbit_u32_s64(long long int input) [[hc]]
    {
        if (input == 0) {
            return -1;
        }

        return input > 0 ?
            __firstbit_u32_u64(input) : __firstbit_u32_u64(~input);
    }

    /** @{ */
    /**
     * Find the first bit set to 1 in a number starting from the least
     * significant bit
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/bit_string.htm">HSA PRM 5.7</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    inline
    unsigned int __lastbit_u32_u32(unsigned int input) [[hc]]
    {
        return input == 0 ? -1 : __builtin_ctz(input);
    }

    extern "C"
    inline
    unsigned int __lastbit_u32_u64(unsigned long long int input) [[hc]]
    {
        return input == 0 ? -1 : __builtin_ctzl(input);
    }

    extern "C"
    inline
    unsigned int __lastbit_u32_s32(int input) [[hc]]
    {
        return __lastbit_u32_u32(input);
    }

    extern "C"
    inline unsigned int __lastbit_u32_s64(unsigned long long input) [[hc]]
    {
        return __lastbit_u32_u64(input);
    }
    /** @} */

    /** @{ */
    /**
     * Copy and interleave the lower half of the elements from
     * each source into the destination
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/packed_data.htm">HSA PRM 5.9</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    unsigned int __unpacklo_u8x4(unsigned int src0, unsigned int src1) [[hc]];

    extern "C"
    std::uint64_t __unpacklo_u8x8(
        std::uint64_t src0, std::uint64_t src1) [[hc]];

    extern "C"
    unsigned int __unpacklo_u16x2(unsigned int src0, unsigned int src1) [[hc]];

    extern "C"
    std::uint64_t __unpacklo_u16x4(
        std::uint64_t src0, std::uint64_t src1) [[hc]];

    extern "C"
    std::uint64_t __unpacklo_u32x2(
        std::uint64_t src0, std::uint64_t src1) [[hc]];

    extern "C"
    int __unpacklo_s8x4(int src0, int src1) [[hc]];

    extern "C"
    std::int64_t __unpacklo_s8x8(std::int64_t src0, std::int64_t src1) [[hc]];

    extern "C"
    int __unpacklo_s16x2(int src0, int src1) [[hc]];

    extern "C"
    std::int64_t __unpacklo_s16x4(std::int64_t src0, std::int64_t src1) [[hc]];

    extern "C"
    std::int64_t __unpacklo_s32x2(std::int64_t src0, std::int64_t src1) [[hc]];
    /** @} */

    /** @{ */
    /**
     * Copy and interleave the upper half of the elements from
     * each source into the destination
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/packed_data.htm">HSA PRM 5.9</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    unsigned int __unpackhi_u8x4(unsigned int src0, unsigned int src1) [[hc]];

    extern "C"
    std::uint64_t __unpackhi_u8x8(
        std::uint64_t src0, std::uint64_t src1) [[hc]];

    extern "C"
    unsigned int __unpackhi_u16x2(unsigned int src0, unsigned int src1) [[hc]];

    extern "C"
    std::uint64_t __unpackhi_u16x4(
        std::uint64_t src0, std::uint64_t src1) [[hc]];

    extern "C"
    std::uint64_t __unpackhi_u32x2(
        std::uint64_t src0, std::uint64_t src1) [[hc]];

    extern "C"
    int __unpackhi_s8x4(int src0, int src1) [[hc]];

    extern "C"
    std::int64_t __unpackhi_s8x8(std::int64_t src0, std::int64_t src1) [[hc]];

    extern "C"
    int __unpackhi_s16x2(int src0, int src1) [[hc]];

    extern "C"
    std::int64_t __unpackhi_s16x4(std::int64_t src0, std::int64_t src1) [[hc]];

    extern "C"
    std::int64_t __unpackhi_s32x2(std::int64_t src0, std::int64_t src1) [[hc]];
    /** @} */

    /** @{ */
    /**
     * Assign the elements of the packed value in src0, replacing
     * the element specified by src2 with the value from src1
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/packed_data.htm">HSA PRM 5.9</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    unsigned int __pack_u8x4_u32(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];

    extern "C"
    std::uint64_t __pack_u8x8_u32(
        std::uint64_t src0, unsigned int src1, unsigned int src2) [[hc]];

    extern "C"
    unsigned __pack_u16x2_u32(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];

    extern "C"
    std::uint64_t __pack_u16x4_u32(
        std::uint64_t src0, unsigned int src1, unsigned int src2) [[hc]];

    extern "C"
    std::uint64_t __pack_u32x2_u32(
        std::uint64_t src0, unsigned int src1, unsigned int src2) [[hc]];

    extern "C"
    int __pack_s8x4_s32(int src0, int src1, unsigned int src2) [[hc]];

    extern "C"
    std::int64_t __pack_s8x8_s32(
        std::int64_t src0, int src1, unsigned int src2) [[hc]];

    extern "C"
    int __pack_s16x2_s32(int src0, int src1, unsigned int src2) [[hc]];

    extern "C"
    std::int64_t __pack_s16x4_s32(
        std::int64_t src0, int src1, unsigned int src2) [[hc]];

    extern "C"
    std::int64_t __pack_s32x2_s32(
        std::int64_t src0, int src1, unsigned int src2) [[hc]];

    extern "C"
    double __pack_f32x2_f32(double src0, float src1, unsigned int src2) [[hc]];
    /** @} */

    /** @{ */
    /**
     * Assign the elements specified by src1 from the packed value in src0
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/packed_data.htm">HSA PRM 5.9</a>
     * for more detailed specification of these functions.
     */
    extern "C"
    unsigned int __unpack_u32_u8x4(unsigned int src0, unsigned int src1) [[hc]];

    extern "C"
    unsigned int __unpack_u32_u8x8(uint64_t src0, unsigned int src1) [[hc]];

    extern "C"
    unsigned int __unpack_u32_u16x2(
        unsigned int src0, unsigned int src1) [[hc]];

    extern "C"
    unsigned int __unpack_u32_u16x4(
        std::uint64_t src0, unsigned int src1) [[hc]];

    extern "C"
    unsigned int __unpack_u32_u32x2(
        std::uint64_t src0, unsigned int src1) [[hc]];

    extern "C"
    int __unpack_s32_s8x4(int src0, unsigned int src1) [[hc]];

    extern "C"
    int __unpack_s32_s8x8(std::int64_t src0, unsigned int src1) [[hc]];

    extern "C"
    int __unpack_s32_s16x2(int src0, unsigned int src1) [[hc]];

    extern "C"
    int __unpack_s32_s16x4(std::int64_t src0, unsigned int src1) [[hc]];

    extern "C"
    int __unpack_s32_s3x2(std::int64_t src0, unsigned int src1) [[hc]];

    extern "C"
    float __unpack_f32_f32x2(double src0, unsigned int src1) [[hc]];
    /** @} */

    /**
     * Align 32 bits within 64 bits of data on an arbitrary bit boundary
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a>
     * for more detailed specification.
     */
    extern "C"
    unsigned int __bitalign_b32(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];

    /**
     * Align 32 bits within 64 bis of data on an arbitrary byte boundary
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a>
     * for more detailed specification.
     */
    extern "C"
    unsigned int __bytealign_b32(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];

    /**
     * Do linear interpolation and computes the unsigned 8-bit average of packed
     * data
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a>
     * for more detailed specification.
     */
    extern "C"
    unsigned int __lerp_u8x4(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];

    /**
     * Takes four floating-point number, convers them to unsigned integer
     * values, and packs them into a packed u8x4 value
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a>
     * for more detailed specification.
     */
    extern "C"
    unsigned int __packcvt_u8x4_f32(
        float src0, float src1, float src2, float src3) [[hc]];

    /**
     * Unpacks a single element from a packed u8x4 value and converts it to an
     * f32.
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a>
     * for more detailed specification.
     */
    extern "C"
    float __unpackcvt_f32_u8x4(unsigned int src0, unsigned int src1) [[hc]];

    /** @{ */
    /**
     * Computes the sum of the absolute differences of src0 and src1 and then
     * adds src2 to the result
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a> 
     * for more detailed specification.
     */
    extern "C"
    unsigned int __sad_u32_u32(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];

    extern "C"
    unsigned int __sad_u32_u16x2(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];

    extern "C"
    unsigned int __sad_u32_u8x4(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];
    /** @} */

    /**
     * This function is mostly the same as sad except the sum of absolute
     * differences is added to the most significant 16 bits of the result
     *
     * Please refer to <a href="http://www.hsafoundation.com/html/Content/PRM/Topics/05_Arithmetic/multimedia.htm">HSA PRM 5.15</a>
     * for more detailed specification.
     */
    extern "C"
    unsigned int __sadhi_u16x2_u8x4(
        unsigned int src0, unsigned int src1, unsigned int src2) [[hc]];

    /**
     * Get system timestamp
     */
    extern "C"
    std::uint64_t __clock_u64() [[hc]];

    /**
     * Get hardware cycle count
     *
     * Notice the return value of this function is implementation defined.
     */
    extern "C"
    std::uint64_t __cycle_u64() [[hc]];

    /**
     * Get the count of the number of earlier (in flattened
     * work-item order) active work-items within the same wavefront.
     *
     * @return The result will be in the range 0 to WAVESIZE - 1.
     */
    extern "C"
    unsigned int __activelaneid_u32() [[hc]];

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
    extern "C"
    std::uint64_t __activelanemask_v4_b64_b1(unsigned int input) [[hc]];

    /**
     * Count the number of active work-items in the current
     * wavefront that have a non-zero input.
     *
     * @param[in] input An unsigned 32-bit integer.
     * @return The number of active work-items in the current wavefront that
     *         have a non-zero input.
     */
    extern "C"
    inline
    unsigned int __activelanecount_u32_b1(unsigned int input) [[hc]]
    {
        return  __popcount_u32_b64(__activelanemask_v4_b64_b1(input));
    }

    // ------------------------------------------------------------------------
    // Wavefront Vote Functions
    // ------------------------------------------------------------------------

    /**
     * Evaluate predicate for all active work-items in the wavefront and return
     * non-zero if and only if predicate evaluates to non-zero for any of them.
     */
    extern "C"
    bool __ockl_wfany_i32(int) [[hc]];
    extern "C"
    inline
    int __any(int predicate) [[hc]]
    {
        return __ockl_wfany_i32(predicate);
    }

    /**
     * Evaluate predicate for all active work-items in the wavefront and return
     * non-zero if and only if predicate evaluates to non-zero for all of them.
     */
    extern "C"
    bool __ockl_wfall_i32(int) [[hc]];
    extern "C"
    inline
    int __all(int predicate) [[hc]]
    {
        return __ockl_wfall_i32(predicate);
    }

    /**
     * Evaluate predicate for all active work-items in the wavefront and return
     * an integer whose Nth bit is set if and only if predicate evaluates to
     * non-zero for the Nth work-item of the wavefront and the Nth work-item is
     * active.
     */

    // XXX from llvm/include/llvm/IR/InstrTypes.h
    __attribute__((convergent))
    unsigned long long __llvm_amdgcn_icmp_i32(
        unsigned int x,
        unsigned int y,
        unsigned int z) [[hc]] __asm("llvm.amdgcn.icmp.i32");
    extern "C"
    inline
    std::uint64_t __ballot(int predicate) [[hc]]
    {
        static constexpr unsigned int ICMP_NE = 33;
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
     * Work-items may only read data from another work-item which is active in
     * the current wavefront. If the target work-item is inactive, the retrieved
     * value is fixed as 0.
     *
     * The function returns the value of var held by the work-item whose ID is
     * given by srcLane. If width is less than __HSA_WAVEFRONT_SIZE__ then each
     * subsection of the wavefront behaves as a separate entity with a starting
     * logical work-item ID of 0. If srcLane is outside the range [0:width-1],
     * the value returned corresponds to the value of var held by:
     * srcLane modulo width (i.e. within the same subsection).
     *
     * The optional width parameter must have a value which is a power of 2;
     * results are undefined if it is not a power of 2, or is number greater
     * than __HSA_WAVEFRONT_SIZE__.
     */

    #if __hcc_backend__ == HCC_BACKEND_AMDGPU
        /*
        * FIXME: We need to add __builtin_amdgcn_mbcnt_{lo,hi} to clang and call
        * them here instead.
        */

        int __amdgcn_mbcnt_lo(
            int mask, int src) [[hc]] __asm("llvm.amdgcn.mbcnt.lo");
        int __amdgcn_mbcnt_hi(
            int mask, int src) [[hc]] __asm("llvm.amdgcn.mbcnt.hi");

        inline
        int __lane_id(void) [[hc]]
        {
            int lo = __amdgcn_mbcnt_lo(-1, 0);
            return __amdgcn_mbcnt_hi(-1, lo);
        }
    #endif

    #if __hcc_backend__ == HCC_BACKEND_AMDGPU
        /**
         * ds_bpermute intrinsic
         * FIXME: We need to add __builtin_amdgcn_ds_bpermute to clang and call
         * it here instead.
         */
        int __amdgcn_ds_bpermute(
            int index, int src) [[hc]] __asm("llvm.amdgcn.ds.bpermute");
        inline
        unsigned int __amdgcn_ds_bpermute(int index, unsigned int src) [[hc]]
        {
            __u tmp; tmp.u = src;
            tmp.i = __amdgcn_ds_bpermute(index, tmp.i);
            return tmp.u;
        }
        inline
        float __amdgcn_ds_bpermute(int index, float src) [[hc]]
        {
            __u tmp; tmp.f = src;
            tmp.i = __amdgcn_ds_bpermute(index, tmp.i);
            return tmp.f;
        }

        /**
         * ds_permute intrinsic
         */
        extern "C"
        int __amdgcn_ds_permute(int index, int src) [[hc]];
        inline
        unsigned int __amdgcn_ds_permute(int index, unsigned int src) [[hc]]
        {
            __u tmp; tmp.u = src;
            tmp.i = __amdgcn_ds_permute(index, tmp.i);
            return tmp.u;
        }
        inline
        float __amdgcn_ds_permute(int index, float src) [[hc]]
        {
            __u tmp; tmp.f = src;
            tmp.i = __amdgcn_ds_permute(index, tmp.i);
            return tmp.f;
        }

        /**
         * ds_swizzle intrinsic
         */
        extern "C"
        int __amdgcn_ds_swizzle(int src, int pattern) [[hc]];
        inline
        unsigned int __amdgcn_ds_swizzle(unsigned int src, int pattern) [[hc]]
        {
            __u tmp; tmp.u = src;
            tmp.i = __amdgcn_ds_swizzle(tmp.i, pattern);
            return tmp.u;
        }
        inline
        float __amdgcn_ds_swizzle(float src, int pattern) [[hc]]
        {
            __u tmp; tmp.f = src;
            tmp.i = __amdgcn_ds_swizzle(tmp.i, pattern);
            return tmp.f;
        }

        /**
         * move DPP intrinsic
         */
        extern "C"
        int __amdgcn_move_dpp(
            int src,
            int dpp_ctrl,
            int row_mask,
            int bank_mask,
            bool bound_ctrl) [[hc]];

        /**
         * Shift the value of src to the right by one thread within a wavefront.
         *
         * @param[in] src variable being shifted
         * @param[in] bound_ctrl When set to true, a zero will be shifted into
         *                       thread 0; otherwise, the original value will be
         *                       returned for thread 0
         * @return value of src being shifted into from the neighboring lane
         *
         */
        extern "C"
        int __amdgcn_wave_sr1(int src, bool bound_ctrl) [[hc]];
        inline
        unsigned int __amdgcn_wave_sr1(unsigned int src, bool bound_ctrl) [[hc]]
        {
            __u tmp; tmp.u = src;
            tmp.i = __amdgcn_wave_sr1(tmp.i, bound_ctrl);
            return tmp.u;
        }
        inline
        float __amdgcn_wave_sr1(float src, bool bound_ctrl) [[hc]]
        {
            __u tmp; tmp.f = src;
            tmp.i = __amdgcn_wave_sr1(tmp.i, bound_ctrl);
            return tmp.f;
        }

        /**
         * Shift the value of src to the left by one thread within a wavefront.
         *
         * @param[in] src variable being shifted
         * @param[in] bound_ctrl When set to true, a zero will be shifted into
         *                       thread 63; otherwise, the original value will
         *                       be returned for thread 63
         * @return value of src being shifted into from the neighboring lane
         *
         */
        extern "C"
        int __amdgcn_wave_sl1(int src, bool bound_ctrl) [[hc]];
        inline
        unsigned int __amdgcn_wave_sl1(unsigned int src, bool bound_ctrl) [[hc]]
        {
            __u tmp; tmp.u = src;
            tmp.i = __amdgcn_wave_sl1(tmp.i, bound_ctrl);
            return tmp.u;
        }
        inline
        float __amdgcn_wave_sl1(float src, bool bound_ctrl) [[hc]]
        {
            __u tmp; tmp.f = src;
            tmp.i = __amdgcn_wave_sl1(tmp.i, bound_ctrl);
            return tmp.f;
        }

        /**
         * Rotate the value of src to the right by one thread within a
         * wavefront.
         *
         * @param[in] src variable being rotated
         * @return value of src being rotated into from the neighboring lane
         *
         */
        extern "C"
        int __amdgcn_wave_rr1(int src) [[hc]];
        inline
        unsigned int __amdgcn_wave_rr1(unsigned int src) [[hc]]
        {
            __u tmp; tmp.u = src;
            tmp.i = __amdgcn_wave_rr1(tmp.i);
            return tmp.u;
        }
        inline
        float __amdgcn_wave_rr1(float src) [[hc]]
        {
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
        extern "C"
        int __amdgcn_wave_rl1(int src) [[hc]];
        inline
        unsigned int __amdgcn_wave_rl1(unsigned int src) [[hc]]
        {
            __u tmp; tmp.u = src;
            tmp.i = __amdgcn_wave_rl1(tmp.i);
            return tmp.u;
        }
        inline
        float __amdgcn_wave_rl1(float src) [[hc]]
        {
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

    #if __hcc_backend__ == HCC_BACKEND_AMDGPU
        inline
        int __shfl(
            int var, int srcLane, int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            int self = __lane_id();
            int index = srcLane + (self & ~(width-1));
            return __amdgcn_ds_bpermute(index<<2, var);
        }

        inline
        unsigned int __shfl(
            unsigned int var,
            int srcLane,
            int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            __u tmp; tmp.u = var;
            tmp.i = __shfl(tmp.i, srcLane, width);
            return tmp.u;
        }


        inline
        float __shfl(
            float var, int srcLane, int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            __u tmp; tmp.f = var;
            tmp.i = __shfl(tmp.i, srcLane, width);
            return tmp.f;
        }
    #endif

    // FIXME: support half type
    /** @} */

    /** @{ */
    /**
     * Copy from an active work-item with lower ID relative to caller within a
     * wavefront.
     *
     * Work-items may only read data from another work-item which is active in
     * the current wavefront. If the target work-item is inactive, the retrieved
     * value is fixed as 0.
     *
     * The function calculates a source work-item ID by subtracting delta from
     * the caller's work-item ID within the wavefront. The value of var held by
     * the resulting lane ID is returned: in effect, var is shifted up the
     * wavefront by delta work-items. If width is less than
     * __HSA_WAVEFRONT_SIZE__ then each subsection of the wavefront behaves as a
     * separate entity with a starting logical work-item ID of 0. The source
     * work-item index will not wrap around the value of width, so effectively
     * the lower delta work-items will be unchanged.
     *
     * The optional width parameter must have a value which is a power of 2;
     * results are undefined if it is not a power of 2, or is number greater
     * than __HSA_WAVEFRONT_SIZE__.
     */

    #if __hcc_backend__ == HCC_BACKEND_AMDGPU
        inline
        int __shfl_up(
            int var,
            unsigned int delta,
            int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            int self = __lane_id();
            int index = self - delta;
            index = (index < (self & ~(width-1)))?self:index;
            return __amdgcn_ds_bpermute(index<<2, var);
        }

        inline
        unsigned int __shfl_up(
            unsigned int var,
            unsigned int delta,
            int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            __u tmp; tmp.u = var;
            tmp.i = __shfl_up(tmp.i, delta, width);
            return tmp.u;
        }

        inline
        float __shfl_up(
            float var,
            unsigned int delta,
            int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
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
     * Work-items may only read data from another work-item which is active in
     * the current wavefront. If the target work-item is inactive, the retrieved
     * value is fixed as 0.
     *
     * The function calculates a source work-item ID by adding delta from the
     * caller's work-item ID within the wavefront. The value of var held by the
     * resulting lane ID is returned: this has the effect of shifting var up the
     * wavefront by delta work-items. If width is less than
     * __HSA_WAVEFRONT_SIZE__ then each subsection of the wavefront behaves as a
     * separate entity with a starting logical work-item ID of 0. The ID number
     * of the source work-item index will not wrap around the value of width, so
     * the upper delta work-items will remain unchanged.
     *
     * The optional width parameter must have a value which is a power of 2;
     * results are undefined if it is not a power of 2, or is number greater
     * than __HSA_WAVEFRONT_SIZE__.
     */

    #if __hcc_backend__ == HCC_BACKEND_AMDGPU
        inline
        int __shfl_down(
            int var,
            unsigned int delta,
            int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            int self = __lane_id();
            int index = self + delta;
            index = (int)((self&(width-1))+delta) >= width?self:index;
            return __amdgcn_ds_bpermute(index<<2, var);
        }

        inline
        unsigned int __shfl_down(
            unsigned int var,
            unsigned int delta,
            int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            __u tmp; tmp.u = var;
            tmp.i = __shfl_down(tmp.i, delta, width);
            return tmp.u;
        }

        inline
        float __shfl_down(
            float var,
            unsigned int delta,
            int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            __u tmp; tmp.f = var;
            tmp.i = __shfl_down(tmp.i, delta, width);
            return tmp.f;
        }
    #endif

    // FIXME: support half type
    /** @} */

    /** @{ */
    /**
     * Copy from an active work-item based on bitwise XOR of caller work-item ID
     * within a wavefront.
     *
     * Work-items may only read data from another work-item which is active in
     * the current wavefront. If the target work-item is inactive, the retrieved
     * value is fixed as 0.
     *
     * THe function calculates a source work-item ID by performing a bitwise XOR
     * of the caller's work-item ID with laneMask: the value of var held by the
     * resulting work-item ID is returned.
     *
     * The optional width parameter must have a value which is a power of 2;
     * results are undefined if it is not a power of 2, or is number greater
     * than __HSA_WAVEFRONT_SIZE__.
     */

    #if __hcc_backend__ == HCC_BACKEND_AMDGPU
        inline
        int __shfl_xor(
            int var, int laneMask, int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            int self = __lane_id();
            int index = self^laneMask;
            index = index >= ((self+width)&~(width-1))?self:index;
            return __amdgcn_ds_bpermute(index<<2, var);
        }

        inline
        float __shfl_xor(
            float var, int laneMask, int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            __u tmp; tmp.f = var;
            tmp.i = __shfl_xor(tmp.i, laneMask, width);
            return tmp.f;
        }

        // FIXME: support half type
        /** @} */

        inline
        unsigned int __shfl_xor(
            unsigned int var,
            int laneMask,
            int width = __HSA_WAVEFRONT_SIZE__) [[hc]]
        {
            __u tmp; tmp.u = var;
            tmp.i = __shfl_xor(tmp.i, laneMask, width);
            return tmp.u;
        }
    #endif

    /**
     * Multiply two unsigned integers (x,y) but only the lower 24 bits will be
     * used in the multiplication.
     *
     * @param[in] x 24-bit unsigned integer multiplier
     * @param[in] y 24-bit unsigned integer multiplicand
     * @return 32-bit unsigned integer product
     */
    inline
    unsigned int __mul24(unsigned int x, unsigned int y) [[hc]]
    {
        return (x & 0x00FFFFFF) * (y & 0x00FFFFFF);
    }

    /**
     * Multiply two integers (x,y) but only the lower 24 bits will be used in
     * the multiplication.
     *
     * @param[in] x 24-bit integer multiplier
     * @param[in] y 24-bit integer multiplicand
     * @return 32-bit integer product
     */
    inline
    int __mul24(int x, int y) [[hc]]
    {
        return  ((x << 8) >> 8) * ((y << 8) >> 8);
    }

    /**
     * Multiply two unsigned integers (x,y) but only the lower 24 bits will be
     * used in the multiplication and then add the product to a 32-bit unsigned
     * integer
     *
     * @param[in] x 24-bit unsigned integer multiplier
     * @param[in] y 24-bit unsigned integer multiplicand
     * @param[in] z 32-bit unsigned integer to be added to the product
     * @return 32-bit unsigned integer result of mad24
     */
    inline
    unsigned int __mad24(unsigned int x, unsigned int y, unsigned int z) [[hc]]
    {
        return __mul24(x,y) + z;
    }

    /**
     * Multiply two integers (x,y) but only the lower 24 bits will be used in
     * the multiplication and then add the product to a 32-bit integer
     *
     * @param[in] x 24-bit integer multiplier
     * @param[in] y 24-bit integer multiplicand
     * @param[in] z 32-bit integer to be added to the product
     * @return 32-bit integer result of mad24
     */
    inline
    int __mad24(int x, int y, int z) [[hc]]
    {
        return __mul24(x,y) + z;
    }

    inline
    void abort() [[hc]]
    {
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
    extern "C" unsigned int get_group_segment_size() [[hc]];

    /**
     * Fetch the size of static group segment
     *
     * @return The size of static group segment used by the kernel in bytes.
     */
    extern "C" unsigned int get_static_group_segment_size() [[hc]];

    /**
     * Fetch the address of the beginning of group segment.
     */
    extern "C" void* get_group_segment_base_pointer() [[hc]];

    /**
     * Fetch the address of the beginning of dynamic group segment.
     */
    extern "C" void* get_dynamic_group_segment_base_pointer() [[hc]];

    // ------------------------------------------------------------------------
    // tiled_barrier
    // ------------------------------------------------------------------------

    /**
     * The tile_barrier class is a capability class that is only creatable by
     * the system, and passed to a tiled parallel_for_each function object as
     * part of the tiled_index parameter. It provides member functions, such as
     * wait, whose purpose is to synchronize execution of threads running within
     * the thread tile.
     */
    class tile_barrier {
    public:
        /**
         * Copy constructor. Constructs a new tile_barrier from the supplied
         * argument "other".
         *
         * @param[in] other An object of type tile_barrier from which to
         *                  initialize this.
         */
        tile_barrier(const tile_barrier&) [[cpu, hc]] = default;

        /**
         * Blocks execution of all threads in the thread tile until all threads
         * in the tile have reached this call. Establishes a memory fence on all
         * tile_static and global memory operations executed by the threads in
         * the tile such that all memory operations issued prior to hitting the
         * barrier are visible to all other threads after the barrier has
         * completed and none of the memory operations occurring after the
         * barrier are executed before hitting the barrier. This is identical to
         * wait_with_all_memory_fence().
         */
        void wait() const noexcept [[hc]]
        {
            wait_with_all_memory_fence();
        }

        /**
         * Blocks execution of all threads in the thread tile until all threads in
         * the tile have reached this call. Establishes a memory fence on all
         * tile_static and global memory operations executed by the threads in the
         * tile such that all memory operations issued prior to hitting the barrier
         * are visible to all other threads after the barrier has completed and
         * none of the memory operations occurring after the barrier are executed
         * before hitting the barrier. This is identical to wait().
         */
        void wait_with_all_memory_fence() const noexcept [[hc]]
        {
            hc_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }

        /**
         * Blocks execution of all threads in the thread tile until all threads
         * in the tile have reached this call. Establishes a memory fence on
         * global memory operations (but not tile-static memory operations)
         * executed by the threads in the tile such that all global memory
         * operations issued prior to hitting the barrier are visible to all
         * other threads after the barrier has completed and none of the global
         * memory operations occurring after the barrier are executed before
         * hitting the barrier.
         */
        void wait_with_global_memory_fence() const noexcept [[hc]]
        {
            hc_barrier(CLK_GLOBAL_MEM_FENCE);
        }

        /**
         * Blocks execution of all threads in the thread tile until all threads
         * in the tile have reached this call. Establishes a memory fence on
         * tile-static memory operations (but not global memory operations)
         * executed by the threads in the tile such that all tile_static memory
         * operations issued prior to hitting the barrier are visible to all
         * other threads after the barrier has completed and none of the
         * tile-static memory operations occurring after the barrier are
         * executed before hitting the barrier.
         */
        void wait_with_tile_static_memory_fence() const [[hc]] {
            hc_barrier(CLK_LOCAL_MEM_FENCE);
        }

    private:
        tile_barrier() [[hc]] = default;

        template <int N> friend
            class tiled_index;
    };

    // ------------------------------------------------------------------------
    // other memory fences
    // ------------------------------------------------------------------------

    /**
     * Establishes a thread-tile scoped memory fence for both global and
     * tile-static memory operations. This function does not imply a barrier and
     * is therefore permitted in divergent code.
     */
    // FIXME: this functions has not been implemented.
    void all_memory_fence(const tile_barrier&) [[hc]];

    /**
     * Establishes a thread-tile scoped memory fence for global (but not
     * tile-static) memory operations. This function does not imply a barrier
     * and is therefore permitted in divergent code.
     */
    // FIXME: this functions has not been implemented.
    void global_memory_fence(const tile_barrier&) [[hc]];

    /**
     * Establishes a thread-tile scoped memory fence for tile-static (but not
     * global) memory operations. This function does not imply a barrier and is
     * therefore permitted in divergent code.
     */
    // FIXME: this functions has not been implemented.
    void tile_static_memory_fence(const tile_barrier&) [[hc]];

    // ------------------------------------------------------------------------
    // tiled_index
    // ------------------------------------------------------------------------

    /**
     * Represents a set of related indices subdivided into 1-, 2-, or
     * 3-dimensional tiles.
     *
     * @tparam n Tile dimension.
     */
    template<int n>
    class tiled_index {
        friend struct detail::Indexer;

        template<typename Kernel>
        friend
        completion_future parallel_for_each(
            const accelerator_view&, const tiled_extent<n>&, const Kernel&);

        // TODO: convert to using the hc_ flavoured functions.
        template<int m = n, typename std::enable_if<m == 1>::type* = nullptr>
        tiled_index() [[hc]]
            : global{hc_get_workitem_absolute_id(0)},
            local{hc_get_workitem_id(0)},
            tile{hc_get_group_id(0)},
            tile_origin{global[0] - local[0]},
            tile_dim{hc_get_group_size(0)}
        {}
        template<int m = n, typename std::enable_if<m == 2>::type* = nullptr>
        tiled_index() [[hc]]
            : global{
                hc_get_workitem_absolute_id(1), hc_get_workitem_absolute_id(0)},
            local{hc_get_workitem_id(1), hc_get_workitem_id(0)},
            tile{hc_get_group_id(1), hc_get_group_id(0)},
            tile_origin{global[0] - local[0], global[1] - local[1]},
            tile_dim{hc_get_group_size(1), hc_get_group_size(0)}
        {}

        template<int m = n, typename std::enable_if<m == 3>::type* = nullptr>
        tiled_index() [[hc]]
            :
            global{
                hc_get_workitem_absolute_id(2),
                hc_get_workitem_absolute_id(1),
                hc_get_workitem_absolute_id(0)},
            local{
                hc_get_workitem_id(2),
                hc_get_workitem_id(1),
                hc_get_workitem_id(0)},
            tile{hc_get_group_id(2), hc_get_group_id(1), hc_get_group_id(0)},
            tile_origin{
                global[0] - local[0],
                global[1] - local[1],
                global[2] - local[2]},
            tile_dim{
                hc_get_group_size(2),
                hc_get_group_size(1),
                hc_get_group_size(0)}
        {}
    public:
        /**
         * A static member of tiled_index that contains the rank of this tiled
         * extent, and is either 1, 2, or 3 depending on the specialization
         * used.
         */
        static constexpr int rank{n};

        tiled_index(const index<n>& g) [[cpu, hc]] : tiled_index{}
        {
            const_cast<index<n>&>(global) = g; // TODO: remove yucky cast.
        }

        /**
         * Copy constructor. Constructs a new tiled_index from the supplied
         * argument "other".
         *
         * @param[in] other An object of type tiled_index from which to
         *                  initialize this.
         */
        tiled_index(const tiled_index&) [[cpu, hc]] = default;
        tiled_index(tiled_index&&) [[cpu, hc]] = default;

        /**
         * An index of rank 1, 2, or 3 that represents the global index within
         * an extent.
         */
        const index<n> global;

        /**
         * An index of rank 1, 2, or 3 that represents the relative index within
         * the current tile of a tiled extent.
         */
        const index<n> local;

        /**
         * An index of rank 1, 2, or 3 that represents the coordinates of the
         * current tile of a tiled extent.
         */
        const index<n> tile;

        /**
         * An index of rank 1, 2, or 3 that represents the global coordinates of
         * the origin of the current tile within a tiled extent.
         */
        const index<n> tile_origin;

        /**
         * An object which represents a barrier within the current tile of
         * threads.
         */
        const tile_barrier barrier;

        /**
         * An index of rank 1, 2, 3 that represents the size of the tile.
         */
        const index<n> tile_dim;

        /**
         * Implicit conversion operator that converts a tiled_index<N> into
         * an index<N>. The implicit conversion converts to the .global index
         * member.
         */
        operator index<n>() const [[cpu, hc]]
        {
            return global;
        }
    };

    // ------------------------------------------------------------------------
    // utility helper classes for array_view
    // ------------------------------------------------------------------------

    template<typename T>
    struct __has_data {
    private:
        struct two {char __lx; char __lxx;};
        template<typename C>
        static
        char test(decltype(std::declval<C>().data()));
        template<typename C>
        static two test(...);
    public:
        static constexpr bool value = sizeof(test<T>(0)) == 1;
    };

    template<typename T>
    struct __has_size {
    private:
        struct two {char __lx; char __lxx;};
        template <typename C> static char test(decltype(&C::size));
        template <typename C> static two test(...);
    public:
        static constexpr bool value = sizeof(test<T>(0)) == 1;
    };

    template<typename T>
    struct __is_container {
        using _T = typename std::remove_reference<T>::type;
        static constexpr bool value =
            __has_size<_T>::value && __has_data<_T>::value;
    };


    // ------------------------------------------------------------------------
    // forward declarations of copy routines used by array / array_view
    // ------------------------------------------------------------------------

    template<typename T, int N>
    void copy(const array_view<const T, N>& src, const array_view<T, N>& dest);

    template<typename T, int N>
    void copy(const array_view<T, N>& src, const array_view<T, N>& dest);

    template<typename T, int N>
    void copy(const array<T, N>& src, const array_view<T, N>& dest);

    template<typename T, int N>
    void copy(const array<T, N>& src, array<T, N>& dest);

    template<typename T, int N>
    void copy(const array_view<const T, N>& src, array<T, N>& dest);

    template<typename T, int N>
    void copy(const array_view<T, N>& src, array<T, N>& dest);

    template<typename InputIter, typename T, int N>
    void copy(
        InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest);

    template<typename InputIter, typename T, int N>
    void copy(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest);

    template<typename InputIter, typename T, int N>
    void copy(InputIter srcBegin, const array_view<T, N>& dest);

    template<typename InputIter, typename T, int N>
    void copy(InputIter srcBegin, array<T, N>& dest);

    template<typename OutputIter, typename T, int N>
    void copy(const array_view<T, N> &src, OutputIter destBegin);

    template<typename OutputIter, typename T, int N>
    void copy(const array<T, N> &src, OutputIter destBegin);

    // ------------------------------------------------------------------------
    // array
    // ------------------------------------------------------------------------

    /**
     * Represents an N-dimensional region of memory (with type T) located on an
     * accelerator.
     *
     * @tparam T The element type of this array
     * @tparam N The dimensionality of the array, defaults to 1 if elided.
     */
    struct array_base {
        struct Deleter {
            template<typename T>
            void operator()(T* ptr)
            {
                if (!ptr) return;
                if (hsa_memory_free(ptr) == HSA_STATUS_SUCCESS) return;

                std::cerr << "Failed to deallocate array memory; HC runtime may"
                    << " be in an inconsistent state." << std::endl;
            }
        };
        using Guarded_locked_ptr = std::pair<
            std::atomic_flag, std::pair<const void*, void*>>;

        static constexpr std::size_t max_array_cnt_{65536u};

        inline static std::array< // TODO: this is a placeholder, and most dubious.
            std::pair<
                std::atomic<std::uint32_t>,
                std::pair<
                    std::mutex, std::forward_list<std::shared_future<void>>>>,
            max_array_cnt_> writers_{};
        inline static std::array<
            Guarded_locked_ptr, max_array_cnt_> locked_ptrs_{};
        inline thread_local static std::vector<std::size_t> captured_{};

        static
        std::size_t writers_for_()
        {
            for (decltype(writers_.size()) i = 0u; i != writers_.size(); ++i) {
                if (writers_[i].first++ == 0) return i;
                else --writers_[i].first;
            }

            throw std::runtime_error{"Failed to associate writers for array."};
        }
    };

    template <typename T, int N = 1>
    class array : private array_base {
        static_assert(!std::is_const<T>{}, "array<const T> is not supported");
        static_assert(
            std::is_trivially_copyable<T>{},
            "Only trivially copyable types are supported.");
        static_assert(
            std::is_trivially_destructible<T>{},
            "Only trivially destructible types are supported.");

        accelerator_view owner_;
        accelerator_view associate_;
        extent<N> extent_;
        access_type cpu_access_;
        std::unique_ptr<T[], Deleter> data_;
        std::size_t this_idx_{max_array_cnt_};
        std::size_t writers_for_this_{max_array_cnt_};

        template<typename U, int M>
        friend
        void copy(const array<U, M>&, array<U, M>&);
        template<typename U, int M>
        friend
        void copy(const array<U, M>&, const array_view<U, M>&);
        template<typename O, typename U, int M>
        friend
        void copy(const array<U, M>&, O);
        template<typename U, int M>
        friend
        void copy(const array<U, M>&, const array_view<U, M>&);
        template<typename U, int M>
        friend
        void copy(const array_view<const U, M>&, array<U, M>&);

        void add_to_captured_() const
        {
            captured_.push_back(writers_for_this_);
        }

        T* allocate_()
        {
            hsa_region_t* r{nullptr};
            switch (cpu_access_) {
            case access_type_none: case access_type_auto:
                r = static_cast<hsa_region_t*>(
                    owner_.get_accelerator().get_hsa_am_region());
                break;
            default:
                r = static_cast<hsa_region_t*>(
                    owner_.get_accelerator().get_hsa_am_system_region());
            }

            void* tmp{nullptr};

            auto s = hsa_memory_allocate(*r, extent_.size() * sizeof(T), &tmp);
            if (s != HSA_STATUS_SUCCESS) {
                throw std::runtime_error{"Failed to allocate array storage."};
            }

            return static_cast<T*>(tmp);
        }

        static
        constexpr
        std::uint64_t make_bitmask_(
            std::uint8_t first, std::uint8_t last) noexcept [[cpu, hc]]
        {
            return (first == last) ?
                0u : ((UINT64_MAX >> (64u - (first - last))) << last);
        }

        static
        std::uint32_t k_r_hash_(const void* ptr) [[cpu, hc]]
        {
            static constexpr auto byte_offset_bits = 2u;
            static constexpr auto set_bits = 10u;
            static constexpr auto tag_bits =
                sizeof(std::uintptr_t) * CHAR_BIT - set_bits - byte_offset_bits;

            static const auto byte_offset = [](const void* p) {
                constexpr auto mask = make_bitmask_(byte_offset_bits, 0u);

                return reinterpret_cast<std::uintptr_t>(p) & mask;
            };
            static const auto set = [](const void* p) {
                constexpr auto mask = make_bitmask_(
                    set_bits + byte_offset_bits, byte_offset_bits);

                return (reinterpret_cast<std::uintptr_t>(p) & mask) >>
                    byte_offset_bits;
            };
            static const auto tag = [](const void* p) {
                constexpr auto mask = make_bitmask_(
                    tag_bits + set_bits + byte_offset_bits,
                    set_bits + byte_offset_bits);

                return (reinterpret_cast<std::uintptr_t>(p) & mask) >>
                    (set_bits + byte_offset_bits);
            };

            return set(ptr) * (max_array_cnt_ / 1024);
        }

        std::size_t lock_this_()
        {
            const auto n = k_r_hash_(this);
            do {
                auto idx = 0;
                do {
                    idx = 0;
                    while (idx != max_array_cnt_ / 1024) {
                        if (!locked_ptrs_[n + idx].first.test_and_set()) break;
                        ++idx;
                    }
                } while (idx == max_array_cnt_ / 1024);

                auto s = hsa_amd_memory_lock(
                    this,
                    sizeof(*this),
                    static_cast<hsa_agent_t*>(
                        owner_.get_accelerator().get_hsa_agent()),
                    1,
                    reinterpret_cast<void**>(
                        &locked_ptrs_[n + idx].second.second));

                if (s != HSA_STATUS_SUCCESS) {
                    throw std::runtime_error{"Failed to lock array address."};
                }

                locked_ptrs_[n + idx].second.first = this;

                return n + idx;
            } while (true); // TODO: add termination after a number of attempts.
        }

        array* const this_() const [[hc]]
        {
            const auto n = k_r_hash_(this);

            for (auto i = 0; i != max_array_cnt_ / 1024; ++i) {
                if (locked_ptrs_[n + i].second.first != this) continue;

                return static_cast<array* const>(
                    locked_ptrs_[n + i].second.second);
            }

            return nullptr;
        }

        void wait_for_all_pending_writers_() const
        {
            decltype(writers_[writers_for_this_].second.second) tmp;
            {
                std::lock_guard<std::mutex> lck{
                    writers_[writers_for_this_].second.first};

                std::swap(tmp, writers_[writers_for_this_].second.second);
            }
            for (auto&& x : tmp) if (x.valid()) x.wait();
        }
    public:
        /**
         * The rank of this array.
         */
        static constexpr int rank = N;

        /**
         * The element type of this array.
         */
        using value_type = T;

        /**
         * There is no default constructor for array<T,N>.
         */
        array() = delete;

        /**
         * Copy constructor. Constructs a new array<T,N> from the supplied
         * argument other. The new array is located on the same accelerator_view
         * as the source array. A deep copy is performed.
         *
         * @param[in] other An object of type array<T,N> from which to
         *                  initialize this new array.
         */
        array(const array& other)
            : array{other.extent_, other.owner_, other.associate_}
        {   // TODO: if both arrays resolve to the same slot this will deadlock.
            copy(other, *this);
        }

        /**
         * Move constructor. Constructs a new array<T,N> by moving from the
         * supplied argument other.
         *
         * @param[in] other An object of type array<T,N> from which to
         *                  initialize this new array.
         */
        array(array&& other)
            :
            owner_{std::move(other.owner_)},
            associate_{std::move(other.associate_)},
            extent_{std::move(other.extent_)},
            cpu_access_{other.cpu_access_},
            data_{std::move(other.data_)},
            writers_for_this_{other.writers_for_this_}
        {
            this_idx_ = lock_this_();
            other.writers_for_this_ = max_array_cnt_;
        }

        /**
         * Constructs a new array with the supplied extent, located on the
         * default view of the default accelerator. If any components of the
         * extent are non-positive, an exception will be thrown.
         *
         * @param[in] ext The extent in each dimension of this array.
         */
        explicit
        array(const hc::extent<N>& ext)
            : array{ext, accelerator::get_auto_selection_view()}
        {}

        /** @{ */
        /**
         * Equivalent to construction using
         * "array(extent<N>(e0 [, e1 [, e2 ]]))".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array.
         */
        explicit
        array(int e0) : array{hc::extent<N>{e0}}
        {
            static_assert(N == 1, "illegal");
        }
        explicit
        array(int e0, int e1) : array{hc::extent<N>{e0, e1}}
        {
            static_assert(N == 2, "illegal");
        }
        explicit
        array(int e0, int e1, int e2) : array{hc::extent<N>{e0, e1, e2}}
        {
            static_assert(N == 3, "illegal");
        }

        /** @} */

        /** @{ */
        /**
         * Constructs a new array with the supplied extent, located on the
         * default accelerator, initialized with the contents of a source
         * container specified by a beginning and optional ending iterator. The
         * source data is copied by value into this array as if by calling
         * "copy()".
         *
         * If the number of available container elements is less than
         * this->extent.size(), undefined behavior results.
         *
         * @param[in] ext The extent in each dimension of this array.
         * @param[in] srcBegin A beginning iterator into the source container.
         * @param[in] srcEnd An ending iterator into the source container.
         */
        template<typename InputIter>
        array(const hc::extent<N>& ext, InputIter srcBegin)
            : array{ext, srcBegin, accelerator::get_auto_selection_view()}
        {}
        template<typename InputIter>
        array(const hc::extent<N>& ext, InputIter srcBegin, InputIter srcEnd)
            :
            array{ext, srcBegin, srcEnd, accelerator::get_auto_selection_view()}
        {}

        /** @} */

        /** @{ */
        /**
         * Equivalent to construction using
         * "array(extent<N>(e0 [, e1 [, e2 ]]), src)".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array.
         * @param[in] srcBegin A beginning iterator into the source container.
         * @param[in] srcEnd An ending iterator into the source container.
         */
        template<typename InputIter>
        array(int e0, InputIter srcBegin)
            : array{hc::extent<N>{e0}, srcBegin}
        {}
        template<typename InputIter>
        array(int e0, InputIter srcBegin, InputIter srcEnd)
            : array{hc::extent<N>{e0}, srcBegin, srcEnd}
        {}
        template<typename InputIter>
        array(int e0, int e1, InputIter srcBegin)
            : array{hc::extent<N>{e0, e1}, srcBegin}
        {}
        template<typename InputIter>
        array(int e0, int e1, InputIter srcBegin, InputIter srcEnd)
            : array{hc::extent<N>{e0, e1}, srcBegin, srcEnd}
        {}
        template<typename InputIter>
        array(int e0, int e1, int e2, InputIter srcBegin)
            : array{hc::extent<N>{e0, e1, e2}, srcBegin}
        {}
        template<typename InputIter>
        array(int e0, int e1, int e2, InputIter srcBegin, InputIter srcEnd)
            : array{hc::extent<N>{e0, e1, e2}, srcBegin, srcEnd}
        {}

        /** @} */

        /**
         * Constructs a new array, located on the default view of the default
         * accelerator, initialized with the contents of the array_view "src".
         * The extent of this array is taken from the extent of the source
         * array_view. The "src" is copied by value into this array as if by
         * calling "copy(src, *this)".
         *
         * @param[in] src An array_view object from which to copy the data into
         *                this array (and also to determine the extent of this
         *                array).
         */
        explicit
        array(const array_view<const T, N>& src)
            : array{src.get_extent(), accelerator::get_auto_selection_view()}
        {
            copy(src, *this);
        }

        /**
         * Constructs a new array with the supplied extent, located on the
         * accelerator bound to the accelerator_view "av".
         *
         * Users can optionally specify the type of CPU access desired for
         * "this" array thus requesting creation of an array that is accessible
         * both on the specified accelerator_view "av" as well as the CPU (with
         * the specified CPU access_type). If a value other than
         * access_type_auto or access_type_none is specified for the
         * cpu_access_type parameter and the accelerator corresponding to the
         * accelerator_view "av" does not support cpu_shared_memory, a
         * runtime_exception is thrown. The cpu_access_type parameter has a
         * default value of access_type_auto which leaves it up to the
         * implementation to decide what type of allowed CPU access should the
         * array be created with. The actual CPU access_type allowed for the
         * created array can be queried using the get_cpu_access_type member
         * method.
         *
         * @param[in] ext The extent in each dimension of this array.
         * @param[in] av An accelerator_view object which specifies the location
         *               of this array.
         * @param[in] access_type The type of CPU access desired for this array.
         */
        array(
            const hc::extent<N>& ext,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
        try :
            owner_{std::move(av)},
            associate_{owner_},
            extent_{ext},
            cpu_access_{cpu_access_type},
            data_{allocate_(), Deleter{}},
            this_idx_{lock_this_()},
            writers_for_this_{writers_for_()}
        {}
        catch (const std::exception& ex) {
            if (ext.size() != 0) throw ex;

            throw std::domain_error{"Tried to construct zero-sized array."};
        }

        /** @{ */
        /**
         * Constructs an array instance based on the given pointer on the device
         * memory.
         */
        array(int e0, void* accelerator_pointer)
            :
            array{
                hc::extent<N>{e0},
                static_cast<T*>(accelerator_pointer),
                accelerator::get_auto_selection_view(),
                access_type_none}
        {}
        array(int e0, int e1, void* accelerator_pointer)
            :
            array{
                hc::extent<N>{e0, e1},
                static_cast<T*>(accelerator_pointer),
                accelerator::get_auto_selection_view(),
                access_type_none}
        {}
        array(int e0, int e1, int e2, void* accelerator_pointer)
            :
            array{
                hc::extent<N>{e0, e1, e2},
                static_cast<T*>(accelerator_pointer),
                accelerator::get_auto_selection_view(),
                access_type_none}
        {}

        array(const hc::extent<N>& ext, void* accelerator_pointer)
            :
            array{
                ext,
                static_cast<T*>(accelerator_pointer),
                accelerator::get_auto_selection_view(),
                access_type_none}
        {}
        /** @} */

        /**
         * Constructs an array instance based on the given pointer on the device
         * memory.
         *
         * @param[in] ext The extent in each dimension of this array.
         * @param[in] av An accelerator_view object which specifies the location
         *               of this array.
         * @param[in] accelerator_pointer The pointer to the device memory.
         * @param[in] access_type The type of CPU access desired for this array.
         */
        array(
            const extent<N>& ext,
            accelerator_view av,
            void* accelerator_pointer,
            access_type cpu_access_type = access_type_none)
            :
            array{
                ext,
                static_cast<T*>(accelerator_pointer),
                std::move(av),
                cpu_access_type}
        {
            // TODO: handle access types other than none.
        }

        /** @{ */
        /**
         * Equivalent to construction using
         * "array(extent<N>(e0 [, e1 [, e2 ]]), av, cpu_access_type)".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array.
         * @param[in] av An accelerator_view object which specifies the location
         *               of this array.
         * @param[in] access_type The type of CPU access desired for this array.
         */
        array(
            int e0,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            : array{hc::extent<N>{e0}, std::move(av), cpu_access_type}
        {}
        array(
            int e0,
            int e1,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            : array{hc::extent<N>{e0, e1}, std::move(av), cpu_access_type}
        {}
        array(
            int e0,
            int e1,
            int e2,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            : array{hc::extent<N>{e0, e1, e2}, std::move(av), cpu_access_type}
        {}

        /** @} */

        /**
         * Constructs a new array with the supplied extent, located on the
         * accelerator bound to the accelerator_view "av", initialized with the
         * contents of the source container specified by a beginning and
         * optional ending iterator. The data is copied by value into this array
         * as if by calling "copy()".
         *
         * Users can optionally specify the type of CPU access desired for
         * "this" array thus requesting creation of an array that is accessible
         * both on the specified accelerator_view "av" as well as the CPU (with
         * the specified CPU access_type). If a value other than
         * access_type_auto or access_type_none is specified for the
         * cpu_access_type parameter and the accelerator corresponding to the
         * accelerator_view "av" does not support cpu_shared_memory, a
         * runtime_exception is thrown. The cpu_access_type parameter has a
         * default value of access_type_auto which leaves it up to the
         * implementation to decide what type of allowed CPU access should the
         * array be created with. The actual CPU access_type allowed for the
         * created array can be queried using the get_cpu_access_type member
         * method.
         *
         * @param[in] ext The extent in each dimension of this array.
         * @param[in] srcBegin A beginning iterator into the source container.
         * @param[in] srcEnd An ending iterator into the source container.
         * @param[in] av An accelerator_view object which specifies the home
         *               location of this array.
         * @param[in] access_type The type of CPU access desired for this array.
         */
        template<typename InputIter>
        array(
            const hc::extent<N>& ext,
            InputIter srcBegin,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            : array{ext, std::move(av), cpu_access_type}
        {
            copy(srcBegin, *this);
        }
        template<typename InputIter>
        array(
            const hc::extent<N>& ext,
            InputIter srcBegin,
            InputIter srcEnd,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            : array{ext, std::move(av), cpu_access_type}
        {
            copy(srcBegin, srcEnd, *this);
        }

        /** @} */

        /**
         * Constructs a new array initialized with the contents of the
         * array_view "src". The extent of this array is taken from the extent
         * of the source array_view. The "src" is copied by value into this
         * array as if by calling "copy(src, *this)". The new array is located
         * on the accelerator bound to the accelerator_view "av".
         *
         * Users can optionally specify the type of CPU access desired for
         * "this" array thus requesting creation of an array that is accessible
         * both on the specified accelerator_view "av" as well as the CPU (with
         * the specified CPU access_type). If a value other than
         * access_type_auto or access_type_none is specified for the
         * cpu_access_type parameter and the accelerator corresponding to the
         * accelerator_view “av” does not support cpu_shared_memory, a
         * runtime_exception is thrown. The cpu_access_type parameter has a
         * default value of access_type_auto which leaves it up to the
         * implementation to decide what type of allowed CPU access should the
         * array be created with. The actual CPU access_type allowed for the
         * created array can be queried using the get_cpu_access_type member
         * method.
         *
         * @param[in] src An array_view object from which to copy the data into
         *                this array (and also to determine the extent of this
         *                array).
         * @param[in] av An accelerator_view object which specifies the home
         *               location of this array.
         * @param[in] access_type The type of CPU access desired for this array.
         */
        array(
            const array_view<const T, N>& src,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            : array{src.get_extent(), std::move(av), cpu_access_type}
        {
            copy(src, *this);
        }

        /** @{ */
        /**
         * Equivalent to construction using
         * "array(
         *     extent<N>(e0 [, e1 [, e2 ]]),
         *     srcBegin [, srcEnd],
         *     av,
         *     cpu_access_type)".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array.
         * @param[in] srcBegin A beginning iterator into the source container.
         * @param[in] srcEnd An ending iterator into the source container.
         * @param[in] av An accelerator_view object which specifies the home
         *               location of this array.
         * @param[in] access_type The type of CPU access desired for this array.
         */
        template<typename InputIter>
        array(
            int e0,
            InputIter srcBegin,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            : array{hc::extent<N>{e0}, srcBegin, std::move(av), cpu_access_type}
        {}
        template<typename InputIter>
        array(
            int e0,
            InputIter srcBegin,
            InputIter srcEnd,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            :
            array{
                hc::extent<N>{e0},
                srcBegin,
                srcEnd,
                std::move(av),
                cpu_access_type}
        {}
        template<typename InputIter>
        array(
            int e0,
            int e1,
            InputIter srcBegin,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            :
            array{
                hc::extent<N>{e0, e1}, srcBegin, std::move(av), cpu_access_type}
        {}
        template<typename InputIter>
        array(
            int e0,
            int e1,
            InputIter srcBegin,
            InputIter srcEnd,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            :
            array{
                hc::extent<N>{e0, e1},
                srcBegin,
                srcEnd,
                std::move(av),
                cpu_access_type}
        {}
        template<typename InputIter>
        array(
            int e0,
            int e1,
            int e2,
            InputIter srcBegin,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            :
            array{
                hc::extent<N>{e0, e1, e2},
                srcBegin,
                std::move(av),
                cpu_access_type}
        {}
        template<typename InputIter>
        array(
            int e0,
            int e1,
            int e2,
            InputIter srcBegin,
            InputIter srcEnd,
            accelerator_view av,
            access_type cpu_access_type = access_type_auto)
            :
            array{
                hc::extent<N>{e0, e1, e2},
                srcBegin,
                srcEnd,
                std::move(av),
                cpu_access_type}
        {}

        /** @} */

        /**
         * Constructs a staging array with the given extent, which acts as a
         * staging area between accelerator views "av" and "associated_av". If
         * "av" is a cpu accelerator view, this will construct a staging array
         * which is optimized for data transfers between the CPU and
         * "associated_av".
         *
         * @param[in] ext The extent in each dimension of this array.
         * @param[in] av An accelerator_view object which specifies the home
         *               location of this array.
         * @param[in] associated_av An accelerator_view object which specifies a
         *                          target device accelerator.
         */
        array(
            const hc::extent<N>& ext,
            accelerator_view av,
            accelerator_view associated_av)
        try :
            owner_{std::move(av)},
            associate_{std::move(associated_av)},
            extent_{ext},
            cpu_access_{access_type_auto},
            data_{allocate_(), Deleter{}},
            this_idx_{lock_this_()},
            writers_for_this_{writers_for_()}
        {}
        catch (const std::exception& ex) {
            if (ext.size() != 0) throw ex;

            throw std::domain_error{"Tried to construct zero-sized array."};
        }

        /** @{ */
        /**
         * Equivalent to construction using
         * "array(extent<N>(e0 [, e1 [, e2 ]]), av, associated_av)".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array.
         * @param[in] av An accelerator_view object which specifies the home
         *               location of this array.
         * @param[in] associated_av An accelerator_view object which specifies a
         *                          target device accelerator.
         */
        array(int e0, accelerator_view av, accelerator_view associated_av)
            : array{hc::extent<N>{e0}, std::move(av), associated_av}
        {}
        array(
            int e0, int e1, accelerator_view av, accelerator_view associated_av)
            : array{hc::extent<N>{e0, e1}, std::move(av), associated_av}
        {}
        array(
            int e0,
            int e1,
            int e2,
            accelerator_view av,
            accelerator_view associated_av)
            : array{hc::extent<N>{e0, e1, e2}, std::move(av), associated_av}
        {}

        /** @} */

        /** @{ */
        /**
         * Constructs a staging array with the given extent, which acts as a
         * staging area between accelerator_views "av" (which must be the CPU
         * accelerator) and "associated_av". The staging array will be
         * initialized with the data specified by "src" as if by calling
         * "copy(src, *this)".
         *
         * @param[in] ext The extent in each dimension of this array.
         * @param[in] srcBegin A beginning iterator into the source container.
         * @param[in] srcEnd An ending iterator into the source container.
         * @param[in] av An accelerator_view object which specifies the home
         *               location of this array.
         * @param[in] associated_av An accelerator_view object which specifies a
         *                          target device accelerator.
         */
        template<typename InputIter>
        array(
            const hc::extent<N>& ext,
            InputIter srcBegin,
            accelerator_view av,
            accelerator_view associated_av)
            : array{ext, std::move(av), std::move(associated_av)}
        {
            copy(srcBegin, *this);
        }
        template<typename InputIter>
        array(
            const hc::extent<N>& ext,
            InputIter srcBegin,
            InputIter srcEnd,
            accelerator_view av,
            accelerator_view associated_av)
            : array{ext, std::move(av), associated_av}
        {
            copy(srcBegin, srcEnd, *this);
        }

        /** @} */

        /**
         * Constructs a staging array initialized with the array_view given by
         * "src", which acts as a staging area between accelerator_views "av"
         * (which must be the CPU accelerator) and "associated_av". The extent
         * of this array is taken from the extent of the source array_view. The
         * staging array will be initialized from "src" as if by calling
         * "copy(src, *this)".
         *
         * @param[in] src An array_view object from which to copy the data into
         *                this array (and also to determine the extent of this
         *                array).
         * @param[in] av An accelerator_view object which specifies the home
         *               location of this array.
         * @param[in] associated_av An accelerator_view object which specifies a
         *                          target device accelerator.
         */
        array(
            const array_view<const T, N>& src,
            accelerator_view av,
            accelerator_view associated_av)
            : array{src.get_extent(), std::move(av), associated_av}
        {
            copy(src, *this);
        }

        /** @{ */
        /**
         * Equivalent to construction using
         * "array(extent<N>(e0 [, e1 [, e2 ]]), src, av, associated_av)".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array.
         * @param[in] srcBegin A beginning iterator into the source container.
         * @param[in] srcEnd An ending iterator into the source container.
         * @param[in] av An accelerator_view object which specifies the home
         *               location of this array.
         * @param[in] associated_av An accelerator_view object which specifies a
         *                          target device accelerator.
         */
        template<typename InputIter>
        array(
            int e0,
            InputIter srcBegin,
            accelerator_view av,
            accelerator_view associated_av)
            : array{hc::extent<N>{e0}, srcBegin, std::move(av), associated_av}
        {}
        template<typename InputIter>
        array(
            int e0,
            InputIter srcBegin,
            InputIter srcEnd,
            accelerator_view av,
            accelerator_view associated_av)
            :
            array{
                hc::extent<N>{e0},
                srcBegin,
                srcEnd,
                std::move(av),
                associated_av}
        {}
        template<typename InputIter>
        array(
            int e0,
            int e1,
            InputIter srcBegin,
            accelerator_view av,
            accelerator_view associated_av)
            :
            array{hc::extent<N>{e0, e1}, srcBegin, std::move(av), associated_av}
        {}
        template<typename InputIter>
        array(
            int e0,
            int e1,
            InputIter srcBegin,
            InputIter srcEnd,
            accelerator_view av,
            accelerator_view associated_av)
            :
            array{
                hc::extent<N>{e0, e1},
                srcBegin,
                srcEnd,
                std::move(av),
                associated_av}
        {}
        template<typename InputIter>
        array(
            int e0,
            int e1,
            int e2,
            InputIter srcBegin,
            accelerator_view av,
            accelerator_view associated_av)
            :
            array{
                hc::extent<N>{e0, e1, e2},
                srcBegin,
                std::move(av),
                associated_av}
        {}
        template<typename InputIter>
        array(
            int e0,
            int e1,
            int e2,
            InputIter srcBegin,
            InputIter srcEnd,
            accelerator_view av,
            accelerator_view associated_av)
            :
            array{
                hc::extent<N>{e0, e1, e2},
                srcBegin,
                srcEnd,
                std::move(av),
                associated_av}
        {}

        /** @} */

        /**
         * Access the extent that defines the shape of this array.
         */
        hc::extent<N> get_extent() const [[cpu, hc]]
        {
            return extent_;
        }

        /**
         * This property returns the accelerator_view representing the location
         * where this array has been allocated.
         */
        accelerator_view get_accelerator_view() const
        {
            return owner_;
        }

        /**
         * This property returns the accelerator_view representing the preferred
         * target where this array can be copied.
         */
        accelerator_view get_associated_accelerator_view() const
        {
            return associate_;
        }

        /**
         * This property returns the CPU "access_type" allowed for this array.
         */
        access_type get_cpu_access_type() const
        {
            return cpu_access_;
        }

        /**
         * Assigns the contents of the array "other" to this array, using a deep
         * copy.
         *
         * @param[in] other An object of type array<T,N> from which to copy into
         *                  this array.
         * @return Returns *this.
         */
        array& operator=(const array& other) {
            if (this != &other) {
                array arr(other);
                *this = std::move(arr);
            }
            return *this;
        }

        /**
         * Moves the contents of the array "other" to this array.
         *
         * @param[in] other An object of type array<T,N> from which to move into
         *                  this array.
         * @return Returns *this.
         */
        array& operator=(array&& other)
        {   // TODO: fix infinite recursion, this is temporary bad, explosive juju.
            array tmp{std::move(other)};
            std::swap(*this, tmp);

            return *this;
        }

        /**
         * Assigns the contents of the array_view "src", as if by calling
         * "copy(src, *this)".
         *
         * @param[in] src An object of type array_view<T,N> from which to copy
         *                into this array.
         * @return Returns *this.
         */
        array& operator=(const array_view<const T,N>& src)
        {
            using std::swap;

            array tmp{src};
            swap(*this, tmp);

            return *this;
        }

        /**
         * Copies the contents of this array to the array given by "dest", as
         * if by calling "copy(*this, dest)".
         *
         * @param[out] dest An object of type array<T,N> to which to copy data
         *                  from this array.
         */
        void copy_to(array& dest) const
        {
            copy(*this, dest);
        }

        /**
         * Copies the contents of this array to the array_view given by "dest",
         * as if by calling "copy(*this, dest)".
         *
         * @param[out] dest An object of type array_view<T,N> to which to copy
         *                  data from this array.
         */
        void copy_to(const array_view<T,N>& dest) const
        {
            copy(*this, dest);
        }

        /**
         * Returns a pointer to the raw data underlying this array.
         *
         * @return A pointer to the (const) first element in the linearised
         *         array.
         */
        T* data() const [[cpu, hc]]
        {
            return data_.get();
        }

        /**
         * Returns a pointer to the device memory underlying this array.
         *
         * @return A (const) pointer to the first element in the array on the
         *         device memory.
         */
        T* accelerator_pointer() const [[cpu, hc]]
        {   // TODO: this is dumb, array is an owning owned container i.e. data_
            //       IS an accelerator pointer; it is NOT array_view, and this
            //       function should be removed.
            return data_.get();
        }

        /**
         * Implicitly converts an array to a std::vector, as if by
         * "copy(*this, vector)".
         *
         * @return An object of type vector<T> which contains a copy of the data
         *         contained on the array.
         */
        operator std::vector<T>() const {
            std::vector<T> vec(extent_.size());
            hc::copy(*this, vec.data());
            return vec;
        }

        /** @{ */
        /**
         * Returns a reference to the element of this array that is at the
         * location in N-dimensional space specified by "idx". Accessing array
         * data on a location where it is not resident (e.g. from the CPU when
         * it is resident on a GPU) results in an exception (in CPU context) or
         * undefined behavior (in GPU context).
         *
         * @param[in] idx An object of type index<N> from that specifies the
         *                location of the element.
         */
        T& operator[](const index<N>& idx) [[cpu]]
        {   // TODO: simplify, this is a placeholder.
            static const accelerator cpu{L"cpu"};

            switch (cpu_access_) {
            case access_type_none:
                throw
                    runtime_exception{"The array is not accessible on CPU.", 0};
            case access_type_auto:
                if (owner_.get_accelerator() == cpu) break;
                throw
                    runtime_exception{"The array is not accessible on CPU.", 0};
            default:
                break;
            }

            return data_[detail::amp_helper<
                N, index<N>, hc::extent<N>>::flatten(idx, extent_)];
        }
        T& operator[](const index<N>& idx) [[hc]]
        {
            return this_()->data_[detail::amp_helper<
                N, index<N>, hc::extent<N>>::flatten(idx, this_()->extent_)];
        }
        template<int m = N, typename std::enable_if<m == 1>::type* = nullptr>
        T& operator[](int i0) [[cpu, hc]]
        {
            return operator[](index<1>{i0});
        }
        T& operator()(const index<N>& idx) [[cpu, hc]]
        {
            return (*this)[idx];
        }

        /** @} */

        /** @{ */
        /**
         * Returns a const reference to the element of this array that is at the
         * location in N-dimensional space specified by "idx". Accessing array
         * data on a location where it is not resident (e.g. from the CPU when
         * it is resident on a GPU) results in an exception (in cpu context) or
         * undefined behavior (in GPU context).
         *
         * @param[in] idx An object of type index<N> from that specifies the
         *                location of the element.
         */
        const T& operator[](const index<N>& idx) const [[cpu, hc]]
        {   // TODO: semi-ghastly, even though Scott Meyers approves of it.
            return (*const_cast<array* const>(this))[idx];
        }
        template<int m = N, typename std::enable_if<m == 1>::type* = nullptr>
        const T& operator[](int i0) const [[cpu, hc]]
        {
            return operator[](index<m>{i0});
        }
        const T& operator()(const index<N>& idx) const [[cpu, hc]]
        {
            return operator[](idx);
        }

        /** @} */

        /** @{ */
        /**
         * Equivalent to
         * "array<T,N>::operator()(index<N>(i0 [, i1 [, i2 ]]))".
         *
         * @param[in] i0,i1,i2 The component values that will form the index
         *                     into this array.
         */
        template<int m = N, typename std::enable_if<m == 1>::type* = nullptr>
        T& operator()(int i0) [[cpu, hc]]
        {
            return operator[](index<1>{i0});
        }
        template<int m = N, typename std::enable_if<m == 2>::type* = nullptr>
        T& operator()(int i0, int i1) [[cpu, hc]]
        {
            return operator[](index<2>{i0, i1});
        }
        template<int m = N, typename std::enable_if<m == 3>::type* = nullptr>
        T& operator()(int i0, int i1, int i2) [[cpu, hc]]
        {
            return operator[](index<3>{i0, i1, i2});
        }

        /** @} */

        /** @{ */
        /**
         * Equivalent to
         * "array<T,N>::operator()(index<N>(i0 [, i1 [, i2 ]])) const".
         *
         * @param[in] i0,i1,i2 The component values that will form the index
         *                     into this array.
         */
        template<int m = N, typename std::enable_if<m == 1>::type* = nullptr>
        const T& operator()(int i0) const [[cpu, hc]]
        {
            return (*const_cast<array* const>(this))(i0);
        }
        template<int m = N, typename std::enable_if<m == 2>::type* = nullptr>
        const T& operator()(int i0, int i1) const [[cpu, hc]]
        {
            return (*const_cast<array* const>(this))(i0, i1);
        }
        template<int m = N, typename std::enable_if<m == 3>::type* = nullptr>
        const T& operator()(int i0, int i1, int i2) const [[cpu, hc]]
        {
            return (*const_cast<array* const>(this))(i0, i1, i2);
        }

        /** @{ */
        /**
         * This overload is defined for array<T,N> where @f$N \ge 2@f$.
         * This mode of indexing is equivalent to projecting on the
         * most-significant dimension. It allows C-style indexing. For example:
         *
         * @code{.cpp}
         * array<float,4> myArray(myExtents, …);
         * myArray[index<4>(5,4,3,2)] = 7;
         * assert(myArray[5][4][3][2] == 7);
         * @endcode
         *
         * @param i0 An integer that is the index into the most-significant
         *           dimension of this array.
         * @return Returns an array_view whose dimension is one lower than that
         *         of this array.
         */
        template<int m = N, typename std::enable_if<(m > 1)>::type* = nullptr>
        array_view<T, m - 1> operator[](int i0) [[cpu, hc]]
        {
            hc::extent<m - 1> tmp;
            for (auto i = 1; i != m; ++i) tmp[i - 1] = extent_[i];

            return array_view<T, m - 1>{tmp, data() + i0 * tmp.size()};
        }

        template<int m = N, typename std::enable_if<(m > 1)>::type* = nullptr>
        array_view<const T, m - 1> operator[](int i0) const [[cpu, hc]]
        {
            hc::extent<m - 1> tmp;
            for (auto i = 1; i != m; ++i) tmp[i - 1] = extent_[i];

            return array_view<const T, m - 1>{tmp, data() + i0 * tmp.size()};
        }

        template<int m = N, typename std::enable_if<(m > 1)>::type* = nullptr>
        array_view<T, m - 1> operator()(int i0) [[cpu, hc]]
        {
            return (*this)[i0];
        }

        template<int m = N, typename std::enable_if<(m > 1)>::type* = nullptr>
        array_view<const T, m - 1> operator()(int i0) const [[cpu, hc]]
        {
            return (*this)[i0];
        }

        /** @} */

        /** @{ */
        /**
         * Returns a subsection of the source array view at the origin specified
         * by "idx" and with the extent specified by "ext".
         *
         * Example:
         * @code{.cpp}
         * array<float,2> a(extent<2>(200,100));
         * array_view<float,2> v1(a); // v1.extent = <200,100>
         * array_view<float,2> v2 =
         *     v1.section(index<2>(15,25), extent<2>(40,50));
         * assert(v2(0,0) == v1(15,25));
         * @endcode
         *
         * @param[in] origin Provides the offset/origin of the resulting
         *            section.
         * @param[in] ext Provides the extent of the resulting section.
         * @return Returns a subsection of the source array at specified origin,
         *         and with the specified extent.
         */
        array_view<T, N> section(
            const index<N>& origin, const hc::extent<N>& ext) [[cpu]]
        {
            if (extent_.size() < (ext + origin).size()) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<T, N>{*this}.section(origin, ext);
        }
        array_view<T, N> section(
            const index<N>& origin, const hc::extent<N>& ext) [[hc]]
        {
            return array_view<T, N>{*this}.section(origin, ext);
        }

        array_view<const T, N> section(
            const index<N>& origin, const hc::extent<N>& ext) const [[cpu]]
        {
            if (extent_.size() < (ext + origin).size()) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<const T, N>{*this}.section(origin, ext);
        }
        array_view<const T, N> section(
            const index<N>& origin, const hc::extent<N>& ext) const [[hc]]
        {
            return array_view<const T, N>{*this}.section(origin, ext);
        }

        /** @} */

        /** @{ */
        /**
         * Equivalent to "section(idx, this->extent – idx)".
         */
        array_view<T, N> section(const index<N>& idx) [[cpu]]
        {
            if (!extent_.contains(idx)) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<T, N>{*this}.section(idx);
        }
        array_view<T, N> section(const index<N>& idx) [[hc]]
        {
            return array_view<T, N>{*this}.section(idx);
        }

        array_view<const T, N> section(const index<N>& idx) const [[cpu]]
        {
            if (!extent_.contains(idx)) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<const T, N>{*this}.section(idx);
        }
        array_view<const T, N> section(const index<N>& idx) const [[hc]]
        {
            return array_view<const T, N>{*this}.section(idx);
        }

        /** @} */

        /** @{ */
        /**
         * Equivalent to "section(index<N>(), ext)".
         */
        array_view<T, N> section(const hc::extent<N>& ext) [[cpu, hc]]
        {
            return array_view<T, N>{*this}.section(ext);
        }
        array_view<const T, N> section(
            const hc::extent<N>& ext) const [[cpu, hc]]
        {
            return array_view<const T, N>{*this}.section(ext);
        }

        /** @} */

        /** @{ */
        /**
         * Equivalent to
         * "array<T,N>::section(
         *      index<N>{i0 [, i1 [, i2 ]]},
         *      extent<N>{e0 [, e1 [, e2 ]]}) const".
         *
         * @param[in] i0,i1,i2 The component values that will form the origin of
         *                     the section
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     the section
         */
        array_view<T, 1> section(int i0, int e0) [[cpu, hc]]
        {
            static_assert(N == 1, "Rank must be 1.");

            return section(index<1>{i0}, hc::extent<1>{e0});
        }
        array_view<T, 2> section(int i0, int i1, int e0, int e1) [[cpu, hc]]
        {
            static_assert(N == 2, "Rank must be 2.");

            return section(index<2>{i0, i1}, hc::extent<2>{e0, e1});
        }
        array_view<T, 3> section(
            int i0, int i1, int i2, int e0, int e1, int e2) [[cpu, hc]]
        {
            static_assert(N == 3, "Rank must be 3.");

            return section(index<3>{i0, i1, i2}, hc::extent<3>{e0, e1, e2});
        }

        array_view<const T, 1> section(int i0, int e0) const [[cpu, hc]]
        {
            static_assert(N == 1, "Rank must be 1.");

            return section(index<1>{i0}, hc::extent<1>{e0});
        }
        array_view<const T, 2> section(
            int i0, int i1, int e0, int e1) const [[cpu, hc]]
        {
            static_assert(N == 2, "Rank must be 2.");

            return section(index<2>{i0, i1}, hc::extent<2>{e0, e1});
        }
        array_view<const T, 3> section(
            int i0, int i1, int i2, int e0, int e1, int e2) const [[cpu, hc]]
        {
            static_assert(N == 3, "Rank must be 3.");

            return section(index<3>{i0, i1, i2}, hc::extent<3>{e0, e1, e2});
        }

        /** @} */

        /** @{ */
        /**
         * Sometimes it is desirable to view the data of an N-dimensional array
         * as a linear array, possibly with a (unsafe) reinterpretation of the
         * element type. This can be achieved through the reinterpret_as member
         * function. Example:
         *
         * @code{.cpp}
         * struct RGB { float r; float g; float b; };
         * array<RGB,3> a = ...;
         * array_view<float,1> v = a.reinterpret_as<float>();
         * assert(v.extent == 3*a.extent);
         * @endcode
         *
         * The size of the reinterpreted ElementType must evenly divide into the
         * total size of this array.
         *
         * @return Returns an array_view from this array<T,N> with the element
         *         type reinterpreted from T to ElementType, and the rank
         *         reduced from N to 1.
         */
        template<typename U>
        array_view<U, 1> reinterpret_as() [[cpu]]
        {
            int size{extent_.size() / sizeof(U) * sizeof(T)};

            if (size * sizeof(U) != extent_.size() * sizeof(T)) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<U, 1>{extent<1>{size}, data()};
        }
        template<typename U>
        array_view<U, 1> reinterpret_as() [[hc]]
        {
            int size{extent_.size() / sizeof(U) * sizeof(T)};

            return array_view<U, 1>{extent<1>{size}, data()};
        }

        template<typename U>
        array_view<const U, 1> reinterpret_as() const [[cpu]]
        {
            int size{extent_.size() / sizeof(U) * sizeof(T)};

            if (size * sizeof(U) != extent_.size() * sizeof(T)) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<const U, 1>{extent<1>{size}, data()};
        }
        template<typename U>
        array_view<const U, 1> reinterpret_as() const [[hc]]
        {
            int size{extent_.size() / sizeof(U) * sizeof(T)};

            return array_view<const U, 1>{extent<1>{size}, data()};
        }

        /** @} */

        /** @{ */
        /**
         * An array of higher rank can be reshaped into an array of lower rank,
         * or vice versa, using the view_as member function. Example:
         *
         * @code{.cpp}
         * array<float,1> a(100);
         * array_view<float,2> av = a.view_as(extent<2>(2,50));
         * @endcode
         *
         * @return Returns an array_view from this array<T,N> with the rank
         *         changed to K from N.
         */
        template<int m>
        array_view<T, m> view_as(const hc::extent<m>& view_extent) [[cpu]]
        {
            if (extent_.size() < view_extent.size()) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<T, m>{view_extent, data()};
        }
        template<int m>
        array_view<T, m> view_as(const hc::extent<m>& view_extent) [[hc]]
        {
            return array_view<T, m>{view_extent, data()};
        }

        template<int m>
        array_view<const T, m> view_as(
            const hc::extent<m>& view_extent) const [[cpu]]
        {
            if (extent_.size() < view_extent.size()) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<const T, m>{view_extent, data()};
        }
        template<int m>
        array_view<const T, m> view_as(
            const hc::extent<m>& view_extent) const [[hc]]
        {
            return array_view<const T, m>{view_extent, data()};
        }

        /** @} */

        ~array()
        {
            static constexpr auto force_emission_ = &array::add_to_captured_;

            if (writers_for_this_ != max_array_cnt_) {
                --writers_[writers_for_this_].first;
            }
            if (this_idx_ == max_array_cnt_) return;

            if (hsa_amd_memory_unlock(this) != HSA_STATUS_SUCCESS) {
                std::cerr << "Failed to unlock locked array pointer; HC runtime"
                    << " may be in an inconsistent state." << std::endl;
            }

            locked_ptrs_[this_idx_].first.clear();
        }
    };

    // ------------------------------------------------------------------------
    // array_view
    // ------------------------------------------------------------------------
    /**
     * The array_view<T, N> type represents a possibly cached view into the data
     * held in an array<T, N>, or a section thereof. It also provides such views
     * over native CPU data. It exposes an indexing interface congruent to that
     * of array<T, N>.
     */
    struct array_view_base {
        static constexpr std::size_t max_array_view_cnt_{65536};

        inline static std::array< // TODO: this is a placeholder, and most dubious.
            std::pair<
                std::atomic<std::uint32_t>,
                std::pair<std::mutex, std::forward_list<std::shared_future<void>>>>,
            max_array_view_cnt_> writers_{};
        inline static std::mutex mutex_{}; // TODO: use shared_mutex if C++17 feasible.
        inline static std::unordered_map<
            const void*, std::shared_ptr<void>> cache_{};
        inline thread_local static std::vector<std::size_t> captured_{};

        static
        const std::shared_ptr<void>& cache_for_sourceless_(
            void* ptr, std::size_t byte_cnt)
        {
            static const accelerator acc{};

            auto s = hsa_memory_allocate(
                *static_cast<hsa_region_t*>(acc.get_hsa_am_system_region()),
                byte_cnt,
                &ptr);

            if (s != HSA_STATUS_SUCCESS) {
                throw std::runtime_error{
                    "Failed cache allocation for sourceless array_view."};
            }

            std::lock_guard<std::mutex> lck{mutex_};

            return cache_.emplace(
                std::piecewise_construct, std::make_tuple(ptr),
                std::make_tuple(ptr, hsa_memory_free)).first->second;
        }

        const std::shared_ptr<void>& cache_for_(
            const void* ptr, std::size_t byte_cnt)
        {
            if (ptr == this) return cache_for_sourceless_(this, byte_cnt);

            std::lock_guard<std::mutex> lck{mutex_};

            const auto it = cache_.find(ptr);

            if (it != cache_.cend()) return it->second;

            static const accelerator acc{};

            void* tmp{nullptr};
            auto s = hsa_memory_allocate(
                *static_cast<hsa_region_t*>(acc.get_hsa_am_system_region()),
                byte_cnt,
                &tmp);

            if (s != HSA_STATUS_SUCCESS) {
                throw std::runtime_error{
                    "Failed cache allocation for array_view."};
            }

            return cache_.emplace(
                std::piecewise_construct,
                std::make_tuple(ptr),
                std::make_tuple(tmp, hsa_memory_free)).first->second;
        }

        static
        std::size_t writers_for_()
        {
            for (decltype(writers_.size()) i = 0u; i != writers_.size(); ++i) {
                if (writers_[i].first++ == 0) return i;
                else --writers_[i].first;
            }

            throw std::runtime_error{
                "Failed to associate writers for array_view."};
        }
    };

    template <typename T, int N = 1>
    class array_view : private array_view_base {
        static_assert(
            std::is_trivially_copyable<T>{},
            "Only trivially copyable types are supported.");
        static_assert(
            std::is_trivially_destructible<T>{},
            "Only trivially destructible types are supported.");

        using ValT_ = typename std::remove_const<T>::type;

        // TODO: compress data layout to make array_view more pointer like in cost.
        #if !defined(__HCC_ACCELERATOR__) // TODO: temporary, assess shared_ptr use.
            std::shared_ptr<void> data_;
        #else
            struct {
                typename std::aligned_storage<
                    sizeof(std::shared_ptr<void>),
                    alignof(std::shared_ptr<void>)>::type pad_;

                void* get() const [[cpu, hc]] { return nullptr; }
            } data_;
        #endif
        const accelerator* owner_;
        hc::extent<N> extent_;
        T* base_ptr_;
        typename std::conditional<
            std::is_const<T>{}, const void*, void*>::type source_;
        std::size_t writers_for_this_;

        template<typename, int> friend class array;
        template<typename, int> friend class array_view;

        template<typename Q, int K>
        friend
        void copy(const array<Q, K>&, const array_view<Q, K>&);
        template<typename InputIter, typename Q, int K>
        friend
        void copy(InputIter, InputIter, const array_view<Q, K>&);
        template<typename Q, int K>
        friend
        void copy(const array_view<const Q, K>&, array<Q, K>&);
        template<typename OutputIter, typename Q, int K>
        friend
        void copy(const array_view<Q, K>&, OutputIter);
        template<typename Q, int K>
        friend
        void copy(const array_view<const Q, K>&, const array_view<Q, K>&);

        T* updated_data_() const [[cpu]]
        {
            if (writers_for_this_ == max_array_view_cnt_) return base_ptr_;
            if (writers_[writers_for_this_].second.second.empty()) {
                return base_ptr_;
            }

            decltype(writers_[writers_for_this_].second.second) tmp;
            {
                std::lock_guard<std::mutex> lck{
                    writers_[writers_for_this_].second.first};

                for (auto&& x : writers_[writers_for_this_].second.second) {
                    if (!x.valid()) continue;
                    x.wait();
                }

                std::swap(writers_[writers_for_this_].second.second, tmp);
            }

            return base_ptr_;
        }
        T* updated_data_() const [[hc]]
        {
            return base_ptr_;
        }
    public:
        /**
         * The rank of this array.
         */
        static const int rank = N;

        /**
         * The element type of this array.
         */
        typedef T value_type;

        /**
         * There is no default constructor for array_view<T,N>.
         */
        array_view() = delete;

        /**
         * Constructs an array_view which is bound to the data contained in the
         * "src" array. The extent of the array_view is that of the src array,
         * and the origin of the array view is at zero.
         *
         * @param[in] src An array which contains the data that this array_view
         *                is bound to.
         */
        array_view(hc::array<T, N>& src) [[cpu]]
            : array_view{src.get_extent(), src.data()}
        {   // TODO: refactor to pass owner directly to delegated to ctor.
            static const auto accs = accelerator::get_all();

            for (auto&& acc : accs) {
                if (acc != src.get_accelerator_view().get_accelerator()) continue;

                owner_ = &acc;
                break;
            }

            copy(src, base_ptr_); // TODO: could directly re-use the array storage.
        }
        array_view(hc::array<T, N>& src) [[hc]]
            : array_view{src.get_extent(), src.data()}
        {}

        template<
            typename Container,
            typename std::enable_if<
                N == 1 && __is_container<Container>::value>::type* = nullptr>
        explicit
        array_view(Container& src) : array_view{hc::extent<1>(src.size()), src}
        {}
        template<int m>
        explicit
        array_view(value_type (&src)[m]) [[cpu, hc]]
            : array_view{hc::extent<1>{m}, src}
        {}

        /**
         * Constructs an array_view which is bound to the data contained in the
         * "src" container. The extent of the array_view is that given by the
         * "extent" argument, and the origin of the array view is at zero.
         *
         * @param[in] src A template argument that must resolve to a linear
         *                container that supports .data() and .size() members
         *                (such as std::vector or std::array)
         * @param[in] extent The extent of this array_view.
         */
        template<   // TODO: redo the type predicates.
            typename Container,
            typename std::enable_if<
                __is_container<Container>::value>::type* = nullptr>
        array_view(const hc::extent<N>& extent, Container& src)
            : array_view{extent, src.data()}
        {
            static_assert(
                std::is_same<typename Container::value_type, ValT_>::value,
                "container element type and array view element type must "
                    "match");
        }

        /**
         * Constructs an array_view which is bound to the data contained in the
         * "src" container. The extent of the array_view is that given by the
         * "extent" argument, and the origin of the array view is at zero.
         *
         * @param[in] src A pointer to the source data this array_view will bind
         *                to. If the number of elements pointed to is less than
         *                the size of extent, the behavior is undefined.
         * @param[in] ext The extent of this array_view.
         */
        array_view(const hc::extent<N>& ext, value_type* src) [[cpu]]
        try :
            data_{cache_for_(src, ext.size() * sizeof(T))},
            owner_{nullptr},
            extent_{ext},
            base_ptr_{static_cast<T*>(data_.get())},
            source_{
                (src == reinterpret_cast<value_type*>(this)) ? base_ptr_ : src},
            writers_for_this_{
                std::is_const<T>{} ? max_array_view_cnt_ : writers_for_()}
        {
            if (source_ == base_ptr_) return;

            auto s = hsa_memory_copy(
                const_cast<ValT_*>(base_ptr_), //
                source_,
                extent_.size() * sizeof(T));

            if (s == HSA_STATUS_SUCCESS) return;

            throw std::runtime_error{
                "Failed to copy source data into array_view."};
        }
        catch (const std::exception& ex) {
            if (ext.size() != 0) throw ex;

            throw
                std::domain_error{"Tried to construct zero-sized array_view."};
        }
        array_view(const hc::extent<N>& ext, value_type* src) [[hc]]
            :
            owner_{nullptr},
            extent_{ext},
            base_ptr_{src},
            source_{nullptr},
            writers_for_this_{max_array_view_cnt_}
        {}

        /**
         * Constructs an array_view which is not bound to a data source. The
         * extent of the array_view is that given by the "extent" argument, and
         * the origin of the array view is at zero. An array_view thus
         * constructed represents uninitialized data and the underlying
         * allocations are created lazily as the array_view is accessed on
         * different locations (on an accelerator_view or on the CPU).
         *
         * @param[in] ext The extent of this array_view.
         */
        explicit
        array_view(const hc::extent<N>& ext)
            : array_view{ext, reinterpret_cast<value_type*>(this)}
        {}

        /**
         * Equivalent to construction using
         * "array_view(extent<N>(e0 [, e1 [, e2 ]]), src)".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array_view.
         * @param[in] src A template argument that must resolve to a contiguous
         *                container that supports .data() and .size() members
         *                (such as std::vector or std::array)
         */
        template<
            typename Container,
            typename std::enable_if<
                N == 1 && __is_container<Container>::value>::type* = nullptr>
        array_view(int e0, Container& src)
            : array_view{hc::extent<N>{e0}, src}
        {}
        template<
            typename Container,
            typename std::enable_if<
                N == 2 && __is_container<Container>::value>::type* = nullptr>
        array_view(int e0, int e1, Container& src)
            : array_view{hc::extent<N>{e0, e1}, src}
        {}
        template<
            typename Container,
            typename std::enable_if<
                N == 3 && __is_container<Container>::value>::type* = nullptr>
        array_view(int e0, int e1, int e2, Container& src)
            : array_view{hc::extent<N>{e0, e1, e2}, src}
        {}

        /**
         * Equivalent to construction using
         * "array_view(extent<N>(e0 [, e1 [, e2 ]]), src)".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array_view.
         * @param[in] src A pointer to the source data this array_view will bind
         *                to. If the number of elements pointed to is less than
         *                the size of extent, the behavior is undefined.
         */
        template<int m = N, typename std::enable_if<m == 1>::type* = nullptr>
        array_view(int e0, value_type *src) [[cpu, hc]]
            : array_view{hc::extent<N>{e0}, src}
        {}
        template<int m = N, typename std::enable_if<m == 2>::type* = nullptr>
        array_view(int e0, int e1, value_type *src) [[cpu, hc]]
            : array_view{hc::extent<N>{e0, e1}, src}
        {}
        template<int m = N, typename std::enable_if<m == 3>::type* = nullptr>
        array_view(int e0, int e1, int e2, value_type *src) [[cpu, hc]]
            : array_view{hc::extent<N>{e0, e1, e2}, src}
        {}

        /**
         * Equivalent to construction using
         * "array_view(extent<N>(e0 [, e1 [, e2 ]]))".
         *
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     this array_view.
         */
        template<int m = N, typename std::enable_if<m == 1>::type* = nullptr>
        explicit
        array_view(int e0) : array_view{hc::extent<N>{e0}}
        {}
        template<int m = N, typename std::enable_if<m == 2>::type* = nullptr>
        array_view(int e0, int e1) : array_view{hc::extent<N>{e0, e1}}
        {}
        template<int m = N, typename std::enable_if<m == 3>::type* = nullptr>
        array_view(int e0, int e1, int e2)
            : array_view{hc::extent<N>{e0, e1, e2}}
        {}

        /**
         * Copy constructor. Constructs an array_view from the supplied argument
         * other. A shallow copy is performed.
         *
         * @param[in] other An object of type array_view<T,N> or
         *                  array_view<const T,N> from which to initialize this
         *                  new array_view.
         */
        template<
            typename U = T,
            typename std::enable_if<!std::is_const<U>{}>::type* = nullptr>
        array_view(const array_view& other) [[cpu]]
            :
            data_{other.data_},
            owner_{other.owner_},
            extent_{other.extent_},
            base_ptr_{other.base_ptr_},
            source_{other.source_},
            writers_for_this_{other.writers_for_this_}
        {   // N.B.: this is coupled with make_registered_kernel, and relies on
            //       it copying the user provided Callable.
            ++writers_[writers_for_this_].first;
            captured_.push_back(writers_for_this_);
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_const<U>{}>::type* = nullptr>
        array_view(const array_view& other) [[cpu]]
            :
            data_{other.data_},
            owner_{other.owner_},
            extent_{other.extent_},
            base_ptr_{other.base_ptr_},
            source_{other.source_},
            writers_for_this_{other.writers_for_this_}
        {
            if (writers_for_this_ == max_array_view_cnt_) return;

            // N.B.: this is coupled with make_registered_kernel, and relies on
            //       it copying the user provided Callable. It causes a spurious
            //       writer registration that inserts a needless wait; TODO - fix.
            captured_.push_back(writers_for_this_);
        }

        array_view(const array_view& other) [[hc]]
            :
            owner_{nullptr},
            extent_{other.extent_},
            base_ptr_{other.base_ptr_},
            writers_for_this_{max_array_view_cnt_}
        {}

        template<
            typename U,
            typename V = T,
            typename std::enable_if<
                !std::is_const<U>{} && std::is_const<V>{}>::type* = nullptr>
        array_view(const array_view<U, N>& other) [[cpu]]
            :
            data_{other.data_},
            owner_{other.owner_},
            extent_{other.extent_},
            base_ptr_{other.base_ptr_},
            source_{other.source_},
            writers_for_this_{other.writers_for_this_}
        {
            ++writers_[writers_for_this_].first;
        }
        template<
            typename U,
            typename V = T,
            typename std::enable_if<
                !std::is_const<U>{} && std::is_const<V>{}>::type* = nullptr>
        array_view(const array_view<U, N>& other) [[hc]]
            :
            owner_{nullptr},
            extent_{other.extent_},
            base_ptr_{other.base_ptr_},
            writers_for_this_{max_array_view_cnt_}
        {}
        /**
         * Move constructor. Constructs an array_view from the supplied argument
         * other.
         *
         * @param[in] other An object of type array_view<T,N> or
         *                  array_view<const T,N> from which to initialize this
         *                  new array_view.
         */
        array_view(array_view&& other) [[cpu, hc]]
            :
            data_{std::move(other.data_)},
            owner_{other.owner_},
            extent_{std::move(other.extent_)},
            base_ptr_{other.base_ptr_},
            source_{other.source_},
            writers_for_this_{other.writers_for_this_}
        {
            other.base_ptr_ = nullptr;
            other.source_ = nullptr;
            other.writers_for_this_ = max_array_view_cnt_;
        }

        /**
         * Access the extent that defines the shape of this array_view.
         */
        hc::extent<N> get_extent() const [[cpu, hc]]
        {
            return extent_;
        }

        /**
         * Access the accelerator_view where the data source of the array_view
         * is located.
         *
         * When the data source of the array_view is native CPU memory, the
         * method returns
         * accelerator{accelerator::cpu_accelerator}.default_view. When the data
         * source underlying the array_view is an array, the method returns the
         * accelerator_view where the source array is located.
         */
        accelerator_view get_source_accelerator_view() const
        {
            static const auto cpu_av{
                accelerator{accelerator::cpu_accelerator}.get_default_view()};

            return owner_ ? owner_->get_default_view() : cpu_av;
        }

        /**
         * Assigns the contents of the array_view "other" to this array_view,
         * using a shallow copy. Both array_views will refer to the same data.
         *
         * @param[in] other An object of type array_view<T,N> from which to copy
         *                  into this array.
         * @return Returns *this.
         */
        array_view& operator=(const array_view& other) [[cpu, hc]]
        {
            using std::swap;

            array_view tmp{other};
            swap(*this, tmp);

            return *this;
        }

        /**
         * Moves the contents of the array_view "other" to this array_view,
         * leaving "other" in a moved-from state.
         *
         * @param[in] other An object of type array_view<T,N> from which to move
         *                  into this array.
         * @return Returns *this.
         */
        array_view& operator=(array_view&& other) [[cpu]]
        {   // TODO: redo.
            using std::swap;

            swap(data_, other.data_);
            swap(owner_, other.owner_);
            swap(extent_, other.extent_);
            swap(base_ptr_, other.base_ptr_);
            swap(source_, other.source_);
            swap(writers_for_this_, other.writers_for_this_);

            return *this;
        }
        array_view& operator=(array_view&& other) [[hc]]
        {   // TODO: redo.
            using std::swap;

            swap(owner_, other.owner_);
            swap(extent_, other.extent_);
            swap(base_ptr_, other.base_ptr_);

            return *this;
        }

        /**
         * Copies the data referred to by this array_view to the array given by
         * "dest", as if by calling "copy(*this, dest)"
         *
         * @param[in] dest An object of type array <T,N> to which to copy data
         *                 from this array.
         */
        void copy_to(array<T, N>& dest) const
        {
            copy(*this, dest);
        }

        /**
         * Copies the contents of this array_view to the array_view given by
         * "dest", as if by calling "copy(*this, dest)"
         *
         * @param[in] dest An object of type array_view<T,N> to which to copy
         *                 data from this array.
         */
        void copy_to(const array_view& dest) const
        {
            copy(*this, dest);
        }

        /**
         * Returns a pointer to the first data element underlying this
         * array_view. This is only available on array_views of rank 1.
         *
         * When the data source of the array_view is native CPU memory, the
         * pointer returned by data() is valid for the lifetime of the data
         * source.
         *
         * When the data source underlying the array_view is an array, or the
         * array_view is created without a data source, the pointer returned by
         * data() in CPU context is ephemeral and is invalidated when the
         * original data source or any of its views are accessed on an
         * accelerator_view through a parallel_for_each or a copy operation.
         *
         * @return A pointer to the first element in the linearised array.
         */
        T* data() const [[cpu]]
        {
            static_assert(
                N == 1, "data() is only permissible on array views of rank 1");

            return updated_data_();
        }
        T* data() const [[hc]]
        {
            static_assert(
                N == 1, "data() is only permissible on array views of rank 1");

            return base_ptr_;
        }

        /**
         * Returns a pointer to the device memory underlying this array_view.
         *
         * @return A (const) pointer to the first element in the array_view on
         *         the device memory.
         */
        T* accelerator_pointer() const [[cpu, hc]] // TODO: this should be removed.
        {
            return base_ptr_;
        }

        /**
         * Calling this member function informs the array_view that its bound
         * memory has been modified outside the array_view interface. This will
         * render all cached information stale.
         */
        void refresh() const
        {
            static const accelerator cpu{accelerator::cpu_accelerator};

            if (owner_ && *owner_ == cpu) return;
            if (base_ptr_ == source_) return;

            auto s = hsa_memory_copy(
                const_cast<ValT_*>(base_ptr_),
                source_,
                extent_.size() * sizeof(T));
            if (s == HSA_STATUS_SUCCESS) return;

            throw std::runtime_error{"Failed to refresh cache for array_view."};
        }

        /**
         * Calling this member function synchronizes any modifications made to
         * the data underlying "this" array_view to its source data container.
         * For example, for an array_view on system memory, if the data
         * underlying the view are modified on a remote accelerator_view through
         * a parallel_for_each invocation, calling synchronize ensures that the
         * modifications are synchronized to the source data and will be visible
         * through the system memory pointer which the array_view was created
         * over.
         *
         * For writable array_view objects, callers of this functional can
         * optionally specify the type of access desired on the source data
         * container through the "type" parameter. For example specifying a
         * "access_type_read" (which is also the default value of the parameter)
         * indicates that the data has been synchronized to its source location
         * only for reading. On the other hand, specifying an access_type of
         * "access_type_read_write" synchronizes the data to its source location
         * both for reading and writing; i.e. any modifications to the source
         * data directly through the source data container are legal after
         * synchronizing the array_view with write access and before
         * subsequently accessing the array_view on another remote location.
         *
         * It is advisable to be precise about the access_type specified in the
         * synchronize call; i.e. if only write access it required, specifying
         * access_type_write may yield better performance that calling synchronize
         * with "access_type_read_write" since the later may require any
         * modifications made to the data on remote locations to be synchronized to
         * the source location, which is unnecessary if the contents are
         * intended to be overwritten without reading.
         *
         * @param[in] type An argument of type "access_type" which specifies the
         *                 type of access on the data source that the array_view
         *                 is synchronized for.
         */
        template<
            typename U = T,
            typename std::enable_if<!std::is_const<U>{}>::type* = nullptr>
        void synchronize(access_type type = access_type_read) const
        {
            if (type == access_type_none || type == access_type_write) return;

            decltype(writers_[writers_for_this_].second.second) tmp;
            {
                std::lock_guard<std::mutex> lck{
                    writers_[writers_for_this_].second.first};

                std::swap(writers_[writers_for_this_].second.second, tmp);
            }
            for (auto&& x : tmp) if (x.valid()) x.wait();

            if (source_ == base_ptr_) return;

            auto s = hsa_memory_copy(
                source_, base_ptr_, extent_.size() * sizeof(T));

            if (s == HSA_STATUS_SUCCESS) return;

            throw std::runtime_error{"Failed to synchronise array_view."};
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_const<U>{}>::type* = nullptr>
        void synchronize(access_type = access_type_read) const
        {
            return;
        }

        /**
         * An asynchronous version of synchronize, which returns a completion
         * future object. When the future is ready, the synchronization
         * operation is complete.
         *
         * @return An object of type completion_future that can be used to
         *         determine the status of the asynchronous operation or can be
         *         used to chain other operations to be executed after the
         *         completion of the asynchronous operation.
         */
        completion_future synchronize_async(
            access_type type = access_type_read) const
        {
            if (type == access_type_none || type == access_type_write) {
                return completion_future{
                    std::async(std::launch::deferred, [](){}).share()};
            }

            return completion_future{
                std::async([this]() { synchronize(); }).share()};
        }

        /**
         * Calling this member function synchronizes any modifications made to
         * the data underlying "this" array_view to the specified
         * accelerator_view "av". For example, for an array_view on system
         * memory, if the data underlying the view is modified on the CPU, and
         * synchronize_to is called on "this" array_view, then the array_view
         * contents are cached on the specified accelerator_view location.
         *
         * For writable array_view objects, callers of this functional can
         * optionally specify the type of access desired on the specified target
         * accelerator_view "av", through the "type" parameter. For example
         * specifying a "access_type_read" (which is also the default value of
         * the parameter) indicates that the data has been synchronized to "av"
         * only for reading. On the other hand, specifying an access_type of
         * "access_type_read_write" synchronizes the data to "av" both for
         * reading and writing; i.e. any modifications to the data on "av" are
         * legal after synchronizing the array_view with write access and before
         * subsequently accessing the array_view on a location other than "av".
         *
         * It is advisable to be precise about the access_type specified in the
         * synchronize call; i.e. if only write access it required, specifying
         * access_type_write may yield better performance that calling
         * synchronize with "access_type_read_write" since the later may require
         * any modifications made to the data on remote locations to be
         * synchronized to "av", which is unnecessary if the contents are
         * intended to be immediately overwritten without reading.
         *
         * @param[in] av The target accelerator_view that "this" array_view is
         *               synchronized for access on.
         * @param[in] type An argument of type "access_type" which specifies the
         *                 type of access on the data source that the array_view
         *                 is synchronized for.
         */
        void synchronize_to(
            const accelerator_view& av,
            access_type type = access_type_read) const
        {   // TODO: assess optimisation opportunities.
            if (owner_ && av.get_accelerator() == *owner_) return;

            synchronize(type);
        }

        /**
         * An asynchronous version of synchronize_to, which returns a completion
         * future object. When the future is ready, the synchronization
         * operation is complete.
         *
         * @param[in] av The target accelerator_view that "this" array_view is
         *               synchronized for access on.
         * @param[in] type An argument of type "access_type" which specifies the
         *                 type of access on the data source that the array_view
         *                 is synchronized for.
         * @return An object of type completion_future that can be used to
         *         determine the status of the asynchronous operation or can be
         *         used to chain other operations to be executed after the
         *         completion of the asynchronous operation.
         */
        completion_future synchronize_to_async(
            const accelerator_view& av,
            access_type type = access_type_read) const
        {
            if (type == access_type_none || type == access_type_write) {
                return  completion_future{
                    std::async(std::launch::deferred, [](){}).share()};
            }
            if (owner_ && av.get_accelerator() == *owner_) return {};

            return synchronize_async(type);
        }

        /**
         * Indicates to the runtime that it may discard the current logical
         * contents of this array_view. This is an optimization hint to the
         * runtime used to avoid copying the current contents of the view to a
         * target accelerator_view, and its use is recommended if the existing
         * content is not needed.
         */
        void discard_data() const
        {
            decltype(writers_[writers_for_this_].second.second) tmp;

            {
                std::lock_guard<std::mutex> lck{
                    writers_[writers_for_this_].second.first};

                std::swap(writers_[writers_for_this_].second.second, tmp);
            }
        }

        /** @{ */
        /**
         * Returns a reference to the element of this array_view that is at the
         * location in N-dimensional space specified by "idx".
         *
         * @param[in] idx An object of type index<N> that specifies the location
         *                of the element.
         */
        T& operator[](const index<N>& idx) const [[cpu, hc]]
        {
            return updated_data_()[detail::amp_helper<
                N, index<N>, hc::extent<N>>::flatten(idx, extent_)];
        }

        template<int m = N, typename std::enable_if<(m == 1)>::type* = nullptr>
        T& operator[](int i0) const [[cpu]][[hc]]
        {
            return operator[](index<1>{i0});
        }


        T& operator()(const index<N>& idx) const [[cpu, hc]]
        {
            return operator[](idx);
        }

        /** @} */

        /**
         * Returns a reference to the element of this array_view that is at the
         * location in N-dimensional space specified by "idx".
         *
         * Unlike the other indexing operators for accessing the array_view on
         * the CPU, this method does not implicitly synchronize this
         * array_view's contents to the CPU. After accessing the array_view on a
         * remote location or performing a copy operation involving this
         * array_view, users are responsible to explicitly synchronize the
         * array_view to the CPU before calling this method. Failure to do so
         * results in undefined behavior.
         */
        T& get_ref(const index<N>& idx) const [[cpu, hc]]
        {
            return base_ptr_[detail::amp_helper<N, index<N>, hc::extent<N>>::
                flatten(idx, extent_)];
        }

        /** @{ */
        /**
         * Equivalent to
         * "array_view<T,N>::operator()(index<N>(i0 [, i1 [, i2 ]]))".
         *
         * @param[in] i0,i1,i2 The component values that will form the index 
         *                     into this array.
         */
        T& operator()(int i0) const [[cpu, hc]]
        {
            static_assert(
                N == 1,
                "T& array_view::operator()(int) is only permissible on "
                    "array_view<T, 1>");

            return operator[](index<1>{i0});
        }
        T& operator()(int i0, int i1) const [[cpu, hc]]
        {
            static_assert(
                N == 2,
                "T& array_view::operator()(int, int) is only permissible on "
                    "array_view<T, 2>");

            return operator[](index<2>{i0, i1});
        }
        T& operator()(int i0, int i1, int i2) const [[cpu, hc]]
        {
            static_assert(
                N == 3,
                "T& array_view::operator()(int, int, int) is only permissible "
                    "on array_view<T, 3>");

            return operator[](index<3>{i0, i1, i2});
        }

        /** @} */

        /** @{ */
        /**
         * This overload is defined for array_view<T,N> where @f$N \ge 2@f$.
         *
         * This mode of indexing is equivalent to projecting on the
         * most-significant dimension. It allows C-style indexing. For example:
         *
         * @code{.cpp}
         * array<float,4> myArray(myExtents, ...);
         *
         * myArray[index<4>(5,4,3,2)] = 7;
         * assert(myArray[5][4][3][2] == 7);
         * @endcode
         *
         * @param[in] i0 An integer that is the index into the most-significant
         *               dimension of this array.
         * @return Returns an array_view whose dimension is one lower than that
         *         of this array_view.
         */
        template<int m = N, typename std::enable_if<(m > 1)>::type* = nullptr>
        array_view<T, N - 1> operator[](int i0) const [[cpu, hc]]
        {
            hc::extent<N - 1> ext;
            for (auto i = 1; i != N; ++i) ext[i - 1] = extent_[i];

            array_view<T, N - 1> tmp{ext, static_cast<T*>(base_ptr_)}; // TODO: this is incorrect.
            tmp.base_ptr_ += i0 * ext.size();

            return tmp;
        }

        template<int m = N, typename std::enable_if<(m > 1)>::type* = nullptr>
        array_view<T, N - 1> operator()(int i0) const [[cpu, hc]]
        {
            return operator[](i0);
        }
        /** @} */

        /**
         * Returns a subsection of the source array view at the origin specified
         * by "idx" and with the extent specified by "ext".
         *
         * Example:
         *
         * @code{.cpp}
         * array<float,2> a(extent<2>(200,100));
         * array_view<float,2> v1(a); // v1.extent = <200,100>
         * array_view<float,2> v2 =
         *     v1.section(index<2>(15,25), extent<2>(40,50));
         * assert(v2(0,0) == v1(15,25));
         * @endcode
         *
         * @param[in] idx Provides the offset/origin of the resulting section.
         * @param[in] ext Provides the extent of the resulting section.
         * @return Returns a subsection of the source array at specified origin,
         *         and with the specified extent.
         */
        array_view<T, N> section(
            const index<N>& origin, const hc::extent<N>& ext) const [[cpu]]
        {
            if (extent_.size() < (ext + origin).size()) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            const auto dx = detail::amp_helper<N, index<N>, hc::extent<N>>::
                flatten(origin, extent_);

            array_view<T, N> tmp{*this};
            tmp.extent_ = ext;
            tmp.base_ptr_ += dx;
            tmp.source_ = static_cast<T*>(tmp.source_) + dx;

            return tmp;
        }
        array_view<T, N> section(
            const index<N>& origin, const hc::extent<N>& ext) const [[hc]]
        {
            const auto dx = detail::amp_helper<N, index<N>, hc::extent<N>>::
                flatten(origin, extent_);

            array_view<T, N> tmp{*this};
            tmp.extent_ = ext;
            tmp.base_ptr_ += dx;
            tmp.source_ = static_cast<T*>(tmp.source_) + dx;

            return tmp;
        }

        /**
         * Equivalent to "section(idx, this->extent – idx)".
         */
        array_view<T, N> section(const index<N>& idx) const [[cpu, hc]]
        {
            hc::extent<N> ext{extent_};
            detail::amp_helper<N, index<N>, hc::extent<N>>::minus(idx, ext);

            return section(idx, ext);
        }

        /**
         * Equivalent to "section(index<N>(), ext)".
         */
        array_view<T, N> section(const hc::extent<N>& ext) const [[cpu, hc]]
        {
            return section(index<N>{}, ext);
        }

        /** @{ */
        /**
         * Equivalent to
         * "section(index<N>(i0 [, i1 [, i2 ]]), extent<N>(e0 [, e1 [, e2 ]]))".
         *
         * @param[in] i0,i1,i2 The component values that will form the origin of
         *                     the section
         * @param[in] e0,e1,e2 The component values that will form the extent of
         *                     the section
         */
        array_view<T, 1> section(int i0, int e0) const [[cpu, hc]]
        {
            static_assert(N == 1, "Rank must be 1.");

            return section(index<1>{i0}, hc::extent<1>{e0});
        }

        array_view<T, 2> section(
            int i0, int i1, int e0, int e1) const [[cpu, hc]]
        {
            static_assert(N == 2, "Rank must be 2.");

            return section(index<2>{i0, i1}, hc::extent<2>{e0, e1});
        }

        array_view<T, 3> section(
            int i0, int i1, int i2, int e0, int e1, int e2) const [[cpu, hc]]
        {
            static_assert(N == 3, "Rank must be 3.");

            return section(index<3>{i0, i1, i2}, hc::extent<3>{e0, e1, e2});
        }

        /** @} */

        /**
         * This member function is similar to "array<T,N>::reinterpret_as",
         * although it only supports array_views of rank 1 (only those guarantee
         * that all elements are laid out contiguously).
         *
         * The size of the reinterpreted ElementType must evenly divide into the
         * total size of this array_view.
         *
         * @return Returns an array_view from this array_view<T,1> with the
         *         element type reinterpreted from T to ElementType.
         */
        template<typename U>
        array_view<U, 1> reinterpret_as() const [[cpu]]
        {
            static_assert(
                N == 1,
                "reinterpret_as is only permissible on array views of rank 1.");

            hc::extent<1> tmp{extent_.size() / sizeof(U)};

            if (extent_.size() * sizeof(T) != tmp.size() * sizeof(U)) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            if (source_) return array_view<U, 1>{tmp, source_};
            return array_view<U, 1>{tmp};
        }
        template<typename U>
        array_view<U, 1> reinterpret_as() const [[hc]]
        {
            static_assert(
                N == 1,
                "reinterpret_as is only permissible on array views of rank 1.");

            hc::extent<1> tmp{extent_.size() / sizeof(U)};

            return array_view<U, 1>{tmp, base_ptr_};
        }

        /**
         * This member function is similar to "array<T,N>::view_as", although it
         * only supports array_views of rank 1 (only those guarantee that all
         * elements are laid out contiguously).
         *
         * @return Returns an array_view from this array_view<T,1> with the rank
         * changed to K from 1.
         */
        template<int m>
        array_view<T, m> view_as(const hc::extent<m>& view_extent) const [[cpu]]
        {
            static_assert(
                N == 1, "view_as is only permissible on array views of rank 1");

            if (extent_.size() < view_extent.size()) {
                throw runtime_exception{"errorMsg_throw", 0};
            }

            return array_view<T, m>{view_extent, source_};
        }
        template<int m>
        array_view<T, m> view_as(const hc::extent<m>& view_extent) const [[hc]]
        {
            static_assert(
                N == 1, "view_as is only permissible on array views of rank 1");

            return array_view<T, m>{view_extent, source_};
        }

        ~array_view() [[cpu]][[hc]]
        {
            #if __HCC_ACCELERATOR__ != 1
                if (!data_) return;

                {
                    std::lock_guard<std::mutex> lck{mutex_};

                    if (data_.use_count() == 2) cache_.erase(source_);
                }

                if (writers_for_this_ == max_array_view_cnt_) return;
                if (--writers_[writers_for_this_].first != 0) return;

                try {
                    synchronize(access_type_read_write);
                }
                catch (const std::exception& ex) {
                    std::cerr << ex.what() << std::endl;
                }
            #endif
        }
    };

    // ------------------------------------------------------------------------
    // copy
    // ------------------------------------------------------------------------

    /**
     * The contents of "src" are copied into "dest". The source and destination
     * may reside on different accelerators. If the extents of "src" and "dest"
     * don't match, a runtime exception is thrown.
     *
     * @param[in] src An object of type array<T,N> to be copied from.
     * @param[out] dest An object of type array<T,N> to be copied to.
     */
    template<typename T, int N>
    inline
    void copy(const array<T, N>& src, array<T, N>& dest)
    {
        if (src.get_extent() != dest.get_extent()) {
            throw std::logic_error{
                "Tried to copy arrays of mismatched extents."};
        }

        src.wait_for_all_pending_writers_();

        auto s = hsa_memory_copy(
            dest.data(), src.data(), src.get_extent().size() * sizeof(T));

        if (s == HSA_STATUS_SUCCESS) return;

        throw std::runtime_error{"Array copy failed."};
    }

    /** @{ */
    /**
     * The contents of "src" are copied into "dest". If the extents of "src" and
     * "dest" don't match, a runtime exception is thrown.
     *
     * @param[in] src An object of type array<T,N> to be copied from.
     * @param[out] dest An object of type array_view<T,N> to be copied to.
     */
    template<typename T, int N>
    inline
    void copy(const array<T, N>& src, const array_view<T, N>& dest)
    {   // TODO: assess optimisation opportunities.
        if (src.get_extent() != dest.get_extent()) {
            throw std::logic_error{
                "Tried to copy array to an array_view with a mismatched "
                "extent."};
        }

        src.wait_for_all_pending_writers_();

        auto s = hsa_memory_copy(
            dest.data(), src.base_ptr_, src.get_extent().size() * sizeof(T));

        if (s == HSA_STATUS_SUCCESS) return;

        throw std::runtime_error{"array_view to array copy failed."};
    }
    /** @} */

    /** @{ */
    /**
     * The contents of "src" are copied into "dest". If the extents of "src" and
     * "dest" don't match, a runtime exception is thrown.
     *
     * @param[in] src An object of type array_view<T,N> (or
     *                array_view<const T, N>) to be copied from.
     * @param[out] dest An object of type array<T,N> to be copied to.
     */
    template<typename T, int N>
    inline
    void copy(const array_view<const T, N>& src, array<T, N>& dest)
    {
        if (src.get_extent() != dest.get_extent()) {
            throw std::logic_error{
                "Tried to copy array_view to an array with a mismatched "
                "extent."};
        }

        auto s = hsa_memory_copy(
            dest.data(), src.data(), src.get_extent().size() * sizeof(T));

        if (s == HSA_STATUS_SUCCESS) return;

        throw std::runtime_error{"array_view to array copy failed."};
    }

    template<typename T, int N>
    inline
    void copy(const array_view<T, N>& src, array<T, N>& dest)
    {
        copy(array_view<const T, N>{src}, dest);
    }
    /** @} */

    /** @{ */
    /**
     * The contents of "src" are copied into "dest". If the extents of "src" and
     * "dest" don't match, a runtime exception is thrown.
     *
     * @param[in] src An object of type array_view<T,N> (or
     *                array_view<const T, N>) to be copied from.
     * @param[out] dest An object of type array_view<T,N> to be copied to.
     */
    template<typename T, int N>
    inline
    void copy(const array_view<const T, N>& src, const array_view<T, N>& dest)
    {
        if (src.get_extent() != dest.get_extent()) {
            throw std::logic_error{
                "Tried to copy array_views with mismatched extents."};
        }

        auto s = hsa_memory_copy(
            dest.base_ptr_, src.data(), src.get_extent().size() * sizeof(T));

        if (s == HSA_STATUS_SUCCESS) return;

        throw std::runtime_error{"array_view to array_view copy failed."};
    }

    template <typename T, int N>
    inline
    void copy(const array_view<T, N>& src, const array_view<T, N>& dest)
    {
        copy(array_view<const T, N>{src}, dest);
    }
    /** @} */

    /** @{ */
    /**
     * The contents of a source container from the iterator range
     * [srcBegin,srcEnd) are copied into "dest". If the number of elements in
     * the iterator range is not equal to "dest.extent.size()", an exception is
     * thrown.
     *
     * In the overloads which don't take an end-iterator it is assumed that the
     * source iterator is able to provide at least dest.extent.size() elements,
     * but no checking is performed (nor possible).
     *
     * @param[in] srcBegin An iterator to the first element of a source
     *            container.
     * @param[in] srcEnd An interator to the end of a source container.
     * @param[out] dest An object of type array<T,N> to be copied to.
     */
    template<typename InputIter, typename T, int N>
    inline
    void copy(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::value_type, T>{},
            "Only same type copies supported.");

        if (srcBegin == srcEnd) return;

        if (std::distance(srcBegin, srcEnd) != dest.get_extent().size()) {
            throw std::logic_error{"Mismatched copy sizes."};
        }

        copy(srcBegin, dest);
    }

    template<typename InputIter, typename T, int N>
    inline
    void copy(InputIter srcBegin, array<T, N>& dest)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<typename std::iterator_traits<InputIter>::value_type, T>{},
            "Only same type copies supported.");

        auto s = hsa_memory_copy( // TODO: add to_address() and use it instead of &*.
            dest.data(), &*srcBegin, dest.get_extent().size() * sizeof(T));

        if (s == HSA_STATUS_SUCCESS) return;

        throw std::runtime_error{"Failed iterator range to array copy."};
    }

    /** @} */

    /** @{ */
    /**
     * The contents of a source container from the iterator range
     * [srcBegin,srcEnd) are copied into "dest". If the number of elements in
     * the iterator range is not equal to "dest.extent.size()", an exception is
     * thrown.
     *
     * In the overloads which don't take an end-iterator it is assumed that the
     * source iterator is able to provide at least dest.extent.size() elements,
     * but no checking is performed (nor possible).
     *
     * @param[in] srcBegin An iterator to the first element of a source
     *            container.
     * @param[in] srcEnd An interator to the end of a source container.
     * @param[out] dest An object of type array_view<T,N> to be copied to.
     */
    template<typename InputIter, typename T, int N>
    inline
    void copy(
        InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::value_type, T>{},
            "Only same type copies supported.");

        if (srcBegin == srcEnd) return;

        if (std::distance(srcBegin, srcEnd) != dest.get_extent().size()) {
            throw std::logic_error{"Mismatched copy sizes."};
        }

        auto s = hsa_memory_copy( // TODO: add to_address() and use it instead of &*.
            dest.base_ptr_, &*srcBegin, dest.get_extent().size() * sizeof(T));

        if (s == HSA_STATUS_SUCCESS) return;

        throw std::runtime_error{"Failed iterator range to array_view copy."};
    }

    template<typename InputIter, typename T, int N>
    inline
    void copy(InputIter srcBegin, const array_view<T, N>& dest)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::value_type, T>{},
            "Only same type copies supported.");

        copy(srcBegin, srcBegin + dest.get_extent().size(), dest);
    }

    /** @} */

    /**
     * The contents of a source array are copied into "dest" starting with
     * iterator destBegin. If the number of elements in the range starting
     * destBegin in the destination container is smaller than
     * "src.extent.size()", the behavior is undefined.
     *
     * @param[in] src An object of type array<T,N> to be copied from.
     * @param[out] destBegin An output iterator addressing the position of the
     *                       first element in the destination container.
     */
    template<typename OutputIter, typename T, int N>
    inline
    void copy(const array<T, N> &src, OutputIter destBegin)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<OutputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<OutputIter>::value_type, T>{},
            "Only same type copies supported.");

        src.wait_for_all_pending_writers_();

        // TODO: must add to_address() and use instead of &*.
        auto s = hsa_memory_copy(
            &*destBegin, src.data(), src.get_extent().size() * sizeof(T));

        if (s == HSA_STATUS_SUCCESS) return;

        throw std::runtime_error{"array to iterator range copy failed."};
    }

    /**
     * The contents of a source array are copied into "dest" starting with
     * iterator destBegin. If the number of elements in the range starting
     * destBegin in the destination container is smaller than
     * "src.extent.size()", the behavior is undefined.
     *
     * @param[in] src An object of type array_view<T,N> to be copied from.
     * @param[out] destBegin An output iterator addressing the position of the
     *                       first element in the destination container.
     */
    template<typename OutputIter, typename T, int N>
    inline
    void copy(const array_view<T, N> &src, OutputIter destBegin)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<OutputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<OutputIter>::value_type, T>{},
            "Only same type copies supported.");

        src.synchronize(); // TODO: conservative, temporary.

        // TODO: must add to_address() and use instead of &*.
        auto s = hsa_memory_copy(
            &*destBegin, src.data(), src.get_extent().size() * sizeof(T));

        if (s == HSA_STATUS_SUCCESS) return;

        throw std::runtime_error{"array_view to iterator range copy failed."};
    }

    // ------------------------------------------------------------------------
    // copy_async
    // ------------------------------------------------------------------------

    /**
     * The contents of "src" are copied into "dest". The source and destination
     * may reside on different accelerators. If the extents of "src" and "dest"
     * don't match, a runtime exception is thrown.
     *
     * @param[in] src An object of type array<T,N> to be copied from.
     * @param[out] dest An object of type array<T,N> to be copied to.
     */
    template<typename T, int N>
    inline
    completion_future copy_async(const array<T, N>& src, array<T, N>& dest)
    {
        return
            completion_future{std::async([&]() { copy(src, dest); }).share()};
    }

    /**
     * The contents of "src" are copied into "dest". If the extents of "src" and
     * "dest" don't match, a runtime exception is thrown.
     *
     * @param[in] src An object of type array<T,N> to be copied from.
     * @param[out] dest An object of type array_view<T,N> to be copied to.
     */
    template<typename T, int N>
    inline
    completion_future copy_async(
        const array<T, N>& src, const array_view<T, N>& dest)
    {   // TODO: should this count as a writer to the array_view?
        return completion_future{
            std::async([&, dest]() { copy(src, dest); }).share()};
    }

    /** @{ */
    /**
     * The contents of "src" are copied into "dest". If the extents of "src" and
     * "dest" don't match, a runtime exception is thrown.
     *
     * @param[in] src An object of type array_view<T,N> (or
     *                array_view<const T, N>) to be copied from.
     * @param[out] dest An object of type array<T,N> to be copied to.
     */
    template<typename T, int N>
    inline
    completion_future copy_async(
        const array_view<const T, N>& src, array<T, N>& dest)
    {
        return completion_future{
            std::async([&, src]() { copy(src, dest); }).share()};
    }

    template<typename T, int N>
    inline
    completion_future copy_async(const array_view<T, N>& src, array<T, N>& dest)
    {
        return completion_future{
            std::async([&, src]() { copy(src, dest); }).share()};
    }

    /** @} */

    /** @{ */
    /**
     * The contents of "src" are copied into "dest". If the extents of "src" and
     * "dest" don't match, a runtime exception is thrown.
     *
     * @param[in] src An object of type array_view<T,N> (or
     *                array_view<const T, N>) to be copied from.
     * @param[out] dest An object of type array_view<T,N> to be copied to.
     */
    template<typename T, int N>
    inline
    completion_future copy_async(
        const array_view<const T, N>& src, const array_view<T, N>& dest)
    {   // TODO: should this count as a writer to the array_view?
        return
            completion_future{std::async([=]() { copy(src, dest); }).share()};
    }

    template<typename T, int N>
    inline
    completion_future copy_async(
        const array_view<T, N>& src, const array_view<T, N>& dest)
    {   // TODO: should this count as a writer to the array_view?
        return
            completion_future{std::async([=]() { copy(src, dest); }).share()};
    }

    /** @} */

    /** @{ */
    /**
     * The contents of a source container from the iterator range
     * [srcBegin,srcEnd) are copied into "dest". If the number of elements in
     * the iterator range is not equal to "dest.extent.size()", an exception is
     * thrown.
     *
     * In the overloads which don't take an end-iterator it is assumed that the
     * source iterator is able to provide at least dest.extent.size() elements,
     * but no checking is performed (nor possible).
     *
     * @param[in] srcBegin An iterator to the first element of a source
     * container.
     * @param[in] srcEnd An interator to the end of a source container.
     * @param[out] dest An object of type array<T,N> to be copied to.
     */
    template<typename InputIter, typename T, int N>
    inline
    completion_future copy_async(
        InputIter srcBegin, InputIter srcEnd, array<T, N>& dest)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::value_type, T>{},
            "Only same type copies supported.");

        if (std::distance(srcBegin, srcEnd) != dest.get_extent().size()) {
            throw std::logic_error{"Mismatched copy sizes."};
        }

        return completion_future{
            std::async([=, &dest]() { copy(srcBegin, srcEnd, dest); }).share()};
    }

    template<typename InputIter, typename T, int N>
    inline
    completion_future copy_async(InputIter srcBegin, array<T, N>& dest)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::value_type, T>{},
            "Only same type copies supported.");

        return copy_async(srcBegin, srcBegin + dest.get_extent().size(), dest);
    }

    /** @} */

    /** @{ */
    /**
     * The contents of a source container from the iterator range
     * [srcBegin,srcEnd) are copied into "dest". If the number of elements in
     * the iterator range is not equal to "dest.extent.size()", an exception is
     * thrown.
     *
     * In the overloads which don't take an end-iterator it is assumed that the
     * source iterator is able to provide at least dest.extent.size() elements,
     * but no checking is performed (nor possible).
     *
     * @param[in] srcBegin An iterator to the first element of a source
     *            container.
     * @param[in] srcEnd An interator to the end of a source container.
     * @param[out] dest An object of type array_view<T,N> to be copied to.
     */
    template<typename InputIter, typename T, int N>
    inline
    completion_future copy_async(
        InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::value_type, T>{},
            "Only same type copies supported.");

        if (std::distance(srcBegin, srcEnd) != dest.get_extent().size()) {
            throw std::logic_error{"Mismatched copy sizes."};
        }

        return completion_future{
            std::async([=]() { copy(srcBegin, srcEnd, dest); }).share()};
    }

    template<typename InputIter, typename T, int N>
    inline
    completion_future copy_async(
        InputIter srcBegin, const array_view<T, N>& dest)
    {
    static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<InputIter>::value_type, T>{},
            "Only same type copies supported.");

        return copy_async(srcBegin, srcBegin + dest.get_extent().size(), dest);
    }

    /** @} */

    /**
     * The contents of a source array are copied into "dest" starting with
     * iterator destBegin. If the number of elements in the range starting
     * destBegin in the destination container is smaller than
     * "src.extent.size()", the behavior is undefined.
     *
     * @param[in] src An object of type array<T,N> to be copied from.
     * @param[out] destBegin An output iterator addressing the position of the
     *                       first element in the destination container.
     */
    template<typename OutputIter, typename T, int N>
    inline
    completion_future copy_async(const array<T, N>& src, OutputIter destBegin)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<OutputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<OutputIter>::value_type, T>{},
            "Only same type copies supported.");

        return completion_future{
            std::async([&, destBegin]() { copy(src, destBegin); }).share()};
    }

    /**
     * The contents of a source array are copied into "dest" starting with
     * iterator destBegin. If the number of elements in the range starting
     * destBegin in the destination container is smaller than
     * "src.extent.size()", the behavior is undefined.
     *
     * @param[in] src An object of type array_view<T,N> to be copied from.
     * @param[out] destBegin An output iterator addressing the position of the
     *                       first element in the destination container.
     */
    template<typename OutputIter, typename T, int N>
    inline
    completion_future copy_async(
        const array_view<T, N>& src, OutputIter destBegin)
    {
        static_assert(
            std::is_same<
                typename std::iterator_traits<OutputIter>::iterator_category,
                std::random_access_iterator_tag>{},
            "Only contiguous random access iterators supported.");
        static_assert(
            std::is_same<
                typename std::iterator_traits<OutputIter>::value_type, T>{},
            "Only same type copies supported.");

        return completion_future{
            std::async([=]() { copy(src, destBegin); }).share()};
    }

    // ------------------------------------------------------------------------
    // parallel_for_each
    // ------------------------------------------------------------------------

    template<typename Kernel, int n>
    completion_future parallel_for_each(
        const accelerator_view&, const hc::extent<n>&, const Kernel&);

    template<typename Kernel, int n>
    completion_future parallel_for_each(
        const accelerator_view&, const tiled_extent<n>&, const Kernel&);

    template<typename Kernel, int n>
    inline
    completion_future parallel_for_each(
        const hc::extent<n>& compute_domain, const Kernel& f)
    {
        return parallel_for_each(
            accelerator::get_auto_selection_view(), compute_domain, f);
    }

    template<int n, typename Kernel>
    inline
    completion_future parallel_for_each(
        const tiled_extent<n>& compute_domain, const Kernel& f) {
        return parallel_for_each(
            accelerator::get_auto_selection_view(), compute_domain, f);
    }

    template<int n>
    inline
    void validate_compute_domain(const hc::extent<n>& compute_domain)
    {
        std::size_t sz{1};
        for (auto i = 0; i != n; ++i) {
            sz *= compute_domain[i];

            if (sz < 1) throw invalid_compute_domain{"Extent is not positive."};
            if (sz > UINT_MAX) {
                throw invalid_compute_domain{"Extent is too large."};
            }
        }
    }

    template<typename Kernel>
    inline
    std::forward_list<std::shared_future<void>> predecessors_for(const Kernel& f)
    {   // TODO: cleanup & optimise; the iteration can be collapsed.
        using AR = array_base;
        using AV = array_view_base;

        AV::captured_.clear();
        auto trigger_registration = f;

        std::forward_list<std::shared_future<void>> r;
        for (auto&& widx : AR::captured_) {
            std::lock_guard<std::mutex> lck{AR::writers_[widx].second.first};

            r.splice_after(
                r.before_begin(),
                std::move(AR::writers_[widx].second.second),
                AR::writers_[widx].second.second.before_begin());
        }
        for (auto&& widx : AV::captured_) {
            std::lock_guard<std::mutex> lck{AV::writers_[widx].second.first};

            r.splice_after(
                r.before_begin(),
                std::move(AV::writers_[widx].second.second),
                AV::writers_[widx].second.second.before_begin());
        }

        return r;
    }

    inline
    void register_writer(const completion_future& pending_task)
    {   // TODO: cleanup & optimise; the iteration can be collapsed.
        using AR = array_base;
        using AV = array_view_base;

        for (auto&& widx : AR::captured_) {
            std::lock_guard<std::mutex> lck{AR::writers_[widx].second.first};

            AR::writers_[widx].second.second.emplace_front(pending_task);
        }
        for (auto&& widx : AV::captured_) {
            std::lock_guard<std::mutex> lck{AV::writers_[widx].second.first};

            AV::writers_[widx].second.second.emplace_front(pending_task);
        }

        AR::captured_.clear();
    }

    //ND parallel_for_each, nontiled
    template<typename Kernel, int n>
    inline
    __attribute__((annotate("__HC_PFE__")))
    completion_future parallel_for_each(
        const accelerator_view& av,
        const hc::extent<n>& compute_domain,
        const Kernel& f)
    {   // TODO: unify with tiled, everything is essentially tiled
        if (compute_domain.size() == 0) {
            return completion_future{std::async([](){}).share()};
        }

        if (av.get_accelerator().get_device_path() == L"cpu") {
        throw hc::runtime_exception{
            detail::__errorMsg_UnsupportedAccelerator, detail::E_FAIL};
        }

        validate_compute_domain(compute_domain);

        for (auto&& x : predecessors_for(f)) if (x.valid()) x.wait();

        completion_future tmp{
            detail::launch_kernel_async(av, compute_domain, f)};
        av.add_pending_task_(tmp);

        register_writer(tmp);

        return tmp;
    }

    template<int n>
    inline
    void validate_tiled_compute_domain(const tiled_extent<n>& compute_domain)
    {
        validate_compute_domain(compute_domain);

        size_t sz{1};
        for (auto i = 0u; i != n; ++i) {
            if (compute_domain.tile_dim[i] < 0) {
                throw invalid_compute_domain{
                    "The extent of the tile must be positive."};
            }

            constexpr int max_tile_dim{1024}; // Should be read via the HSArt.
            sz *= compute_domain.tile_dim[i];
            if (max_tile_dim < sz) {
                throw invalid_compute_domain{
                    "The extent of the tile exceeds the device limit"};
            }

            if (compute_domain[i] < compute_domain.tile_dim[i]) {
                throw invalid_compute_domain{
                    "The extent of the tile exceeds the compute grid extent"};
            }
        }
    }

    //ND parallel_for_each, tiled
    template <typename Kernel, int n>
    inline
    __attribute__((annotate("__HC_PFE__")))
    completion_future parallel_for_each(
        const accelerator_view& av,
        const tiled_extent<n>& compute_domain,
        const Kernel& f)
    {   // TODO: optimise, this spuriously does one extra copy of Kernel.
        if (compute_domain.size() == 0) {
            return completion_future{std::async([](){}).share()};
        }

        if (av.get_accelerator().get_device_path() == L"cpu") {
            throw hc::runtime_exception{
                detail::__errorMsg_UnsupportedAccelerator, detail::E_FAIL};
        }

        validate_tiled_compute_domain(compute_domain);

        for (auto&& x : predecessors_for(f)) if (x.valid()) x.wait();

        completion_future tmp{
            detail::launch_kernel_async(av, compute_domain, f)};
        av.add_pending_task_(tmp);

        register_writer(tmp);

        return tmp;
    }
} // namespace hc