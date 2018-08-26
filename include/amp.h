//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

/**
 * @file amp.h
 * C++ AMP API.
 */

#pragma once

#warning "C++AMP support is deprecated in ROCm 1.9 and will be removed in ROCm 2.0!"

#include "atomics.hpp"
#include "hc.hpp"
#include "hc_defines.h"
#include "kalmar_exception.h"
#include "kalmar_index.h"
#include "kalmar_runtime.h"
#include "kalmar_buffer.h"
#include "kalmar_serialize.h"
#include "kalmar_launch.h"

#include <climits>
#include <cstddef>
#include <type_traits>

// forward declaration
namespace Concurrency {
class completion_future;
class accelerator;
class accelerator_view;
template <int N> class extent;
template <int D0, int D1=0, int D2=0> class tiled_extent;
} // namespace Concurrency

// namespace alias
// namespace concurrency is an alias of namespace Concurrency
namespace concurrency = Concurrency;


// type alias
namespace Concurrency {

using hc::array;
using hc::array_view;

/**
 * Represents a unique position in N-dimensional space.
 */
template <int N>
using index = detail::index<N>;

using runtime_exception = detail::runtime_exception;
using invalid_compute_domain = detail::invalid_compute_domain;
using accelerator_view_removed = detail::accelerator_view_removed;
} // namespace Concurrency


/**
 * @namespace Concurrency
 * C++ AMP namespace
 */
namespace Concurrency {

using namespace hc::atomics;
using namespace detail::enums;
using namespace detail::CLAMP;

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
     */
    void wait() { pQueue->wait(); }

    /**
     * Sends the queued up commands in the accelerator_view to the device for
     * execution.
     *
     * An accelerator_view internally maintains a buffer of commands such as
     * data transfers between the host memory and device buffers, and kernel
     * invocations (parallel_for_each calls). This member function sends the
     * commands to the device for processing. Normally, these commands are sent
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
     * after the device finishes executing the buffered commands. However, the
     * commands will eventually always complete.
     *
     * If the queuing_mode is queuing_mode_immediate, this function does
     * nothing.
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
     * @return A future which can be waited on, and will block until the
     *         current batch of commands has completed.
     */
    // FIXME: dummy implementation now
    completion_future create_marker();
  
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

private:
    accelerator_view(std::shared_ptr<detail::HCCQueue> pQueue) : pQueue(pQueue) {}
    std::shared_ptr<detail::HCCQueue> pQueue;
    friend class accelerator;

    template<typename Domain, typename Kernel>
    friend
    void detail::launch_kernel(
        const std::shared_ptr<detail::HCCQueue>&,
        const Domain&,
        const Kernel&);
    template<typename Domain, typename Kernel>
    friend
    std::shared_future<detail::HCCAsyncOp> detail::launch_kernel_async(
        const std::shared_ptr<detail::HCCQueue>&,
        const Domain&,
        const Kernel&);

    template<typename, int> friend class hc::array;
    template<typename, int> friend class hc::array_view;
    template <int N, typename Kernel>
    friend
    void parallel_for_each(const Concurrency::extent<N>&, const Kernel&);
    template <int N, typename Kernel>
    friend
    void parallel_for_each(
        const accelerator_view&, const Concurrency::extent<N>&, const Kernel&);

    template<typename Kernel, int... dims>
    friend
    void parallel_for_each(const tiled_extent<dims...>&, const Kernel&);
    template<typename Kernel, int... dims>
    friend
    void parallel_for_each(
        const accelerator_view&, const tiled_extent<dims...>&, const Kernel&);
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
  
    /** @{ */
    /** 
     * These are static constant string literals that represent device paths for
     * known accelerators, or in the case of "default_accelerator", direct the
     * runtime to choose an accelerator automatically.
     *
     * default_accelerator: The string L"default" represents the default
     * accelerator, which directs the runtime to choose the fastest accelerator
     * available. The selection criteria are discussed in section 3.2.1 Default
     * Accelerator.
     *
     * cpu_accelerator: The string L"cpu" represents the host system. This
     * accelerator is used to provide a location for system-allocated memory
     * such as host arrays and staging arrays. It is not a valid target for
     * accelerated computations.
     */
    static const wchar_t default_accelerator[];   // = L"default"
    static const wchar_t cpu_accelerator[];       // = L"cpu"
  
    /** @} */
  
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
    accelerator() : accelerator(default_accelerator) {}
  
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
        : pDev(detail::getContext()->getDevice(path)) {}
  
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
        auto Devices = detail::getContext()->getDevices();
        std::vector<accelerator> ret(Devices.size());
        for (std::size_t i = 0; i < ret.size(); ++i)
            ret[i] = Devices[i];
        return std::move(ret);
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
        return detail::getContext()->set_default(path);
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
        return detail::getContext()->auto_select();
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
    accelerator_view create_view(queuing_mode qmode = queuing_mode_automatic) {
        auto pQueue = pDev->createQueue();
        pQueue->set_mode(qmode);
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
    bool set_default_cpu_access_type(access_type default_cpu_access_type) {
        pDev->set_access(default_cpu_access_type);
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

private:
    accelerator(detail::HCCDevice* pDev) : pDev(pDev) {}
    friend class accelerator_view;
    detail::HCCDevice* pDev;
};

// ------------------------------------------------------------------------
// completion_future
// ------------------------------------------------------------------------

/**
 * This class is the return type of all C++ AMP asynchronous APIs and has an
 * interface analogous to std::shared_future<void>. Similar to
 * std::shared_future, this type provides member methods such as wait and get
 * to wait for C++ AMP asynchronous operations to finish, and the type
 * additionally provides a member method then(), to specify a completion
 * callback functor to be executed upon completion of a C++ AMP asynchronous
 * operation.
 */
class completion_future {
public:

    /**
     * Default constructor. Constructs an empty uninitialized completion_future
     * object which does not refer to any asynchronous operation. Default
     * constructed completion_future objects have valid() == false
     */
    completion_future() {};

    /**
     * Copy constructor. Constructs a new completion_future object that referes
     * to the same asynchronous operation as the other completion_future object.
     *
     * @param[in] other An object of type completion_future from which to
     *                  initialize this.
     */
    completion_future(const completion_future& other)
        : __amp_future(other.__amp_future), __thread_then(other.__thread_then) {}

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
        : __amp_future(std::move(other.__amp_future)), __thread_then(other.__thread_then) {}

    /**
     * Copy assignment. Copy assigns the contents of other to this. This method
     * causes this to stop referring its current asynchronous operation and
     * start referring the same asynchronous operation as other.
     *
     * @param[in] other An object of type completion_future which is copy
     *                  assigned to this.
     */
    completion_future& operator=(const completion_future& other) {
        if (this != &other) {
           __amp_future = other.__amp_future;
           __thread_then = other.__thread_then;
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
    completion_future& operator=(completion_future&& other) {
        if (this != &other) {
            __amp_future = std::move(other.__amp_future);
            __thread_then = other.__thread_then;
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
     */
    void wait() const {
        if(this->valid())
          __amp_future.wait();
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
#if __KALMAR_ACCELERATOR__ != 1
      // could only assign once
      if (__thread_then == nullptr) {
        // spawn a new thread to wait on the future and then execute the callback functor
        __thread_then = new std::thread([&]() restrict(cpu) {
          this->wait();
          if(this->valid())
            func();
        });
      }
#endif
    }

    ~completion_future() {
      if (__thread_then != nullptr) {
        __thread_then->join();
      }
      delete __thread_then;
      __thread_then = nullptr;
    }

private:
    std::shared_future<void> __amp_future;
    std::thread* __thread_then = nullptr;

    completion_future(const std::shared_future<void> &__future)
        : __amp_future(__future) {}

    template <typename T, int N> friend
        completion_future copy_async(const array_view<const T, N>& src, const array_view<T, N>& dest);
    template <typename T, int N> friend
        completion_future copy_async(const array<T, N>& src, array<T, N>& dest);
    template <typename T, int N> friend
        completion_future copy_async(const array<T, N>& src, const array_view<T, N>& dest);
    template <typename T, int N> friend
        completion_future copy_async(const array_view<T, N>& src, const array_view<T, N>& dest);
    template <typename T, int N> friend
        completion_future copy_async(const array_view<const T, N>& src, array<T, N>& dest);

    template <typename InputIter, typename T, int N> friend
        completion_future copy_async(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest);
    template <typename InputIter, typename T, int N> friend
        completion_future copy_async(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest);
    template <typename InputIter, typename T, int N> friend
        completion_future copy_async(InputIter srcBegin, array<T, N>& dest);
    template <typename InputIter, typename T, int N> friend
        completion_future copy_async(InputIter srcBegin, const array_view<T, N>& dest);
    template <typename OutputIter, typename T, int N> friend
        completion_future copy_async(const array<T, N>& src, OutputIter destBegin);
    template <typename OutputIter, typename T, int N> friend
        completion_future copy_async(const array_view<T, N>& src, OutputIter destBegin);

    template<typename, int> friend class hc::array_view;
};

// ------------------------------------------------------------------------
// member function implementations
// ------------------------------------------------------------------------

inline accelerator accelerator_view::get_accelerator() const { return pQueue->getDev(); }

inline completion_future accelerator_view::create_marker(){ return completion_future(); }

inline unsigned int accelerator_view::get_version() const { return get_accelerator().get_version(); }


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
template <int N>
class extent {
public:

    /**
     * A static member of extent<N> that contains the rank of this extent.
     */
    static const int rank = N;

    /**
     * The element type of extent<N>.
     */
    typedef int value_type;

    /**
     * Default constructor. The value at each dimension is initialized to zero.
     * Thus, "extent<3> ix;" initializes the variable to the position (0,0,0).
     */
    extent() restrict(amp,cpu) : base_() {
      static_assert(N > 0, "Dimensionality must be positive");
    };

    /**
     * Copy constructor. Constructs a new extent<N> from the supplied argument.
     *
     * @param other An object of type extent<N> from which to initialize this
     *              new extent.
     */
    extent(const extent& other) restrict(amp,cpu)
        : base_(other.base_) {}

    extent(const hc::extent<N>& other) restrict(cpu, amp)
        : extent{reinterpret_cast<const extent&>(other)}
    {}

    /** @{ */
    /**
     * Constructs an extent<N> with the coordinate values provided by @f$e_{0..2}@f$.
     * These are specialized constructors that are only valid when the rank of
     * the extent @f$N \in \{1,2,3\}@f$. Invoking a specialized constructor
     * whose argument @f$count \ne N@f$ will result in a compilation error.
     *
     * @param[in] e0 The component values of the extent vector.
     */
    explicit extent(int e0) restrict(amp,cpu)
        : base_(e0) {}

    template <typename ..._Tp>
        explicit extent(_Tp ... __t) restrict(amp,cpu)
        : base_(__t...) {
      static_assert(sizeof...(__t) <= 3, "Can only supply at most 3 individual coordinates in the constructor");
      static_assert(sizeof...(__t) == N, "rank should be consistency");
    }

    /** @} */

    /**
     * Constructs an extent<N> with the coordinate values provided the array of
     * int component values. If the coordinate array length @f$\ne@f$ N, the
     * behavior is undefined. If the array value is NULL or not a valid
     * pointer, the behavior is undefined.
     *
     * @param[in] components An array of N int values.
     */
    explicit extent(const int components[]) restrict(amp,cpu)
        : base_(components) {}

    /**
     * Constructs an extent<N> with the coordinate values provided the array of
     * int component values. If the coordinate array length @f$\ne@f$ N, the
     * behavior is undefined. If the array value is NULL or not a valid
     * pointer, the behavior is undefined.
     *
     * @param[in] components An array of N int values.
     */
    // FIXME: this function is not defined in C++AMP specification.
    explicit extent(int components[]) restrict(amp,cpu)
        : base_(components) {}

    /**
     * Assigns the component values of "other" to this extent<N> object.
     *
     * @param[in] other An object of type extent<N> from which to copy into
     *                  this extent.
     * @return Returns *this.
     */
    extent& operator=(const extent& other) restrict(amp,cpu) {
        base_.operator=(other.base_);
        return *this;
    }
    
    /** @{ */
    /**
     * Returns the extent component value at position c.
     *
     * @param[in] c The dimension axis whose coordinate is to be accessed.
     * @return A the component value at position c.
     */
    int operator[] (unsigned int c) const restrict(amp,cpu) {
        return base_[c];
    }
    int& operator[] (unsigned int c) restrict(amp,cpu) {
        return base_[c];
    }

    /** @} */

    /**
     * Tests whether the index "idx" is properly contained within this extent
     * (with an assumed origin of zero).
     *
     * @param[in] idx An object of type index<N>
     * @return Returns true if the "idx" is contained within the space defined
     *         by this extent (with an assumed origin of zero).
     */
    bool contains(const index<N>& idx) const restrict(amp,cpu) {
        return detail::amp_helper<N, index<N>, extent<N>>::contains(idx, *this);
    }

    /**
     * This member function returns the total linear size of this extent<N> (in
     * units of elements), which is computed as:
     * extent[0] * extent[1] ... * extent[N-1]
     */
    unsigned int size() const restrict(amp,cpu) {
        return detail::index_helper<N, extent<N>>::count_size(*this);
    }


    /** @{ */
    /**
     * Produces a tiled_extent object with the tile extents given by D0, D1,
     * and D2.
     *
     * tile<D0,D1,D2>() is only supported on extent<3>. It will produce a
     * compile-time error if used on an extent where N @f$\ne@f$ 3.
     * tile<D0,D1>() is only supported on extent<2>. It will produce a
     * compile-time error if used on an extent where N @f$\ne@f$ 2.
     * tile<D0>() is only supported on extent<1>. It will produce a
     * compile-time error if used on an extent where N @f$\ne@f$ 1.
     */
    template <int D0>
        tiled_extent<D0> tile() const restrict(amp,cpu) {
            static_assert(N == 1, "One-dimensional tile() method only available on extent<1>");
            static_assert(D0 >0, "All tile dimensions must be positive");
            return tiled_extent<D0>(*this);
        }
    template <int D0, int D1>
        tiled_extent<D0, D1> tile() const restrict(amp,cpu) {
            static_assert(N == 2, "Two-dimensional tile() method only available on extent<2>");
            static_assert(D0 >0 && D1 > 0, "All tile dimensions must be positive");
            return tiled_extent<D0, D1>(*this);
        }
    template <int D0, int D1, int D2>
        tiled_extent<D0, D1, D2> tile() const restrict(amp,cpu) {
            static_assert(N == 3, "Three-dimensional tile() method only available on extent<3>");
            static_assert(D0 >0 && D1 > 0 && D2 > 0, "All tile dimensions must be positive");
            return tiled_extent<D0, D1, D2>(*this);
        }

    /** @} */

    /** @{ */
    /**
     * Compares two objects of extent<N>.
     *
     * The expression
     * leftExt @f$\oplus@f$ rightExt
     * is true if leftExt[i] @f$\oplus@f$ rightExt[i] for every i from 0 to N-1.
     *
     * @param[in] other The right-hand extent<N> to be compared.
     */
    // FIXME: the signature is not entirely the same as defined in:
    //        C++AMP spec v1.2 #1255
    bool operator==(const extent& other) const restrict(amp,cpu) {
        return detail::index_helper<N, extent<N> >::equal(*this, other);
    }
    bool operator!=(const extent& other) const restrict(amp,cpu) {
        return !(*this == other);
    }

    /** @} */

    /** @{ */
    /**
     * Adds (or subtracts) an object of type extent<N> from this extent to form
     * a new extent. The result extent<N> is such that for a given operator @f$\oplus@f$,
     * result[i] = this[i] @f$\oplus@f$ ext[i]
     *
     * @param[in] ext The right-hand extent<N> to be added or subtracted.
     */
    extent& operator+=(const extent& __r) restrict(amp,cpu) {
        base_.operator+=(__r.base_);
        return *this;
    }
    extent& operator-=(const extent& __r) restrict(amp,cpu) {
        base_.operator-=(__r.base_);
        return *this;
    }

    // FIXME: this function is not defined in C++AMP specification.
    extent& operator*=(const extent& __r) restrict(amp,cpu) {
        base_.operator*=(__r.base_);
        return *this;
    }
    // FIXME: this function is not defined in C++AMP specification.
    extent& operator/=(const extent& __r) restrict(amp,cpu) {
        base_.operator/=(__r.base_);
        return *this;
    }
    // FIXME: this function is not defined in C++AMP specification.
    extent& operator%=(const extent& __r) restrict(amp,cpu) {
        base_.operator%=(__r.base_);
        return *this;
    }
    /** @} */

    /** @{ */
    /**
     * Adds (or subtracts) an object of type index<N> from this extent to form
     * a new extent. The result extent<N> is such that for a given operator @f$\oplus@f$,
     * result[i] = this[i] @f$\oplus@f$ idx[i]
     *
     * @param[in] idx The right-hand index<N> to be added or subtracted.
     */
    extent operator+(const index<N>& idx) restrict(amp,cpu) {
        extent __r = *this;
        __r += idx;
        return __r;
    }
    extent operator-(const index<N>& idx) restrict(amp,cpu) {
        extent __r = *this;
        __r -= idx;
        return __r;
    }
    extent& operator+=(const index<N>& idx) restrict(amp,cpu) {
        base_.operator+=(idx.base_);
        return *this;
    }
    extent& operator-=(const index<N>& idx) restrict(amp,cpu) {
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
    extent& operator+=(int value) restrict(amp,cpu) {
        base_.operator+=(value);
        return *this;
    }
    extent& operator-=(int value) restrict(amp,cpu) {
        base_.operator-=(value);
        return *this;
    }
    extent& operator*=(int value) restrict(amp,cpu) {
        base_.operator*=(value);
        return *this;
    }
    extent& operator/=(int value) restrict(amp,cpu) {
        base_.operator/=(value);
        return *this;
    }
    extent& operator%=(int value) restrict(amp,cpu) {
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
    extent& operator++() restrict(amp,cpu) {
        base_.operator+=(1);
        return *this;
    }
    extent operator++(int) restrict(amp,cpu) {
        extent ret = *this;
        base_.operator+=(1);
        return ret;
    }
    extent& operator--() restrict(amp,cpu) {
        base_.operator-=(1);
        return *this;
    }
    extent operator--(int) restrict(amp,cpu) {
        extent ret = *this;
        base_.operator-=(1);
        return ret;
    }

    /** @} */

    // FIXME: this function is not defined in C++AMP specification.
    template <int D0, int D1, int D2>
        explicit extent(const tiled_extent<D0, D1, D2>& other) restrict(amp,cpu)
            : base_(other.base_) {}

    constexpr
    operator const hc::extent<N>&() const
    {   // TODO: temporary, icky.
        return *reinterpret_cast<const hc::extent<N>* const>(this);
    }
private:
    typedef detail::index_impl<typename detail::__make_indices<N>::type> base;
    base base_;
    template <int K, typename Q> friend struct detail::index_helper;
    template <int K, typename Q1, typename Q2> friend struct detail::amp_helper;
};

// ------------------------------------------------------------------------
// utility class for tiled_barrier
// ------------------------------------------------------------------------

#ifndef CLK_LOCAL_MEM_FENCE
#define CLK_LOCAL_MEM_FENCE (1)
#endif

#ifndef CLK_GLOBAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE (2)
#endif

// ------------------------------------------------------------------------
// tile_barrier
// ------------------------------------------------------------------------

/**
 * The tile_barrier class is a capability class that is only creatable by the
 * system, and passed to a tiled parallel_for_each function object as part of
 * the tiled_index parameter. It provides member functions, such as wait, whose
 * purpose is to synchronize execution of threads running within the thread
 * tile.
 */
class tile_barrier {
public:
    /**
     * Copy constructor. Constructs a new tile_barrier from the supplied
     * argument "other".
     *
     * @param[in] other An object of type tile_barrier from which to initialize
     *                  this.
     */
    tile_barrier(const tile_barrier& other) restrict(amp,cpu) {}

    /**
     * Blocks execution of all threads in the thread tile until all threads in
     * the tile have reached this call. Establishes a memory fence on all
     * tile_static and global memory operations executed by the threads in the
     * tile such that all memory operations issued prior to hitting the barrier
     * are visible to all other threads after the barrier has completed and
     * none of the memory operations occurring after the barrier are executed
     * before hitting the barrier. This is identical to
     * wait_with_all_memory_fence().
     */
    void wait() const restrict(amp) {
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
    void wait_with_all_memory_fence() const restrict(amp) {
        amp_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    /**
     * Blocks execution of all threads in the thread tile until all threads in
     * the tile have reached this call. Establishes a memory fence on global
     * memory operations (but not tile-static memory operations) executed by
     * the threads in the tile such that all global memory operations issued
     * prior to hitting the barrier are visible to all other threads after the
     * barrier has completed and none of the global memory operations occurring
     * after the barrier are executed before hitting the barrier.
     */
    void wait_with_global_memory_fence() const restrict(amp) {
        amp_barrier(CLK_GLOBAL_MEM_FENCE);
    }

    /**
     * Blocks execution of all threads in the thread tile until all threads in
     * the tile have reached this call. Establishes a memory fence on
     * tile-static memory operations (but not global memory operations)
     * executed by the threads in the tile such that all tile_static memory
     * operations issued prior to hitting the barrier are visible to all other
     * threads after the barrier has completed and none of the tile-static
     * memory operations occurring after the barrier are executed before
     * hitting the barrier.
     */
    void wait_with_tile_static_memory_fence() const restrict(amp) {
        amp_barrier(CLK_LOCAL_MEM_FENCE);
    }

private:
    tile_barrier() restrict(amp) = default;

    template<int D0, int D1, int D2> friend
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
void all_memory_fence(const tile_barrier&) restrict(amp);

/**
 * Establishes a thread-tile scoped memory fence for global (but not
 * tile-static) memory operations. This function does not imply a barrier and
 * is therefore permitted in divergent code.
 */
// FIXME: this functions has not been implemented.
void global_memory_fence(const tile_barrier&) restrict(amp);

/**
 * Establishes a thread-tile scoped memory fence for tile-static (but not
 * global) memory operations. This function does not imply a barrier and is
 * therefore permitted in divergent code.
 */
// FIXME: this functions has not been implemented.
void tile_static_memory_fence(const tile_barrier&) restrict(amp);

// ------------------------------------------------------------------------
// tiled_index
// ------------------------------------------------------------------------

/**
 * Represents a set of related indices subdivided into 1-, 2-, or 3-dimensional
 * tiles.
 *
 * @tparam D0,D1,D2 The length of the tile in each specified dimension, where
 *                  D0 is the most-significant dimension and D2 is the
 *                  least-significant.
 */
template <int D0, int D1=0, int D2=0>
class tiled_index {
public:
    /**
     * A static member of tiled_index that contains the rank of this tiled
     * extent, and is either 1, 2, or 3 depending on the specialization used.
     */
    static const int rank = 3;

    // FIXME: missing constructor:
    // tiled_index(const index<N>& global,
    //             const index<N>& local,
    //             const index<N>& tile,
    //             const index<N>& tile_origin,
    //             const tile_barrier& barrier) restrict(amp,cpu);

    /**
     * Copy constructor. Constructs a new tiled_index from the supplied
     * argument "other".
     *
     * @param[in] other An object of type tiled_index from which to initialize
     *                  this.
     */
    tiled_index(const tiled_index<D0, D1, D2>& o) restrict(amp, cpu)
        : global(o.global), local(o.local), tile(o.tile), tile_origin(o.tile_origin), barrier(o.barrier) {}

    /**
     * An index of rank 1, 2, or 3 that represents the global index within an
     * extent.
     */
    const index<3> global;

    /**
     * An index of rank 1, 2, or 3 that represents the relative index within
     * the current tile of a tiled extent.
     */
    const index<3> local;

    /**
     * An index of rank 1, 2, or 3 that represents the coordinates of the
     * current tile of a tiled extent.
     */
    const index<3> tile;

    /**
     * An index of rank 1, 2, or 3 that represents the global coordinates of
     * the origin of the current tile within a tiled extent.
     */
    const index<3> tile_origin;

    /**
     * An object which represents a barrier within the current tile of threads.
     */
    const tile_barrier barrier;

    /**
     * Implicit conversion operator that converts a tiled_index<D0,D1,D2> into
     * an index<N>. The implicit conversion converts to the .global index
     * member.
     */
    operator const index<3>() const restrict(amp,cpu) {
        return global;
    }

    /** @{ */
    /**
     * Returns an instance of an extent<N> that captures the values of the
     * tiled_index template arguments D0, D1, and D2. For example:
     *
     * @code{.cpp}
     * index<3> zero;
     * tiled_index<64,16,4> ti(index<3>(256,256,256), zero, zero, zero, mybarrier);
     * extent<3> myTileExtent = ti.tile_extent;
     * assert(myTileExtent.tile_dim0 == 64);
     * assert(myTileExtent.tile_dim1 == 16);
     * assert(myTileExtent.tile_dim2 == 4);
     * @endcode
     */
    Concurrency::extent<3> get_tile_extent() const restrict(amp, cpu) {
      return tile_extent;
    }
    const Concurrency::extent<3> tile_extent;

    /** @} */

    /** @{ */
    /**
     * These constants allow access to the template arguments of tiled_index.
     */
    static const int tile_dim0 = D0;
    static const int tile_dim1 = D1;
    static const int tile_dim2 = D2;

    /** @} */


    // FIXME: this function is not defined in C++AMP specification.
    tiled_index(const index<3>& g) restrict(amp, cpu) : global(g) {}

private:
    tiled_index() restrict(amp)
        : global(index<3>(amp_get_global_id(2), amp_get_global_id(1), amp_get_global_id(0))),
          local(index<3>(amp_get_local_id(2), amp_get_local_id(1), amp_get_local_id(0))),
          tile(index<3>(amp_get_group_id(2), amp_get_group_id(1), amp_get_group_id(0))),
          tile_origin(index<3>(amp_get_global_id(2)-amp_get_local_id(2),
                               amp_get_global_id(1)-amp_get_local_id(1),
                               amp_get_global_id(0)-amp_get_local_id(0))),
          tile_extent(D0, D1, D2)
    {}

    template<typename K>
    friend
    void parallel_for_each(
        const accelerator_view&, const tiled_extent<D0, D1, D2>&, const K&);
    friend
    struct detail::Indexer;
};

/**
 * Represents a set of related indices subdivided into 1-, 2-, or 3-dimensional
 * tiles.
 *
 * @tparam D0,D1,D2 The length of the tile in each specified dimension, where
 *                  D0 is the most-significant dimension and D2 is the
 *                  least-significant.
 */
template <int D0>
class tiled_index<D0, 0, 0> {
public:
    /**
     * A static member of tiled_index that contains the rank of this tiled
     * extent, and is either 1, 2, or 3 depending on the specialization used.
     */
    static const int rank = 3;

    // FIXME: missing constructor:
    // tiled_index(const index<N>& global,
    //             const index<N>& local,
    //             const index<N>& tile,
    //             const index<N>& tile_origin,
    //             const tile_barrier& barrier) restrict(amp,cpu);

    /**
     * Copy constructor. Constructs a new tiled_index from the supplied
     * argument "other".
     *
     * @param[in] other An object of type tiled_index from which to initialize
     *                  this.
     */
    tiled_index(const tiled_index<D0>& o) restrict(amp, cpu)
        : global(o.global), local(o.local), tile(o.tile), tile_origin(o.tile_origin), barrier(o.barrier) {}

    /**
     * An index of rank 1, 2, or 3 that represents the global index within an
     * extent.
     */
    const index<1> global;

    /**
     * An index of rank 1, 2, or 3 that represents the relative index within
     * the current tile of a tiled extent.
     */
    const index<1> local;

    /**
     * An index of rank 1, 2, or 3 that represents the coordinates of the
     * current tile of a tiled extent.
     */
    const index<1> tile;

    /**
     * An index of rank 1, 2, or 3 that represents the global coordinates of
     * the origin of the current tile within a tiled extent.
     */
    const index<1> tile_origin;

    /**
     * An object which represents a barrier within the current tile of threads.
     */
    const tile_barrier barrier;

    /**
     * Implicit conversion operator that converts a tiled_index<D0,D1,D2> into
     * an index<N>. The implicit conversion converts to the .global index
     * member.
     */
    operator const index<1>() const restrict(amp,cpu) {
        return global;
    }

    /** @{ */
    /**
     * Returns an instance of an extent<N> that captures the values of the
     * tiled_index template arguments D0, D1, and D2. For example:
     *
     * @code{.cpp}
     * index<3> zero;
     * tiled_index<64,16,4> ti(index<3>(256,256,256), zero, zero, zero, mybarrier);
     * extent<3> myTileExtent = ti.tile_extent;
     * assert(myTileExtent.tile_dim0 == 64);
     * assert(myTileExtent.tile_dim1 == 16);
     * assert(myTileExtent.tile_dim2 == 4);
     * @endcode
     */
    Concurrency::extent<1> get_tile_extent() const restrict(amp, cpu) {
      return tile_extent;
    }
    const Concurrency::extent<1> tile_extent;

    /** @} */

    /** @{ */
    /**
     * These constants allow access to the template arguments of tiled_index.
     */
    static const int tile_dim0 = D0;

    /** @} */

    // FIXME: this function is not defined in C++AMP specification.
    tiled_index(const index<1>& g) restrict(amp, cpu) : global(g) {}

private:
    tiled_index() restrict(amp)
        : global(index<1>(amp_get_global_id(0))),
          local(index<1>(amp_get_local_id(0))),
          tile(index<1>(amp_get_group_id(0))),
          tile_origin(index<1>(amp_get_global_id(0)-amp_get_local_id(0))),
          tile_extent(D0)
    {}

    template<typename K> friend
    void parallel_for_each(
        const accelerator_view&, const tiled_extent<D0>&, const K&);
    friend
    struct detail::Indexer;
};

/**
 * Represents a set of related indices subdivided into 1-, 2-, or 3-dimensional
 * tiles.
 *
 * @tparam D0,D1,D2 The length of the tile in each specified dimension, where
 *                  D0 is the most-significant dimension and D2 is the
 *                  least-significant.
 */
template <int D0, int D1>
class tiled_index<D0, D1, 0> {
public:
    /**
     * A static member of tiled_index that contains the rank of this tiled
     * extent, and is either 1, 2, or 3 depending on the specialization used.
     */
    static const int rank = 2;

    // FIXME: missing constructor:
    // tiled_index(const index<N>& global,
    //             const index<N>& local,
    //             const index<N>& tile,
    //             const index<N>& tile_origin,
    //             const tile_barrier& barrier) restrict(amp,cpu);

    /**
     * Copy constructor. Constructs a new tiled_index from the supplied
     * argument "other".
     *
     * @param[in] other An object of type tiled_index from which to initialize
     *                  this.
     */
    tiled_index(const tiled_index<D0, D1>& o) restrict(amp, cpu)
        : global(o.global), local(o.local), tile(o.tile), tile_origin(o.tile_origin), barrier(o.barrier) {}

    /**
     * An index of rank 1, 2, or 3 that represents the global index within an
     * extent.
     */
    const index<2> global;

    /**
     * An index of rank 1, 2, or 3 that represents the relative index within
     * the current tile of a tiled extent.
     */
    const index<2> local;

    /**
     * An index of rank 1, 2, or 3 that represents the coordinates of the
     * current tile of a tiled extent.
     */
    const index<2> tile;

    /**
     * An index of rank 1, 2, or 3 that represents the global coordinates of
     * the origin of the current tile within a tiled extent.
     */
    const index<2> tile_origin;

    /**
     * An object which represents a barrier within the current tile of threads.
     */
    const tile_barrier barrier;

    /**
     * Implicit conversion operator that converts a tiled_index<D0,D1,D2> into
     * an index<N>. The implicit conversion converts to the .global index
     * member.
     */
    operator const index<2>() const restrict(amp,cpu) {
      return global;
    }

    /** @{ */
    /**
     * Returns an instance of an extent<N> that captures the values of the
     * tiled_index template arguments D0, D1, and D2. For example:
     *
     * @code{.cpp}
     * index<3> zero;
     * tiled_index<64,16,4> ti(index<3>(256,256,256), zero, zero, zero, mybarrier);
     * extent<3> myTileExtent = ti.tile_extent;
     * assert(myTileExtent.tile_dim0 == 64);
     * assert(myTileExtent.tile_dim1 == 16);
     * assert(myTileExtent.tile_dim2 == 4);
     * @endcode
     */
    Concurrency::extent<2> get_tile_extent() const restrict(amp, cpu) {
      return tile_extent;
    }
    const Concurrency::extent<2> tile_extent;

    /** @} */

    /** @{ */
    /**
     * These constants allow access to the template arguments of tiled_index.
     */
    static const int tile_dim0 = D0;
    static const int tile_dim1 = D1;

    /** @} */


    // FIXME: this function is not defined in C++AMP specification.
    tiled_index(const index<2>& g) restrict(amp, cpu) : global(g) {}

private:
    tiled_index() restrict(amp)
        : global(index<2>(amp_get_global_id(1), amp_get_global_id(0))),
          local(index<2>(amp_get_local_id(1), amp_get_local_id(0))),
          tile(index<2>(amp_get_group_id(1), amp_get_group_id(0))),
          tile_origin(index<2>(amp_get_global_id(1)-amp_get_local_id(1),
                               amp_get_global_id(0)-amp_get_local_id(0))),
          tile_extent(D0, D1)
    {}

    template<typename K>
    friend
    void parallel_for_each(
        const accelerator_view&, const tiled_extent<D0, D1>&, const K&);
    friend
    struct detail::Indexer;
};

// ------------------------------------------------------------------------
// tiled_extent
// ------------------------------------------------------------------------

/**
 * Represents an extent subdivided into 1-, 2-, or 3-dimensional tiles.
 *
 * @tparam D0,D1,D2 The length of the tile in each specified dimension, where
 *                  D0 is the most-significant dimension and D2 is the
 *                  least-significant.
 */
template <int D0, int D1/*=0*/, int D2/*=0*/>
class tiled_extent : public extent<3>
{
public:
    static_assert(D0 > 0, "Tile size must be positive");
    static_assert(D1 > 0, "Tile size must be positive");
    static_assert(D2 > 0, "Tile size must be positive");
    static const int rank = 3;

    /**
     * Default constructor. The origin and extent is default-constructed and
     * thus zero.
     */
    tiled_extent() restrict(amp,cpu) { }

    /**
     * Copy constructor. Constructs a new tiled_extent from the supplied
     * argument "other".
     *
     * @param[in] other An object of type tiled_extent from which to initialize
     *                  this new extent.
     */
    tiled_extent(const tiled_extent& other) restrict(amp,cpu): extent(other[0], other[1], other[2]) {}

    /**
     * Constructs a tiled_extent<N> with the extent "ext".
     * Notice that this constructor allows implicit conversions from extent<N>
     * to tiled_extent<N>.
     *
     * @param[in] ext The extent of this tiled_extent
     */
    tiled_extent(const extent<3>& ext) restrict(amp,cpu): extent(ext) {}

    /**
     * Assigns the component values of "other" to this tiled_extent<N> object.
     *
     * @param[in] other An object of type tiled_extent<N> from which to copy
     *                  into this.
     * @return Returns *this.
     */
    tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);

    /**
     * Returns a new tiled_extent with the extents adjusted up to be evenly
     * divisible by the tile dimensions. The origin of the new tiled_extent is
     * the same as the origin of this one.
     */
    tiled_extent pad() const restrict(amp,cpu) {
        tiled_extent padded(*this);
        padded[0] = (padded[0] <= D0) ? D0 : (((padded[0] + D0 - 1) / D0) * D0);
        padded[1] = (padded[1] <= D1) ? D1 : (((padded[1] + D1 - 1) / D1) * D1);
        padded[2] = (padded[2] <= D2) ? D2 : (((padded[2] + D2 - 1) / D2) * D2);
        return padded;
    }

    /**
     * Returns a new tiled_extent with the extents adjusted down to be evenly
     * divisible by the tile dimensions. The origin of the new tiled_extent is
     * the same as the origin of this one.
     */
    tiled_extent truncate() const restrict(amp,cpu) {
        tiled_extent trunc(*this);
        trunc[0] = (trunc[0]/D0) * D0;
        trunc[1] = (trunc[1]/D1) * D1;
        trunc[2] = (trunc[2]/D2) * D2;
        return trunc;
    }
  
    /**
     * Returns an instance of an extent<N> that captures the values of the
     * tiled_extent template arguments D0, D1, and D2. For example:
     *
     * @code{.cpp}
     * tiled_extent<64,16,4> tg;
     * extent<3> myTileExtent = tg.tile_extent;
     * assert(myTileExtent[0] == 64);
     * assert(myTileExtent[1] == 16);
     * assert(myTileExtent[2] == 4);
     * @endcode
     */
    // FIXME: this functions has not been implemented.
    extent<3> get_tile_extent() const;

    /** @{ */
    /**
     * These constants allow access to the template arguments of tiled_extent.
     */
    static const int tile_dim0 = D0;
    static const int tile_dim1 = D1;
    static const int tile_dim2 = D2;
    /** @} */

    /** @{ */
    /**
     * Compares two objects of tiled_extent<N>.
     *
     * The expression
     * lhs @f$\oplus@f$ rhs
     * is true if lhs.extent @f$\oplus@f$ rhs.extent and lhs.origin @f$\oplus@f$ rhs.origin.
     */
    friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
    friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);

    /** @} */
};

/**
 * Represents an extent subdivided into 1-, 2-, or 3-dimensional tiles.
 *
 * @tparam D0,D1,D2 The length of the tile in each specified dimension, where
 *                  D0 is the most-significant dimension and D2 is the
 *                  least-significant.
 */
template <int D0, int D1>
class tiled_extent<D0,D1,0> : public extent<2>
{
public:
    static_assert(D0 > 0, "Tile size must be positive");
    static_assert(D1 > 0, "Tile size must be positive");
    static const int rank = 2;

    /**
     * Default constructor. The origin and extent is default-constructed and
     * thus zero.
     */
    tiled_extent() restrict(amp,cpu) { }

    /**
     * Copy constructor. Constructs a new tiled_extent from the supplied
     * argument "other".
     *
     * @param[in] other An object of type tiled_extent from which to initialize
     *                  this new extent.
     */
    tiled_extent(const tiled_extent& other) restrict(amp,cpu):extent(other[0], other[1]) {}

    /**
     * Constructs a tiled_extent<N> with the extent "ext".
     * Notice that this constructor allows implicit conversions from extent<N>
     * to tiled_extent<N>.
     *
     * @param[in] ext The extent of this tiled_extent
     */
    tiled_extent(const extent<2>& ext) restrict(amp,cpu):extent(ext) {}

    /**
     * Assigns the component values of "other" to this tiled_extent<N> object.
     *
     * @param[in] other An object of type tiled_extent<N> from which to copy
     *                  into this.
     * @return Returns *this.
     */
    tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);

    /**
     * Returns a new tiled_extent with the extents adjusted up to be evenly
     * divisible by the tile dimensions. The origin of the new tiled_extent is
     * the same as the origin of this one.
     */
    tiled_extent pad() const restrict(amp,cpu) {
        tiled_extent padded(*this);
        padded[0] = (padded[0] <= D0) ? D0 : (((padded[0] + D0 - 1) / D0) * D0);
        padded[1] = (padded[1] <= D1) ? D1 : (((padded[1] + D1 - 1) / D1) * D1);
        return padded;
    }

    /**
     * Returns a new tiled_extent with the extents adjusted down to be evenly
     * divisible by the tile dimensions. The origin of the new tiled_extent is
     * the same as the origin of this one.
     */
    tiled_extent truncate() const restrict(amp,cpu) {
        tiled_extent trunc(*this);
        trunc[0] = (trunc[0]/D0) * D0;
        trunc[1] = (trunc[1]/D1) * D1;
        return trunc;
    }

    /**
     * Returns an instance of an extent<N> that captures the values of the
     * tiled_extent template arguments D0, D1, and D2. For example:
     *
     * @code{.cpp}
     * tiled_extent<64,16,4> tg;
     * extent<3> myTileExtent = tg.tile_extent;
     * assert(myTileExtent[0] == 64);
     * assert(myTileExtent[1] == 16);
     * assert(myTileExtent[2] == 4);
     * @endcode
     */
    // FIXME: this functions has not been implemented.
    extent<2> get_tile_extent() const;

    /** @{ */
    /**
     * These constants allow access to the template arguments of tiled_extent.
     */
    static const int tile_dim0 = D0;
    static const int tile_dim1 = D1;
    /** @} */

    /** @{ */
    /**
     * Compares two objects of tiled_extent<N>.
     *
     * The expression
     * lhs @f$\oplus@f$ rhs
     * is true if lhs.extent @f$\oplus@f$ rhs.extent and lhs.origin @f$\oplus@f$ rhs.origin.
     */
    friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
    friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);

    /** @} */
};

/**
 * Represents an extent subdivided into 1-, 2-, or 3-dimensional tiles.
 *
 * @tparam D0,D1,D2 The length of the tile in each specified dimension, where
 *                  D0 is the most-significant dimension and D2 is the
 *                  least-significant.
 */
template <int D0>
class tiled_extent<D0,0,0> : public extent<1>
{
public:
    static_assert(D0 > 0, "Tile size must be positive");
    static const int rank = 1;

    /**
     * Default constructor. The origin and extent is default-constructed and
     * thus zero.
     */
    tiled_extent() restrict(amp,cpu) { }

    /**
     * Copy constructor. Constructs a new tiled_extent from the supplied
     * argument "other".
     *
     * @param[in] other An object of type tiled_extent from which to initialize
     *                  this new extent.
     */
    tiled_extent(const tiled_extent& other) restrict(amp,cpu): extent(other[0]) {}

    /**
     * Constructs a tiled_extent<N> with the extent "ext".
     * Notice that this constructor allows implicit conversions from extent<N>
     * to tiled_extent<N>.
     *
     * @param[in] ext The extent of this tiled_extent
     */
    tiled_extent(const extent<1>& ext) restrict(amp,cpu):extent(ext) {}

    /**
     * Assigns the component values of "other" to this tiled_extent<N> object.
     *
     * @param[in] other An object of type tiled_extent<N> from which to copy
     *                  into this.
     * @return Returns *this.
     */
    tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);

    /**
     * Returns a new tiled_extent with the extents adjusted up to be evenly
     * divisible by the tile dimensions. The origin of the new tiled_extent is
     * the same as the origin of this one.
     */
    tiled_extent pad() const restrict(amp,cpu) {
        tiled_extent padded(*this);
        padded[0] = (padded[0] <= D0) ? D0 : (((padded[0] + D0 - 1) / D0) * D0);
        return padded;
    }

    /**
     * Returns a new tiled_extent with the extents adjusted down to be evenly
     * divisible by the tile dimensions. The origin of the new tiled_extent is
     * the same as the origin of this one.
     */
    tiled_extent truncate() const restrict(amp,cpu) {
        tiled_extent trunc(*this);
        trunc[0] = (trunc[0]/D0) * D0;
        return trunc;
    }

    /**
     * Returns an instance of an extent<N> that captures the values of the
     * tiled_extent template arguments D0, D1, and D2. For example:
     *
     * @code{.cpp}
     * tiled_extent<64,16,4> tg;
     * extent<3> myTileExtent = tg.tile_extent;
     * assert(myTileExtent[0] == 64);
     * assert(myTileExtent[1] == 16);
     * assert(myTileExtent[2] == 4);
     * @endcode
     */
    // FIXME: this functions has not been implemented.
    extent<1> get_tile_extent() const;

    /** @{ */
    /**
     * These constants allow access to the template arguments of tiled_extent.
     */
    static const int tile_dim0 = D0;
    /** @} */


    /** @{ */
    /**
     * Compares two objects of tiled_extent<N>.
     *
     * The expression
     * lhs @f$\oplus@f$ rhs
     * is true if lhs.extent @f$\oplus@f$ rhs.extent and lhs.origin @f$\oplus@f$ rhs.origin.
     */
    friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
    friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);

    /** @} */
};

// ------------------------------------------------------------------------
// utility helper classes for array_view
// ------------------------------------------------------------------------

template <typename T, int N>
struct projection_helper
{
    // array_view<T,N>, where N>1
    //    array_view<T,N-1> operator[](int i) const restrict(amp,cpu)
    static_assert(N > 1, "projection_helper is only supported on array_view with a rank of 2 or higher");
    typedef array_view<T, N - 1> result_type;
    static result_type project(array_view<T, N>& now, int stride) restrict(amp,cpu) {
        int ext[N - 1], i, idx[N - 1], ext_o[N - 1];
        for (i = N - 1; i > 0; --i) {
            ext_o[i - 1] = now.extent[i];
            ext[i - 1] = now.extent_base[i];
            idx[i - 1] = now.index_base[i];
        }
        stride += now.index_base[0];
        Concurrency::extent<N - 1> ext_now(ext_o);
        Concurrency::extent<N - 1> ext_base(ext);
        Concurrency::index<N - 1> idx_base(idx);
        return result_type (now.cache, ext_now, ext_base, idx_base,
                            now.offset + ext_base.size() * stride);
    }
    static result_type project(const array_view<T, N>& now, int stride) restrict(amp,cpu) {
        int ext[N - 1], i, idx[N - 1], ext_o[N - 1];
        for (i = N - 1; i > 0; --i) {
            ext_o[i - 1] = now.extent[i];
            ext[i - 1] = now.extent_base[i];
            idx[i - 1] = now.index_base[i];
        }
        stride += now.index_base[0];
        Concurrency::extent<N - 1> ext_now(ext_o);
        Concurrency::extent<N - 1> ext_base(ext);
        Concurrency::index<N - 1> idx_base(idx);
        return result_type (now.cache, ext_now, ext_base, idx_base,
                            now.offset + ext_base.size() * stride);
    }
};
template <typename T>
struct projection_helper<T, 1>
{
    // array_view<T,1>
    //      T& operator[](int i) const restrict(amp,cpu);
    typedef T& result_type;
    static result_type project(array_view<T, 1>& now, int i) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
        now.cache.get_cpu_access(true);
#endif
        T *ptr = reinterpret_cast<T *>(now.cache.get() + i + now.offset + now.index_base[0]);
        return *ptr;
    }
    static result_type project(const array_view<T, 1>& now, int i) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
        now.cache.get_cpu_access(true);
#endif
        T *ptr = reinterpret_cast<T *>(now.cache.get() + i + now.offset + now.index_base[0]);
        return *ptr;
    }
};
template <typename T, int N>
struct projection_helper<const T, N>
{
    // array_view<T,N>, where N>1
    //    array_view<const T,N-1> operator[](int i) const restrict(amp,cpu);
    static_assert(N > 1, "projection_helper is only supported on array_view with a rank of 2 or higher");
    typedef array_view<const T, N - 1> const_result_type;
    static const_result_type project(array_view<const T, N>& now, int stride) restrict(amp,cpu) {
        int ext[N - 1], i, idx[N - 1], ext_o[N - 1];
        for (i = N - 1; i > 0; --i) {
            ext_o[i - 1] = now.extent[i];
            ext[i - 1] = now.extent_base[i];
            idx[i - 1] = now.index_base[i];
        }
        stride += now.index_base[0];
        Concurrency::extent<N - 1> ext_now(ext_o);
        Concurrency::extent<N - 1> ext_base(ext);
        Concurrency::index<N - 1> idx_base(idx);
        auto ret = const_result_type (now.cache, ext_now, ext_base, idx_base,
                                      now.offset + ext_base.size() * stride);
        return ret;
    }
    static const_result_type project(const array_view<const T, N>& now, int stride) restrict(amp,cpu) {
        int ext[N - 1], i, idx[N - 1], ext_o[N - 1];
        for (i = N - 1; i > 0; --i) {
            ext_o[i - 1] = now.extent[i];
            ext[i - 1] = now.extent_base[i];
            idx[i - 1] = now.index_base[i];
        }
        stride += now.index_base[0];
        Concurrency::extent<N - 1> ext_now(ext_o);
        Concurrency::extent<N - 1> ext_base(ext);
        Concurrency::index<N - 1> idx_base(idx);
        auto ret = const_result_type (now.cache, ext_now, ext_base, idx_base,
                                      now.offset + ext_base.size() * stride);
        return ret;
    }
};
template <typename T>
struct projection_helper<const T, 1>
{
    // array_view<const T,1>
    //      const T& operator[](int i) const restrict(amp,cpu);
    typedef const T& const_result_type;
    static const_result_type project(array_view<const T, 1>& now, int i) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
        now.cache.get_cpu_access();
#endif
        const T *ptr = reinterpret_cast<const T *>(now.cache.get() + i + now.offset + now.index_base[0]);
        return *ptr;
    }
    static const_result_type project(const array_view<const T, 1>& now, int i) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
        now.cache.get_cpu_access();
#endif
        const T *ptr = reinterpret_cast<const T *>(now.cache.get() + i + now.offset + now.index_base[0]);
        return *ptr;
    }
};

// ------------------------------------------------------------------------
// utility helper classes for array
// ------------------------------------------------------------------------

template <typename T, int N>
struct array_projection_helper
{
    // array<T,N>, where N>1
    //     array_view<T,N-1> operator[](int i0) restrict(amp,cpu);
    //     array_view<const T,N-1> operator[](int i0) const restrict(amp,cpu);
    static_assert(N > 1, "projection_helper is only supported on array with a rank of 2 or higher");
    typedef array_view<T, N - 1> result_type;
    typedef array_view<const T, N - 1> const_result_type;
    static result_type project(array<T, N>& now, int stride) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
        if( stride < 0)
          throw runtime_exception("errorMsg_throw", 0);
#endif
        int comp[N - 1], i;
        for (i = N - 1; i > 0; --i)
            comp[i - 1] = now.extent[i];
        Concurrency::extent<N - 1> ext(comp);
        int offset = ext.size() * stride;
#if __KALMAR_ACCELERATOR__ != 1
        if( offset >= now.extent.size())
          throw runtime_exception("errorMsg_throw", 0);
#endif
        return result_type(now.m_device, ext, ext, index<N - 1>(), offset);
    }
    static const_result_type project(const array<T, N>& now, int stride) restrict(amp,cpu) {
        int comp[N - 1], i;
        for (i = N - 1; i > 0; --i)
            comp[i - 1] = now.extent[i];
        Concurrency::extent<N - 1> ext(comp);
        int offset = ext.size() * stride;
        return const_result_type(now.m_device, ext, ext, index<N - 1>(), offset);
    }
};
template <typename T>
struct array_projection_helper<T, 1>
{
    // array<T,1>
    //    T& operator[](int i0) restrict(amp,cpu);
    //    const T& operator[](int i0) const restrict(amp,cpu);
    typedef T& result_type;
    typedef const T& const_result_type;
    static result_type project(array<T, 1>& now, int i) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
        now.m_device.synchronize(true);
#endif
        T *ptr = reinterpret_cast<T *>(now.m_device.get() + i);
        return *ptr;
    }
    static const_result_type project(const array<T, 1>& now, int i) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
        now.m_device.synchronize();
#endif
        const T *ptr = reinterpret_cast<const T *>(now.m_device.get() + i);
        return *ptr;
    }
};

template <int N>
const Concurrency::extent<N>& check(const Concurrency::extent<N>& ext)
{
#if __KALMAR_ACCELERATOR__ != 1
    for (int i = 0; i < N; i++)
    {
        if(ext[i] <=0)
            throw runtime_exception("errorMsg_throw", 0);
    }
#endif
    return ext;
}

// ------------------------------------------------------------------------
// forward declarations of copy routines used by array / array_view
// ------------------------------------------------------------------------

template <typename T, int N>
void copy(const array_view<const T, N>& src, const array_view<T, N>& dest);

template <typename T, int N>
void copy(const array_view<T, N>& src, const array_view<T, N>& dest);

template <typename T, int N>
void copy(const array<T, N>& src, const array_view<T, N>& dest);

template <typename T, int N>
void copy(const array<T, N>& src, array<T, N>& dest);

template <typename T, int N>
void copy(const array_view<const T, N>& src, array<T, N>& dest);

template <typename T, int N>
void copy(const array_view<T, N>& src, array<T, N>& dest);

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest);

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest);

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, const array_view<T, N>& dest);

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, array<T, N>& dest);

template <typename OutputIter, typename T, int N>
void copy(const array_view<T, N> &src, OutputIter destBegin);

template <typename OutputIter, typename T, int N>
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

// ------------------------------------------------------------------------
// utility classes for array_view
// ------------------------------------------------------------------------

template <typename T>
struct __has_data
{
private:
    struct two {char __lx; char __lxx;};
    template <typename C> static char test(decltype(std::declval<C>().data()));
    template <typename C> static two test(...);
public:
    static const bool value = sizeof(test<T>(0)) == 1;
};

template <typename T>
struct __has_size
{
private:
    struct two {char __lx; char __lxx;};
    template <typename C> static char test(decltype(&C::size));
    template <typename C> static two test(...);
public:
    static const bool value = sizeof(test<T>(0)) == 1;
};

template <typename T>
struct __is_container
{
    using _T = typename std::remove_reference<T>::type;
    static const bool value = __has_size<_T>::value && __has_data<_T>::value;
};

// ------------------------------------------------------------------------
// array_view<T,N>
// ------------------------------------------------------------------------

/**
 * The array_view<T,N> type represents a possibly cached view into the data
 * held in an array<T,N>, or a section thereof. It also provides such views
 * over native CPU data. It exposes an indexing interface congruent to that of
 * array<T,N>.
 */

// ------------------------------------------------------------------------
// array_view<const T,N>
// ------------------------------------------------------------------------

/**
 * The partial specialization array_view<const T,N> represents a view over
 * elements of type const T with rank N. The elements are readonly. At the
 * boundary of a call site (such as parallel_for_each), this form of array_view
 * need only be copied to the target accelerator if it isn't already there. It
 * will not be copied out.
 */

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
template <int N>
extent<N> operator+(const extent<N>& lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r += rhs;
    return __r;
}
template <int N>
extent<N> operator-(const extent<N>& lhs, const extent<N>& rhs) restrict(amp,cpu) {
    extent<N> __r = lhs;
    __r -= rhs;
    return __r;
}

/** @} */

/** @{ */
/**
 * Binary arithmetic operations that produce a new extent<N> that is the result
 * of performing the corresponding binary arithmetic operation on the elements
 * of the extent operands. The result extent<N> is such that for a given
 * operator @f$\oplus@f$,
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
template <int N>
extent<N> operator+(const extent<N>& ext, int value) restrict(amp,cpu) {
    extent<N> __r = ext;
    __r += value;
    return __r;
}
template <int N>
extent<N> operator+(int value, const extent<N>& ext) restrict(amp,cpu) {
    extent<N> __r = ext;
    __r += value;
    return __r;
}
template <int N>
extent<N> operator-(const extent<N>& ext, int value) restrict(amp,cpu) {
    extent<N> __r = ext;
    __r -= value;
    return __r;
}
template <int N>
extent<N> operator-(int value, const extent<N>& ext) restrict(amp,cpu) {
    extent<N> __r(value);
    __r -= ext;
    return __r;
}
template <int N>
extent<N> operator*(const extent<N>& ext, int value) restrict(amp,cpu) {
    extent<N> __r = ext;
    __r *= value;
    return __r;
}
template <int N>
extent<N> operator*(int value, const extent<N>& ext) restrict(amp,cpu) {
    extent<N> __r = ext;
    __r *= value;
    return __r;
}
template <int N>
extent<N> operator/(const extent<N>& ext, int value) restrict(amp,cpu) {
    extent<N> __r = ext;
    __r /= value;
    return __r;
}
template <int N>
extent<N> operator/(int value, const extent<N>& ext) restrict(amp,cpu) {
    extent<N> __r(value);
    __r /= ext;
    return __r;
}
template <int N>
extent<N> operator%(const extent<N>& ext, int value) restrict(amp,cpu) {
    extent<N> __r = ext;
    __r %= value;
    return __r;
}
template <int N>
extent<N> operator%(int value, const extent<N>& ext) restrict(amp,cpu) {
    extent<N> __r(value);
    __r %= ext;
    return __r;
}

/** @} */

// ------------------------------------------------------------------------
// utility functions for copy
// ------------------------------------------------------------------------

template<typename T, int N>
static inline bool is_flat(const array_view<T, N>& av) noexcept {
    return av.extent == av.extent_base && av.index_base == index<N>();
}

template<typename T>
static inline bool is_flat(const array_view<T, 1>& av) noexcept { return true; }

template <typename InputIter, typename T, int N, int dim>
struct copy_input
{
    void operator()(InputIter& It, T* ptr, const extent<N>& ext,
                    const extent<N>& base, const index<N>& idx)
    {
        size_t stride = 1;
        for (int i = dim; i < N; i++)
            stride *= base[i];
        ptr += stride * idx[dim - 1];
        for (int i = 0; i < ext[dim - 1]; i++) {
            copy_input<InputIter, T, N, dim + 1>()(It, ptr, ext, base, idx);
            ptr += stride;
        }
    }
};

template <typename InputIter, typename T, int N>
struct copy_input<InputIter, T, N, N>
{
    void operator()(InputIter& It, T* ptr, const extent<N>& ext,
                    const extent<N>& base, const index<N>& idx)
    {
        InputIter end = It;
        std::advance(end, ext[N - 1]);
        std::copy(It, end, ptr + idx[N - 1]);
        It = end;
    }
};

template <typename OutputIter, typename T, int N, int dim>
struct copy_output
{
    void operator()(const T* ptr, OutputIter& It, const extent<N>& ext,
                    const extent<N>& base, const index<N>& idx)
    {
        size_t stride = 1;
        for (int i = dim; i < N; i++)
            stride *= base[i];
        ptr += stride * idx[dim - 1];
        for (int i = 0; i < ext[dim - 1]; i++) {
            copy_output<OutputIter, T, N, dim + 1>()(ptr, It, ext, base, idx);
            ptr += stride;
        }
    }
};

template <typename OutputIter, typename T, int N>
struct copy_output<OutputIter, T, N, N>
{
    void operator()(const T* ptr, OutputIter& It, const extent<N>& ext,
                    const extent<N>& base, const index<N>& idx)
    {
        ptr += idx[N - 1];
        It = std::copy(ptr, ptr + ext[N - 1], It);
    }
};

template <typename T, int N, int dim>
struct copy_bidir
{
    void operator()(const T* src, T* dst, const extent<N>& ext,
                    const extent<N>& base1, const index<N>& idx1,
                    const extent<N>& base2, const index<N>& idx2)
    {
        size_t stride1 = 1;
        for (int i = dim; i < N; i++)
            stride1 *= base1[i];
        src += stride1 * idx1[dim - 1];

        size_t stride2 = 1;
        for (int i = dim; i < N; i++)
            stride2 *= base2[i];
        dst += stride2 * idx2[dim - 1];

        for (int i = 0; i < ext[dim - 1]; i++) {
            copy_bidir<T, N, dim + 1>()(src, dst, ext, base1, idx1, base2, idx2);
            src += stride1;
            dst += stride2;
        }
    }
};

template <typename T, int N>
struct copy_bidir<T, N, N>
{
    void operator()(const T* src, T* dst, const extent<N>& ext,
                    const extent<N>& base1, const index<N>& idx1,
                    const extent<N>& base2, const index<N>& idx2)
    {
        src += idx1[N - 1];
        dst += idx2[N - 1];
        std::copy(src, src + ext[N - 1], dst);
    }
};

template <typename Iter, typename T, int N>
struct do_copy
{
    template<template <typename, int> class _amp_container>
    void operator()(Iter srcBegin, Iter srcEnd, const _amp_container<T, N>& dest) {
        size_t size = dest.get_extent().size();
        size_t offset = dest.get_offset();
        bool modify = true;

        T* ptr = dest.internal().map_ptr(modify, size, offset);
        std::copy(srcBegin, srcEnd, ptr);
        dest.internal().unmap_ptr(ptr, modify, size, offset);
    }
    template<template <typename, int> class _amp_container>
    void operator()(const _amp_container<T, N> &src, Iter destBegin) {
        size_t size = src.get_extent().size();
        size_t offset = src.get_offset();
        bool modify = false;

        const T* ptr = src.internal().map_ptr(modify, size, offset);
        std::copy(ptr, ptr + src.get_extent().size(), destBegin);
        src.internal().unmap_ptr(ptr, modify, size, offset);
    }
};

template <typename Iter, typename T>
struct do_copy<Iter, T, 1>
{
    template<template <typename, int> class _amp_container>
    void operator()(Iter srcBegin, Iter srcEnd, const _amp_container<T, 1>& dest) {
        size_t size = dest.get_extent().size();
        size_t offset = dest.get_offset() + dest.get_index_base()[0];
        bool modify = true;

        T* ptr = dest.internal().map_ptr(modify, size, offset);
        std::copy(srcBegin, srcEnd, ptr);
        dest.internal().unmap_ptr(ptr, modify, size, offset);
    }
    template<template <typename, int> class _amp_container>
    void operator()(const _amp_container<T, 1> &src, Iter destBegin) {
        size_t size = src.get_extent().size();
        size_t offset = src.get_offset() + src.get_index_base()[0];
        bool modify = false;

        const T* ptr = src.internal().map_ptr(modify, size, offset);
        std::copy(ptr, ptr + src.get_extent().size(), destBegin);
        src.internal().unmap_ptr(ptr, modify, size, offset);
    }
};

template <typename T, int N>
struct do_copy<T*, T, N>
{
    template<template <typename, int> class _amp_container>
    void operator()(T* srcBegin, T* srcEnd, const _amp_container<T, N>& dest) {
        dest.internal().write(srcBegin, std::distance(srcBegin, srcEnd), dest.get_offset(), true);
    }
    template<template <typename, int> class _amp_container>
    void operator()(const _amp_container<T, N> &src, T* destBegin) {
        src.internal().read(destBegin, src.get_extent().size(), src.get_offset());
    }
};

template <typename T>
struct do_copy<T*, T, 1>
{
    template<template <typename, int> class _amp_container>
    void operator()(const T* srcBegin, const T* srcEnd, const _amp_container<T, 1>& dest) {
        dest.internal().write(srcBegin, std::distance(srcBegin, srcEnd),
                              dest.get_offset() + dest.get_index_base()[0], true);
    }
    template<template <typename, int> class _amp_container>
    void operator()(const _amp_container<T, 1> &src, T* destBegin) {
        src.internal().read(destBegin, src.get_extent().size(),
                            src.get_offset() + src.get_index_base()[0]);
    }
};

// ------------------------------------------------------------------------
// copy
// ------------------------------------------------------------------------

/**
 * The contents of "src" are copied into "dest". The source and destination may
 * reside on different accelerators. If the extents of "src" and "dest" don't
 * match, a runtime exception is thrown.
 *
 * @param[in] src An object of type array<T,N> to be copied from.
 * @param[out] dest An object of type array<T,N> to be copied to.
 */
template <typename T, int N>
void copy(const array<T, N>& src, array<T, N>& dest) {
    src.internal().copy(dest.internal(), 0, 0, 0);
}

/** @{ */
/**
 * The contents of "src" are copied into "dest". If the extents of "src" and
 * "dest" don't match, a runtime exception is thrown.
 *
 * @param[in] src An object of type array<T,N> to be copied from.
 * @param[out] dest An object of type array_view<T,N> to be copied to.
 */
template <typename T, int N>
void copy(const array<T, N>& src, const array_view<T, N>& dest) {
    if (is_flat(dest))
        src.internal().copy(dest.internal(), src.get_offset(),
                            dest.get_offset(), dest.get_extent().size());
    else {
        // FIXME: logic here deserve to be reviewed
        size_t srcSize = src.extent.size();
        size_t srcOffset = 0;
        bool srcModify = false;
        size_t destSize = dest.extent_base.size();
        size_t destOffset = dest.offset;
        bool destModify = true;

        T* pSrc = src.internal().map_ptr(srcModify, srcSize, srcOffset);
        T* p = pSrc;
        T* pDst = dest.internal().map_ptr(destModify, destSize, destOffset);
        copy_input<T*, T, N, 1>()(pSrc, pDst, dest.extent, dest.extent_base, dest.index_base);
        dest.internal().unmap_ptr(pDst, destModify, destSize, destOffset);
        src.internal().unmap_ptr(p, srcModify, srcSize, srcOffset);
    }
}

template <typename T>
void copy(const array<T, 1>& src, const array_view<T, 1>& dest) {
    src.internal().copy(dest.internal(),
                        src.get_offset() + src.get_index_base()[0],
                        dest.get_offset() + dest.get_index_base()[0],
                        dest.get_extent().size());
}

/** @} */

/** @{ */
/**
 * The contents of "src" are copied into "dest". If the extents of "src" and
 * "dest" don't match, a runtime exception is thrown.
 *
 * @param[in] src An object of type array_view<T,N> (or array_view<const T, N>)
 *                to be copied from.
 * @param[out] dest An object of type array<T,N> to be copied to.
 */
template <typename T, int N>
void copy(const array_view<const T, N>& src, array<T, N>& dest) {
    if (is_flat(src)) {
        src.internal().copy(dest.internal(), src.get_offset(),
                            dest.get_offset(), dest.get_extent().size());
    } else {
        // FIXME: logic here deserve to be reviewed
        size_t srcSize = src.extent_base.size();
        size_t srcOffset = src.offset;
        bool srcModify = false;
        size_t destSize = dest.extent.size();
        size_t destOffset = 0;
        bool destModify = true;

        T* pDst = dest.internal().map_ptr(destModify, destSize, destOffset);
        T* p = pDst;
        const T* pSrc = src.internal().map_ptr(srcModify, srcSize, srcOffset);
        copy_output<T*, T, N, 1>()(pSrc, pDst, src.extent, src.extent_base, src.index_base);
        src.internal().unmap_ptr(pSrc, srcModify, srcSize, srcOffset);
        dest.internal().unmap_ptr(p, destModify, destSize, destOffset);
    }
}

template <typename T, int N>
void copy(const array_view<T, N>& src, array<T, N>& dest) {
    const array_view<const T, N> buf(src);
    copy(buf, dest);
}

template <typename T>
void copy(const array_view<const T, 1>& src, array<T, 1>& dest) {
    src.internal().copy(dest.internal(),
                        src.get_offset() + src.get_index_base()[0],
                        dest.get_offset() + dest.get_index_base()[0],
                        dest.get_extent().size());
}

/** @} */

/** @{ */
/**
 * The contents of "src" are copied into "dest". If the extents of "src" and
 * "dest" don't match, a runtime exception is thrown.
 *
 * @param[in] src An object of type array_view<T,N> (or array_view<const T, N>)
 *                to be copied from.
 * @param[out] dest An object of type array_view<T,N> to be copied to.
 */
template <typename T, int N>
void copy(const array_view<const T, N>& src, const array_view<T, N>& dest) {
    if (is_flat(src)) {
        if (is_flat(dest))
            src.internal().copy(dest.internal(), src.get_offset(),
                                dest.get_offset(), dest.get_extent().size());
        else {
            // FIXME: logic here deserve to be reviewed
            size_t srcSize = src.extent.size();
            size_t srcOffset = 0;
            bool srcModify = false;
            size_t destSize = dest.extent_base.size();
            size_t destOffset = dest.offset;
            bool destModify = true;

            const T* pSrc = src.internal().map_ptr(srcModify, srcSize, srcOffset);
            const T* p = pSrc;
            T* pDst = dest.internal().map_ptr(destModify, destSize, destOffset);
            copy_input<const T*, T, N, 1>()(pSrc, pDst, dest.extent, dest.extent_base, dest.index_base);
            dest.internal().unmap_ptr(pDst, destModify, destSize, destOffset);
            src.internal().unmap_ptr(p, srcModify, srcSize, srcOffset);
        }
    } else {
        if (is_flat(dest)) {
            // FIXME: logic here deserve to be reviewed
            size_t srcSize = src.extent_base.size();
            size_t srcOffset = src.offset;
            bool srcModify = false;
            size_t destSize = dest.extent.size();
            size_t destOffset = 0;
            bool destModify = true;

            T* pDst = dest.internal().map_ptr(destModify, destSize, destOffset);
            T* p = pDst;
            const T* pSrc = src.internal().map_ptr(srcModify, srcSize, srcOffset);
            copy_output<T*, T, N, 1>()(pSrc, pDst, src.extent, src.extent_base, src.index_base);
            dest.internal().unmap_ptr(p, destModify, destSize, destOffset);
            src.internal().unmap_ptr(pSrc, srcModify, srcSize, srcOffset);
        } else {
            // FIXME: logic here deserve to be reviewed
            size_t srcSize = src.extent_base.size();
            size_t srcOffset = src.offset;
            bool srcModify = false;
            size_t destSize = dest.extent_base.size();
            size_t destOffset = dest.offset;
            bool destModify = true;

            const T* pSrc = src.internal().map_ptr(srcModify, srcSize, srcOffset);
            T* pDst = dest.internal().map_ptr(destModify, destSize, destOffset);
            copy_bidir<T, N, 1>()(pSrc, pDst, src.extent, src.extent_base,
                                  src.index_base, dest.extent_base, dest.index_base);
            dest.internal().unmap_ptr(pDst, destModify, destSize, destOffset);
            src.internal().unmap_ptr(pSrc, srcModify, srcSize, srcOffset);
        }
    }
}

template <typename T, int N>
void copy(const array_view<T, N>& src, const array_view<T, N>& dest) {
    const array_view<const T, N> buf(src);
    copy(buf, dest);
}

template <typename T>
void copy(const array_view<const T, 1>& src, const array_view<T, 1>& dest) {
    src.internal().copy(dest.internal(),
                        src.get_offset() + src.get_index_base()[0],
                        dest.get_offset() + dest.get_index_base()[0],
                        dest.get_extent().size());
}

/** @} */

/** @{ */
/**
 * The contents of a source container from the iterator range [srcBegin,srcEnd)
 * are copied into "dest". If the number of elements in the iterator range is
 * not equal to "dest.extent.size()", an exception is thrown.
 *
 * In the overloads which don't take an end-iterator it is assumed that the
 * source iterator is able to provide at least dest.extent.size() elements, but
 * no checking is performed (nor possible).
 *
 * @param[in] srcBegin An iterator to the first element of a source container.
 * @param[in] srcEnd An interator to the end of a source container.
 * @param[out] dest An object of type array<T,N> to be copied to.
 */
template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest) {
#if __KALMAR_ACCELERATOR__ != 1
    if( ( std::distance(srcBegin,srcEnd) <=0 )||( std::distance(srcBegin,srcEnd) < dest.get_extent().size() ))
      throw runtime_exception("errorMsg_throw ,copy between different types", 0);
#endif
    do_copy<InputIter, T, N>()(srcBegin, srcEnd, dest);
}

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, array<T, N>& dest) {
    InputIter srcEnd = srcBegin;
    std::advance(srcEnd, dest.get_extent().size());
    Concurrency::copy(srcBegin, srcEnd, dest);
}

/** @} */

/** @{ */
/**
 * The contents of a source container from the iterator range [srcBegin,srcEnd)
 * are copied into "dest". If the number of elements in the iterator range is
 * not equal to "dest.extent.size()", an exception is thrown.
 *
 * In the overloads which don't take an end-iterator it is assumed that the
 * source iterator is able to provide at least dest.extent.size() elements, but
 * no checking is performed (nor possible).
 *
 * @param[in] srcBegin An iterator to the first element of a source container.
 * @param[in] srcEnd An interator to the end of a source container.
 * @param[out] dest An object of type array_view<T,N> to be copied to.
 */
template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest) {
    if (is_flat(dest))
        do_copy<InputIter, T, N>()(srcBegin, srcEnd, dest);
   else {
        size_t size = dest.extent_base.size();
        size_t offset = dest.offset;
        bool modify = true;

        T* ptr = dest.internal().map_ptr(modify, size, offset);
        copy_input<InputIter, T, N, 1>()(srcBegin, ptr, dest.extent, dest.extent_base, dest.index_base);
        dest.internal().unmap_ptr(ptr, modify, size, offset);
    }
}

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, const array_view<T, N>& dest) {
    InputIter srcEnd = srcBegin;
    std::advance(srcEnd, dest.get_extent().size());
    copy(srcBegin, srcEnd, dest);
}

/** @} */

/**
 * The contents of a source array are copied into "dest" starting with iterator
 * destBegin. If the number of elements in the range starting destBegin in the
 * destination container is smaller than "src.extent.size()", the behavior is
 * undefined.
 *
 * @param[in] src An object of type array<T,N> to be copied from.
 * @param[out] destBegin An output iterator addressing the position of the
 *                       first element in the destination container.
 */
template <typename OutputIter, typename T, int N>
void copy(const array<T, N> &src, OutputIter destBegin) {
    do_copy<OutputIter, T, N>()(src, destBegin);
}

/**
 * The contents of a source array are copied into "dest" starting with iterator
 * destBegin. If the number of elements in the range starting destBegin in the
 * destination container is smaller than "src.extent.size()", the behavior is
 * undefined.
 *
 * @param[in] src An object of type array_view<T,N> to be copied from.
 * @param[out] destBegin An output iterator addressing the position of the
 *                       first element in the destination container.
 */
template <typename OutputIter, typename T, int N>
void copy(const array_view<T, N> &src, OutputIter destBegin) {
    if (is_flat(src))
        do_copy<OutputIter, T, N>()(src, destBegin);
    else {
        size_t size = src.extent_base.size();
        size_t offset = src.offset;
        bool modify = false;

        T* ptr = src.internal().map_ptr(modify, size, offset);
        copy_output<OutputIter, T, N, 1>()(ptr, destBegin, src.extent, src.extent_base, src.index_base);
        src.internal().unmap_ptr(ptr, modify, size, offset);
    }
}

// ------------------------------------------------------------------------
// utility function for copy_async
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copy_async
// ------------------------------------------------------------------------

/**
 * The contents of "src" are copied into "dest". The source and destination may
 * reside on different accelerators. If the extents of "src" and "dest" don't
 * match, a runtime exception is thrown.
 *
 * @param[in] src An object of type array<T,N> to be copied from.
 * @param[out] dest An object of type array<T,N> to be copied to.
 */
template <typename T, int N>
completion_future copy_async(const array<T, N>& src, array<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

/**
 * The contents of "src" are copied into "dest". If the extents of "src" and
 * "dest" don't match, a runtime exception is thrown.
 *
 * @param[in] src An object of type array<T,N> to be copied from.
 * @param[out] dest An object of type array_view<T,N> to be copied to.
 */
template <typename T, int N>
completion_future copy_async(const array<T, N>& src, const array_view<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

/** @{ */
/**
 * The contents of "src" are copied into "dest". If the extents of "src" and
 * "dest" don't match, a runtime exception is thrown.
 *
 * @param[in] src An object of type array_view<T,N> (or array_view<const T, N>)
 *                to be copied from.
 * @param[out] dest An object of type array<T,N> to be copied to.
 */
template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, array<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, array<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

/** @} */

/** @{ */
/**
 * The contents of "src" are copied into "dest". If the extents of "src" and
 * "dest" don't match, a runtime exception is thrown.
 *
 * @param[in] src An object of type array_view<T,N> (or array_view<const T, N>)
 *                to be copied from.
 * @param[out] dest An object of type array_view<T,N> to be copied to.
 */
template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, const array_view<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, const array_view<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

/** @} */

/** @{ */
/**
 * The contents of a source container from the iterator range [srcBegin,srcEnd)
 * are copied into "dest". If the number of elements in the iterator range is
 * not equal to "dest.extent.size()", an exception is thrown.
 *
 * In the overloads which don't take an end-iterator it is assumed that the
 * source iterator is able to provide at least dest.extent.size() elements, but
 * no checking is performed (nor possible).
 *
 * @param[in] srcBegin An iterator to the first element of a source container.
 * @param[in] srcEnd An interator to the end of a source container.
 * @param[out] dest An object of type array<T,N> to be copied to.
 */
template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&, srcBegin, srcEnd]() mutable { copy(srcBegin, srcEnd, dest); });
    return completion_future(fut.share());
}

template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, array<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&, srcBegin]() mutable { copy(srcBegin, dest); });
    return completion_future(fut.share());
}

/** @} */

/** @{ */
/**
 * The contents of a source container from the iterator range [srcBegin,srcEnd)
 * are copied into "dest". If the number of elements in the iterator range is
 * not equal to "dest.extent.size()", an exception is thrown.
 *
 * In the overloads which don't take an end-iterator it is assumed that the
 * source iterator is able to provide at least dest.extent.size() elements, but
 * no checking is performed (nor possible).
 *
 * @param[in] srcBegin An iterator to the first element of a source container.
 * @param[in] srcEnd An interator to the end of a source container.
 * @param[out] dest An object of type array_view<T,N> to be copied to.
 */
template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&, srcBegin, srcEnd]() mutable { copy(srcBegin, srcEnd, dest); });
    return completion_future(fut.share());
}

template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, const array_view<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&, srcBegin]() mutable { copy(srcBegin, dest); });
    return completion_future(fut.share());
}

/** @} */

/**
 * The contents of a source array are copied into "dest" starting with iterator
 * destBegin. If the number of elements in the range starting destBegin in the
 * destination container is smaller than "src.extent.size()", the behavior is
 * undefined.
 *
 * @param[in] src An object of type array<T,N> to be copied from.
 * @param[out] destBegin An output iterator addressing the position of the
 *                       first element in the destination container.
 */
template <typename OutputIter, typename T, int N>
completion_future copy_async(const array<T, N>& src, OutputIter destBegin) {
    std::future<void> fut = std::async(std::launch::deferred, [&, destBegin]() mutable { copy(src, destBegin); });
    return completion_future(fut.share());
}

/**
 * The contents of a source array are copied into "dest" starting with iterator
 * destBegin. If the number of elements in the range starting destBegin in the
 * destination container is smaller than "src.extent.size()", the behavior is
 * undefined.
 *
 * @param[in] src An object of type array_view<T,N> to be copied from.
 * @param[out] destBegin An output iterator addressing the position of the
 *                       first element in the destination container.
 */
template <typename OutputIter, typename T, int N>
completion_future copy_async(const array_view<T, N>& src, OutputIter destBegin) {
    std::future<void> fut = std::async(std::launch::deferred, [&, destBegin]() mutable { copy(src, destBegin); });
    return completion_future(fut.share());
}

// FIXME: these functions are not defined in C++ AMP specification
template <typename T, int N>
completion_future copy_async(const array<T, N>& src, const array<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, const array<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, const array<T, N>& dest) {
    std::future<void> fut = std::async(std::launch::deferred, [&]() mutable { copy(src, dest); });
    return completion_future(fut.share());
}

// ------------------------------------------------------------------------
// parallel_for_each
// ------------------------------------------------------------------------

template<int N, typename Kernel>
inline
void parallel_for_each(const extent<N>& compute_domain, const Kernel& f)
{
    parallel_for_each(
        accelerator::get_auto_selection_view(), compute_domain, f);
}

template<typename Kernel, int... dims>
inline
void parallel_for_each(
    const tiled_extent<dims...>& compute_domain, const Kernel& f)
{
    parallel_for_each(
        accelerator::get_auto_selection_view(), compute_domain, f);
}

template<int n>
inline
void validate_compute_domain(const Concurrency::extent<n>& compute_domain)
{
    std::size_t sz{1};
    for (auto i = 0; i != n; ++i) {
        sz *= compute_domain[i];

        if (sz < 1) throw invalid_compute_domain{"Extent is not positive."};
        if (sz > UINT_MAX) throw invalid_compute_domain{"Extent is too large."};
    }
}

template<int N, typename Kernel>
inline
void parallel_for_each(
    const accelerator_view& av,
    const extent<N>& compute_domain,
    const Kernel& f)
{
    if (av.get_accelerator().get_device_path() == L"cpu") {
      throw runtime_exception{
          detail::__errorMsg_UnsupportedAccelerator, E_FAIL};
    }

    validate_compute_domain(compute_domain);

    detail::launch_kernel(av.pQueue, compute_domain, f);
}


// parallel_for_each, tiled
template<typename...>
inline
void validate_tile_dims()
{}

template<int dim, int... dims>
inline
void validate_tile_dims()
{
    static_assert(dim >= 0, "The number of threads in a tile must be positive.");
    static_assert(
        dim <= 1024, "The maximum number of threads in a tile is 1024.");

    validate_tile_dims<dims...>();
}

template<int... dims>
inline
void validate_tiled_compute_domain(const tiled_extent<dims...>& compute_domain)
{
    validate_tile_dims<dims...>();
    validate_compute_domain(compute_domain);

    constexpr int tmp[]{dims...};
    for (auto i = 0u; i != compute_domain.rank; ++i) {
        if (compute_domain[i] % tmp[i]) {
            throw invalid_compute_domain{"Extent not divisible by tile size."};
        }
    }
}

template <typename Kernel, int... dims>
inline
void parallel_for_each(
    const accelerator_view& av,
    const tiled_extent<dims...>& compute_domain,
    const Kernel& f)
{
    if (av.get_accelerator().get_device_path() == L"cpu") {
        throw runtime_exception{
            detail::__errorMsg_UnsupportedAccelerator, E_FAIL};
    }

    validate_tiled_compute_domain(compute_domain);

    detail::launch_kernel(av.pQueue, compute_domain, f);
}
} // namespace Concurrency
