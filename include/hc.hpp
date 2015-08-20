#pragma once

#include <kalmar_defines.h>
#include <kalmar_exception.h>
#include <kalmar_index.h>
#include <kalmar_runtime.h>
#include <kalmar_serialize.h>
#include <kalmar_launch.h>
#include <kalmar_buffer.h>
#include <kalmar_math.h>

#include <hsa_atomic.h>

#include <am.h>

namespace hc {

using namespace Kalmar::enums;

// forward declaration
class accelerator;
class accelerator_view;
class completion_future;
template <int N> class extent;
template <int N> class tiled_extent;
class ts_allocator;
template <typename T, int N> class array_view;
template <typename T, int N> class array;

// namespace alias
// namespace hc::fast_math is an alias of namespace Kalmar::fast_math
namespace fast_math = Kalmar::fast_math;

// namespace hc::precise_math is an alias of namespace Kalmar::precise_math
namespace precise_math = Kalmar::precise_math;

// type alias

/**
 * Represents a unique position in N-dimensional space.
 */
template <int N>
using index = Kalmar::index<N>;

using runtime_exception = Kalmar::runtime_exception;
using invalid_compute_domain = Kalmar::invalid_compute_domain;
using accelerator_view_removed = Kalmar::accelerator_view_removed;

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
    // FIXME: dummy implementation now
    unsigned int get_version() const { return 0; } 

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
     */
    int getPendingAsyncOps() {
        return pQueue->getPendingAsyncOps();
    }

    /**
     * Returns an opaque handle which points to the underlying HSA queue.
     *
     * @return An opaque handle of the underlying HSA queue, if the accelerator
     *         view is based on HSA.  NULL if otherwise.
     */
    void* getHSAQueue() {
        return pQueue->getHSAQueue();
    }

    /**
     * Returns if the accelerator view is based on HSA.
     */
    bool hasHSAInterOp() {
        return pQueue->hasHSAInterOp();
    }

private:
    accelerator_view(std::shared_ptr<Kalmar::KalmarQueue> pQueue) : pQueue(pQueue) {}
    std::shared_ptr<Kalmar::KalmarQueue> pQueue;

    friend class accelerator;
    template <typename Q, int K> friend class array;
    template <typename Q, int K> friend class array_view;
  
    template<typename Kernel> friend
        void* Kalmar::mcw_cxxamp_get_kernel(const std::shared_ptr<Kalmar::KalmarQueue>&, const Kernel&);
    template<typename Kernel, int dim_ext> friend
        void Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory(const std::shared_ptr<Kalmar::KalmarQueue>&, size_t *, size_t *, const Kernel&, void*, size_t);
    template<typename Kernel, int dim_ext> friend
        std::shared_ptr<Kalmar::KalmarAsyncOp> Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async(const std::shared_ptr<Kalmar::KalmarQueue>&, size_t *, size_t *, const Kernel&, void*, size_t);
    template<typename Kernel, int dim_ext> friend
        void Kalmar::mcw_cxxamp_launch_kernel(const std::shared_ptr<Kalmar::KalmarQueue>&, size_t *, size_t *, const Kernel&);
    template<typename Kernel, int dim_ext> friend
        std::shared_ptr<Kalmar::KalmarAsyncOp> Kalmar::mcw_cxxamp_launch_kernel_async(const std::shared_ptr<Kalmar::KalmarQueue>&, size_t *, size_t *, const Kernel&);
  
    // FIXME: enable CPU execution path for HC
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
    template <typename Kernel, int N> friend
        void Kalmar::launch_cpu_task(const std::shared_ptr<Kalmar::KalmarQueue>&, Kernel const&, extent<N> const&);
#endif

    // non-tiled parallel_for_each with dynamic group segment
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<1>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<2>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<3>&, ts_allocator&, const Kernel&);
  
    // non-tiled parallel_for_each with dynamic group segment
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<1>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<2>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<3>&, ts_allocator&, const Kernel&);
  
    // tiled parallel_for_each with dynamic group segment
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, ts_allocator&, const Kernel&);
  
    // tiled parallel_for_each with dynamic group segment
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<1>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<2>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<3>&, ts_allocator&, const Kernel&);
  
    // non-tiled parallel_for_each
    // generic version
    template <int N, typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<N>&, const Kernel&);
  
    // 1D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<1>&, const Kernel&);
  
    // 2D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<2>&, const Kernel&);
  
    // 3D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<3>&, const Kernel&);
  
    // tiled parallel_for_each, 3D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, const Kernel&);
  
    // tiled parallel_for_each, 2D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, const Kernel&);
  
    // tiled parallel_for_each, 1D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, const Kernel&);

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
public:
#endif
    __attribute__((annotate("user_deserialize")))
    accelerator_view() restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
        throw runtime_exception("errorMsg_throw", 0);
#endif
    }
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
    static const wchar_t default_accelerator[];
    static const wchar_t cpu_accelerator[];

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
        std::vector<accelerator> ret(Devices.size());
        for (int i = 0; i < ret.size(); ++i)
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
    accelerator_view create_view(queuing_mode mode = queuing_mode_automatic) {
        auto pQueue = pDev->createQueue();
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
    // FIXME: dummy implementation now
    unsigned int get_version() const { return 0; }

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
  
private:
    accelerator(Kalmar::KalmarDevice* pDev) : pDev(pDev) {}
    friend class accelerator_view;
    Kalmar::KalmarDevice* pDev;
};

// FIXME: this will cause troubles later in separated compilation
const wchar_t accelerator::cpu_accelerator[] = L"cpu";
const wchar_t accelerator::default_accelerator[] = L"default";

// ------------------------------------------------------------------------
// completion_future
// ------------------------------------------------------------------------

class completion_future {
public:

    completion_future() : __amp_future(), __thread_then(nullptr), __asyncOp(nullptr) {};

    completion_future(const completion_future& _Other)
        : __amp_future(_Other.__amp_future), __thread_then(_Other.__thread_then), __asyncOp(_Other.__asyncOp) {}

    completion_future(completion_future&& _Other)
        : __amp_future(std::move(_Other.__amp_future)), __thread_then(_Other.__thread_then), __asyncOp(_Other.__asyncOp) {}

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

    completion_future& operator=(const completion_future& _Other) {
        if (this != &_Other) {
           __amp_future = _Other.__amp_future;
           __thread_then = _Other.__thread_then;
           __asyncOp = _Other.__asyncOp;
        }
        return (*this);
    }

    completion_future& operator=(completion_future&& _Other) {
        if (this != &_Other) {
            __amp_future = std::move(_Other.__amp_future);
            __thread_then = _Other.__thread_then;
           __asyncOp = _Other.__asyncOp;
        }
        return (*this);
    }

    void get() const {
        __amp_future.get();
    }

    bool valid() const {
        return __amp_future.valid();
    }
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

    operator std::shared_future<void>() const {
        return __amp_future;
    }

    // notice we removed const from the signature here
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

    void* getNativeHandle() {
      if (__asyncOp != nullptr) {
        return __asyncOp->getNativeHandle();
      } else {
        return nullptr;
      }
    }

private:
    std::shared_future<void> __amp_future;
    std::thread* __thread_then = nullptr;
    std::shared_ptr<Kalmar::KalmarAsyncOp> __asyncOp;

    completion_future(std::shared_ptr<Kalmar::KalmarAsyncOp> event) : __amp_future(*(event->getFuture())), __asyncOp(event) {}

    completion_future(const std::shared_future<void> &__future)
        : __amp_future(__future), __thread_then(nullptr), __asyncOp(nullptr) {}

    // non-tiled parallel_for_each with dynamic group segment
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<1>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<2>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<3>&, ts_allocator&, const Kernel&);
  
    // non-tiled parallel_for_each with dynamic group segment
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<1>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<2>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const extent<3>&, ts_allocator&, const Kernel&);
  
    // tiled parallel_for_each with dynamic group segment
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, ts_allocator&, const Kernel&);
    template<typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, ts_allocator&, const Kernel&);
  
    // tiled parallel_for_each with dynamic group segment
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<1>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<2>&, ts_allocator&, const Kernel&);
    template <typename Kernel> friend
        completion_future parallel_for_each(const tiled_extent<3>&, ts_allocator&, const Kernel&);

    // non-tiled parallel_for_each
    // generic version
    template <int N, typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<N>&, const Kernel&);

    // 1D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<1>&, const Kernel&);

    // 2D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<2>&, const Kernel&);

    // 3D specialization
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const extent<3>&, const Kernel&);

    // tiled parallel_for_each, 3D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, const Kernel&);

    // tiled parallel_for_each, 2D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, const Kernel&);

    // tiled parallel_for_each, 1D version
    template <typename Kernel> friend
        completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, const Kernel&);

    // implementation of copy_async
    template <typename InputIter, typename OutputIter>
        friend completion_future __amp_copy_async_impl(InputIter& src, OutputIter& dst);

    // copy_async
    template <typename InputIter, typename T, int N>
        friend completion_future copy_async(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest);
    template <typename InputIter, typename T, int N>
        friend completion_future copy_async(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest);
    template <typename InputIter, typename T, int N>
        friend completion_future copy_async(InputIter srcBegin, array<T, N>& dest);
    template <typename InputIter, typename T, int N>
        friend completion_future copy_async(InputIter srcBegin, const array_view<T, N>& dest);
    template <typename OutputIter, typename T, int N>
        friend completion_future copy_async(const array<T, N>& src, OutputIter destBegin);
    template <typename OutputIter, typename T, int N>
        friend completion_future copy_async(const array_view<T, N>& src, OutputIter destBegin);

    // array_view
    template <typename T, int N> friend class array_view;

    // accelerator_view
    friend class accelerator_view;
};


// ------------------------------------------------------------------------
// member function implementations
// ------------------------------------------------------------------------

inline accelerator accelerator_view::get_accelerator() const { return pQueue->getDev(); }

inline completion_future accelerator_view::create_marker() {
    return completion_future(pQueue->EnqueueMarker());
}

// ------------------------------------------------------------------------
// extent
// ------------------------------------------------------------------------

template <int N>
class extent {
public:
    static const int rank = N;
    typedef int value_type;

    extent() restrict(amp,cpu) : base_() {
      static_assert(N > 0, "Dimensionality must be positive");
    };
    extent(const extent& other) restrict(amp,cpu)
        : base_(other.base_) {}
    template <typename ..._Tp>
        explicit extent(_Tp ... __t) restrict(amp,cpu)
        : base_(__t...) {
      static_assert(sizeof...(__t) <= 3, "Can only supply at most 3 individual coordinates in the constructor");
      static_assert(sizeof...(__t) == N, "rank should be consistency");
    }
    explicit extent(int component) restrict(amp,cpu)
        : base_(component) {}
    explicit extent(int components[]) restrict(amp,cpu)
        : base_(components) {}
    explicit extent(const int components[]) restrict(amp,cpu)
        : base_(components) {}

    extent& operator=(const extent& other) restrict(amp,cpu) {
        base_.operator=(other.base_);
        return *this;
    }

    int operator[] (unsigned int c) const restrict(amp,cpu) {
        return base_[c];
    }
    int& operator[] (unsigned int c) restrict(amp,cpu) {
        return base_[c];
    }

    bool operator==(const extent& other) const restrict(amp,cpu) {
        return Kalmar::index_helper<N, extent<N> >::equal(*this, other);
    }
    bool operator!=(const extent& other) const restrict(amp,cpu) {
        return !(*this == other);
    }

    unsigned int size() const restrict(amp,cpu) {
        return Kalmar::index_helper<N, extent<N>>::count_size(*this);
    }
    bool contains(const index<N>& idx) const restrict(amp,cpu) {
        return Kalmar::amp_helper<N, index<N>, extent<N>>::contains(idx, *this);
    }
    tiled_extent<1> tile(int t0) const;
    tiled_extent<2> tile(int t0, int t1) const;
    tiled_extent<3> tile(int t0, int t1, int t2) const;

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
    extent& operator+=(const extent& __r) restrict(amp,cpu) {
        base_.operator+=(__r.base_);
        return *this;
    }
    extent& operator-=(const extent& __r) restrict(amp,cpu) {
        base_.operator-=(__r.base_);
        return *this;
    }
    extent& operator*=(const extent& __r) restrict(amp,cpu) {
        base_.operator*=(__r.base_);
        return *this;
    }
    extent& operator/=(const extent& __r) restrict(amp,cpu) {
        base_.operator/=(__r.base_);
        return *this;
    }
    extent& operator%=(const extent& __r) restrict(amp,cpu) {
        base_.operator%=(__r.base_);
        return *this;
    }
    extent& operator+=(int __r) restrict(amp,cpu) {
        base_.operator+=(__r);
        return *this;
    }
    extent& operator-=(int __r) restrict(amp,cpu) {
        base_.operator-=(__r);
        return *this;
    }
    extent& operator*=(int __r) restrict(amp,cpu) {
        base_.operator*=(__r);
        return *this;
    }
    extent& operator/=(int __r) restrict(amp,cpu) {
        base_.operator/=(__r);
        return *this;
    }
    extent& operator%=(int __r) restrict(amp,cpu) {
        base_.operator%=(__r);
        return *this;
    }
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
private:
    typedef Kalmar::index_impl<typename Kalmar::__make_indices<N>::type> base;
    base base_;
    template <int K, typename Q> friend struct Kalmar::index_helper;
    template <int K, typename Q1, typename Q2> friend struct Kalmar::amp_helper;
};

// ------------------------------------------------------------------------
// tiled_extent
// ------------------------------------------------------------------------

// tile extent supporting dynamic tile size
template <int N>
class tiled_extent : public extent<N> {
public:
  static const int rank = N;
  int tile_dim[N];
  tiled_extent() restrict(amp,cpu) : extent<N>(), tile_dim{0} {}
  tiled_extent(const tiled_extent& other) restrict(amp,cpu) : extent<N>(other) {
    for (int i = 0; i < N; ++i) {
      tile_dim[i] = other.tile_dim[i];
    }
  }
};

// tile extent supporting dynamic tile size
// 1D specialization
template <>
class tiled_extent<1> : public extent<1> {
public:
  static const int rank = 1;
  int tile_dim[1];
  tiled_extent() restrict(amp,cpu) : extent(0), tile_dim{0} {}
  tiled_extent(int e0, int t0) restrict(amp,cpu) : extent(e0), tile_dim{t0} {}
  tiled_extent(const tiled_extent<1>& other) restrict(amp,cpu) : extent(other[0]), tile_dim{other.tile_dim[0]} {}
  tiled_extent(const extent<1>& ext, int t0) restrict(amp,cpu) : extent(ext), tile_dim{t0} {} 
};

// tile extent supporting dynamic tile size
// 2D specialization
template <>
class tiled_extent<2> : public extent<2> {
public:
  static const int rank = 2;
  int tile_dim[2];
  tiled_extent() restrict(amp,cpu) : extent(0, 0), tile_dim{0, 0} {}
  tiled_extent(int e0, int e1, int t0, int t1) restrict(amp,cpu) : extent(e0, e1), tile_dim{t0, t1} {}
  tiled_extent(const tiled_extent<2>& other) restrict(amp,cpu) : extent(other[0], other[1]), tile_dim{other.tile_dim[0], other.tile_dim[1]} {}
  tiled_extent(const extent<2>& ext, int t0, int t1) restrict(amp,cpu) : extent(ext), tile_dim{t0, t1} {}
};

// tile extent supporting dynamic tile size
// 3D specialization
template <>
class tiled_extent<3> : public extent<3> {
public:
  static const int rank = 3;
  int tile_dim[3];
  tiled_extent() restrict(amp,cpu) : extent(0, 0, 0), tile_dim{0, 0, 0} {}
  tiled_extent(int e0, int e1, int e2, int t0, int t1, int t2) restrict(amp,cpu) : extent(e0, e1, e2), tile_dim{t0, t1, t2} {}
  tiled_extent(const tiled_extent<3>& other) restrict(amp,cpu) : extent(other[0], other[1], other[2]), tile_dim{other.tile_dim[0], other.tile_dim[1], other.tile_dim[2]} {}
  tiled_extent(const extent<3>& ext, int t0, int t1, int t2) restrict(amp,cpu) : extent(ext), tile_dim{t0, t1, t2} {}
};

// ------------------------------------------------------------------------
// implementation of extent<N>::tile()
// ------------------------------------------------------------------------

template <int N>
inline
tiled_extent<1> extent<N>::tile(int t0) const restrict(amp,cpu) {
  static_assert(N == 1, "One-dimensional tile() method only available on extent<1>");
  return tiled_extent<1>(*this, t0);
}

template <int N>
inline
tiled_extent<2> extent<N>::tile(int t0, int t1) const restrict(amp,cpu) {
  static_assert(N == 2, "Two-dimensional tile() method only available on extent<2>");
  return tiled_extent<2>(*this, t0, t1);
}

template <int N>
inline
tiled_extent<3> extent<N>::tile(int t0, int t1, int t2) const restrict(amp,cpu) {
  static_assert(N == 3, "Three-dimensional tile() method only available on extent<3>");
  return tiled_extent<3>(*this, t0, t1, t2);
}

// ------------------------------------------------------------------------
// ts_allocator
// ------------------------------------------------------------------------

/// getLDS : C interface of HSA builtin function to fetch an address within group segment
extern "C" __attribute__((address_space(3))) void* getLDS(unsigned int offset) restrict(amp);

class ts_allocator {
private:
  unsigned int static_group_segment_size;
  unsigned int dynamic_group_segment_size;
  int cursor;

  void setStaticGroupSegmentSize(unsigned int size) restrict(cpu) {
    static_group_segment_size = size;
  } 

  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const extent<1>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const extent<2>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const extent<3>&, ts_allocator&, const Kernel&);

  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, ts_allocator&, const Kernel&);
  template <typename Kernel> friend
    completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, ts_allocator&, const Kernel&);

public:
  ts_allocator() :
    static_group_segment_size(0), 
    dynamic_group_segment_size(0),
    cursor(0) {}

  ~ts_allocator() {}

  unsigned int getStaticGroupSegmentSize() restrict(amp,cpu) {
    return static_group_segment_size;
  }

  void setDynamicGroupSegmentSize(unsigned int size) restrict(cpu) {
    dynamic_group_segment_size = size;
  }

  unsigned int getDynamicGroupSegmentSize() restrict(amp,cpu) {
    return dynamic_group_segment_size;
  }

  void reset() restrict(amp,cpu) {
    cursor = 0;
  }

  // Allocate the requested size in tile static memory and return its pointer
  // returns NULL if the requested size can't be allocated
  // It requires all threads in a tile to hit the same ts_alloc call site at the
  // same time.
  // Only one instance of the tile static memory will be allocated per call site
  // and all threads within a tile will get the same tile static memory address.
  __attribute__((address_space(3))) void* alloc(unsigned int size) restrict(amp) {
    int offset = cursor;

    // only the first workitem in the workgroup moves the cursor
    if (amp_get_local_id(0) == 0 && amp_get_local_id(1) == 0 && amp_get_local_id(2) == 0) {
      cursor += size;
    }

    // fetch the beginning address of dynamic group segment
    __attribute__((address_space(3))) unsigned char* lds = (__attribute__((address_space(3))) unsigned char*) getLDS(static_group_segment_size);

    // return the address
    return lds + offset;
  }   
};  

// ------------------------------------------------------------------------
// utility class for tiled_barrier
// ------------------------------------------------------------------------

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
template <typename Ker, typename Ti>
void bar_wrapper(Ker *f, Ti *t)
{
    (*f)(*t);
}
struct barrier_t {
    std::unique_ptr<ucontext_t[]> ctx;
    int idx;
    barrier_t (int a) :
        ctx(new ucontext_t[a + 1]) {}
    template <typename Ti, typename Ker>
    void setctx(int x, char *stack, Ker& f, Ti* tidx, int S) {
        getcontext(&ctx[x]);
        ctx[x].uc_stack.ss_sp = stack;
        ctx[x].uc_stack.ss_size = S;
        ctx[x].uc_link = &ctx[x - 1];
        makecontext(&ctx[x], (void (*)(void))bar_wrapper<Ker, Ti>, 2, &f, tidx);
    }
    void swap(int a, int b) {
        swapcontext(&ctx[a], &ctx[b]);
    }
    void wait() {
        --idx;
        swapcontext(&ctx[idx + 1], &ctx[idx]);
    }
};
#endif

// ------------------------------------------------------------------------
// tiled_barrier
// ------------------------------------------------------------------------

#ifndef CLK_LOCAL_MEM_FENCE
#define CLK_LOCAL_MEM_FENCE (1)
#endif

#ifndef CLK_GLOBAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE (2)
#endif

class tile_barrier {
 public:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  using pb_t = std::shared_ptr<barrier_t>;
  tile_barrier(pb_t pb) : pbar(pb) {}
  tile_barrier(const tile_barrier& other) restrict(amp,cpu) : pbar(other.pbar) {}
#else
  tile_barrier(const tile_barrier& other) restrict(amp,cpu) {}
#endif
  void wait() const restrict(amp) {
#if __KALMAR_ACCELERATOR__ == 1
    wait_with_all_memory_fence();
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      pbar->wait();
#endif
  }
  void wait_with_all_memory_fence() const restrict(amp) {
#if __KALMAR_ACCELERATOR__ == 1
    amp_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      pbar->wait();
#endif
  }
  void wait_with_global_memory_fence() const restrict(amp) {
#if __KALMAR_ACCELERATOR__ == 1
    amp_barrier(CLK_GLOBAL_MEM_FENCE);
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      pbar->wait();
#endif
  }
  void wait_with_tile_static_memory_fence() const restrict(amp) {
#if __KALMAR_ACCELERATOR__ == 1
    amp_barrier(CLK_LOCAL_MEM_FENCE);
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      pbar->wait();
#endif
  }
 private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  tile_barrier() restrict(amp,cpu) = default;
  pb_t pbar;
#else
  tile_barrier() restrict(amp) {}
#endif
  template <int N> friend
    class tiled_index;

  friend class tiled_index_1D;
  friend class tiled_index_2D;
  friend class tiled_index_3D;
};

// ------------------------------------------------------------------------
// tiled_index
// ------------------------------------------------------------------------

template <int N=3>
class tiled_index {
public:
  const index<3> global;
  const index<3> local;
  const index<3> tile;
  const index<3> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<3>& g) restrict(amp,cpu) : global(g) {}
  tiled_index(const tiled_index& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<3>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index(int a0, int a1, int a2, int b0, int b1, int b2, int c0, int c1, int c2, tile_barrier& pb) restrict(amp,cpu) :
    global(a2, a1, a0), local(b2, b1, b0), tile(c2, c1, c0), tile_origin(a2 - b2, a1 - b1, a0 - b0), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index() restrict(amp) :
    global(index<3>(amp_get_global_id(2), amp_get_global_id(1), amp_get_global_id(0))),
    local(index<3>(amp_get_local_id(2), amp_get_local_id(1), amp_get_local_id(0))),
    tile(index<3>(amp_get_group_id(2), amp_get_group_id(1), amp_get_group_id(0))),
    tile_origin(index<3>(amp_get_global_id(2) - amp_get_local_id(2),
                         amp_get_global_id(1) - amp_get_local_id(1),
                         amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<N>&, ts_allocator&, const Kernel&);

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<N>&, const Kernel&);
};

template<>
class tiled_index<1> {
public:
  const index<1> global;
  const index<1> local;
  const index<1> tile;
  const index<1> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<1>& g) restrict(amp,cpu) : global(g) {}
  tiled_index(const tiled_index& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<1>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index(int a, int b, int c, tile_barrier& pb) restrict(amp,cpu) :
    global(a), local(b), tile(c), tile_origin(a - b), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index() restrict(amp) :
    global(index<1>(amp_get_global_id(0))),
    local(index<1>(amp_get_local_id(0))),
    tile(index<1>(amp_get_group_id(0))),
    tile_origin(index<1>(amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, ts_allocator&, const Kernel&);

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, const Kernel&);
};

template<>
class tiled_index<2> {
public:
  const index<2> global;
  const index<2> local;
  const index<2> tile;
  const index<2> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<2>& g) restrict(amp,cpu) : global(g) {}
  tiled_index(const tiled_index& other) restrict(amp,cpu) : global(other.global), local(other.local), tile(other.tile), tile_origin(other.tile_origin), barrier(other.barrier) {}
  operator const index<2>() const restrict(amp,cpu) {
    return global;
  }
private:
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index(int a0, int a1, int b0, int b1, int c0, int c1, tile_barrier& pb) restrict(amp,cpu) :
    global(a1, a0), local(b1, b0), tile(c1, c0), tile_origin(a1 - b1, a0 - b0), barrier(pb) {}
#endif

  __attribute__((annotate("__cxxamp_opencl_index")))
#if __KALMAR_ACCELERATOR__ == 1
  __attribute__((always_inline)) tiled_index() restrict(amp) :
    global(index<2>(amp_get_global_id(1), amp_get_global_id(0))),
    local(index<2>(amp_get_local_id(1), amp_get_local_id(0))),
    tile(index<2>(amp_get_group_id(1), amp_get_group_id(0))),
    tile_origin(index<2>(amp_get_global_id(1) - amp_get_local_id(1),
                         amp_get_global_id(0) - amp_get_local_id(0)))
#elif __KALMAR__ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  __attribute__((always_inline)) tiled_index() restrict(amp,cpu)
#else
  __attribute__((always_inline)) tiled_index() restrict(amp)
#endif // __KALMAR_ACCELERATOR__
  {}

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, ts_allocator&, const Kernel&);

  template<typename Kernel> friend
      completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, const Kernel&);
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
        extent<N - 1> ext_now(ext_o);
        extent<N - 1> ext_base(ext);
        index<N - 1> idx_base(idx);
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
        extent<N - 1> ext_now(ext_o);
        extent<N - 1> ext_base(ext);
        index<N - 1> idx_base(idx);
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
        extent<N - 1> ext_now(ext_o);
        extent<N - 1> ext_base(ext);
        index<N - 1> idx_base(idx);
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
        extent<N - 1> ext_now(ext_o);
        extent<N - 1> ext_base(ext);
        index<N - 1> idx_base(idx);
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
// utility helper classes for array_view
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
        extent<N - 1> ext(comp);
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
        extent<N - 1> ext(comp);
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
const extent<N>& check(const extent<N>& ext)
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

template <typename T, int N = 1>
class array {
  static_assert(!std::is_const<T>::value, "array<const T> is not supported");
  static_assert(0 == (sizeof(T) % sizeof(int)), "only value types whose size is a multiple of the size of an integer are allowed in array");
public:
#if __KALMAR_ACCELERATOR__ == 1
  typedef Kalmar::_data<T> acc_buffer_t;
#else
  typedef Kalmar::_data_host<T> acc_buffer_t;
#endif

  static const int rank = N;
  typedef T value_type;
  array() = delete;

  array(const extent<N>& ext, accelerator_view av, accelerator_view associated_av)
#if __KALMAR_ACCELERATOR__ == 1
      : m_device(ext.size()), extent(ext) {}
#else
      : m_device(av.pQueue, associated_av.pQueue, check(ext).size(), access_type_auto), extent(ext) {}
#endif
  array(int e0, accelerator_view av, accelerator_view associated_av)
      : array(hc::extent<N>(e0), av, associated_av) {}
  array(int e0, int e1, accelerator_view av, accelerator_view associated_av)
      : array(hc::extent<N>(e0, e1), av, associated_av) {}
  array(int e0, int e1, int e2, accelerator_view av, accelerator_view associated_av)
      : array(hc::extent<N>(e0, e1, e2), av, associated_av) {}

  array(const extent<N>& ext, accelerator_view av,
        access_type cpu_access_type = access_type_auto)
#if __KALMAR_ACCELERATOR__ == 1
      : m_device(ext.size()), extent(ext) {}
#else
      : m_device(av.pQueue, av.pQueue, check(ext).size(), cpu_access_type), extent(ext) {}
#endif
  array(int e0, accelerator_view av,
        access_type cpu_access_type = access_type_auto)
      : array(hc::extent<N>(e0), av, cpu_access_type) {}
  array(int e0, int e1, accelerator_view av,
        access_type cpu_access_type = access_type_auto)
      : array(hc::extent<N>(e0, e1), av, cpu_access_type) {}
  array(int e0, int e1, int e2, accelerator_view av,
        access_type cpu_access_type = access_type_auto)
      : array(hc::extent<N>(e0, e1, e2), av, cpu_access_type) {}


  explicit array(const extent<N>& ext) :
      array(ext, accelerator(L"default").get_default_view()) {}
  explicit array(int e0) : array(hc::extent<N>(e0)) { static_assert(N == 1, "illegal"); }
  explicit array(int e0, int e1) : array(hc::extent<N>(e0, e1)) {}
  explicit array(int e0, int e1, int e2) : array(hc::extent<N>(e0, e1, e2)) {}

  template <typename InputIter>
      array(const extent<N>& ext, InputIter srcBegin,
            accelerator_view av, accelerator_view associated_av)
      : array(ext, av, associated_av) { copy(srcBegin, *this); }
  template <typename InputIter>
      array(const extent<N>& ext, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, accelerator_view associated_av)
      : array(ext, av, associated_av) {
          if(ext.size() < std::distance(srcBegin, srcEnd))
              throw runtime_exception("errorMsg_throw", 0);
          copy(srcBegin, srcEnd, *this);
      }
  template <typename InputIter>
      array(int e0, InputIter srcBegin,
            accelerator_view av, accelerator_view associated_av)
      : array(extent<N>(e0), srcBegin, av, associated_av) {}
  template <typename InputIter>
      array(int e0, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, accelerator_view associated_av)
      : array(extent<N>(e0), srcBegin, srcEnd, av, associated_av) {}
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin,
            accelerator_view av, accelerator_view associated_av)
      : array(hc::extent<N>(e0, e1), srcBegin, av, associated_av) {}
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, accelerator_view associated_av)
      : array(hc::extent<N>(e0, e1), srcBegin, srcEnd, av, associated_av) {}
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin,
            accelerator_view av, accelerator_view associated_av)
      : array(hc::extent<N>(e0, e1, e2), srcBegin, av, associated_av) {}
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, accelerator_view associated_av)
      : array(hc::extent<N>(e0, e1, e2), srcBegin, srcEnd, av, associated_av) {}

  template <typename InputIter>
      array(const extent<N>& ext, InputIter srcBegin, accelerator_view av,
            access_type cpu_access_type = access_type_auto)
      : array(ext, av, cpu_access_type) { copy(srcBegin, *this); }
  template <typename InputIter>
      array(const extent<N>& ext, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, access_type cpu_access_type = access_type_auto)
      : array(ext, av, cpu_access_type) {
          if(ext.size() < std::distance(srcBegin, srcEnd))
              throw runtime_exception("errorMsg_throw", 0);
          copy(srcBegin, srcEnd, *this);
      }
  template <typename InputIter>
      array(int e0, InputIter srcBegin, accelerator_view av,
            access_type cpu_access_type = access_type_auto)
      : array(extent<N>(e0), srcBegin, av, cpu_access_type) {}
  template <typename InputIter>
      array(int e0, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, access_type cpu_access_type = access_type_auto)
      : array(extent<N>(e0), srcBegin, srcEnd, av, cpu_access_type) {}
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin, accelerator_view av,
            access_type cpu_access_type = access_type_auto)
      : array(hc::extent<N>(e0, e1), srcBegin, av, cpu_access_type) {}
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, access_type cpu_access_type = access_type_auto)
      : array(hc::extent<N>(e0, e1), srcBegin, srcEnd, av, cpu_access_type) {}
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin, accelerator_view av,
            access_type cpu_access_type = access_type_auto)
      : array(hc::extent<N>(e0, e1, e2), srcBegin, av, cpu_access_type) {}
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, access_type cpu_access_type = access_type_auto)
      : array(hc::extent<N>(e0, e1, e2), srcBegin, srcEnd, av, cpu_access_type) {}

  template <typename InputIter>
      array(const extent<N>& ext, InputIter srcBegin)
      : array(ext, srcBegin, accelerator(L"default").get_default_view()) {}
  template <typename InputIter>
      array(const extent<N>& ext, InputIter srcBegin, InputIter srcEnd)
      : array(ext, srcBegin, srcEnd, accelerator(L"default").get_default_view()) {}
  template <typename InputIter>
      array(int e0, InputIter srcBegin)
      : array(extent<N>(e0), srcBegin) {}
  template <typename InputIter>
      array(int e0, InputIter srcBegin, InputIter srcEnd)
      : array(extent<N>(e0), srcBegin, srcEnd) {}
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin)
      : array(hc::extent<N>(e0, e1), srcBegin) {}
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin, InputIter srcEnd)
      : array(hc::extent<N>(e0, e1), srcBegin, srcEnd) {}
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin)
      : array(hc::extent<N>(e0, e1, e2), srcBegin) {}
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin, InputIter srcEnd)
      : array(hc::extent<N>(e0, e1, e2), srcBegin, srcEnd) {}


  explicit array(const array_view<const T, N>& src)
      : array(src.get_extent(), accelerator(L"default").get_default_view())
  { copy(src, *this); }

  array(const array_view<const T, N>& src, accelerator_view av,
        access_type cpu_access_type = access_type_auto)
      : array(src.get_extent(), av, cpu_access_type) { copy(src, *this); }
  array(const array_view<const T, N>& src, accelerator_view av,
        accelerator_view associated_av) : array(src.get_extent(), av, associated_av)
    { copy(src, *this); }


  array(const array& other) : array(other.get_extent(), other.get_accelerator_view())
    { copy(other, *this); }
  array(array&& other) : m_device(other.m_device), extent(other.extent)
    { other.m_device.reset(); }

  array& operator=(const array& other) {
    if(this != &other) {
        array arr(other);
        *this = std::move(arr);
    }
    return *this;
  }
  array& operator=(array&& other) {
    if(this != &other) {
      extent = other.extent;
      m_device = other.m_device;
      other.m_device.reset();
    }
    return *this;
  }
  array& operator=(const array_view<T,N>& src) {
      array arr(src);
      *this = std::move(arr);
      return *this;
  }

  void copy_to(array& dest) const {
#if __KALMAR_ACCELERATOR__ != 1
      for(int i = 0 ; i < N ; i++)
      {
        if(dest.extent[i] < this->extent[i] )
          throw runtime_exception("errorMsg_throw", 0);
      }
#endif
      copy(*this, dest);
  }

  void copy_to(const array_view<T,N>& dest) const { copy(*this, dest); }
  extent<N> get_extent() const restrict(amp,cpu) { return extent; }

  accelerator_view get_accelerator_view() const { return m_device.get_av(); }
  accelerator_view get_associated_accelerator_view() const { return m_device.get_stage(); }
  access_type get_cpu_access_type() const { return m_device.get_access(); }

  T& operator[](const index<N>& idx) restrict(amp,cpu) {
#ifndef __KALMAR_ACCELERATOR__
      if (!m_device.get())
          throw runtime_exception("The array is not accessible on CPU.", 0);
      m_device.synchronize(true);
#endif
      T *ptr = reinterpret_cast<T*>(m_device.get());
      return ptr[Kalmar::amp_helper<N, index<N>, hc::extent<N>>::flatten(idx, extent)];
  }
  const T& operator[](const index<N>& idx) const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      if (!m_device.get())
          throw runtime_exception("The array is not accessible on CPU.", 0);
      m_device.synchronize();
#endif
      T *ptr = reinterpret_cast<T*>(m_device.get());
      return ptr[Kalmar::amp_helper<N, index<N>, hc::extent<N>>::flatten(idx, extent)];
  }

  typename array_projection_helper<T, N>::result_type
      operator[] (int i) restrict(amp,cpu) {
          return array_projection_helper<T, N>::project(*this, i);
      }
  typename array_projection_helper<T, N>::const_result_type
      operator[] (int i) const restrict(amp,cpu) {
          return array_projection_helper<T, N>::project(*this, i);
      }

  T& operator()(const index<N>& idx) restrict(amp,cpu) {
    return (*this)[idx];
  }
  const T& operator()(const index<N>& idx) const restrict(amp,cpu) {
    return (*this)[idx];
  }
  typename array_projection_helper<T, N>::result_type
      operator()(int i0) restrict(amp,cpu) {
          return (*this)[i0];
  }
  typename array_projection_helper<T, N>::const_result_type
      operator()(int i0) const restrict(amp,cpu) {
          return (*this)[i0];
  }
  T& operator()(int i0, int i1) restrict(amp,cpu) {
      return (*this)[index<2>(i0, i1)];
  }
  const T& operator()(int i0, int i1) const restrict(amp,cpu) {
      return (*this)[index<2>(i0, i1)];
  }
  T& operator()(int i0, int i1, int i2) restrict(amp,cpu) {
      return (*this)[index<3>(i0, i1, i2)];
  }
  const T& operator()(int i0, int i1, int i2) const restrict(amp,cpu) {
      return (*this)[index<3>(i0, i1, i2)];
  }

  array_view<T, N> section(const index<N>& idx, const extent<N>& ext) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      if(  !Kalmar::amp_helper<N, index<N>, hc::extent<N>>::contains(idx,  ext ,this->extent) )
        throw runtime_exception("errorMsg_throw", 0);
#endif
      array_view<T, N> av(*this);
      return av.section(idx, ext);
  }
  array_view<const T, N> section(const index<N>& idx, const extent<N>& ext) const restrict(amp,cpu) {
      array_view<const T, N> av(*this);
      return av.section(idx, ext);
  }
  array_view<T, N> section(const index<N>& idx) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      if(  !Kalmar::amp_helper<N, index<N>, hc::extent<N>>::contains(idx, this->extent ) )
        throw runtime_exception("errorMsg_throw", 0);
#endif
      array_view<T, N> av(*this);
      return av.section(idx);
  }
  array_view<const T, N> section(const index<N>& idx) const restrict(amp,cpu) {
      array_view<const T, N> av(*this);
      return av.section(idx);
  }
  array_view<T,N> section(const extent<N>& ext) restrict(amp,cpu) {
      array_view<T, N> av(*this);
      return av.section(ext);
  }
  array_view<const T,N> section(const extent<N>& ext) const restrict(amp,cpu) {
      array_view<const T, N> av(*this);
      return av.section(ext);
  }

  array_view<T, 1> section(int i0, int e0) restrict(amp,cpu) {
      static_assert(N == 1, "Rank must be 1");
      return section(index<1>(i0), hc::extent<1>(e0));
  }
  array_view<const T, 1> section(int i0, int e0) const restrict(amp,cpu) {
      static_assert(N == 1, "Rank must be 1");
      return section(index<1>(i0), hc::extent<1>(e0));
  }
  array_view<T, 2> section(int i0, int i1, int e0, int e1) const restrict(amp,cpu) {
      static_assert(N == 2, "Rank must be 2");
      return section(index<2>(i0, i1), hc::extent<2>(e0, e1));
  }
  array_view<T, 2> section(int i0, int i1, int e0, int e1) restrict(amp,cpu) {
      static_assert(N == 2, "Rank must be 2");
      return section(index<2>(i0, i1), hc::extent<2>(e0, e1));
  }
  array_view<T, 3> section(int i0, int i1, int i2, int e0, int e1, int e2) restrict(amp,cpu) {
      static_assert(N == 3, "Rank must be 3");
      return section(index<3>(i0, i1, i2), hc::extent<3>(e0, e1, e2));
  }
  array_view<const T, 3> section(int i0, int i1, int i2, int e0, int e1, int e2) const restrict(amp,cpu) {
      static_assert(N == 3, "Rank must be 3");
      return section(index<3>(i0, i1, i2), hc::extent<3>(e0, e1, e2));
  }

  template <typename ElementType>
      array_view<ElementType, 1> reinterpret_as() restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
          if( (extent.size() * sizeof(T)) % sizeof(ElementType))
              throw runtime_exception("errorMsg_throw", 0);
#endif
          int size = extent.size() * sizeof(T) / sizeof(ElementType);
          using buffer_type = typename array_view<ElementType, 1>::acc_buffer_t;
          array_view<ElementType, 1> av(buffer_type(m_device), extent<1>(size), 0);
          return av;
      }
  template <typename ElementType>
      array_view<const ElementType, 1> reinterpret_as() const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
#endif
          int size = extent.size() * sizeof(T) / sizeof(ElementType);
          using buffer_type = typename array_view<ElementType, 1>::acc_buffer_t;
          array_view<const ElementType, 1> av(buffer_type(m_device), extent<1>(size), 0);
          return av;
      }

  template <int K> array_view<T, K>
      view_as(const extent<K>& viewExtent) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
    if( viewExtent.size() > extent.size())
      throw runtime_exception("errorMsg_throw", 0);
#endif
          array_view<T, K> av(m_device, viewExtent, 0);
          return av;
      }
  template <int K> array_view<const T, K>
      view_as(const extent<K>& viewExtent) const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
    if( viewExtent.size() > extent.size())
      throw runtime_exception("errorMsg_throw", 0);
#endif
          const array_view<T, K> av(m_device, viewExtent, 0);
          return av;
      }

  operator std::vector<T>() const {
      std::vector<T> vec(extent.size());
      copy(*this, vec.data());
      return std::move(vec);
  }

  T* data() const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      if (!m_device.get())
          return nullptr;
    m_device.synchronize(true);
#endif
    return reinterpret_cast<T*>(m_device.get());
  }
  ~array() {}


  const acc_buffer_t& internal() const restrict(amp,cpu) { return m_device; }
  int get_offset() const restrict(amp,cpu) { return 0; }
  index<N> get_index_base() const restrict(amp,cpu) { return index<N>(); }
private:
  template <typename K, int Q> friend struct projection_helper;
  template <typename K, int Q> friend struct array_projection_helper;
  acc_buffer_t m_device;
  extent<N> extent;
};

// ------------------------------------------------------------------------
// array_view
// ------------------------------------------------------------------------

template <typename T, int N = 1>
class array_view
{
  static_assert(0 == (sizeof(T) % sizeof(int)), "only value types whose size is a multiple of the size of an integer are allowed in array views");
public:
  typedef typename std::remove_const<T>::type nc_T;
#if __KALMAR_ACCELERATOR__ == 1
  typedef Kalmar::_data<T> acc_buffer_t;
#else
  typedef Kalmar::_data_host<T> acc_buffer_t;
#endif

  static const int rank = N;
  typedef T value_type;
  array_view() = delete;

  ~array_view() restrict(amp,cpu) {}

  array_view(array<T, N>& src) restrict(amp,cpu)
      : cache(src.internal()), extent(src.get_extent()), extent_base(extent), index_base(),
      offset(0) {}

  template <typename Container, class = typename std::enable_if<__is_container<Container>::value>::type>
      array_view(const extent<N>& extent, Container& src)
      : array_view(extent, src.data())
  { static_assert( std::is_same<decltype(src.data()), T*>::value, "container element type and array view element type must match"); }
  template <typename Container, class = typename std::enable_if<__is_container<Container>::value>::type>
      array_view(int e0, Container& src)
      : array_view(hc::extent<N>(e0), src) {}
  template <typename Container, class = typename std::enable_if<__is_container<Container>::value>::type>
      array_view(int e0, int e1, Container& src)
      : array_view(hc::extent<N>(e0, e1), src) {}
  template <typename Container, class = typename std::enable_if<__is_container<Container>::value>::type>
      array_view(int e0, int e1, int e2, Container& src)
      : array_view(hc::extent<N>(e0, e1, e2), src) {}

  array_view(const extent<N>& ext, value_type* src) restrict(amp,cpu)
#if __KALMAR_ACCELERATOR__ == 1
      : cache((T *)(src)), extent(ext), extent_base(ext), offset(0) {}
#else
      : cache(ext.size(), (T *)(src)), extent(ext), extent_base(ext), offset(0) {}
#endif
  array_view(int e0, value_type *src) restrict(amp,cpu)
      : array_view(hc::extent<N>(e0), src) {}
  array_view(int e0, int e1, value_type *src) restrict(amp,cpu)
      : array_view(hc::extent<N>(e0, e1), src) {}
  array_view(int e0, int e1, int e2, value_type *src) restrict(amp,cpu)
      : array_view(hc::extent<N>(e0, e1, e2), src) {}


  explicit array_view(const extent<N>& ext)
  : cache(ext.size()), extent(ext), extent_base(ext), offset(0) {}
  explicit array_view(int e0) : array_view(hc::extent<N>(e0)) {}
  explicit array_view(int e0, int e1)
      : array_view(hc::extent<N>(e0, e1)) {}
  explicit array_view(int e0, int e1, int e2)
      : array_view(hc::extent<N>(e0, e1, e2)) {}

  array_view(const array_view& other) restrict(amp,cpu) : cache(other.cache),
    extent(other.extent), extent_base(other.extent_base), index_base(other.index_base),
    offset(other.offset) {}
  array_view& operator=(const array_view& other) restrict(amp,cpu) {
      if (this != &other) {
          cache = other.cache;
          extent = other.extent;
          index_base = other.index_base;
          extent_base = other.extent_base;
          offset = other.offset;
      }
      return *this;
  }
  void copy_to(array<T,N>& dest) const {
#if __KALMAR_ACCELERATOR__ != 1
      for(int i= 0 ;i< N;i++)
      {
        if(dest.get_extent()[i] < this->extent[i])
          throw runtime_exception("errorMsg_throw", 0);
      }
#endif
      copy(*this, dest);
  }
  void copy_to(const array_view& dest) const { copy(*this, dest); }
  extent<N> get_extent() const restrict(amp,cpu) { return extent; }

  T& operator[](const index<N>& idx) const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      cache.get_cpu_access(true);
#endif
      T *ptr = reinterpret_cast<T*>(cache.get() + offset);
      return ptr[Kalmar::amp_helper<N, index<N>, hc::extent<N>>::flatten(idx + index_base, extent_base)];
  }

/* FIXME: implement this with hc tiled_index */
/*
  template <int D0, int D1=0, int D2=0>
  T& operator[](const tiled_index<D0, D1, D2>& idx) const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      cache.get_cpu_access(true);
#endif
      T *ptr = reinterpret_cast<T*>(cache.get() + offset);
      return ptr[Kalmar::amp_helper<N, index<N>, hc::extent<N>>::flatten(idx.global + index_base, extent_base)];
  }
*/

  typename projection_helper<T, N>::result_type
      operator[] (int i) const restrict(amp,cpu) {
          return projection_helper<T, N>::project(*this, i);
      }
  T& operator()(const index<N>& idx) const restrict(amp,cpu) {
      return (*this)[idx];
  }
  typename projection_helper<T, N>::result_type
      operator()(int i0) const restrict(amp,cpu) { return (*this)[i0]; }
  T& operator()(int i0, int i1) const restrict(amp,cpu) {
      static_assert(N == 2, "T& array_view::operator()(int,int) is only permissible on array_view<T, 2>");
      return (*this)[index<2>(i0, i1)];
  }
  T& operator()(int i0, int i1, int i2) const restrict(amp,cpu) {
      static_assert(N == 3, "T& array_view::operator()(int,int, int) is only permissible on array_view<T, 3>");
      return (*this)[index<3>(i0, i1, i2)];
  }
  template <typename ElementType>
      array_view<ElementType, N> reinterpret_as() const restrict(amp,cpu) {
          static_assert(N == 1, "reinterpret_as is only permissible on array views of rank 1");
#if __KALMAR_ACCELERATOR__ != 1
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
          if( (extent.size() * sizeof(T)) % sizeof(ElementType))
              throw runtime_exception("errorMsg_throw", 0);
#endif
          int size = extent.size() * sizeof(T) / sizeof(ElementType);
          using buffer_type = typename array_view<ElementType, 1>::acc_buffer_t;
          array_view<ElementType, 1> av(buffer_type(cache),
                                        extent<1>(size),
                                        (offset + index_base[0])* sizeof(T) / sizeof(ElementType));
          return av;
      }

  array_view<T, N> section(const index<N>& idx,
                           const extent<N>& ext) const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      if(  !Kalmar::amp_helper<N, index<N>, hc::extent<N>>::contains(idx, ext,this->extent ) )
        throw runtime_exception("errorMsg_throw", 0);
#endif
      array_view<T, N> av(cache, ext, extent_base, idx + index_base, offset);
      return av;
  }
  array_view<T, N> section(const index<N>& idx) const restrict(amp,cpu) {
      hc::extent<N> ext(extent);
      Kalmar::amp_helper<N, index<N>, hc::extent<N>>::minus(idx, ext);
      return section(idx, ext);
  }
  array_view<T, N> section(const extent<N>& ext) const restrict(amp,cpu) {
      index<N> idx;
      return section(idx, ext);
  }
  array_view<T, 1> section(int i0, int e0) const restrict(amp,cpu) {
      static_assert(N == 1, "Rank must be 1");
      return section(index<1>(i0), extent<1>(e0));
  }
  array_view<T, 2> section(int i0, int i1, int e0, int e1) const restrict(amp,cpu) {
      static_assert(N == 2, "Rank must be 2");
      return section(index<2>(i0, i1), hc::extent<2>(e0, e1));
  }
  array_view<T, 3> section(int i0, int i1, int i2, int e0, int e1, int e2) const restrict(amp,cpu) {
      static_assert(N == 3, "Rank must be 3");
      return section(index<3>(i0, i1, i2), hc::extent<3>(e0, e1, e2));
  }
  template <int K>
  array_view<T, K> view_as(extent<K> viewExtent) const restrict(amp,cpu) {
    static_assert(N == 1, "view_as is only permissible on array views of rank 1");
#if __KALMAR_ACCELERATOR__ != 1
    if( viewExtent.size() > extent.size())
      throw runtime_exception("errorMsg_throw", 0);
#endif
    array_view<T, K> av(cache, viewExtent, offset + index_base[0]);
    return av;
  }

  void synchronize() const { cache.get_cpu_access(); }
  void synchronize_to(const accelerator_view& av) const {
#if __KALMAR_ACCELERATOR__ != 1
      cache.sync_to(av.pQueue);
#endif
  }
  completion_future synchronize_async() const {
      std::future<void> fut = std::async([&]() mutable { synchronize(); });
      return completion_future(fut.share());
  }
  void refresh() const { cache.refresh(); }
  void discard_data() const {
#if __KALMAR_ACCELERATOR__ != 1
      cache.discard();
#endif
  }
  T* data() const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      cache.get_cpu_access(true);
#endif
    static_assert(N == 1, "data() is only permissible on array views of rank 1");
    return reinterpret_cast<T*>(cache.get() + offset + index_base[0]);
  }

  accelerator_view get_source_accelerator_view() const { return cache.get_av(); }

  const acc_buffer_t& internal() const restrict(amp,cpu) { return cache; }
  int get_offset() const restrict(amp,cpu) { return offset; }
  index<N> get_index_base() const restrict(amp,cpu) { return index_base; }
private:
  template <typename K, int Q> friend struct projection_helper;
  template <typename K, int Q> friend struct array_projection_helper;
  template <typename Q, int K> friend class array;
  template <typename Q, int K> friend class array_view;

  template<typename Q, int K>
      friend bool is_flat(const array_view<Q, K>&) noexcept;
  template <typename Q, int K>
      friend void copy(const array<Q, K>&, const array_view<Q, K>&);
  template <typename InputIter, typename Q, int K>
      friend void copy(InputIter, InputIter, const array_view<Q, K>&);
  template <typename Q, int K>
      friend void copy(const array_view<const Q, K>&, array<Q, K>&);
  template <typename OutputIter, typename Q, int K>
      friend void copy(const array_view<Q, K>&, OutputIter);
  template <typename Q, int K>
      friend void copy(const array_view<const Q, K>& src, const array_view<Q, K>& dest);


  // used by view_as and reinterpret_as
  array_view(const acc_buffer_t& cache, const hc::extent<N>& ext,
             int offset) restrict(amp,cpu)
      : cache(cache), extent(ext), extent_base(ext), offset(offset) {}
  // used by section and projection
  array_view(const acc_buffer_t& cache, const hc::extent<N>& ext_now,
             const hc::extent<N>& ext_b,
             const index<N>& idx_b, int off) restrict(amp,cpu)
      : cache(cache), extent(ext_now), extent_base(ext_b), index_base(idx_b),
      offset(off) {}

  acc_buffer_t cache;
  hc::extent<N> extent;
  hc::extent<N> extent_base;
  index<N> index_base;
  int offset;
};

// ------------------------------------------------------------------------
// array_view (read-only)
// ------------------------------------------------------------------------

template <typename T, int N>
class array_view<const T, N>
{
public:
  typedef typename std::remove_const<T>::type nc_T;
  static const int rank = N;
  typedef const T value_type;

#if __KALMAR_ACCELERATOR__ == 1
  typedef Kalmar::_data<nc_T> acc_buffer_t;
#else
  typedef Kalmar::_data_host<const T> acc_buffer_t;
#endif

  array_view() = delete;

  ~array_view() restrict(amp,cpu) {}

  array_view(const array<T,N>& src) restrict(amp,cpu)
      : cache(src.internal()), extent(src.get_extent()), extent_base(extent), index_base(),
      offset(0) {}
  template <typename Container, class = typename std::enable_if<__is_container<Container>::value>::type>
    array_view(const extent<N>& extent, const Container& src)
        : array_view(extent, src.data())
  { static_assert( std::is_same<typename std::remove_const<typename std::remove_reference<decltype(*src.data())>::type>::type, T>::value, "container element type and array view element type must match"); }
    template <typename Container, class = typename std::enable_if<__is_container<Container>::value>::type>
      array_view(int e0, Container& src) : array_view(hc::extent<1>(e0), src) {}
    template <typename Container, class = typename std::enable_if<__is_container<Container>::value>::type>
      array_view(int e0, int e1, Container& src)
      : array_view(hc::extent<N>(e0, e1), src) {}
    template <typename Container, class = typename std::enable_if<__is_container<Container>::value>::type>
      array_view(int e0, int e1, int e2, Container& src)
      : array_view(hc::extent<N>(e0, e1, e2), src) {}

  array_view(const extent<N>& ext, const value_type* src) restrict(amp,cpu)
#if __KALMAR_ACCELERATOR__ == 1
      : cache((nc_T*)(src)), extent(ext), extent_base(ext), offset(0) {}
#else
      : cache(ext.size(), src), extent(ext), extent_base(ext),
          offset(0) {}
#endif
  array_view(int e0, value_type *src) restrict(amp,cpu)
      : array_view(hc::extent<1>(e0), src) {}
  array_view(int e0, int e1, value_type *src) restrict(amp,cpu)
      : array_view(hc::extent<2>(e0, e1), src) {}
  array_view(int e0, int e1, int e2, value_type *src) restrict(amp,cpu)
      : array_view(hc::extent<3>(e0, e1, e2), src) {}

  array_view(const array_view<nc_T, N>& other) restrict(amp,cpu) : cache(other.cache),
    extent(other.extent), extent_base(other.extent_base), index_base(other.index_base),
    offset(other.offset) {}

  array_view(const array_view& other) restrict(amp,cpu) : cache(other.cache),
    extent(other.extent), extent_base(other.extent_base), index_base(other.index_base),
    offset(other.offset) {}

  array_view& operator=(const array_view<T,N>& other) restrict(amp,cpu) {
    cache = other.cache;
    extent = other.extent;
    index_base = other.index_base;
    extent_base = other.extent_base;
    offset = other.offset;
    return *this;
  }

  array_view& operator=(const array_view& other) restrict(amp,cpu) {
    if (this != &other) {
      cache = other.cache;
      extent = other.extent;
      index_base = other.index_base;
      extent_base = other.extent_base;
      offset = other.offset;
    }
    return *this;
  }

  void copy_to(array<T,N>& dest) const { copy(*this, dest); }
  void copy_to(const array_view<T,N>& dest) const { copy(*this, dest); }

  extent<N> get_extent() const restrict(amp,cpu) { return extent; }
  accelerator_view get_source_accelerator_view() const { return cache.get_av(); }

  const T& operator[](const index<N>& idx) const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      cache.get_cpu_access();
#endif
    const T *ptr = reinterpret_cast<const T*>(cache.get() + offset);
    return ptr[Kalmar::amp_helper<N, index<N>, hc::extent<N>>::flatten(idx + index_base, extent_base)];
  }

  typename projection_helper<const T, N>::const_result_type
      operator[] (int i) const restrict(amp,cpu) {
    return projection_helper<const T, N>::project(*this, i);
  }

  const T& get_ref(const index<N>& idx) const restrict(amp,cpu);

  const T& operator()(const index<N>& idx) const restrict(amp,cpu) {
    return (*this)[idx];
  }
  const T& operator()(int i0) const restrict(amp,cpu) {
    static_assert(N == 1, "const T& array_view::operator()(int) is only permissible on array_view<T, 1>");
    return (*this)[index<1>(i0)];
  }

  const T& operator()(int i0, int i1) const restrict(amp,cpu) {
    static_assert(N == 2, "const T& array_view::operator()(int,int) is only permissible on array_view<T, 2>");
    return (*this)[index<2>(i0, i1)];
  }
  const T& operator()(int i0, int i1, int i2) const restrict(amp,cpu) {
    static_assert(N == 3, "const T& array_view::operator()(int,int, int) is only permissible on array_view<T, 3>");
    return (*this)[index<3>(i0, i1, i2)];
  }

  template <typename ElementType>
      array_view<const ElementType, N> reinterpret_as() const restrict(amp,cpu) {
          static_assert(N == 1, "reinterpret_as is only permissible on array views of rank 1");
#if __KALMAR_ACCELERATOR__ != 1
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
#endif
          int size = extent.size() * sizeof(T) / sizeof(ElementType);
          using buffer_type = typename array_view<ElementType, 1>::acc_buffer_t;
          array_view<const ElementType, 1> av(buffer_type(cache),
                                              extent<1>(size),
                                              (offset + index_base[0])* sizeof(T) / sizeof(ElementType));
          return av;
      }
  array_view<const T, N> section(const index<N>& idx,
                     const extent<N>& ext) const restrict(amp,cpu) {
    array_view<const T, N> av(cache, ext, extent_base, idx + index_base, offset);
    return av;
  }
  array_view<const T, N> section(const index<N>& idx) const restrict(amp,cpu) {
    hc::extent<N> ext(extent);
    Kalmar::amp_helper<N, index<N>, hc::extent<N>>::minus(idx, ext);
    return section(idx, ext);
  }

  array_view<const T, N> section(const extent<N>& ext) const restrict(amp,cpu) {
    index<N> idx;
    return section(idx, ext);
  }
  array_view<const T, 1> section(int i0, int e0) const restrict(amp,cpu) {
    static_assert(N == 1, "Rank must be 1");
    return section(index<1>(i0), hc::extent<1>(e0));
  }
  array_view<const T, 2> section(int i0, int i1, int e0, int e1) const restrict(amp,cpu) {
    static_assert(N == 2, "Rank must be 2");
    return section(index<2>(i0, i1), hc::extent<2>(e0, e1));
  }
  array_view<const T, 3> section(int i0, int i1, int i2, int e0, int e1, int e2) const restrict(amp,cpu) {
    static_assert(N == 3, "Rank must be 3");
    return section(index<3>(i0, i1, i2), hc::extent<3>(e0, e1, e2));
  }

  template <int K>
    array_view<const T, K> view_as(extent<K> viewExtent) const restrict(amp,cpu) {
      static_assert(N == 1, "view_as is only permissible on array views of rank 1");
#if __KALMAR_ACCELERATOR__ != 1
    if( viewExtent.size() > extent.size())
      throw runtime_exception("errorMsg_throw", 0);
#endif
      array_view<const T, K> av(cache, viewExtent, offset + index_base[0]);
      return av;
    }

  void synchronize() const { cache.get_cpu_access(); }
  completion_future synchronize_async() const {
      std::future<void> fut = std::async([&]() mutable { synchronize(); });
      return completion_future(fut.share());
  }

  void synchronize_to(const accelerator_view& av) const {
#if __KALMAR_ACCELERATOR__ != 1
      cache.sync_to(av.pQueue);
#endif
  }
  completion_future synchronize_to_async(const accelerator_view& av) const;

  void refresh() const { cache.refresh(); }
  const T* data() const restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
      cache.get_cpu_access();
#endif
    static_assert(N == 1, "data() is only permissible on array views of rank 1");
    return reinterpret_cast<const T*>(cache.get() + offset + index_base[0]);
  }
  const acc_buffer_t& internal() const restrict(amp,cpu) { return cache; }
  int get_offset() const restrict(amp,cpu) { return offset; }
  index<N> get_index_base() const restrict(amp,cpu) { return index_base; }
private:
  template <typename K, int Q> friend struct projection_helper;
  template <typename K, int Q> friend struct array_projection_helper;
  template <typename Q, int K> friend class array;
  template <typename Q, int K> friend class array_view;


  template<typename Q, int K>
      friend bool is_flat(const array_view<Q, K>&) noexcept;
  template <typename Q, int K>
      friend void copy(const array<Q, K>&, const array_view<Q, K>&);
  template <typename InputIter, typename Q, int K>
      friend void copy(InputIter, InputIter, const array_view<Q, K>&);
  template <typename Q, int K>
      friend void copy(const array_view<const Q, K>&, array<Q, K>&);
  template <typename OutputIter, typename Q, int K>
      friend void copy(const array_view<Q, K>&, OutputIter);
  template <typename Q, int K>
      friend void copy(const array_view<const Q, K>& src, const array_view<Q, K>& dest);

  // used by view_as and reinterpret_as
  array_view(const acc_buffer_t& cache, const hc::extent<N>& ext,
             int offset) restrict(amp,cpu)
      : cache(cache), extent(ext), extent_base(ext), offset(offset) {}

  // used by section and projection
  array_view(const acc_buffer_t& cache, const hc::extent<N>& ext_now,
             const extent<N>& ext_b,
             const index<N>& idx_b, int off) restrict(amp,cpu)
      : cache(cache), extent(ext_now), extent_base(ext_b), index_base(idx_b),
      offset(off) {}

  acc_buffer_t cache;
  hc::extent<N> extent;
  hc::extent<N> extent_base;
  index<N> index_base;
  int offset;
};

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
        T* ptr = dest.internal().map_ptr(true, dest.get_extent().size(), dest.get_offset());
        std::copy(srcBegin, srcEnd, ptr);
        dest.internal().unmap_ptr(ptr);
    }
    template<template <typename, int> class _amp_container>
    void operator()(const _amp_container<T, N> &src, Iter destBegin) {
        const T* ptr = src.internal().map_ptr(false, src.get_extent().size(), src.get_offset());
        std::copy(ptr, ptr + src.get_extent().size(), destBegin);
        src.internal().unmap_ptr(ptr);
    }
};

template <typename Iter, typename T>
struct do_copy<Iter, T, 1>
{
    template<template <typename, int> class _amp_container>
    void operator()(Iter srcBegin, Iter srcEnd, const _amp_container<T, 1>& dest) {
        T* ptr = dest.internal().map_ptr(true, dest.get_extent().size(),
                                         dest.get_offset() + dest.get_index_base()[0]);
        std::copy(srcBegin, srcEnd, ptr);
        dest.internal().unmap_ptr(ptr);
    }
    template<template <typename, int> class _amp_container>
    void operator()(const _amp_container<T, 1> &src, Iter destBegin) {
        const T* ptr = src.internal().map_ptr(false, src.get_extent().size(),
                                              src.get_offset() + src.get_index_base()[0]);
        std::copy(ptr, ptr + src.get_extent().size(), destBegin);
        src.internal().unmap_ptr(ptr);
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

template <typename T, int N>
void copy(const array_view<const T, N>& src, const array_view<T, N>& dest) {
    if (is_flat(src)) {
        if (is_flat(dest))
            src.internal().copy(dest.internal(), src.get_offset(),
                                dest.get_offset(), dest.get_extent().size());
        else {
            const T* pSrc = src.internal().map_ptr();
            const T* p = pSrc;
            T* pDst = dest.internal().map_ptr(true, dest.extent_base.size(), dest.offset);
            copy_input<const T*, T, N, 1>()(pSrc, pDst, dest.extent, dest.extent_base, dest.index_base);
            dest.internal().unmap_ptr(pDst);
            src.internal().unmap_ptr(p);
        }
    } else {
        if (is_flat(dest)) {
            T* pDst = dest.internal().map_ptr(true);
            T* p = pDst;
            const T* pSrc = src.internal().map_ptr(false, src.extent_base.size(), src.offset);
            copy_output<T*, T, N, 1>()(pSrc, pDst, src.extent, src.extent_base, src.index_base);
            dest.internal().unmap_ptr(p);
            src.internal().unmap_ptr(pSrc);
        } else {
            const T* pSrc = src.internal().map_ptr(false, src.extent_base.size(), src.offset);
            T* pDst = dest.internal().map_ptr(true, dest.extent_base.size(), dest.offset);
            copy_bidir<T, N, 1>()(pSrc, pDst, src.extent, src.extent_base,
                                  src.index_base, dest.extent_base, dest.index_base);
            dest.internal().unmap_ptr(pDst);
            src.internal().unmap_ptr(pSrc);

        }
    }
}

template <typename T>
void copy(const array_view<const T, 1>& src, const array_view<T, 1>& dest) {
    src.internal().copy(dest.internal(),
                        src.get_offset() + src.get_index_base()[0],
                        dest.get_offset() + dest.get_index_base()[0],
                        dest.get_extent().size());
}

template <typename T, int N>
void copy(const array_view<T, N>& src, const array_view<T, N>& dest) {
    const array_view<const T, N> buf(src);
    copy(buf, dest);
}

template <typename T, int N>
void copy(const array_view<const T, N>& src, array<T, N>& dest) {
    if (is_flat(src)) {
        src.internal().copy(dest.internal(), src.get_offset(),
                            dest.get_offset(), dest.get_extent().size());
    } else {
        T* pDst = dest.internal().map_ptr(true);
        T* p = pDst;
        const T* pSrc = src.internal().map_ptr(false, src.extent_base.size(), src.offset);
        copy_output<T*, T, N, 1>()(pSrc, pDst, src.extent, src.extent_base, src.index_base);
        src.internal().unmap_ptr(pSrc);
        dest.internal().unmap_ptr(p);
    }
}

template <typename T>
void copy(const array_view<const T, 1>& src, array<T, 1>& dest) {
    src.internal().copy(dest.internal(),
                        src.get_offset() + src.get_index_base()[0],
                        dest.get_offset() + dest.get_index_base()[0],
                        dest.get_extent().size());
}

template <typename T, int N>
void copy(const array_view<T, N>& src, array<T, N>& dest) {
    const array_view<const T, N> buf(src);
    copy(buf, dest);
}

template <typename T, int N>
void copy(const array<T, N>& src, const array_view<T, N>& dest) {
    if (is_flat(dest))
        src.internal().copy(dest.internal(), src.get_offset(),
                            dest.get_offset(), dest.get_extent().size());
    else {
        T* pSrc = src.internal().map_ptr();
        T* p = pSrc;
        T* pDst = dest.internal().map_ptr(true, dest.extent_base.size(), dest.offset);
        copy_input<T*, T, N, 1>()(pSrc, pDst, dest.extent, dest.extent_base, dest.index_base);
        dest.internal().unmap_ptr(pDst);
        src.internal().unmap_ptr(p);
    }
}

template <typename T>
void copy(const array<T, 1>& src, const array_view<T, 1>& dest) {
    src.internal().copy(dest.internal(),
                        src.get_offset() + src.get_index_base()[0],
                        dest.get_offset() + dest.get_index_base()[0],
                        dest.get_extent().size());
}

template <typename T, int N>
void copy(const array<T, N>& src, array<T, N>& dest) {
    src.internal().copy(dest.internal(), 0, 0, 0);
}

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest) {
    if (is_flat(dest))
        do_copy<InputIter, T, N>()(srcBegin, srcEnd, dest);
   else {
        T* ptr = dest.internal().map_ptr(true, dest.extent_base.size(), dest.offset);
        copy_input<InputIter, T, N, 1>()(srcBegin, ptr, dest.extent, dest.extent_base, dest.index_base);
        dest.internal().unmap_ptr(ptr);
    }
}

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest) {
#if __KALMAR_ACCELERATOR__ != 1
    if( ( std::distance(srcBegin,srcEnd) <=0 )||( std::distance(srcBegin,srcEnd) < dest.get_extent().size() ))
      throw runtime_exception("errorMsg_throw ,copy between different types", 0);
#endif
    do_copy<InputIter, T, N>()(srcBegin, srcEnd, dest);
}

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, const array_view<T, N>& dest) {
    InputIter srcEnd = srcBegin;
    std::advance(srcEnd, dest.get_extent().size());
    copy(srcBegin, srcEnd, dest);
}

template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, array<T, N>& dest) {
    InputIter srcEnd = srcBegin;
    std::advance(srcEnd, dest.get_extent().size());
    copy(srcBegin, srcEnd, dest);
}

template <typename OutputIter, typename T, int N>
void copy(const array_view<T, N> &src, OutputIter destBegin) {
    if (is_flat(src))
        do_copy<OutputIter, T, N>()(src, destBegin);
    else {
        T* ptr = src.internal().map_ptr(false, src.extent_base.size(), src.offset);
        copy_output<OutputIter, T, N, 1>()(ptr, destBegin, src.extent, src.extent_base, src.index_base);
        src.internal().unmap_ptr(ptr);
    }
}

template <typename OutputIter, typename T, int N>
void copy(const array<T, N> &src, OutputIter destBegin) {
    do_copy<OutputIter, T, N>()(src, destBegin);
}

// ------------------------------------------------------------------------
// utility function for copy_async_
// ------------------------------------------------------------------------

template <typename InputIter, typename OutputIter>
completion_future __amp_copy_async_impl(InputIter& src, OutputIter& dst) {
    std::future<void> fut = std::async([&]() mutable { copy(src, dst); });
    return completion_future(fut.share());
}

// ------------------------------------------------------------------------
// copy_async
// ------------------------------------------------------------------------

template <typename T, int N>
completion_future copy_async(const array<T, N>& src, array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array<T, N>& src, const array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array<T, N>& src, const array_view<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}


template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, const array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, const array_view<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}


template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, const array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, const array_view<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}


template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest) {
    std::future<void> fut = std::async([&]() mutable { copy(srcBegin, srcEnd, dest); });
    return completion_future(fut.share());
}

template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest) {
    std::future<void> fut = std::async([&]() mutable { copy(srcBegin, srcEnd, dest); });
    return completion_future(fut.share());
}


template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, array<T, N>& dest) {
    std::future<void> fut = std::async([&]() mutable { copy(srcBegin, dest); });
    return completion_future(fut.share());
}
template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, const array_view<T, N>& dest) {
    std::future<void> fut = std::async([&]() mutable { copy(srcBegin, dest); });
    return completion_future(fut.share());
}


template <typename OutputIter, typename T, int N>
completion_future copy_async(const array<T, N>& src, OutputIter destBegin) {
    std::future<void> fut = std::async([&]() mutable { copy(src, destBegin); });
    return completion_future(fut.share());
}
template <typename OutputIter, typename T, int N>
completion_future copy_async(const array_view<T, N>& src, OutputIter destBegin) {
    std::future<void> fut = std::async([&]() mutable { copy(src, destBegin); });
    return completion_future(fut.share());
}

// ------------------------------------------------------------------------
// parallel_for_each
// ------------------------------------------------------------------------

template <int N, typename Kernel>
completion_future parallel_for_each(const accelerator_view&, const extent<N>&, const Kernel&);

template <typename Kernel>
completion_future parallel_for_each(const accelerator_view&, const tiled_extent<3>&, const Kernel&);

template <typename Kernel>
completion_future parallel_for_each(const accelerator_view&, const tiled_extent<2>&, const Kernel&);

template <typename Kernel>
completion_future parallel_for_each(const accelerator_view&, const tiled_extent<1>&, const Kernel&);

template <int N, typename Kernel>
completion_future parallel_for_each(const extent<N>& compute_domain, const Kernel& f) {
    return parallel_for_each(accelerator::get_auto_selection_view(), compute_domain, f);
}

template <typename Kernel>
completion_future parallel_for_each(const tiled_extent<3>& compute_domain, const Kernel& f) {
    return parallel_for_each(accelerator::get_auto_selection_view(), compute_domain, f);
}

template <typename Kernel>
completion_future parallel_for_each(const tiled_extent<2>& compute_domain, const Kernel& f) {
    return parallel_for_each(accelerator::get_auto_selection_view(), compute_domain, f);
}

template <typename Kernel>
completion_future parallel_for_each(const tiled_extent<1>& compute_domain, const Kernel& f) {
    return parallel_for_each(accelerator::get_auto_selection_view(), compute_domain, f);
}

template <int N, typename Kernel, typename _Tp>
struct pfe_helper
{
    static inline void call(Kernel& k, _Tp& idx) restrict(amp,cpu) {
        int i;
        for (i = 0; i < k.ext[N - 1]; ++i) {
            idx[N - 1] = i;
            pfe_helper<N - 1, Kernel, _Tp>::call(k, idx);
        }
    }
};
template <typename Kernel, typename _Tp>
struct pfe_helper<0, Kernel, _Tp>
{
    static inline void call(Kernel& k, _Tp& idx) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ == 1
        k.k(idx);
#endif
    }
};

template <int N, typename Kernel>
class pfe_wrapper
{
public:
    explicit pfe_wrapper(const extent<N>& other, const Kernel& f) restrict(amp,cpu)
        : ext(other), k(f) {}
    void operator() (index<N> idx) restrict(amp,cpu) {
        pfe_helper<N - 3, pfe_wrapper<N, Kernel>, index<N>>::call(*this, idx);
    }
private:
    const extent<N> ext;
    const Kernel k;
    template <int K, typename Ker, typename _Tp>
        friend struct pfe_helper;
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma clang diagnostic ignored "-Wunused-variable"
//ND parallel_for_each, nontiled
template <int N, typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av,
    const extent<N>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
    size_t compute_domain_size = 1;
    for(int i = 0 ; i < N ; i++)
    {
      if(compute_domain[i]<=0)
        throw invalid_compute_domain("Extent is less or equal than 0.");
      if (static_cast<size_t>(compute_domain[i]) > 4294967295L)
        throw invalid_compute_domain("Extent size too large.");
      compute_domain_size *= static_cast<size_t>(compute_domain[i]);
      if (compute_domain_size > 4294967295L)
        throw invalid_compute_domain("Extent size too large.");
    }
    size_t ext[3] = {static_cast<size_t>(compute_domain[N - 1]),
        static_cast<size_t>(compute_domain[N - 2]),
        static_cast<size_t>(compute_domain[N - 3])};
    if (av.get_accelerator().get_device_path() == L"cpu") {
      throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
    }
    const pfe_wrapper<N, Kernel> _pf(compute_domain, f);
    return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<pfe_wrapper<N, Kernel>, 3>(av.pQueue, ext, NULL, _pf));
#else
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  int* foo1 = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
    auto bar = &pfe_wrapper<N, Kernel>::operator();
    auto qq = &index<N>::__cxxamp_opencl_index;
    int* foo = reinterpret_cast<int*>(&pfe_wrapper<N, Kernel>::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//1D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const extent<1>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext = compute_domain[0];
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 1>(av.pQueue, &ext, NULL, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//2D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const extent<2>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = {static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 2>(av.pQueue, ext, NULL, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//3D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const extent<3>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[3] = {static_cast<size_t>(compute_domain[2]),
                   static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 3>(av.pQueue, ext, NULL, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//1D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const tiled_extent<1>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext = compute_domain[0];
  size_t tile = compute_domain.tile_dim[0];
  if (static_cast<size_t>(compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if(ext % tile != 0) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 1>(av.pQueue, &ext, &tile, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<1> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//2D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const tiled_extent<2>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = { static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[2] = { static_cast<size_t>(compute_domain.tile_dim[1]),
                     static_cast<size_t>(compute_domain.tile_dim[0]) };
  if (static_cast<size_t>(compute_domain.tile_dim[1] * compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 2>(av.pQueue, ext, tile, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<2> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//3D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future parallel_for_each(
    const accelerator_view& av, const tiled_extent<3>& compute_domain, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[3] = { static_cast<size_t>(compute_domain[2]),
                    static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[3] = { static_cast<size_t>(compute_domain.tile_dim[2]),
                     static_cast<size_t>(compute_domain.tile_dim[1]),
                     static_cast<size_t>(compute_domain.tile_dim[0]) };
  if (static_cast<size_t>(compute_domain.tile_dim[2] * compute_domain.tile_dim[1]* compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0) || (ext[2] % tile[2] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  return completion_future(Kalmar::mcw_cxxamp_launch_kernel_async<Kernel, 3>(av.pQueue, ext, tile, f));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<3> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//1D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const extent<1>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
      return;
  }
#endif
  size_t ext = compute_domain[0];
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 1>(av.pQueue, &ext, NULL, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//2D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const extent<2>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
      return;
  }
#endif
  size_t ext[2] = {static_cast<size_t>(compute_domain[1]),
      static_cast<size_t>(compute_domain[0])};
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 2>(av.pQueue, ext, NULL, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//3D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const extent<3>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
      return;
  }
#endif
  size_t ext[3] = {static_cast<size_t>(compute_domain[2]),
      static_cast<size_t>(compute_domain[1]),
      static_cast<size_t>(compute_domain[0])};
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 3>(av.pQueue, ext, NULL, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//1D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const tiled_extent<1>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(amp,cpu) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L) {
    throw invalid_compute_domain("Extent size too large.");
  }
  size_t ext = compute_domain[0];
  size_t tile = compute_domain.tile_dim[0];
  if (static_cast<size_t>(compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if (ext % tile != 0) {
    throw invalid_compute_domain("Extent can't be evenly divisible by tile size.");
  }
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
      return;
  }
#endif
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 1>(av.pQueue, &ext, &tile, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<1> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//2D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const tiled_extent<2>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = { static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[2] = { static_cast<size_t>(compute_domain.tile_dim[1]),
                     static_cast<size_t>(compute_domain.tile_dim[0]) };
  if (static_cast<size_t>(compute_domain.tile_dim[1] * compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
  } else
#endif
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 2>(av.pQueue, ext, tile, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<2> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
// variants of parallel_for_each that supports runtime allocation of tile static
//3D parallel_for_each, tiled
template <typename Kernel>
__attribute__((noinline,used))
completion_future parallel_for_each(const accelerator_view& av,
                       const tiled_extent<3>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[3] = { static_cast<size_t>(compute_domain[2]),
                    static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[3] = { static_cast<size_t>(compute_domain.tile_dim[2]),
                     static_cast<size_t>(compute_domain.tile_dim[1]),
                     static_cast<size_t>(compute_domain.tile_dim[0]) };
  if (static_cast<size_t>(compute_domain.tile_dim[2] * compute_domain.tile_dim[1]* compute_domain.tile_dim[0]) > 1024) {
    throw invalid_compute_domain("The maximum number of threads in a tile is 1024");
  }
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0) || (ext[2] % tile[2] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  if (CLAMP::is_cpu()) {
      launch_cpu_task(av.pQueue, f, compute_domain);
  } else
#endif
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(Kalmar::__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  void *kernel = Kalmar::mcw_cxxamp_get_kernel<Kernel>(av.pQueue, f);
  allocator.setStaticGroupSegmentSize(av.pQueue->GetGroupSegmentSize(kernel));
  return completion_future(Kalmar::mcw_cxxamp_execute_kernel_with_dynamic_group_memory_async<Kernel, 3>(av.pQueue, ext, tile, f, kernel, allocator.getDynamicGroupSegmentSize()));
#else //if __KALMAR_ACCELERATOR__ != 1
  tiled_index<3> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
#endif
}
#pragma clang diagnostic pop

template <typename Kernel>
completion_future parallel_for_each(const extent<1>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template <typename Kernel>
completion_future parallel_for_each(const extent<2>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template <typename Kernel>
completion_future parallel_for_each(const extent<3>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template <typename Kernel>
completion_future parallel_for_each(const tiled_extent<1>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template<typename Kernel>
completion_future parallel_for_each(const tiled_extent<2>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

template<typename Kernel>
completion_future parallel_for_each(const tiled_extent<3>& compute_domain,
                       ts_allocator& allocator,
                       const Kernel& f) {
  auto que = Kalmar::get_availabe_que(f);
  const accelerator_view av(que);
  return parallel_for_each(av, compute_domain, allocator, f);
}

} // namespace hc
