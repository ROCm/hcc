#pragma once

#include <hsa/hsa.h>

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <utility>

namespace hc
{
    class accelerator_view;
    template<typename, int> class array;
    template<typename, int> class array_view;
    template<int> class extent;
    template<int> class tiled_extent;

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
        struct State_ {
            std::shared_future<void> future{};
            std::once_flag maybe_then{};

            State_(std::shared_future<void> fut)
                : future{std::move(fut)}, maybe_then{}
            {}
        };

        std::shared_ptr<State_> state_{};

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

        // CREATORS
        completion_future(std::shared_future<void> future)
            : state_{std::make_shared<State_>(std::move(future))}
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

        ~completion_future()
        {
            if (!state_) return;
            if (state_.use_count() > 1) return;

            if (state_->future.valid()) state_->future.wait();
        }
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
            if (state_) state_->future.get();
        }

        /**
         * This method is functionally identical to
         * std::shared_future<void>::valid. This returns true if this
         * completion_future is associated with an asynchronous operation.
         */
        bool valid() const
        {
            return state_ ? state_->future.valid() : false;
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
            if (state_) state_->future.wait();

            // TODO: printf:(
            //detail::getContext()->flushPrintfBuffer();
        }

        template<typename Rep, typename Period>
        std::future_status wait_for(
            const std::chrono::duration<Rep, Period>& rel_time) const
        {   // TODO: this should probably be an exception if !state_.
            return state_ ?
                state_->future.wait_for(rel_time) :
                std::future_status::deferred;
        }

        template<typename Clock, typename Duration>
        std::future_status wait_until(
            const std::chrono::time_point<Clock, Duration>& abs_time) const
        {
            return state_ ?
                state_->future.wait_until(abs_time) :
                std::future_status::deferred;
        }

        /** @} */

        /**
         * Conversion operator to std::shared_future<void>. This method returns
         * a shared_future<void> object corresponding to this completion_future
         * object and refers to the same asynchronous operation.
         */
        operator std::shared_future<void>() const
        {
            return state_ ? state_->future : std::shared_future<void>{};
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
                state_->maybe_then, [=](std::shared_future<void> fut) {
                std::thread{[=]() { fut.wait(); func(); }}.detach();
            }, state_->future);
        }

        /**
         * Get if the async operations has been completed.
         *
         * @return True if the async operation has been completed, false if not.
         */
        bool is_ready()
        {
            return state_->future.wait_for(std::chrono::nanoseconds{0}) ==
                std::future_status::ready;
        }
    };
} // Namespace hc.