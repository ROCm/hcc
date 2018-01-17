//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "type_support.hpp"

#include <type_traits>
#include <utility>

namespace hc2
{
    template<typename T, typename D>
    class RAII_handle {
        friend
        inline
        const T& handle(const RAII_handle& x) { return x.h_; }

        friend
        inline
        T& handle(RAII_handle& x) { return x.h_; }

        T h_;
        D d_;
    public:
        RAII_handle() = default;
        RAII_handle(const RAII_handle&) = default;
        RAII_handle(RAII_handle&&) = default;

        RAII_handle(T h, D d) : h_{std::move(h)}, d_{std::move(d)} {}

        template<
            typename E,
            typename std::enable_if<std::is_convertible<E, D>{}>::type* = nullptr>
        RAII_handle(T h, E d) : RAII_handle{std::move(h), std::move(d)} {}

        RAII_handle& operator=(const RAII_handle&) = default;
        RAII_handle& operator=(RAII_handle&&) = default;

        operator T() const { return h_; }

        ~RAII_handle() { d_(h_); }
    };

    template<typename T, typename D>
    class RAII_move_only_handle :
        public Swappable<RAII_move_only_handle<T, D>> {
        friend class Swappable<RAII_move_only_handle>;

        friend
        inline
        const T& handle(const RAII_move_only_handle& x) { return x.h_; }

        friend
        inline
        T& handle(RAII_move_only_handle& x) { return x.h_; }

        T h_;
        D d_;
        bool v_ = false;

        void swp_(RAII_move_only_handle& x)
        {
            using std::swap;

            swap(h_, x.h_);
            swap(d_, x.d_);
            swap(v_, x.v_);
        }
    public:
        RAII_move_only_handle() = default;
        RAII_move_only_handle(const RAII_move_only_handle&) = delete;
        RAII_move_only_handle(RAII_move_only_handle&& x)
            : RAII_move_only_handle{std::move(x.h_), std::move(x.d_)}
        {
            x.h_ = T{};
            x.v_ = false;
        }

        RAII_move_only_handle(T h, D d)
            : h_{std::move(h)}, d_{std::move(d)}, v_{true}
        {}

        template<
            typename E,
            typename std::enable_if<std::is_convertible<E, D>{}>::type* = nullptr>
        RAII_move_only_handle(T h, E d)
            : RAII_move_only_handle{std::move(h), std::move(d)}
        {}

        RAII_move_only_handle& operator=(RAII_move_only_handle x)
        {
            using std::swap;

            swap(*this, x);

            return *this;
        }

        ~RAII_move_only_handle() { if (v_) d_(h_); v_ = false; }
    };

    template<typename D>
    class RAII_stateless_handle {
        D d_;
    public:
        RAII_stateless_handle() = default;
        RAII_stateless_handle(const RAII_stateless_handle&) = default;
        RAII_stateless_handle(RAII_stateless_handle&&) = default;

        template<typename C>
        RAII_stateless_handle(const C& ctor, D dtor) : d_{std::move(dtor)}
        {
            ctor();
        }

        RAII_stateless_handle& operator=(
            const RAII_stateless_handle&) = default;
        RAII_stateless_handle& operator=(RAII_stateless_handle&&) = default;

        ~RAII_stateless_handle() { d_(); }
    };
}