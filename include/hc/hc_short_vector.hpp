//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "hc_defines.h"
#include "hc_norm_unorm.hpp"

#include <type_traits>

namespace hc
{
    namespace short_vector
    {
        template<typename T, int n>
        class Vector_base {
            using VecT = typename std::conditional<
                std::is_same<T, norm>{} || std::is_same<T, unorm>{},
                float,
                T>::type __attribute__((ext_vector_type(n)));

            union { // TODO: revise, this is only used for ref_n() functions.
                VecT data_;
                T components_[n]{};
            };

            friend class Vector_base<T, 2>;
            friend class Vector_base<T, 3>;
            friend class Vector_base<T, 4>;

            friend
            inline
            Vector_base operator+(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} += y;
            }
            friend
            inline
            Vector_base operator-(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} -= y;
            }
            friend
            inline
            Vector_base operator*(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} *= y;
            }
            friend
            inline
            Vector_base operator/(
                const Vector_base& x, const Vector_base& y) [[cpu, hc]]
            {
                return Vector_base{x} /= y;
            }
            friend
            inline
            bool operator==(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                auto tmp = x.data_ == y.data_;
                for (auto i = 0; i != n; ++i) if (tmp[i] == 0) return false;

                return true;
            }
            friend
            inline
            bool operator!=(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return !(x == y);
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            friend
            inline
            Vector_base operator%(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} %= y;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            friend
            inline
            Vector_base operator^(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} ^= y;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            friend
            inline
            Vector_base operator|(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} |= y;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            friend
            inline
            Vector_base operator&(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} &= y;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            friend
            inline
            Vector_base operator<<(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} <<= y;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            friend
            inline
            Vector_base operator>>(
                const Vector_base& x, const Vector_base& y) noexcept [[cpu, hc]]
            {
                return Vector_base{x} >>= y;
            }

            explicit
            Vector_base(VecT x) noexcept [[cpu, hc]] : data_{x} {}
        public:
            using value_type = T;

            static constexpr int size{n};

            // CREATORS
            Vector_base() [[cpu, hc]] = default;
            Vector_base(const Vector_base&) [[cpu, hc]] = default;
            Vector_base(Vector_base&&) = default;
            constexpr
            Vector_base(T x) noexcept [[cpu, hc]] : data_(x) {}
            template<
                int m = n, typename std::enable_if<m == 2>::type* = nullptr>
            constexpr
            Vector_base(T x, T y) noexcept [[cpu, hc]] : data_{x, y} {}
            template<
                int m = n, typename std::enable_if<m == 3>::type* = nullptr>
            constexpr
            Vector_base(T x, T y, T z) noexcept [[cpu, hc]] : data_{x, y, z} {}
            template<
                int m = n, typename std::enable_if<m == 4>::type* = nullptr>
            constexpr
            Vector_base(T x, T y, T z, T w) noexcept [[cpu, hc]]
                : data_{x, y, z, w}
            {}
            template<
                typename U,
                int m,
                typename std::enable_if<
                    std::is_convertible<U, T>{} && m == n>::type* = nullptr>
            Vector_base(const Vector_base<U, m>& x)
            {   // TODO: optimise.
                for (auto i = 0; i != m; ++i) data_[i] = x.data_[i];
            }
            ~Vector_base() [[cpu, hc]] = default;

            // MANIPULATORS
            Vector_base& operator=(const Vector_base&) [[cpu, hc]] = default;
            Vector_base& operator=(Vector_base&&) [[cpu, hc]] = default;
            Vector_base& operator+=(const Vector_base& x) noexcept [[cpu, hc]]
            {
                data_ += x.data_;
                return *this;
            }
            Vector_base& operator-=(const Vector_base& x) noexcept [[cpu, hc]]
            {
                data_ -= x.data_;
                return *this;
            }
            Vector_base& operator*=(const Vector_base& x) noexcept [[cpu, hc]]
            {
                data_ *= x.data_;
                return *this;
            }
            Vector_base& operator/=(const Vector_base& x) [[cpu, hc]]
            {
                data_ /= x.data_;
                return *this;
            }
            Vector_base& operator++() noexcept [[cpu, hc]]
            {
                ++data_;
                return *this;
            }
            Vector_base operator++(int) noexcept [[cpu, hc]]
            {
                Vector_base tmp{*this};
                ++*this;
                return tmp;
            }
            Vector_base& operator--() noexcept [[cpu, hc]]
            {
                --data_;
                return *this;
            }
            Vector_base operator--(int) noexcept [[cpu, hc]]
            {
                Vector_base tmp{*this};
                --*this;
                return tmp;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            Vector_base operator%=(const Vector_base& x) [[cpu, hc]]
            {
                data_ %= x.data_;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            Vector_base operator^=(const Vector_base& x) noexcept [[cpu, hc]]
            {
                data_ ^= x.data_;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            Vector_base operator|=(const Vector_base& x) noexcept [[cpu, hc]]
            {
                data_ |= x.data_;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            Vector_base operator&=(const Vector_base& x) noexcept [[cpu, hc]]
            {
                data_ &= x.data_;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            Vector_base operator>>=(const Vector_base& x) [[cpu, hc]]
            {
                data_ >>= x.data_;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            Vector_base operator<<=(const Vector_base& x) [[cpu, hc]]
            {
                data_ <<= x.data_;
                return *this;
            }

            // one-component access
            void set_x(T x) noexcept [[cpu, hc]] { data_.x = x; }
            template<
                int m = n, typename std::enable_if<m == 2>::type* = nullptr>
            void set_y(T x) noexcept [[cpu, hc]] { data_.y = x; }
            template<
                int m = n, typename std::enable_if<m == 3>::type* = nullptr>
            void set_z(T x) noexcept [[cpu, hc]] { data_.z = x; }
            template<
                int m = n, typename std::enable_if<m == 4>::type* = nullptr>
            void set_w(T x) noexcept [[cpu, hc]] { data_.w = x; }
            void set_r(T x) noexcept [[cpu, hc]] { set_x(x); }
            template<
                int m = n, typename std::enable_if<m == 2>::type* = nullptr>
            void set_g(T x) noexcept [[cpu, hc]] { set_y(x); }
            template<
                int m = n, typename std::enable_if<m == 3>::type* = nullptr>
            void set_b(T x) noexcept [[cpu, hc]] { set_z(x); }
            template<
                int m = n, typename std::enable_if<m == 4>::type* = nullptr>
            void set_a(T x) noexcept [[cpu, hc]] { set_w(x); }

            T& ref_x() noexcept [[cpu, hc]] { return components_[0]; }
            template<
                int m = n, typename std::enable_if<m == 2>::type* = nullptr>
            T& ref_y() noexcept [[cpu, hc]] { return components_[1]; }
            template<
                int m = n, typename std::enable_if<m == 3>::type* = nullptr>
            T& ref_z() noexcept [[cpu, hc]] { return components_[2]; }
            template<
                int m = n, typename std::enable_if<m == 4>::type* = nullptr>
            T& ref_w() noexcept [[cpu, hc]] { return components_[3]; }
            T& ref_r() noexcept [[cpu, hc]] { return ref_x(); }
            template<
                int m = n, typename std::enable_if<m == 2>::type* = nullptr>
            T& ref_g() noexcept [[cpu, hc]] { return ref_y(); }
            template<
                int m = n, typename std::enable_if<m == 3>::type* = nullptr>
            T& ref_b() noexcept [[cpu, hc]] { return ref_z(); }
            template<
                int m = n, typename std::enable_if<m == 4>::type* = nullptr>
            T& ref_a() noexcept [[cpu, hc]] { return ref_w(); }

            // two-component access
            template<
                int m = n, typename std::enable_if<(m > 1)>::type* = nullptr>
            void set_xy(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.xy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 1)>::type* = nullptr>
            void set_yx(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.yx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_xz(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.xz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_zx(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.zx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xw(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.xw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wx(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.wx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_yz(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.yz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_zy(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.zy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_yw(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.yw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wy(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.wy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zw(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.zw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wz(Vector_base<T, 2> x) noexcept [[cpu, hc]]
            {
                data_.wz = x.data_;
            }
            // three-component access
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_xyz(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.xyz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_yzx(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.yzx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_zxy(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.zxy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_xzy(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.xzy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_yxz(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.yxz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            void set_zyx(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.zyx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xyw(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.xyw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_ywx(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.ywx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wxy(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.wxy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xwy(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.xwy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_yxw(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.yxw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wyx(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.wyx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xzw(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.xzw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zwx(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.zwx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wxz(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.wxz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xwz(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.xwz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zxw(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.zxw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wzx(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.wzx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_yzw(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.yzw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zwy(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.zwy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wyz(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.wyz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wzy(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.wzy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_ywz(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.ywz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zyw(Vector_base<T, 3> x) noexcept [[cpu, hc]]
            {
                data_.zyw = x.data_;
            }

            // four-component access
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xyzw(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.xyzw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xzwy(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.xzwy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xwyz(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.xwyz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xzyw(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.xzyw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xywz(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.xywz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_xwzy(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.xwzy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_yzwx(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.yzwx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_ywxz(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.ywxz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_yxzw(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.yxzw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_yxwz(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.yxwz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_yzxw(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.yzxw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_ywzx(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.ywzx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zwxy(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.zwxy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zxyw(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.zxyw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zywx(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.zywx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zyxw(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.zyxw = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zwyx(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.zwyx = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_zxwy(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.zxwy = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wxyz(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.wxyz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wyzx(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.wxyz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wzxy(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.wxyz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wzyx(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.wxyz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wxzy(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.wxyz = x.data_;
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            void set_wyxz(Vector_base<T, 4> x) noexcept [[cpu, hc]]
            {
                data_.wxyz = x.data_;
            }

            // ACCESSORS
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            Vector_base operator~() const noexcept [[cpu, hc]]
            {
                Vector_base tmp{*this};
                tmp.data_ = ~tmp.data_;
                return tmp;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
            Vector_base operator-() const noexcept [[cpu, hc]]
            {
                Vector_base tmp{*this};
                tmp.data_ = -tmp.data_;
                return tmp;
            }

            // one-component access
            T get_x() const noexcept [[cpu, hc]] { return T{data_.x}; }
            template<
                int m = n, typename std::enable_if<(m > 1)>::type* = nullptr>
            T get_y() const noexcept [[cpu, hc]] { return T{data_.y}; }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            T get_z() const noexcept [[cpu, hc]] { return T{data_.z}; }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            T get_w() const noexcept [[cpu, hc]] { return T{data_.w}; }
            T get_r() const noexcept [[cpu, hc]] { return get_x(); }
            template<
                int m = n, typename std::enable_if<(m > 1)>::type* = nullptr>
            T get_g() const noexcept [[cpu, hc]] { return get_y(); }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            T get_b() const noexcept [[cpu, hc]] { return get_z(); }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            T get_a() const noexcept [[cpu, hc]] { return get_w(); }

            // two-component access
            template<
                int m = n, typename std::enable_if<(m > 1)>::type* = nullptr>
            Vector_base<T, 2> get_xy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.xy};
            }
            template<
                int m = n, typename std::enable_if<(m > 1)>::type* = nullptr>
            Vector_base<T, 2> get_yx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.yx};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 2> get_xz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.xz};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 2> get_zx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.zx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 2> get_xw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.xw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 2> get_wx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.wx};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 2> get_yz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.yz};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 2> get_zy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.zy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 2> get_yw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.yw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 2> get_wy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.wy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 2> get_zw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.zw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 2> get_wz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 2>{data_.wz};
            }

            // three-component access
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 3> get_xyz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.xyz};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 3> get_yzx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.yzx};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 3> get_zxy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.zxy};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 3> get_xzy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.xzy};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 3> get_yxz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.yxz};
            }
            template<
                int m = n, typename std::enable_if<(m > 2)>::type* = nullptr>
            Vector_base<T, 3> get_zyx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.zyx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_xyw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.xyw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_ywx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.ywx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_wxy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.wxy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_xwy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.xwy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_yxw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.yxw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_wyx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.wyx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_xzw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.xzw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_zwx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.zwx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_wxz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.wxz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_xwz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.xwz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_zxw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.zxw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_wzx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.wzx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_yzw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.yzw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_zwy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.zwy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_wyz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.wyz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_wzy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.wzy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_ywz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.ywz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 3> get_zyw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 3>{data_.zyw};
            }

            // four-component access
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_xyzw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.xyzw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_xzwy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.xzwy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_xwyz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.xwyz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_xzyw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.xzyw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_xywz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.xywz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_xwzy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.xwzy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_yzwx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.yzwx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_ywxz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.ywxz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_yxzw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.yxzw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_yxwz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.yxwz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_yzxw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.yzxw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_ywzx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.ywzx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_zwxy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.zwxy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_zxyw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.zxyw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_zywx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.zywx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_zyxw() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.zyxw};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_zwyx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.zwyx};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_zxwy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.zxwy};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_wxyz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.wxyz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_wyzx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.wxyz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_wzxy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.wxyz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_wzyx() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.wxyz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_wxzy() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.wxyz};
            }
            template<
                int m = n, typename std::enable_if<(m > 3)>::type* = nullptr>
            Vector_base<T, 4> get_wyxz() const noexcept [[cpu, hc]]
            {
                return Vector_base<T, 4>{data_.wxyz};
            }
        };

        template<typename, int> struct short_vector;
        template<typename> struct short_vector_traits; // TODO: don't use macro.

        #define MAKE_HC_VECTOR_TYPE(T, n) \
            using T##_##n = Vector_base<T, n>;\
            using T##n = Vector_base<T, n>;\
            template<> struct short_vector<T, n> { using type = T##_##n; };\
            template<>\
            struct short_vector_traits<T##_##n> {\
                using value_type = T;\
                static constexpr int size{n};\
            };

        using uchar = unsigned char;
        using ushort = unsigned short;
        using uint = unsigned int;
        using ulong = unsigned long;
        using longlong = long long;
        using ulonglong = unsigned long long;
        using half = _Float16;

        MAKE_HC_VECTOR_TYPE(char, 1)
        MAKE_HC_VECTOR_TYPE(char, 2)
        MAKE_HC_VECTOR_TYPE(char, 3)
        MAKE_HC_VECTOR_TYPE(char, 4)
        MAKE_HC_VECTOR_TYPE(uchar, 1)
        MAKE_HC_VECTOR_TYPE(uchar, 2)
        MAKE_HC_VECTOR_TYPE(uchar, 3)
        MAKE_HC_VECTOR_TYPE(uchar, 4)
        MAKE_HC_VECTOR_TYPE(short, 1)
        MAKE_HC_VECTOR_TYPE(short, 2)
        MAKE_HC_VECTOR_TYPE(short, 3)
        MAKE_HC_VECTOR_TYPE(short, 4)
        MAKE_HC_VECTOR_TYPE(ushort, 1)
        MAKE_HC_VECTOR_TYPE(ushort, 2)
        MAKE_HC_VECTOR_TYPE(ushort, 3)
        MAKE_HC_VECTOR_TYPE(ushort, 4)
        MAKE_HC_VECTOR_TYPE(int, 1)
        MAKE_HC_VECTOR_TYPE(int, 2)
        MAKE_HC_VECTOR_TYPE(int, 3)
        MAKE_HC_VECTOR_TYPE(int, 4)
        MAKE_HC_VECTOR_TYPE(uint, 1)
        MAKE_HC_VECTOR_TYPE(uint, 2)
        MAKE_HC_VECTOR_TYPE(uint, 3)
        MAKE_HC_VECTOR_TYPE(uint, 4)
        MAKE_HC_VECTOR_TYPE(long, 1)
        MAKE_HC_VECTOR_TYPE(long, 2)
        MAKE_HC_VECTOR_TYPE(long, 3)
        MAKE_HC_VECTOR_TYPE(long, 4)
        MAKE_HC_VECTOR_TYPE(ulong, 1)
        MAKE_HC_VECTOR_TYPE(ulong, 2)
        MAKE_HC_VECTOR_TYPE(ulong, 3)
        MAKE_HC_VECTOR_TYPE(ulong, 4)
        MAKE_HC_VECTOR_TYPE(longlong, 1)
        MAKE_HC_VECTOR_TYPE(longlong, 2)
        MAKE_HC_VECTOR_TYPE(longlong, 3)
        MAKE_HC_VECTOR_TYPE(longlong, 4)
        MAKE_HC_VECTOR_TYPE(ulonglong, 1)
        MAKE_HC_VECTOR_TYPE(ulonglong, 2)
        MAKE_HC_VECTOR_TYPE(ulonglong, 3)
        MAKE_HC_VECTOR_TYPE(ulonglong, 4)
        MAKE_HC_VECTOR_TYPE(half, 1)
        MAKE_HC_VECTOR_TYPE(half, 2)
        MAKE_HC_VECTOR_TYPE(half, 3)
        MAKE_HC_VECTOR_TYPE(half, 4)
        MAKE_HC_VECTOR_TYPE(float, 1)
        MAKE_HC_VECTOR_TYPE(float, 2)
        MAKE_HC_VECTOR_TYPE(float, 3)
        MAKE_HC_VECTOR_TYPE(float, 4)
        MAKE_HC_VECTOR_TYPE(double, 1)
        MAKE_HC_VECTOR_TYPE(double, 2)
        MAKE_HC_VECTOR_TYPE(double, 3)
        MAKE_HC_VECTOR_TYPE(double, 4)
        MAKE_HC_VECTOR_TYPE(norm, 1)
        MAKE_HC_VECTOR_TYPE(norm, 2)
        MAKE_HC_VECTOR_TYPE(norm, 3)
        MAKE_HC_VECTOR_TYPE(norm, 4)
        MAKE_HC_VECTOR_TYPE(unorm, 1)
        MAKE_HC_VECTOR_TYPE(unorm, 2)
        MAKE_HC_VECTOR_TYPE(unorm, 3)
        MAKE_HC_VECTOR_TYPE(unorm, 4)
    } // namespace short_vector
} // namespace hc