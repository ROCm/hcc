//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

namespace hc
{
    namespace short_vector
    {
        template<typename T>
        constexpr
        inline
        T _clamp(T x, T x_min, T x_max) [[cpu, hc]]
        {   // TODO: consider using med3 for [[hc]]
            return (x < x_min) ? x_min : ((x_max < x) ? x_max : x);
        }

        class unorm;

        class norm {
            float x_{};

            friend class unorm;

            friend
            inline
            norm operator+(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return norm{x} += y;
            }
            friend
            inline
            norm operator-(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return norm{x} -= y;
            }
            friend
            inline
            norm operator*(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return norm{x} *= y;
            }
            friend
            inline
            norm operator/(const norm& x, const norm& y) [[cpu, hc]]
            {
                return norm{x} /= y;
            }
            friend
            inline
            bool operator==(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return x.x_ == y.x_;
            }
            friend
            inline
            bool operator!=(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return !(x == y);
            }
            friend
            inline
            bool operator<(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return x.x_ < y.x_;
            }
            friend
            inline
            bool operator<=(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return !(y < x);
            }
            friend
            inline
            bool operator>(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return y < x;
            }
            friend
            inline
            bool operator>=(const norm& x, const norm& y) noexcept [[cpu, hc]]
            {
                return !(x < y);
            }
        public:
            // CREATORS
            norm() [[cpu, hc]] = default;
            norm(const norm&) [[cpu, hc]] = default;
            norm(norm&&) [[cpu, hc]] = default;
            constexpr
            norm(const unorm& x) noexcept [[cpu, hc]];
            constexpr
            explicit
            norm(float x) noexcept [[cpu, hc]] : x_{_clamp(x, -1.0f, 1.0f)} {}
            constexpr
            explicit
            norm(unsigned int x) noexcept [[cpu, hc]]
                : norm{static_cast<float>(x)}
            {}
            constexpr
            explicit
            norm(int x) noexcept [[cpu, hc]] : norm{static_cast<float>(x)} {}
            constexpr
            explicit
            norm(double x) noexcept [[cpu, hc]] : norm{static_cast<float>(x)} {}
            ~norm() [[cpu, hc]] = default;

            // MANIPULATORS
            norm& operator=(const norm&) [[cpu, hc]] = default;
            norm& operator=(norm&&) [[cpu, hc]] = default;
            norm& operator+=(const norm& x) noexcept [[cpu, hc]]
            {
                return *this = norm{x_ + x.x_};
            }
            norm& operator-=(const norm& x) noexcept [[cpu, hc]]
            {
                return *this = norm{x_ - x.x_};
            }
            norm& operator*=(const norm& x) noexcept [[cpu, hc]]
            {
                return *this = norm{x_ * x.x_};
            }
            norm& operator/=(const norm& x) [[cpu, hc]]
            {
                return *this = norm{x_ / x.x_};
            }
            norm& operator++() noexcept [[cpu, hc]]
            {
                return *this = norm{++x_};
            }
            norm operator++(int) noexcept [[cpu, hc]]
            {
                norm tmp{*this};
                ++*this;
                return tmp;
            }
            norm& operator--() noexcept [[cpu, hc]]
            {
                return *this = norm{--x_};
            }
            norm operator--(int) noexcept [[cpu, hc]]
            {
                norm tmp{*this};
                --*this;
                return tmp;
            }

            // ACCESSORS
            constexpr
            operator float() const noexcept [[cpu, hc]] { return x_; }
            constexpr
            norm operator-() const noexcept [[cpu, hc]] { return norm{-x_}; }
        };

        static constexpr norm NORM_MAX{1.0f};
        static constexpr norm NORM_MIN{-1.0f};
        static constexpr norm NORM_ZERO{0.0f};

        class unorm {
            float x_{};

            friend class norm;

            friend
            inline
            unorm operator+(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return unorm{x} += y;
            }
            friend
            inline
            unorm operator-(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return unorm{x} -= y;
            }
            friend
            inline
            unorm operator*(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return unorm{x} *= y;
            }
            friend
            inline
            unorm operator/(const unorm& x, const unorm& y) [[cpu, hc]]
            {
                return unorm{x} /= y;
            }
            friend
            inline
            bool operator==(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return x.x_ == y.x_;
            }
            friend
            inline
            bool operator!=(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return !(x == y);
            }
            friend
            inline
            bool operator<(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return x.x_ < y.x_;
            }
            friend
            inline
            bool operator<=(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return !(y < x);
            }
            friend
            inline
            bool operator>(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return y < x;
            }
            friend
            inline
            bool operator>=(const unorm& x, const unorm& y) noexcept [[cpu, hc]]
            {
                return !(x < y);
            }
        public:
            // CREATORS
            unorm() [[cpu, hc]] = default;
            unorm(const unorm&) [[cpu, hc]] = default;
            unorm(unorm&&) [[cpu, hc]] = default;
            constexpr
            explicit
            unorm(const norm& x) noexcept [[cpu, hc]] : unorm{x.x_} {}
            constexpr
            explicit
            unorm(float x) noexcept [[cpu, hc]] : x_{_clamp(x, 0.0f, 1.0f)} {}
            constexpr
            explicit
            unorm(unsigned int x) noexcept [[cpu, hc]]
                : unorm{static_cast<float>(x)}
            {}
            constexpr
            explicit
            unorm(int x) noexcept [[cpu, hc]] : unorm{static_cast<float>(x)} {}
            constexpr
            explicit
            unorm(double x) noexcept [[cpu, hc]]
                : unorm{static_cast<float>(x)}
            {}
            ~unorm() [[cpu, hc]] = default;

            // MANIPULATORS
            unorm& operator=(const unorm&) [[cpu, hc]] = default;
            unorm& operator=(unorm&&) [[cpu, hc]] = default;
            unorm& operator+=(const unorm& x) noexcept [[cpu, hc]]
            {
                return *this = unorm{x_ + x.x_};
            }
            unorm& operator-=(const unorm& x) noexcept [[cpu, hc]]
            {
                return *this = unorm{x_ - x.x_};
            }
            unorm& operator*=(const unorm& x) noexcept [[cpu, hc]]
            {
                return *this = unorm{x_ * x.x_};
            }
            unorm& operator/=(const unorm& x) [[cpu, hc]]
            {
                return *this = unorm{x_ / x.x_};
            }
            unorm& operator++() noexcept [[cpu, hc]]
            {
                return *this = unorm{++x_};
            }
            unorm operator++(int) noexcept [[cpu, hc]]
            {
                unorm tmp{*this};
                ++*this;
                return tmp;
            }
            unorm& operator--() noexcept [[cpu, hc]]
            {
                return *this = unorm{--x_};
            }
            unorm operator--(int) noexcept [[cpu, hc]]
            {
                unorm tmp{*this};
                --*this;
                return tmp;
            }

            // ACCESSORS
            constexpr
            operator float() const noexcept [[cpu, hc]] { return x_; }
        };

        // TODO: use levelisation to fix the weird late definition.
        constexpr
        inline
        norm::norm(const unorm& x) noexcept [[cpu, hc]] : x_{x.x_} {}

        static constexpr unorm UNORM_MAX{1.0f};
        static constexpr unorm UNORM_MIN{0.0f};
        static constexpr unorm UNORM_ZERO{0.0f};
    } // Namespace hc::short_vector.
} // Namespace hc.

namespace std
{   // TODO: add additional specialisations.
    template<>
    struct is_unsigned<hc::short_vector::unorm> : public std::true_type {};
}