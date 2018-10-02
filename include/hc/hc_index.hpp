//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

namespace hc
{
    template<int> class extent;

    namespace detail
    {
        /** \cond HIDDEN_SYMBOLS */
        template <int...> struct __indices {};

        template <int _Sp, class _IntTuple, int _Ep>
        struct __make_indices_imp;

        template <int _Sp, int ..._Indices, int _Ep>
        struct __make_indices_imp<_Sp, __indices<_Indices...>, _Ep> {
            using type = typename __make_indices_imp<
                _Sp+1, __indices<_Indices..., _Sp>, _Ep>::type;
        };

        template <int _Ep, int ..._Indices>
        struct __make_indices_imp<_Ep, __indices<_Indices...>, _Ep> {
            typedef __indices<_Indices...> type;
        };

        template <int _Ep, int _Sp = 0>
        struct __make_indices {
            static_assert(_Sp <= _Ep, "__make_indices input error");
            using type =
                typename __make_indices_imp<_Sp, __indices<>, _Ep>::type;
        };

        template <int _Ip>
        class __index_leaf {
            int __idx;
            int dummy;
        public:
            explicit
            __index_leaf(int __t) noexcept [[cpu, hc]] : __idx(__t) {}

            __index_leaf& operator=(const int __t) noexcept [[cpu, hc]]
            {
                __idx = __t;
                return *this;
            }
            __index_leaf& operator+=(const int __t) noexcept [[cpu, hc]]
            {
                __idx += __t;
                return *this;
            }
            __index_leaf& operator-=(const int __t) noexcept [[cpu, hc]]
            {
                __idx -= __t;
                return *this;
            }
            __index_leaf& operator*=(const int __t) noexcept [[cpu, hc]]
            {
                __idx *= __t;
                return *this;
            }
            __index_leaf& operator/=(const int __t) noexcept [[cpu, hc]]
            {
                __idx /= __t;
                return *this;
            }
            __index_leaf& operator%=(const int __t) noexcept [[cpu, hc]]
            {
                __idx %= __t;
                return *this;
            }
            int& get() noexcept [[cpu, hc]] { return __idx; }
            const int& get() const noexcept [[cpu, hc]] { return __idx; }
        };

        template <class _Indx> struct index_impl;

        template <int ...N>
        struct index_impl<__indices<N...> > : public __index_leaf<N>...  {
            index_impl() [[cpu, hc]] : __index_leaf<N>(0)... {}

            template<class ..._Up>
                explicit
                index_impl(_Up... __u) [[cpu, hc]]
                    : __index_leaf<N>(__u)... {}

            index_impl(const index_impl& other) [[cpu, hc]]
                :
                index_impl(static_cast<const __index_leaf<N>&>(other).get()...)
            {}

            index_impl(int component) [[cpu, hc]]
                : __index_leaf<N>(component)... {}
            index_impl(int components[]) [[cpu, hc]]
                : __index_leaf<N>(components[N])... {}
            index_impl(const int components[]) [[cpu, hc]]
                : __index_leaf<N>(components[N])... {}

            template<class ..._Tp>
            inline
            void __swallow(_Tp...) [[cpu, hc]] {}

            int operator[](unsigned int c) const [[cpu, hc]]
            {
                return static_cast<const __index_leaf<0>&>(
                    *((__index_leaf<0> *)this + c)).get();
            }
            int& operator[](unsigned int c) [[cpu, hc]]
            {
                return static_cast<__index_leaf<0>&>(
                    *((__index_leaf<0> *)this + c)).get();
            }
            index_impl& operator=(const index_impl& __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator=(
                    static_cast<const __index_leaf<N>&>(__t).get())...);
                return *this;
            }
            index_impl& operator+=(const index_impl& __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator+=(
                    static_cast<const __index_leaf<N>&>(__t).get())...);
                return *this;
            }
            index_impl& operator-=(const index_impl& __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator-=(
                    static_cast<const __index_leaf<N>&>(__t).get())...);
                return *this;
            }
            index_impl& operator*=(const index_impl& __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator*=(
                    static_cast<const __index_leaf<N>&>(__t).get())...);
                return *this;
            }
            index_impl& operator/=(const index_impl& __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator/=(
                    static_cast<const __index_leaf<N>&>(__t).get())...);
                return *this;
            }
            index_impl& operator%=(const index_impl& __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator%=(
                    static_cast<const __index_leaf<N>&>(__t).get())...);
                return *this;
            }
            index_impl& operator+=(const int __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator+=(__t)...);
                return *this;
            }
            index_impl& operator-=(const int __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator-=(__t)...);
                return *this;
            }
            index_impl& operator*=(const int __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator*=(__t)...);
                return *this;
            }
            index_impl& operator/=(const int __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator/=(__t)...);
                return *this;
            }
            index_impl& operator%=(const int __t) [[cpu, hc]]
            {
                __swallow(__index_leaf<N>::operator%=(__t)...);
                return *this;
            }
        };

        template<int N, typename _Tp>
        struct index_helper {
            static
            inline
            void set(_Tp& now) [[cpu, hc]]
            {
                now[N - 1] = hc_get_global_id(_Tp::rank - N);
                index_helper<N - 1, _Tp>::set(now);
            }
            static
            inline
            bool equal(const _Tp& _lhs, const _Tp& _rhs) [[cpu, hc]]
            {
                return (_lhs[N - 1] == _rhs[N - 1]) &&
                    (index_helper<N - 1, _Tp>::equal(_lhs, _rhs));
            }
            static
            inline
            int count_size(const _Tp& now) [[cpu, hc]]
            {
                return now[N - 1] * index_helper<N - 1, _Tp>::count_size(now);
            }
        };

        template<typename _Tp>
        struct index_helper<1, _Tp> {
            static
            inline
            void set(_Tp& now) [[cpu, hc]]
            {
                now[0] = hc_get_global_id(_Tp::rank - 1);
            }
            static
            inline
            bool equal(const _Tp& _lhs, const _Tp& _rhs) [[cpu, hc]]
            {
                return (_lhs[0] == _rhs[0]);
            }
            static
            inline
            int count_size(const _Tp& now) [[cpu, hc]]
            {
                return now[0];
            }
        };

        template<int N, typename T, typename U>
        struct amp_helper {
            static
            bool
            inline contains(const T& idx, const U& ext) [[cpu, hc]]
            {
                return idx[N - 1] >= 0 && idx[N - 1] < ext[N - 1] &&
                    amp_helper<N - 1, T, U>::contains(idx, ext);
            }

            static
            bool
            inline contains(
                const T& idx, const U& ext,const U& ext2) [[cpu, hc]]
            {
                return idx[N - 1] >= 0 &&
                    ext[N - 1] > 0 &&
                    (idx[N - 1] + ext[N - 1]) <= ext2[N - 1] &&
                    amp_helper<N - 1, T, U>::contains(idx, ext, ext2);
            }

            static
            inline
            int flatten(const T& idx, const U& ext) [[cpu, hc]]
            {
                return idx[N - 1] +
                    ext[N - 1] * amp_helper<N - 1, T, U>::flatten(idx, ext);
            }
            static
            inline
            void minus(const T& idx, U& ext) [[cpu, hc]]
            {
                ext.base_ -= idx.base_;
            }
        };

        template<typename T, typename U>
        struct amp_helper<1, T, U> {
            static
            inline
            bool contains(const T& idx, const U& ext) [[cpu, hc]]
            {
                return idx[0] >= 0 && idx[0] < ext[0];
            }

            static
            inline
            bool contains(const T& idx, const U& ext,const U& ext2) [[cpu, hc]]
            {
                return
                    idx[0] >= 0 && ext[0] > 0 && (idx[0] + ext[0]) <= ext2[0];
            }

            static
            inline
            int flatten(const T& idx, const U&) [[cpu, hc]]
            {
                return idx[0];
            }
            static
            inline
            void minus(const T& idx, U& ext) [[cpu, hc]]
            {
                ext.base_ -= idx.base_;
            }
        };
        /** \endcond */

        /**
         * Represents a unique position in N-dimensional space.
         *
         * @tparam N The dimensionality space into which this index applies.
         *           Special constructors are supplied for the cases where
         *           @f$N \in \{1,2,3\}@f$, but N can be any integer greater
         *           than 0.
         */
        template<int N>
        class index {
            static_assert(N > 0, "rank should greater than 0.");

            using base = index_impl<typename __make_indices<N>::type>;
            base base_;

            template<int> friend class hc::extent;
            template<int, typename> friend struct index_helper;
            template<int, typename, typename> friend struct amp_helper;
        public:
            /**
             * A static member of index<N> that contains the rank of this index.
             */
            static constexpr int rank = N;

            /**
             * The element type of index<N>.
             */
            using value_type = int;

            /**
             * Default constructor. The value at each dimension is initialized
             * to zero. Thus, "index<3> ix;" initializes the variable to the
             * position (0,0,0).
             */
            index() [[cpu, hc]] = default;

            /**
             * Copy constructor. Constructs a new index<N> from the supplied
             * argument "other".
             *
             * @param[in] other An object of type index<N> from which to
             *                  initialize this new index.
             */
            index(const index&) [[cpu, hc]] = default;
            index(index&&) [[cpu, hc]] = default;

            /** @{ */
            /**
             * Constructs an index<N> with the coordinate values provided by
             * @f$i_{0..2}@f$. These are specialized constructors that are only
             * valid when the rank of the index @f$N \in \{1,2,3\}@f$. Invoking
             * a specialized constructor whose argument @f$count \ne N@f$ will
             * result in a compilation error.
             *
             * @param[in] i0 The component values of the index vector.
             */
            template<
                typename... Ts,
                typename std::enable_if<sizeof...(Ts) == N>::type* = nullptr>
            explicit
            index(Ts... i_n) [[cpu, hc]] : base_{static_cast<int>(i_n)...}
            {
                static_assert(
                    sizeof...(Ts) <= 3,
                    "Explicit constructor with rank greater than 3 is not "
                        "allowed");
            }

            /** @} */

            /**
             * Constructs an index<N> with the coordinate values provided the
             * array of int component values. If the coordinate array length
             * @f$\ne@f$ N, the behavior is undefined. If the array value is
             * NULL or not a valid pointer, the behavior is undefined.
             *
             * @param[in] components An array of N int values.
             */
            explicit
            index(const int components[]) [[cpu, hc]] : base_{components} {}

            /**
             * Assigns the component values of "other" to this index<N> object.
             *
             * @param[in] other An object of type index<N> from which to copy
             *                  into this index.
             * @return Returns *this.
             */
            index& operator=(const index&) [[cpu, hc]] = default;
            index& operator=(index&&) [[cpu, hc]] = default;

            /** @{ */
            /**
             * Returns the index component value at position c.
             *
             * @param[in] c The dimension axis whose coordinate is to be
             *              accessed.
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

            /** @{ */
            /**
             * Compares two objects of index<N>.
             *
             * The expression
             * @f$leftIdx \oplus rightIdx@f$
             * is true if @f$leftIdx[i] \oplus rightIdx[i]@f$ for every i from 0
             * to N-1.
             *
             * @param[in] other The right-hand index<N> to be compared.
             */
            // FIXME: the signature is not entirely the same as defined in:
            //        C++AMP spec v1.2 #1137
            bool operator==(const index& other) const [[cpu, hc]]
            {
                return index_helper<N, index<N> >::equal(*this, other);
            }
            bool operator!=(const index& other) const [[cpu, hc]]
            {
                return !(*this == other);
            }

            /** @} */

            /** @{ */
            /**
             * For a given operator @f$\oplus@f$, produces the same effect as
             * (*this) = (*this) @f$\oplus@f$ rhs;
             * The return value is "*this".
             *
             * @param[in] rhs The right-hand index<N> of the arithmetic
             *                operation.
             */
            index& operator+=(const index& rhs) [[cpu, hc]]
            {
                base_ += rhs.base_;
                return *this;
            }
            index& operator-=(const index& rhs) [[cpu, hc]]
            {
                base_ -= rhs.base_;
                return *this;
            }

            /** @} */

            /** @{ */
            /**
             * For a given operator @f$\oplus@f$, produces the same effect as
             * (*this) = (*this) @f$\oplus@f$ value;
             * The return value is "*this".
             *
             * @param[in] value The right-hand int of the arithmetic operation.
             */
            index& operator+=(int value) [[cpu, hc]]
            {
                base_  += value;
                return *this;
            }
            index& operator-=(int value) [[cpu, hc]]
            {
                base_ -= value;
                return *this;
            }
            index& operator*=(int value) [[cpu, hc]]
            {
                base_ *= value;
                return *this;
            }
            index& operator/=(int value) [[cpu, hc]]
            {
                base_ /= value;
                return *this;
            }
            index& operator%=(int value) [[cpu, hc]]
            {
                base_ %= value;
                return *this;
            }

            /** @} */

            /** @{ */
            /**
             * For a given operator @f$\oplus@f$, produces the same effect as
             * (*this) = (*this) @f$\oplus@f$ 1;
             *
             * For prefix increment and decrement, the return value is "*this".
             * Otherwise a new index<N> is returned.
             */
            index& operator++() [[cpu, hc]]
            {
                return *this += 1;
            }
            index operator++(int) [[cpu, hc]]
            {
                index ret = *this;
                ++*this;
                return ret;
            }
            index& operator--() [[cpu, hc]]
            {
                return *this -= 1;
            }
            index operator--(int) [[cpu, hc]]
            {
                index ret = *this;
                --*this;
                return ret;
            }

            /** @} */
        };


        ////////////////////////////////////////////////////////////////////////
        // operators for index<N>
        ////////////////////////////////////////////////////////////////////////

        /** @{ */
        /**
         * Binary arithmetic operations that produce a new index<N> that is the
         * result of performing the corresponding pair-wise binary arithmetic
         * operation on the elements of the operands. The result index<N> is
         * such that for a given operator @f$\oplus@f$,
         * @f$result[i] = leftIdx[i] \oplus rightIdx[i]@f$
         * for every i from 0 to N-1.
         *
         * @param[in] lhs The left-hand index<N> of the arithmetic operation.
         * @param[in] rhs The right-hand index<N> of the arithmetic operation.
         */
        // FIXME: the signature is not entirely the same as defined in:
        //        C++AMP spec v1.2 #1138
        template<int N>
        index<N> operator+(const index<N>& lhs, const index<N>& rhs) [[cpu, hc]]
        {
            index<N> __r = lhs;
            __r += rhs;
            return __r;
        }
        template<int N>
        index<N> operator-(const index<N>& lhs, const index<N>& rhs) [[cpu, hc]]
        {
            index<N> __r = lhs;
            __r -= rhs;
            return __r;
        }

        /** @} */

        /** @{ */
        /**
         * Binary arithmetic operations that produce a new index<N> that is the
         * result of performing the corresponding binary arithmetic operation on
         * the elements of the index operands. The result index<N> is such that
         * for a given operator @f$\oplus@f$,
         * result[i] = idx[i] @f$\oplus@f$ value
         * or
         * result[i] = value @f$\oplus@f$ idx[i]
         * for every i from 0 to N-1.
         *
         * @param[in] idx The index<N> operand
         * @param[in] value The integer operand
         */
        // FIXME: the signature is not entirely the same as defined in:
        //        C++AMP spec v1.2 #1141
        template<int N>
        index<N> operator+(const index<N>& idx, int value) [[cpu, hc]]
        {
            index<N> __r = idx;
            __r += value;
            return __r;
        }
        template<int N>
        index<N> operator+(int value, const index<N>& idx) [[cpu, hc]]
        {
            index<N> __r = idx;
            __r += value;
            return __r;
        }
        template<int N>
        index<N> operator-(const index<N>& idx, int value) [[cpu, hc]]
        {
            index<N> __r = idx;
            __r -= value;
            return __r;
        }
        template<int N>
        index<N> operator-(int value, const index<N>& idx) [[cpu, hc]]
        {
            index<N> __r(value);
            __r -= idx;
            return __r;
        }
        template<int N>
        index<N> operator*(const index<N>& idx, int value) [[cpu, hc]]
        {
            index<N> __r = idx;
            __r *= value;
            return __r;
        }
        template<int N>
        index<N> operator*(int value, const index<N>& idx) [[cpu, hc]]
        {
            index<N> __r(value);
            __r *= idx;
            return __r;
        }
        template<int N>
        index<N> operator/(const index<N>& idx, int value) [[cpu, hc]]
        {
            index<N> __r = idx;
            __r /= value;
            return __r;
        }
        template<int N>
        index<N> operator/(int value, const index<N>& idx) [[cpu, hc]]
        {
            index<N> __r(value);
            __r /= idx;
            return __r;
        }
        template<int N>
        index<N> operator%(const index<N>& idx, int value) [[cpu, hc]]
        {
            index<N> __r = idx;
            __r %= value;
            return __r;
        }
        template<int N>
        index<N> operator%(int value, const index<N>& idx) [[cpu, hc]]
        {
            index<N> __r(value);
            __r %= idx;
            return __r;
        }

        /** @} */
    } // Namespace hc::detail.
} // Namespace hc.