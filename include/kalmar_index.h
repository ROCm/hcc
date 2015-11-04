#pragma once

//forward declaration
namespace Concurrency {
template <int N> class extent;
} // namespace Concurrency

//forward declaration
namespace hc {
template <int N> class extent;
} // namespace hc

namespace Kalmar {

/** \cond HIDDEN_SYMBOLS */
template <int...> struct __indices {};

template <int _Sp, class _IntTuple, int _Ep>
struct __make_indices_imp;

template <int _Sp, int ..._Indices, int _Ep>
struct __make_indices_imp<_Sp, __indices<_Indices...>, _Ep> {
    typedef typename __make_indices_imp<_Sp+1, __indices<_Indices..., _Sp>, _Ep>::type type;
};

template <int _Ep, int ..._Indices>
struct __make_indices_imp<_Ep, __indices<_Indices...>, _Ep> {
    typedef __indices<_Indices...> type;
};

template <int _Ep, int _Sp = 0>
struct __make_indices {
    static_assert(_Sp <= _Ep, "__make_indices input error");
    typedef typename __make_indices_imp<_Sp, __indices<>, _Ep>::type type;
};

template <int _Ip>
class __index_leaf {
    int __idx;
    int dummy;
public:
    explicit __index_leaf(int __t) restrict(amp,cpu) : __idx(__t) {}

    __index_leaf& operator=(const int __t) restrict(amp,cpu) {
        __idx = __t;
        return *this;
    }
    __index_leaf& operator+=(const int __t) restrict(amp,cpu) {
        __idx += __t;
        return *this;
    }
    __index_leaf& operator-=(const int __t) restrict(amp,cpu) {
        __idx -= __t;
        return *this;
    }
    __index_leaf& operator*=(const int __t) restrict(amp,cpu) {
        __idx *= __t;
        return *this;
    }
    __index_leaf& operator/=(const int __t) restrict(amp,cpu) {
        __idx /= __t;
        return *this;
    }
    __index_leaf& operator%=(const int __t) restrict(amp,cpu) {
        __idx %= __t;
        return *this;
    }
          int& get()       restrict(amp,cpu) { return __idx; }
    const int& get() const restrict(amp,cpu) { return __idx; }
};

template <class _Indx> struct index_impl;

template <int ...N>
struct index_impl<__indices<N...> > : public __index_leaf<N>...  {
    index_impl() restrict(amp,cpu) : __index_leaf<N>(0)... {}

    template<class ..._Up>
        explicit index_impl(_Up... __u) restrict(amp,cpu)
        : __index_leaf<N>(__u)... {}

    index_impl(const index_impl& other) restrict(amp,cpu)
        : index_impl(static_cast<const __index_leaf<N>&>(other).get()...) {}

    index_impl(int component) restrict(amp,cpu)
        : __index_leaf<N>(component)... {}
    index_impl(int components[]) restrict(amp,cpu)
        : __index_leaf<N>(components[N])... {}
    index_impl(const int components[]) restrict(amp,cpu)
        : __index_leaf<N>(components[N])... {}

    template<class ..._Tp>
        inline void __swallow(_Tp...) restrict(amp,cpu) {}

    int operator[] (unsigned int c) const restrict(amp,cpu) {
        return static_cast<const __index_leaf<0>&>(*((__index_leaf<0> *)this + c)).get();
    }
    int& operator[] (unsigned int c) restrict(amp,cpu) {
        return static_cast<__index_leaf<0>&>(*((__index_leaf<0> *)this + c)).get();
    }
    index_impl& operator=(const index_impl& __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator=(static_cast<const __index_leaf<N>&>(__t).get())...);
        return *this;
    }
    index_impl& operator+=(const index_impl& __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator+=(static_cast<const __index_leaf<N>&>(__t).get())...);
        return *this;
    }
    index_impl& operator-=(const index_impl& __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator-=(static_cast<const __index_leaf<N>&>(__t).get())...);
        return *this;
    }
    index_impl& operator*=(const index_impl& __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator*=(static_cast<const __index_leaf<N>&>(__t).get())...);
        return *this;
    }
    index_impl& operator/=(const index_impl& __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator/=(static_cast<const __index_leaf<N>&>(__t).get())...);
        return *this;
    }
    index_impl& operator%=(const index_impl& __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator%=(static_cast<const __index_leaf<N>&>(__t).get())...);
        return *this;
    }
    index_impl& operator+=(const int __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator+=(__t)...);
        return *this;
    }
    index_impl& operator-=(const int __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator-=(__t)...);
        return *this;
    }
    index_impl& operator*=(const int __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator*=(__t)...);
        return *this;
    }
    index_impl& operator/=(const int __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator/=(__t)...);
        return *this;
    }
    index_impl& operator%=(const int __t) restrict(amp,cpu) {
        __swallow(__index_leaf<N>::operator%=(__t)...);
        return *this;
    }
};

template <int N, typename _Tp>
struct index_helper
{
    static inline void set(_Tp& now) restrict(amp,cpu) {
        now[N - 1] = amp_get_global_id(_Tp::rank - N);
        index_helper<N - 1, _Tp>::set(now);
    }
    static inline bool equal(const _Tp& _lhs, const _Tp& _rhs) restrict(amp,cpu) {
        return (_lhs[N - 1] == _rhs[N - 1]) &&
            (index_helper<N - 1, _Tp>::equal(_lhs, _rhs));
    }
    static inline int count_size(const _Tp& now) restrict(amp,cpu) {
        return now[N - 1] * index_helper<N - 1, _Tp>::count_size(now);
    }
};

template<typename _Tp>
struct index_helper<1, _Tp>
{
    static inline void set(_Tp& now) restrict(amp,cpu) {
        now[0] = amp_get_global_id(_Tp::rank - 1);
    }
    static inline bool equal(const _Tp& _lhs, const _Tp& _rhs) restrict(amp,cpu) {
        return (_lhs[0] == _rhs[0]);
    }
    static inline int count_size(const _Tp& now) restrict(amp,cpu) {
        return now[0];
    }
};

template <int N, typename _Tp1, typename _Tp2>
struct amp_helper
{
    static bool inline contains(const _Tp1& idx, const _Tp2& ext) restrict(amp,cpu) {
        return idx[N - 1] >= 0 && idx[N - 1] < ext[N - 1] &&
            amp_helper<N - 1, _Tp1, _Tp2>::contains(idx, ext);
    }

    static bool inline contains(const _Tp1& idx, const _Tp2& ext,const _Tp2& ext2) restrict(amp,cpu) {
        return idx[N - 1] >= 0 && ext[N - 1] > 0 && (idx[N - 1] + ext[N - 1]) <= ext2[N - 1] &&
            amp_helper<N - 1, _Tp1, _Tp2>::contains(idx, ext,ext2);
    }

    static int inline flatten(const _Tp1& idx, const _Tp2& ext) restrict(amp,cpu) {
        return idx[N - 1] + ext[N - 1] * amp_helper<N - 1, _Tp1, _Tp2>::flatten(idx, ext);
    }
    static void inline minus(const _Tp1& idx, _Tp2& ext) restrict(amp,cpu) {
        ext.base_ -= idx.base_;
    }
};

template <typename _Tp1, typename _Tp2>
struct amp_helper<1, _Tp1, _Tp2>
{
    static bool inline contains(const _Tp1& idx, const _Tp2& ext) restrict(amp,cpu) {
        return idx[0] >= 0 && idx[0] < ext[0];
    }

    static bool inline contains(const _Tp1& idx, const _Tp2& ext,const _Tp2& ext2) restrict(amp,cpu) {
        return idx[0] >= 0 && ext[0] > 0 && (idx[0] + ext[0]) <= ext2[0] ;
    }

    static int inline flatten(const _Tp1& idx, const _Tp2& ext) restrict(amp,cpu) {
        return idx[0];
    }
    static void inline minus(const _Tp1& idx, _Tp2& ext) restrict(amp,cpu) {
        ext.base_ -= idx.base_;
    }
};
/** \endcond */

/**
 * Represents a unique position in N-dimensional space.
 *
 * @tparam N The dimensionality space into which this index applies. Special
 *           constructors are supplied for the cases where @f$N \in \{1,2,3\}@f$,
 *           but N can be any integer greater than 0.
 */
template <int N>
class index {
public:
    /**
     * A static member of index<N> that contains the rank of this index.
     */
    static const int rank = N;

    /**
     * The element type of index<N>.
     */
    typedef int value_type;

    /**
     * Default constructor. The value at each dimension is initialized to zero.
     * Thus, "index<3> ix;" initializes the variable to the position (0,0,0).
     */
    index() restrict(amp,cpu) : base_() {
        static_assert( N>0, "rank should bigger than 0 ");
    };

    /**
     * Copy constructor. Constructs a new index<N> from the supplied argument
     * "other".
     *
     * @param[in] other An object of type index<N> from which to initialize
     *                  this new index.
     */
    index(const index& other) restrict(amp,cpu)
        : base_(other.base_) {}

    /** @{ */
    /**
     * Constructs an index<N> with the coordinate values provided by @f$i_{0..2}@f$.
     * These are specialized constructors that are only valid when the rank of
     * the index @f$N \in \{1,2,3\}@f$. Invoking a specialized constructor whose argument
     * @f$count \ne N@f$ will result in a compilation error.
     *
     * @param[in] i0 The component values of the index vector.
     */
    explicit index(int i0) restrict(amp,cpu)
        : base_(i0) {}

    template <typename ..._Tp>
        explicit index(_Tp ... __t) restrict(amp,cpu)
        : base_(__t...) {
            static_assert(sizeof...(_Tp) <= 3, "Explicit constructor with rank greater than 3 is not allowed");
            static_assert(sizeof...(_Tp) == N, "rank should be consistency");
        }

    /** @} */

    /**
     * Constructs an index<N> with the coordinate values provided the array of
     * int component values. If the coordinate array length @f$\ne@f$ N, the
     * behavior is undefined. If the array value is NULL or not a valid
     * pointer, the behavior is undefined.
     *
     * @param[in] components An array of N int values.
     */
    explicit index(const int components[]) restrict(amp,cpu)
        : base_(components) {}

    /**
     * Constructs an index<N> with the coordinate values provided the array of
     * int component values. If the coordinate array length @f$\ne@f$ N, the
     * behavior is undefined. If the array value is NULL or not a valid
     * pointer, the behavior is undefined.
     *
     * @param[in] components An array of N int values.
     */
    // FIXME: this function is not defined in C++AMP specification.
    explicit index(int components[]) restrict(amp,cpu)
        : base_(components) {}

    /**
     * Assigns the component values of "other" to this index<N> object.
     *
     * @param[in] other An object of type index<N> from which to copy into this
     *                  index.
     * @return Returns *this.
     */
    index& operator=(const index& other) restrict(amp,cpu) {
        base_.operator=(other.base_);
        return *this;
    }

    /** @{ */
    /**
     * Returns the index component value at position c.
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

    /** @{ */
    /**
     * Compares two objects of index<N>.
     *
     * The expression
     * @f$leftIdx \oplus rightIdx@f$
     * is true if @f$leftIdx[i] \oplus rightIdx[i]@f$ for every i from 0 to N-1.
     *
     * @param[in] other The right-hand index<N> to be compared.
     */
    // FIXME: the signature is not entirely the same as defined in:
    //        C++AMP spec v1.2 #1137
    bool operator== (const index& other) const restrict(amp,cpu) {
        return index_helper<N, index<N> >::equal(*this, other);
    }
    bool operator!= (const index& other) const restrict(amp,cpu) {
        return !(*this == other);
    }

    /** @} */

    /** @{ */
    /**
     * For a given operator @f$\oplus@f$, produces the same effect as
     * (*this) = (*this) @f$\oplus@f$ rhs;
     * The return value is "*this".
     *
     * @param[in] rhs The right-hand index<N> of the arithmetic operation.
     */
    index& operator+=(const index& rhs) restrict(amp,cpu) {
        base_.operator+=(rhs.base_);
        return *this;
    }
    index& operator-=(const index& rhs) restrict(amp,cpu) {
        base_.operator-=(rhs.base_);
        return *this;
    }

    // FIXME: this function is not defined in C++AMP specification.
    index& operator*=(const index& __r) restrict(amp,cpu) {
        base_.operator*=(__r.base_);
        return *this;
    }
    // FIXME: this function is not defined in C++AMP specification.
    index& operator/=(const index& __r) restrict(amp,cpu) {
        base_.operator/=(__r.base_);
        return *this;
    }
    // FIXME: this function is not defined in C++AMP specification.
    index& operator%=(const index& __r) restrict(amp,cpu) {
        base_.operator%=(__r.base_);
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
    index& operator+=(int value) restrict(amp,cpu) {
        base_.operator+=(value);
        return *this;
    }
    index& operator-=(int value) restrict(amp,cpu) {
        base_.operator-=(value);
        return *this;
    }
    index& operator*=(int value) restrict(amp,cpu) {
        base_.operator*=(value);
        return *this;
    }
    index& operator/=(int value) restrict(amp,cpu) {
        base_.operator/=(value);
        return *this;
    }
    index& operator%=(int value) restrict(amp,cpu) {
        base_.operator%=(value);
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
    index& operator++() restrict(amp,cpu) {
        base_.operator+=(1);
        return *this;
    }
    index operator++(int) restrict(amp,cpu) {
        index ret = *this;
        base_.operator+=(1);
        return ret;
    }
    index& operator--() restrict(amp,cpu) {
        base_.operator-=(1);
        return *this;
    }
    index operator--(int) restrict(amp,cpu) {
        index ret = *this;
        base_.operator-=(1);
        return ret;
    }

    /** @} */

private:
    typedef index_impl<typename __make_indices<N>::type> base;
    base base_;
    template <int T> friend class Concurrency::extent;
    template <int T> friend class hc::extent;
    template <int K, typename Q> friend struct index_helper;
    template <int K, typename Q1, typename Q2> friend struct amp_helper;

public:
    __attribute__((annotate("__cxxamp_opencl_index")))
    void __cxxamp_opencl_index() restrict(amp,cpu)
#if __KALMAR_ACCELERATOR__ == 1
    {
        index_helper<N, index<N>>::set(*this);
    }
#elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
    {}
#else
    ;
#endif
};

///////////////////////////////////////////////////////////////////////////////
// explicit instantions
///////////////////////////////////////////////////////////////////////////////
template class index<1>;
template class index<2>;
template class index<3>;

///////////////////////////////////////////////////////////////////////////////
// operators for index<N>
///////////////////////////////////////////////////////////////////////////////

/** @{ */
/**
 * Binary arithmetic operations that produce a new index<N> that is the result
 * of performing the corresponding pair-wise binary arithmetic operation on the
 * elements of the operands. The result index<N> is such that for a given
 * operator @f$\oplus@f$,
 * @f$result[i] = leftIdx[i] \oplus rightIdx[i]@f$
 * for every i from 0 to N-1.
 *
 * @param[in] lhs The left-hand index<N> of the arithmetic operation.
 * @param[in] rhs The right-hand index<N> of the arithmetic operation.
 */
// FIXME: the signature is not entirely the same as defined in:
//        C++AMP spec v1.2 #1138
template <int N>
index<N> operator+(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r += rhs;
    return __r;
}
template <int N>
index<N> operator-(const index<N>& lhs, const index<N>& rhs) restrict(amp,cpu) {
    index<N> __r = lhs;
    __r -= rhs;
    return __r;
}

/** @} */

/** @{ */
/**
 * Binary arithmetic operations that produce a new index<N> that is the result
 * of performing the corresponding binary arithmetic operation on the elements
 * of the index operands. The result index<N> is such that for a given
 * operator @f$\oplus@f$,
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
template <int N>
index<N> operator+(const index<N>& idx, int value) restrict(amp,cpu) {
    index<N> __r = idx;
    __r += value;
    return __r;
}
template <int N>
index<N> operator+(int value, const index<N>& idx) restrict(amp,cpu) {
    index<N> __r = idx;
    __r += value;
    return __r;
}
template <int N>
index<N> operator-(const index<N>& idx, int value) restrict(amp,cpu) {
    index<N> __r = idx;
    __r -= value;
    return __r;
}
template <int N>
index<N> operator-(int value, const index<N>& idx) restrict(amp,cpu) {
    index<N> __r(value);
    __r -= idx;
    return __r;
}
template <int N>
index<N> operator*(const index<N>& idx, int value) restrict(amp,cpu) {
    index<N> __r = idx;
    __r *= value;
    return __r;
}
template <int N>
index<N> operator*(int value, const index<N>& idx) restrict(amp,cpu) {
    index<N> __r(value);
    __r *= idx;
    return __r;
}
template <int N>
index<N> operator/(const index<N>& idx, int value) restrict(amp,cpu) {
    index<N> __r = idx;
    __r /= value;
    return __r;
}
template <int N>
index<N> operator/(int value, const index<N>& idx) restrict(amp,cpu) {
    index<N> __r(value);
    __r /= idx;
    return __r;
}
template <int N>
index<N> operator%(const index<N>& idx, int value) restrict(amp,cpu) {
    index<N> __r = idx;
    __r %= value;
    return __r;
}
template <int N>
index<N> operator%(int value, const index<N>& idx) restrict(amp,cpu) {
    index<N> __r(value);
    __r %= idx;
    return __r;
}

/** @} */


} // namespace Kalmar

