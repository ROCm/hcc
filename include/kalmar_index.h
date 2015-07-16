#pragma once

namespace Kalmar {

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

} // namespace Kalmar

