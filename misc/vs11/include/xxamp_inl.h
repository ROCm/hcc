/***
* ==++==
*
* Copyright (c) Microsoft Corporation.  All rights reserved.
*
* ==--==
* =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
*
* xxamp_inl.h
*
* C++ AMP Library helper classes implementations.
*
* This file contains the bodies of medthods declared in xxamp which rely on
* amp.h class defintions.
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/

#pragma once

#include <xxamp.h>


namespace Concurrency
{
namespace details
{

// Projection Helpers

template <typename _T, int _R>
/*static*/ array_view<_T,_R-1> _Array_view_projection_helper<_T,_R>::_Project0(const array_view<_T,_R>* _Arr_view, int _I) __GPU
{
    array_view<_T,_R-1> _Projected_view;
    _Arr_view->_Project0(_I, _Projected_view);
    return _Projected_view;
}

template <typename _T>
/*static*/ _T& _Array_view_projection_helper<_T,1>::_Project0(const array_view<_T,1>* _Arr_view, int _I) __GPU
{
    return _Arr_view->operator[](index<1>(_I));
}

template <typename _T, int _R>
/*static*/ array_view<const _T,_R-1> _Array_projection_helper<_T,_R>::_Project0(const array<_T, _R>* _Array, int _I) __GPU
{
    array_view<const _T,_R> _Temp(*_Array);
	array_view<const _T,_R-1> _Temp2 = _Array_view_projection_helper<const _T,_R>::_Project0(&_Temp, _I);
    return _Temp2;
}

template <typename _T, int _R>
/*static*/ array_view<_T,_R-1> _Array_projection_helper<_T,_R>::_Project0(_In_ array<_T, _R>* _Array, int _I) __GPU
{
    array_view<_T,_R> _Temp(*_Array);
	array_view<_T,_R-1> _Temp2 = _Array_view_projection_helper<_T,_R>::_Project0(&_Temp, _I);
    return _Temp2;
}

template <typename _T>
/*static*/ const _T& _Array_projection_helper<_T,1>::_Project0(const array<_T,1>* _Array, int _I) __GPU
{
    return _Array->operator[](index<1>(_I));
}

template <typename _T>
/*static*/ _T& _Array_projection_helper<_T,1>::_Project0(_In_ array<_T,1>* _Array, int _I) __GPU
{
    return _Array->operator[](index<1>(_I));
}

} // namespace details
} // namespace concurrency
