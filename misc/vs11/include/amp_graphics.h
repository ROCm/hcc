/***
* ==++==
*
* Copyright (c) Microsoft Corporation.  All rights reserved.
*
* ==--==
* =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
*
* amp_graphics.h
*
* C++ AMP Graphics Library
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/

#pragma once

#include <amp_short_vectors.h>
#include <vector>

#define _AMP_GRAPHICS_H

namespace Concurrency
{

namespace graphics
{

namespace details
{

#pragma warning( push )
#pragma warning( disable : 6326 ) // Potential comparison of a constant with another constant

template<typename _Ty>
struct _Short_vector_type_traits
{
    typedef void _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = false;
    static const _Short_vector_base_type_id _Format_base_type_id = _Invalid_type;
    static const unsigned int _Num_channels = 0;
    static const unsigned int _Default_bits_per_channel = 0;
};

template<>
struct _Short_vector_type_traits<unsigned int>
{
    typedef unsigned int _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Uint_type;
    static const unsigned int _Num_channels = 1;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<uint_2>
{
    typedef uint_2::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
   static const _Short_vector_base_type_id _Format_base_type_id = _Uint_type;
    static const unsigned int _Num_channels = 2;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<uint_3>
{
    typedef uint_3::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = false;
    static const _Short_vector_base_type_id _Format_base_type_id = _Invalid_type;
    static const unsigned int _Num_channels = 0;
    static const unsigned int _Default_bits_per_channel = 0;
};

template<>
struct _Short_vector_type_traits<uint_4>
{
    typedef uint_4::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
   static const _Short_vector_base_type_id _Format_base_type_id = _Uint_type;
    static const unsigned int _Num_channels = 4;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<int>
{
    typedef int _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
   static const _Short_vector_base_type_id _Format_base_type_id = _Int_type;
    static const unsigned int _Num_channels = 1;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<int_2>
{
    typedef int_2::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
   static const _Short_vector_base_type_id _Format_base_type_id = _Int_type;
    static const unsigned int _Num_channels = 2;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<int_3>
{
    typedef int_3::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = false;
    static const _Short_vector_base_type_id _Format_base_type_id = _Invalid_type;
    static const unsigned int _Num_channels = 0;
    static const unsigned int _Default_bits_per_channel = 0;
};

template<>
struct _Short_vector_type_traits<int_4>
{
    typedef int_4::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
   static const _Short_vector_base_type_id _Format_base_type_id = _Int_type;
    static const unsigned int _Num_channels = 4;
    static const unsigned int _Default_bits_per_channel = 32;
};


template<>
struct _Short_vector_type_traits<float>
{
    typedef float _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Float_type;
    static const unsigned int _Num_channels = 1;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<float_2>
{
    typedef float_2::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Float_type;
    static const unsigned int _Num_channels = 2;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<float_3>
{
    typedef float_3::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = false;
    static const _Short_vector_base_type_id _Format_base_type_id = _Invalid_type;
    static const unsigned int _Num_channels = 0;
    static const unsigned int _Default_bits_per_channel = 0;
};

template<>
struct _Short_vector_type_traits<float_4>
{
    typedef float_4::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Float_type;
    static const unsigned int _Num_channels = 4;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<unorm>
{
    typedef unorm _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Unorm_type;
    static const unsigned int _Num_channels = 1;
    static const unsigned int _Default_bits_per_channel = 16;
};

template<>
struct _Short_vector_type_traits<unorm_2>
{
    typedef unorm_2::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Unorm_type;
    static const unsigned int _Num_channels = 2;
    static const unsigned int _Default_bits_per_channel = 16;
};

template<>
struct _Short_vector_type_traits<unorm_3>
{
    typedef unorm_3::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = false;
    static const _Short_vector_base_type_id _Format_base_type_id = _Invalid_type;
    static const unsigned int _Num_channels = 0;
    static const unsigned int _Default_bits_per_channel = 0;
};

template<>
struct _Short_vector_type_traits<unorm_4>
{
    typedef unorm_4::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Unorm_type;
    static const unsigned int _Num_channels = 4;
    static const unsigned int _Default_bits_per_channel = 16;
};

template<>
struct _Short_vector_type_traits<norm>
{
    typedef norm _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Norm_type;
    static const unsigned int _Num_channels = 1;
    static const unsigned int _Default_bits_per_channel = 16;
};

template<>
struct _Short_vector_type_traits<norm_2>
{
    typedef norm_2::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Norm_type;
    static const unsigned int _Num_channels = 2;
    static const unsigned int _Default_bits_per_channel = 16;
};

template<>
struct _Short_vector_type_traits<norm_3>
{
    typedef norm_3::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = false;
    static const _Short_vector_base_type_id _Format_base_type_id = _Invalid_type;
    static const unsigned int _Num_channels = 0;
    static const unsigned int _Default_bits_per_channel = 0;
};

template<>
struct _Short_vector_type_traits<norm_4>
{
    typedef norm_4::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Norm_type;
    static const unsigned int _Num_channels = 4;
    static const unsigned int _Default_bits_per_channel = 16;
};


template<>
struct _Short_vector_type_traits<double>
{
    typedef double _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Double_type;
    static const unsigned int _Num_channels = 2;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<double_2>
{
    typedef double_2::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = true;
    static const _Short_vector_base_type_id _Format_base_type_id = _Double_type;
    static const unsigned int _Num_channels = 4;
    static const unsigned int _Default_bits_per_channel = 32;
};

template<>
struct _Short_vector_type_traits<double_3>
{
    typedef double_3::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = false;
    static const _Short_vector_base_type_id _Format_base_type_id = _Invalid_type;
    static const unsigned int _Num_channels = 0;
    static const unsigned int _Default_bits_per_channel = 0;
};

template<>
struct _Short_vector_type_traits<double_4>
{
    typedef double_4::value_type _Scalar_type;
    static const bool _Is_valid_SVT_for_texture = false;
    static const _Short_vector_base_type_id _Format_base_type_id = _Invalid_type;
    static const unsigned int _Num_channels = 0;
    static const unsigned int _Default_bits_per_channel = 0;
};

template<typename _Short_vector_type>
unsigned int _Get_default_bits_per_scalar_element() 
{
    return _Short_vector_type_traits<_Short_vector_type>::_Format_base_type_id == _Double_type ? 
            _Short_vector_type_traits<_Short_vector_type>::_Default_bits_per_channel * 2 :
            _Short_vector_type_traits<_Short_vector_type>::_Default_bits_per_channel;
}

template<int _Rank>
void _Get_dimensions(const Concurrency::extent<_Rank> & _Ext, size_t & _Width, size_t & _Height, size_t & _Depth)
{
    // For un-used dimensions, use value 1.
    switch((_Rank)) {
    case 1:
        _Width = static_cast<size_t>(_Ext[0]);
        _Height = 1;
        _Depth = 1;
        break;
    case 2:
        _Width = static_cast<size_t>(_Ext[1]);
        _Height = static_cast<size_t>(_Ext[0]);
        _Depth = 1;
        break;
    case 3:
        _Width = static_cast<size_t>(_Ext[2]);
        _Height = static_cast<size_t>(_Ext[1]);
        _Depth = static_cast<size_t>(_Ext[0]);
        break;
    default:
        _ASSERTE(false);
        break;
    }
}

template<int _Rank>
Concurrency::extent<_Rank> _Create_extent(size_t _Width, size_t _Height, size_t _Depth)
{
    extent<_Rank> _Ext;
    switch((_Rank)) {
    case 1:
        _Ext[0] = static_cast<int>(_Width);
        break;
    case 2:
        _Ext[0] = static_cast<int>(_Height);
        _Ext[1] = static_cast<int>(_Width);
        break;
    case 3:
        _Ext[0] = static_cast<int>(_Depth);
        _Ext[1] = static_cast<int>(_Height);
        _Ext[2] = static_cast<int>(_Width);
        break;
    default:
        _ASSERTE(false);
        break;
    }
    return _Ext;
}

// The base class for texture, writeonly_texture_view
template <typename _Value_type, int _Rank>
class _Texture_base
{
    // Friends
    template<typename _T>
    friend const _Texture_descriptor& Concurrency::details::_Get_texture_descriptor(const _T& _Tex) __GPU;
    template<typename _T>
    friend _Texture_ptr Concurrency::details::_Get_texture(const _T& _Tex) __CPU_ONLY;
    template<typename _Value_type, int _Rank>
    friend void _Copy_impl(const _Texture_base<_Value_type, _Rank>& _Src, const _Texture_base<_Value_type, _Rank>& _Dest) __CPU_ONLY;

public:
    /// <summary>
    ///     Returns the extent that defines the shape of this texture or writeonly_texture_view. 
    /// </summary>
    __declspec(property(get=get_extent)) Concurrency::extent<_Rank> extent;
    Concurrency::extent<_Rank> get_extent() const __GPU
    {
        return _M_extent;
    }

    /// <summary>
    ///     Returns the accelerator_view where this texture/writeonly_texture_view is located.
    /// </summary>
    __declspec(property(get=get_accelerator_view)) Concurrency::accelerator_view accelerator_view;
    Concurrency::accelerator_view get_accelerator_view() const __CPU_ONLY
    {
        return _Get_texture()->_Get_accelerator_view();
    }

    /// <summary>
    ///     Returns the number of bits per scalar element
    /// </summary>
    __declspec(property(get=get_bits_per_scalar_element)) unsigned int bits_per_scalar_element;
    unsigned int get_bits_per_scalar_element() const __CPU_ONLY
    {
        unsigned int _Bits_per_channel = _Get_texture()->_Get_bits_per_channel();
        return _Short_vector_type_traits<_Value_type>::_Format_base_type_id == _Double_type ? _Bits_per_channel * (sizeof(double)/sizeof(int)) : _Bits_per_channel;
    }

    /// <summary>
    ///     Returns the physical data length (in bytes) that is required in order to represent 
    ///     the texture on the host side with its native format.
    /// </summary>
    __declspec(property(get=get_data_length)) unsigned int data_length;
    unsigned int get_data_length() const __CPU_ONLY
    {
        return _Get_texture()->_Get_data_length();
    }

protected:
    // internal storage abstraction
    typedef Concurrency::details::_Texture_descriptor _Texture_descriptor;

    _Texture_base() __CPU_ONLY
    {
    }

    _Texture_base(const Concurrency::extent<_Rank>& _Ext) __CPU_ONLY
        : _M_extent(_Ext)
    {
        Concurrency::details::_Is_valid_extent(_M_extent);
    }

    // shallow copy
    _Texture_base(const _Texture_base & _Src) __GPU
        : _M_extent(_Src.get_extent()), _M_texture_descriptor(_Src._Get_descriptor())
    {
    }

    _Texture_base(const Concurrency::extent<_Rank>& _Ext, const _Texture_descriptor & _Desc)
        : _M_extent(_Ext), _M_texture_descriptor(_Desc)
    {
        Concurrency::details::_Is_valid_extent(_M_extent);
    }

    void _Copy_to(const _Texture_base & _Dest) const
    {
        if (!(*this == _Dest))
        {
            details::_Copy_impl(*this, _Dest);
        }
    }

    bool operator==(const _Texture_base & _Other) const
    {
        return _Other._M_extent == _M_extent && _Other._M_texture_descriptor == _M_texture_descriptor;
    }

    ~_Texture_base() __GPU
    {
    }

    _Texture_ptr _Get_texture() __CPU_ONLY const
    {
        return _M_texture_descriptor._Get_texture_ptr(); 
    }

    const _Texture_descriptor & _Get_descriptor() const __GPU
    {
        return _M_texture_descriptor;
    }
protected:
    Concurrency::extent<_Rank> _M_extent;
    _Texture_descriptor _M_texture_descriptor;

};

// forward declaration
template <typename _Input_iterator, typename _Value_type, int _Rank>
_Event _Copy_async_impl(_Input_iterator _First, _Input_iterator _Last, const _Texture_base<_Value_type, _Rank>& _Dest);
template <typename _Value_type, int _Rank>
_Event _Copy_async_impl(const void * _Src, unsigned int _Src_byte_size, const _Texture_base<_Value_type, _Rank>& _Dest);
template<typename _Value_type, int _Rank>
void _Copy_impl(const _Texture_base<_Value_type, _Rank>& _Src, const _Texture_base<_Value_type, _Rank>& _Dest);

} // namespace details


using Concurrency::graphics::details::_Short_vector_type_traits;

// forward declarations
template <typename _Value_type, int _Rank>
class texture;
template <typename _Value_type, int _Rank>
class writeonly_texture_view;
template <typename _Value_type, int _Rank>
void copy(const void * _Src, unsigned int _Src_byte_size, texture<_Value_type, _Rank>& _Dst);

namespace direct3d
{
template<typename _Value_type, int _Rank>
texture<_Value_type, _Rank> make_texture(const Concurrency::accelerator_view &_Av, _In_ IUnknown *_D3D_texture) __CPU_ONLY;
} // namespace direct3d

/// <summary>
///     A texture is a data aggregate on an accelerator_view in the extent domain.
///     It is a collection of variables, one for each element in an extent domain.
///     Each variable holds a value corresponding to C++ primitive type (unsigned int, 
///     int, float, double), or scalar type norm, or unorm (defined in concurrency::graphics),
///     or eligible short vector types defined in concurrency::graphics.
/// </summary>
/// <param name="_Value_type">
///     The type of the elements in the texture aggregates. 
/// </param>
/// <param name="_Rank">
///     The _Rank of the corresponding extent domain.
/// </param>
template <typename _Value_type, int _Rank> class texture : public details::_Texture_base<_Value_type, _Rank>
{
    static_assert(_Rank > 0 && _Rank <= 3, "texture is only supported for rank 1, 2, and 3.");
    static_assert(_Short_vector_type_traits<_Value_type>::_Is_valid_SVT_for_texture, "invalid value_type for class texture<value_type, rank>.");
    template<typename _Value_type, int _Rank>
    friend texture<_Value_type,_Rank> direct3d::make_texture(const Concurrency::accelerator_view &_Av, _In_ IUnknown *_D3D_texture) __CPU_ONLY;

public:
    static const int rank = _Rank;
    typedef typename _Value_type value_type;
    typedef typename _Short_vector_type_traits<_Value_type>::_Scalar_type scalar_type;

public:

    /// <summary>
    ///     Construct a texture from extents.
    /// </summary>
    /// <param name="_Extent">
    ///     An extent that describes the shape of the texture. 
    /// </param>
    texture(const Concurrency::extent<_Rank>& _Ext) __CPU_ONLY
        : _Texture_base(_Ext)
    {
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view);
    }

    /// <summary>
    ///     Construct texture&lt;T,1&gt; with the extent _E0
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of this texture (width). 
    /// </param>
    texture(int _E0) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0))
    {
        static_assert(_Rank == 1, "texture(int) is only permissible on texture<value_type, 1>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view);
    }

    /// <summary>
    ///     Construct a texture&lt;T,2&gt; from two integer extents.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    texture(int _E0, int _E1) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1))
    {
        static_assert(_Rank == 2, "texture(int, int) is only permissible on texture<value_type, 2>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view);
    }

    /// <summary>
    ///     Construct a texture&lt;T,3&gt; from three integer extents.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (depth). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the next-to-most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E2">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    texture(int _E0, int _E1, int _E2) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1, _E2))
    {
        static_assert(_Rank == 3, "texture(int, int, int) is only permissible on texture<value_type, 3>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view);
    }

    /// <summary>
    ///     Construct a texture from extents, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_Extent">
    ///     An extent that describes the shape of the texture. 
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(const Concurrency::extent<_Rank>& _Ext, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(_Ext)
    {
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(_Av);
    }

    /// <summary>
    ///     Construct a texture&lt;T,1&gt; with the extent _E0, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of this texture (width). 
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0))
    {
        static_assert(_Rank == 1, "texture(int, accelerator_view) is only permissible on texture<value_type, 1>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(_Av);
    }

    /// <summary>
    ///     Construct a texture&lt;T,2&gt; from two integer extents, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, int _E1, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1))
    {
        static_assert(_Rank == 2, "texture(int, int, accelerator_view) is only permissible on texture<value_type, 2>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(_Av);
    }

    /// <summary>
    ///     Construct a texture&lt;T,3&gt; from three integer extents, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (depth). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the next-to-most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E2">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, int _E1, int _E2, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1, _E2))
    {
        static_assert(_Rank == 3, "texture(int, int, int, accelerator_view) is only permissible on texture<value_type, 3>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(_Av);
    }

    /// <summary>
    ///     Construct a texture initialized from a pair of iterators into a container.
    /// </summary>
    /// <param name="_Extent">
    ///     An extent that describes the shape of the texture. 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Src_last">
    ///     An ending iterator into the source container.
    /// </param>
    template<typename _Input_iterator> texture(const Concurrency::extent<_Rank>& _Ext, _Input_iterator _Src_first, _Input_iterator _Src_last) __CPU_ONLY
        : _Texture_base(_Ext)
    {
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Src_first, _Src_last);
    }

    /// <summary>
    ///     Construct a texture&lt;T,1&gt; with the extent _E0 and from a pair of iterators into a container.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of this texture (width). 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Src_last">
    ///     An ending iterator into the source container.
    /// </param>
    template<typename _Input_iterator> texture(int _E0, _Input_iterator _Src_first, _Input_iterator _Src_last) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0))
    {
        static_assert(_Rank == 1, "texture(int, iterator, iterator) is only permissible on texture<value_type, 1>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Src_first, _Src_last);
    }

    /// <summary>
    ///     Construct a texture&lt;T,2&gt; with two integers and initialized from a pair of iterators into a container.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Src_last">
    ///     An ending iterator into the source container.
    /// </param>
    template<typename _Input_iterator> texture(int _E0, int _E1, _Input_iterator _Src_first, _Input_iterator _Src_last) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1))
    {
        static_assert(_Rank == 2, "texture(int, int, iterator, iterator) is only permissible on texture<value_type, 2>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Src_first, _Src_last);
    }


    /// <summary>
    ///     Construct a texture&lt;T,3&gt; with three integers and initialized from a pair of iterators into a container.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (depth). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the next-to-most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E2">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Src_last">
    ///     An ending iterator into the source container.
    /// </param>
    template<typename _Input_iterator> texture(int _E0, int _E1, int _E2, _Input_iterator _Src_first, _Input_iterator _Src_last) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1, _E2))
    {
        static_assert(_Rank == 3, "texture(int, int, int, iterator, iterator) is only permissible on texture<value_type, 3>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Src_first, _Src_last);
    }

    /// <summary>
    ///     Construct a texture initialized from a pair of iterators into a container, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_Extent">
    ///     An extent that describes the shape of the texture. 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Src_last">
    ///     An ending iterator into the source container.
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    template<typename _Input_iterator> texture(const Concurrency::extent<_Rank>& _Ext, _Input_iterator _Src_first, _Input_iterator _Src_last, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(_Ext)
    {
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(_Av, _Src_first, _Src_last);
    }

    /// <summary>
    ///     Construct a texture&lt;T,1&gt; with interger _E0 and initialized from a pair of iterators into a container, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of this texture (width). 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Src_last">
    ///     An ending iterator into the source container.
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    template<typename _Input_iterator> texture(int _E0, _Input_iterator _Src_first, _Input_iterator _Src_last, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0))
    {
        static_assert(_Rank == 1, "texture(int, iterator, iterator, accelerator_view) is only permissible on texture<value_type, 1>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(_Av, _Src_first, _Src_last);
    }

    /// <summary>
    ///     Construct a texture&lt;T,2&gt; with two integers and initialized from a pair of iterators into a container, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Src_last">
    ///     An ending iterator into the source container.
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    template<typename _Input_iterator> texture(int _E0, int _E1, _Input_iterator _Src_first, _Input_iterator _Src_last, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1))
    {
        static_assert(_Rank == 2, "texture(int, int, iterator, iterator, accelerator_view) is only permissible on texture<value_type, 2>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(_Av, _Src_first, _Src_last);
    }

    /// <summary>
    ///     Construct a texture&lt;T,3&gt; with three integers and initialized from a pair of iterators into a container, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (depth). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the next-to-most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E2">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Src_last">
    ///     An ending iterator into the source container.
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    template<typename _Input_iterator> texture(int _E0, int _E1, int _E2, _Input_iterator _Src_first, _Input_iterator _Src_last, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1, _E2))
    {
        static_assert(_Rank == 3, "texture(int, int, int, iterator, iterator, accelerator_view) is only permissible on texture<value_type, 3>.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "texture cannot be constructed from unorm based short vectors via this constructor.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "texture cannot be constructed from norm based short vectors via this constructor.");
        _Initialize(_Av, _Src_first, _Src_last);
    }


    /// <summary>
    ///     Construct a texture from extents and specified bits per scalar element
    /// </summary>
    /// <param name="_Extent">
    ///     An extent that describes the shape of the texture. 
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(const Concurrency::extent<_Rank>& _Ext, unsigned int _Bits_per_scalar_element) __CPU_ONLY
        : _Texture_base(_Ext)
    {
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T,1&gt; with interger _E0 and specified bits per scalar element
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of this texture (width). 
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(int _E0, unsigned int _Bits_per_scalar_element) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0))
    {
        static_assert(_Rank == 1, "texture(int, unsigned int) is only permissible on texture<value_type, 1>.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T,2&gt; with two integers and specified bits per scalar element
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(int _E0, int _E1, unsigned int _Bits_per_scalar_element) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1))
    {
        static_assert(_Rank == 2, "texture(int, int, unsigned int) is only permissible on texture<value_type, 2>.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T,3&gt; with three integers and specified bits per scalar element
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (depth). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the next-to-most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E2">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Src_first">
    ///     A beginning iterator into the source container.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(int _E0, int _E1, int _E2, unsigned int _Bits_per_scalar_element) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1, _E2))
    {
        static_assert(_Rank == 3, "texture(int, int, int, unsigned int) is only permissible on texture<value_type, 3>.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Bits_per_scalar_element);
    }


    /// <summary>
    ///     Construct a texture from extents and specified bits per scalar element, bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_Extent">
    ///     An extent that describes the shape of the texture. 
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(const Concurrency::extent<_Rank>& _Ext, unsigned int _Bits_per_scalar_element, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(_Ext)
    {
        _Initialize(_Av, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T, 1&gt; with integer _E0 and specified bits per scalar element, bound to a specific accelerator.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, unsigned int _Bits_per_scalar_element, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0))
    {
        static_assert(_Rank == 1, "texture(int, unsigned int, accelerator_view) is only permissible on texture<value_type, 1>.");
        _Initialize(_Av, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T,2&gt; with two integers and specified bits per scalar element, bound to a specific accelerator.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, int _E1, unsigned int _Bits_per_scalar_element, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1))
    {
        static_assert(_Rank == 2, "texture(int, int, unsigned int, accelerator_view) is only permissible on texture<value_type, 2>.");
        _Initialize(_Av, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T,3&gt; with three integers and specified bits per scalar element, bound to a specific accelerator.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (depth). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E2">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, int _E1, int _E2, unsigned int _Bits_per_scalar_element, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1, _E2))
    {
        static_assert(_Rank == 3, "texture(int, int, int, unsigned int, accelerator_view) is only permissible on texture<value_type, 3>.");
        _Initialize(_Av, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture from extents and specified bits per scalar element, initialized from a host buffer.
    /// </summary>
    /// <param name="_Extent">
    ///     An extent that describes the shape of the texture. 
    /// </param>
    /// <param name="_Source">
    ///     A pointer to a host buffer.
    /// </param>
    /// <param name="_Source_byte_size">
    ///     Number of bytes in the source buffer.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(const Concurrency::extent<_Rank>& _Ext, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element) __CPU_ONLY
        : _Texture_base(_Ext)
    {
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Source, _Src_byte_size, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T,1&gt; with integer _E0 and specified bits per scalar element, initialized from a host buffer.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of this texture (width). 
    /// </param>
    /// <param name="_Source">
    ///     A pointer to a host buffer.
    /// </param>
    /// <param name="_Source_byte_size">
    ///     Number of bytes in the source buffer.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(int _E0, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0))
    {
        static_assert(_Rank == 1, "texture(int, void *, unsigned int, unsigned int) is only permissible on texture<value_type, 1>.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Source, _Src_byte_size, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T,2&gt; with two integers and specified bits per scalar element, initialized from a host buffer.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Source">
    ///     A pointer to a host buffer.
    /// </param>
    /// <param name="_Source_byte_size">
    ///     Number of bytes in the source buffer.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(int _E0, int _E1, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1))
    {
        static_assert(_Rank == 2, "texture(int, int, void *, unsigned int, unsigned int) is only permissible on texture<value_type, 2>.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Source, _Src_byte_size, _Bits_per_scalar_element);
    }


    /// <summary>
    ///     Construct a texture&lt;T,3&gt; with three integers and specified bits per scalar element, initialized from a host buffer.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (depth). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E2">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Source">
    ///     A pointer to a host buffer.
    /// </param>
    /// <param name="_Source_byte_size">
    ///     Number of bytes in the source buffer.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    texture(int _E0, int _E1, int _E2, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1, _E2))
    {
        static_assert(_Rank == 3, "texture(int, int, int, void *, unsigned int, unsigned int) is only permissible on texture<value_type, 3>.");
        _Initialize(Concurrency::details::_Select_default_accelerator().default_view, _Source, _Src_byte_size, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture from extents and specified bits per scalar element, initialized from a host buffer,  bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_Extent">
    ///     An extent that describes the shape of the texture. 
    /// </param>
    /// <param name="_Source">
    ///     A pointer to a host buffer.
    /// </param>
    /// <param name="_Source_byte_size">
    ///     Number of bytes in the source buffer.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(const Concurrency::extent<_Rank>& _Ext, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(_Ext)
    {
        _Initialize(_Av, _Source, _Src_byte_size, _Bits_per_scalar_element);
    }


    /// <summary>
    ///     Construct a texture&lt;T, 1&gt; with integer _E0 and specified bits per scalar element, initialized from a host buffer,  bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of this texture (width). 
    /// </param>
    /// <param name="_Source">
    ///     A pointer to a host buffer.
    /// </param>
    /// <param name="_Source_byte_size">
    ///     Number of bytes in the source buffer.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0))
    {
        static_assert(_Rank == 1, "texture(int, void *, unsigned int, unsigned int, accelerator_view) is only permissible on texture<value_type, 1>.");
        _Initialize(_Av, _Source, _Src_byte_size, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T, 2&gt; with two integers and specified bits per scalar element, initialized from a host buffer,  bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Source">
    ///     A pointer to a host buffer.
    /// </param>
    /// <param name="_Source_byte_size">
    ///     Number of bytes in the source buffer.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, int _E1, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1))
    {
        static_assert(_Rank == 2, "texture(int, int, void *, unsigned int, unsigned int, accelerator_view) is only permissible on texture<value_type, 2>.");
        _Initialize(_Av, _Source, _Src_byte_size, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Construct a texture&lt;T, 3&gt; with three integers and specified bits per scalar element, initialized from a host buffer,  bound to a specific accelerator_view.
    /// </summary>
    /// <param name="_E0">
    ///     An integer that is the length of the most-significant dimension of this texture (depth). 
    /// </param>
    /// <param name="_E1">
    ///     An integer that is the length of the least-significant dimension of this texture (height). 
    /// </param>
    /// <param name="_E2">
    ///     An integer that is the length of the least-significant dimension of this texture (width). 
    /// </param>
    /// <param name="_Source">
    ///     A pointer to a host buffer.
    /// </param>
    /// <param name="_Source_byte_size">
    ///     Number of bytes in the source buffer.
    /// </param>
    /// <param name="_Bits_per_scalar_element">
    ///     Number of bits per each scalar element in the underlying scalar type of the texture. 
    ///     In general, supported value is 0, 8, 16, 32, 64.
    ///     If 0 is specified, the number of bits picks defaulted value for the underlying scalar_type.
    ///     64 is only valid for double based textures
    /// </param>
    /// <param name="_Av">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(int _E0, int _E1, int _E2, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element, const Concurrency::accelerator_view& _Av) __CPU_ONLY
        : _Texture_base(Concurrency::extent<_Rank>(_E0, _E1, _E2))
    {
        static_assert(_Rank == 3, "texture(int, int, int, void *, unsigned int, unsigned int, accelerator_view) is only permissible on texture<value_type, 3>.");
        _Initialize(_Av, _Source, _Src_byte_size, _Bits_per_scalar_element);
    }

    /// <summary>
    ///     Copy constructor. Deep copy
    /// </summary>
    /// <param name="_Src">
    ///     The texture to copy from.
    /// </param>
    texture(const texture & _Src)
        : _Texture_base(_Src.extent)
    {
        _Initialize(_Src.accelerator_view, _Src);
    }

    /// <summary>
    ///     Copy constructor. Deep copy
    /// </summary>
    /// <param name="_Src">
    ///     The texture to copy from.
    /// </param>
    /// <param name="_Acc_view">
    ///     An accelerator_view where this texture resides.
    /// </param>
    texture(const texture & _Src, const Concurrency::accelerator_view & _Acc_view)
        : _Texture_base(_Src.extent)
    {
        _Initialize(_Acc_view, _Src);
    }

    /// <summary>
    ///     Copy assignment operator. Deep copy
    /// </summary>
    /// <param name="_Src">
    ///     The texture to copy from.
    /// </param>
    /// <returns>
    ///     A reference to this texture.
    /// </returns>
    texture& operator=(const texture & _Other)
    {
        if (this != &_Other)
        {
            _M_extent = _Other._M_extent;
            _Initialize(_Other.accelerator_view, _Other);
        }
        return *this;
    }
    
    /// <summary>
    ///     Copy-to, deep copy
    /// </summary>
    /// <param name="_Dest">
    ///     The destionation texture to copy to.
    /// </param>
    void copy_to(texture & _Dest) const
    {
        auto _Span_id = concurrency::details::_Get_amp_trace()->_Start_copy_event_helper(concurrency::details::_Get_texture_descriptor(*this),
                                                                            concurrency::details::_Get_texture_descriptor(_Dest),
                                                                            this->get_data_length());

        _Texture_base::_Copy_to(_Dest);

        concurrency::details::_Get_amp_trace()->_Write_end_event(_Span_id);
    }

    /// <summary>
    ///     Copy-to, deep copy
    /// </summary>
    /// <param name="_Dest">
    ///     The destionation writeonly_texture_view to copy to.
    /// </param>
    void copy_to(const writeonly_texture_view<_Value_type, _Rank> & _Dest) const
    {
        auto _Span_id = concurrency::details::_Get_amp_trace()->_Start_copy_event_helper(concurrency::details::_Get_texture_descriptor(*this),
                                                                            concurrency::details::_Get_texture_descriptor(_Dest),
                                                                            this->get_data_length());

        _Texture_base::_Copy_to(_Dest);

        concurrency::details::_Get_amp_trace()->_Write_end_event(_Span_id);
    }

    /// <summary>
    ///     Move constructor
    /// </summary>
    /// <param name="_Other">
    ///     The source texture to move from.
    /// </param>
    texture(texture && _Other)
    {
        *this = std::move(_Other);
    }

    /// <summary>
    ///     Move assignment operator
    /// </summary>
    /// <param name="_Other">
    ///     The source texture to move from.
    /// </param>
    /// <returns>
    ///     A reference to this texture.
    /// </returns>
    texture& operator=(texture<_Value_type, _Rank> && _Other)
    {
        if (this != &_Other)
        {
            _M_extent = _Other._M_extent;
            _M_texture_descriptor = _Other._M_texture_descriptor;

            _Other._M_texture_descriptor._M_data_ptr = NULL;
            _Other._M_texture_descriptor._Set_texture_ptr(NULL);
        }
        return *this;
    }

    /// <summary>
    ///     Destructor
    /// </summary>
    ~texture() __CPU_ONLY
    {
    }

    /// <summary>
    ///     Get the element value indexed by _Index.
    /// </summary>
    /// <param name="_Index">
    ///     The index.
    /// </param>
    /// <returns>
    ///     The element value indexed by _Index.
    /// </returns>
    const value_type operator[] (const index<_Rank>& _Index) const __GPU_ONLY
    {
        value_type _Tmp;
        _Texture_read_helper<index<_Rank>, _Rank>::func(_M_texture_descriptor._M_data_ptr, &_Tmp, _Index);
        return _Tmp;
    }

    /// <summary>
    ///     Get the element value indexed by _I.
    /// </summary>
    /// <param name="_I">
    ///     The index.
    /// </param>
    /// <returns>
    ///     The element value indexed by _I.
    /// </returns>
    const value_type operator[] (int _I0) const __GPU_ONLY
    {
        static_assert(_Rank == 1, "value_type texture::operator[](int) is only permissible on texture<value_type, 1>.");
        return (*this)[index<1>(_I0)];
    }

    /// <summary>
    ///     Get the element value indexed by _Index.
    /// </summary>
    /// <param name="_Index">
    ///     The index.
    /// </param>
    /// <returns>
    ///     The element value indexed by _Index.
    /// </returns>
    const value_type operator() (const index<_Rank>& _Index) const __GPU_ONLY
    {
        return (*this)[_Index];
    }

    /// <summary>
    ///     Get the element value indexed by _I0
    /// </summary>
    /// <param name="_I0">
    ///     The index.
    /// </param>
    /// <returns>
    ///     The element value indexed by _I0.
    /// </returns>
    const value_type operator() (int _I0) const __GPU_ONLY
    {
        static_assert(_Rank == 1, "value_type texture::operator()(int) is only permissible on texture<value_type, 1>.");
        return (*this)[index<1>(_I0)];
    }

    /// <summary>
    ///     Get the element value indexed by (_I0,_I1)
    /// </summary>
    /// <param name="_I0">
    ///     The most-significant component of the index
    /// </param>
    /// <param name="_I1">
    ///     The least-significant component of the index
    /// </param>
    /// <returns>
    ///     The element value indexed by (_I0,_I1)
    /// </returns>
    const value_type operator() (int _I0, int _I1) const __GPU_ONLY
    {
        static_assert(_Rank == 2, "value_type texture::operator()(int, int) is only permissible on texture<value_type, 2>.");
        return (*this)[index<2>(_I0, _I1)];
    }

    /// <summary>
    ///     Get the element value indexed by (_I0,_I1,_I2)
    /// </summary>
    /// <param name="_I0">
    ///     The most-significant component of the index
    /// </param>
    /// <param name="_I1">
    ///     The next-to-most-significant component of the index
    /// </param>
    /// <param name="_I2">
    ///     The least-significant component of the index
    /// </param>
    /// <returns>
    ///     The element value indexed by (_I0,_I1,_I2)
    /// </returns>
    const value_type operator() (int _I0, int _I1, int _I2) const __GPU_ONLY
    {
        static_assert(_Rank == 3, "value_type texture::operator()(int, int, int) is only permissible on texture<value_type, 3>.");
        return (*this)[index<3>(_I0, _I1, _I2)];
    }

    /// <summary>
    ///     Get the element value indexed by _Index.
    /// </summary>
    /// <param name="_Index">
    ///     The index.
    /// </param>
    /// <returns>
    ///     The element value indexed by _Index.
    /// </returns>
    const value_type get(const index<_Rank>& _Index) const __GPU_ONLY
    {
        return (*this)[_Index];
    }

    /// <summary>
    ///     Set the element indexed by _Index with value _Value.
    /// </summary>
    /// <param name="_Index">
    ///     The index.
    /// </param>
    /// <param name="_Value">
    ///     The value to be set to the element indexed by _Index.
    /// </param>
    void set(const index<_Rank>& _Index, const value_type& _Value) __GPU_ONLY
    {
        static_assert(_Short_vector_type_traits<_Value_type>::_Num_channels == 1, "Invalid value_type for set method.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Unorm_type, "Invalid value_type for set method.");
        static_assert(_Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Norm_type, "Invalid value_type for set method.");
        _Texture_write_helper<index<_Rank>, _Rank>::func(_M_texture_descriptor._M_data_ptr, &_Value, _Index);
    }


private:
    texture(const Concurrency::extent<_Rank> & _Ext, const _Texture_descriptor & _Descriptor)
        : _Texture_base(_Ext, _Descriptor)
    {
    }

    void _Initialize(Concurrency::accelerator_view _Av, unsigned int _Bits_per_scalar_element) __CPU_ONLY
    {
        if (_Bits_per_scalar_element != 8 && _Bits_per_scalar_element != 16 && 
            _Bits_per_scalar_element != 32 && _Bits_per_scalar_element != 64)
        {
            throw runtime_exception("Invalid _Bits_per_scalar_element argument - it can only be 8, 16, 32, or 64.", E_INVALIDARG);
        }

        // special cases for 64 and for double based textures

#pragma warning( push )
#pragma warning( disable : 4127 ) // conditional expression is constant
        if (_Bits_per_scalar_element == 64 && _Short_vector_type_traits<_Value_type>::_Format_base_type_id != _Double_type)
        {
            throw runtime_exception("Invalid _Bits_per_scalar_element argument - 64 is only valid for texture of double based short vector types.", E_INVALIDARG);
        }

        if (_Bits_per_scalar_element != 64 && _Short_vector_type_traits<_Value_type>::_Format_base_type_id == _Double_type)
        {
            throw runtime_exception("Invalid _Bits_per_scalar_element argument - it can only be 64 for texture of double based short vector types.", E_INVALIDARG);
        }

        // the rest of the check is done by _Texture::_Create_texture, it depends on the underlying supported DXGI formats.

        unsigned int _Bits_per_channel = _Bits_per_scalar_element;

        if (_Short_vector_type_traits<_Value_type>::_Format_base_type_id == _Double_type)
        {
            _Bits_per_channel = _Short_vector_type_traits<_Value_type>::_Default_bits_per_channel;
        }

        size_t _Width, _Height, _Depth;
        Concurrency::graphics::details::_Get_dimensions(_M_extent, _Width, _Height, _Depth);
        _Texture_ptr _Tex_ptr = _Texture::_Create_texture(_Av, _Rank, _Width, _Height, _Depth, 1 /*_Mip_level_count */,
            _Short_vector_type_traits<_Value_type>::_Format_base_type_id == _Double_type ? _Uint_type : _Short_vector_type_traits<_Value_type>::_Format_base_type_id,
            _Short_vector_type_traits<_Value_type>::_Num_channels,
            _Bits_per_channel);

        _M_texture_descriptor._Set_texture_ptr(_Tex_ptr);
#pragma warning( pop )
    }
    
    void _Initialize(Concurrency::accelerator_view _Av) __CPU_ONLY
    {
        _Initialize(_Av, Concurrency::graphics::details::_Get_default_bits_per_scalar_element<_Value_type>());
    }

    template<typename _Input_iterator>
    void _Initialize(Concurrency::accelerator_view _Av, _Input_iterator _Src_first, _Input_iterator _Src_last) __CPU_ONLY
    {
        _Initialize(_Av);

        auto _Span_id = Concurrency::details::_Get_amp_trace()->_Start_copy_event_helper(nullptr,
                                                                                         Concurrency::details::_Get_texture_descriptor(*this),
                                                                                         this->get_data_length());

        Concurrency::graphics::details::_Copy_async_impl(_Src_first, _Src_last, *this)._Get();

        Concurrency::details::_Get_amp_trace()->_Write_end_event(_Span_id);
    }
    
    void _Initialize(Concurrency::accelerator_view _Av, const void * _Source, unsigned int _Src_byte_size, unsigned int _Bits_per_scalar_element) __CPU_ONLY
    {
        _Initialize(_Av, _Bits_per_scalar_element);
        Concurrency::graphics::copy(_Source, _Src_byte_size, *this);
    }
    
    void _Initialize(Concurrency::accelerator_view _Av, const void * _Source, unsigned int _Src_byte_size) __CPU_ONLY
    {
        _Initialize(_Av);
        Concurrency::graphics::copy(_Source, _Src_byte_size, *this);
    }    

    void _Initialize(Concurrency::accelerator_view _Av, const _Texture_base<_Value_type, _Rank> & _Src) __CPU_ONLY
    {
        _Initialize(_Av, _Src.bits_per_scalar_element);

        auto _Span_id = Concurrency::details::_Get_amp_trace()->_Start_copy_event_helper(Concurrency::details::_Get_texture_descriptor(_Src),
                                                                                         Concurrency::details::_Get_texture_descriptor(*this),
                                                                                         this->get_data_length());

        Concurrency::graphics::details::_Copy_impl(_Src, *this);

        Concurrency::details::_Get_amp_trace()->_Write_end_event(_Span_id);
    }
};

/// <summary>
///     A writeonly_texture_view provides writeonly access to a texture.
/// </summary>
/// <param name="_Value_type">
///     The type of the elements in the texture aggregates. 
/// </param>
/// <param name="_Rank">
///     The _Rank of the corresponding extent domain.
/// </param>
template <typename _Value_type, int _Rank> class writeonly_texture_view : public details::_Texture_base<_Value_type, _Rank>
{
    static_assert(_Rank > 0 && _Rank <= 3, "texture is only supported for rank 1, 2, and 3.");
    static_assert(_Short_vector_type_traits<_Value_type>::_Is_valid_SVT_for_texture, "invalid value_type for class writeonly_texture_view<value_type, rank>.");
public:
    static const int rank = _Rank;
    typedef typename _Value_type value_type;
    typedef typename _Short_vector_type_traits<_Value_type>::_Scalar_type scalar_type;

    /// <summary>
    ///     Construct a writeonly_texture_view of a texture _Src.
    /// </summary>
    /// <param name="_Src">
    ///     The texture where the writeonly view is created on.
    /// </param>
    writeonly_texture_view(texture<_Value_type, _Rank>& _Src) __CPU_ONLY
        : _Texture_base(_Src)
    {
    }

    /// <summary>
    ///     Construct a writeonly_texture_view of a texture _Src.
    /// </summary>
    /// <param name="_Src">
    ///     The texture where the writeonly view is created on.
    /// </param>
    writeonly_texture_view(texture<_Value_type, _Rank>& _Src) __GPU_ONLY
        : _Texture_base(_Src)
    {
        static_assert(_Short_vector_type_traits<_Value_type>::_Num_channels == 1, 
                      "In an amp-restricted function, writeonly_texture_view<value_type, rank> can only be constructed when value_type is a short vector type with a single (non double) scalar element.");
    }

    /// <summary>
    ///     Construct a writeonly_texture_view from another writeonly_texture_view. Both are the view of the same texture.
    /// </summary>
    /// <param name="_Src">
    ///     The writeonly_texture_view where the current view is created with.
    /// </param>
    writeonly_texture_view(const writeonly_texture_view<_Value_type, _Rank>& _Src) __GPU 
        : _Texture_base(_Src)
    {
    }

    /// <summary>
    ///     Assignment operator. This writeonly_texture_view becomes a view of the same texture which _Other is a view of.
    /// </summary>
    /// <param name="_Other">
    ///     The source writeonly_texture_view.
    /// </param>
    writeonly_texture_view<_Value_type, _Rank>& operator=(const writeonly_texture_view<_Value_type, _Rank>& _Other) __GPU
    {
        if (this != &_Other)
        {
            _M_extent = _Other._M_extent;
            _M_texture_descriptor = _Other._M_texture_descriptor;
        }
        return *this;
    }

    /// <summary>
    ///     Destructor
    /// </summary>
    ~writeonly_texture_view() __GPU 
    {
    }

    /// <summary>
    ///     Set the element indexed by _Index with value _Value.
    /// </summary>
    /// <param name="_Index">
    ///     The index.
    /// </param>
    /// <param name="_Value">
    ///     The value to be set to the element indexed by _Index.
    /// </param>
    void set(const index<_Rank>& _Index, const value_type& _Value) const __GPU_ONLY
    {
        _Texture_write_helper<index<_Rank>, _Rank>::func(_M_texture_descriptor._M_data_ptr, &_Value, _Index);
    }
};

namespace details
{
#pragma warning ( push )
#pragma warning ( disable : 6101 )
// Supress "warning C6101: Returning uninitialized memory '*_Dst'.:  A successful"
// "path through the function does not set the named _Out_ parameter."
// The callers to _Copy_data_on_host all have static_assert that _Rank has to be 1, 2, or 3 dimensions for texture
//
inline void _Copy_data_on_host(int _Rank, _Out_ void * _Dst, const void * _Src, 
                               size_t _Size,
                               size_t _Height, size_t _Depth,
                               size_t _Row_size,
                               size_t _Dst_row_size, size_t _Dst_depth_slice_size,
                               size_t _Src_row_size, size_t _Src_depth_slice_size)
{
    unsigned char * _Dst_ptr = reinterpret_cast<unsigned char *>(_Dst);
    const unsigned char * _Src_ptr = reinterpret_cast<const unsigned char *>(_Src);
    switch(_Rank)
    {
    case 1:
        memcpy_s(_Dst_ptr, _Size, _Src_ptr, _Size);
        break;
    case 2:
        for (size_t _I = 0, _Dst_offset = 0, _Src_offset = 0; 
            _I < _Height; 
            _I++, _Dst_offset += _Dst_row_size, _Src_offset += _Src_row_size)
        {
            memcpy_s(_Dst_ptr + _Dst_offset, _Row_size, _Src_ptr + _Src_offset, _Row_size);
        }
        break;
    case 3:
        for (size_t _I = 0, _Dst_slice = 0, _Src_slice = 0;
             _I < _Depth;
             _I++, _Dst_slice += _Dst_depth_slice_size, _Src_slice += _Src_depth_slice_size) 
        {
            for (size_t _J = 0, _Dst_offset = _Dst_slice, _Src_offset = _Src_slice;
                _J < _Height; 
                _J++, _Dst_offset += _Dst_row_size, _Src_offset += _Src_row_size)
            {
                memcpy_s(_Dst_ptr + _Dst_offset, _Row_size, _Src_ptr + _Src_offset, _Row_size);
            }
        }
        break;
    default:
        _ASSERTE(FALSE);
        break;
    }
}
#pragma warning ( pop ) // disable : 6101

template<typename _Value_type, int _Rank>
_Event _Copy_async_impl(const _Texture_base<_Value_type, _Rank>& _Src, _Out_ void * _Dst, unsigned int _Dst_byte_size)
{
    if (_Src.data_length > _Dst_byte_size) 
    {
        throw runtime_exception("Invalid _Dst_byte_size argument. _Dst_byte_size is smaller than the size of _Src.", E_INVALIDARG);
    }

    _Texture_ptr _Src_tex_ptr = _Get_texture(_Src);

    // The src is on the device. We need to copy it out to a temporary staging texture
    _Texture_ptr _Src_staging_tex_ptr = _Texture::_Create_stage_texture(
        _Src.accelerator_view, accelerator(accelerator::cpu_accelerator).default_view,
        _Rank, _Src_tex_ptr->_Get_width(), _Src_tex_ptr->_Get_height(), _Src_tex_ptr->_Get_depth(),
        _Src_tex_ptr->_Get_mip_levels(), _Src_tex_ptr->_Get_format(), true /*_Is_temp */);

    _Event _Ev = _Src_tex_ptr->_Copy_to_async(_Src_staging_tex_ptr);

    size_t _Size_to_copy = _Src.data_length;

    return _Ev._Add_continuation(std::function<_Event()>([_Src_staging_tex_ptr, _Dst, _Size_to_copy]() mutable -> _Event {
        
        // Now copy from the staging texture to the output
        size_t _Width = _Src_staging_tex_ptr->_Get_width();
        size_t _Height = _Src_staging_tex_ptr->_Get_height();
        size_t _Depth = _Src_staging_tex_ptr->_Get_depth();
        _ASSERTE(_Src_staging_tex_ptr->_Get_bits_per_element() * _Width % 8 == 0);
        size_t _Row_size = (_Src_staging_tex_ptr->_Get_bits_per_element() * _Width) >> 3; // in bytes
        size_t _Depth_slice_size = _Row_size * _Height;

        size_t _Row_pitch = _Src_staging_tex_ptr->_Get_row_pitch();
        size_t _Depth_pitch = _Src_staging_tex_ptr->_Get_depth_pitch();
        _ASSERTE(_Row_pitch >= _Row_size);
        _ASSERTE(_Depth_pitch >= _Depth_slice_size);

        _Copy_data_on_host(_Rank, _Dst, _Src_staging_tex_ptr->_Get_host_ptr(),
                           _Size_to_copy, _Height, _Depth, _Row_size,
                           _Row_size, _Depth_slice_size, _Row_pitch, _Depth_pitch);
        return _Event();
    }));
}

template <typename _Value_type, int _Rank>
_Event _Copy_async_impl(const void * _Src, unsigned int _Src_byte_size, const _Texture_base<_Value_type, _Rank>& _Dest)
{
    if (_Src_byte_size < _Dest.data_length)
    {
        throw runtime_exception("Invalid _Src_byte_size argument(s). _Src_byte_size is smaller than total size of the _Dest.", E_INVALIDARG);
    }

    // dest is on a device. Lets create a temp staging buffer on the dest 
    // accelerator_view and copy the input over
    _Texture_ptr _Dest_tex_ptr = _Get_texture(_Dest);
    size_t _Width = _Dest_tex_ptr->_Get_width();
    size_t _Height = _Dest_tex_ptr->_Get_height();
    size_t _Depth = _Dest_tex_ptr->_Get_depth();
    _ASSERTE((_Dest_tex_ptr->_Get_bits_per_element() * _Width) % 8 == 0);
    size_t _Row_size = (_Dest_tex_ptr->_Get_bits_per_element() * _Width) >> 3; // in bytes
    size_t _Depth_slice_size = _Row_size * _Height;

    _Texture_ptr _Dest_staging_tex_ptr = _Texture::_Create_stage_texture(
        _Dest.accelerator_view, accelerator(accelerator::cpu_accelerator).default_view, 
        _Rank, _Width, _Height, _Depth,
        _Dest_tex_ptr->_Get_mip_levels(), _Dest_tex_ptr->_Get_format(), true /* _Is_temp */);
    
    _Dest_staging_tex_ptr->_Map_stage_buffer(_Write_access, true /* _Wait */);

    size_t _Row_pitch = _Dest_staging_tex_ptr->_Get_row_pitch();
    size_t _Depth_pitch = _Dest_staging_tex_ptr->_Get_depth_pitch();
    _ASSERTE(_Row_pitch >= _Row_size);
    _ASSERTE(_Depth_pitch >= _Depth_slice_size);

    // Copy from input to the staging
    _Copy_data_on_host(_Rank, _Dest_staging_tex_ptr->_Get_host_ptr(), _Src, _Dest.data_length,
                       _Height, _Depth, _Row_size,
                       _Row_pitch, _Depth_pitch, _Row_size, _Depth_slice_size);

    return _Dest_staging_tex_ptr->_Copy_to_async(_Dest_tex_ptr);
}

template <typename _Input_iterator, typename _Value_type, int _Rank>
_Event _Copy_async_impl(_Input_iterator _First, _Input_iterator _Last, const _Texture_base<_Value_type, _Rank>& _Dest)
{
    if ((unsigned int)std::distance(_First, _Last) < _Dest.extent.size())
    {
        throw runtime_exception("Inadequate amount of data supplied through the iterators", E_INVALIDARG);
    }

    // dest is on a device. Lets create a temp staging buffer on the dest 
    // accelerator_view and copy the input over
    _Texture_ptr _Dest_tex_ptr = _Get_texture(_Dest);
    size_t _Width = _Dest_tex_ptr->_Get_width();
    size_t _Height = _Dest_tex_ptr->_Get_height();
    size_t _Depth = _Dest_tex_ptr->_Get_depth();
    _ASSERTE((_Dest_tex_ptr->_Get_bits_per_element() * _Width) % 8 == 0);
    size_t _Row_size = (_Dest_tex_ptr->_Get_bits_per_element() * _Width) >> 3; // in bytes
    size_t _Depth_slice_size = _Row_size * _Height;

    _Texture_ptr _Dest_staging_tex_ptr = _Texture::_Create_stage_texture(
        _Dest.accelerator_view, accelerator(accelerator::cpu_accelerator).default_view, 
        _Rank, _Width, _Height, _Depth,
        _Dest_tex_ptr->_Get_mip_levels(), _Dest_tex_ptr->_Get_format(), true /* _Is_temp */);
    
    _Dest_staging_tex_ptr->_Map_stage_buffer(_Write_access, true /* _Wait */);

    size_t _Row_pitch = _Dest_staging_tex_ptr->_Get_row_pitch();
    size_t _Depth_pitch = _Dest_staging_tex_ptr->_Get_depth_pitch();
    _ASSERTE(_Row_pitch >= _Row_size);
    _ASSERTE(_Depth_pitch >= _Depth_slice_size);
    UNREFERENCED_PARAMETER(_Depth_slice_size);
   
    unsigned char * _Dst_ptr = reinterpret_cast<unsigned char *>(_Dest_staging_tex_ptr->_Get_host_ptr());

    // Copy from input to the staging
    switch((_Rank))
    {
    case 1:
        {
            _Input_iterator _End = _First;
            std::advance(_End, _Width);
            std::copy(_First, _End, stdext::make_unchecked_array_iterator(reinterpret_cast<_Value_type*>(_Dst_ptr)));
        }
        break;
    case 2:
        {
            _Input_iterator _Src_start = _First, _Src_end = _First;
            for (size_t _I = 0, _Dst_offset = 0;
                _I < _Height; 
                _I++, _Dst_offset += _Row_pitch, _Src_start = _Src_end)
            {
                std::advance(_Src_end, _Width);
                std::copy(_Src_start, _Src_end, 
                          stdext::make_unchecked_array_iterator(reinterpret_cast<_Value_type*>(_Dst_ptr + _Dst_offset)));
            }
        }
        break;
    case 3:
        {
            _Input_iterator _Src_depth_slice_start = _First, _Src_deptch_slice_end = _First;
            for (size_t _I = 0, _Dst_slice = 0;
                 _I < _Depth;
                 _I++, _Dst_slice += _Depth_pitch, 
                 _Src_depth_slice_start = _Src_deptch_slice_end)
            {
                std::advance(_Src_deptch_slice_end, _Width * _Height);
                _Input_iterator _Src_start = _Src_depth_slice_start, _Src_end = _Src_start;
                for (size_t _J = 0, _Dst_offset = _Dst_slice;
                    _J < _Height; 
                    _J++, _Dst_offset += _Row_pitch, _Src_start = _Src_end)
                {
                    std::advance(_Src_end, _Width);
                    std::copy(_Src_start, _Src_end, 
                              stdext::make_unchecked_array_iterator(reinterpret_cast<_Value_type*>(_Dst_ptr + _Dst_offset)));
                }
            }
        }
        break;
    }

    return _Dest_staging_tex_ptr->_Copy_to_async(_Dest_tex_ptr);
}


template<typename _Value_type, int _Rank>
void _Copy_impl(const _Texture_base<_Value_type, _Rank>& _Src, const _Texture_base<_Value_type, _Rank>& _Dest)
{
    // Must be exactly the same extent
    
    for (int i = 0; i < _Rank; i++)
    {
        if (_Src.extent[i] != _Dest.extent[i])
        {
            throw runtime_exception("The source and destination textures must have the exactly the same extent.", E_INVALIDARG);
        }
    }
    _Texture_ptr _Src_tex = _Src._Get_texture();
    _Texture_ptr _Dst_tex = _Dest._Get_texture();
    // format must be compatible
    if (_Src_tex->_Get_num_channels() != _Dst_tex->_Get_num_channels() ||
        _Src_tex->_Get_bits_per_channel() != _Dst_tex->_Get_bits_per_channel())
    {
        throw runtime_exception("The source and destination textures are not compatible.", E_INVALIDARG);
    }

    unsigned int _Size = _Src.data_length;
    std::vector<unsigned char> _Host_buffer(_Size);
    _Copy_async_impl(_Src, reinterpret_cast<void *>(_Host_buffer.data()), _Size)._Get();
    _Copy_async_impl(reinterpret_cast<void *>(_Host_buffer.data()), _Size, _Dest)._Get();
}

} // namespace details

/// <summary>
///     Asynchronously copies the contents of the source texture into the destination host buffer.
/// </summary>
/// <param name="_Rank">
///     The rank of the source texture.
/// </param>
/// <param name="_Value_type">
///     The type of the elements of the source texture.
/// </param>
/// <param name="_Src">
///     The source texture.
/// </param>
/// <param name="_Dst">
///     The destination host buffer.
/// </param>
/// <param name="_Dst_byte_size">
///     Number of bytes in the destination buffer.
/// </param>
/// <returns>
///     A future upon which to wait for the operation to complete.
/// </returns>
template<typename _Value_type, int _Rank> concurrency::completion_future copy_async(const texture<_Value_type, _Rank>& _Src, _Out_ void * _Dst, unsigned int _Dst_byte_size)
{
    auto _Async_op_id = concurrency::details::_Get_amp_trace()->_Launch_async_copy_event_helper(concurrency::details::_Get_texture_descriptor(_Src), 
                                                                                   nullptr,
                                                                                   _Src.get_data_length());

    _Event _Ev = details::_Copy_async_impl(_Src, _Dst, _Dst_byte_size);

    return concurrency::details::_Get_amp_trace()->_Start_async_op_wait_event_helper(_Async_op_id, _Ev);
}

/// <summary>
///     Copies the contents of the source texture into the destination host buffer.
/// </summary>
/// <param name="_Rank">
///     The rank of the source texture.
/// </param>
/// <param name="_Value_type">
///     The type of the elements of the source texture.
/// </param>
/// <param name="_Src">
///     The source texture.
/// </param>
/// <param name="_Dst">
///     The destination host buffer.
/// </param>
/// <param name="_Dst_byte_size">
///     Number of bytes in the destination buffer.
/// </param>
template <typename _Value_type, int _Rank> void copy(const texture<_Value_type, _Rank>& _Src, _Out_ void * _Dst, unsigned int _Dst_byte_size)
{
    auto _Span_id = concurrency::details::_Get_amp_trace()->_Start_copy_event_helper(concurrency::details::_Get_texture_descriptor(_Src),
                                                                        nullptr,
                                                                        _Src.get_data_length());

    copy_async(_Src, _Dst, _Dst_byte_size).get();

    concurrency::details::_Get_amp_trace()->_Write_end_event(_Span_id);
}

/// <summary>
///     Asynchronously copies the contents of the source host buffer into the destination texture.
/// </summary>
/// <param name="_Rank">
///     The rank of the destination texture.
/// </param>
/// <param name="_Value_type">
///     The type of the elements of the destination texture.
/// </param>
/// <param name="_Src">
///     The source host buffer.
/// </param>
/// <param name="_Src_byte_size">
///     Number of bytes in the source buffer.
/// </param>
/// <param name="_Dst">
///     The destination texture.
/// </param>
/// <returns>
///     A future upon which to wait for the operation to complete.
/// </returns>
template <typename _Value_type, int _Rank> concurrency::completion_future copy_async(const void * _Src, unsigned int _Src_byte_size, texture<_Value_type, _Rank>& _Dst)
{
    auto _Async_op_id = concurrency::details::_Get_amp_trace()->_Launch_async_copy_event_helper(nullptr, 
                                                                                   concurrency::details::_Get_texture_descriptor(_Dst), 
                                                                                   _Dst.get_data_length());

    _Event _Ev = details::_Copy_async_impl(_Src, _Src_byte_size, _Dst);

    return concurrency::details::_Get_amp_trace()->_Start_async_op_wait_event_helper(_Async_op_id, _Ev);
}

/// <summary>
///     Copies the contents of the source host buffer into the destination texture.
/// </summary>
/// <param name="_Rank">
///     The rank of the destination texture.
/// </param>
/// <param name="_Value_type">
///     The type of the elements of the destination texture.
/// </param>
/// <param name="_Src">
///     The source host buffer.
/// </param>
/// <param name="_Src_byte_size">
///     Number of bytes in the source buffer.
/// </param>
/// <param name="_Dst">
///     The destination texture.
/// </param>
template <typename _Value_type, int _Rank> void copy(const void * _Src, unsigned int _Src_byte_size, texture<_Value_type, _Rank>& _Dst)
{
    auto _Span_id = concurrency::details::_Get_amp_trace()->_Start_copy_event_helper(nullptr, 
                                                                        concurrency::details::_Get_texture_descriptor(_Dst),
                                                                        _Dst.get_data_length());

    copy_async(_Src, _Src_byte_size, _Dst).get();

    concurrency::details::_Get_amp_trace()->_Write_end_event(_Span_id);
}

/// <summary>
///     Asynchronously copies the contents of the source host buffer into the destination texture viewed by _Dst.
/// </summary>
/// <param name="_Rank">
///     The rank of the destination texture.
/// </param>
/// <param name="_Value_type">
///     The type of the elements of the destination texture.
/// </param>
/// <param name="_Src">
///     The source host buffer.
/// </param>
/// <param name="_Src_byte_size">
///     Number of bytes in the source buffer.
/// </param>
/// <param name="_Dst">
///     A writeonly_texture_view.
/// </param>
/// <returns>
///     A future upon which to wait for the operation to complete.
/// </returns>
template <typename _Value_type, int _Rank> concurrency::completion_future copy_async(const void * _Src, unsigned int _Src_byte_size, const writeonly_texture_view<_Value_type, _Rank>& _Dst)
{
    auto _Async_op_id = concurrency::details::_Get_amp_trace()->_Launch_async_copy_event_helper(nullptr, 
                                                                                   concurrency::details::_Get_texture_descriptor(_Dst), 
                                                                                   _Dst.get_data_length());

    _Event _Ev = details::_Copy_async_impl(_Src, _Src_byte_size, _Dst);

    return concurrency::details::_Get_amp_trace()->_Start_async_op_wait_event_helper(_Async_op_id, _Ev);
}

/// <summary>
///     Copies the contents of the source host buffer into the destination texture viewed by _Dst.
/// </summary>
/// <param name="_Rank">
///     The rank of the destination texture.
/// </param>
/// <param name="_Value_type">
///     The type of the elements of the destination texture.
/// </param>
/// <param name="_Src">
///     The source host buffer.
/// </param>
/// <param name="_Src_byte_size">
///     Number of bytes in the source buffer.
/// </param>
/// <param name="_Dst">
///     A writeonly_texture_view.
/// </param>
template <typename _Value_type, int _Rank> void copy(const void * _Src, unsigned int _Src_byte_size, const writeonly_texture_view<_Value_type, _Rank>& _Dst)
{
    auto _Span_id = concurrency::details::_Get_amp_trace()->_Start_copy_event_helper(nullptr,
                                                                        concurrency::details::_Get_texture_descriptor(_Dst),
                                                                        _Dst.get_data_length());

    copy_async(_Src, _Src_byte_size, _Dst).get();

    concurrency::details::_Get_amp_trace()->_Write_end_event(_Span_id);
}

namespace details
{
template<int _Rank>
Concurrency::extent<_Rank> _Make_texture(const Concurrency::accelerator_view &_Av, _In_ IUnknown *_D3D_texture, _Texture_base_type_id _Id, _Inout_ _Texture ** _Tex) __CPU_ONLY
{
    if (_D3D_texture == NULL)
    {
        throw runtime_exception("NULL D3D texture pointer.", E_INVALIDARG);
    }

    if (!Concurrency::details::_Is_D3D_accelerator_view(_Av)) {
        throw runtime_exception("Cannot create D3D texture on a non-D3D accelerator_view.", E_INVALIDARG);
    }

    _Texture * _Tex_ptr = _Texture::_Create_texture(_Rank, _Id, _D3D_texture, _Av);
    Concurrency::extent<_Rank> _Ext = Concurrency::graphics::details::_Create_extent<_Rank>(_Tex_ptr->_Get_width(), _Tex_ptr->_Get_height(), _Tex_ptr->_Get_depth());
    *_Tex = _Tex_ptr;
    return _Ext;
}

#pragma warning( pop )
} // namespace details

namespace direct3d
{
    /// <summary>
    ///     Get the D3D texture interface underlying an texture.
    /// </summary>
    /// <param name="_Rank">
    ///     The rank of the texture to get underlying D3D texture of.
    /// </param>
    /// <param name="_Value_type">
    ///     The type of the elements in the texture to get underlying D3D texture of.
    /// </param>
    /// <param name="_Texture">
    ///     A texture on a D3D accelerator_view for which the underlying D3D texture interface is returned.
    /// </param>
    /// <returns>
    ///     The IUnknown interface pointer corresponding to the D3D texture underlying the texture.
    /// </returns>
    template<typename _Value_type, int _Rank> _Ret_ IUnknown *get_texture(const texture<_Value_type, _Rank> &_Texture) __CPU_ONLY
    {
        return Concurrency::details::_D3D_interop::_Get_D3D_texture(Concurrency::details::_Get_texture(_Texture));
    }

    /// <summary>
    ///     Get the D3D texture interface underlying an texture viewed by a writeonly_texture_view.
    /// </summary>
    /// <param name="_Rank">
    ///     The rank of the texture to get underlying D3D texture of.
    /// </param>
    /// <param name="_Value_type">
    ///     The type of the elements in the texture to get underlying D3D texture of.
    /// </param>
    /// <param name="_Texture">
    ///     A writeonly_texture_view of a texture on a D3D accelerator_view for which the underlying D3D texture interface is returned.
    /// </param>
    /// <returns>
    ///     The IUnknown interface pointer corresponding to the D3D texture underlying the texture.
    /// </returns>
    template<typename _Value_type, int _Rank> _Ret_ IUnknown *get_texture(const writeonly_texture_view<_Value_type, _Rank> &_Texture) __CPU_ONLY
    {
        return Concurrency::details::_D3D_interop::_Get_D3D_buffer(Concurrency::details::_Get_texture(_Texture));
    }

    /// <summary>
    ///     Create an texture from a D3D texture interface pointer.
    /// </summary>
    /// <param name="_Rank">
    ///     The rank of the texture to be created from the D3D texture.
    /// </param>
    /// <param name="_Value_type">
    ///     The type of the elements of the texture to be created from the D3D texture.
    /// </param>
    /// <param name="_Av">
    ///     A D3D accelerator view on which the texture is to be created.
    /// </param>
    /// <param name="_D3D_texture">
    ///     IUnknown interface pointer of the D3D texture to create the texture from. 
    /// </param>
    /// <returns>
    ///     A texture using the provided D3D texture.
    /// </returns>
    template<typename _Value_type, int _Rank> texture<_Value_type, _Rank> make_texture(const Concurrency::accelerator_view &_Av, _In_ IUnknown *_D3D_texture) __CPU_ONLY
    {
        _Texture * _Tex_ptr = NULL;
#pragma warning( suppress: 6326 ) // Potential comparison of a constant with another constant       
        Concurrency::extent<_Rank> _Ext = Concurrency::graphics::details::_Make_texture<_Rank>(_Av, _D3D_texture,
         _Short_vector_type_traits<_Value_type>::_Format_base_type_id == _Double_type ? _Uint_type : _Short_vector_type_traits<_Value_type>::_Format_base_type_id,
         &_Tex_ptr);
        _ASSERTE(_Tex_ptr);
        return texture<_Value_type, _Rank>(_Ext, _Texture_descriptor(_Tex_ptr));
    }

} // namespace direct3d

} //namespace graphics
} //namespace Concurrency



