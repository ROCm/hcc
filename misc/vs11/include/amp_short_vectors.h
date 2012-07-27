/***
* ==++==
*
* Copyright (c) Microsoft Corporation.  All rights reserved.
*
* ==--==
* =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
*
* amp_short_vectors.h
*
* C++ AMP Short Vector Types
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/


// !!! DO NOT HAND EDIT !!!
// This file was generated.

#pragma once

#pragma warning(push)
#pragma warning(disable : 4100)
#include <amp.h>
#define _AMP_SHORT_VECTORS_H
namespace Concurrency
{
namespace graphics
{
class unorm;
class norm;

/// <summary>
///     Represent a unorm number.
///     Each element is a floating point number in the range of [0.0f, 1.0f].
/// </summary>
class unorm
{
    friend class norm;
private:
    float _Value;
    void _Set(float _Val) __CPU_ONLY
    {
        _Val = _Val < 0.0f ? 0.0f : _Val;
        _Val = _Val > 1.0f ? 1.0f : _Val;
        _Value = _Val;
    }

    void _Set(float _Val) __GPU_ONLY
    {
        _Value = Concurrency::direct3d::clamp(_Val, 0.0f, 1.0f);
    }
public:

    /// <summary>
    ///     Default constructor. Initialize to 0.0f.
    /// </summary>
    unorm(void) __GPU
    {
        _Value = 0.0f;
    }

    /// <summary>
    ///     Constructor. Initialize by clamping _V to the range of [0.0f, 1.0f].
    /// </summary>
    /// <param name="_V">
    ///     The value used to initialize.
    /// </param>
    explicit unorm(float _V) __GPU
    {
        _Set(_V);
    }

    /// <summary>
    ///     Constructor. Initialize by casting _V to float, then clamping to the range of [0.0f, 1.0f].
    /// </summary>
    /// <param name="_V">
    ///     The value used to initialize.
    /// </param>
    explicit unorm(unsigned int _V) __GPU
    {
        _Set(static_cast<float>(_V));
    }

    /// <summary>
    ///     Constructor. Initialize by casting _V to float, then clamping to the range of [0.0f, 1.0f].
    /// </summary>
    /// <param name="_V">
    ///     The value used to initialize.
    /// </param>
    explicit unorm(int _V) __GPU
    {
        _Set(static_cast<float>(_V));
    }

    /// <summary>
    ///     Constructor. Initialize by casting _V to float, then clamping to the range of [0.0f, 1.0f].
    /// </summary>
    /// <param name="_V">
    ///     The value used to initialize.
    /// </param>
    explicit unorm(double _V) __GPU
    {
        _Set(static_cast<float>(_V));
    }

    /// <summary>
    ///     Copy constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object to copy from.
    /// </param>
    unorm(const unorm& _Other) __GPU
    {
        _Value = _Other._Value;
    }

    /// <summary>
    ///     Constructor. Initialize by clamping _Other to the range of [0.0f, 1.0f].
    /// </summary>
    /// <param name="_Other">
    ///     The norm object used to initialize.
    /// </param>
    inline explicit unorm(const norm& _Other) __GPU;

    unorm& operator=(const unorm& _Other) __GPU
    {
        _Value = _Other._Value;
        return *this;
    }

    /// <summary>
    ///     Conversion operator. Convert the unorm number to a floating point value.
    /// </summary>
    operator float(void) const __GPU
    {
        return _Value;
    }

    unorm& operator+=(const unorm& _Other) __GPU
    {
        float _Res = _Value;
        _Res += _Other._Value;
        _Set(_Res);
        return *this;
    }

    unorm& operator-=(const unorm& _Other) __GPU
    {
        float _Res = _Value;
        _Res -= _Other._Value;
        _Set(_Res);
        return *this;
    }

    unorm& operator*=(const unorm& _Other) __GPU
    {
        float _Res = _Value;
        _Res *= _Other._Value;
        _Set(_Res);
        return *this;
    }

    unorm& operator/=(const unorm& _Other) __GPU
    {
        float _Res = _Value;
        _Res /= _Other._Value;
        _Set(_Res);
        return *this;
    }

    unorm& operator++() __GPU
    {
        float _Res = _Value;
        ++_Res;
        _Set(_Res);
        return *this;
    }

    unorm operator++(int) __GPU
    {
        unorm _Res = *this;
        ++(*this);
        return _Res;
    }

    unorm& operator--() __GPU
    {
        float _Res = _Value;
        --_Res;
        _Set(_Res);
        return *this;
    }

    unorm operator--(int) __GPU
    {
        unorm _Res = *this;
        --(*this);
        return _Res;
    }

};

/// <summary>
///     Represent a norm number.
///     Each element is a floating point number in the range of [-1.0f, 1.0f].
/// </summary>
class norm
{
    friend class unorm;
private:
    float _Value;
    void _Set(float _Val) __CPU_ONLY
    {
        _Val = _Val < -1.0f ? -1.0f : _Val;
        _Val = _Val > 1.0f ? 1.0f : _Val;
        _Value = _Val;
    }

    void _Set(float _Val) __GPU_ONLY
    {
        _Value = Concurrency::direct3d::clamp(_Val, -1.0f, 1.0f);
    }
public:

    /// <summary>
    ///     Default constructor. Initialize to 0.0f.
    /// </summary>
    norm(void) __GPU
    {
        _Value = 0.0f;
    }

    /// <summary>
    ///     Constructor. Initialize by clamping _V to the range of [-1.0f, 1.0f].
    /// </summary>
    /// <param name="_V">
    ///     The value used to initialize.
    /// </param>
    explicit norm(float _V) __GPU
    {
        _Set(_V);
    }

    /// <summary>
    ///     Constructor. Initialize by casting _V to float, then clamping to the range of [-1.0f, 1.0f].
    /// </summary>
    /// <param name="_V">
    ///     The value used to initialize.
    /// </param>
    explicit norm(unsigned int _V) __GPU
    {
        _Set(static_cast<float>(_V));
    }

    /// <summary>
    ///     Constructor. Initialize by casting _V to float, then clamping to the range of [-1.0f, 1.0f].
    /// </summary>
    /// <param name="_V">
    ///     The value used to initialize.
    /// </param>
    explicit norm(int _V) __GPU
    {
        _Set(static_cast<float>(_V));
    }

    /// <summary>
    ///     Constructor. Initialize by casting _V to float, then clamping to the range of [-1.0f, 1.0f].
    /// </summary>
    /// <param name="_V">
    ///     The value used to initialize.
    /// </param>
    explicit norm(double _V) __GPU
    {
        _Set(static_cast<float>(_V));
    }

    /// <summary>
    ///     Copy constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object to copy from.
    /// </param>
    norm(const norm& _Other) __GPU
    {
        _Value = _Other._Value;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    norm(const unorm& _Other) __GPU
    {
        _Value = _Other._Value;
    }

    norm& operator=(const norm& _Other) __GPU
    {
        _Value = _Other._Value;
        return *this;
    }

    /// <summary>
    ///     Conversion operator. Convert the norm number to a floating point value.
    /// </summary>
    operator float(void) const __GPU
    {
        return _Value;
    }

    norm& operator+=(const norm& _Other) __GPU
    {
        float _Res = _Value;
        _Res += _Other._Value;
        _Set(_Res);
        return *this;
    }

    norm& operator-=(const norm& _Other) __GPU
    {
        float _Res = _Value;
        _Res -= _Other._Value;
        _Set(_Res);
        return *this;
    }

    norm& operator*=(const norm& _Other) __GPU
    {
        float _Res = _Value;
        _Res *= _Other._Value;
        _Set(_Res);
        return *this;
    }

    norm& operator/=(const norm& _Other) __GPU
    {
        float _Res = _Value;
        _Res /= _Other._Value;
        _Set(_Res);
        return *this;
    }

    norm& operator++() __GPU
    {
        float _Res = _Value;
        ++_Res;
        _Set(_Res);
        return *this;
    }

    norm operator++(int) __GPU
    {
        norm _Res = *this;
        ++(*this);
        return _Res;
    }

    norm& operator--() __GPU
    {
        float _Res = _Value;
        --_Res;
        _Set(_Res);
        return *this;
    }

    norm operator--(int) __GPU
    {
        norm _Res = *this;
        --(*this);
        return _Res;
    }

    norm operator-(void) const __GPU
    {
        norm _Ret;
        _Ret._Value = -_Value;
        return _Ret;
    }

};

unorm::unorm(const norm& _Other) __GPU
{
    _Set(_Other._Value);
}

inline unorm operator+(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return unorm(float(_Lhs) + float(_Rhs));
}

inline norm operator+(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return norm(float(_Lhs) + float(_Rhs));
}

inline unorm operator-(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return unorm(float(_Lhs) - float(_Rhs));
}

inline norm operator-(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return norm(float(_Lhs) - float(_Rhs));
}

inline unorm operator*(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return unorm(float(_Lhs) * float(_Rhs));
}

inline norm operator*(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return norm(float(_Lhs) * float(_Rhs));
}

inline unorm operator/(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return unorm(float(_Lhs) / float(_Rhs));
}

inline norm operator/(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return norm(float(_Lhs) / float(_Rhs));
}

inline bool operator==(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return float(_Lhs) == float(_Rhs);
}

inline bool operator==(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return float(_Lhs) == float(_Rhs);
}

inline bool operator!=(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return float(_Lhs) != float(_Rhs);
}

inline bool operator!=(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return float(_Lhs) != float(_Rhs);
}

inline bool operator>(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return float(_Lhs) > float(_Rhs);
}

inline bool operator>(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return float(_Lhs) > float(_Rhs);
}

inline bool operator<(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return float(_Lhs) < float(_Rhs);
}

inline bool operator<(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return float(_Lhs) < float(_Rhs);
}

inline bool operator>=(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return float(_Lhs) >= float(_Rhs);
}

inline bool operator>=(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return float(_Lhs) >= float(_Rhs);
}

inline bool operator<=(const unorm& _Lhs, const unorm& _Rhs) __GPU
{
    return float(_Lhs) <= float(_Rhs);
}

inline bool operator<=(const norm& _Lhs, const norm& _Rhs) __GPU
{
    return float(_Lhs) <= float(_Rhs);
}

#define UNORM_ZERO ((concurrency::graphics::unorm)0.0f)
#define UNORM_MIN ((concurrency::graphics::unorm)0.0f)
#define UNORM_MAX ((concurrency::graphics::unorm)1.0f)
#define NORM_ZERO ((concurrency::graphics::norm)0.0f)
#define NORM_MIN  ((concurrency::graphics::norm)-1.0f)
#define NORM_MAX  ((concurrency::graphics::norm)1.0f)


typedef unsigned int uint;
// Forward Declarations
class uint_2;
class uint_3;
class uint_4;
class int_2;
class int_3;
class int_4;
class float_2;
class float_3;
class float_4;
class unorm_2;
class unorm_3;
class unorm_4;
class norm_2;
class norm_3;
class norm_4;
class double_2;
class double_3;
class double_4;

/// <summary>
///    Represent a short vector of 2 unsigned int's.
/// </summary>
class uint_2
{
public:
    typedef unsigned int value_type;
    static const int size = 2;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Uint_type;
private:
    value_type _M_x;
    value_type _M_y;

public:
    __declspec( property( get = get_x, put = set_x) ) unsigned int x;
    __declspec( property( get = get_x, put = set_x) ) unsigned int r;

    unsigned int get_x() const __GPU {
        return _M_x;
    }

    void set_x(unsigned int _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) unsigned int y;
    __declspec( property( get = get_y, put = set_y) ) unsigned int g;

    unsigned int get_y() const __GPU {
        return _M_y;
    }

    void set_y(unsigned int _Value) __GPU
    {
        _M_y = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    uint_2() __GPU
    {
        _M_x = 0;
        _M_y = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    uint_2(unsigned int _V0, unsigned int _V1) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    uint_2(unsigned int _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_2(const int_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_2(const float_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_2(const unorm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_2(const norm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_2(const double_2& _Other) __GPU;

    uint_2 operator~() const __GPU
    {
        uint_2 _Value = *this;
        return uint_2(~_Value.x, ~_Value.y);
    }

    uint_2& operator++() __GPU
    {
        uint_2 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        *this = _Value;
        return *this;
    }

    uint_2 operator++(int) __GPU
    {
        uint_2 _Result = *this;
        ++(*this);
        return _Result;
    }

    uint_2& operator--() __GPU
    {
        uint_2 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        *this = _Value;
        return *this;
    }

    uint_2 operator--(int) __GPU
    {
        uint_2 _Result = *this;
        --(*this);
        return _Result;
    }

    uint_2& operator+=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator-=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator*=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator/=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator%=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x %= _Value2.x;
        _Value1.y %= _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator^=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x ^= _Value2.x;
        _Value1.y ^= _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator|=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x |= _Value2.x;
        _Value1.y |= _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator&=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x &= _Value2.x;
        _Value1.y &= _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator>>=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x >>= _Value2.x;
        _Value1.y >>= _Value2.y;
        *this = _Value1;
        return *this;
    }

    uint_2& operator<<=(const uint_2& _Other) __GPU
    {
        uint_2 _Value1 = *this;
        uint_2 _Value2 = _Other;
        _Value1.x <<= _Value2.x;
        _Value1.y <<= _Value2.y;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) uint_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) uint_2 rg;

    uint_2 get_xy() const __GPU {
        return uint_2(_M_x, _M_y);
    }

    void set_xy(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) uint_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) uint_2 gr;

    uint_2 get_yx() const __GPU {
        return uint_2(_M_y, _M_x);
    }

    void set_yx(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

};

/// <summary>
///    Represent a short vector of 3 unsigned int's.
/// </summary>
class uint_3
{
public:
    typedef unsigned int value_type;
    static const int size = 3;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Uint_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;

public:
    __declspec( property( get = get_x, put = set_x) ) unsigned int x;
    __declspec( property( get = get_x, put = set_x) ) unsigned int r;

    unsigned int get_x() const __GPU {
        return _M_x;
    }

    void set_x(unsigned int _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) unsigned int y;
    __declspec( property( get = get_y, put = set_y) ) unsigned int g;

    unsigned int get_y() const __GPU {
        return _M_y;
    }

    void set_y(unsigned int _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) unsigned int z;
    __declspec( property( get = get_z, put = set_z) ) unsigned int b;

    unsigned int get_z() const __GPU {
        return _M_z;
    }

    void set_z(unsigned int _Value) __GPU
    {
        _M_z = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    uint_3() __GPU
    {
        _M_x = 0;
        _M_y = 0;
        _M_z = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    uint_3(unsigned int _V0, unsigned int _V1, unsigned int _V2) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    uint_3(unsigned int _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_3(const int_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_3(const float_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_3(const unorm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_3(const norm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_3(const double_3& _Other) __GPU;

    uint_3 operator~() const __GPU
    {
        uint_3 _Value = *this;
        return uint_3(~_Value.x, ~_Value.y, ~_Value.z);
    }

    uint_3& operator++() __GPU
    {
        uint_3 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        *this = _Value;
        return *this;
    }

    uint_3 operator++(int) __GPU
    {
        uint_3 _Result = *this;
        ++(*this);
        return _Result;
    }

    uint_3& operator--() __GPU
    {
        uint_3 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        *this = _Value;
        return *this;
    }

    uint_3 operator--(int) __GPU
    {
        uint_3 _Result = *this;
        --(*this);
        return _Result;
    }

    uint_3& operator+=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator-=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator*=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator/=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator%=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x %= _Value2.x;
        _Value1.y %= _Value2.y;
        _Value1.z %= _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator^=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x ^= _Value2.x;
        _Value1.y ^= _Value2.y;
        _Value1.z ^= _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator|=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x |= _Value2.x;
        _Value1.y |= _Value2.y;
        _Value1.z |= _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator&=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x &= _Value2.x;
        _Value1.y &= _Value2.y;
        _Value1.z &= _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator>>=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x >>= _Value2.x;
        _Value1.y >>= _Value2.y;
        _Value1.z >>= _Value2.z;
        *this = _Value1;
        return *this;
    }

    uint_3& operator<<=(const uint_3& _Other) __GPU
    {
        uint_3 _Value1 = *this;
        uint_3 _Value2 = _Other;
        _Value1.x <<= _Value2.x;
        _Value1.y <<= _Value2.y;
        _Value1.z <<= _Value2.z;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) uint_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) uint_2 rg;

    uint_2 get_xy() const __GPU {
        return uint_2(_M_x, _M_y);
    }

    void set_xy(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) uint_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) uint_2 rb;

    uint_2 get_xz() const __GPU {
        return uint_2(_M_x, _M_z);
    }

    void set_xz(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) uint_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) uint_2 gr;

    uint_2 get_yx() const __GPU {
        return uint_2(_M_y, _M_x);
    }

    void set_yx(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) uint_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) uint_2 gb;

    uint_2 get_yz() const __GPU {
        return uint_2(_M_y, _M_z);
    }

    void set_yz(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) uint_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) uint_2 br;

    uint_2 get_zx() const __GPU {
        return uint_2(_M_z, _M_x);
    }

    void set_zx(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) uint_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) uint_2 bg;

    uint_2 get_zy() const __GPU {
        return uint_2(_M_z, _M_y);
    }

    void set_zy(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) uint_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) uint_3 rgb;

    uint_3 get_xyz() const __GPU {
        return uint_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) uint_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) uint_3 rbg;

    uint_3 get_xzy() const __GPU {
        return uint_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) uint_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) uint_3 grb;

    uint_3 get_yxz() const __GPU {
        return uint_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) uint_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) uint_3 gbr;

    uint_3 get_yzx() const __GPU {
        return uint_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) uint_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) uint_3 brg;

    uint_3 get_zxy() const __GPU {
        return uint_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) uint_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) uint_3 bgr;

    uint_3 get_zyx() const __GPU {
        return uint_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

};

/// <summary>
///    Represent a short vector of 4 unsigned int's.
/// </summary>
class uint_4
{
public:
    typedef unsigned int value_type;
    static const int size = 4;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Uint_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;
    value_type _M_w;

public:
    __declspec( property( get = get_x, put = set_x) ) unsigned int x;
    __declspec( property( get = get_x, put = set_x) ) unsigned int r;

    unsigned int get_x() const __GPU {
        return _M_x;
    }

    void set_x(unsigned int _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) unsigned int y;
    __declspec( property( get = get_y, put = set_y) ) unsigned int g;

    unsigned int get_y() const __GPU {
        return _M_y;
    }

    void set_y(unsigned int _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) unsigned int z;
    __declspec( property( get = get_z, put = set_z) ) unsigned int b;

    unsigned int get_z() const __GPU {
        return _M_z;
    }

    void set_z(unsigned int _Value) __GPU
    {
        _M_z = _Value;
    }

    __declspec( property( get = get_w, put = set_w) ) unsigned int w;
    __declspec( property( get = get_w, put = set_w) ) unsigned int a;

    unsigned int get_w() const __GPU {
        return _M_w;
    }

    void set_w(unsigned int _Value) __GPU
    {
        _M_w = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    uint_4() __GPU
    {
        _M_x = 0;
        _M_y = 0;
        _M_z = 0;
        _M_w = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    uint_4(unsigned int _V0, unsigned int _V1, unsigned int _V2, unsigned int _V3) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
        _M_w = _V3;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    uint_4(unsigned int _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
        _M_w = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_4(const int_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_4(const float_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_4(const unorm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_4(const norm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline uint_4(const double_4& _Other) __GPU;

    uint_4 operator~() const __GPU
    {
        uint_4 _Value = *this;
        return uint_4(~_Value.x, ~_Value.y, ~_Value.z, ~_Value.w);
    }

    uint_4& operator++() __GPU
    {
        uint_4 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        ++_Value._M_w;
        *this = _Value;
        return *this;
    }

    uint_4 operator++(int) __GPU
    {
        uint_4 _Result = *this;
        ++(*this);
        return _Result;
    }

    uint_4& operator--() __GPU
    {
        uint_4 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        --_Value._M_w;
        *this = _Value;
        return *this;
    }

    uint_4 operator--(int) __GPU
    {
        uint_4 _Result = *this;
        --(*this);
        return _Result;
    }

    uint_4& operator+=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        _Value1.w += _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator-=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        _Value1.w -= _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator*=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        _Value1.w *= _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator/=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        _Value1.w /= _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator%=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x %= _Value2.x;
        _Value1.y %= _Value2.y;
        _Value1.z %= _Value2.z;
        _Value1.w %= _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator^=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x ^= _Value2.x;
        _Value1.y ^= _Value2.y;
        _Value1.z ^= _Value2.z;
        _Value1.w ^= _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator|=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x |= _Value2.x;
        _Value1.y |= _Value2.y;
        _Value1.z |= _Value2.z;
        _Value1.w |= _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator&=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x &= _Value2.x;
        _Value1.y &= _Value2.y;
        _Value1.z &= _Value2.z;
        _Value1.w &= _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator>>=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x >>= _Value2.x;
        _Value1.y >>= _Value2.y;
        _Value1.z >>= _Value2.z;
        _Value1.w >>= _Value2.w;
        *this = _Value1;
        return *this;
    }

    uint_4& operator<<=(const uint_4& _Other) __GPU
    {
        uint_4 _Value1 = *this;
        uint_4 _Value2 = _Other;
        _Value1.x <<= _Value2.x;
        _Value1.y <<= _Value2.y;
        _Value1.z <<= _Value2.z;
        _Value1.w <<= _Value2.w;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) uint_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) uint_2 rg;

    uint_2 get_xy() const __GPU {
        return uint_2(_M_x, _M_y);
    }

    void set_xy(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) uint_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) uint_2 rb;

    uint_2 get_xz() const __GPU {
        return uint_2(_M_x, _M_z);
    }

    void set_xz(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xw, put = set_xw) ) uint_2 xw;
    __declspec( property( get = get_xw, put = set_xw) ) uint_2 ra;

    uint_2 get_xw() const __GPU {
        return uint_2(_M_x, _M_w);
    }

    void set_xw(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) uint_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) uint_2 gr;

    uint_2 get_yx() const __GPU {
        return uint_2(_M_y, _M_x);
    }

    void set_yx(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) uint_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) uint_2 gb;

    uint_2 get_yz() const __GPU {
        return uint_2(_M_y, _M_z);
    }

    void set_yz(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yw, put = set_yw) ) uint_2 yw;
    __declspec( property( get = get_yw, put = set_yw) ) uint_2 ga;

    uint_2 get_yw() const __GPU {
        return uint_2(_M_y, _M_w);
    }

    void set_yw(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) uint_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) uint_2 br;

    uint_2 get_zx() const __GPU {
        return uint_2(_M_z, _M_x);
    }

    void set_zx(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) uint_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) uint_2 bg;

    uint_2 get_zy() const __GPU {
        return uint_2(_M_z, _M_y);
    }

    void set_zy(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_zw, put = set_zw) ) uint_2 zw;
    __declspec( property( get = get_zw, put = set_zw) ) uint_2 ba;

    uint_2 get_zw() const __GPU {
        return uint_2(_M_z, _M_w);
    }

    void set_zw(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_wx, put = set_wx) ) uint_2 wx;
    __declspec( property( get = get_wx, put = set_wx) ) uint_2 ar;

    uint_2 get_wx() const __GPU {
        return uint_2(_M_w, _M_x);
    }

    void set_wx(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_wy, put = set_wy) ) uint_2 wy;
    __declspec( property( get = get_wy, put = set_wy) ) uint_2 ag;

    uint_2 get_wy() const __GPU {
        return uint_2(_M_w, _M_y);
    }

    void set_wy(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_wz, put = set_wz) ) uint_2 wz;
    __declspec( property( get = get_wz, put = set_wz) ) uint_2 ab;

    uint_2 get_wz() const __GPU {
        return uint_2(_M_w, _M_z);
    }

    void set_wz(const uint_2& _Value) __GPU
    {
        uint_2 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) uint_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) uint_3 rgb;

    uint_3 get_xyz() const __GPU {
        return uint_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xyw, put = set_xyw) ) uint_3 xyw;
    __declspec( property( get = get_xyw, put = set_xyw) ) uint_3 rga;

    uint_3 get_xyw() const __GPU {
        return uint_3(_M_x, _M_y, _M_w);
    }

    void set_xyw(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) uint_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) uint_3 rbg;

    uint_3 get_xzy() const __GPU {
        return uint_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xzw, put = set_xzw) ) uint_3 xzw;
    __declspec( property( get = get_xzw, put = set_xzw) ) uint_3 rba;

    uint_3 get_xzw() const __GPU {
        return uint_3(_M_x, _M_z, _M_w);
    }

    void set_xzw(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xwy, put = set_xwy) ) uint_3 xwy;
    __declspec( property( get = get_xwy, put = set_xwy) ) uint_3 rag;

    uint_3 get_xwy() const __GPU {
        return uint_3(_M_x, _M_w, _M_y);
    }

    void set_xwy(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xwz, put = set_xwz) ) uint_3 xwz;
    __declspec( property( get = get_xwz, put = set_xwz) ) uint_3 rab;

    uint_3 get_xwz() const __GPU {
        return uint_3(_M_x, _M_w, _M_z);
    }

    void set_xwz(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) uint_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) uint_3 grb;

    uint_3 get_yxz() const __GPU {
        return uint_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxw, put = set_yxw) ) uint_3 yxw;
    __declspec( property( get = get_yxw, put = set_yxw) ) uint_3 gra;

    uint_3 get_yxw() const __GPU {
        return uint_3(_M_y, _M_x, _M_w);
    }

    void set_yxw(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) uint_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) uint_3 gbr;

    uint_3 get_yzx() const __GPU {
        return uint_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_yzw, put = set_yzw) ) uint_3 yzw;
    __declspec( property( get = get_yzw, put = set_yzw) ) uint_3 gba;

    uint_3 get_yzw() const __GPU {
        return uint_3(_M_y, _M_z, _M_w);
    }

    void set_yzw(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_ywx, put = set_ywx) ) uint_3 ywx;
    __declspec( property( get = get_ywx, put = set_ywx) ) uint_3 gar;

    uint_3 get_ywx() const __GPU {
        return uint_3(_M_y, _M_w, _M_x);
    }

    void set_ywx(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_ywz, put = set_ywz) ) uint_3 ywz;
    __declspec( property( get = get_ywz, put = set_ywz) ) uint_3 gab;

    uint_3 get_ywz() const __GPU {
        return uint_3(_M_y, _M_w, _M_z);
    }

    void set_ywz(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) uint_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) uint_3 brg;

    uint_3 get_zxy() const __GPU {
        return uint_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zxw, put = set_zxw) ) uint_3 zxw;
    __declspec( property( get = get_zxw, put = set_zxw) ) uint_3 bra;

    uint_3 get_zxw() const __GPU {
        return uint_3(_M_z, _M_x, _M_w);
    }

    void set_zxw(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) uint_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) uint_3 bgr;

    uint_3 get_zyx() const __GPU {
        return uint_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zyw, put = set_zyw) ) uint_3 zyw;
    __declspec( property( get = get_zyw, put = set_zyw) ) uint_3 bga;

    uint_3 get_zyw() const __GPU {
        return uint_3(_M_z, _M_y, _M_w);
    }

    void set_zyw(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zwx, put = set_zwx) ) uint_3 zwx;
    __declspec( property( get = get_zwx, put = set_zwx) ) uint_3 bar;

    uint_3 get_zwx() const __GPU {
        return uint_3(_M_z, _M_w, _M_x);
    }

    void set_zwx(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zwy, put = set_zwy) ) uint_3 zwy;
    __declspec( property( get = get_zwy, put = set_zwy) ) uint_3 bag;

    uint_3 get_zwy() const __GPU {
        return uint_3(_M_z, _M_w, _M_y);
    }

    void set_zwy(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxy, put = set_wxy) ) uint_3 wxy;
    __declspec( property( get = get_wxy, put = set_wxy) ) uint_3 arg;

    uint_3 get_wxy() const __GPU {
        return uint_3(_M_w, _M_x, _M_y);
    }

    void set_wxy(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxz, put = set_wxz) ) uint_3 wxz;
    __declspec( property( get = get_wxz, put = set_wxz) ) uint_3 arb;

    uint_3 get_wxz() const __GPU {
        return uint_3(_M_w, _M_x, _M_z);
    }

    void set_wxz(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wyx, put = set_wyx) ) uint_3 wyx;
    __declspec( property( get = get_wyx, put = set_wyx) ) uint_3 agr;

    uint_3 get_wyx() const __GPU {
        return uint_3(_M_w, _M_y, _M_x);
    }

    void set_wyx(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wyz, put = set_wyz) ) uint_3 wyz;
    __declspec( property( get = get_wyz, put = set_wyz) ) uint_3 agb;

    uint_3 get_wyz() const __GPU {
        return uint_3(_M_w, _M_y, _M_z);
    }

    void set_wyz(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wzx, put = set_wzx) ) uint_3 wzx;
    __declspec( property( get = get_wzx, put = set_wzx) ) uint_3 abr;

    uint_3 get_wzx() const __GPU {
        return uint_3(_M_w, _M_z, _M_x);
    }

    void set_wzx(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wzy, put = set_wzy) ) uint_3 wzy;
    __declspec( property( get = get_wzy, put = set_wzy) ) uint_3 abg;

    uint_3 get_wzy() const __GPU {
        return uint_3(_M_w, _M_z, _M_y);
    }

    void set_wzy(const uint_3& _Value) __GPU
    {
        uint_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xyzw, put = set_xyzw) ) uint_4 xyzw;
    __declspec( property( get = get_xyzw, put = set_xyzw) ) uint_4 rgba;

    uint_4 get_xyzw() const __GPU {
        return uint_4(_M_x, _M_y, _M_z, _M_w);
    }

    void set_xyzw(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xywz, put = set_xywz) ) uint_4 xywz;
    __declspec( property( get = get_xywz, put = set_xywz) ) uint_4 rgab;

    uint_4 get_xywz() const __GPU {
        return uint_4(_M_x, _M_y, _M_w, _M_z);
    }

    void set_xywz(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xzyw, put = set_xzyw) ) uint_4 xzyw;
    __declspec( property( get = get_xzyw, put = set_xzyw) ) uint_4 rbga;

    uint_4 get_xzyw() const __GPU {
        return uint_4(_M_x, _M_z, _M_y, _M_w);
    }

    void set_xzyw(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xzwy, put = set_xzwy) ) uint_4 xzwy;
    __declspec( property( get = get_xzwy, put = set_xzwy) ) uint_4 rbag;

    uint_4 get_xzwy() const __GPU {
        return uint_4(_M_x, _M_z, _M_w, _M_y);
    }

    void set_xzwy(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_xwyz, put = set_xwyz) ) uint_4 xwyz;
    __declspec( property( get = get_xwyz, put = set_xwyz) ) uint_4 ragb;

    uint_4 get_xwyz() const __GPU {
        return uint_4(_M_x, _M_w, _M_y, _M_z);
    }

    void set_xwyz(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xwzy, put = set_xwzy) ) uint_4 xwzy;
    __declspec( property( get = get_xwzy, put = set_xwzy) ) uint_4 rabg;

    uint_4 get_xwzy() const __GPU {
        return uint_4(_M_x, _M_w, _M_z, _M_y);
    }

    void set_xwzy(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_yxzw, put = set_yxzw) ) uint_4 yxzw;
    __declspec( property( get = get_yxzw, put = set_yxzw) ) uint_4 grba;

    uint_4 get_yxzw() const __GPU {
        return uint_4(_M_y, _M_x, _M_z, _M_w);
    }

    void set_yxzw(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yxwz, put = set_yxwz) ) uint_4 yxwz;
    __declspec( property( get = get_yxwz, put = set_yxwz) ) uint_4 grab;

    uint_4 get_yxwz() const __GPU {
        return uint_4(_M_y, _M_x, _M_w, _M_z);
    }

    void set_yxwz(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_yzxw, put = set_yzxw) ) uint_4 yzxw;
    __declspec( property( get = get_yzxw, put = set_yzxw) ) uint_4 gbra;

    uint_4 get_yzxw() const __GPU {
        return uint_4(_M_y, _M_z, _M_x, _M_w);
    }

    void set_yzxw(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yzwx, put = set_yzwx) ) uint_4 yzwx;
    __declspec( property( get = get_yzwx, put = set_yzwx) ) uint_4 gbar;

    uint_4 get_yzwx() const __GPU {
        return uint_4(_M_y, _M_z, _M_w, _M_x);
    }

    void set_yzwx(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_ywxz, put = set_ywxz) ) uint_4 ywxz;
    __declspec( property( get = get_ywxz, put = set_ywxz) ) uint_4 garb;

    uint_4 get_ywxz() const __GPU {
        return uint_4(_M_y, _M_w, _M_x, _M_z);
    }

    void set_ywxz(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_ywzx, put = set_ywzx) ) uint_4 ywzx;
    __declspec( property( get = get_ywzx, put = set_ywzx) ) uint_4 gabr;

    uint_4 get_ywzx() const __GPU {
        return uint_4(_M_y, _M_w, _M_z, _M_x);
    }

    void set_ywzx(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zxyw, put = set_zxyw) ) uint_4 zxyw;
    __declspec( property( get = get_zxyw, put = set_zxyw) ) uint_4 brga;

    uint_4 get_zxyw() const __GPU {
        return uint_4(_M_z, _M_x, _M_y, _M_w);
    }

    void set_zxyw(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zxwy, put = set_zxwy) ) uint_4 zxwy;
    __declspec( property( get = get_zxwy, put = set_zxwy) ) uint_4 brag;

    uint_4 get_zxwy() const __GPU {
        return uint_4(_M_z, _M_x, _M_w, _M_y);
    }

    void set_zxwy(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zyxw, put = set_zyxw) ) uint_4 zyxw;
    __declspec( property( get = get_zyxw, put = set_zyxw) ) uint_4 bgra;

    uint_4 get_zyxw() const __GPU {
        return uint_4(_M_z, _M_y, _M_x, _M_w);
    }

    void set_zyxw(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zywx, put = set_zywx) ) uint_4 zywx;
    __declspec( property( get = get_zywx, put = set_zywx) ) uint_4 bgar;

    uint_4 get_zywx() const __GPU {
        return uint_4(_M_z, _M_y, _M_w, _M_x);
    }

    void set_zywx(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zwxy, put = set_zwxy) ) uint_4 zwxy;
    __declspec( property( get = get_zwxy, put = set_zwxy) ) uint_4 barg;

    uint_4 get_zwxy() const __GPU {
        return uint_4(_M_z, _M_w, _M_x, _M_y);
    }

    void set_zwxy(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zwyx, put = set_zwyx) ) uint_4 zwyx;
    __declspec( property( get = get_zwyx, put = set_zwyx) ) uint_4 bagr;

    uint_4 get_zwyx() const __GPU {
        return uint_4(_M_z, _M_w, _M_y, _M_x);
    }

    void set_zwyx(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wxyz, put = set_wxyz) ) uint_4 wxyz;
    __declspec( property( get = get_wxyz, put = set_wxyz) ) uint_4 argb;

    uint_4 get_wxyz() const __GPU {
        return uint_4(_M_w, _M_x, _M_y, _M_z);
    }

    void set_wxyz(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wxzy, put = set_wxzy) ) uint_4 wxzy;
    __declspec( property( get = get_wxzy, put = set_wxzy) ) uint_4 arbg;

    uint_4 get_wxzy() const __GPU {
        return uint_4(_M_w, _M_x, _M_z, _M_y);
    }

    void set_wxzy(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wyxz, put = set_wyxz) ) uint_4 wyxz;
    __declspec( property( get = get_wyxz, put = set_wyxz) ) uint_4 agrb;

    uint_4 get_wyxz() const __GPU {
        return uint_4(_M_w, _M_y, _M_x, _M_z);
    }

    void set_wyxz(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wyzx, put = set_wyzx) ) uint_4 wyzx;
    __declspec( property( get = get_wyzx, put = set_wyzx) ) uint_4 agbr;

    uint_4 get_wyzx() const __GPU {
        return uint_4(_M_w, _M_y, _M_z, _M_x);
    }

    void set_wyzx(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wzxy, put = set_wzxy) ) uint_4 wzxy;
    __declspec( property( get = get_wzxy, put = set_wzxy) ) uint_4 abrg;

    uint_4 get_wzxy() const __GPU {
        return uint_4(_M_w, _M_z, _M_x, _M_y);
    }

    void set_wzxy(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wzyx, put = set_wzyx) ) uint_4 wzyx;
    __declspec( property( get = get_wzyx, put = set_wzyx) ) uint_4 abgr;

    uint_4 get_wzyx() const __GPU {
        return uint_4(_M_w, _M_z, _M_y, _M_x);
    }

    void set_wzyx(const uint_4& _Value) __GPU
    {
        uint_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

};

/// <summary>
///    Represent a short vector of 2 int's.
/// </summary>
class int_2
{
public:
    typedef int value_type;
    static const int size = 2;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Int_type;
private:
    value_type _M_x;
    value_type _M_y;

public:
    __declspec( property( get = get_x, put = set_x) ) int x;
    __declspec( property( get = get_x, put = set_x) ) int r;

    int get_x() const __GPU {
        return _M_x;
    }

    void set_x(int _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) int y;
    __declspec( property( get = get_y, put = set_y) ) int g;

    int get_y() const __GPU {
        return _M_y;
    }

    void set_y(int _Value) __GPU
    {
        _M_y = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    int_2() __GPU
    {
        _M_x = 0;
        _M_y = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    int_2(int _V0, int _V1) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    int_2(int _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_2(const uint_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_2(const float_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_2(const unorm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_2(const norm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_2(const double_2& _Other) __GPU;

    int_2 operator~() const __GPU
    {
        int_2 _Value = *this;
        return int_2(~_Value.x, ~_Value.y);
    }

    int_2 operator-() const __GPU
    {
        int_2 _Value = *this;
        return int_2(-_Value.x, -_Value.y);
    }

    int_2& operator++() __GPU
    {
        int_2 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        *this = _Value;
        return *this;
    }

    int_2 operator++(int) __GPU
    {
        int_2 _Result = *this;
        ++(*this);
        return _Result;
    }

    int_2& operator--() __GPU
    {
        int_2 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        *this = _Value;
        return *this;
    }

    int_2 operator--(int) __GPU
    {
        int_2 _Result = *this;
        --(*this);
        return _Result;
    }

    int_2& operator+=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator-=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator*=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator/=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator%=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x %= _Value2.x;
        _Value1.y %= _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator^=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x ^= _Value2.x;
        _Value1.y ^= _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator|=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x |= _Value2.x;
        _Value1.y |= _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator&=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x &= _Value2.x;
        _Value1.y &= _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator>>=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x >>= _Value2.x;
        _Value1.y >>= _Value2.y;
        *this = _Value1;
        return *this;
    }

    int_2& operator<<=(const int_2& _Other) __GPU
    {
        int_2 _Value1 = *this;
        int_2 _Value2 = _Other;
        _Value1.x <<= _Value2.x;
        _Value1.y <<= _Value2.y;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) int_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) int_2 rg;

    int_2 get_xy() const __GPU {
        return int_2(_M_x, _M_y);
    }

    void set_xy(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) int_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) int_2 gr;

    int_2 get_yx() const __GPU {
        return int_2(_M_y, _M_x);
    }

    void set_yx(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

};

/// <summary>
///    Represent a short vector of 3 int's.
/// </summary>
class int_3
{
public:
    typedef int value_type;
    static const int size = 3;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Int_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;

public:
    __declspec( property( get = get_x, put = set_x) ) int x;
    __declspec( property( get = get_x, put = set_x) ) int r;

    int get_x() const __GPU {
        return _M_x;
    }

    void set_x(int _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) int y;
    __declspec( property( get = get_y, put = set_y) ) int g;

    int get_y() const __GPU {
        return _M_y;
    }

    void set_y(int _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) int z;
    __declspec( property( get = get_z, put = set_z) ) int b;

    int get_z() const __GPU {
        return _M_z;
    }

    void set_z(int _Value) __GPU
    {
        _M_z = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    int_3() __GPU
    {
        _M_x = 0;
        _M_y = 0;
        _M_z = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    int_3(int _V0, int _V1, int _V2) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    int_3(int _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_3(const uint_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_3(const float_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_3(const unorm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_3(const norm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_3(const double_3& _Other) __GPU;

    int_3 operator~() const __GPU
    {
        int_3 _Value = *this;
        return int_3(~_Value.x, ~_Value.y, ~_Value.z);
    }

    int_3 operator-() const __GPU
    {
        int_3 _Value = *this;
        return int_3(-_Value.x, -_Value.y, -_Value.z);
    }

    int_3& operator++() __GPU
    {
        int_3 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        *this = _Value;
        return *this;
    }

    int_3 operator++(int) __GPU
    {
        int_3 _Result = *this;
        ++(*this);
        return _Result;
    }

    int_3& operator--() __GPU
    {
        int_3 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        *this = _Value;
        return *this;
    }

    int_3 operator--(int) __GPU
    {
        int_3 _Result = *this;
        --(*this);
        return _Result;
    }

    int_3& operator+=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator-=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator*=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator/=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator%=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x %= _Value2.x;
        _Value1.y %= _Value2.y;
        _Value1.z %= _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator^=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x ^= _Value2.x;
        _Value1.y ^= _Value2.y;
        _Value1.z ^= _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator|=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x |= _Value2.x;
        _Value1.y |= _Value2.y;
        _Value1.z |= _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator&=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x &= _Value2.x;
        _Value1.y &= _Value2.y;
        _Value1.z &= _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator>>=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x >>= _Value2.x;
        _Value1.y >>= _Value2.y;
        _Value1.z >>= _Value2.z;
        *this = _Value1;
        return *this;
    }

    int_3& operator<<=(const int_3& _Other) __GPU
    {
        int_3 _Value1 = *this;
        int_3 _Value2 = _Other;
        _Value1.x <<= _Value2.x;
        _Value1.y <<= _Value2.y;
        _Value1.z <<= _Value2.z;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) int_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) int_2 rg;

    int_2 get_xy() const __GPU {
        return int_2(_M_x, _M_y);
    }

    void set_xy(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) int_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) int_2 rb;

    int_2 get_xz() const __GPU {
        return int_2(_M_x, _M_z);
    }

    void set_xz(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) int_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) int_2 gr;

    int_2 get_yx() const __GPU {
        return int_2(_M_y, _M_x);
    }

    void set_yx(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) int_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) int_2 gb;

    int_2 get_yz() const __GPU {
        return int_2(_M_y, _M_z);
    }

    void set_yz(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) int_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) int_2 br;

    int_2 get_zx() const __GPU {
        return int_2(_M_z, _M_x);
    }

    void set_zx(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) int_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) int_2 bg;

    int_2 get_zy() const __GPU {
        return int_2(_M_z, _M_y);
    }

    void set_zy(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) int_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) int_3 rgb;

    int_3 get_xyz() const __GPU {
        return int_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) int_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) int_3 rbg;

    int_3 get_xzy() const __GPU {
        return int_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) int_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) int_3 grb;

    int_3 get_yxz() const __GPU {
        return int_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) int_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) int_3 gbr;

    int_3 get_yzx() const __GPU {
        return int_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) int_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) int_3 brg;

    int_3 get_zxy() const __GPU {
        return int_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) int_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) int_3 bgr;

    int_3 get_zyx() const __GPU {
        return int_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

};

/// <summary>
///    Represent a short vector of 4 int's.
/// </summary>
class int_4
{
public:
    typedef int value_type;
    static const int size = 4;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Int_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;
    value_type _M_w;

public:
    __declspec( property( get = get_x, put = set_x) ) int x;
    __declspec( property( get = get_x, put = set_x) ) int r;

    int get_x() const __GPU {
        return _M_x;
    }

    void set_x(int _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) int y;
    __declspec( property( get = get_y, put = set_y) ) int g;

    int get_y() const __GPU {
        return _M_y;
    }

    void set_y(int _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) int z;
    __declspec( property( get = get_z, put = set_z) ) int b;

    int get_z() const __GPU {
        return _M_z;
    }

    void set_z(int _Value) __GPU
    {
        _M_z = _Value;
    }

    __declspec( property( get = get_w, put = set_w) ) int w;
    __declspec( property( get = get_w, put = set_w) ) int a;

    int get_w() const __GPU {
        return _M_w;
    }

    void set_w(int _Value) __GPU
    {
        _M_w = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    int_4() __GPU
    {
        _M_x = 0;
        _M_y = 0;
        _M_z = 0;
        _M_w = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    int_4(int _V0, int _V1, int _V2, int _V3) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
        _M_w = _V3;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    int_4(int _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
        _M_w = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_4(const uint_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_4(const float_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_4(const unorm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_4(const norm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline int_4(const double_4& _Other) __GPU;

    int_4 operator~() const __GPU
    {
        int_4 _Value = *this;
        return int_4(~_Value.x, ~_Value.y, ~_Value.z, ~_Value.w);
    }

    int_4 operator-() const __GPU
    {
        int_4 _Value = *this;
        return int_4(-_Value.x, -_Value.y, -_Value.z, -_Value.w);
    }

    int_4& operator++() __GPU
    {
        int_4 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        ++_Value._M_w;
        *this = _Value;
        return *this;
    }

    int_4 operator++(int) __GPU
    {
        int_4 _Result = *this;
        ++(*this);
        return _Result;
    }

    int_4& operator--() __GPU
    {
        int_4 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        --_Value._M_w;
        *this = _Value;
        return *this;
    }

    int_4 operator--(int) __GPU
    {
        int_4 _Result = *this;
        --(*this);
        return _Result;
    }

    int_4& operator+=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        _Value1.w += _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator-=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        _Value1.w -= _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator*=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        _Value1.w *= _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator/=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        _Value1.w /= _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator%=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x %= _Value2.x;
        _Value1.y %= _Value2.y;
        _Value1.z %= _Value2.z;
        _Value1.w %= _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator^=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x ^= _Value2.x;
        _Value1.y ^= _Value2.y;
        _Value1.z ^= _Value2.z;
        _Value1.w ^= _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator|=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x |= _Value2.x;
        _Value1.y |= _Value2.y;
        _Value1.z |= _Value2.z;
        _Value1.w |= _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator&=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x &= _Value2.x;
        _Value1.y &= _Value2.y;
        _Value1.z &= _Value2.z;
        _Value1.w &= _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator>>=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x >>= _Value2.x;
        _Value1.y >>= _Value2.y;
        _Value1.z >>= _Value2.z;
        _Value1.w >>= _Value2.w;
        *this = _Value1;
        return *this;
    }

    int_4& operator<<=(const int_4& _Other) __GPU
    {
        int_4 _Value1 = *this;
        int_4 _Value2 = _Other;
        _Value1.x <<= _Value2.x;
        _Value1.y <<= _Value2.y;
        _Value1.z <<= _Value2.z;
        _Value1.w <<= _Value2.w;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) int_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) int_2 rg;

    int_2 get_xy() const __GPU {
        return int_2(_M_x, _M_y);
    }

    void set_xy(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) int_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) int_2 rb;

    int_2 get_xz() const __GPU {
        return int_2(_M_x, _M_z);
    }

    void set_xz(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xw, put = set_xw) ) int_2 xw;
    __declspec( property( get = get_xw, put = set_xw) ) int_2 ra;

    int_2 get_xw() const __GPU {
        return int_2(_M_x, _M_w);
    }

    void set_xw(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) int_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) int_2 gr;

    int_2 get_yx() const __GPU {
        return int_2(_M_y, _M_x);
    }

    void set_yx(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) int_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) int_2 gb;

    int_2 get_yz() const __GPU {
        return int_2(_M_y, _M_z);
    }

    void set_yz(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yw, put = set_yw) ) int_2 yw;
    __declspec( property( get = get_yw, put = set_yw) ) int_2 ga;

    int_2 get_yw() const __GPU {
        return int_2(_M_y, _M_w);
    }

    void set_yw(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) int_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) int_2 br;

    int_2 get_zx() const __GPU {
        return int_2(_M_z, _M_x);
    }

    void set_zx(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) int_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) int_2 bg;

    int_2 get_zy() const __GPU {
        return int_2(_M_z, _M_y);
    }

    void set_zy(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_zw, put = set_zw) ) int_2 zw;
    __declspec( property( get = get_zw, put = set_zw) ) int_2 ba;

    int_2 get_zw() const __GPU {
        return int_2(_M_z, _M_w);
    }

    void set_zw(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_wx, put = set_wx) ) int_2 wx;
    __declspec( property( get = get_wx, put = set_wx) ) int_2 ar;

    int_2 get_wx() const __GPU {
        return int_2(_M_w, _M_x);
    }

    void set_wx(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_wy, put = set_wy) ) int_2 wy;
    __declspec( property( get = get_wy, put = set_wy) ) int_2 ag;

    int_2 get_wy() const __GPU {
        return int_2(_M_w, _M_y);
    }

    void set_wy(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_wz, put = set_wz) ) int_2 wz;
    __declspec( property( get = get_wz, put = set_wz) ) int_2 ab;

    int_2 get_wz() const __GPU {
        return int_2(_M_w, _M_z);
    }

    void set_wz(const int_2& _Value) __GPU
    {
        int_2 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) int_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) int_3 rgb;

    int_3 get_xyz() const __GPU {
        return int_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xyw, put = set_xyw) ) int_3 xyw;
    __declspec( property( get = get_xyw, put = set_xyw) ) int_3 rga;

    int_3 get_xyw() const __GPU {
        return int_3(_M_x, _M_y, _M_w);
    }

    void set_xyw(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) int_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) int_3 rbg;

    int_3 get_xzy() const __GPU {
        return int_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xzw, put = set_xzw) ) int_3 xzw;
    __declspec( property( get = get_xzw, put = set_xzw) ) int_3 rba;

    int_3 get_xzw() const __GPU {
        return int_3(_M_x, _M_z, _M_w);
    }

    void set_xzw(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xwy, put = set_xwy) ) int_3 xwy;
    __declspec( property( get = get_xwy, put = set_xwy) ) int_3 rag;

    int_3 get_xwy() const __GPU {
        return int_3(_M_x, _M_w, _M_y);
    }

    void set_xwy(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xwz, put = set_xwz) ) int_3 xwz;
    __declspec( property( get = get_xwz, put = set_xwz) ) int_3 rab;

    int_3 get_xwz() const __GPU {
        return int_3(_M_x, _M_w, _M_z);
    }

    void set_xwz(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) int_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) int_3 grb;

    int_3 get_yxz() const __GPU {
        return int_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxw, put = set_yxw) ) int_3 yxw;
    __declspec( property( get = get_yxw, put = set_yxw) ) int_3 gra;

    int_3 get_yxw() const __GPU {
        return int_3(_M_y, _M_x, _M_w);
    }

    void set_yxw(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) int_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) int_3 gbr;

    int_3 get_yzx() const __GPU {
        return int_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_yzw, put = set_yzw) ) int_3 yzw;
    __declspec( property( get = get_yzw, put = set_yzw) ) int_3 gba;

    int_3 get_yzw() const __GPU {
        return int_3(_M_y, _M_z, _M_w);
    }

    void set_yzw(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_ywx, put = set_ywx) ) int_3 ywx;
    __declspec( property( get = get_ywx, put = set_ywx) ) int_3 gar;

    int_3 get_ywx() const __GPU {
        return int_3(_M_y, _M_w, _M_x);
    }

    void set_ywx(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_ywz, put = set_ywz) ) int_3 ywz;
    __declspec( property( get = get_ywz, put = set_ywz) ) int_3 gab;

    int_3 get_ywz() const __GPU {
        return int_3(_M_y, _M_w, _M_z);
    }

    void set_ywz(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) int_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) int_3 brg;

    int_3 get_zxy() const __GPU {
        return int_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zxw, put = set_zxw) ) int_3 zxw;
    __declspec( property( get = get_zxw, put = set_zxw) ) int_3 bra;

    int_3 get_zxw() const __GPU {
        return int_3(_M_z, _M_x, _M_w);
    }

    void set_zxw(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) int_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) int_3 bgr;

    int_3 get_zyx() const __GPU {
        return int_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zyw, put = set_zyw) ) int_3 zyw;
    __declspec( property( get = get_zyw, put = set_zyw) ) int_3 bga;

    int_3 get_zyw() const __GPU {
        return int_3(_M_z, _M_y, _M_w);
    }

    void set_zyw(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zwx, put = set_zwx) ) int_3 zwx;
    __declspec( property( get = get_zwx, put = set_zwx) ) int_3 bar;

    int_3 get_zwx() const __GPU {
        return int_3(_M_z, _M_w, _M_x);
    }

    void set_zwx(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zwy, put = set_zwy) ) int_3 zwy;
    __declspec( property( get = get_zwy, put = set_zwy) ) int_3 bag;

    int_3 get_zwy() const __GPU {
        return int_3(_M_z, _M_w, _M_y);
    }

    void set_zwy(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxy, put = set_wxy) ) int_3 wxy;
    __declspec( property( get = get_wxy, put = set_wxy) ) int_3 arg;

    int_3 get_wxy() const __GPU {
        return int_3(_M_w, _M_x, _M_y);
    }

    void set_wxy(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxz, put = set_wxz) ) int_3 wxz;
    __declspec( property( get = get_wxz, put = set_wxz) ) int_3 arb;

    int_3 get_wxz() const __GPU {
        return int_3(_M_w, _M_x, _M_z);
    }

    void set_wxz(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wyx, put = set_wyx) ) int_3 wyx;
    __declspec( property( get = get_wyx, put = set_wyx) ) int_3 agr;

    int_3 get_wyx() const __GPU {
        return int_3(_M_w, _M_y, _M_x);
    }

    void set_wyx(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wyz, put = set_wyz) ) int_3 wyz;
    __declspec( property( get = get_wyz, put = set_wyz) ) int_3 agb;

    int_3 get_wyz() const __GPU {
        return int_3(_M_w, _M_y, _M_z);
    }

    void set_wyz(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wzx, put = set_wzx) ) int_3 wzx;
    __declspec( property( get = get_wzx, put = set_wzx) ) int_3 abr;

    int_3 get_wzx() const __GPU {
        return int_3(_M_w, _M_z, _M_x);
    }

    void set_wzx(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wzy, put = set_wzy) ) int_3 wzy;
    __declspec( property( get = get_wzy, put = set_wzy) ) int_3 abg;

    int_3 get_wzy() const __GPU {
        return int_3(_M_w, _M_z, _M_y);
    }

    void set_wzy(const int_3& _Value) __GPU
    {
        int_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xyzw, put = set_xyzw) ) int_4 xyzw;
    __declspec( property( get = get_xyzw, put = set_xyzw) ) int_4 rgba;

    int_4 get_xyzw() const __GPU {
        return int_4(_M_x, _M_y, _M_z, _M_w);
    }

    void set_xyzw(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xywz, put = set_xywz) ) int_4 xywz;
    __declspec( property( get = get_xywz, put = set_xywz) ) int_4 rgab;

    int_4 get_xywz() const __GPU {
        return int_4(_M_x, _M_y, _M_w, _M_z);
    }

    void set_xywz(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xzyw, put = set_xzyw) ) int_4 xzyw;
    __declspec( property( get = get_xzyw, put = set_xzyw) ) int_4 rbga;

    int_4 get_xzyw() const __GPU {
        return int_4(_M_x, _M_z, _M_y, _M_w);
    }

    void set_xzyw(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xzwy, put = set_xzwy) ) int_4 xzwy;
    __declspec( property( get = get_xzwy, put = set_xzwy) ) int_4 rbag;

    int_4 get_xzwy() const __GPU {
        return int_4(_M_x, _M_z, _M_w, _M_y);
    }

    void set_xzwy(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_xwyz, put = set_xwyz) ) int_4 xwyz;
    __declspec( property( get = get_xwyz, put = set_xwyz) ) int_4 ragb;

    int_4 get_xwyz() const __GPU {
        return int_4(_M_x, _M_w, _M_y, _M_z);
    }

    void set_xwyz(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xwzy, put = set_xwzy) ) int_4 xwzy;
    __declspec( property( get = get_xwzy, put = set_xwzy) ) int_4 rabg;

    int_4 get_xwzy() const __GPU {
        return int_4(_M_x, _M_w, _M_z, _M_y);
    }

    void set_xwzy(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_yxzw, put = set_yxzw) ) int_4 yxzw;
    __declspec( property( get = get_yxzw, put = set_yxzw) ) int_4 grba;

    int_4 get_yxzw() const __GPU {
        return int_4(_M_y, _M_x, _M_z, _M_w);
    }

    void set_yxzw(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yxwz, put = set_yxwz) ) int_4 yxwz;
    __declspec( property( get = get_yxwz, put = set_yxwz) ) int_4 grab;

    int_4 get_yxwz() const __GPU {
        return int_4(_M_y, _M_x, _M_w, _M_z);
    }

    void set_yxwz(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_yzxw, put = set_yzxw) ) int_4 yzxw;
    __declspec( property( get = get_yzxw, put = set_yzxw) ) int_4 gbra;

    int_4 get_yzxw() const __GPU {
        return int_4(_M_y, _M_z, _M_x, _M_w);
    }

    void set_yzxw(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yzwx, put = set_yzwx) ) int_4 yzwx;
    __declspec( property( get = get_yzwx, put = set_yzwx) ) int_4 gbar;

    int_4 get_yzwx() const __GPU {
        return int_4(_M_y, _M_z, _M_w, _M_x);
    }

    void set_yzwx(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_ywxz, put = set_ywxz) ) int_4 ywxz;
    __declspec( property( get = get_ywxz, put = set_ywxz) ) int_4 garb;

    int_4 get_ywxz() const __GPU {
        return int_4(_M_y, _M_w, _M_x, _M_z);
    }

    void set_ywxz(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_ywzx, put = set_ywzx) ) int_4 ywzx;
    __declspec( property( get = get_ywzx, put = set_ywzx) ) int_4 gabr;

    int_4 get_ywzx() const __GPU {
        return int_4(_M_y, _M_w, _M_z, _M_x);
    }

    void set_ywzx(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zxyw, put = set_zxyw) ) int_4 zxyw;
    __declspec( property( get = get_zxyw, put = set_zxyw) ) int_4 brga;

    int_4 get_zxyw() const __GPU {
        return int_4(_M_z, _M_x, _M_y, _M_w);
    }

    void set_zxyw(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zxwy, put = set_zxwy) ) int_4 zxwy;
    __declspec( property( get = get_zxwy, put = set_zxwy) ) int_4 brag;

    int_4 get_zxwy() const __GPU {
        return int_4(_M_z, _M_x, _M_w, _M_y);
    }

    void set_zxwy(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zyxw, put = set_zyxw) ) int_4 zyxw;
    __declspec( property( get = get_zyxw, put = set_zyxw) ) int_4 bgra;

    int_4 get_zyxw() const __GPU {
        return int_4(_M_z, _M_y, _M_x, _M_w);
    }

    void set_zyxw(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zywx, put = set_zywx) ) int_4 zywx;
    __declspec( property( get = get_zywx, put = set_zywx) ) int_4 bgar;

    int_4 get_zywx() const __GPU {
        return int_4(_M_z, _M_y, _M_w, _M_x);
    }

    void set_zywx(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zwxy, put = set_zwxy) ) int_4 zwxy;
    __declspec( property( get = get_zwxy, put = set_zwxy) ) int_4 barg;

    int_4 get_zwxy() const __GPU {
        return int_4(_M_z, _M_w, _M_x, _M_y);
    }

    void set_zwxy(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zwyx, put = set_zwyx) ) int_4 zwyx;
    __declspec( property( get = get_zwyx, put = set_zwyx) ) int_4 bagr;

    int_4 get_zwyx() const __GPU {
        return int_4(_M_z, _M_w, _M_y, _M_x);
    }

    void set_zwyx(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wxyz, put = set_wxyz) ) int_4 wxyz;
    __declspec( property( get = get_wxyz, put = set_wxyz) ) int_4 argb;

    int_4 get_wxyz() const __GPU {
        return int_4(_M_w, _M_x, _M_y, _M_z);
    }

    void set_wxyz(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wxzy, put = set_wxzy) ) int_4 wxzy;
    __declspec( property( get = get_wxzy, put = set_wxzy) ) int_4 arbg;

    int_4 get_wxzy() const __GPU {
        return int_4(_M_w, _M_x, _M_z, _M_y);
    }

    void set_wxzy(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wyxz, put = set_wyxz) ) int_4 wyxz;
    __declspec( property( get = get_wyxz, put = set_wyxz) ) int_4 agrb;

    int_4 get_wyxz() const __GPU {
        return int_4(_M_w, _M_y, _M_x, _M_z);
    }

    void set_wyxz(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wyzx, put = set_wyzx) ) int_4 wyzx;
    __declspec( property( get = get_wyzx, put = set_wyzx) ) int_4 agbr;

    int_4 get_wyzx() const __GPU {
        return int_4(_M_w, _M_y, _M_z, _M_x);
    }

    void set_wyzx(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wzxy, put = set_wzxy) ) int_4 wzxy;
    __declspec( property( get = get_wzxy, put = set_wzxy) ) int_4 abrg;

    int_4 get_wzxy() const __GPU {
        return int_4(_M_w, _M_z, _M_x, _M_y);
    }

    void set_wzxy(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wzyx, put = set_wzyx) ) int_4 wzyx;
    __declspec( property( get = get_wzyx, put = set_wzyx) ) int_4 abgr;

    int_4 get_wzyx() const __GPU {
        return int_4(_M_w, _M_z, _M_y, _M_x);
    }

    void set_wzyx(const int_4& _Value) __GPU
    {
        int_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

};

/// <summary>
///    Represent a short vector of 2 float's.
/// </summary>
class float_2
{
public:
    typedef float value_type;
    static const int size = 2;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Float_type;
private:
    value_type _M_x;
    value_type _M_y;

public:
    __declspec( property( get = get_x, put = set_x) ) float x;
    __declspec( property( get = get_x, put = set_x) ) float r;

    float get_x() const __GPU {
        return _M_x;
    }

    void set_x(float _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) float y;
    __declspec( property( get = get_y, put = set_y) ) float g;

    float get_y() const __GPU {
        return _M_y;
    }

    void set_y(float _Value) __GPU
    {
        _M_y = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    float_2() __GPU
    {
        _M_x = 0;
        _M_y = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    float_2(float _V0, float _V1) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    float_2(float _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_2(const uint_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_2(const int_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_2(const unorm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_2(const norm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_2(const double_2& _Other) __GPU;

    float_2 operator-() const __GPU
    {
        float_2 _Value = *this;
        return float_2(-_Value.x, -_Value.y);
    }

    float_2& operator++() __GPU
    {
        float_2 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        *this = _Value;
        return *this;
    }

    float_2 operator++(int) __GPU
    {
        float_2 _Result = *this;
        ++(*this);
        return _Result;
    }

    float_2& operator--() __GPU
    {
        float_2 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        *this = _Value;
        return *this;
    }

    float_2 operator--(int) __GPU
    {
        float_2 _Result = *this;
        --(*this);
        return _Result;
    }

    float_2& operator+=(const float_2& _Other) __GPU
    {
        float_2 _Value1 = *this;
        float_2 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        *this = _Value1;
        return *this;
    }

    float_2& operator-=(const float_2& _Other) __GPU
    {
        float_2 _Value1 = *this;
        float_2 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        *this = _Value1;
        return *this;
    }

    float_2& operator*=(const float_2& _Other) __GPU
    {
        float_2 _Value1 = *this;
        float_2 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        *this = _Value1;
        return *this;
    }

    float_2& operator/=(const float_2& _Other) __GPU
    {
        float_2 _Value1 = *this;
        float_2 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) float_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) float_2 rg;

    float_2 get_xy() const __GPU {
        return float_2(_M_x, _M_y);
    }

    void set_xy(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) float_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) float_2 gr;

    float_2 get_yx() const __GPU {
        return float_2(_M_y, _M_x);
    }

    void set_yx(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

};

/// <summary>
///    Represent a short vector of 3 float's.
/// </summary>
class float_3
{
public:
    typedef float value_type;
    static const int size = 3;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Float_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;

public:
    __declspec( property( get = get_x, put = set_x) ) float x;
    __declspec( property( get = get_x, put = set_x) ) float r;

    float get_x() const __GPU {
        return _M_x;
    }

    void set_x(float _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) float y;
    __declspec( property( get = get_y, put = set_y) ) float g;

    float get_y() const __GPU {
        return _M_y;
    }

    void set_y(float _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) float z;
    __declspec( property( get = get_z, put = set_z) ) float b;

    float get_z() const __GPU {
        return _M_z;
    }

    void set_z(float _Value) __GPU
    {
        _M_z = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    float_3() __GPU
    {
        _M_x = 0;
        _M_y = 0;
        _M_z = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    float_3(float _V0, float _V1, float _V2) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    float_3(float _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_3(const uint_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_3(const int_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_3(const unorm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_3(const norm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_3(const double_3& _Other) __GPU;

    float_3 operator-() const __GPU
    {
        float_3 _Value = *this;
        return float_3(-_Value.x, -_Value.y, -_Value.z);
    }

    float_3& operator++() __GPU
    {
        float_3 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        *this = _Value;
        return *this;
    }

    float_3 operator++(int) __GPU
    {
        float_3 _Result = *this;
        ++(*this);
        return _Result;
    }

    float_3& operator--() __GPU
    {
        float_3 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        *this = _Value;
        return *this;
    }

    float_3 operator--(int) __GPU
    {
        float_3 _Result = *this;
        --(*this);
        return _Result;
    }

    float_3& operator+=(const float_3& _Other) __GPU
    {
        float_3 _Value1 = *this;
        float_3 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        *this = _Value1;
        return *this;
    }

    float_3& operator-=(const float_3& _Other) __GPU
    {
        float_3 _Value1 = *this;
        float_3 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        *this = _Value1;
        return *this;
    }

    float_3& operator*=(const float_3& _Other) __GPU
    {
        float_3 _Value1 = *this;
        float_3 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        *this = _Value1;
        return *this;
    }

    float_3& operator/=(const float_3& _Other) __GPU
    {
        float_3 _Value1 = *this;
        float_3 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) float_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) float_2 rg;

    float_2 get_xy() const __GPU {
        return float_2(_M_x, _M_y);
    }

    void set_xy(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) float_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) float_2 rb;

    float_2 get_xz() const __GPU {
        return float_2(_M_x, _M_z);
    }

    void set_xz(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) float_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) float_2 gr;

    float_2 get_yx() const __GPU {
        return float_2(_M_y, _M_x);
    }

    void set_yx(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) float_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) float_2 gb;

    float_2 get_yz() const __GPU {
        return float_2(_M_y, _M_z);
    }

    void set_yz(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) float_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) float_2 br;

    float_2 get_zx() const __GPU {
        return float_2(_M_z, _M_x);
    }

    void set_zx(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) float_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) float_2 bg;

    float_2 get_zy() const __GPU {
        return float_2(_M_z, _M_y);
    }

    void set_zy(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) float_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) float_3 rgb;

    float_3 get_xyz() const __GPU {
        return float_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) float_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) float_3 rbg;

    float_3 get_xzy() const __GPU {
        return float_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) float_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) float_3 grb;

    float_3 get_yxz() const __GPU {
        return float_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) float_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) float_3 gbr;

    float_3 get_yzx() const __GPU {
        return float_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) float_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) float_3 brg;

    float_3 get_zxy() const __GPU {
        return float_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) float_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) float_3 bgr;

    float_3 get_zyx() const __GPU {
        return float_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

};

/// <summary>
///    Represent a short vector of 4 float's.
/// </summary>
class float_4
{
public:
    typedef float value_type;
    static const int size = 4;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Float_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;
    value_type _M_w;

public:
    __declspec( property( get = get_x, put = set_x) ) float x;
    __declspec( property( get = get_x, put = set_x) ) float r;

    float get_x() const __GPU {
        return _M_x;
    }

    void set_x(float _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) float y;
    __declspec( property( get = get_y, put = set_y) ) float g;

    float get_y() const __GPU {
        return _M_y;
    }

    void set_y(float _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) float z;
    __declspec( property( get = get_z, put = set_z) ) float b;

    float get_z() const __GPU {
        return _M_z;
    }

    void set_z(float _Value) __GPU
    {
        _M_z = _Value;
    }

    __declspec( property( get = get_w, put = set_w) ) float w;
    __declspec( property( get = get_w, put = set_w) ) float a;

    float get_w() const __GPU {
        return _M_w;
    }

    void set_w(float _Value) __GPU
    {
        _M_w = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    float_4() __GPU
    {
        _M_x = 0;
        _M_y = 0;
        _M_z = 0;
        _M_w = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    float_4(float _V0, float _V1, float _V2, float _V3) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
        _M_w = _V3;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    float_4(float _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
        _M_w = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_4(const uint_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_4(const int_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_4(const unorm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_4(const norm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline float_4(const double_4& _Other) __GPU;

    float_4 operator-() const __GPU
    {
        float_4 _Value = *this;
        return float_4(-_Value.x, -_Value.y, -_Value.z, -_Value.w);
    }

    float_4& operator++() __GPU
    {
        float_4 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        ++_Value._M_w;
        *this = _Value;
        return *this;
    }

    float_4 operator++(int) __GPU
    {
        float_4 _Result = *this;
        ++(*this);
        return _Result;
    }

    float_4& operator--() __GPU
    {
        float_4 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        --_Value._M_w;
        *this = _Value;
        return *this;
    }

    float_4 operator--(int) __GPU
    {
        float_4 _Result = *this;
        --(*this);
        return _Result;
    }

    float_4& operator+=(const float_4& _Other) __GPU
    {
        float_4 _Value1 = *this;
        float_4 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        _Value1.w += _Value2.w;
        *this = _Value1;
        return *this;
    }

    float_4& operator-=(const float_4& _Other) __GPU
    {
        float_4 _Value1 = *this;
        float_4 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        _Value1.w -= _Value2.w;
        *this = _Value1;
        return *this;
    }

    float_4& operator*=(const float_4& _Other) __GPU
    {
        float_4 _Value1 = *this;
        float_4 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        _Value1.w *= _Value2.w;
        *this = _Value1;
        return *this;
    }

    float_4& operator/=(const float_4& _Other) __GPU
    {
        float_4 _Value1 = *this;
        float_4 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        _Value1.w /= _Value2.w;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) float_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) float_2 rg;

    float_2 get_xy() const __GPU {
        return float_2(_M_x, _M_y);
    }

    void set_xy(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) float_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) float_2 rb;

    float_2 get_xz() const __GPU {
        return float_2(_M_x, _M_z);
    }

    void set_xz(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xw, put = set_xw) ) float_2 xw;
    __declspec( property( get = get_xw, put = set_xw) ) float_2 ra;

    float_2 get_xw() const __GPU {
        return float_2(_M_x, _M_w);
    }

    void set_xw(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) float_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) float_2 gr;

    float_2 get_yx() const __GPU {
        return float_2(_M_y, _M_x);
    }

    void set_yx(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) float_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) float_2 gb;

    float_2 get_yz() const __GPU {
        return float_2(_M_y, _M_z);
    }

    void set_yz(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yw, put = set_yw) ) float_2 yw;
    __declspec( property( get = get_yw, put = set_yw) ) float_2 ga;

    float_2 get_yw() const __GPU {
        return float_2(_M_y, _M_w);
    }

    void set_yw(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) float_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) float_2 br;

    float_2 get_zx() const __GPU {
        return float_2(_M_z, _M_x);
    }

    void set_zx(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) float_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) float_2 bg;

    float_2 get_zy() const __GPU {
        return float_2(_M_z, _M_y);
    }

    void set_zy(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_zw, put = set_zw) ) float_2 zw;
    __declspec( property( get = get_zw, put = set_zw) ) float_2 ba;

    float_2 get_zw() const __GPU {
        return float_2(_M_z, _M_w);
    }

    void set_zw(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_wx, put = set_wx) ) float_2 wx;
    __declspec( property( get = get_wx, put = set_wx) ) float_2 ar;

    float_2 get_wx() const __GPU {
        return float_2(_M_w, _M_x);
    }

    void set_wx(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_wy, put = set_wy) ) float_2 wy;
    __declspec( property( get = get_wy, put = set_wy) ) float_2 ag;

    float_2 get_wy() const __GPU {
        return float_2(_M_w, _M_y);
    }

    void set_wy(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_wz, put = set_wz) ) float_2 wz;
    __declspec( property( get = get_wz, put = set_wz) ) float_2 ab;

    float_2 get_wz() const __GPU {
        return float_2(_M_w, _M_z);
    }

    void set_wz(const float_2& _Value) __GPU
    {
        float_2 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) float_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) float_3 rgb;

    float_3 get_xyz() const __GPU {
        return float_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xyw, put = set_xyw) ) float_3 xyw;
    __declspec( property( get = get_xyw, put = set_xyw) ) float_3 rga;

    float_3 get_xyw() const __GPU {
        return float_3(_M_x, _M_y, _M_w);
    }

    void set_xyw(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) float_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) float_3 rbg;

    float_3 get_xzy() const __GPU {
        return float_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xzw, put = set_xzw) ) float_3 xzw;
    __declspec( property( get = get_xzw, put = set_xzw) ) float_3 rba;

    float_3 get_xzw() const __GPU {
        return float_3(_M_x, _M_z, _M_w);
    }

    void set_xzw(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xwy, put = set_xwy) ) float_3 xwy;
    __declspec( property( get = get_xwy, put = set_xwy) ) float_3 rag;

    float_3 get_xwy() const __GPU {
        return float_3(_M_x, _M_w, _M_y);
    }

    void set_xwy(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xwz, put = set_xwz) ) float_3 xwz;
    __declspec( property( get = get_xwz, put = set_xwz) ) float_3 rab;

    float_3 get_xwz() const __GPU {
        return float_3(_M_x, _M_w, _M_z);
    }

    void set_xwz(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) float_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) float_3 grb;

    float_3 get_yxz() const __GPU {
        return float_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxw, put = set_yxw) ) float_3 yxw;
    __declspec( property( get = get_yxw, put = set_yxw) ) float_3 gra;

    float_3 get_yxw() const __GPU {
        return float_3(_M_y, _M_x, _M_w);
    }

    void set_yxw(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) float_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) float_3 gbr;

    float_3 get_yzx() const __GPU {
        return float_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_yzw, put = set_yzw) ) float_3 yzw;
    __declspec( property( get = get_yzw, put = set_yzw) ) float_3 gba;

    float_3 get_yzw() const __GPU {
        return float_3(_M_y, _M_z, _M_w);
    }

    void set_yzw(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_ywx, put = set_ywx) ) float_3 ywx;
    __declspec( property( get = get_ywx, put = set_ywx) ) float_3 gar;

    float_3 get_ywx() const __GPU {
        return float_3(_M_y, _M_w, _M_x);
    }

    void set_ywx(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_ywz, put = set_ywz) ) float_3 ywz;
    __declspec( property( get = get_ywz, put = set_ywz) ) float_3 gab;

    float_3 get_ywz() const __GPU {
        return float_3(_M_y, _M_w, _M_z);
    }

    void set_ywz(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) float_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) float_3 brg;

    float_3 get_zxy() const __GPU {
        return float_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zxw, put = set_zxw) ) float_3 zxw;
    __declspec( property( get = get_zxw, put = set_zxw) ) float_3 bra;

    float_3 get_zxw() const __GPU {
        return float_3(_M_z, _M_x, _M_w);
    }

    void set_zxw(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) float_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) float_3 bgr;

    float_3 get_zyx() const __GPU {
        return float_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zyw, put = set_zyw) ) float_3 zyw;
    __declspec( property( get = get_zyw, put = set_zyw) ) float_3 bga;

    float_3 get_zyw() const __GPU {
        return float_3(_M_z, _M_y, _M_w);
    }

    void set_zyw(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zwx, put = set_zwx) ) float_3 zwx;
    __declspec( property( get = get_zwx, put = set_zwx) ) float_3 bar;

    float_3 get_zwx() const __GPU {
        return float_3(_M_z, _M_w, _M_x);
    }

    void set_zwx(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zwy, put = set_zwy) ) float_3 zwy;
    __declspec( property( get = get_zwy, put = set_zwy) ) float_3 bag;

    float_3 get_zwy() const __GPU {
        return float_3(_M_z, _M_w, _M_y);
    }

    void set_zwy(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxy, put = set_wxy) ) float_3 wxy;
    __declspec( property( get = get_wxy, put = set_wxy) ) float_3 arg;

    float_3 get_wxy() const __GPU {
        return float_3(_M_w, _M_x, _M_y);
    }

    void set_wxy(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxz, put = set_wxz) ) float_3 wxz;
    __declspec( property( get = get_wxz, put = set_wxz) ) float_3 arb;

    float_3 get_wxz() const __GPU {
        return float_3(_M_w, _M_x, _M_z);
    }

    void set_wxz(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wyx, put = set_wyx) ) float_3 wyx;
    __declspec( property( get = get_wyx, put = set_wyx) ) float_3 agr;

    float_3 get_wyx() const __GPU {
        return float_3(_M_w, _M_y, _M_x);
    }

    void set_wyx(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wyz, put = set_wyz) ) float_3 wyz;
    __declspec( property( get = get_wyz, put = set_wyz) ) float_3 agb;

    float_3 get_wyz() const __GPU {
        return float_3(_M_w, _M_y, _M_z);
    }

    void set_wyz(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wzx, put = set_wzx) ) float_3 wzx;
    __declspec( property( get = get_wzx, put = set_wzx) ) float_3 abr;

    float_3 get_wzx() const __GPU {
        return float_3(_M_w, _M_z, _M_x);
    }

    void set_wzx(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wzy, put = set_wzy) ) float_3 wzy;
    __declspec( property( get = get_wzy, put = set_wzy) ) float_3 abg;

    float_3 get_wzy() const __GPU {
        return float_3(_M_w, _M_z, _M_y);
    }

    void set_wzy(const float_3& _Value) __GPU
    {
        float_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xyzw, put = set_xyzw) ) float_4 xyzw;
    __declspec( property( get = get_xyzw, put = set_xyzw) ) float_4 rgba;

    float_4 get_xyzw() const __GPU {
        return float_4(_M_x, _M_y, _M_z, _M_w);
    }

    void set_xyzw(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xywz, put = set_xywz) ) float_4 xywz;
    __declspec( property( get = get_xywz, put = set_xywz) ) float_4 rgab;

    float_4 get_xywz() const __GPU {
        return float_4(_M_x, _M_y, _M_w, _M_z);
    }

    void set_xywz(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xzyw, put = set_xzyw) ) float_4 xzyw;
    __declspec( property( get = get_xzyw, put = set_xzyw) ) float_4 rbga;

    float_4 get_xzyw() const __GPU {
        return float_4(_M_x, _M_z, _M_y, _M_w);
    }

    void set_xzyw(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xzwy, put = set_xzwy) ) float_4 xzwy;
    __declspec( property( get = get_xzwy, put = set_xzwy) ) float_4 rbag;

    float_4 get_xzwy() const __GPU {
        return float_4(_M_x, _M_z, _M_w, _M_y);
    }

    void set_xzwy(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_xwyz, put = set_xwyz) ) float_4 xwyz;
    __declspec( property( get = get_xwyz, put = set_xwyz) ) float_4 ragb;

    float_4 get_xwyz() const __GPU {
        return float_4(_M_x, _M_w, _M_y, _M_z);
    }

    void set_xwyz(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xwzy, put = set_xwzy) ) float_4 xwzy;
    __declspec( property( get = get_xwzy, put = set_xwzy) ) float_4 rabg;

    float_4 get_xwzy() const __GPU {
        return float_4(_M_x, _M_w, _M_z, _M_y);
    }

    void set_xwzy(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_yxzw, put = set_yxzw) ) float_4 yxzw;
    __declspec( property( get = get_yxzw, put = set_yxzw) ) float_4 grba;

    float_4 get_yxzw() const __GPU {
        return float_4(_M_y, _M_x, _M_z, _M_w);
    }

    void set_yxzw(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yxwz, put = set_yxwz) ) float_4 yxwz;
    __declspec( property( get = get_yxwz, put = set_yxwz) ) float_4 grab;

    float_4 get_yxwz() const __GPU {
        return float_4(_M_y, _M_x, _M_w, _M_z);
    }

    void set_yxwz(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_yzxw, put = set_yzxw) ) float_4 yzxw;
    __declspec( property( get = get_yzxw, put = set_yzxw) ) float_4 gbra;

    float_4 get_yzxw() const __GPU {
        return float_4(_M_y, _M_z, _M_x, _M_w);
    }

    void set_yzxw(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yzwx, put = set_yzwx) ) float_4 yzwx;
    __declspec( property( get = get_yzwx, put = set_yzwx) ) float_4 gbar;

    float_4 get_yzwx() const __GPU {
        return float_4(_M_y, _M_z, _M_w, _M_x);
    }

    void set_yzwx(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_ywxz, put = set_ywxz) ) float_4 ywxz;
    __declspec( property( get = get_ywxz, put = set_ywxz) ) float_4 garb;

    float_4 get_ywxz() const __GPU {
        return float_4(_M_y, _M_w, _M_x, _M_z);
    }

    void set_ywxz(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_ywzx, put = set_ywzx) ) float_4 ywzx;
    __declspec( property( get = get_ywzx, put = set_ywzx) ) float_4 gabr;

    float_4 get_ywzx() const __GPU {
        return float_4(_M_y, _M_w, _M_z, _M_x);
    }

    void set_ywzx(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zxyw, put = set_zxyw) ) float_4 zxyw;
    __declspec( property( get = get_zxyw, put = set_zxyw) ) float_4 brga;

    float_4 get_zxyw() const __GPU {
        return float_4(_M_z, _M_x, _M_y, _M_w);
    }

    void set_zxyw(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zxwy, put = set_zxwy) ) float_4 zxwy;
    __declspec( property( get = get_zxwy, put = set_zxwy) ) float_4 brag;

    float_4 get_zxwy() const __GPU {
        return float_4(_M_z, _M_x, _M_w, _M_y);
    }

    void set_zxwy(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zyxw, put = set_zyxw) ) float_4 zyxw;
    __declspec( property( get = get_zyxw, put = set_zyxw) ) float_4 bgra;

    float_4 get_zyxw() const __GPU {
        return float_4(_M_z, _M_y, _M_x, _M_w);
    }

    void set_zyxw(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zywx, put = set_zywx) ) float_4 zywx;
    __declspec( property( get = get_zywx, put = set_zywx) ) float_4 bgar;

    float_4 get_zywx() const __GPU {
        return float_4(_M_z, _M_y, _M_w, _M_x);
    }

    void set_zywx(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zwxy, put = set_zwxy) ) float_4 zwxy;
    __declspec( property( get = get_zwxy, put = set_zwxy) ) float_4 barg;

    float_4 get_zwxy() const __GPU {
        return float_4(_M_z, _M_w, _M_x, _M_y);
    }

    void set_zwxy(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zwyx, put = set_zwyx) ) float_4 zwyx;
    __declspec( property( get = get_zwyx, put = set_zwyx) ) float_4 bagr;

    float_4 get_zwyx() const __GPU {
        return float_4(_M_z, _M_w, _M_y, _M_x);
    }

    void set_zwyx(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wxyz, put = set_wxyz) ) float_4 wxyz;
    __declspec( property( get = get_wxyz, put = set_wxyz) ) float_4 argb;

    float_4 get_wxyz() const __GPU {
        return float_4(_M_w, _M_x, _M_y, _M_z);
    }

    void set_wxyz(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wxzy, put = set_wxzy) ) float_4 wxzy;
    __declspec( property( get = get_wxzy, put = set_wxzy) ) float_4 arbg;

    float_4 get_wxzy() const __GPU {
        return float_4(_M_w, _M_x, _M_z, _M_y);
    }

    void set_wxzy(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wyxz, put = set_wyxz) ) float_4 wyxz;
    __declspec( property( get = get_wyxz, put = set_wyxz) ) float_4 agrb;

    float_4 get_wyxz() const __GPU {
        return float_4(_M_w, _M_y, _M_x, _M_z);
    }

    void set_wyxz(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wyzx, put = set_wyzx) ) float_4 wyzx;
    __declspec( property( get = get_wyzx, put = set_wyzx) ) float_4 agbr;

    float_4 get_wyzx() const __GPU {
        return float_4(_M_w, _M_y, _M_z, _M_x);
    }

    void set_wyzx(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wzxy, put = set_wzxy) ) float_4 wzxy;
    __declspec( property( get = get_wzxy, put = set_wzxy) ) float_4 abrg;

    float_4 get_wzxy() const __GPU {
        return float_4(_M_w, _M_z, _M_x, _M_y);
    }

    void set_wzxy(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wzyx, put = set_wzyx) ) float_4 wzyx;
    __declspec( property( get = get_wzyx, put = set_wzyx) ) float_4 abgr;

    float_4 get_wzyx() const __GPU {
        return float_4(_M_w, _M_z, _M_y, _M_x);
    }

    void set_wzyx(const float_4& _Value) __GPU
    {
        float_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

};

/// <summary>
///    Represent a short vector of 2 unorm's.
/// </summary>
class unorm_2
{
public:
    typedef unorm value_type;
    static const int size = 2;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Unorm_type;
private:
    value_type _M_x;
    value_type _M_y;

public:
    __declspec( property( get = get_x, put = set_x) ) unorm x;
    __declspec( property( get = get_x, put = set_x) ) unorm r;

    unorm get_x() const __GPU {
        return _M_x;
    }

    void set_x(unorm _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) unorm y;
    __declspec( property( get = get_y, put = set_y) ) unorm g;

    unorm get_y() const __GPU {
        return _M_y;
    }

    void set_y(unorm _Value) __GPU
    {
        _M_y = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    unorm_2() __GPU
    {
        _M_x = unorm(0.0f);
        _M_y = unorm(0.0f);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    unorm_2(unorm _V0, unorm _V1) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    unorm_2(float _V0, float _V1) __GPU
    {
        _M_x = unorm(_V0);
        _M_y = unorm(_V1);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    unorm_2(unorm _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
    }

    explicit unorm_2(float _V) __GPU
    {
        _M_x = unorm(_V);
        _M_y = unorm(_V);
    }

    /// <summary>
    ///     Copy constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object to copy from.
    /// </param>
    unorm_2(const unorm_2& _Other) __GPU
    {
        *this = _Other;
    }

    unorm_2& operator=(const unorm_2& _Other) __GPU
    {
        _M_x = _Other._M_x;
        _M_y = _Other._M_y;
        return *this;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_2(const uint_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_2(const int_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_2(const float_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_2(const norm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_2(const double_2& _Other) __GPU;

    unorm_2& operator++() __GPU
    {
        unorm_2 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        *this = _Value;
        return *this;
    }

    unorm_2 operator++(int) __GPU
    {
        unorm_2 _Result = *this;
        ++(*this);
        return _Result;
    }

    unorm_2& operator--() __GPU
    {
        unorm_2 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        *this = _Value;
        return *this;
    }

    unorm_2 operator--(int) __GPU
    {
        unorm_2 _Result = *this;
        --(*this);
        return _Result;
    }

    unorm_2& operator+=(const unorm_2& _Other) __GPU
    {
        unorm_2 _Value1 = *this;
        unorm_2 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        *this = _Value1;
        return *this;
    }

    unorm_2& operator-=(const unorm_2& _Other) __GPU
    {
        unorm_2 _Value1 = *this;
        unorm_2 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        *this = _Value1;
        return *this;
    }

    unorm_2& operator*=(const unorm_2& _Other) __GPU
    {
        unorm_2 _Value1 = *this;
        unorm_2 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        *this = _Value1;
        return *this;
    }

    unorm_2& operator/=(const unorm_2& _Other) __GPU
    {
        unorm_2 _Value1 = *this;
        unorm_2 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) unorm_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) unorm_2 rg;

    unorm_2 get_xy() const __GPU {
        return unorm_2(_M_x, _M_y);
    }

    void set_xy(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) unorm_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) unorm_2 gr;

    unorm_2 get_yx() const __GPU {
        return unorm_2(_M_y, _M_x);
    }

    void set_yx(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

};

/// <summary>
///    Represent a short vector of 3 unorm's.
/// </summary>
class unorm_3
{
public:
    typedef unorm value_type;
    static const int size = 3;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Unorm_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;

public:
    __declspec( property( get = get_x, put = set_x) ) unorm x;
    __declspec( property( get = get_x, put = set_x) ) unorm r;

    unorm get_x() const __GPU {
        return _M_x;
    }

    void set_x(unorm _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) unorm y;
    __declspec( property( get = get_y, put = set_y) ) unorm g;

    unorm get_y() const __GPU {
        return _M_y;
    }

    void set_y(unorm _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) unorm z;
    __declspec( property( get = get_z, put = set_z) ) unorm b;

    unorm get_z() const __GPU {
        return _M_z;
    }

    void set_z(unorm _Value) __GPU
    {
        _M_z = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    unorm_3() __GPU
    {
        _M_x = unorm(0.0f);
        _M_y = unorm(0.0f);
        _M_z = unorm(0.0f);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    unorm_3(unorm _V0, unorm _V1, unorm _V2) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    unorm_3(float _V0, float _V1, float _V2) __GPU
    {
        _M_x = unorm(_V0);
        _M_y = unorm(_V1);
        _M_z = unorm(_V2);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    unorm_3(unorm _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
    }

    explicit unorm_3(float _V) __GPU
    {
        _M_x = unorm(_V);
        _M_y = unorm(_V);
        _M_z = unorm(_V);
    }

    /// <summary>
    ///     Copy constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object to copy from.
    /// </param>
    unorm_3(const unorm_3& _Other) __GPU
    {
        *this = _Other;
    }

    unorm_3& operator=(const unorm_3& _Other) __GPU
    {
        _M_x = _Other._M_x;
        _M_y = _Other._M_y;
        _M_z = _Other._M_z;
        return *this;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_3(const uint_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_3(const int_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_3(const float_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_3(const norm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_3(const double_3& _Other) __GPU;

    unorm_3& operator++() __GPU
    {
        unorm_3 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        *this = _Value;
        return *this;
    }

    unorm_3 operator++(int) __GPU
    {
        unorm_3 _Result = *this;
        ++(*this);
        return _Result;
    }

    unorm_3& operator--() __GPU
    {
        unorm_3 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        *this = _Value;
        return *this;
    }

    unorm_3 operator--(int) __GPU
    {
        unorm_3 _Result = *this;
        --(*this);
        return _Result;
    }

    unorm_3& operator+=(const unorm_3& _Other) __GPU
    {
        unorm_3 _Value1 = *this;
        unorm_3 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        *this = _Value1;
        return *this;
    }

    unorm_3& operator-=(const unorm_3& _Other) __GPU
    {
        unorm_3 _Value1 = *this;
        unorm_3 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        *this = _Value1;
        return *this;
    }

    unorm_3& operator*=(const unorm_3& _Other) __GPU
    {
        unorm_3 _Value1 = *this;
        unorm_3 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        *this = _Value1;
        return *this;
    }

    unorm_3& operator/=(const unorm_3& _Other) __GPU
    {
        unorm_3 _Value1 = *this;
        unorm_3 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) unorm_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) unorm_2 rg;

    unorm_2 get_xy() const __GPU {
        return unorm_2(_M_x, _M_y);
    }

    void set_xy(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) unorm_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) unorm_2 rb;

    unorm_2 get_xz() const __GPU {
        return unorm_2(_M_x, _M_z);
    }

    void set_xz(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) unorm_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) unorm_2 gr;

    unorm_2 get_yx() const __GPU {
        return unorm_2(_M_y, _M_x);
    }

    void set_yx(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) unorm_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) unorm_2 gb;

    unorm_2 get_yz() const __GPU {
        return unorm_2(_M_y, _M_z);
    }

    void set_yz(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) unorm_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) unorm_2 br;

    unorm_2 get_zx() const __GPU {
        return unorm_2(_M_z, _M_x);
    }

    void set_zx(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) unorm_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) unorm_2 bg;

    unorm_2 get_zy() const __GPU {
        return unorm_2(_M_z, _M_y);
    }

    void set_zy(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) unorm_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) unorm_3 rgb;

    unorm_3 get_xyz() const __GPU {
        return unorm_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) unorm_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) unorm_3 rbg;

    unorm_3 get_xzy() const __GPU {
        return unorm_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) unorm_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) unorm_3 grb;

    unorm_3 get_yxz() const __GPU {
        return unorm_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) unorm_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) unorm_3 gbr;

    unorm_3 get_yzx() const __GPU {
        return unorm_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) unorm_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) unorm_3 brg;

    unorm_3 get_zxy() const __GPU {
        return unorm_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) unorm_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) unorm_3 bgr;

    unorm_3 get_zyx() const __GPU {
        return unorm_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

};

/// <summary>
///    Represent a short vector of 4 unorm's.
/// </summary>
class unorm_4
{
public:
    typedef unorm value_type;
    static const int size = 4;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Unorm_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;
    value_type _M_w;

public:
    __declspec( property( get = get_x, put = set_x) ) unorm x;
    __declspec( property( get = get_x, put = set_x) ) unorm r;

    unorm get_x() const __GPU {
        return _M_x;
    }

    void set_x(unorm _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) unorm y;
    __declspec( property( get = get_y, put = set_y) ) unorm g;

    unorm get_y() const __GPU {
        return _M_y;
    }

    void set_y(unorm _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) unorm z;
    __declspec( property( get = get_z, put = set_z) ) unorm b;

    unorm get_z() const __GPU {
        return _M_z;
    }

    void set_z(unorm _Value) __GPU
    {
        _M_z = _Value;
    }

    __declspec( property( get = get_w, put = set_w) ) unorm w;
    __declspec( property( get = get_w, put = set_w) ) unorm a;

    unorm get_w() const __GPU {
        return _M_w;
    }

    void set_w(unorm _Value) __GPU
    {
        _M_w = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    unorm_4() __GPU
    {
        _M_x = unorm(0.0f);
        _M_y = unorm(0.0f);
        _M_z = unorm(0.0f);
        _M_w = unorm(0.0f);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    unorm_4(unorm _V0, unorm _V1, unorm _V2, unorm _V3) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
        _M_w = _V3;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    unorm_4(float _V0, float _V1, float _V2, float _V3) __GPU
    {
        _M_x = unorm(_V0);
        _M_y = unorm(_V1);
        _M_z = unorm(_V2);
        _M_w = unorm(_V3);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    unorm_4(unorm _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
        _M_w = _V;
    }

    explicit unorm_4(float _V) __GPU
    {
        _M_x = unorm(_V);
        _M_y = unorm(_V);
        _M_z = unorm(_V);
        _M_w = unorm(_V);
    }

    /// <summary>
    ///     Copy constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object to copy from.
    /// </param>
    unorm_4(const unorm_4& _Other) __GPU
    {
        *this = _Other;
    }

    unorm_4& operator=(const unorm_4& _Other) __GPU
    {
        _M_x = _Other._M_x;
        _M_y = _Other._M_y;
        _M_z = _Other._M_z;
        _M_w = _Other._M_w;
        return *this;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_4(const uint_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_4(const int_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_4(const float_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_4(const norm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline unorm_4(const double_4& _Other) __GPU;

    unorm_4& operator++() __GPU
    {
        unorm_4 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        ++_Value._M_w;
        *this = _Value;
        return *this;
    }

    unorm_4 operator++(int) __GPU
    {
        unorm_4 _Result = *this;
        ++(*this);
        return _Result;
    }

    unorm_4& operator--() __GPU
    {
        unorm_4 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        --_Value._M_w;
        *this = _Value;
        return *this;
    }

    unorm_4 operator--(int) __GPU
    {
        unorm_4 _Result = *this;
        --(*this);
        return _Result;
    }

    unorm_4& operator+=(const unorm_4& _Other) __GPU
    {
        unorm_4 _Value1 = *this;
        unorm_4 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        _Value1.w += _Value2.w;
        *this = _Value1;
        return *this;
    }

    unorm_4& operator-=(const unorm_4& _Other) __GPU
    {
        unorm_4 _Value1 = *this;
        unorm_4 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        _Value1.w -= _Value2.w;
        *this = _Value1;
        return *this;
    }

    unorm_4& operator*=(const unorm_4& _Other) __GPU
    {
        unorm_4 _Value1 = *this;
        unorm_4 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        _Value1.w *= _Value2.w;
        *this = _Value1;
        return *this;
    }

    unorm_4& operator/=(const unorm_4& _Other) __GPU
    {
        unorm_4 _Value1 = *this;
        unorm_4 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        _Value1.w /= _Value2.w;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) unorm_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) unorm_2 rg;

    unorm_2 get_xy() const __GPU {
        return unorm_2(_M_x, _M_y);
    }

    void set_xy(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) unorm_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) unorm_2 rb;

    unorm_2 get_xz() const __GPU {
        return unorm_2(_M_x, _M_z);
    }

    void set_xz(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xw, put = set_xw) ) unorm_2 xw;
    __declspec( property( get = get_xw, put = set_xw) ) unorm_2 ra;

    unorm_2 get_xw() const __GPU {
        return unorm_2(_M_x, _M_w);
    }

    void set_xw(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) unorm_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) unorm_2 gr;

    unorm_2 get_yx() const __GPU {
        return unorm_2(_M_y, _M_x);
    }

    void set_yx(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) unorm_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) unorm_2 gb;

    unorm_2 get_yz() const __GPU {
        return unorm_2(_M_y, _M_z);
    }

    void set_yz(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yw, put = set_yw) ) unorm_2 yw;
    __declspec( property( get = get_yw, put = set_yw) ) unorm_2 ga;

    unorm_2 get_yw() const __GPU {
        return unorm_2(_M_y, _M_w);
    }

    void set_yw(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) unorm_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) unorm_2 br;

    unorm_2 get_zx() const __GPU {
        return unorm_2(_M_z, _M_x);
    }

    void set_zx(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) unorm_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) unorm_2 bg;

    unorm_2 get_zy() const __GPU {
        return unorm_2(_M_z, _M_y);
    }

    void set_zy(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_zw, put = set_zw) ) unorm_2 zw;
    __declspec( property( get = get_zw, put = set_zw) ) unorm_2 ba;

    unorm_2 get_zw() const __GPU {
        return unorm_2(_M_z, _M_w);
    }

    void set_zw(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_wx, put = set_wx) ) unorm_2 wx;
    __declspec( property( get = get_wx, put = set_wx) ) unorm_2 ar;

    unorm_2 get_wx() const __GPU {
        return unorm_2(_M_w, _M_x);
    }

    void set_wx(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_wy, put = set_wy) ) unorm_2 wy;
    __declspec( property( get = get_wy, put = set_wy) ) unorm_2 ag;

    unorm_2 get_wy() const __GPU {
        return unorm_2(_M_w, _M_y);
    }

    void set_wy(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_wz, put = set_wz) ) unorm_2 wz;
    __declspec( property( get = get_wz, put = set_wz) ) unorm_2 ab;

    unorm_2 get_wz() const __GPU {
        return unorm_2(_M_w, _M_z);
    }

    void set_wz(const unorm_2& _Value) __GPU
    {
        unorm_2 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) unorm_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) unorm_3 rgb;

    unorm_3 get_xyz() const __GPU {
        return unorm_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xyw, put = set_xyw) ) unorm_3 xyw;
    __declspec( property( get = get_xyw, put = set_xyw) ) unorm_3 rga;

    unorm_3 get_xyw() const __GPU {
        return unorm_3(_M_x, _M_y, _M_w);
    }

    void set_xyw(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) unorm_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) unorm_3 rbg;

    unorm_3 get_xzy() const __GPU {
        return unorm_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xzw, put = set_xzw) ) unorm_3 xzw;
    __declspec( property( get = get_xzw, put = set_xzw) ) unorm_3 rba;

    unorm_3 get_xzw() const __GPU {
        return unorm_3(_M_x, _M_z, _M_w);
    }

    void set_xzw(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xwy, put = set_xwy) ) unorm_3 xwy;
    __declspec( property( get = get_xwy, put = set_xwy) ) unorm_3 rag;

    unorm_3 get_xwy() const __GPU {
        return unorm_3(_M_x, _M_w, _M_y);
    }

    void set_xwy(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xwz, put = set_xwz) ) unorm_3 xwz;
    __declspec( property( get = get_xwz, put = set_xwz) ) unorm_3 rab;

    unorm_3 get_xwz() const __GPU {
        return unorm_3(_M_x, _M_w, _M_z);
    }

    void set_xwz(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) unorm_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) unorm_3 grb;

    unorm_3 get_yxz() const __GPU {
        return unorm_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxw, put = set_yxw) ) unorm_3 yxw;
    __declspec( property( get = get_yxw, put = set_yxw) ) unorm_3 gra;

    unorm_3 get_yxw() const __GPU {
        return unorm_3(_M_y, _M_x, _M_w);
    }

    void set_yxw(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) unorm_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) unorm_3 gbr;

    unorm_3 get_yzx() const __GPU {
        return unorm_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_yzw, put = set_yzw) ) unorm_3 yzw;
    __declspec( property( get = get_yzw, put = set_yzw) ) unorm_3 gba;

    unorm_3 get_yzw() const __GPU {
        return unorm_3(_M_y, _M_z, _M_w);
    }

    void set_yzw(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_ywx, put = set_ywx) ) unorm_3 ywx;
    __declspec( property( get = get_ywx, put = set_ywx) ) unorm_3 gar;

    unorm_3 get_ywx() const __GPU {
        return unorm_3(_M_y, _M_w, _M_x);
    }

    void set_ywx(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_ywz, put = set_ywz) ) unorm_3 ywz;
    __declspec( property( get = get_ywz, put = set_ywz) ) unorm_3 gab;

    unorm_3 get_ywz() const __GPU {
        return unorm_3(_M_y, _M_w, _M_z);
    }

    void set_ywz(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) unorm_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) unorm_3 brg;

    unorm_3 get_zxy() const __GPU {
        return unorm_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zxw, put = set_zxw) ) unorm_3 zxw;
    __declspec( property( get = get_zxw, put = set_zxw) ) unorm_3 bra;

    unorm_3 get_zxw() const __GPU {
        return unorm_3(_M_z, _M_x, _M_w);
    }

    void set_zxw(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) unorm_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) unorm_3 bgr;

    unorm_3 get_zyx() const __GPU {
        return unorm_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zyw, put = set_zyw) ) unorm_3 zyw;
    __declspec( property( get = get_zyw, put = set_zyw) ) unorm_3 bga;

    unorm_3 get_zyw() const __GPU {
        return unorm_3(_M_z, _M_y, _M_w);
    }

    void set_zyw(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zwx, put = set_zwx) ) unorm_3 zwx;
    __declspec( property( get = get_zwx, put = set_zwx) ) unorm_3 bar;

    unorm_3 get_zwx() const __GPU {
        return unorm_3(_M_z, _M_w, _M_x);
    }

    void set_zwx(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zwy, put = set_zwy) ) unorm_3 zwy;
    __declspec( property( get = get_zwy, put = set_zwy) ) unorm_3 bag;

    unorm_3 get_zwy() const __GPU {
        return unorm_3(_M_z, _M_w, _M_y);
    }

    void set_zwy(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxy, put = set_wxy) ) unorm_3 wxy;
    __declspec( property( get = get_wxy, put = set_wxy) ) unorm_3 arg;

    unorm_3 get_wxy() const __GPU {
        return unorm_3(_M_w, _M_x, _M_y);
    }

    void set_wxy(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxz, put = set_wxz) ) unorm_3 wxz;
    __declspec( property( get = get_wxz, put = set_wxz) ) unorm_3 arb;

    unorm_3 get_wxz() const __GPU {
        return unorm_3(_M_w, _M_x, _M_z);
    }

    void set_wxz(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wyx, put = set_wyx) ) unorm_3 wyx;
    __declspec( property( get = get_wyx, put = set_wyx) ) unorm_3 agr;

    unorm_3 get_wyx() const __GPU {
        return unorm_3(_M_w, _M_y, _M_x);
    }

    void set_wyx(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wyz, put = set_wyz) ) unorm_3 wyz;
    __declspec( property( get = get_wyz, put = set_wyz) ) unorm_3 agb;

    unorm_3 get_wyz() const __GPU {
        return unorm_3(_M_w, _M_y, _M_z);
    }

    void set_wyz(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wzx, put = set_wzx) ) unorm_3 wzx;
    __declspec( property( get = get_wzx, put = set_wzx) ) unorm_3 abr;

    unorm_3 get_wzx() const __GPU {
        return unorm_3(_M_w, _M_z, _M_x);
    }

    void set_wzx(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wzy, put = set_wzy) ) unorm_3 wzy;
    __declspec( property( get = get_wzy, put = set_wzy) ) unorm_3 abg;

    unorm_3 get_wzy() const __GPU {
        return unorm_3(_M_w, _M_z, _M_y);
    }

    void set_wzy(const unorm_3& _Value) __GPU
    {
        unorm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xyzw, put = set_xyzw) ) unorm_4 xyzw;
    __declspec( property( get = get_xyzw, put = set_xyzw) ) unorm_4 rgba;

    unorm_4 get_xyzw() const __GPU {
        return unorm_4(_M_x, _M_y, _M_z, _M_w);
    }

    void set_xyzw(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xywz, put = set_xywz) ) unorm_4 xywz;
    __declspec( property( get = get_xywz, put = set_xywz) ) unorm_4 rgab;

    unorm_4 get_xywz() const __GPU {
        return unorm_4(_M_x, _M_y, _M_w, _M_z);
    }

    void set_xywz(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xzyw, put = set_xzyw) ) unorm_4 xzyw;
    __declspec( property( get = get_xzyw, put = set_xzyw) ) unorm_4 rbga;

    unorm_4 get_xzyw() const __GPU {
        return unorm_4(_M_x, _M_z, _M_y, _M_w);
    }

    void set_xzyw(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xzwy, put = set_xzwy) ) unorm_4 xzwy;
    __declspec( property( get = get_xzwy, put = set_xzwy) ) unorm_4 rbag;

    unorm_4 get_xzwy() const __GPU {
        return unorm_4(_M_x, _M_z, _M_w, _M_y);
    }

    void set_xzwy(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_xwyz, put = set_xwyz) ) unorm_4 xwyz;
    __declspec( property( get = get_xwyz, put = set_xwyz) ) unorm_4 ragb;

    unorm_4 get_xwyz() const __GPU {
        return unorm_4(_M_x, _M_w, _M_y, _M_z);
    }

    void set_xwyz(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xwzy, put = set_xwzy) ) unorm_4 xwzy;
    __declspec( property( get = get_xwzy, put = set_xwzy) ) unorm_4 rabg;

    unorm_4 get_xwzy() const __GPU {
        return unorm_4(_M_x, _M_w, _M_z, _M_y);
    }

    void set_xwzy(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_yxzw, put = set_yxzw) ) unorm_4 yxzw;
    __declspec( property( get = get_yxzw, put = set_yxzw) ) unorm_4 grba;

    unorm_4 get_yxzw() const __GPU {
        return unorm_4(_M_y, _M_x, _M_z, _M_w);
    }

    void set_yxzw(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yxwz, put = set_yxwz) ) unorm_4 yxwz;
    __declspec( property( get = get_yxwz, put = set_yxwz) ) unorm_4 grab;

    unorm_4 get_yxwz() const __GPU {
        return unorm_4(_M_y, _M_x, _M_w, _M_z);
    }

    void set_yxwz(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_yzxw, put = set_yzxw) ) unorm_4 yzxw;
    __declspec( property( get = get_yzxw, put = set_yzxw) ) unorm_4 gbra;

    unorm_4 get_yzxw() const __GPU {
        return unorm_4(_M_y, _M_z, _M_x, _M_w);
    }

    void set_yzxw(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yzwx, put = set_yzwx) ) unorm_4 yzwx;
    __declspec( property( get = get_yzwx, put = set_yzwx) ) unorm_4 gbar;

    unorm_4 get_yzwx() const __GPU {
        return unorm_4(_M_y, _M_z, _M_w, _M_x);
    }

    void set_yzwx(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_ywxz, put = set_ywxz) ) unorm_4 ywxz;
    __declspec( property( get = get_ywxz, put = set_ywxz) ) unorm_4 garb;

    unorm_4 get_ywxz() const __GPU {
        return unorm_4(_M_y, _M_w, _M_x, _M_z);
    }

    void set_ywxz(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_ywzx, put = set_ywzx) ) unorm_4 ywzx;
    __declspec( property( get = get_ywzx, put = set_ywzx) ) unorm_4 gabr;

    unorm_4 get_ywzx() const __GPU {
        return unorm_4(_M_y, _M_w, _M_z, _M_x);
    }

    void set_ywzx(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zxyw, put = set_zxyw) ) unorm_4 zxyw;
    __declspec( property( get = get_zxyw, put = set_zxyw) ) unorm_4 brga;

    unorm_4 get_zxyw() const __GPU {
        return unorm_4(_M_z, _M_x, _M_y, _M_w);
    }

    void set_zxyw(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zxwy, put = set_zxwy) ) unorm_4 zxwy;
    __declspec( property( get = get_zxwy, put = set_zxwy) ) unorm_4 brag;

    unorm_4 get_zxwy() const __GPU {
        return unorm_4(_M_z, _M_x, _M_w, _M_y);
    }

    void set_zxwy(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zyxw, put = set_zyxw) ) unorm_4 zyxw;
    __declspec( property( get = get_zyxw, put = set_zyxw) ) unorm_4 bgra;

    unorm_4 get_zyxw() const __GPU {
        return unorm_4(_M_z, _M_y, _M_x, _M_w);
    }

    void set_zyxw(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zywx, put = set_zywx) ) unorm_4 zywx;
    __declspec( property( get = get_zywx, put = set_zywx) ) unorm_4 bgar;

    unorm_4 get_zywx() const __GPU {
        return unorm_4(_M_z, _M_y, _M_w, _M_x);
    }

    void set_zywx(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zwxy, put = set_zwxy) ) unorm_4 zwxy;
    __declspec( property( get = get_zwxy, put = set_zwxy) ) unorm_4 barg;

    unorm_4 get_zwxy() const __GPU {
        return unorm_4(_M_z, _M_w, _M_x, _M_y);
    }

    void set_zwxy(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zwyx, put = set_zwyx) ) unorm_4 zwyx;
    __declspec( property( get = get_zwyx, put = set_zwyx) ) unorm_4 bagr;

    unorm_4 get_zwyx() const __GPU {
        return unorm_4(_M_z, _M_w, _M_y, _M_x);
    }

    void set_zwyx(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wxyz, put = set_wxyz) ) unorm_4 wxyz;
    __declspec( property( get = get_wxyz, put = set_wxyz) ) unorm_4 argb;

    unorm_4 get_wxyz() const __GPU {
        return unorm_4(_M_w, _M_x, _M_y, _M_z);
    }

    void set_wxyz(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wxzy, put = set_wxzy) ) unorm_4 wxzy;
    __declspec( property( get = get_wxzy, put = set_wxzy) ) unorm_4 arbg;

    unorm_4 get_wxzy() const __GPU {
        return unorm_4(_M_w, _M_x, _M_z, _M_y);
    }

    void set_wxzy(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wyxz, put = set_wyxz) ) unorm_4 wyxz;
    __declspec( property( get = get_wyxz, put = set_wyxz) ) unorm_4 agrb;

    unorm_4 get_wyxz() const __GPU {
        return unorm_4(_M_w, _M_y, _M_x, _M_z);
    }

    void set_wyxz(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wyzx, put = set_wyzx) ) unorm_4 wyzx;
    __declspec( property( get = get_wyzx, put = set_wyzx) ) unorm_4 agbr;

    unorm_4 get_wyzx() const __GPU {
        return unorm_4(_M_w, _M_y, _M_z, _M_x);
    }

    void set_wyzx(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wzxy, put = set_wzxy) ) unorm_4 wzxy;
    __declspec( property( get = get_wzxy, put = set_wzxy) ) unorm_4 abrg;

    unorm_4 get_wzxy() const __GPU {
        return unorm_4(_M_w, _M_z, _M_x, _M_y);
    }

    void set_wzxy(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wzyx, put = set_wzyx) ) unorm_4 wzyx;
    __declspec( property( get = get_wzyx, put = set_wzyx) ) unorm_4 abgr;

    unorm_4 get_wzyx() const __GPU {
        return unorm_4(_M_w, _M_z, _M_y, _M_x);
    }

    void set_wzyx(const unorm_4& _Value) __GPU
    {
        unorm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

};

/// <summary>
///    Represent a short vector of 2 norm's.
/// </summary>
class norm_2
{
public:
    typedef norm value_type;
    static const int size = 2;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Norm_type;
private:
    value_type _M_x;
    value_type _M_y;

public:
    __declspec( property( get = get_x, put = set_x) ) norm x;
    __declspec( property( get = get_x, put = set_x) ) norm r;

    norm get_x() const __GPU {
        return _M_x;
    }

    void set_x(norm _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) norm y;
    __declspec( property( get = get_y, put = set_y) ) norm g;

    norm get_y() const __GPU {
        return _M_y;
    }

    void set_y(norm _Value) __GPU
    {
        _M_y = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    norm_2() __GPU
    {
        _M_x = norm(0.0f);
        _M_y = norm(0.0f);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    norm_2(norm _V0, norm _V1) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    norm_2(float _V0, float _V1) __GPU
    {
        _M_x = norm(_V0);
        _M_y = norm(_V1);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    norm_2(unorm _V0, unorm _V1) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    norm_2(norm _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
    }

    explicit norm_2(float _V) __GPU
    {
        _M_x = norm(_V);
        _M_y = norm(_V);
    }

    /// <summary>
    ///     Copy constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object to copy from.
    /// </param>
    norm_2(const norm_2& _Other) __GPU
    {
        *this = _Other;
    }

    norm_2& operator=(const norm_2& _Other) __GPU
    {
        _M_x = _Other._M_x;
        _M_y = _Other._M_y;
        return *this;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_2(const uint_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_2(const int_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_2(const float_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_2(const unorm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_2(const double_2& _Other) __GPU;

    norm_2 operator-() const __GPU
    {
        norm_2 _Value = *this;
        return norm_2(-_Value.x, -_Value.y);
    }

    norm_2& operator++() __GPU
    {
        norm_2 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        *this = _Value;
        return *this;
    }

    norm_2 operator++(int) __GPU
    {
        norm_2 _Result = *this;
        ++(*this);
        return _Result;
    }

    norm_2& operator--() __GPU
    {
        norm_2 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        *this = _Value;
        return *this;
    }

    norm_2 operator--(int) __GPU
    {
        norm_2 _Result = *this;
        --(*this);
        return _Result;
    }

    norm_2& operator+=(const norm_2& _Other) __GPU
    {
        norm_2 _Value1 = *this;
        norm_2 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        *this = _Value1;
        return *this;
    }

    norm_2& operator-=(const norm_2& _Other) __GPU
    {
        norm_2 _Value1 = *this;
        norm_2 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        *this = _Value1;
        return *this;
    }

    norm_2& operator*=(const norm_2& _Other) __GPU
    {
        norm_2 _Value1 = *this;
        norm_2 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        *this = _Value1;
        return *this;
    }

    norm_2& operator/=(const norm_2& _Other) __GPU
    {
        norm_2 _Value1 = *this;
        norm_2 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) norm_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) norm_2 rg;

    norm_2 get_xy() const __GPU {
        return norm_2(_M_x, _M_y);
    }

    void set_xy(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) norm_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) norm_2 gr;

    norm_2 get_yx() const __GPU {
        return norm_2(_M_y, _M_x);
    }

    void set_yx(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

};

/// <summary>
///    Represent a short vector of 3 norm's.
/// </summary>
class norm_3
{
public:
    typedef norm value_type;
    static const int size = 3;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Norm_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;

public:
    __declspec( property( get = get_x, put = set_x) ) norm x;
    __declspec( property( get = get_x, put = set_x) ) norm r;

    norm get_x() const __GPU {
        return _M_x;
    }

    void set_x(norm _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) norm y;
    __declspec( property( get = get_y, put = set_y) ) norm g;

    norm get_y() const __GPU {
        return _M_y;
    }

    void set_y(norm _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) norm z;
    __declspec( property( get = get_z, put = set_z) ) norm b;

    norm get_z() const __GPU {
        return _M_z;
    }

    void set_z(norm _Value) __GPU
    {
        _M_z = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    norm_3() __GPU
    {
        _M_x = norm(0.0f);
        _M_y = norm(0.0f);
        _M_z = norm(0.0f);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    norm_3(norm _V0, norm _V1, norm _V2) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    norm_3(float _V0, float _V1, float _V2) __GPU
    {
        _M_x = norm(_V0);
        _M_y = norm(_V1);
        _M_z = norm(_V2);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    norm_3(unorm _V0, unorm _V1, unorm _V2) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    norm_3(norm _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
    }

    explicit norm_3(float _V) __GPU
    {
        _M_x = norm(_V);
        _M_y = norm(_V);
        _M_z = norm(_V);
    }

    /// <summary>
    ///     Copy constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object to copy from.
    /// </param>
    norm_3(const norm_3& _Other) __GPU
    {
        *this = _Other;
    }

    norm_3& operator=(const norm_3& _Other) __GPU
    {
        _M_x = _Other._M_x;
        _M_y = _Other._M_y;
        _M_z = _Other._M_z;
        return *this;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_3(const uint_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_3(const int_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_3(const float_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_3(const unorm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_3(const double_3& _Other) __GPU;

    norm_3 operator-() const __GPU
    {
        norm_3 _Value = *this;
        return norm_3(-_Value.x, -_Value.y, -_Value.z);
    }

    norm_3& operator++() __GPU
    {
        norm_3 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        *this = _Value;
        return *this;
    }

    norm_3 operator++(int) __GPU
    {
        norm_3 _Result = *this;
        ++(*this);
        return _Result;
    }

    norm_3& operator--() __GPU
    {
        norm_3 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        *this = _Value;
        return *this;
    }

    norm_3 operator--(int) __GPU
    {
        norm_3 _Result = *this;
        --(*this);
        return _Result;
    }

    norm_3& operator+=(const norm_3& _Other) __GPU
    {
        norm_3 _Value1 = *this;
        norm_3 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        *this = _Value1;
        return *this;
    }

    norm_3& operator-=(const norm_3& _Other) __GPU
    {
        norm_3 _Value1 = *this;
        norm_3 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        *this = _Value1;
        return *this;
    }

    norm_3& operator*=(const norm_3& _Other) __GPU
    {
        norm_3 _Value1 = *this;
        norm_3 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        *this = _Value1;
        return *this;
    }

    norm_3& operator/=(const norm_3& _Other) __GPU
    {
        norm_3 _Value1 = *this;
        norm_3 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) norm_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) norm_2 rg;

    norm_2 get_xy() const __GPU {
        return norm_2(_M_x, _M_y);
    }

    void set_xy(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) norm_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) norm_2 rb;

    norm_2 get_xz() const __GPU {
        return norm_2(_M_x, _M_z);
    }

    void set_xz(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) norm_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) norm_2 gr;

    norm_2 get_yx() const __GPU {
        return norm_2(_M_y, _M_x);
    }

    void set_yx(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) norm_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) norm_2 gb;

    norm_2 get_yz() const __GPU {
        return norm_2(_M_y, _M_z);
    }

    void set_yz(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) norm_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) norm_2 br;

    norm_2 get_zx() const __GPU {
        return norm_2(_M_z, _M_x);
    }

    void set_zx(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) norm_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) norm_2 bg;

    norm_2 get_zy() const __GPU {
        return norm_2(_M_z, _M_y);
    }

    void set_zy(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) norm_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) norm_3 rgb;

    norm_3 get_xyz() const __GPU {
        return norm_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) norm_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) norm_3 rbg;

    norm_3 get_xzy() const __GPU {
        return norm_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) norm_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) norm_3 grb;

    norm_3 get_yxz() const __GPU {
        return norm_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) norm_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) norm_3 gbr;

    norm_3 get_yzx() const __GPU {
        return norm_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) norm_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) norm_3 brg;

    norm_3 get_zxy() const __GPU {
        return norm_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) norm_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) norm_3 bgr;

    norm_3 get_zyx() const __GPU {
        return norm_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

};

/// <summary>
///    Represent a short vector of 4 norm's.
/// </summary>
class norm_4
{
public:
    typedef norm value_type;
    static const int size = 4;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Norm_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;
    value_type _M_w;

public:
    __declspec( property( get = get_x, put = set_x) ) norm x;
    __declspec( property( get = get_x, put = set_x) ) norm r;

    norm get_x() const __GPU {
        return _M_x;
    }

    void set_x(norm _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) norm y;
    __declspec( property( get = get_y, put = set_y) ) norm g;

    norm get_y() const __GPU {
        return _M_y;
    }

    void set_y(norm _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) norm z;
    __declspec( property( get = get_z, put = set_z) ) norm b;

    norm get_z() const __GPU {
        return _M_z;
    }

    void set_z(norm _Value) __GPU
    {
        _M_z = _Value;
    }

    __declspec( property( get = get_w, put = set_w) ) norm w;
    __declspec( property( get = get_w, put = set_w) ) norm a;

    norm get_w() const __GPU {
        return _M_w;
    }

    void set_w(norm _Value) __GPU
    {
        _M_w = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    norm_4() __GPU
    {
        _M_x = norm(0.0f);
        _M_y = norm(0.0f);
        _M_z = norm(0.0f);
        _M_w = norm(0.0f);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    norm_4(norm _V0, norm _V1, norm _V2, norm _V3) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
        _M_w = _V3;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    norm_4(float _V0, float _V1, float _V2, float _V3) __GPU
    {
        _M_x = norm(_V0);
        _M_y = norm(_V1);
        _M_z = norm(_V2);
        _M_w = norm(_V3);
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    norm_4(unorm _V0, unorm _V1, unorm _V2, unorm _V3) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
        _M_w = _V3;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    norm_4(norm _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
        _M_w = _V;
    }

    explicit norm_4(float _V) __GPU
    {
        _M_x = norm(_V);
        _M_y = norm(_V);
        _M_z = norm(_V);
        _M_w = norm(_V);
    }

    /// <summary>
    ///     Copy constructor.
    /// </summary>
    /// <param name="_Other">
    ///     The object to copy from.
    /// </param>
    norm_4(const norm_4& _Other) __GPU
    {
        *this = _Other;
    }

    norm_4& operator=(const norm_4& _Other) __GPU
    {
        _M_x = _Other._M_x;
        _M_y = _Other._M_y;
        _M_z = _Other._M_z;
        _M_w = _Other._M_w;
        return *this;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_4(const uint_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_4(const int_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_4(const float_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_4(const unorm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline norm_4(const double_4& _Other) __GPU;

    norm_4 operator-() const __GPU
    {
        norm_4 _Value = *this;
        return norm_4(-_Value.x, -_Value.y, -_Value.z, -_Value.w);
    }

    norm_4& operator++() __GPU
    {
        norm_4 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        ++_Value._M_w;
        *this = _Value;
        return *this;
    }

    norm_4 operator++(int) __GPU
    {
        norm_4 _Result = *this;
        ++(*this);
        return _Result;
    }

    norm_4& operator--() __GPU
    {
        norm_4 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        --_Value._M_w;
        *this = _Value;
        return *this;
    }

    norm_4 operator--(int) __GPU
    {
        norm_4 _Result = *this;
        --(*this);
        return _Result;
    }

    norm_4& operator+=(const norm_4& _Other) __GPU
    {
        norm_4 _Value1 = *this;
        norm_4 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        _Value1.w += _Value2.w;
        *this = _Value1;
        return *this;
    }

    norm_4& operator-=(const norm_4& _Other) __GPU
    {
        norm_4 _Value1 = *this;
        norm_4 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        _Value1.w -= _Value2.w;
        *this = _Value1;
        return *this;
    }

    norm_4& operator*=(const norm_4& _Other) __GPU
    {
        norm_4 _Value1 = *this;
        norm_4 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        _Value1.w *= _Value2.w;
        *this = _Value1;
        return *this;
    }

    norm_4& operator/=(const norm_4& _Other) __GPU
    {
        norm_4 _Value1 = *this;
        norm_4 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        _Value1.w /= _Value2.w;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) norm_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) norm_2 rg;

    norm_2 get_xy() const __GPU {
        return norm_2(_M_x, _M_y);
    }

    void set_xy(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) norm_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) norm_2 rb;

    norm_2 get_xz() const __GPU {
        return norm_2(_M_x, _M_z);
    }

    void set_xz(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xw, put = set_xw) ) norm_2 xw;
    __declspec( property( get = get_xw, put = set_xw) ) norm_2 ra;

    norm_2 get_xw() const __GPU {
        return norm_2(_M_x, _M_w);
    }

    void set_xw(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) norm_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) norm_2 gr;

    norm_2 get_yx() const __GPU {
        return norm_2(_M_y, _M_x);
    }

    void set_yx(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) norm_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) norm_2 gb;

    norm_2 get_yz() const __GPU {
        return norm_2(_M_y, _M_z);
    }

    void set_yz(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yw, put = set_yw) ) norm_2 yw;
    __declspec( property( get = get_yw, put = set_yw) ) norm_2 ga;

    norm_2 get_yw() const __GPU {
        return norm_2(_M_y, _M_w);
    }

    void set_yw(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) norm_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) norm_2 br;

    norm_2 get_zx() const __GPU {
        return norm_2(_M_z, _M_x);
    }

    void set_zx(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) norm_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) norm_2 bg;

    norm_2 get_zy() const __GPU {
        return norm_2(_M_z, _M_y);
    }

    void set_zy(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_zw, put = set_zw) ) norm_2 zw;
    __declspec( property( get = get_zw, put = set_zw) ) norm_2 ba;

    norm_2 get_zw() const __GPU {
        return norm_2(_M_z, _M_w);
    }

    void set_zw(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_wx, put = set_wx) ) norm_2 wx;
    __declspec( property( get = get_wx, put = set_wx) ) norm_2 ar;

    norm_2 get_wx() const __GPU {
        return norm_2(_M_w, _M_x);
    }

    void set_wx(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_wy, put = set_wy) ) norm_2 wy;
    __declspec( property( get = get_wy, put = set_wy) ) norm_2 ag;

    norm_2 get_wy() const __GPU {
        return norm_2(_M_w, _M_y);
    }

    void set_wy(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_wz, put = set_wz) ) norm_2 wz;
    __declspec( property( get = get_wz, put = set_wz) ) norm_2 ab;

    norm_2 get_wz() const __GPU {
        return norm_2(_M_w, _M_z);
    }

    void set_wz(const norm_2& _Value) __GPU
    {
        norm_2 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) norm_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) norm_3 rgb;

    norm_3 get_xyz() const __GPU {
        return norm_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xyw, put = set_xyw) ) norm_3 xyw;
    __declspec( property( get = get_xyw, put = set_xyw) ) norm_3 rga;

    norm_3 get_xyw() const __GPU {
        return norm_3(_M_x, _M_y, _M_w);
    }

    void set_xyw(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) norm_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) norm_3 rbg;

    norm_3 get_xzy() const __GPU {
        return norm_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xzw, put = set_xzw) ) norm_3 xzw;
    __declspec( property( get = get_xzw, put = set_xzw) ) norm_3 rba;

    norm_3 get_xzw() const __GPU {
        return norm_3(_M_x, _M_z, _M_w);
    }

    void set_xzw(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xwy, put = set_xwy) ) norm_3 xwy;
    __declspec( property( get = get_xwy, put = set_xwy) ) norm_3 rag;

    norm_3 get_xwy() const __GPU {
        return norm_3(_M_x, _M_w, _M_y);
    }

    void set_xwy(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xwz, put = set_xwz) ) norm_3 xwz;
    __declspec( property( get = get_xwz, put = set_xwz) ) norm_3 rab;

    norm_3 get_xwz() const __GPU {
        return norm_3(_M_x, _M_w, _M_z);
    }

    void set_xwz(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) norm_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) norm_3 grb;

    norm_3 get_yxz() const __GPU {
        return norm_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxw, put = set_yxw) ) norm_3 yxw;
    __declspec( property( get = get_yxw, put = set_yxw) ) norm_3 gra;

    norm_3 get_yxw() const __GPU {
        return norm_3(_M_y, _M_x, _M_w);
    }

    void set_yxw(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) norm_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) norm_3 gbr;

    norm_3 get_yzx() const __GPU {
        return norm_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_yzw, put = set_yzw) ) norm_3 yzw;
    __declspec( property( get = get_yzw, put = set_yzw) ) norm_3 gba;

    norm_3 get_yzw() const __GPU {
        return norm_3(_M_y, _M_z, _M_w);
    }

    void set_yzw(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_ywx, put = set_ywx) ) norm_3 ywx;
    __declspec( property( get = get_ywx, put = set_ywx) ) norm_3 gar;

    norm_3 get_ywx() const __GPU {
        return norm_3(_M_y, _M_w, _M_x);
    }

    void set_ywx(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_ywz, put = set_ywz) ) norm_3 ywz;
    __declspec( property( get = get_ywz, put = set_ywz) ) norm_3 gab;

    norm_3 get_ywz() const __GPU {
        return norm_3(_M_y, _M_w, _M_z);
    }

    void set_ywz(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) norm_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) norm_3 brg;

    norm_3 get_zxy() const __GPU {
        return norm_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zxw, put = set_zxw) ) norm_3 zxw;
    __declspec( property( get = get_zxw, put = set_zxw) ) norm_3 bra;

    norm_3 get_zxw() const __GPU {
        return norm_3(_M_z, _M_x, _M_w);
    }

    void set_zxw(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) norm_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) norm_3 bgr;

    norm_3 get_zyx() const __GPU {
        return norm_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zyw, put = set_zyw) ) norm_3 zyw;
    __declspec( property( get = get_zyw, put = set_zyw) ) norm_3 bga;

    norm_3 get_zyw() const __GPU {
        return norm_3(_M_z, _M_y, _M_w);
    }

    void set_zyw(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zwx, put = set_zwx) ) norm_3 zwx;
    __declspec( property( get = get_zwx, put = set_zwx) ) norm_3 bar;

    norm_3 get_zwx() const __GPU {
        return norm_3(_M_z, _M_w, _M_x);
    }

    void set_zwx(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zwy, put = set_zwy) ) norm_3 zwy;
    __declspec( property( get = get_zwy, put = set_zwy) ) norm_3 bag;

    norm_3 get_zwy() const __GPU {
        return norm_3(_M_z, _M_w, _M_y);
    }

    void set_zwy(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxy, put = set_wxy) ) norm_3 wxy;
    __declspec( property( get = get_wxy, put = set_wxy) ) norm_3 arg;

    norm_3 get_wxy() const __GPU {
        return norm_3(_M_w, _M_x, _M_y);
    }

    void set_wxy(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxz, put = set_wxz) ) norm_3 wxz;
    __declspec( property( get = get_wxz, put = set_wxz) ) norm_3 arb;

    norm_3 get_wxz() const __GPU {
        return norm_3(_M_w, _M_x, _M_z);
    }

    void set_wxz(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wyx, put = set_wyx) ) norm_3 wyx;
    __declspec( property( get = get_wyx, put = set_wyx) ) norm_3 agr;

    norm_3 get_wyx() const __GPU {
        return norm_3(_M_w, _M_y, _M_x);
    }

    void set_wyx(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wyz, put = set_wyz) ) norm_3 wyz;
    __declspec( property( get = get_wyz, put = set_wyz) ) norm_3 agb;

    norm_3 get_wyz() const __GPU {
        return norm_3(_M_w, _M_y, _M_z);
    }

    void set_wyz(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wzx, put = set_wzx) ) norm_3 wzx;
    __declspec( property( get = get_wzx, put = set_wzx) ) norm_3 abr;

    norm_3 get_wzx() const __GPU {
        return norm_3(_M_w, _M_z, _M_x);
    }

    void set_wzx(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wzy, put = set_wzy) ) norm_3 wzy;
    __declspec( property( get = get_wzy, put = set_wzy) ) norm_3 abg;

    norm_3 get_wzy() const __GPU {
        return norm_3(_M_w, _M_z, _M_y);
    }

    void set_wzy(const norm_3& _Value) __GPU
    {
        norm_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xyzw, put = set_xyzw) ) norm_4 xyzw;
    __declspec( property( get = get_xyzw, put = set_xyzw) ) norm_4 rgba;

    norm_4 get_xyzw() const __GPU {
        return norm_4(_M_x, _M_y, _M_z, _M_w);
    }

    void set_xyzw(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xywz, put = set_xywz) ) norm_4 xywz;
    __declspec( property( get = get_xywz, put = set_xywz) ) norm_4 rgab;

    norm_4 get_xywz() const __GPU {
        return norm_4(_M_x, _M_y, _M_w, _M_z);
    }

    void set_xywz(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xzyw, put = set_xzyw) ) norm_4 xzyw;
    __declspec( property( get = get_xzyw, put = set_xzyw) ) norm_4 rbga;

    norm_4 get_xzyw() const __GPU {
        return norm_4(_M_x, _M_z, _M_y, _M_w);
    }

    void set_xzyw(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xzwy, put = set_xzwy) ) norm_4 xzwy;
    __declspec( property( get = get_xzwy, put = set_xzwy) ) norm_4 rbag;

    norm_4 get_xzwy() const __GPU {
        return norm_4(_M_x, _M_z, _M_w, _M_y);
    }

    void set_xzwy(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_xwyz, put = set_xwyz) ) norm_4 xwyz;
    __declspec( property( get = get_xwyz, put = set_xwyz) ) norm_4 ragb;

    norm_4 get_xwyz() const __GPU {
        return norm_4(_M_x, _M_w, _M_y, _M_z);
    }

    void set_xwyz(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xwzy, put = set_xwzy) ) norm_4 xwzy;
    __declspec( property( get = get_xwzy, put = set_xwzy) ) norm_4 rabg;

    norm_4 get_xwzy() const __GPU {
        return norm_4(_M_x, _M_w, _M_z, _M_y);
    }

    void set_xwzy(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_yxzw, put = set_yxzw) ) norm_4 yxzw;
    __declspec( property( get = get_yxzw, put = set_yxzw) ) norm_4 grba;

    norm_4 get_yxzw() const __GPU {
        return norm_4(_M_y, _M_x, _M_z, _M_w);
    }

    void set_yxzw(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yxwz, put = set_yxwz) ) norm_4 yxwz;
    __declspec( property( get = get_yxwz, put = set_yxwz) ) norm_4 grab;

    norm_4 get_yxwz() const __GPU {
        return norm_4(_M_y, _M_x, _M_w, _M_z);
    }

    void set_yxwz(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_yzxw, put = set_yzxw) ) norm_4 yzxw;
    __declspec( property( get = get_yzxw, put = set_yzxw) ) norm_4 gbra;

    norm_4 get_yzxw() const __GPU {
        return norm_4(_M_y, _M_z, _M_x, _M_w);
    }

    void set_yzxw(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yzwx, put = set_yzwx) ) norm_4 yzwx;
    __declspec( property( get = get_yzwx, put = set_yzwx) ) norm_4 gbar;

    norm_4 get_yzwx() const __GPU {
        return norm_4(_M_y, _M_z, _M_w, _M_x);
    }

    void set_yzwx(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_ywxz, put = set_ywxz) ) norm_4 ywxz;
    __declspec( property( get = get_ywxz, put = set_ywxz) ) norm_4 garb;

    norm_4 get_ywxz() const __GPU {
        return norm_4(_M_y, _M_w, _M_x, _M_z);
    }

    void set_ywxz(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_ywzx, put = set_ywzx) ) norm_4 ywzx;
    __declspec( property( get = get_ywzx, put = set_ywzx) ) norm_4 gabr;

    norm_4 get_ywzx() const __GPU {
        return norm_4(_M_y, _M_w, _M_z, _M_x);
    }

    void set_ywzx(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zxyw, put = set_zxyw) ) norm_4 zxyw;
    __declspec( property( get = get_zxyw, put = set_zxyw) ) norm_4 brga;

    norm_4 get_zxyw() const __GPU {
        return norm_4(_M_z, _M_x, _M_y, _M_w);
    }

    void set_zxyw(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zxwy, put = set_zxwy) ) norm_4 zxwy;
    __declspec( property( get = get_zxwy, put = set_zxwy) ) norm_4 brag;

    norm_4 get_zxwy() const __GPU {
        return norm_4(_M_z, _M_x, _M_w, _M_y);
    }

    void set_zxwy(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zyxw, put = set_zyxw) ) norm_4 zyxw;
    __declspec( property( get = get_zyxw, put = set_zyxw) ) norm_4 bgra;

    norm_4 get_zyxw() const __GPU {
        return norm_4(_M_z, _M_y, _M_x, _M_w);
    }

    void set_zyxw(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zywx, put = set_zywx) ) norm_4 zywx;
    __declspec( property( get = get_zywx, put = set_zywx) ) norm_4 bgar;

    norm_4 get_zywx() const __GPU {
        return norm_4(_M_z, _M_y, _M_w, _M_x);
    }

    void set_zywx(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zwxy, put = set_zwxy) ) norm_4 zwxy;
    __declspec( property( get = get_zwxy, put = set_zwxy) ) norm_4 barg;

    norm_4 get_zwxy() const __GPU {
        return norm_4(_M_z, _M_w, _M_x, _M_y);
    }

    void set_zwxy(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zwyx, put = set_zwyx) ) norm_4 zwyx;
    __declspec( property( get = get_zwyx, put = set_zwyx) ) norm_4 bagr;

    norm_4 get_zwyx() const __GPU {
        return norm_4(_M_z, _M_w, _M_y, _M_x);
    }

    void set_zwyx(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wxyz, put = set_wxyz) ) norm_4 wxyz;
    __declspec( property( get = get_wxyz, put = set_wxyz) ) norm_4 argb;

    norm_4 get_wxyz() const __GPU {
        return norm_4(_M_w, _M_x, _M_y, _M_z);
    }

    void set_wxyz(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wxzy, put = set_wxzy) ) norm_4 wxzy;
    __declspec( property( get = get_wxzy, put = set_wxzy) ) norm_4 arbg;

    norm_4 get_wxzy() const __GPU {
        return norm_4(_M_w, _M_x, _M_z, _M_y);
    }

    void set_wxzy(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wyxz, put = set_wyxz) ) norm_4 wyxz;
    __declspec( property( get = get_wyxz, put = set_wyxz) ) norm_4 agrb;

    norm_4 get_wyxz() const __GPU {
        return norm_4(_M_w, _M_y, _M_x, _M_z);
    }

    void set_wyxz(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wyzx, put = set_wyzx) ) norm_4 wyzx;
    __declspec( property( get = get_wyzx, put = set_wyzx) ) norm_4 agbr;

    norm_4 get_wyzx() const __GPU {
        return norm_4(_M_w, _M_y, _M_z, _M_x);
    }

    void set_wyzx(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wzxy, put = set_wzxy) ) norm_4 wzxy;
    __declspec( property( get = get_wzxy, put = set_wzxy) ) norm_4 abrg;

    norm_4 get_wzxy() const __GPU {
        return norm_4(_M_w, _M_z, _M_x, _M_y);
    }

    void set_wzxy(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wzyx, put = set_wzyx) ) norm_4 wzyx;
    __declspec( property( get = get_wzyx, put = set_wzyx) ) norm_4 abgr;

    norm_4 get_wzyx() const __GPU {
        return norm_4(_M_w, _M_z, _M_y, _M_x);
    }

    void set_wzyx(const norm_4& _Value) __GPU
    {
        norm_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

};

/// <summary>
///    Represent a short vector of 2 double's.
/// </summary>
class double_2
{
public:
    typedef double value_type;
    static const int size = 2;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Double_type;
private:
    value_type _M_x;
    value_type _M_y;

public:
    __declspec( property( get = get_x, put = set_x) ) double x;
    __declspec( property( get = get_x, put = set_x) ) double r;

    double get_x() const __GPU {
        return _M_x;
    }

    void set_x(double _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) double y;
    __declspec( property( get = get_y, put = set_y) ) double g;

    double get_y() const __GPU {
        return _M_y;
    }

    void set_y(double _Value) __GPU
    {
        _M_y = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    double_2() __GPU
    {
        _M_x = 0;
        _M_y = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    double_2(double _V0, double _V1) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    double_2(double _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_2(const uint_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_2(const int_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_2(const float_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_2(const unorm_2& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_2(const norm_2& _Other) __GPU;

    double_2 operator-() const __GPU
    {
        double_2 _Value = *this;
        return double_2(-_Value.x, -_Value.y);
    }

    double_2& operator++() __GPU
    {
        double_2 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        *this = _Value;
        return *this;
    }

    double_2 operator++(int) __GPU
    {
        double_2 _Result = *this;
        ++(*this);
        return _Result;
    }

    double_2& operator--() __GPU
    {
        double_2 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        *this = _Value;
        return *this;
    }

    double_2 operator--(int) __GPU
    {
        double_2 _Result = *this;
        --(*this);
        return _Result;
    }

    double_2& operator+=(const double_2& _Other) __GPU
    {
        double_2 _Value1 = *this;
        double_2 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        *this = _Value1;
        return *this;
    }

    double_2& operator-=(const double_2& _Other) __GPU
    {
        double_2 _Value1 = *this;
        double_2 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        *this = _Value1;
        return *this;
    }

    double_2& operator*=(const double_2& _Other) __GPU
    {
        double_2 _Value1 = *this;
        double_2 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        *this = _Value1;
        return *this;
    }

    double_2& operator/=(const double_2& _Other) __GPU
    {
        double_2 _Value1 = *this;
        double_2 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) double_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) double_2 rg;

    double_2 get_xy() const __GPU {
        return double_2(_M_x, _M_y);
    }

    void set_xy(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) double_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) double_2 gr;

    double_2 get_yx() const __GPU {
        return double_2(_M_y, _M_x);
    }

    void set_yx(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

};

/// <summary>
///    Represent a short vector of 3 double's.
/// </summary>
class double_3
{
public:
    typedef double value_type;
    static const int size = 3;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Double_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;

public:
    __declspec( property( get = get_x, put = set_x) ) double x;
    __declspec( property( get = get_x, put = set_x) ) double r;

    double get_x() const __GPU {
        return _M_x;
    }

    void set_x(double _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) double y;
    __declspec( property( get = get_y, put = set_y) ) double g;

    double get_y() const __GPU {
        return _M_y;
    }

    void set_y(double _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) double z;
    __declspec( property( get = get_z, put = set_z) ) double b;

    double get_z() const __GPU {
        return _M_z;
    }

    void set_z(double _Value) __GPU
    {
        _M_z = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    double_3() __GPU
    {
        _M_x = 0;
        _M_y = 0;
        _M_z = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    double_3(double _V0, double _V1, double _V2) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    double_3(double _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_3(const uint_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_3(const int_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_3(const float_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_3(const unorm_3& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_3(const norm_3& _Other) __GPU;

    double_3 operator-() const __GPU
    {
        double_3 _Value = *this;
        return double_3(-_Value.x, -_Value.y, -_Value.z);
    }

    double_3& operator++() __GPU
    {
        double_3 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        *this = _Value;
        return *this;
    }

    double_3 operator++(int) __GPU
    {
        double_3 _Result = *this;
        ++(*this);
        return _Result;
    }

    double_3& operator--() __GPU
    {
        double_3 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        *this = _Value;
        return *this;
    }

    double_3 operator--(int) __GPU
    {
        double_3 _Result = *this;
        --(*this);
        return _Result;
    }

    double_3& operator+=(const double_3& _Other) __GPU
    {
        double_3 _Value1 = *this;
        double_3 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        *this = _Value1;
        return *this;
    }

    double_3& operator-=(const double_3& _Other) __GPU
    {
        double_3 _Value1 = *this;
        double_3 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        *this = _Value1;
        return *this;
    }

    double_3& operator*=(const double_3& _Other) __GPU
    {
        double_3 _Value1 = *this;
        double_3 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        *this = _Value1;
        return *this;
    }

    double_3& operator/=(const double_3& _Other) __GPU
    {
        double_3 _Value1 = *this;
        double_3 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) double_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) double_2 rg;

    double_2 get_xy() const __GPU {
        return double_2(_M_x, _M_y);
    }

    void set_xy(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) double_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) double_2 rb;

    double_2 get_xz() const __GPU {
        return double_2(_M_x, _M_z);
    }

    void set_xz(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) double_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) double_2 gr;

    double_2 get_yx() const __GPU {
        return double_2(_M_y, _M_x);
    }

    void set_yx(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) double_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) double_2 gb;

    double_2 get_yz() const __GPU {
        return double_2(_M_y, _M_z);
    }

    void set_yz(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) double_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) double_2 br;

    double_2 get_zx() const __GPU {
        return double_2(_M_z, _M_x);
    }

    void set_zx(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) double_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) double_2 bg;

    double_2 get_zy() const __GPU {
        return double_2(_M_z, _M_y);
    }

    void set_zy(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) double_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) double_3 rgb;

    double_3 get_xyz() const __GPU {
        return double_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) double_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) double_3 rbg;

    double_3 get_xzy() const __GPU {
        return double_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) double_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) double_3 grb;

    double_3 get_yxz() const __GPU {
        return double_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) double_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) double_3 gbr;

    double_3 get_yzx() const __GPU {
        return double_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) double_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) double_3 brg;

    double_3 get_zxy() const __GPU {
        return double_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) double_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) double_3 bgr;

    double_3 get_zyx() const __GPU {
        return double_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

};

/// <summary>
///    Represent a short vector of 4 double's.
/// </summary>
class double_4
{
public:
    typedef double value_type;
    static const int size = 4;
private:
    static const _Short_vector_base_type_id _Base_type_id = _Double_type;
private:
    value_type _M_x;
    value_type _M_y;
    value_type _M_z;
    value_type _M_w;

public:
    __declspec( property( get = get_x, put = set_x) ) double x;
    __declspec( property( get = get_x, put = set_x) ) double r;

    double get_x() const __GPU {
        return _M_x;
    }

    void set_x(double _Value) __GPU
    {
        _M_x = _Value;
    }

    __declspec( property( get = get_y, put = set_y) ) double y;
    __declspec( property( get = get_y, put = set_y) ) double g;

    double get_y() const __GPU {
        return _M_y;
    }

    void set_y(double _Value) __GPU
    {
        _M_y = _Value;
    }

    __declspec( property( get = get_z, put = set_z) ) double z;
    __declspec( property( get = get_z, put = set_z) ) double b;

    double get_z() const __GPU {
        return _M_z;
    }

    void set_z(double _Value) __GPU
    {
        _M_z = _Value;
    }

    __declspec( property( get = get_w, put = set_w) ) double w;
    __declspec( property( get = get_w, put = set_w) ) double a;

    double get_w() const __GPU {
        return _M_w;
    }

    void set_w(double _Value) __GPU
    {
        _M_w = _Value;
    }

public:
    /// <summary>
    ///     Default constructor, initializes all elements with 0.
    /// </summary>
    double_4() __GPU
    {
        _M_x = 0;
        _M_y = 0;
        _M_z = 0;
        _M_w = 0;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V0">
    ///     The value to initialize element 0.
    /// </param>
    /// <param name="_V1">
    ///     The value to initialize element 1.
    /// </param>
    /// <param name="_V2">
    ///     The value to initialize element 2.
    /// </param>
    /// <param name="_V3">
    ///     The value to initialize element 3.
    /// </param>
    double_4(double _V0, double _V1, double _V2, double _V3) __GPU
    {
        _M_x = _V0;
        _M_y = _V1;
        _M_z = _V2;
        _M_w = _V3;
    }

    /// <summary>
    ///     Constructor.
    /// </summary>
    /// <param name="_V">
    ///     The value for initialization.
    /// </param>
    double_4(double _V) __GPU
    {
        _M_x = _V;
        _M_y = _V;
        _M_z = _V;
        _M_w = _V;
    }

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_4(const uint_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_4(const int_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_4(const float_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_4(const unorm_4& _Other) __GPU;

    /// <summary>
    ///     Constructor.
    ///     Each element is initialized by casting from the corresponding element in _Other.
    /// </summary>
    /// <param name="_Other">
    ///     The object used to initialize.
    /// </param>
    explicit inline double_4(const norm_4& _Other) __GPU;

    double_4 operator-() const __GPU
    {
        double_4 _Value = *this;
        return double_4(-_Value.x, -_Value.y, -_Value.z, -_Value.w);
    }

    double_4& operator++() __GPU
    {
        double_4 _Value = *this;
        ++_Value._M_x;
        ++_Value._M_y;
        ++_Value._M_z;
        ++_Value._M_w;
        *this = _Value;
        return *this;
    }

    double_4 operator++(int) __GPU
    {
        double_4 _Result = *this;
        ++(*this);
        return _Result;
    }

    double_4& operator--() __GPU
    {
        double_4 _Value = *this;
        --_Value._M_x;
        --_Value._M_y;
        --_Value._M_z;
        --_Value._M_w;
        *this = _Value;
        return *this;
    }

    double_4 operator--(int) __GPU
    {
        double_4 _Result = *this;
        --(*this);
        return _Result;
    }

    double_4& operator+=(const double_4& _Other) __GPU
    {
        double_4 _Value1 = *this;
        double_4 _Value2 = _Other;
        _Value1.x += _Value2.x;
        _Value1.y += _Value2.y;
        _Value1.z += _Value2.z;
        _Value1.w += _Value2.w;
        *this = _Value1;
        return *this;
    }

    double_4& operator-=(const double_4& _Other) __GPU
    {
        double_4 _Value1 = *this;
        double_4 _Value2 = _Other;
        _Value1.x -= _Value2.x;
        _Value1.y -= _Value2.y;
        _Value1.z -= _Value2.z;
        _Value1.w -= _Value2.w;
        *this = _Value1;
        return *this;
    }

    double_4& operator*=(const double_4& _Other) __GPU
    {
        double_4 _Value1 = *this;
        double_4 _Value2 = _Other;
        _Value1.x *= _Value2.x;
        _Value1.y *= _Value2.y;
        _Value1.z *= _Value2.z;
        _Value1.w *= _Value2.w;
        *this = _Value1;
        return *this;
    }

    double_4& operator/=(const double_4& _Other) __GPU
    {
        double_4 _Value1 = *this;
        double_4 _Value2 = _Other;
        _Value1.x /= _Value2.x;
        _Value1.y /= _Value2.y;
        _Value1.z /= _Value2.z;
        _Value1.w /= _Value2.w;
        *this = _Value1;
        return *this;
    }

public:
    __declspec( property( get = get_xy, put = set_xy) ) double_2 xy;
    __declspec( property( get = get_xy, put = set_xy) ) double_2 rg;

    double_2 get_xy() const __GPU {
        return double_2(_M_x, _M_y);
    }

    void set_xy(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_xz, put = set_xz) ) double_2 xz;
    __declspec( property( get = get_xz, put = set_xz) ) double_2 rb;

    double_2 get_xz() const __GPU {
        return double_2(_M_x, _M_z);
    }

    void set_xz(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xw, put = set_xw) ) double_2 xw;
    __declspec( property( get = get_xw, put = set_xw) ) double_2 ra;

    double_2 get_xw() const __GPU {
        return double_2(_M_x, _M_w);
    }

    void set_xw(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_yx, put = set_yx) ) double_2 yx;
    __declspec( property( get = get_yx, put = set_yx) ) double_2 gr;

    double_2 get_yx() const __GPU {
        return double_2(_M_y, _M_x);
    }

    void set_yx(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_yz, put = set_yz) ) double_2 yz;
    __declspec( property( get = get_yz, put = set_yz) ) double_2 gb;

    double_2 get_yz() const __GPU {
        return double_2(_M_y, _M_z);
    }

    void set_yz(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_yw, put = set_yw) ) double_2 yw;
    __declspec( property( get = get_yw, put = set_yw) ) double_2 ga;

    double_2 get_yw() const __GPU {
        return double_2(_M_y, _M_w);
    }

    void set_yw(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_zx, put = set_zx) ) double_2 zx;
    __declspec( property( get = get_zx, put = set_zx) ) double_2 br;

    double_2 get_zx() const __GPU {
        return double_2(_M_z, _M_x);
    }

    void set_zx(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_zy, put = set_zy) ) double_2 zy;
    __declspec( property( get = get_zy, put = set_zy) ) double_2 bg;

    double_2 get_zy() const __GPU {
        return double_2(_M_z, _M_y);
    }

    void set_zy(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_zw, put = set_zw) ) double_2 zw;
    __declspec( property( get = get_zw, put = set_zw) ) double_2 ba;

    double_2 get_zw() const __GPU {
        return double_2(_M_z, _M_w);
    }

    void set_zw(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
    }

    __declspec( property( get = get_wx, put = set_wx) ) double_2 wx;
    __declspec( property( get = get_wx, put = set_wx) ) double_2 ar;

    double_2 get_wx() const __GPU {
        return double_2(_M_w, _M_x);
    }

    void set_wx(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
    }

    __declspec( property( get = get_wy, put = set_wy) ) double_2 wy;
    __declspec( property( get = get_wy, put = set_wy) ) double_2 ag;

    double_2 get_wy() const __GPU {
        return double_2(_M_w, _M_y);
    }

    void set_wy(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
    }

    __declspec( property( get = get_wz, put = set_wz) ) double_2 wz;
    __declspec( property( get = get_wz, put = set_wz) ) double_2 ab;

    double_2 get_wz() const __GPU {
        return double_2(_M_w, _M_z);
    }

    void set_wz(const double_2& _Value) __GPU
    {
        double_2 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
    }

    __declspec( property( get = get_xyz, put = set_xyz) ) double_3 xyz;
    __declspec( property( get = get_xyz, put = set_xyz) ) double_3 rgb;

    double_3 get_xyz() const __GPU {
        return double_3(_M_x, _M_y, _M_z);
    }

    void set_xyz(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_xyw, put = set_xyw) ) double_3 xyw;
    __declspec( property( get = get_xyw, put = set_xyw) ) double_3 rga;

    double_3 get_xyw() const __GPU {
        return double_3(_M_x, _M_y, _M_w);
    }

    void set_xyw(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xzy, put = set_xzy) ) double_3 xzy;
    __declspec( property( get = get_xzy, put = set_xzy) ) double_3 rbg;

    double_3 get_xzy() const __GPU {
        return double_3(_M_x, _M_z, _M_y);
    }

    void set_xzy(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xzw, put = set_xzw) ) double_3 xzw;
    __declspec( property( get = get_xzw, put = set_xzw) ) double_3 rba;

    double_3 get_xzw() const __GPU {
        return double_3(_M_x, _M_z, _M_w);
    }

    void set_xzw(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_xwy, put = set_xwy) ) double_3 xwy;
    __declspec( property( get = get_xwy, put = set_xwy) ) double_3 rag;

    double_3 get_xwy() const __GPU {
        return double_3(_M_x, _M_w, _M_y);
    }

    void set_xwy(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xwz, put = set_xwz) ) double_3 xwz;
    __declspec( property( get = get_xwz, put = set_xwz) ) double_3 rab;

    double_3 get_xwz() const __GPU {
        return double_3(_M_x, _M_w, _M_z);
    }

    void set_xwz(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxz, put = set_yxz) ) double_3 yxz;
    __declspec( property( get = get_yxz, put = set_yxz) ) double_3 grb;

    double_3 get_yxz() const __GPU {
        return double_3(_M_y, _M_x, _M_z);
    }

    void set_yxz(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_yxw, put = set_yxw) ) double_3 yxw;
    __declspec( property( get = get_yxw, put = set_yxw) ) double_3 gra;

    double_3 get_yxw() const __GPU {
        return double_3(_M_y, _M_x, _M_w);
    }

    void set_yxw(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_yzx, put = set_yzx) ) double_3 yzx;
    __declspec( property( get = get_yzx, put = set_yzx) ) double_3 gbr;

    double_3 get_yzx() const __GPU {
        return double_3(_M_y, _M_z, _M_x);
    }

    void set_yzx(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_yzw, put = set_yzw) ) double_3 yzw;
    __declspec( property( get = get_yzw, put = set_yzw) ) double_3 gba;

    double_3 get_yzw() const __GPU {
        return double_3(_M_y, _M_z, _M_w);
    }

    void set_yzw(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_ywx, put = set_ywx) ) double_3 ywx;
    __declspec( property( get = get_ywx, put = set_ywx) ) double_3 gar;

    double_3 get_ywx() const __GPU {
        return double_3(_M_y, _M_w, _M_x);
    }

    void set_ywx(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_ywz, put = set_ywz) ) double_3 ywz;
    __declspec( property( get = get_ywz, put = set_ywz) ) double_3 gab;

    double_3 get_ywz() const __GPU {
        return double_3(_M_y, _M_w, _M_z);
    }

    void set_ywz(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_zxy, put = set_zxy) ) double_3 zxy;
    __declspec( property( get = get_zxy, put = set_zxy) ) double_3 brg;

    double_3 get_zxy() const __GPU {
        return double_3(_M_z, _M_x, _M_y);
    }

    void set_zxy(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_zxw, put = set_zxw) ) double_3 zxw;
    __declspec( property( get = get_zxw, put = set_zxw) ) double_3 bra;

    double_3 get_zxw() const __GPU {
        return double_3(_M_z, _M_x, _M_w);
    }

    void set_zxw(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zyx, put = set_zyx) ) double_3 zyx;
    __declspec( property( get = get_zyx, put = set_zyx) ) double_3 bgr;

    double_3 get_zyx() const __GPU {
        return double_3(_M_z, _M_y, _M_x);
    }

    void set_zyx(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zyw, put = set_zyw) ) double_3 zyw;
    __declspec( property( get = get_zyw, put = set_zyw) ) double_3 bga;

    double_3 get_zyw() const __GPU {
        return double_3(_M_z, _M_y, _M_w);
    }

    void set_zyw(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
    }

    __declspec( property( get = get_zwx, put = set_zwx) ) double_3 zwx;
    __declspec( property( get = get_zwx, put = set_zwx) ) double_3 bar;

    double_3 get_zwx() const __GPU {
        return double_3(_M_z, _M_w, _M_x);
    }

    void set_zwx(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_zwy, put = set_zwy) ) double_3 zwy;
    __declspec( property( get = get_zwy, put = set_zwy) ) double_3 bag;

    double_3 get_zwy() const __GPU {
        return double_3(_M_z, _M_w, _M_y);
    }

    void set_zwy(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxy, put = set_wxy) ) double_3 wxy;
    __declspec( property( get = get_wxy, put = set_wxy) ) double_3 arg;

    double_3 get_wxy() const __GPU {
        return double_3(_M_w, _M_x, _M_y);
    }

    void set_wxy(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_wxz, put = set_wxz) ) double_3 wxz;
    __declspec( property( get = get_wxz, put = set_wxz) ) double_3 arb;

    double_3 get_wxz() const __GPU {
        return double_3(_M_w, _M_x, _M_z);
    }

    void set_wxz(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wyx, put = set_wyx) ) double_3 wyx;
    __declspec( property( get = get_wyx, put = set_wyx) ) double_3 agr;

    double_3 get_wyx() const __GPU {
        return double_3(_M_w, _M_y, _M_x);
    }

    void set_wyx(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wyz, put = set_wyz) ) double_3 wyz;
    __declspec( property( get = get_wyz, put = set_wyz) ) double_3 agb;

    double_3 get_wyz() const __GPU {
        return double_3(_M_w, _M_y, _M_z);
    }

    void set_wyz(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
    }

    __declspec( property( get = get_wzx, put = set_wzx) ) double_3 wzx;
    __declspec( property( get = get_wzx, put = set_wzx) ) double_3 abr;

    double_3 get_wzx() const __GPU {
        return double_3(_M_w, _M_z, _M_x);
    }

    void set_wzx(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
    }

    __declspec( property( get = get_wzy, put = set_wzy) ) double_3 wzy;
    __declspec( property( get = get_wzy, put = set_wzy) ) double_3 abg;

    double_3 get_wzy() const __GPU {
        return double_3(_M_w, _M_z, _M_y);
    }

    void set_wzy(const double_3& _Value) __GPU
    {
        double_3 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
    }

    __declspec( property( get = get_xyzw, put = set_xyzw) ) double_4 xyzw;
    __declspec( property( get = get_xyzw, put = set_xyzw) ) double_4 rgba;

    double_4 get_xyzw() const __GPU {
        return double_4(_M_x, _M_y, _M_z, _M_w);
    }

    void set_xyzw(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xywz, put = set_xywz) ) double_4 xywz;
    __declspec( property( get = get_xywz, put = set_xywz) ) double_4 rgab;

    double_4 get_xywz() const __GPU {
        return double_4(_M_x, _M_y, _M_w, _M_z);
    }

    void set_xywz(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_x = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xzyw, put = set_xzyw) ) double_4 xzyw;
    __declspec( property( get = get_xzyw, put = set_xzyw) ) double_4 rbga;

    double_4 get_xzyw() const __GPU {
        return double_4(_M_x, _M_z, _M_y, _M_w);
    }

    void set_xzyw(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_xzwy, put = set_xzwy) ) double_4 xzwy;
    __declspec( property( get = get_xzwy, put = set_xzwy) ) double_4 rbag;

    double_4 get_xzwy() const __GPU {
        return double_4(_M_x, _M_z, _M_w, _M_y);
    }

    void set_xzwy(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_x = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_xwyz, put = set_xwyz) ) double_4 xwyz;
    __declspec( property( get = get_xwyz, put = set_xwyz) ) double_4 ragb;

    double_4 get_xwyz() const __GPU {
        return double_4(_M_x, _M_w, _M_y, _M_z);
    }

    void set_xwyz(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_xwzy, put = set_xwzy) ) double_4 xwzy;
    __declspec( property( get = get_xwzy, put = set_xwzy) ) double_4 rabg;

    double_4 get_xwzy() const __GPU {
        return double_4(_M_x, _M_w, _M_z, _M_y);
    }

    void set_xwzy(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_x = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_yxzw, put = set_yxzw) ) double_4 yxzw;
    __declspec( property( get = get_yxzw, put = set_yxzw) ) double_4 grba;

    double_4 get_yxzw() const __GPU {
        return double_4(_M_y, _M_x, _M_z, _M_w);
    }

    void set_yxzw(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yxwz, put = set_yxwz) ) double_4 yxwz;
    __declspec( property( get = get_yxwz, put = set_yxwz) ) double_4 grab;

    double_4 get_yxwz() const __GPU {
        return double_4(_M_y, _M_x, _M_w, _M_z);
    }

    void set_yxwz(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_y = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_yzxw, put = set_yzxw) ) double_4 yzxw;
    __declspec( property( get = get_yzxw, put = set_yzxw) ) double_4 gbra;

    double_4 get_yzxw() const __GPU {
        return double_4(_M_y, _M_z, _M_x, _M_w);
    }

    void set_yzxw(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_yzwx, put = set_yzwx) ) double_4 yzwx;
    __declspec( property( get = get_yzwx, put = set_yzwx) ) double_4 gbar;

    double_4 get_yzwx() const __GPU {
        return double_4(_M_y, _M_z, _M_w, _M_x);
    }

    void set_yzwx(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_y = _Val.x;
        _M_z = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_ywxz, put = set_ywxz) ) double_4 ywxz;
    __declspec( property( get = get_ywxz, put = set_ywxz) ) double_4 garb;

    double_4 get_ywxz() const __GPU {
        return double_4(_M_y, _M_w, _M_x, _M_z);
    }

    void set_ywxz(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_ywzx, put = set_ywzx) ) double_4 ywzx;
    __declspec( property( get = get_ywzx, put = set_ywzx) ) double_4 gabr;

    double_4 get_ywzx() const __GPU {
        return double_4(_M_y, _M_w, _M_z, _M_x);
    }

    void set_ywzx(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_y = _Val.x;
        _M_w = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zxyw, put = set_zxyw) ) double_4 zxyw;
    __declspec( property( get = get_zxyw, put = set_zxyw) ) double_4 brga;

    double_4 get_zxyw() const __GPU {
        return double_4(_M_z, _M_x, _M_y, _M_w);
    }

    void set_zxyw(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zxwy, put = set_zxwy) ) double_4 zxwy;
    __declspec( property( get = get_zxwy, put = set_zxwy) ) double_4 brag;

    double_4 get_zxwy() const __GPU {
        return double_4(_M_z, _M_x, _M_w, _M_y);
    }

    void set_zxwy(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_z = _Val.x;
        _M_x = _Val.y;
        _M_w = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zyxw, put = set_zyxw) ) double_4 zyxw;
    __declspec( property( get = get_zyxw, put = set_zyxw) ) double_4 bgra;

    double_4 get_zyxw() const __GPU {
        return double_4(_M_z, _M_y, _M_x, _M_w);
    }

    void set_zyxw(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_w = _Val.w;
    }

    __declspec( property( get = get_zywx, put = set_zywx) ) double_4 zywx;
    __declspec( property( get = get_zywx, put = set_zywx) ) double_4 bgar;

    double_4 get_zywx() const __GPU {
        return double_4(_M_z, _M_y, _M_w, _M_x);
    }

    void set_zywx(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_z = _Val.x;
        _M_y = _Val.y;
        _M_w = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_zwxy, put = set_zwxy) ) double_4 zwxy;
    __declspec( property( get = get_zwxy, put = set_zwxy) ) double_4 barg;

    double_4 get_zwxy() const __GPU {
        return double_4(_M_z, _M_w, _M_x, _M_y);
    }

    void set_zwxy(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_zwyx, put = set_zwyx) ) double_4 zwyx;
    __declspec( property( get = get_zwyx, put = set_zwyx) ) double_4 bagr;

    double_4 get_zwyx() const __GPU {
        return double_4(_M_z, _M_w, _M_y, _M_x);
    }

    void set_zwyx(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_z = _Val.x;
        _M_w = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wxyz, put = set_wxyz) ) double_4 wxyz;
    __declspec( property( get = get_wxyz, put = set_wxyz) ) double_4 argb;

    double_4 get_wxyz() const __GPU {
        return double_4(_M_w, _M_x, _M_y, _M_z);
    }

    void set_wxyz(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_y = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wxzy, put = set_wxzy) ) double_4 wxzy;
    __declspec( property( get = get_wxzy, put = set_wxzy) ) double_4 arbg;

    double_4 get_wxzy() const __GPU {
        return double_4(_M_w, _M_x, _M_z, _M_y);
    }

    void set_wxzy(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_w = _Val.x;
        _M_x = _Val.y;
        _M_z = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wyxz, put = set_wyxz) ) double_4 wyxz;
    __declspec( property( get = get_wyxz, put = set_wyxz) ) double_4 agrb;

    double_4 get_wyxz() const __GPU {
        return double_4(_M_w, _M_y, _M_x, _M_z);
    }

    void set_wyxz(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_x = _Val.z;
        _M_z = _Val.w;
    }

    __declspec( property( get = get_wyzx, put = set_wyzx) ) double_4 wyzx;
    __declspec( property( get = get_wyzx, put = set_wyzx) ) double_4 agbr;

    double_4 get_wyzx() const __GPU {
        return double_4(_M_w, _M_y, _M_z, _M_x);
    }

    void set_wyzx(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_w = _Val.x;
        _M_y = _Val.y;
        _M_z = _Val.z;
        _M_x = _Val.w;
    }

    __declspec( property( get = get_wzxy, put = set_wzxy) ) double_4 wzxy;
    __declspec( property( get = get_wzxy, put = set_wzxy) ) double_4 abrg;

    double_4 get_wzxy() const __GPU {
        return double_4(_M_w, _M_z, _M_x, _M_y);
    }

    void set_wzxy(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_x = _Val.z;
        _M_y = _Val.w;
    }

    __declspec( property( get = get_wzyx, put = set_wzyx) ) double_4 wzyx;
    __declspec( property( get = get_wzyx, put = set_wzyx) ) double_4 abgr;

    double_4 get_wzyx() const __GPU {
        return double_4(_M_w, _M_z, _M_y, _M_x);
    }

    void set_wzyx(const double_4& _Value) __GPU
    {
        double_4 _Val = _Value;
        _M_w = _Val.x;
        _M_z = _Val.y;
        _M_y = _Val.z;
        _M_x = _Val.w;
    }

};


uint_2::uint_2(const int_2& _Other) __GPU
{
    int_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

uint_2::uint_2(const float_2& _Other) __GPU
{
    float_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

uint_2::uint_2(const unorm_2& _Other) __GPU
{
    unorm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

uint_2::uint_2(const norm_2& _Other) __GPU
{
    norm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

uint_2::uint_2(const double_2& _Other) __GPU
{
    double_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}


uint_3::uint_3(const int_3& _Other) __GPU
{
    int_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

uint_3::uint_3(const float_3& _Other) __GPU
{
    float_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

uint_3::uint_3(const unorm_3& _Other) __GPU
{
    unorm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

uint_3::uint_3(const norm_3& _Other) __GPU
{
    norm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

uint_3::uint_3(const double_3& _Other) __GPU
{
    double_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}


uint_4::uint_4(const int_4& _Other) __GPU
{
    int_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

uint_4::uint_4(const float_4& _Other) __GPU
{
    float_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

uint_4::uint_4(const unorm_4& _Other) __GPU
{
    unorm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

uint_4::uint_4(const norm_4& _Other) __GPU
{
    norm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

uint_4::uint_4(const double_4& _Other) __GPU
{
    double_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}


int_2::int_2(const uint_2& _Other) __GPU
{
    uint_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

int_2::int_2(const float_2& _Other) __GPU
{
    float_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

int_2::int_2(const unorm_2& _Other) __GPU
{
    unorm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

int_2::int_2(const norm_2& _Other) __GPU
{
    norm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

int_2::int_2(const double_2& _Other) __GPU
{
    double_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}


int_3::int_3(const uint_3& _Other) __GPU
{
    uint_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

int_3::int_3(const float_3& _Other) __GPU
{
    float_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

int_3::int_3(const unorm_3& _Other) __GPU
{
    unorm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

int_3::int_3(const norm_3& _Other) __GPU
{
    norm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

int_3::int_3(const double_3& _Other) __GPU
{
    double_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}


int_4::int_4(const uint_4& _Other) __GPU
{
    uint_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

int_4::int_4(const float_4& _Other) __GPU
{
    float_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

int_4::int_4(const unorm_4& _Other) __GPU
{
    unorm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

int_4::int_4(const norm_4& _Other) __GPU
{
    norm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

int_4::int_4(const double_4& _Other) __GPU
{
    double_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}


float_2::float_2(const uint_2& _Other) __GPU
{
    uint_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

float_2::float_2(const int_2& _Other) __GPU
{
    int_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

float_2::float_2(const unorm_2& _Other) __GPU
{
    unorm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

float_2::float_2(const norm_2& _Other) __GPU
{
    norm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

float_2::float_2(const double_2& _Other) __GPU
{
    double_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}


float_3::float_3(const uint_3& _Other) __GPU
{
    uint_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

float_3::float_3(const int_3& _Other) __GPU
{
    int_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

float_3::float_3(const unorm_3& _Other) __GPU
{
    unorm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

float_3::float_3(const norm_3& _Other) __GPU
{
    norm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

float_3::float_3(const double_3& _Other) __GPU
{
    double_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}


float_4::float_4(const uint_4& _Other) __GPU
{
    uint_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

float_4::float_4(const int_4& _Other) __GPU
{
    int_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

float_4::float_4(const unorm_4& _Other) __GPU
{
    unorm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

float_4::float_4(const norm_4& _Other) __GPU
{
    norm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

float_4::float_4(const double_4& _Other) __GPU
{
    double_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}


unorm_2::unorm_2(const uint_2& _Other) __GPU
{
    uint_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

unorm_2::unorm_2(const int_2& _Other) __GPU
{
    int_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

unorm_2::unorm_2(const float_2& _Other) __GPU
{
    float_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

unorm_2::unorm_2(const norm_2& _Other) __GPU
{
    norm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

unorm_2::unorm_2(const double_2& _Other) __GPU
{
    double_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}


unorm_3::unorm_3(const uint_3& _Other) __GPU
{
    uint_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

unorm_3::unorm_3(const int_3& _Other) __GPU
{
    int_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

unorm_3::unorm_3(const float_3& _Other) __GPU
{
    float_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

unorm_3::unorm_3(const norm_3& _Other) __GPU
{
    norm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

unorm_3::unorm_3(const double_3& _Other) __GPU
{
    double_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}


unorm_4::unorm_4(const uint_4& _Other) __GPU
{
    uint_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

unorm_4::unorm_4(const int_4& _Other) __GPU
{
    int_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

unorm_4::unorm_4(const float_4& _Other) __GPU
{
    float_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

unorm_4::unorm_4(const norm_4& _Other) __GPU
{
    norm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

unorm_4::unorm_4(const double_4& _Other) __GPU
{
    double_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}


norm_2::norm_2(const uint_2& _Other) __GPU
{
    uint_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

norm_2::norm_2(const int_2& _Other) __GPU
{
    int_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

norm_2::norm_2(const float_2& _Other) __GPU
{
    float_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

norm_2::norm_2(const unorm_2& _Other) __GPU
{
    unorm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

norm_2::norm_2(const double_2& _Other) __GPU
{
    double_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}


norm_3::norm_3(const uint_3& _Other) __GPU
{
    uint_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

norm_3::norm_3(const int_3& _Other) __GPU
{
    int_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

norm_3::norm_3(const float_3& _Other) __GPU
{
    float_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

norm_3::norm_3(const unorm_3& _Other) __GPU
{
    unorm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

norm_3::norm_3(const double_3& _Other) __GPU
{
    double_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}


norm_4::norm_4(const uint_4& _Other) __GPU
{
    uint_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

norm_4::norm_4(const int_4& _Other) __GPU
{
    int_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

norm_4::norm_4(const float_4& _Other) __GPU
{
    float_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

norm_4::norm_4(const unorm_4& _Other) __GPU
{
    unorm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

norm_4::norm_4(const double_4& _Other) __GPU
{
    double_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}


double_2::double_2(const uint_2& _Other) __GPU
{
    uint_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

double_2::double_2(const int_2& _Other) __GPU
{
    int_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

double_2::double_2(const float_2& _Other) __GPU
{
    float_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

double_2::double_2(const unorm_2& _Other) __GPU
{
    unorm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}

double_2::double_2(const norm_2& _Other) __GPU
{
    norm_2 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
}


double_3::double_3(const uint_3& _Other) __GPU
{
    uint_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

double_3::double_3(const int_3& _Other) __GPU
{
    int_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

double_3::double_3(const float_3& _Other) __GPU
{
    float_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

double_3::double_3(const unorm_3& _Other) __GPU
{
    unorm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}

double_3::double_3(const norm_3& _Other) __GPU
{
    norm_3 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
}


double_4::double_4(const uint_4& _Other) __GPU
{
    uint_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

double_4::double_4(const int_4& _Other) __GPU
{
    int_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

double_4::double_4(const float_4& _Other) __GPU
{
    float_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

double_4::double_4(const unorm_4& _Other) __GPU
{
    unorm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}

double_4::double_4(const norm_4& _Other) __GPU
{
    norm_4 _Value = _Other;
    _M_x = static_cast<value_type>(_Value.x);
    _M_y = static_cast<value_type>(_Value.y);
    _M_z = static_cast<value_type>(_Value.z);
    _M_w = static_cast<value_type>(_Value.w);
}



inline uint_2 operator+(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x + _Value2.x, _Value1.y + _Value2.y);
}

inline uint_2 operator-(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x - _Value2.x, _Value1.y - _Value2.y);
}

inline uint_2 operator*(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x* _Value2.x, _Value1.y* _Value2.y);
}

inline uint_2 operator/(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x / _Value2.x, _Value1.y / _Value2.y);
}

inline bool operator==(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y;
}

inline bool operator!=(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y;
}

inline uint_2 operator%(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x % _Value2.x, _Value1.y % _Value2.y);
}

inline uint_2 operator^(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x ^ _Value2.x, _Value1.y ^ _Value2.y);
}

inline uint_2 operator|(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x | _Value2.x, _Value1.y | _Value2.y);
}

inline uint_2 operator&(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x& _Value2.x, _Value1.y& _Value2.y);
}

inline uint_2 operator<<(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x << _Value2.x, _Value1.y << _Value2.y);
}

inline uint_2 operator>>(const uint_2& _Lhs, const uint_2& _Rhs) __GPU
{
    uint_2 _Value1 = _Lhs;
    uint_2 _Value2 = _Rhs;
    return uint_2(_Value1.x >> _Value2.x, _Value1.y >> _Value2.y);
}

inline uint_3 operator+(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z);
}

inline uint_3 operator-(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z);
}

inline uint_3 operator*(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z);
}

inline uint_3 operator/(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z);
}

inline bool operator==(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z;
}

inline bool operator!=(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z;
}

inline uint_3 operator%(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x % _Value2.x, _Value1.y % _Value2.y, _Value1.z % _Value2.z);
}

inline uint_3 operator^(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x ^ _Value2.x, _Value1.y ^ _Value2.y, _Value1.z ^ _Value2.z);
}

inline uint_3 operator|(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x | _Value2.x, _Value1.y | _Value2.y, _Value1.z | _Value2.z);
}

inline uint_3 operator&(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x& _Value2.x, _Value1.y& _Value2.y, _Value1.z& _Value2.z);
}

inline uint_3 operator<<(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x << _Value2.x, _Value1.y << _Value2.y, _Value1.z << _Value2.z);
}

inline uint_3 operator>>(const uint_3& _Lhs, const uint_3& _Rhs) __GPU
{
    uint_3 _Value1 = _Lhs;
    uint_3 _Value2 = _Rhs;
    return uint_3(_Value1.x >> _Value2.x, _Value1.y >> _Value2.y, _Value1.z >> _Value2.z);
}

inline uint_4 operator+(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z, _Value1.w + _Value2.w);
}

inline uint_4 operator-(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z, _Value1.w - _Value2.w);
}

inline uint_4 operator*(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z, _Value1.w* _Value2.w);
}

inline uint_4 operator/(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z, _Value1.w / _Value2.w);
}

inline bool operator==(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z && _Value1.w == _Value2.w;
}

inline bool operator!=(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z || _Value1.w != _Value2.w;
}

inline uint_4 operator%(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x % _Value2.x, _Value1.y % _Value2.y, _Value1.z % _Value2.z, _Value1.w % _Value2.w);
}

inline uint_4 operator^(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x ^ _Value2.x, _Value1.y ^ _Value2.y, _Value1.z ^ _Value2.z, _Value1.w ^ _Value2.w);
}

inline uint_4 operator|(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x | _Value2.x, _Value1.y | _Value2.y, _Value1.z | _Value2.z, _Value1.w | _Value2.w);
}

inline uint_4 operator&(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x& _Value2.x, _Value1.y& _Value2.y, _Value1.z& _Value2.z, _Value1.w& _Value2.w);
}

inline uint_4 operator<<(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x << _Value2.x, _Value1.y << _Value2.y, _Value1.z << _Value2.z, _Value1.w << _Value2.w);
}

inline uint_4 operator>>(const uint_4& _Lhs, const uint_4& _Rhs) __GPU
{
    uint_4 _Value1 = _Lhs;
    uint_4 _Value2 = _Rhs;
    return uint_4(_Value1.x >> _Value2.x, _Value1.y >> _Value2.y, _Value1.z >> _Value2.z, _Value1.w >> _Value2.w);
}

inline int_2 operator+(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x + _Value2.x, _Value1.y + _Value2.y);
}

inline int_2 operator-(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x - _Value2.x, _Value1.y - _Value2.y);
}

inline int_2 operator*(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x* _Value2.x, _Value1.y* _Value2.y);
}

inline int_2 operator/(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x / _Value2.x, _Value1.y / _Value2.y);
}

inline bool operator==(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y;
}

inline bool operator!=(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y;
}

inline int_2 operator%(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x % _Value2.x, _Value1.y % _Value2.y);
}

inline int_2 operator^(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x ^ _Value2.x, _Value1.y ^ _Value2.y);
}

inline int_2 operator|(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x | _Value2.x, _Value1.y | _Value2.y);
}

inline int_2 operator&(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x& _Value2.x, _Value1.y& _Value2.y);
}

inline int_2 operator<<(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x << _Value2.x, _Value1.y << _Value2.y);
}

inline int_2 operator>>(const int_2& _Lhs, const int_2& _Rhs) __GPU
{
    int_2 _Value1 = _Lhs;
    int_2 _Value2 = _Rhs;
    return int_2(_Value1.x >> _Value2.x, _Value1.y >> _Value2.y);
}

inline int_3 operator+(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z);
}

inline int_3 operator-(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z);
}

inline int_3 operator*(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z);
}

inline int_3 operator/(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z);
}

inline bool operator==(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z;
}

inline bool operator!=(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z;
}

inline int_3 operator%(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x % _Value2.x, _Value1.y % _Value2.y, _Value1.z % _Value2.z);
}

inline int_3 operator^(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x ^ _Value2.x, _Value1.y ^ _Value2.y, _Value1.z ^ _Value2.z);
}

inline int_3 operator|(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x | _Value2.x, _Value1.y | _Value2.y, _Value1.z | _Value2.z);
}

inline int_3 operator&(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x& _Value2.x, _Value1.y& _Value2.y, _Value1.z& _Value2.z);
}

inline int_3 operator<<(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x << _Value2.x, _Value1.y << _Value2.y, _Value1.z << _Value2.z);
}

inline int_3 operator>>(const int_3& _Lhs, const int_3& _Rhs) __GPU
{
    int_3 _Value1 = _Lhs;
    int_3 _Value2 = _Rhs;
    return int_3(_Value1.x >> _Value2.x, _Value1.y >> _Value2.y, _Value1.z >> _Value2.z);
}

inline int_4 operator+(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z, _Value1.w + _Value2.w);
}

inline int_4 operator-(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z, _Value1.w - _Value2.w);
}

inline int_4 operator*(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z, _Value1.w* _Value2.w);
}

inline int_4 operator/(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z, _Value1.w / _Value2.w);
}

inline bool operator==(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z && _Value1.w == _Value2.w;
}

inline bool operator!=(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z || _Value1.w != _Value2.w;
}

inline int_4 operator%(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x % _Value2.x, _Value1.y % _Value2.y, _Value1.z % _Value2.z, _Value1.w % _Value2.w);
}

inline int_4 operator^(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x ^ _Value2.x, _Value1.y ^ _Value2.y, _Value1.z ^ _Value2.z, _Value1.w ^ _Value2.w);
}

inline int_4 operator|(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x | _Value2.x, _Value1.y | _Value2.y, _Value1.z | _Value2.z, _Value1.w | _Value2.w);
}

inline int_4 operator&(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x& _Value2.x, _Value1.y& _Value2.y, _Value1.z& _Value2.z, _Value1.w& _Value2.w);
}

inline int_4 operator<<(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x << _Value2.x, _Value1.y << _Value2.y, _Value1.z << _Value2.z, _Value1.w << _Value2.w);
}

inline int_4 operator>>(const int_4& _Lhs, const int_4& _Rhs) __GPU
{
    int_4 _Value1 = _Lhs;
    int_4 _Value2 = _Rhs;
    return int_4(_Value1.x >> _Value2.x, _Value1.y >> _Value2.y, _Value1.z >> _Value2.z, _Value1.w >> _Value2.w);
}

inline float_2 operator+(const float_2& _Lhs, const float_2& _Rhs) __GPU
{
    float_2 _Value1 = _Lhs;
    float_2 _Value2 = _Rhs;
    return float_2(_Value1.x + _Value2.x, _Value1.y + _Value2.y);
}

inline float_2 operator-(const float_2& _Lhs, const float_2& _Rhs) __GPU
{
    float_2 _Value1 = _Lhs;
    float_2 _Value2 = _Rhs;
    return float_2(_Value1.x - _Value2.x, _Value1.y - _Value2.y);
}

inline float_2 operator*(const float_2& _Lhs, const float_2& _Rhs) __GPU
{
    float_2 _Value1 = _Lhs;
    float_2 _Value2 = _Rhs;
    return float_2(_Value1.x* _Value2.x, _Value1.y* _Value2.y);
}

inline float_2 operator/(const float_2& _Lhs, const float_2& _Rhs) __GPU
{
    float_2 _Value1 = _Lhs;
    float_2 _Value2 = _Rhs;
    return float_2(_Value1.x / _Value2.x, _Value1.y / _Value2.y);
}

inline bool operator==(const float_2& _Lhs, const float_2& _Rhs) __GPU
{
    float_2 _Value1 = _Lhs;
    float_2 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y;
}

inline bool operator!=(const float_2& _Lhs, const float_2& _Rhs) __GPU
{
    float_2 _Value1 = _Lhs;
    float_2 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y;
}

inline float_3 operator+(const float_3& _Lhs, const float_3& _Rhs) __GPU
{
    float_3 _Value1 = _Lhs;
    float_3 _Value2 = _Rhs;
    return float_3(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z);
}

inline float_3 operator-(const float_3& _Lhs, const float_3& _Rhs) __GPU
{
    float_3 _Value1 = _Lhs;
    float_3 _Value2 = _Rhs;
    return float_3(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z);
}

inline float_3 operator*(const float_3& _Lhs, const float_3& _Rhs) __GPU
{
    float_3 _Value1 = _Lhs;
    float_3 _Value2 = _Rhs;
    return float_3(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z);
}

inline float_3 operator/(const float_3& _Lhs, const float_3& _Rhs) __GPU
{
    float_3 _Value1 = _Lhs;
    float_3 _Value2 = _Rhs;
    return float_3(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z);
}

inline bool operator==(const float_3& _Lhs, const float_3& _Rhs) __GPU
{
    float_3 _Value1 = _Lhs;
    float_3 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z;
}

inline bool operator!=(const float_3& _Lhs, const float_3& _Rhs) __GPU
{
    float_3 _Value1 = _Lhs;
    float_3 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z;
}

inline float_4 operator+(const float_4& _Lhs, const float_4& _Rhs) __GPU
{
    float_4 _Value1 = _Lhs;
    float_4 _Value2 = _Rhs;
    return float_4(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z, _Value1.w + _Value2.w);
}

inline float_4 operator-(const float_4& _Lhs, const float_4& _Rhs) __GPU
{
    float_4 _Value1 = _Lhs;
    float_4 _Value2 = _Rhs;
    return float_4(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z, _Value1.w - _Value2.w);
}

inline float_4 operator*(const float_4& _Lhs, const float_4& _Rhs) __GPU
{
    float_4 _Value1 = _Lhs;
    float_4 _Value2 = _Rhs;
    return float_4(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z, _Value1.w* _Value2.w);
}

inline float_4 operator/(const float_4& _Lhs, const float_4& _Rhs) __GPU
{
    float_4 _Value1 = _Lhs;
    float_4 _Value2 = _Rhs;
    return float_4(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z, _Value1.w / _Value2.w);
}

inline bool operator==(const float_4& _Lhs, const float_4& _Rhs) __GPU
{
    float_4 _Value1 = _Lhs;
    float_4 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z && _Value1.w == _Value2.w;
}

inline bool operator!=(const float_4& _Lhs, const float_4& _Rhs) __GPU
{
    float_4 _Value1 = _Lhs;
    float_4 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z || _Value1.w != _Value2.w;
}

inline unorm_2 operator+(const unorm_2& _Lhs, const unorm_2& _Rhs) __GPU
{
    unorm_2 _Value1 = _Lhs;
    unorm_2 _Value2 = _Rhs;
    return unorm_2(_Value1.x + _Value2.x, _Value1.y + _Value2.y);
}

inline unorm_2 operator-(const unorm_2& _Lhs, const unorm_2& _Rhs) __GPU
{
    unorm_2 _Value1 = _Lhs;
    unorm_2 _Value2 = _Rhs;
    return unorm_2(_Value1.x - _Value2.x, _Value1.y - _Value2.y);
}

inline unorm_2 operator*(const unorm_2& _Lhs, const unorm_2& _Rhs) __GPU
{
    unorm_2 _Value1 = _Lhs;
    unorm_2 _Value2 = _Rhs;
    return unorm_2(_Value1.x* _Value2.x, _Value1.y* _Value2.y);
}

inline unorm_2 operator/(const unorm_2& _Lhs, const unorm_2& _Rhs) __GPU
{
    unorm_2 _Value1 = _Lhs;
    unorm_2 _Value2 = _Rhs;
    return unorm_2(_Value1.x / _Value2.x, _Value1.y / _Value2.y);
}

inline bool operator==(const unorm_2& _Lhs, const unorm_2& _Rhs) __GPU
{
    unorm_2 _Value1 = _Lhs;
    unorm_2 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y;
}

inline bool operator!=(const unorm_2& _Lhs, const unorm_2& _Rhs) __GPU
{
    unorm_2 _Value1 = _Lhs;
    unorm_2 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y;
}

inline unorm_3 operator+(const unorm_3& _Lhs, const unorm_3& _Rhs) __GPU
{
    unorm_3 _Value1 = _Lhs;
    unorm_3 _Value2 = _Rhs;
    return unorm_3(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z);
}

inline unorm_3 operator-(const unorm_3& _Lhs, const unorm_3& _Rhs) __GPU
{
    unorm_3 _Value1 = _Lhs;
    unorm_3 _Value2 = _Rhs;
    return unorm_3(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z);
}

inline unorm_3 operator*(const unorm_3& _Lhs, const unorm_3& _Rhs) __GPU
{
    unorm_3 _Value1 = _Lhs;
    unorm_3 _Value2 = _Rhs;
    return unorm_3(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z);
}

inline unorm_3 operator/(const unorm_3& _Lhs, const unorm_3& _Rhs) __GPU
{
    unorm_3 _Value1 = _Lhs;
    unorm_3 _Value2 = _Rhs;
    return unorm_3(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z);
}

inline bool operator==(const unorm_3& _Lhs, const unorm_3& _Rhs) __GPU
{
    unorm_3 _Value1 = _Lhs;
    unorm_3 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z;
}

inline bool operator!=(const unorm_3& _Lhs, const unorm_3& _Rhs) __GPU
{
    unorm_3 _Value1 = _Lhs;
    unorm_3 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z;
}

inline unorm_4 operator+(const unorm_4& _Lhs, const unorm_4& _Rhs) __GPU
{
    unorm_4 _Value1 = _Lhs;
    unorm_4 _Value2 = _Rhs;
    return unorm_4(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z, _Value1.w + _Value2.w);
}

inline unorm_4 operator-(const unorm_4& _Lhs, const unorm_4& _Rhs) __GPU
{
    unorm_4 _Value1 = _Lhs;
    unorm_4 _Value2 = _Rhs;
    return unorm_4(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z, _Value1.w - _Value2.w);
}

inline unorm_4 operator*(const unorm_4& _Lhs, const unorm_4& _Rhs) __GPU
{
    unorm_4 _Value1 = _Lhs;
    unorm_4 _Value2 = _Rhs;
    return unorm_4(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z, _Value1.w* _Value2.w);
}

inline unorm_4 operator/(const unorm_4& _Lhs, const unorm_4& _Rhs) __GPU
{
    unorm_4 _Value1 = _Lhs;
    unorm_4 _Value2 = _Rhs;
    return unorm_4(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z, _Value1.w / _Value2.w);
}

inline bool operator==(const unorm_4& _Lhs, const unorm_4& _Rhs) __GPU
{
    unorm_4 _Value1 = _Lhs;
    unorm_4 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z && _Value1.w == _Value2.w;
}

inline bool operator!=(const unorm_4& _Lhs, const unorm_4& _Rhs) __GPU
{
    unorm_4 _Value1 = _Lhs;
    unorm_4 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z || _Value1.w != _Value2.w;
}

inline norm_2 operator+(const norm_2& _Lhs, const norm_2& _Rhs) __GPU
{
    norm_2 _Value1 = _Lhs;
    norm_2 _Value2 = _Rhs;
    return norm_2(_Value1.x + _Value2.x, _Value1.y + _Value2.y);
}

inline norm_2 operator-(const norm_2& _Lhs, const norm_2& _Rhs) __GPU
{
    norm_2 _Value1 = _Lhs;
    norm_2 _Value2 = _Rhs;
    return norm_2(_Value1.x - _Value2.x, _Value1.y - _Value2.y);
}

inline norm_2 operator*(const norm_2& _Lhs, const norm_2& _Rhs) __GPU
{
    norm_2 _Value1 = _Lhs;
    norm_2 _Value2 = _Rhs;
    return norm_2(_Value1.x* _Value2.x, _Value1.y* _Value2.y);
}

inline norm_2 operator/(const norm_2& _Lhs, const norm_2& _Rhs) __GPU
{
    norm_2 _Value1 = _Lhs;
    norm_2 _Value2 = _Rhs;
    return norm_2(_Value1.x / _Value2.x, _Value1.y / _Value2.y);
}

inline bool operator==(const norm_2& _Lhs, const norm_2& _Rhs) __GPU
{
    norm_2 _Value1 = _Lhs;
    norm_2 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y;
}

inline bool operator!=(const norm_2& _Lhs, const norm_2& _Rhs) __GPU
{
    norm_2 _Value1 = _Lhs;
    norm_2 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y;
}

inline norm_3 operator+(const norm_3& _Lhs, const norm_3& _Rhs) __GPU
{
    norm_3 _Value1 = _Lhs;
    norm_3 _Value2 = _Rhs;
    return norm_3(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z);
}

inline norm_3 operator-(const norm_3& _Lhs, const norm_3& _Rhs) __GPU
{
    norm_3 _Value1 = _Lhs;
    norm_3 _Value2 = _Rhs;
    return norm_3(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z);
}

inline norm_3 operator*(const norm_3& _Lhs, const norm_3& _Rhs) __GPU
{
    norm_3 _Value1 = _Lhs;
    norm_3 _Value2 = _Rhs;
    return norm_3(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z);
}

inline norm_3 operator/(const norm_3& _Lhs, const norm_3& _Rhs) __GPU
{
    norm_3 _Value1 = _Lhs;
    norm_3 _Value2 = _Rhs;
    return norm_3(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z);
}

inline bool operator==(const norm_3& _Lhs, const norm_3& _Rhs) __GPU
{
    norm_3 _Value1 = _Lhs;
    norm_3 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z;
}

inline bool operator!=(const norm_3& _Lhs, const norm_3& _Rhs) __GPU
{
    norm_3 _Value1 = _Lhs;
    norm_3 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z;
}

inline norm_4 operator+(const norm_4& _Lhs, const norm_4& _Rhs) __GPU
{
    norm_4 _Value1 = _Lhs;
    norm_4 _Value2 = _Rhs;
    return norm_4(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z, _Value1.w + _Value2.w);
}

inline norm_4 operator-(const norm_4& _Lhs, const norm_4& _Rhs) __GPU
{
    norm_4 _Value1 = _Lhs;
    norm_4 _Value2 = _Rhs;
    return norm_4(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z, _Value1.w - _Value2.w);
}

inline norm_4 operator*(const norm_4& _Lhs, const norm_4& _Rhs) __GPU
{
    norm_4 _Value1 = _Lhs;
    norm_4 _Value2 = _Rhs;
    return norm_4(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z, _Value1.w* _Value2.w);
}

inline norm_4 operator/(const norm_4& _Lhs, const norm_4& _Rhs) __GPU
{
    norm_4 _Value1 = _Lhs;
    norm_4 _Value2 = _Rhs;
    return norm_4(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z, _Value1.w / _Value2.w);
}

inline bool operator==(const norm_4& _Lhs, const norm_4& _Rhs) __GPU
{
    norm_4 _Value1 = _Lhs;
    norm_4 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z && _Value1.w == _Value2.w;
}

inline bool operator!=(const norm_4& _Lhs, const norm_4& _Rhs) __GPU
{
    norm_4 _Value1 = _Lhs;
    norm_4 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z || _Value1.w != _Value2.w;
}

inline double_2 operator+(const double_2& _Lhs, const double_2& _Rhs) __GPU
{
    double_2 _Value1 = _Lhs;
    double_2 _Value2 = _Rhs;
    return double_2(_Value1.x + _Value2.x, _Value1.y + _Value2.y);
}

inline double_2 operator-(const double_2& _Lhs, const double_2& _Rhs) __GPU
{
    double_2 _Value1 = _Lhs;
    double_2 _Value2 = _Rhs;
    return double_2(_Value1.x - _Value2.x, _Value1.y - _Value2.y);
}

inline double_2 operator*(const double_2& _Lhs, const double_2& _Rhs) __GPU
{
    double_2 _Value1 = _Lhs;
    double_2 _Value2 = _Rhs;
    return double_2(_Value1.x* _Value2.x, _Value1.y* _Value2.y);
}

inline double_2 operator/(const double_2& _Lhs, const double_2& _Rhs) __GPU
{
    double_2 _Value1 = _Lhs;
    double_2 _Value2 = _Rhs;
    return double_2(_Value1.x / _Value2.x, _Value1.y / _Value2.y);
}

inline bool operator==(const double_2& _Lhs, const double_2& _Rhs) __GPU
{
    double_2 _Value1 = _Lhs;
    double_2 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y;
}

inline bool operator!=(const double_2& _Lhs, const double_2& _Rhs) __GPU
{
    double_2 _Value1 = _Lhs;
    double_2 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y;
}

inline double_3 operator+(const double_3& _Lhs, const double_3& _Rhs) __GPU
{
    double_3 _Value1 = _Lhs;
    double_3 _Value2 = _Rhs;
    return double_3(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z);
}

inline double_3 operator-(const double_3& _Lhs, const double_3& _Rhs) __GPU
{
    double_3 _Value1 = _Lhs;
    double_3 _Value2 = _Rhs;
    return double_3(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z);
}

inline double_3 operator*(const double_3& _Lhs, const double_3& _Rhs) __GPU
{
    double_3 _Value1 = _Lhs;
    double_3 _Value2 = _Rhs;
    return double_3(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z);
}

inline double_3 operator/(const double_3& _Lhs, const double_3& _Rhs) __GPU
{
    double_3 _Value1 = _Lhs;
    double_3 _Value2 = _Rhs;
    return double_3(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z);
}

inline bool operator==(const double_3& _Lhs, const double_3& _Rhs) __GPU
{
    double_3 _Value1 = _Lhs;
    double_3 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z;
}

inline bool operator!=(const double_3& _Lhs, const double_3& _Rhs) __GPU
{
    double_3 _Value1 = _Lhs;
    double_3 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z;
}

inline double_4 operator+(const double_4& _Lhs, const double_4& _Rhs) __GPU
{
    double_4 _Value1 = _Lhs;
    double_4 _Value2 = _Rhs;
    return double_4(_Value1.x + _Value2.x, _Value1.y + _Value2.y, _Value1.z + _Value2.z, _Value1.w + _Value2.w);
}

inline double_4 operator-(const double_4& _Lhs, const double_4& _Rhs) __GPU
{
    double_4 _Value1 = _Lhs;
    double_4 _Value2 = _Rhs;
    return double_4(_Value1.x - _Value2.x, _Value1.y - _Value2.y, _Value1.z - _Value2.z, _Value1.w - _Value2.w);
}

inline double_4 operator*(const double_4& _Lhs, const double_4& _Rhs) __GPU
{
    double_4 _Value1 = _Lhs;
    double_4 _Value2 = _Rhs;
    return double_4(_Value1.x* _Value2.x, _Value1.y* _Value2.y, _Value1.z* _Value2.z, _Value1.w* _Value2.w);
}

inline double_4 operator/(const double_4& _Lhs, const double_4& _Rhs) __GPU
{
    double_4 _Value1 = _Lhs;
    double_4 _Value2 = _Rhs;
    return double_4(_Value1.x / _Value2.x, _Value1.y / _Value2.y, _Value1.z / _Value2.z, _Value1.w / _Value2.w);
}

inline bool operator==(const double_4& _Lhs, const double_4& _Rhs) __GPU
{
    double_4 _Value1 = _Lhs;
    double_4 _Value2 = _Rhs;
    return _Value1.x == _Value2.x && _Value1.y == _Value2.y && _Value1.z == _Value2.z && _Value1.w == _Value2.w;
}

inline bool operator!=(const double_4& _Lhs, const double_4& _Rhs) __GPU
{
    double_4 _Value1 = _Lhs;
    double_4 _Value2 = _Rhs;
    return _Value1.x != _Value2.x || _Value1.y != _Value2.y || _Value1.z != _Value2.z || _Value1.w != _Value2.w;
}

namespace direct3d
{

// typedefs
typedef graphics::uint_2 uint2;
typedef graphics::uint_3 uint3;
typedef graphics::uint_4 uint4;
typedef graphics::int_2 int2;
typedef graphics::int_3 int3;
typedef graphics::int_4 int4;
typedef graphics::float_2 float2;
typedef graphics::float_3 float3;
typedef graphics::float_4 float4;
typedef graphics::unorm_2 unorm2;
typedef graphics::unorm_3 unorm3;
typedef graphics::unorm_4 unorm4;
typedef graphics::norm_2 norm2;
typedef graphics::norm_3 norm3;
typedef graphics::norm_4 norm4;
typedef graphics::double_2 double2;
typedef graphics::double_3 double3;
typedef graphics::double_4 double4;


} // namespace direct3d

/// <summary>
///     short_vector provides metaprogramming definitions which are useful for programming short vectors generically.
/// </summary>
/// <param name="_Scalar_type">
///     The scalar type.
/// </param>
/// <param name="_Size">
///     The size of the short vector.
/// </param>
template<typename _Scalar_type, int _Size> struct short_vector
{
    short_vector()
    {
        static_assert(false, "short_vector is not supported for this scalar type (_T) and length (_N)");
    }
};

template<>
struct short_vector<unsigned int, 1>
{
    typedef unsigned int type;
};

template<>
struct short_vector<unsigned int, 2>
{
    typedef uint_2 type;
};

template<>
struct short_vector<unsigned int, 3>
{
    typedef uint_3 type;
};

template<>
struct short_vector<unsigned int, 4>
{
    typedef uint_4 type;
};

template<>
struct short_vector<int, 1>
{
    typedef int type;
};

template<>
struct short_vector<int, 2>
{
    typedef int_2 type;
};

template<>
struct short_vector<int, 3>
{
    typedef int_3 type;
};

template<>
struct short_vector<int, 4>
{
    typedef int_4 type;
};

template<>
struct short_vector<float, 1>
{
    typedef float type;
};

template<>
struct short_vector<float, 2>
{
    typedef float_2 type;
};

template<>
struct short_vector<float, 3>
{
    typedef float_3 type;
};

template<>
struct short_vector<float, 4>
{
    typedef float_4 type;
};

template<>
struct short_vector<unorm, 1>
{
    typedef unorm type;
};

template<>
struct short_vector<unorm, 2>
{
    typedef unorm_2 type;
};

template<>
struct short_vector<unorm, 3>
{
    typedef unorm_3 type;
};

template<>
struct short_vector<unorm, 4>
{
    typedef unorm_4 type;
};

template<>
struct short_vector<norm, 1>
{
    typedef norm type;
};

template<>
struct short_vector<norm, 2>
{
    typedef norm_2 type;
};

template<>
struct short_vector<norm, 3>
{
    typedef norm_3 type;
};

template<>
struct short_vector<norm, 4>
{
    typedef norm_4 type;
};

template<>
struct short_vector<double, 1>
{
    typedef double type;
};

template<>
struct short_vector<double, 2>
{
    typedef double_2 type;
};

template<>
struct short_vector<double, 3>
{
    typedef double_3 type;
};

template<>
struct short_vector<double, 4>
{
    typedef double_4 type;
};

/// <summary>
///     short_vector_traits allows retrival of the underlying vector length and scalar type of a short vector type or a scalar type
/// </summary>
/// <param name="_Type">
///     The short vector type or a scalar type.
/// </param>
template<typename _Type> struct short_vector_traits
{
    short_vector_traits()
    {
        static_assert(false, "short_vector_traits is not supported for this type (_Type)");
    }
};

template<>
struct short_vector_traits<unsigned int>
{
    typedef unsigned int value_type;
    static int const size = 1;
};

template<>
struct short_vector_traits<uint_2>
{
    typedef unsigned int value_type;
    static int const size = 2;
};

template<>
struct short_vector_traits<uint_3>
{
    typedef unsigned int value_type;
    static int const size = 3;
};

template<>
struct short_vector_traits<uint_4>
{
    typedef unsigned int value_type;
    static int const size = 4;
};

template<>
struct short_vector_traits<int>
{
    typedef int value_type;
    static int const size = 1;
};

template<>
struct short_vector_traits<int_2>
{
    typedef int value_type;
    static int const size = 2;
};

template<>
struct short_vector_traits<int_3>
{
    typedef int value_type;
    static int const size = 3;
};

template<>
struct short_vector_traits<int_4>
{
    typedef int value_type;
    static int const size = 4;
};

template<>
struct short_vector_traits<float>
{
    typedef float value_type;
    static int const size = 1;
};

template<>
struct short_vector_traits<float_2>
{
    typedef float value_type;
    static int const size = 2;
};

template<>
struct short_vector_traits<float_3>
{
    typedef float value_type;
    static int const size = 3;
};

template<>
struct short_vector_traits<float_4>
{
    typedef float value_type;
    static int const size = 4;
};

template<>
struct short_vector_traits<unorm>
{
    typedef unorm value_type;
    static int const size = 1;
};

template<>
struct short_vector_traits<unorm_2>
{
    typedef unorm value_type;
    static int const size = 2;
};

template<>
struct short_vector_traits<unorm_3>
{
    typedef unorm value_type;
    static int const size = 3;
};

template<>
struct short_vector_traits<unorm_4>
{
    typedef unorm value_type;
    static int const size = 4;
};

template<>
struct short_vector_traits<norm>
{
    typedef norm value_type;
    static int const size = 1;
};

template<>
struct short_vector_traits<norm_2>
{
    typedef norm value_type;
    static int const size = 2;
};

template<>
struct short_vector_traits<norm_3>
{
    typedef norm value_type;
    static int const size = 3;
};

template<>
struct short_vector_traits<norm_4>
{
    typedef norm value_type;
    static int const size = 4;
};

template<>
struct short_vector_traits<double>
{
    typedef double value_type;
    static int const size = 1;
};

template<>
struct short_vector_traits<double_2>
{
    typedef double value_type;
    static int const size = 2;
};

template<>
struct short_vector_traits<double_3>
{
    typedef double value_type;
    static int const size = 3;
};

template<>
struct short_vector_traits<double_4>
{
    typedef double value_type;
    static int const size = 4;
};


} // namespace graphics

} // namespace Concurrency
#pragma warning(pop)
// End of generated file
