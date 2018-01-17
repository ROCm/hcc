//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <type_traits>

namespace hc2
{
    template<
        typename T,
        typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
    inline
    constexpr
    bool positive(T x)
    {
        return x > T{0};
    }

    template<
        typename T,
        typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
    inline
    constexpr
    bool zero(T x)
    {
        return x == T{0};
    }

}