//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

namespace hc2
{
    template<typename T>
    class Swappable {
        friend
        inline
        void swap(T& x, T& y) { Swappable<T>::swap_(x, y); }
    public:
        static
        void swap_(T& x, T& y) { x.swp_(y); }
    };
}