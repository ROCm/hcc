/***************************************************************************
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/

#pragma once
#if !defined( BOLT_AMP_ITERATOR_TRAITS_H )
#define BOLT_AMP_ITERATOR_TRAITS_H

/*! \file bolt/amp/iterator/iterator_traits.h
    \brief Defines new iterator_traits structures used by the Bolt runtime to make runtime decisions on how to 
    dispatch calls to various supported backends
    \todo This is a minimal version of OpenCL iterator_traits. Needs improvement.
*/

namespace bolt {
namespace amp {

    template< typename iterator >
    struct iterator_traits
    {
        typedef typename iterator::iterator_category iterator_category;
        typedef typename iterator::value_type value_type;
        typedef typename iterator::difference_type difference_type;
        typedef typename iterator::pointer pointer;
        typedef typename iterator::reference reference;
    };

    template< class T >
    struct iterator_traits< T* >
    {
        typedef typename std::random_access_iterator_tag iterator_category;
        typedef T value_type;
        //  difference_type set to int for OpenCL backend
        typedef int difference_type;
        typedef T* pointer;
        typedef T& reference;
    };

    template< class T >
    struct iterator_traits< const T* >
    {
        typedef typename std::random_access_iterator_tag iterator_category;
        typedef T value_type;
        typedef int difference_type;
        typedef const T* pointer;
        typedef const T& reference;
    };

    struct fancy_iterator_tag : public std::random_access_iterator_tag
    {
    };

}
};

#endif