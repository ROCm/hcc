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
#if !defined( BOLT_CL_ITERATOR_TRAITS_H )
#define BOLT_CL_ITERATOR_TRAITS_H

/*! \file bolt/cl/iterator/iterator_traits.h
    \brief Defines new iterator_traits structures used by the Bolt runtime to make runtime decisions on how to
    dispatch calls to various supported backends
*/

// #include <iterator>

namespace bolt {
namespace cl {

    struct input_iterator_tag
    {
        operator std::input_iterator_tag ( ) const
        {
        }
    };

    struct output_iterator_tag
    {
        operator std::output_iterator_tag ( ) const
        {
        }
    };

    struct forward_iterator_tag: public input_iterator_tag, output_iterator_tag
    {
        operator std::forward_iterator_tag ( ) const
        {
        }
    };

    struct bidirectional_iterator_tag: public forward_iterator_tag
    {
        operator std::bidirectional_iterator_tag ( ) const
        {
        }
    };

    struct random_access_iterator_tag: public bidirectional_iterator_tag
    {
        operator std::random_access_iterator_tag ( ) const
        {
        }

        random_access_iterator_tag& operator=( const std::random_access_iterator_tag& )
        {
            return *this;
        }
    };

    struct fancy_iterator_tag: public std::random_access_iterator_tag
    {
        operator fancy_iterator_tag ( ) const
        {
        }
    };

    template< typename iterator >
    struct iterator_traits
    {
        typedef typename iterator::iterator_category memory_system;
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
        //  difference_type set to int for OpenCL backend
        typedef int difference_type;
        typedef const T* pointer;
        typedef const T& reference;
    };

    template<typename Iterator>
      struct iterator_value
    {
      typedef typename bolt::cl::iterator_traits<Iterator>::value_type type;
    }; // end iterator_value


    template<typename Iterator>
      struct iterator_pointer
    {
      typedef typename bolt::cl::iterator_traits<Iterator>::pointer type;
    }; // end iterator_pointer


    template<typename Iterator>
      struct iterator_reference
    {
      typedef typename bolt::cl::iterator_traits<Iterator>::reference type;
    }; // end iterator_reference


    template<typename Iterator>
      struct iterator_difference
    {
      typedef typename bolt::cl::iterator_traits<Iterator>::difference_type type;
    }; // end iterator_difference

//This was causing a resolving sissue. For iterator traversal
    template<typename Iterator>
      struct iterator_category
    {
      typedef typename bolt::cl::iterator_traits<Iterator>::iterator_category type;
    }; // end iterator_category

    template<typename Iterator>
      struct memory_system
    {
      typedef typename Iterator::memory_system type;
    }; // end iterator_category

    ////template< typename newTag, typename InputIterator >
    ////InputIterator retag( InputIterator )
    ////{
    ////    switch( iterator_traits< InputIterator >::iterator_category( ) )
    ////    {
    ////        case std::input_iterator_tag :
    ////            return input_iterator_tag( );
    ////        case std::output_iterator_tag :
    ////            return output_iterator_tag( );
    ////        case std::forward_iterator_tag :
    ////            return forward_iterator_tag( );
    ////        case std::bidirectional_iterator_tag :
    ////            return bidirectional_iterator_tag( );
    ////        case std::random_access_iterator_tag :
    ////            return random_access_iterator_tag( );
    ////    }

    ////    return iter;
    ////}

}
};

#endif
