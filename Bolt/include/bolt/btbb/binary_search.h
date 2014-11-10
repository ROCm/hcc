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
#if !defined( BOLT_BTBB_BINARY_SEARCH_H )
#define BOLT_BTBB_BINARY_SEARCH_H

/*! \file bolt/tbb/binary_search.h
    \brief Searches the input vector for the specified value and returns true if present and false otherwise
*/


namespace bolt {
    namespace btbb {

       template<typename ForwardIterator, typename T>
       bool binary_search( ForwardIterator first, ForwardIterator last, const T & value);
       
       template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
       bool binary_search( ForwardIterator first, ForwardIterator last, const T & value, StrictWeakOrdering comp);
       
    };
};


#include <bolt/btbb/detail/binary_search.inl>

#endif