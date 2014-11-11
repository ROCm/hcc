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
#if !defined( BOLT_BTBB_COPY_H )
#define BOLT_BTBB_COPY_H

/*! \file bolt/tbb/copy.h
    \brief copies the source vector to destination vector
*/


namespace bolt {
    namespace btbb {

       template<typename InputIterator, typename Size, typename OutputIterator>
       OutputIterator copy_n(InputIterator first, Size n, OutputIterator result);

    };
};


#include <bolt/btbb/detail/copy.inl>

#endif