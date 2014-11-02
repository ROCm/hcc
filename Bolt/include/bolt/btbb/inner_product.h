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
#if !defined( BOLT_BTBB_INNER_PRODUCT_H )
#define BOLT_BTBB_INNER_PRODUCT_H

/*! \file bolt/tbb/inner_product.h
    \brief performs inner product of 2 ranges.
*/


namespace bolt {
    namespace btbb {

        template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
        OutputType inner_product( InputIterator first1, InputIterator last1, InputIterator first2, OutputType init,
            BinaryFunction1 f1, BinaryFunction2 f2 );      

    };
};


#include <bolt/btbb/detail/inner_product.inl>

#endif