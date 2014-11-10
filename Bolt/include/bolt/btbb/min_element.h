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
#if !defined( BOLT_BTBB_MIN_ELEMENT_H )
#define BOLT_BTBB_MIN_ELEMENT_H

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"

/*! \file bolt/tbb/min_element.h
    \brief finds the minimum element in the given input vector
*/


namespace bolt {
    namespace btbb {

        template<typename ForwardIterator,typename BinaryPredicate>
        ForwardIterator min_element(ForwardIterator first, ForwardIterator last, BinaryPredicate binary_op);

        template<typename ForwardIterator,typename BinaryPredicate>
        ForwardIterator max_element(ForwardIterator first, ForwardIterator last, BinaryPredicate binary_op);

    };
};


#include <bolt/btbb/detail/min_element.inl>

#endif