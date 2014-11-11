/***************************************************************************
*   Copyright 2012 Advanced Micro Devices, Inc.
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

#if !defined(BOLT_BTBB_REDUCE_BY_KEY_H )
#define BOLT_BTBB_REDUCE_BY_KEY_H
#pragma once

#include "tbb/parallel_scan.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"



/*! \file bolt/btbb/reduce_by_key.h
    \brief Returns the sorted result of all the elements in input according to key values.
*/

namespace bolt {
    namespace btbb {

        template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2>
                unsigned int
                reduce_by_key(
                InputIterator1  keys_first,
                InputIterator1  keys_last,
                InputIterator2  values_first,
                OutputIterator1  keys_output,
                OutputIterator2  values_output);

       template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2,
                typename BinaryPredicate>
                unsigned int
                reduce_by_key(
                InputIterator1  keys_first,
                InputIterator1  keys_last,
                InputIterator2  values_first,
                OutputIterator1  keys_output,
                OutputIterator2  values_output,
                BinaryPredicate binary_pred);

       
            template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2,
                typename BinaryPredicate,
                typename BinaryFunction>
                unsigned int reduce_by_key(  InputIterator1  keys_first,
                                             InputIterator1  keys_last,
                                             InputIterator2  values_first,
                                             OutputIterator1  keys_output,
                                             OutputIterator2  values_output,
                                             BinaryPredicate binary_pred,
                                             BinaryFunction  binary_op);

    }// end of bolt::btbb namespace
}// end of bolt namespace

#include <bolt/btbb/detail/reduce_by_key.inl>

#endif