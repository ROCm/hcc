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

#if !defined(BOLT_BTBB_STABLE_SORT_BY_KEY_H )
#define BOLT_BTBB_STABLE_SORT_BY_KEY_H
#pragma once

#include "tbb/task_scheduler_init.h"


/*! \file bolt/btbb/stable_sort_by_key.h
    \brief Returns the sorted result of all the elements in input according to key values and maintains relative ordering of the duplicate elements.
*/

namespace bolt {
    namespace btbb {

        template< typename RandomAccessIterator1, typename RandomAccessIterator2 > 
        void stable_sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, 
        RandomAccessIterator2 values_first);

        template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering> 
        void stable_sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, 
        StrictWeakOrdering comp);

    }// end of bolt::btbb namespace
}// end of bolt namespace

#include <bolt/btbb/detail/stable_sort_by_key.inl>

#endif