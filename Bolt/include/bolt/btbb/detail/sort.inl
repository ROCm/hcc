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

#if !defined(BOLT_BTBB_SORT_INL )
#define BOLT_BTBB_SORT_INL
#pragma once


namespace bolt {
    namespace btbb {


        template<typename RandomAccessIterator>
        void sort(RandomAccessIterator first,
            RandomAccessIterator last)
        {

        tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
        tbb::parallel_sort(first,last);
        }

        template<typename RandomAccessIterator, typename StrictWeakOrdering>
        void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
        {

        tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
        tbb::parallel_sort(first,last, comp);

        }

    }
}

#endif