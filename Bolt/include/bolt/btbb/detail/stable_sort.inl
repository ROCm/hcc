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

#if !defined( BOLT_BTBB_STABLE_SORT_INL)
#define BOLT_BTBB_STABLE_SORT_INL
#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_invoke.h"
#include <iterator>

namespace bolt{
    namespace btbb {
       
             template<typename RandomAccessIterator>
             struct StableSort
             {

               StableSort () {}  

               void operator() (RandomAccessIterator first, RandomAccessIterator last)
               {
                    int n = (int) std::distance(first, last);

                    if(n == 1)  // Only one element
                         return; // Nothing to Sort!
                    else
                         Parallel_Merge_Sort(first, last);
                    
               }    

               void Parallel_Merge_Sort(RandomAccessIterator beg, RandomAccessIterator end)
               {
                       if (end - beg > 1)
                       {
                             RandomAccessIterator mid = beg + (end - beg) / 2;

                              tbb::parallel_invoke(
                                [&] { Parallel_Merge_Sort( beg, mid); }, 
                                [&] { Parallel_Merge_Sort( mid, end); }  
                              );

                             std::inplace_merge(beg, mid, end);
                       }
                   
               }

           };

 template<typename RandomAccessIterator, typename StrictWeakOrdering>
        void Parallel_Merge_Sort(RandomAccessIterator beg, RandomAccessIterator end,  StrictWeakOrdering comp)
       {
               if (end - beg > 1)
               {
                     RandomAccessIterator mid = beg + (end - beg) / 2;

                      tbb::parallel_invoke(
                        [&] { Parallel_Merge_Sort( beg, mid, comp); }, 
                        [&] { Parallel_Merge_Sort( mid, end, comp); }  
                      );

                     std::inplace_merge(beg, mid, end, comp);
               }
           
       }


           template<typename RandomAccessIterator, typename StrictWeakOrdering>
           struct StableSort_comp
           {

               StableSort_comp () {}  



               void operator() (RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp )
               {
                    int n = (int) std::distance(first, last);

                    if(n == 1)  // Only one element
                         return; // Nothing to Sort!
                    else   
                         Parallel_Merge_Sort(first, last, comp);
                    
               }    



           };

           template<typename RandomAccessIterator>
           void stable_sort(RandomAccessIterator first, RandomAccessIterator last)
           {
                //This allows TBB to choose the number of threads to spawn.
                tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
                StableSort <RandomAccessIterator > stable_sort_op;
                stable_sort_op(first, last);
           }

           template<typename RandomAccessIterator, typename StrictWeakOrdering>
           void stable_sort(RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
           {
               //This allows TBB to choose the number of threads to spawn.
                tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic); 			   
                StableSort_comp <RandomAccessIterator, StrictWeakOrdering > stable_sort_op;
                stable_sort_op(first, last, comp);
           }
       
    } //tbb
} // bolt

#endif //BTBB_STABLE_SORT_INL
