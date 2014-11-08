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

#if !defined( BOLT_BTBB_BINARY_SEARCH_INL)
#define BOLT_BTBB_BINARY_SEARCH_INL
#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include <iterator>

namespace bolt{
    namespace btbb {

             template<typename ForwardIterator, typename T>
             struct BS {

                bool result;

                BS () {result=false;}

				void operator()( ForwardIterator first, int n, const T & val)
                {

                    tbb::parallel_for(  tbb::blocked_range<int>(0, (int) n) ,
                        [&] (const tbb::blocked_range<int> &r) -> void
                        {

                              int low=r.begin();
                              int high=r.end();
                              int mid;
                              T midVal, firstVal;

                              while(low<high)
                              {
                                     mid = (low + high) / 2;

                                     midVal = first[mid];
                                     firstVal = first[low];

                                     if( midVal == val)
                                     {
                                            result = true;
                                            break;
                                     }
                                     else if (midVal < val)
                                            low = mid + 1;
                                     else
                                            high = mid;
                              }

                              //result = std::binary_search(first+r.begin(), first+r.end(), val);
                          });
                 }

            };

             template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
             struct BS_comp {

                bool result;

                BS_comp () {result=false;}

				void operator()( ForwardIterator first, int n, const T & val, StrictWeakOrdering comp)
                {

                    tbb::parallel_for(  tbb::blocked_range<int>(0, (int) n) ,
                        [&] (const tbb::blocked_range<int> &r) -> void
                        {

                              int low=r.begin();
                              int high=r.end();
                              int mid;
                              T midVal, firstVal;

                              while(low<high)
                              {
                                     mid = (low + high) / 2;

                                     midVal = first[mid];
                                     firstVal = first[low];

                                     if( (!comp(midVal, val)) && (!comp(val, midVal)) )
                                     {
                                            this->result = true;
                                            break;
                                     }
                                     else if ( comp(midVal, val) )
                                            low = mid + 1;
                                     else
                                            high = mid;
                              }

                              //result = std::binary_search(first+r.begin(), first+r.end(), val, comp);
                          });
                 }

            };

            template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
            bool binary_search( ForwardIterator first, ForwardIterator last, const T & value, StrictWeakOrdering comp)
            {

               //This allows TBB to choose the number of threads to spawn.
               tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

               int n = (int)std::distance(first, last);

               BS_comp <ForwardIterator, T, StrictWeakOrdering> bs_op;
               bs_op(first, n, value, comp);

               return bs_op.result;
            }

            template<typename ForwardIterator, typename T>
            bool binary_search( ForwardIterator first, ForwardIterator last, const T & value)
            {

               //This allows TBB to choose the number of threads to spawn.
               tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

               int n = (int)std::distance(first, last);

               BS <ForwardIterator, T> bs_op;
               bs_op(first, n, value);

               return bs_op.result;
            }


    } //tbb
} // bolt

#endif //BTBB_BINARY_SEARCH__INL