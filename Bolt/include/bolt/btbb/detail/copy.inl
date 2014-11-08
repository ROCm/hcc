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

#if !defined( BOLT_BTBB_COPY_INL)
#define BOLT_BTBB_COPY_INL
#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include <iterator>

namespace bolt{
    namespace btbb {

             template<typename InputIterator, typename Size, typename OutputIterator>
             struct Copy_n {

               Copy_n () {}

				void operator()( InputIterator first, Size n, OutputIterator result)
                {
                    tbb::parallel_for(  tbb::blocked_range<int>(0, (int) n) ,
                        [&] (const tbb::blocked_range<int> &r) -> void
                        {
                              
                              for(int i = r.begin(); i!=r.end(); i++)
                              {
                                 
                                   *(result+i) = *(first+i);
                              }   
                          
                        });

                }


            };


            template<typename InputIterator, typename Size, typename OutputIterator>
            OutputIterator copy_n(InputIterator first, Size n, OutputIterator result)
            {
               //Gets the number of concurrent threads supported by the underlying platform
               //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();

               //This allows TBB to choose the number of threads to spawn.
               tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

               //Explicitly setting the number of threads to spawn
               //tbb::task_scheduler_init((int) concurentThreadsSupported);

               Copy_n <InputIterator, Size, OutputIterator> copy_op;
               copy_op(first, n, result);

               return result;
            }

       
    } //tbb
} // bolt

#endif //BTBB_COPY_INL