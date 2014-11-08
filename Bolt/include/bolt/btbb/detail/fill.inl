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

#if !defined( BOLT_BTBB_FILL_INL)
#define BOLT_BTBB_FILL_INL
#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
//#include <thread>

namespace bolt{
    namespace btbb {

             template<typename ForwardIterator, typename T>
             struct Fill
             {
                T value;
                
                Fill (const T val) : value(val){}
                Fill (): value(0) {}
                Fill (Fill & s, tbb::split ): value(s.value){}

                void operator()( ForwardIterator first,  ForwardIterator last, T val)
                {
                    typedef typename std::iterator_traits<ForwardIterator>::value_type iType;

                    tbb::parallel_for(  tbb::blocked_range<ForwardIterator>(first, last) ,
                        [=] (const tbb::blocked_range<ForwardIterator> &r) -> void
                        {
                              for(ForwardIterator a = r.begin(); a!=r.end(); a++)
                                 *a = (iType) val;
                        });
                }

            };
      

           template<typename ForwardIterator, typename T>
           void fill( ForwardIterator first, ForwardIterator last, const T & value)
           {
             //Gets the number of concurrent threads supported by the underlying platform
             //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();

             //This allows TBB to choose the number of threads to spawn.
             tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

             //Explicitly setting the number of threads to spawn
             //tbb::task_scheduler_init((int) concurentThreadsSupported);

             Fill <ForwardIterator, T> fill_op(value);
             fill_op(first, last, value);

             //Fill <ForwardIterator, T> fill_op_split(fill_op);
             //fill_op_split(first, last, value);

          }

       
    } //tbb
} // bolt

#endif //BTBB_FILL_INL