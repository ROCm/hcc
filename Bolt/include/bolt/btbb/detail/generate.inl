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

#if !defined( BOLT_BTBB_GENERATE_INL)
#define BOLT_BTBB_GENERATE_INL
#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace bolt{
    namespace btbb {

             template<typename ForwardIterator, typename Generator>
             struct Generate {
                Generator value;

                Generate (const Generator val) : value(val){}
                Generate (): value(0) {}
                Generate (Generate & s, tbb::split ): value(s.value){}

                void operator()( ForwardIterator first,  ForwardIterator last, Generator gen)
                {
                    typedef typename std::iterator_traits<ForwardIterator>::value_type iType;

                    tbb::parallel_for(  tbb::blocked_range<ForwardIterator>(first, last) ,
                        [&] (const tbb::blocked_range<ForwardIterator> &r) -> void
                        {
                              for(ForwardIterator a = r.begin(); a!=r.end(); a++)
                                  *a = (iType) gen();
                        });
                }
            };      
        

            template<typename ForwardIterator, typename Generator>
            void generate( ForwardIterator first, ForwardIterator last, Generator gen)
            {
               //This allows TBB to choose the number of threads to spawn.
               tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
               Generate <ForwardIterator, Generator> generate_obj(gen);
               generate_obj(first, last, gen);
            }       
    } //tbb
} // bolt

#endif //BTBB_GENERATE_INL
