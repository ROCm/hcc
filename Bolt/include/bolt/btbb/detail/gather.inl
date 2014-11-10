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

/*! \file bolt/btbb/scatter.h
*/


#if !defined( BOLT_BTBB_GATHER_INL )
#define BOLT_BTBB_GATHER_INL
#pragma once
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace bolt 
{
    namespace btbb
    {

template<typename InputIterator1,
         typename InputIterator2, 
         typename OutputIterator>

void gather(InputIterator1 mapfirst, 
             InputIterator1 maplast,
             InputIterator2 input, 
             OutputIterator result)
             { 
                // std::cout<<"TBB code path...\n";
                 size_t numElements = static_cast< unsigned int >( std::distance( mapfirst, maplast ) );
                //This allows TBB to choose the number of threads to spawn.               
                 tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);       
                 tbb::parallel_for (tbb::blocked_range<size_t>(0,numElements),[&](const tbb::blocked_range<size_t>& r)
                  {
                    for(size_t iter = r.begin(); iter!=r.end(); iter++)
                        *(result + (int)iter) = * (input + (int)mapfirst[(int)iter]); 
                  });
             }

template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator>

 void gather_if( InputIterator1 mapfirst,
                  InputIterator1 maplast,
                  InputIterator2 stencil,
                  InputIterator3 input,
                  OutputIterator result)
        {
                 //std::cout<<"TBB code path...\n";
                 size_t numElements = static_cast< unsigned int >( std::distance( mapfirst, maplast ) );
                 //This allows TBB to choose the number of threads to spawn.               
                 tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
                 tbb::parallel_for (tbb::blocked_range<size_t>(0,numElements),[&](const tbb::blocked_range<size_t>& r)
                 {
                    for(size_t iter = r.begin(); iter!=r.end(); iter++)
                    {
                         if(stencil[(int)iter]== 1)	   
                                 result[(int)iter] = input[mapfirst[(int)iter]];       
                    }					
                });
        }


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename OutputIterator,
         typename BinaryPredicate>

 void gather_if(  InputIterator1 mapfirst,
                  InputIterator1 maplast,
                  InputIterator2 stencil,
                  InputIterator3 input,
                  OutputIterator result,
                  BinaryPredicate pred)
        {
                 //std::cout<<"TBB code path...\n";
                 size_t numElements = static_cast< unsigned int >( std::distance( mapfirst, maplast) );
                 //This allows TBB to choose the number of threads to spawn.               
                 tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
                 tbb::parallel_for (tbb::blocked_range<size_t>(0,numElements),[&](const tbb::blocked_range<size_t>& r)
                 {
                    for(size_t iter = r.begin(); iter!=r.end(); iter++)
                    {
                         if(pred(stencil[(int)iter]))   
                                  result[(int)iter] = input[mapfirst[(int)iter]]; 						            
                    }					
                });
        }

    }
}

#endif // TBB_GATHER_INL


