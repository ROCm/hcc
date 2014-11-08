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

#if !defined( BOLT_BTBB_REDUCE_INL)
#define BOLT_BTBB_REDUCE_INL
#pragma once

//#include <thread>
#include "tbb/partitioner.h"

namespace bolt{
    namespace btbb {

             /*For documentation on the reduce object see below link
             *http://threadingbuildingblocks.org/docs/help/reference/algorithms/parallel_reduce_func.htm
             *The imperative form of parallel_reduce is used.
             *
            */
            template <typename T, typename InputIterator,typename BinaryFunction>
            struct Reduce {
                T value;
                BinaryFunction op;
                bool flag;

                //TODO - Decide on how many threads to spawn? Usually it should be equal to th enumber of cores
                //You might need to look at the tbb::split and there there cousin's
                //
                Reduce(const T &init) : value(init) {}
                Reduce(const BinaryFunction &_op, const T &init) : op(_op), value(init), flag(false) {}
                Reduce() : value(0) {}
                Reduce( Reduce& s, tbb::split ) : flag(true), op(s.op) {}
                void operator()( const tbb::blocked_range<InputIterator>& r ) {
                    T temp = value;
					InputIterator rend = r.end();
                    for( InputIterator a=r.begin(); a!=rend; ++a ) {
                      if(flag){
                        temp = (T) *a;
                        flag = false;
                      }
                      else
                        temp = (T)op(temp,*a);
                    }
                    value = temp;
                }
                //Join is called by the parent thread after the child finishes to execute.
                void join( Reduce& rhs )
                {
                    value = (T) op(value,rhs.value);
                }
            };

        template<typename InputIterator, typename T, typename BinaryFunction>
        T reduce(InputIterator first,
            InputIterator last,
            T init,
            BinaryFunction binary_op)
        {
            typedef typename std::iterator_traits<InputIterator>::value_type iType;
            //tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

			//Gets the number of concurrent threads supported by the underlying platform
            //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
			unsigned int concurentThreadsSupported = tbb::task_scheduler_init::default_num_threads();
			//Explicitly setting the number of threads to spawn
            tbb::task_scheduler_init((int) concurentThreadsSupported);

            Reduce<T,InputIterator, BinaryFunction> reduce_op(binary_op, init);
            tbb::parallel_reduce( tbb::blocked_range<InputIterator>( first, last, 100000), reduce_op, tbb::auto_partitioner() );
            return reduce_op.value;
        }

        template<typename InputIterator, typename T>
        T   reduce(InputIterator first,
            InputIterator last,
            T init)
        {

            typedef typename std::iterator_traits<InputIterator>::value_type iType;
	    reduce(first,last,iType(),std::plus<iType>());

        }


        template<typename InputIterator>
        typename std::iterator_traits<InputIterator>::value_type
            reduce(InputIterator first,
            InputIterator last)
        {

            typedef typename std::iterator_traits<InputIterator>::value_type iType;
	    reduce(first,last,iType());

        }







    } //tbb
} // bolt

#endif //BTBB_REDUCE_INL
