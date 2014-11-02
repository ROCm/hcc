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

#if !defined( BOLT_BTBB_COUNT_INL)
#define BOLT_BTBB_COUNT_INL
#pragma once

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"

namespace bolt{
    namespace btbb {

            /*For documentation on the reduce object see below link
             *http://threadingbuildingblocks.org/docs/help/reference/algorithms/parallel_reduce_func.htm
             *The imperative form of parallel_reduce is used.
             *
            */
            template <typename T, typename InputIterator,typename Predicate>
            struct Count {
                T value;
                Predicate predicate;

                //TODO - Decide on how many threads to spawn? Usually it should be equal to th enumber of cores
                //You might need to look at the tbb::split and there there cousin's
                //
                Count (const Predicate &_opt) : predicate(_opt),value(0){}
                Count (): value(0) {}

                Count (Count & s, tbb::split ):predicate(s.predicate),value(0){}
                 void operator()( const tbb::blocked_range<InputIterator>& r )
                 {

                    for( InputIterator a=r.begin(); a!=r.end(); ++a )
                    {
                      if(predicate(*a))
                      {
                        value++;
                      }
                    }

                }
                 //Join is called by the parent thread after the child finishes to execute.
                void join(Count & rhs ) {
                    value = (value + rhs.value);
                }
            };

      template <typename T>
        struct CountIfEqual {
            CountIfEqual(const T &targetValue)  : _targetValue(targetValue)
            { };
            CountIfEqual(){}
            bool operator() (const T &x)
            {
                   T temp= _targetValue;
                   return x == temp;
            };

        private:
            T _targetValue;
        };



        template<typename InputIterator, typename Predicate>
        typename std::iterator_traits<InputIterator>::difference_type
            count_if(InputIterator first,
            InputIterator last,
            Predicate predicate)
            {

           			typedef typename std::iterator_traits<InputIterator>::difference_type iType;

                    tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
                    Count<iType,InputIterator,Predicate> count_op(predicate);
                    tbb::parallel_reduce( tbb::blocked_range<InputIterator>( first, last), count_op );
                    return count_op.value;

			}


        template<typename InputIterator, typename EqualityComparable>
        typename std::iterator_traits<InputIterator>::difference_type
            count(InputIterator first,
            InputIterator last,
            const EqualityComparable &value)
            {
				return count_if(first,last,CountIfEqual<EqualityComparable>(value));
			}





    } //tbb
} // bolt

#endif //BTBB_REDUCE_INL