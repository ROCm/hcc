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

#if !defined( BOLT_BTBB_MERGE_INL)
#define BOLT_BTBB_MERGE_INL
#pragma once


#include "tbb/parallel_for.h"
#include "tbb/parallel_invoke.h"
#include "tbb/task_scheduler_init.h"

namespace bolt{
    namespace btbb {

        
		template<typename T1,typename T2>
		struct CompareOp
		{
			bool operator()(const T1 &lhs, const T2 &rhs) const  {return (lhs < rhs) ? true:false;}
		};

            using namespace tbb;

            template<typename InputIterator1 , typename InputIterator2 , typename OutputIterator,
            typename StrictWeakCompare>
            struct ParallelMerge {
                static size_t grainsize;
                InputIterator1 begin1, end1; // [begin1,end1) is 1st sequence to be merged
                InputIterator2 begin2, end2; // [begin2,end2) is 2nd sequence to be merged
                OutputIterator out;               // where to put merged sequence   
                StrictWeakCompare comp;

                bool empty()   const {return (end1-begin1)+(end2-begin2)==0;}
                
                bool is_divisible() const 
                {
			
		  size_t min;
 		   min = end1-begin1 < end2-begin2?end1-begin1:end2-begin2;
                    return  min > grainsize;
                }


                ParallelMerge( ParallelMerge& r, split ) 
                {
                    if( r.end1-r.begin1 < r.end2-r.begin2 ) {
                        std::swap(r.begin1,r.begin2);
                        std::swap(r.end1,r.end2);
                    }
                    begin1 = r.begin1 + (r.end1-r.begin1)/2;
                    begin2 = std::lower_bound( r.begin2, r.end2, *begin1,comp );
                    end1 = r.end1;
                    end2 = r.end2;
                    r.end1 = begin1;
                    r.end2 = begin2;

                    out = r.out + ( r.end1-r.begin1) + ( r.end2-r.begin2);
                }
                ParallelMerge( InputIterator1 begin1_, InputIterator1 end1_, 
                                    InputIterator2 begin2_, InputIterator2 end2_, 
                                    OutputIterator out_,StrictWeakCompare _comp ) :
                    begin1(begin1_), end1(end1_), 
                    begin2(begin2_), end2(end2_), out(out_),comp(comp)
                {}
            };

            template<typename InputIterator1 , typename InputIterator2 , typename OutputIterator,
            typename StrictWeakCompare>
            size_t ParallelMerge<InputIterator1,InputIterator2,OutputIterator,
            StrictWeakCompare>::grainsize = 1000;

            template<typename InputIterator1 , typename InputIterator2 , typename OutputIterator,
            typename StrictWeakCompare>
            struct ParallelMergeCode {
                void operator()( ParallelMerge<InputIterator1,InputIterator2,OutputIterator,
            StrictWeakCompare> & r ) const {
                    std::merge( r.begin1, r.end1, r.begin2, r.end2, r.out,r.comp );
                }
            };

        template<typename InputIterator1 , typename InputIterator2 , typename OutputIterator,
            typename StrictWeakCompare>
            void PMerge( InputIterator1 begin1, InputIterator1 end1, InputIterator2 begin2, 
            InputIterator2 end2, OutputIterator out,StrictWeakCompare comp ) 
        {

            tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

                parallel_for(     
                   btbb::ParallelMerge<InputIterator1,InputIterator2,OutputIterator,
            StrictWeakCompare>(begin1,end1,begin2,end2,out,comp),
                   btbb::ParallelMergeCode<InputIterator1,InputIterator2,OutputIterator,
            StrictWeakCompare> (),
                   simple_partitioner() 
                );

            }




        template<typename InputIterator1 , typename InputIterator2 , typename OutputIterator,
            typename StrictWeakCompare>
        OutputIterator merge (InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
        InputIterator2 last2, OutputIterator result,StrictWeakCompare comp)
        {

            btbb::PMerge(first1,last1,first2,last2,result,comp);
            return result + (last1 - first1) + (last2 - first2);

		}


        template<typename InputIterator1 , typename InputIterator2 , typename OutputIterator >
        OutputIterator merge (InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
        InputIterator2 last2, OutputIterator result)
        {

			return merge(first1,last1,first2,last2,result,CompareOp<InputIterator1,InputIterator2>());

		}


    } //tbb
} // bolt

#endif //BTBB_MERGE_INL
