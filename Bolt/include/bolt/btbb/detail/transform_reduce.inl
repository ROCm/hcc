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

/*! \file bolt/amp/transform.h
	\brief  Applies a specific function object to each element pair in the specified input ranges.
*/


#if !defined( BOLT_BTBB_TRANSFORM_REDUCE_INL )
#define BOLT_BTBB_TRANSFORM_REDUCE_INL
#pragma once

namespace bolt {
	namespace btbb {
			/*For documentation on the reduce object see below link
			 *http://threadingbuildingblocks.org/docs/help/reference/algorithms/parallel_reduce_func.htm
			 *The imperative form of parallel_reduce is used.
			 *
			*/
			template < typename InputIterator, typename UnaryFunction, typename BinaryFunction,typename T>
			struct Transform_Reduce {
				T value;
				BinaryFunction reduce_op;
				UnaryFunction transform_op;
				bool flag;

				//TODO - Decide on how many threads to spawn? Usually it should be equal to th enumber of cores
				//You might need to look at the tbb::split and there there cousin's
				//
				Transform_Reduce(const UnaryFunction &_opt, const BinaryFunction &_opr) : transform_op(_opt), reduce_op(_opr) ,value(0){}
				Transform_Reduce(const UnaryFunction &_opt, const BinaryFunction &_opr, const T &init) : transform_op(_opt), reduce_op(_opr), value(init), flag(false){}

				Transform_Reduce(): value(0) {}
				Transform_Reduce( Transform_Reduce& s, tbb::split ):flag(true),transform_op(s.transform_op),reduce_op(s.reduce_op){}
				 void operator()( const tbb::blocked_range<InputIterator>& r ) {
					T reduce_temp = value, transform_temp;
					for(InputIterator a=r.begin(); a!=r.end(); ++a ) {
					  transform_temp = transform_op(*a);
					  if(flag){
						reduce_temp = transform_temp;
						flag = false;
					  }
					  else
						reduce_temp = reduce_op(reduce_temp,transform_temp);
					}
					value = reduce_temp;
				}
				 //Join is called by the parent thread after the child finishes to execute.
				void join( Transform_Reduce& rhs ) {
					value = reduce_op(value,rhs.value);
				}
			};


		 template<typename InputIterator, typename UnaryFunction, typename T, typename BinaryFunction>
		T transform_reduce(
			InputIterator first,
			InputIterator last,
			UnaryFunction transform_op,
			T init,
			BinaryFunction reduce_op)
		{

				  typedef typename std::iterator_traits< InputIterator >::value_type iType;
					tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
					Transform_Reduce<InputIterator, UnaryFunction, BinaryFunction,T> transform_reduce_op(transform_op, reduce_op, init);
					tbb::parallel_reduce( tbb::blocked_range<InputIterator>( first, last), transform_reduce_op );
					return transform_reduce_op.value;

		}

	}
}
#endif
