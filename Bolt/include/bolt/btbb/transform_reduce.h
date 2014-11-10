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

#pragma once
#if !defined( BOLT_BTBB_TRANSFORM_REDUCE_H )
#define BOLT_BTBB_TRANSFORM_REDUCE_H

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"


/*! \file bolt/btbb/transform_reduce.h
	\brief  Fuses transform and reduce operations together.
*/

namespace bolt {
	namespace btbb {

		/*! \addtogroup algorithms
		 */

		/*! \addtogroup reductions
		*   \ingroup algorithms
		*/

		/*! \addtogroup TBB-transform_reduce
		*   \ingroup reductions
		*   \{
		*/



		/*! \brief \p transform_reduce fuses transform and reduce operations together, increasing performance by
		 *  reducing memory passes.
		 *  \details Logically, a transform operation is performed over the input sequence using the unary function and
		 *  stored into a temporary sequence; then, a reduction operation is applied using the binary function
		 *  to return a single value.
		 *
		 * \param first The beginning of the input sequence.
		 * \param last The end of the input sequence.
		 * \param transform_op A unary tranformation operation.
		 * \param init  The initial value for the accumulator.
		 * \param reduce_op  The binary operation used to combine two values.   By default, the binary operation is
		 *  plus<>().
		 * \return The result of the combined transform and reduction.
		 *
		 *  \tparam T The type of the result.
		 *  \tparam InputIterator is a model of an InputIterator.
		 *         and \c InputIterator's \c value_type is convertible to \c BinaryFunction's \c first_argument_type.
		 *  \tparam UnaryFunction is a model of Unary Function.
		 *          and \c UnaryFunction's \c result_type is convertible to \c InputIterator's \c value_type.
		 *  \tparam BinaryFunction is a model of Binary Function.
		 *           and \c BinaryFunction's \c result_type is convertible to \c InputIterator's \c value_type.
		 *
		 *  \code
		 *  #include <bolt/btbb/transform_reduce.h>
		 *  #include <bolt/btbb/functional.h>
		 *
		 *  int input[10] = {-5,  0,  2,  3,  2,  4, -2,  1,  2,  3};
		 *  int output;
		 *
		 *  bolt::btbb::transform_reduce( input, input + 10, square<int>(), 0, plus<int>() );
		 *
		 *  // output is 76
		 *  \endcode
		 *
		 *  \sa http://www.sgi.com/tech/stl/InputIterator.html
		 *  \sa http://www.sgi.com/tech/stl/OutputIterator.html
		 *  \sa http://www.sgi.com/tech/stl/UnaryFunction.html
		 *  \sa http://www.sgi.com/tech/stl/BinaryFunction.html
		 */

		 template<typename InputIterator, typename UnaryFunction, typename T, typename BinaryFunction>
		T transform_reduce(
			InputIterator first,
			InputIterator last,
			UnaryFunction transform_op,
			T init,
			BinaryFunction reduce_op);


		/*!   \}  */

	};
};

#include <bolt/btbb/detail/transform_reduce.inl>

#endif