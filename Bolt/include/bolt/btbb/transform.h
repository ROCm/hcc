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
#if !defined( BOLT_BTBB_TRANSFORM_H )
#define BOLT_BTBB_TRANSFORM_H

#include "tbb/parallel_for_each.h"
#include "tbb/parallel_for.h"

/*! \file transform.h
*/


namespace bolt
{
	namespace btbb
	{
		/*! \addtogroup algorithms
		 */

		/*! \addtogroup transformations
		*   \ingroup algorithms
		*   \p transform applies a specific function object to each element pair in the specified input ranges, and
		*   writes the result
		*   into the specified output range. For common code between the host
		*   and device, one can take a look at the TypeName implementations. See Bolt Tools for Split-Source
		*   for a detailed description.
		*/

		/*! \addtogroup TBB-transform
		*   \ingroup transformations
		*   \{
		*/


		/*! This version of \p transform applies a unary function to  input sequences and stores the result in the
		 *  corresponding position in an output sequence.
		 *  The input and output sequences can coincide, resulting in an
		 *  in-place transformation.
		 *
		 *  \param first The beginning of the first input sequence.
		 *  \param last The end of the first input sequence.
		 *  \param result The beginning of the output sequence.
		 *  \param op The tranformation operation.
		 *  \return The end of the output sequence.
		 *
		 *  \tparam InputIterator is a model of InputIterator
		 *                        and \c InputIterator's \c value_type is convertible to \c UnaryFunction's
		 * \c second_argument_type.
		 *  \tparam OutputIterator is a model of OutputIterator
		 *  \tparam UnaryFunction is a model of UnaryFunction
		 *                              and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's
		 * \c value_type.
		 *
		 *  The following code snippet demonstrates how to use \p transform.
		 *
		 *  \code
		 *  #include <bolt/amp/transform.h>
		 *  #include <bolt/amp/functional.h>
		 *
		 *  int input[10] = {-5,  0,  2,  3,  2,  4, -2,  1,  2,  3};
		 *  int output[10];
		 *
		 *  bolt::amp::negate<int> op;
		 *  bolt::amp::transform(ctl, input, input + 10, output, op);
		 *
		 *  // output is now {5,  0,  -2,  -3,  -2, - 4, 2,  -1,  -2,  -3};
		 *  \endcode
		 *
		 *  \sa http://www.sgi.com/tech/stl/transform.html
		 *  \sa http://www.sgi.com/tech/stl/InputIterator.html
		 *  \sa http://www.sgi.com/tech/stl/OutputIterator.html
		 *  \sa http://www.sgi.com/tech/stl/UnaryFunction.html
		 *  \sa http://www.sgi.com/tech/stl/BinaryFunction.html
		 */

		template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
		void transform(InputIterator first,
					   InputIterator last,
					   OutputIterator result,
					   UnaryFunction op);




		/*! \breif This version of \p transform applies a binary function to each pair
		 *  of elements from two input sequences and stores the result in the
		 *  corresponding position in an output sequence.
		 *  The input and output sequences can coincide, resulting in an
		 *  in-place transformation.
		 *
		 *  \param first1 The beginning of the first input sequence.
		 *  \param last1 The end of the first input sequence.
		 *  \param first2 The beginning of the second input sequence.
		 *  \param result The beginning of the output sequence.
		 *  \param op The tranformation operation.
		 *  \return The end of the output sequence.
		 *
		 *  \tparam InputIterator1 is a model of InputIterator
		 *                        and \c InputIterator1's \c value_type is convertible to \c BinaryFunction's
		 * \c first_argument_type.
		 *  \tparam InputIterator2 is a model of InputIterator
		 *                        and \c InputIterator2's \c value_type is convertible to \c BinaryFunction's
		 * \c second_argument_type.
		 *  \tparam OutputIterator is a model of OutputIterator
		 *  \tparam BinaryFunction is a model of BinaryFunction
		 *                              and \c BinaryFunction's \c result_type is convertible to \c OutputIterator's
		 * \c value_type.
		 *
		 *  \details The following code snippet demonstrates how to use \p transform.
		 *
		 *  \code
		 *  #include <bolt/amp/transform.h>
		 *  #include <bolt/amp/functional.h>
		 *
		 *  int input1[10] = {-5,  0,  2,  3,  2,  4, -2,  1,  2,  3};
		 *  int input2[10] = { 3,  6, -2,  1,  2,  3, -5,  0,  3,  3};
		 *  int output[10];
		 *
		 *  bolt::plus<int> op;
		 *  bolt::amp::transform(ctl, input1, input1 + 10, input2, output, op);
		 *
		 *  // output is now {-2,  6,  0,  4,  4,  7, -7, 1, 5, 6};
		 *  \endcode
		 *
		 *  \sa http://www.sgi.com/tech/stl/transform.html
		 *  \sa http://www.sgi.com/tech/stl/InputIterator.html
		 *  \sa http://www.sgi.com/tech/stl/OutputIterator.html
		 *  \sa http://www.sgi.com/tech/stl/UnaryFunction.html
		 *  \sa http://www.sgi.com/tech/stl/BinaryFunction.html
		 */

		template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
		void transform(InputIterator1 first1,
					   InputIterator1 last1,
					   InputIterator2 first2,
					   OutputIterator result,
					   BinaryFunction op);
	 /*!   \}  */

	}//tbb namespace ends
}//bolt namespace ends


#include <bolt/btbb/detail/transform.inl>


#endif // TBB_TRANSFORM_H


