/***************************************************************************
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
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

#if !defined( BOLT_CL_INNERPRODUCT_H )
#define BOLT_CL_INNERPRODUCT_H
#pragma once

#include "bolt/cl/device_vector.h"

/*! \file bolt/cl/inner_product.h
    \brief Inner Product returns the inner product of two iterators.
*/



namespace bolt {
    namespace cl {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup reductions
        *   \ingroup algorithms
        */

        /*! \addtogroup CL-inner_product
        *   \ingroup reductions
        *   \{
        */

        /*! \brief Inner Product returns the inner product of two iterators.
        * This is similar to calculating binary transform and then reducing the result.
        * The \p inner_product operation is similar the std::inner_product function.
        *  This function can take  optional \p control structure to control command-queue.
        *
        * \param ctl    \b Optional Control structure to control command-queue, debug, tuning. See bolt::cl::control.
        * \param first1 The beginning of input sequence.
        * \param last1  The end of input sequence.
		    * \param first2 The beginning of the second input sequence.
		    * \param init   The initial value for the accumulator.
        * \param cl_code Optional OpenCL(TM) code to be passed to the OpenCL compiler. The cl_code is inserted first in
        *                the generated code, before the cl_code trait.
        * \tparam InputIterator An iterator that can be dereferenced for an object, and can be incremented to get to
        *                       the next element in a sequence.
        * \tparam OutputType The type of the result.
        * \return The result of the inner product.
        *
        *\details The following code example shows the use of \p inner_product to perform dot product on two vectors of
        * size 10 , using the default multiplies and plus operator.
        * \code
        * #include <bolt/cl/inner_product.h>
        *
        * int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
		    * int b[10] = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55};
        *
        * int ip = bolt::cl::inner_product(a, a+10, b,0);
        * // sum = 1209
        *  \endcode
        *
        * \sa http://www.sgi.com/tech/stl/inner_product.html
        */
		template<typename InputIterator, typename OutputType>
        OutputType inner_product( bolt::cl::control &ctl,  InputIterator first1, InputIterator last1,
            InputIterator first2, OutputType init,
            const std::string& cl_code="");

		template<typename InputIterator, typename OutputType>
        OutputType inner_product( InputIterator first1, InputIterator last1, InputIterator first2, OutputType init,
            const std::string& cl_code="");


        /*! \brief Inner Product returns the inner product of two iterators using user specified binary functors
        * f1 and f2.
        * This is similar to calculating transform and then reducing the result. The functor f1 should be commutative.
        * This function can take  optional \p control structure to control command-queue.
        * The \p inner_product operation is similar the std::inner_product function.
        *
        * \param ctl    \b Optional Control structure to control command-queue, debug, tuning. See FIXME.
        * \param first1 The first position in the input sequence.
        * \param last1  The last position in the input sequence.
		    * \param first2  The beginning of second input sequence.
		    * \param init   The initial value for the accumulator.
		    * \param f1		Binary functor for reduction.
		    * \param f2     Binary functor for transformation.
        * \param cl_code Optional OpenCL(TM) code to be passed to the OpenCL compiler. The cl_code is inserted first in
        *                the generated code, before the cl_code trait.
        * \tparam InputIterator An iterator that can be dereferenced for an object, and can be incremented to get to
        *                       the next element in a sequence.
        * \tparam OutputType The type of the result.
        * \return The result of the inner product.
        *
        *\details The following code example shows the use of \p inner_product on two vectors of size 10, using the
        * user defined functors.
        * \code
        *
        * #include <bolt/cl/functional.h>    //for bolt::cl::plus
        * #include <bolt/cl/inner_product.h>
        *
        * int a[10] = {-5,  0,  2,  3,  2,  4, -2,  1,  2,  3};
		    * int b[10] = {-5,  0,  2,  3,  2,  4, -2,  1,  2,  3};
        *
        * int ip = bolt::cl::inner_product(a, a+10, b, 0, bolt::cl::plus<int>(),bolt::cl::multiplies<int>());
        * // sum = 76
        *  \endcode
        * \sa http://www.sgi.com/tech/stl/inner_product.html
        */

        template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
        OutputType inner_product( bolt::cl::control &ctl,  InputIterator first1, InputIterator last1,
            InputIterator first2, OutputType init,
            BinaryFunction1 f1, BinaryFunction2 f2, const std::string& cl_code="");

        template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
        OutputType inner_product( InputIterator first1, InputIterator last1, InputIterator first2, OutputType init,
            BinaryFunction1 f1, BinaryFunction2 f2, const std::string& cl_code="");

        /*!   \}  */
    };
};

#include <bolt/cl/detail/inner_product.inl>
#endif