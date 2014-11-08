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

#if !defined( BOLT_BTBB_REDUCE_H )
#define BOLT_BTBB_REDUCE_H
#pragma once

/*! \file bolt/tbb/reduce.h
    \brief Returns the result of combining all the elements in the specified range using the specified.
*/

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"



namespace bolt {
    namespace btbb {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup reductions
        *   \ingroup algorithms
        *   Family of reduction operations for boiling data down to a small set by summation, counting, finding
        *   min/max, and more.
        */

        /*! \addtogroup TBB-reduce
        *   \ingroup reductions
        *   \{
        */


        /*! \brief \p reduce returns the result of combining all the elements in the specified range using the specified
        * binary_op.
        * The classic example is a summation, where the binary_op is the plus operator.  By default, the initial value
        * is "0"
        * and the binary operator is "plus<>()".
        *
        * \details \p reduce requires that the binary reduction op ("binary_op") is cummutative.  The order in which
        * \p reduce applies the binary_op
        * is not deterministic.
        *
        * \details The \p reduce operation is similar the std::accumulate function
        *
        * \param first The first position in the sequence to be reduced.
        * \param last  The last position in the sequence to be reduced.
        * \tparam InputIterator An iterator that can be dereferenced for an object, and can be incremented to get to
        * the next element in a sequence.
        * \tparam T The type of the result.
        * \return The result of the reduction.
        *
        * \details The following code example shows the use of \p reduce to sum 10 numbers, using the default plus
        * operator.
        * \code
        * #include <bolt/btbb/reduce.h>
        *
        * int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        *
        * int sum = bolt::btbb::reduce(a, a+10);
        * // sum = 55
        *  \endcode
        * \sa http://www.sgi.com/tech/stl/accumulate.html
        */

        template<typename InputIterator>
        typename std::iterator_traits<InputIterator>::value_type
            reduce(InputIterator first,
            InputIterator last);


        /*! \brief \p reduce returns the result of combining all the elements in the specified range using the
        * specified binary_op.
        * The classic example is a summation, where the binary_op is the plus operator.  By default, the initial value
        * is "0"
        * and the binary operator is "plus<>()".
        *
        * \details \p reduce requires that the binary reduction op ("binary_op") is cummutative.  The order in which
        * \p reduce applies the binary_op
        * is not deterministic.
        *
        * \details The \p reduce operation is similar the std::accumulate function
        *
        * \param first The first position in the sequence to be reduced.
        * \param last  The last position in the sequence to be reduced.
        * \param init  The initial value for the accumulator.
        * \tparam InputIterator An iterator that can be dereferenced for an object, and can be incremented to get to
        * the next element in a sequence.
        * \tparam T The type of the result.
        * \return The result of the reduction.
        *
        * \details The following code example shows the use of \p reduce to sum 10 numbers, using the default plus
        * operator.
        * \code
        * #include <bolt/btbb/reduce.h>
        *
        * int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        *
        * int init = 10;
        *
        * int sum = bolt::tbb::reduce(a, a+10, init);
        * // sum = 65
        *  \endcode
        * \sa http://www.sgi.com/tech/stl/accumulate.html
        */
        template<typename InputIterator, typename T>
        T   reduce(InputIterator first,
            InputIterator last,
            T init);

        /*! \brief \p reduce returns the result of combining all the elements in the specified range using the specified
        * binary_op.
        * The classic example is a summation, where the binary_op is the plus operator.  By default,
        * the binary operator is "plus<>()".  The version takes a bolt::cl::control structure as a first argument.
        *
        * \details \p reduce requires that the binary reduction op ("binary_op") is cummutative.  The order in which
        * \p reduce applies the binary_op
        * is not deterministic.
        *
        * \details The \p reduce operation is similar the std::accumulate function.
        *
        * \param first The first position in the sequence to be reduced.
        * \param last  The last position in the sequence to be reduced.
        * \param init  The initial value for the accumulator.
        * \param binary_op  The binary operation used to combine two values.   By default, the binary operation is
        * plus<>().
        * \tparam InputIterator An iterator that can be dereferenced for an object, and can be incremented to get to
        * the next element in a sequence.
        * \tparam BinaryFunction A function object defining an operation that is applied to consecutive elements in the
        * sequence.
        * \return The result of the reduction.
        *
        * \details The following code example shows the use of \p reduce to find the max of 10 numbers,
        * specifying a specific command-queue and enabling debug messages.
        \code
        #include <bolt/btbb/reduce.h>

        int a[10] = {2, 9, 3, 7, 5, 6, 3, 8, 3, 4};

        int max = bolt::tbb::reduce(ctl, a, a+10, -1,bolt::amp:maximum<int>());
        // max = 9
        \endcode
        * \sa http://www.sgi.com/tech/stl/accumulate.html
        */
        template<typename InputIterator, typename T, typename BinaryFunction>
        T reduce(InputIterator first,
            InputIterator last,
            T init,
            BinaryFunction binary_op)  ;

        /*!   \}  */

    }
}

#include <bolt/btbb/detail/reduce.inl>


#endif //BTBB_REDUCE_H