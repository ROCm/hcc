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

#if !defined(BOLT_BTBB_SORT_H )
#define BOLT_BTBB_SORT_H
#pragma once

#include "tbb/parallel_sort.h"
#include "tbb/task_scheduler_init.h"



/*! \file bolt/btbb/sort.h
    \brief Returns the sorted result of all the elements in input.
*/


namespace bolt {
    namespace btbb {


        /*! \addtogroup algorithms
         */

        /*! \addtogroup sorting
        *   \ingroup algorithms
        *   An Algorithm for sorting the given InputIterator.
        *   It is capable of sorting the arithmetic data types, or the user-defined data types. For common code between
        *   the hostand device, take a look at the ClCode and TypeName implementations.
        *   See the Bolt Tools for Split-Source for a detailed description.
        */

        /*! \addtogroup CL-sort
        *   \ingroup sorting
        *   \{
        */

        /*! \brief This version of \p sort returns the sorted result of all the elements in the \p RandomAccessIterator
        * between the the first and last elements. The routine arranges the elements in an ascending order.
        * \p RandomAccessIterator's value_type must provide operator < overload.

        *
        * \details The \p sort operation is analogus to the std::sort function.
        * See http://www.sgi.com/tech/stl/sort.html
        *  \tparam RandomAccessIterator Is a model of http://www.sgi.com/tech/stl/RandomAccessIterator.html, \n
        *          \p RandomAccessIterator is mutable, \n
        *          \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's \n
        *          \p RandomAccessIterator's \c value_type is \p
        * LessThanComparable http://www.sgi.com/tech/stl/LessThanComparable.html; i.e., the value _type must provide
        * operator < overloaded. \n

        * \param first The first position in the sequence to be sorted.
        * \param last  The last position in the sequence to be sorted.
        * \return The sorted data that is available in place.
        *
        * \details The following code example shows the use of \p sort to sort the elements in the ascending order,
        * specifying a specific command-queue.
        * \code
        * #include <bolt/btbb/sort.h>
        *
        * int a[8] = {2, 9, 3, 7, 5, 6, 3, 8};
        *
        * // for arranging the elements in descending order, greater<int>()
        * bolt::cl::sort(ctl, a, a+8 );
        *
        *  \endcode
        */

        template<typename RandomAccessIterator>
        void sort(RandomAccessIterator first,
            RandomAccessIterator last);

        /*! \brief \p sort returns the sorted result of all the elements in the inputIterator between the the first and
        * last elements using the specified binary_op. You can arrange the elements in an ascending order, where the
        * binary_op is the less<>() operator. This version of \p sort takes a
        * \c functor object defined by \p StrictWeakOrdering.

        *
        *\details  The \p sort operation is analogus to the std::sort function.
        * See http://www.sgi.com/tech/stl/sort.html.
        *  \tparam RandomAccessIterator Is a model of http://www.sgi.com/tech/stl/RandomAccessIterator.html, \n
        *          \p RandomAccessIterator is mutable, \n
        *          \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's \n
        *          \p RandomAccessIterator's \c value_type is
        * \p LessThanComparable http://www.sgi.com/tech/stl/LessThanComparable.html i.e the value _type should provide
        * operator < overloaded. \n
        *  \tparam StrictWeakOrdering Is a model of http://www.sgi.com/tech/stl/StrictWeakOrdering.html. \n

        * \param first The first position in the sequence to be sorted.
        * \param last  The last position in the sequence to be sorted.
        * \param comp  The comparison operation used to compare two values.
        * \return The sorted data that is available in place.
        *
        * \details The following code example shows the use of \p sort to sort the elements in the descending order.
        * \code
        * #include <bolt/btbb/sort.h>
        *
        * int a[8] = {2, 9, 3, 7, 5, 6, 3, 8};
        *
        * bolt::btbb::sort(a, a+8, greater<int>());
        *
        *  \endcode
        */

        template<typename RandomAccessIterator, typename StrictWeakOrdering>
        void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp);


        /*!   \}  */

    }// end of bolt::btbb namespace
}// end of bolt namespace



#include <bolt/btbb/detail/sort.inl>

#endif