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

/*! \file bolt/amp/sort.h
    \brief  Returns the sorted result of all the elements in input.
*/


#if !defined( BOLT_AMP_SORT_H )
#define BOLT_AMP_SORT_H
#pragma once

#include <bolt/amp/bolt.h>
#include <bolt/amp/functional.h>
#include <string>
#include <iostream>


/*! \file sort.h
*/

namespace bolt {
    namespace amp {


        /*! \addtogroup algorithms
         */

        /*! \addtogroup sorting
        *   \ingroup algorithms
        *   An Algorithm for sorting the given InputIterator.
        *   It is capable of sorting the arithmetic data types, or the user-defined data types. For common code between
        *   the hostand device, take a look at the ClCode and TypeName implementations.
        *   See the Bolt Tools for Split-Source for a detailed description.
        */

        /*! \addtogroup AMP-sort
        *   \ingroup sorting
        *   \{
        */

        /*! \breif This version of \p sort returns the sorted result of all the elements in the \p RandomAccessIterator
        * between the the first and last elements.
        * The routine arranges the elements in an ascending order. \p RandomAccessIterator's value_type must provide
        * operator "<" overload.

        *
        * The \p sort operation is analogus to the std::sort function.  See http://www.sgi.com/tech/stl/sort.html
        *  \tparam RandomAccessIterator Is a model of http://www.sgi.com/tech/stl/RandomAccessIterator.html, \n
        *          \p RandomAccessIterator is mutable, \n
        *          \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's \n
        *          \p RandomAccessIterator's \c value_type is \p LessThanComparable
        * http://www.sgi.com/tech/stl/LessThanComparable.html; i.e., the value _type must provide operator
        * '<' overloaded. \n
        *
        * \param ctl \b Optional Bolt control object, to describe the environment under which the function will run.
        * \param first The first position in the sequence to be sorted.
        * \param last  The last position in the sequence to be sorted.
        * \return The sorted data that is available in place.
        *
        * \details The following code example shows the use of \p sort to sort the elements in the ascending order,
        * specifying a specific accelerator.
        * \code
        * #include <bolt/amp/sort.h>
        *
        * int a[8] = {2, 9, 3, 7, 5, 6, 3, 8};
        *
        * //Create an AMP Control object using the default accelerator
        * ::Concurrency::accelerator accel(::Concurrency::accelerator::default_accelerator);
        *
        * bolt::amp::control ctl(accel);
        * bolt::amp::sort(ctl, a, a+8);
        *
        *  \endcode
        */
        template<typename RandomAccessIterator>
        void sort(bolt::amp::control &ctl,
            RandomAccessIterator first,
            RandomAccessIterator last);

        template<typename RandomAccessIterator>
        void sort(RandomAccessIterator first,
            RandomAccessIterator last);

        /*! \breif  \p sort returns the sorted result of all the elements in the inputIterator between the the first
        * and last elements using the specified binary_op.
        * You can arrange the elements in an ascending order, where the binary_op is the less<>() operator.
        * This version of \p sort takes a bolt::amp::control structure as a first argument and compares objects using
        * \c functor object defined by \p StrictWeakOrdering.

        *
        * The \p sort operation is analogus to the std::sort function.  See http://www.sgi.com/tech/stl/sort.html.
        *  \tparam RandomAccessIterator Is a model of http://www.sgi.com/tech/stl/RandomAccessIterator.html, \n
        *          \p RandomAccessIterator is mutable, \n
        *          \p RandomAccessIterator's \c value_type is convertible to \p StrictWeakOrdering's \n
        *          \p RandomAccessIterator's \c value_type is
        * \p LessThanComparable http://www.sgi.com/tech/stl/LessThanComparable.html i.e the value _type should provide
        * operator '<' overloaded. \n
        *  \tparam StrictWeakOrdering Is a model of http://www.sgi.com/tech/stl/StrictWeakOrdering.html. \n

        * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc. See bolt::amp::control.
        * \param first The first position in the sequence to be sorted.
        * \param last  The last position in the sequence to be sorted.
        * \param comp  The comparison operation used to compare two values.
        * \return The sorted data that is available in place.
        *
        * \details The following code example shows the use of \p sort to sort the elements in the descending order,
        * specifying a specific accelerator.
        * \code
        * #include <bolt/amp/sort.h>
        * #include <bolt/amp/functional.h>
        *
        * int a[8] = {2, 9, 3, 7, 5, 6, 3, 8};
        *
        * //Create an AMP Control object using the default accelerator
        * ::Concurrency::accelerator accel(::Concurrency::accelerator::default_accelerator);
        *
        * bolt::amp::control ctl(accel);
        *
        * // for arranging the elements in descending order, use bolt::amp::greater<int>()
        * bolt::amp::sort(ctl, a, a+8, bolt::amp::greater<int>());
        *
        *  \endcode
        */

        template<typename RandomAccessIterator, typename StrictWeakOrdering>
        void sort(bolt::amp::control &ctl,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp);


        template<typename RandomAccessIterator, typename StrictWeakOrdering>
        void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp);


        /*!   \}  */

    }// end of bolt::amp namespace
}// end of bolt namespace

#include <bolt/amp/detail/sort.inl>
#endif