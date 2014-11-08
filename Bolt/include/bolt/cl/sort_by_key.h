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

#if !defined( BOLT_CL_SORT_BY_KEY_H )
#define BOLT_CL_SORT_BY_KEY_H
#pragma once

#include "bolt/cl/device_vector.h"
#include "bolt/cl/functional.h"

/*! \file bolt/cl/sort_by_key.h
    \brief Returns the sorted result of all the elements in input based on equivalent keys.
*/

namespace bolt {
    namespace cl {


        /*! \addtogroup algorithms
         */

        /*! \addtogroup sorting
        *   \ingroup algorithms
        *   An Algorithm for sorting the given InputIterator.
        *   It is capable of sorting the arithmetic data types, or the user-defined data types. For common code between
        *   the host and device, take a look at the ClCode and TypeName implementations. See the Bolt Tools for
        *   Split-Source
        *   for a detailed description.
        */

        /*! \addtogroup CL-sort_by_key
        *   \ingroup sorting
        *   \{
        */

        /*! \brief This version of \p sort_by_key  returns the sorted result of all the elements in the
        * \p RandomAccessIterator between the the first and last elements key elements and corresponding values. The
        * routine arranges the elements in an ascending order.
        *
        *
        * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc. See bolt::cl::control.
        * \param keys_first The first position in the sequence to be sorted.
        * \param keys_last  The last position in the sequence to be sorted.
        * \param values_first  The  first position in the value sequence.
        * \param cl_code Optional OpenCL(TM) code to be passed to the OpenCL compiler. The cl_code is inserted first in
        * the generated code, before the cl_code traits. This can be used for any extra cl code that is to be passed
        * when compiling the OpenCl Kernel.
        * \return The sorted key,value pair that is available in place.
        *
        *
        *  \tparam RandomAccessIterator1 Is a model of http://www.sgi.com/tech/stl/RandomAccessIterator.html
        *  \tparam RandomAccessIterator2  Is a model of http://www.sgi.com/tech/stl/RandomAccessIterator.html
        *
        *
        * \details The following code example shows the use of \p sort_by_key to sort the key,value pair in the
        * ascending order.
        * \code
        * #include <bolt/cl/sort_by_key.h>
        *
        * int keys[8] = {2, 9, 3, 7, 5, 6, 3, 8};
        * int values[8] = {100, 200, 16, 50, 15, 8, 3, 5};
        *
        * bolt::cl::sort_by_key(keys, keys+8,values);
        *  //Output
        * //keys[8] =   {2,3,3,5,6,7,8,9}
        * //values[8] = {100,16,3,15,8,50,5,200}
        *
        *  \endcode
        */
        template<typename RandomAccessIterator1 , typename RandomAccessIterator2>
        void sort_by_key(bolt::cl::control &ctl,
                         RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         const std::string& cl_code="");

        template<typename RandomAccessIterator1 , typename RandomAccessIterator2>
        void sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         const std::string& cl_code="");

        /*! \brief This version of \p sort_by_key  returns the sorted result of all the elements in the
        * \p RandomAccessIterator between the the first and last elements key elements and corresponding values. The
        * routine arranges the elements in an ascending order.
        * \details This routine uses function object comp to compare the key objects.
        *
        * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc. See bolt::cl::control.
        * \param keys_first The first position in the sequence to be sorted.
        * \param keys_last  The last position in the sequence to be sorted.
        * \param values_first  The  first position in the value sequence.
        * \param comp  The comparison operation used to compare two values.
        * \param cl_code Optional OpenCL(TM) code to be passed to the OpenCL compiler. The cl_code is inserted first in
        * the generated code, before the cl_code traits. This can be used for any extra cl code that is to be passed
        * when compiling the OpenCl Kernel.
        * \return The sorted key,value pair that is available in place.
        *
        *  \tparam RandomAccessIterator1 Is a model of http://www.sgi.com/tech/stl/RandomAccessIterator.html
        *  \tparam RandomAccessIterator2  Is a model of http://www.sgi.com/tech/stl/RandomAccessIterator.html
        *  \tparam StrictWeakOrdering Is a model of http://www.sgi.com/tech/stl/StrictWeakOrdering.html.
        *
        * \details The following code example shows the use of \p sort_by_key to sort the key,value pair in the
        * ascending order.
        * \code
        * #include <bolt/cl/sort_by_key.h>
        *
        * int keys[8] = {2, 9, 3, 7, 5, 6, 3, 8};
        * int values[8] = {100, 200, 16, 50, 15, 8, 3, 5};
        *
        * // for arranging the elements in descending order, use bolt::cl::greater<int>()
        * bolt::cl::sort_by_key(keys, keys+8,values,bolt::cl::less<int>());
        *  //Output
        * //keys[8] =   {9,8,7,6,5,3,3,2}
        * //values[8] = {200,5,50,8,15,3,16,100}
        *
        *  \endcode
        */

        template<typename RandomAccessIterator1 , typename RandomAccessIterator2 , typename StrictWeakOrdering>
        void sort_by_key(bolt::cl::control &ctl,
                  RandomAccessIterator1 keys_first,
                  RandomAccessIterator1 keys_last,
                  RandomAccessIterator2 values_first,
                  StrictWeakOrdering comp,
                  const std::string& cl_code="");


        template<typename RandomAccessIterator1 , typename RandomAccessIterator2 , typename StrictWeakOrdering>
        void sort_by_key(RandomAccessIterator1 keys_first,
                  RandomAccessIterator1 keys_last,
                  RandomAccessIterator2 values_first,
                  StrictWeakOrdering comp,
                  const std::string& cl_code="");



        /*!   \}  */

    }// end of bolt::cl namespace
}// end of bolt namespace

#include <bolt/cl/detail/sort_by_key.inl>
#endif