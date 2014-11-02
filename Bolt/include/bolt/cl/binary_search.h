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

#if !defined( BOLT_CL_BINARY_SEARCH_H )
#define BOLT_CL_BINARY_SEARCH_H
#pragma once

#include "bolt/cl/device_vector.h"
#include "bolt/cl/functional.h"
#include <string>

/*! \file bolt/cl/binary_search.h
    \brief Returns true if the search element is found in the given input range and false otherwise.
*/

namespace bolt {
    namespace cl {
      
	    /*! \addtogroup algorithms
        */

        /*! \addtogroup searching
        *   \ingroup algorithms
        *   An Algorithm for performing binary search of a specified value on the given InputIterator.
        *   It is capable of searching the arithmetic data types, or the user-defined data types. 
        */

        /*! \addtogroup CL-search
        *   \ingroup searching
        *   \{
        */

        /*! \brief This version of binary search returns true if the search value is present within the given input range
        * and false otherwise. The routine requires the elements to be arranged in ascending/descending order. 
		* It returns true iff there exists an iterator i in [first, last) such that *i < value and value < *i are both false.
        *
        * \details The \p binary_search operation is analogus to the std::binary_search function.
        * 
        *
        * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc. See bolt::cl::control.
        * \param first The first position in the sequence to search.
        * \param last  The last position in the sequence to search.
		* \param value The value to search.
        * \param cl_code Optional OpenCL(TM) code to be passed to the OpenCL compiler. The cl_code is inserted first
        * in the generated code, before the cl_code traits.
        * This can be used for any extra cl code that is to be passed when compiling the OpenCl Kernel.
		* \tparam ForwardIterator An iterator that can be dereferenced for an object, and can be incremented to get to the next element in a sequence. 
        * \tparam T The type of the search element. 
        * \return boolean value - true if search value present and false otherwise.
        *
        * \details The following code example shows the use of \p binary_search on the elements in the ascending order.
        * \code
        * #include <bolt/cl/binary_search.h>
		* #include <bolt/cl/sort.h>
        *
        * int a[8] = {2, 9, 3, 7, 5, 6, 3, 8};
        *
        * // for arranging the elements in ascending order
        * bolt::cl::sort( a, a+8 );
		* int val = a[2];
		* bolt::cl::binary_search( a, a+8 , val);
        *
        * \endcode
		* \sa http://www.sgi.com/tech/stl/binary_search.html
        */
		
        template<typename ForwardIterator, typename T>
        bool binary_search(bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const T & value,
            const std::string& cl_code="");

        template<typename ForwardIterator, typename T>
        bool binary_search(ForwardIterator first,
            ForwardIterator last,
            const T & value,
            const std::string& cl_code="");

        /*! \brief This version of binary search returns true if the search value is present within the given input range
        * and false otherwise. The routine requires the elements to be arranged in ascending/descending order.
		* It returns true iff there exists an iterator i in [first, last) such that comp(*i, value) and comp(value, *i) are both false.
        *
        * \details The \p binary_search operation is analogus to the std::binary_search function.
        *
        * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc. See bolt::cl::control.
        * \param first The first position in the sequence to search.
        * \param last  The last position in the sequence to search.
		* \param value The value to search.
		* \param comp  The comparison operation used to compare two values.
        * \param cl_code Optional OpenCL(TM) code to be passed to the OpenCL compiler. The cl_code is inserted first
        * in the generated code, before the cl_code traits.
        * This can be used for any extra cl code that is to be passed when compiling the OpenCl Kernel.
		* \tparam ForwardIterator An iterator that can be dereferenced for an object, and can be incremented to get to the next element in a sequence. 
        * \tparam T The type of the search element. 
        * \return boolean value - true if search value present and false otherwise.
        *
        * \details The following code example shows the use of \p binary_search on the elements in the descending order.
        * \code
        * #include <bolt/cl/binary_search.h>
		* #include <bolt/cl/sort.h>
        *
        * int a[8] = {2, 9, 3, 7, 5, 6, 3, 8};
        *
        * // for arranging the elements in descending order
        * bolt::cl::sort( a, a+8, bolt::cl::greater<int>());
		* int val = a[2];
		* bolt::cl::binary_search( a, a+8 , val, bolt::cl::greater<int>());
        *
        * \endcode
		* \sa http://www.sgi.com/tech/stl/binary_search.html
        */
		
        template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
        bool binary_search(bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const T & value,
            StrictWeakOrdering comp,
            const std::string& cl_code="");


        template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
        bool binary_search(ForwardIterator first,
            ForwardIterator last,
            const T & value,
            StrictWeakOrdering comp,
            const std::string& cl_code="");

    }// end of bolt::cl namespace
}// end of bolt namespace

#include <bolt/cl/detail/binary_search.inl>
#endif
