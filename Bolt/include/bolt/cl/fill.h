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

#if !defined( BOLT_CL_FILL_H )
#define BOLT_CL_FILL_H
#pragma once

#include "bolt/cl/device_vector.h"

/*! \file bolt/cl/fill.h
    \brief Fills a range with values passed in the function.
*/


namespace bolt {
    namespace cl {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup transformations
        *   \ingroup algorithms
        *   \p Fill fills a range with values passed in the function.
        */

        /*! \addtogroup CL-filling
        *   \ingroup transformations
        *   \{
        */

        /*! \brief  Fill assigns the value of \c value to each element in the range [first,last].
         *
		     *  \param ctl      \b Optional control structure to control command-queue, debug, tuning, etc.
         *                  See bolt::cl::control.
         *  \param first    The first element in the range of interest.
         *  \param last     The last element in the range of interest.
         *  \param value    Sets this value to elements in the range [first,last].
         *  \param cl_code  Optional OpenCL(TM) code to be prepended to any OpenCL kernels used by this function.
         *
         *  \tparam ForwardIterator is a model of \c Forward Iterator, and \c InputIterator is mutable.
         *  \tparam T is a model of Assignable.
		     *
         * \details The following code snippet demonstrates how to fill a device_vector with a float value.
         *
         *  \code
         *  #include <bolt/cl/fill.h>
         *  #include <bolt/cl/device_vector.h>
         *  #include <stdlib.h>
         *  ...
         *
         *  bolt::cl::device_vector<float> v(10);
         *
         *  float x=25.0f;
         *  bolt::cl::fill(v.begin(), v.end(), x);
         *
         *  // the elements of v are now assigned to the float value.
         *  \endcode
         *
         *  \sa http://www.sgi.com/tech/stl/fill.html
         */

        template<typename ForwardIterator, typename T>
        void fill( const bolt::cl::control &ctl, ForwardIterator first, ForwardIterator last, const T & value,
            const std::string& cl_code="");

        template<typename ForwardIterator, typename T>
        void fill( ForwardIterator first, ForwardIterator last, const T & value, const std::string& cl_code="");

        /*! \brief fill_n assigns the value \c value to every element in the range [first,first+n].
         *  The return value is first + n.
         *
         *  \param ctl      \b Optional control structure to control command-queue, debug, tuning, etc.
         *                  See bolt::cl::control.
         *  \param first    The first element in the range of interest.
         *  \param n        The size of the range of interest.
         *  \param value    Sets this value to elements in the range [first,first+n].
         *  \param cl_code  Optional OpenCL(TM) code to be prepended to any OpenCL kernels used by this function.
         *
         *  \tparam OutputIterator	is a model of Output Iterator
         *  \tparam Size            is an integral type (either signed or unsigned).
         *  \tparam T				is a model of \c Assignable
         *
         *  \return first+n.
         *
         *  \details The following code snippet demonstrates how to fill a device_vector with a float value.
         *
         *  \code
         *  #include <bolt/cl/fill.h>
         *  #include <bolt/cl/device_vector.h>
         *  #include <stdlib.h>
         *
         *  ...
         *
         *  bolt::cl::device_vector<float> v(10);
         *
         *  float x=25.0f;
         *  bolt::cl::fill_n(v.begin(), 10, x);
         *
         *  // the elements of v are now assigned to the float value.
         *  \endcode
         *
         *  \sa http://www.sgi.com/tech/stl/fill_n.html
         */

        template<typename OutputIterator, typename Size, typename T>
        OutputIterator fill_n( const bolt::cl::control &ctl, OutputIterator first, Size n, const T & value,
            const std::string& cl_code="");

        template<typename OutputIterator, typename Size, typename T>
        OutputIterator fill_n( OutputIterator first, Size n, const T & value, const std::string& cl_code="");

        /*!   \}  */
    };
};

#include <bolt/cl/detail/fill.inl>
#endif
