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

#if !defined( BOLT_CL_COPY_H )
#define BOLT_CL_COPY_H
#pragma once

#include "bolt/cl/device_vector.h"

#include <string>
#include <iostream>

/*! \file bolt/cl/copy.h
    \brief Copies each element from the sequence to result.
*/

namespace bolt {
    namespace cl {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup copying
        *   \ingroup algorithms
        *   \p copy copies each element from the sequence [first, last) to [result, result + (last - first)).
        */

        /*! \addtogroup CL-copy
        *   \ingroup copying
        *   \{
        */

        /*! copy copies each element from the sequence [first, last) to [result, result + (last - first)), i.e.,
         *  it assigns *result = *first, then *(result + 1) = *(first + 1), and so on.
         *
         *  Calling copy with overlapping source and destination ranges has undefined behavior, as the order
         *  of copying on the GPU is not guaranteed.
         *
         * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc.See bolt::cl::control.
         * \param first Beginning of the source copy sequence.
         * \param last  End of the source copy sequence.
         * \param result Beginning of the destination sequence.
         * \param user_code  Optional OpenCL(TM) code to be prepended to any OpenCL kernels used by this function.
         * \return result + (last - first).
         *
         * \tparam InputIterator is a model of InputIterator
         * and \c InputIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
         * \tparam OutputIterator is a model of OutputIterator
         *
         *  \details The following demonstrates how to use \p copy.
         *
         *  \code
         *  #include <bolt/cl/copy.h>
         *  ...
         *
         *  std::vector<float> vecSrc(128);
         *  std::vector<float> vecDest(128);
         *  bolt::cl::control ctrl = control::getDefault();
         *  ...
         *
         *  bolt::cl::copy(ctrl, vecSrc.begin(), vecSrc.end(), vecDest.begin());
         *
         *  // vecDest is now a copy of vecSrc
         *  \endcode
         *
         *  \sa http://www.sgi.com/tech/stl/copy.html
         *  \sa http://www.sgi.com/tech/stl/InputIterator.html
         *  \sa http://www.sgi.com/tech/stl/OutputIterator.html
         */
        template<typename InputIterator, typename OutputIterator>
        OutputIterator copy(
            const bolt::cl::control &ctl,
            InputIterator first,
            InputIterator last,
            OutputIterator result,
            const std::string& user_code="");

        template<typename InputIterator, typename OutputIterator>
        OutputIterator copy(
            InputIterator first,
            InputIterator last,
            OutputIterator result,
            const std::string& user_code="");

        /*! copy_n copies each element from the sequence [first, first+n) to [result, result + n), i.e.,
         *  it assigns *result = *first, then *(result + 1) = *(first + 1), and so on.
         *
         *  Calling copy_n with overlapping source and destination ranges has undefined behavior, as the order
         *  of copying on the GPU is not guaranteed.
         *

         * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc.See bolt::cl::control.
         *  \param first Beginning of the source copy sequence.
         *  \param n  Number of elements to copy.
         *  \param result Beginning of the destination sequence.
         * \param user_code Optional OpenCL&tm; code to be passed to the OpenCL compiler. The cl_code is inserted
         *   first in the generated code, before the cl_code trait.
         *  \return result + n.
         *
         *  \tparam InputIterator is a model of InputIterator
         *  and \c InputIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
         *  \tparam Size is an integral type.
         *  \tparam OutputIterator is a model of OutputIterator
         *
         * \details The following demonstrates how to use \p copy.
         *
         *  \code
         *  #include <bolt/cl/copy.h>
         *  ...
         *
         *  std::vector<float> vecSrc(128);
         *  std::vector<float> vecDest(128);
         *  ...
         *
         *  bolt::cl::copy_n(vecSrc.begin(), 128, vecDest.begin());
         *
         *  // vecDest is now a copy of vecSrc
         *  \endcode
         *
         *  \sa http://www.sgi.com/tech/stl/copy_n.html
         *  \sa http://www.sgi.com/tech/stl/InputIterator.html
         *  \sa http://www.sgi.com/tech/stl/OutputIterator.html
         */

        template<typename InputIterator, typename Size, typename OutputIterator>
        OutputIterator copy_n(
            const bolt::cl::control &ctl,
            InputIterator first,
            Size n,
            OutputIterator result,
            const std::string& user_code="");

        template<typename InputIterator, typename Size, typename OutputIterator>
        OutputIterator copy_n(
            InputIterator first,
            Size n,
            OutputIterator result,
            const std::string& user_code="");

        /*!   \}  */
    };
};

#include <bolt/cl/detail/copy.inl>
#endif