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

#if !defined( BOLT_AMP_COPY_H )
#define BOLT_AMP_COPY_H
#pragma once

#include <bolt/amp/bolt.h>
#include <bolt/amp/device_vector.h>

#include <string>
#include <iostream>

/*! \file bolt/amp/copy.h
    \brief Copies each element from the sequence to result.
*/

namespace bolt {
    namespace amp {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup copying
        *   \ingroup algorithms
        *   \p copy copies each element from the sequence [first, last) to [result, result + (last - first)).
        */

        /*! \addtogroup AMP-copy
        *   \ingroup copying
        *   \{
        */

        /*! copy copies each element from the sequence [first, last) to [result, result + (last - first)), i.e.,
         *  it assigns *result = *first, then *(result + 1) = *(first + 1), and so on.
         *
         *  Calling copy with overlapping source and destination ranges has undefined behavior, as the order
         *  of copying on the GPU is not guaranteed.
         *
         * \param ctl \b Optional Control structure to control accelerator, debug, tuning, etc.See bolt::amp::control.
         * \param first Beginning of the source copy sequence.
         * \param last  End of the source copy sequence.
         * \param result Beginning of the destination sequence.
         * \return result + (last - first).
         *
         * \tparam InputIterator is a model of InputIterator
         * and \c InputIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
         * \tparam OutputIterator is a model of OutputIterator
         *
         *  \details The following demonstrates how to use \p copy.
         *
         *  \code
         *  #include <bolt/amp/copy.h>
         *  ...
         *
         *  std::vector<float> vecSrc(128);
         *  std::vector<float> vecDest(128);
         *  bolt::amp::control ctrl = control::getDefault();
         *  ...
         *
         *  bolt::amp::copy(ctrl, vecSrc.begin(), vecSrc.end(), vecDest.begin());
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
            const bolt::amp::control &ctl,
            InputIterator first,
            InputIterator last,
            OutputIterator result);

        template<typename InputIterator, typename OutputIterator>
        OutputIterator copy(
            InputIterator first,
            InputIterator last,
            OutputIterator result);

        /*! copy_n copies each element from the sequence [first, first+n) to [result, result + n), i.e.,
         *  it assigns *result = *first, then *(result + 1) = *(first + 1), and so on.
         *
         *  Calling copy_n with overlapping source and destination ranges has undefined behavior, as the order
         *  of copying on the GPU is not guaranteed.
         *

         * \param ctl \b Optional Control structure to control accelerator, debug, tuning, etc.See bolt::amp::control.
         *  \param first Beginning of the source copy sequence.
         *  \param n  Number of elements to copy.
         *  \param result Beginning of the destination sequence.
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
         *  #include <bolt/amp/copy.h>
         *  ...
         *
         *  std::vector<float> vecSrc(128);
         *  std::vector<float> vecDest(128);
         *  ...
         *
         *  bolt::amp::copy_n(vecSrc.begin(), 128, vecDest.begin());
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
            const bolt::amp::control &ctl,
            InputIterator first,
            Size n,
            OutputIterator result);

        template<typename InputIterator, typename Size, typename OutputIterator>
        OutputIterator copy_n(
            InputIterator first,
            Size n,
            OutputIterator result);

        /*!   \}  */
    };
};

#include <bolt/amp/detail/copy.inl>
#endif