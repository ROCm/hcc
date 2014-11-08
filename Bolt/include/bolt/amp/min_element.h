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

#if !defined( BOLT_AMP_MIN_ELEMENT_H )
#define BOLT_AMP_MIN_ELEMENT_H
#pragma once

#include <amp.h>
#include "bolt/amp/functional.h"

#include "bolt/amp/bolt.h"
#include <string>
#include <assert.h>

/*! \file bolt/amp/min_element.h
    \brief min_element returns the location of the first minimum element in the specified range.
*/


namespace bolt {
    namespace amp {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup reductions
        *   \ingroup algorithms
        *    The min_element finds the location of the first smallest element in the range [first, last]
        */

        /*! \addtogroup AMP-min_element
        *   \ingroup reductions
        *   \{
        */

        /*! \brief The min_element returns the location of the first minimum element in the specified range.
        *
        * \param ctl \b Optional Control structure to control accelerator, debug, tuning, etc.See bolt::amp::control.
        * \param first A forward iterator addressing the position of the first element in the range to be searched for
        *  the minimum element
        * \param last  A forward iterator addressing the position one past the final element in the range to be
        *  searched for the minimum element
        * \tparam ForwardIterator An iterator that can be dereferenced for an object, and can be incremented to get to
        * the next element in a sequence.
        * \return The position of the min_element.
        * \details The following code example shows how to find the position of the  \p min_element of 10 numbers,
        * using the default BinaryPredicate.
        * \code
        * #include <bolt/amp/min_element.h>
        *
        * int a[10] = {4, 8, 6, 1, 5, 3, 10, 2, 9, 7};
        *
        * int min_pos = bolt::amp::min_element(a, a+10);
        * // min_pos = 3
        *  \endcode
        * \sa http://www.sgi.com/tech/stl/min_element.html
        */

        template<typename ForwardIterator>
        ForwardIterator  min_element(bolt::amp::control &ctl,
            ForwardIterator first,
            ForwardIterator last);

        template<typename ForwardIterator>
        ForwardIterator min_element(ForwardIterator first,
            ForwardIterator last);


        /*! \brief The min_element returns the location of the first minimum element in the specified range using the
        * specified binary_op.
        *
        * \param ctl \b Optional Control structure to control accelerator, debug, tuning, etc.See bolt::amp::control.
        * \param first A forward iterator addressing the position of the first element in the range to be searched for
        * the minimum element
        * \param last  A forward iterator addressing the position one past the final element in the range to be
        * searched for the minimum element
        * \param binary_op  The binary operation used to combine two values.   By default, the binary operation is
        * less<>().
        * \tparam ForwardIterator An iterator that can be dereferenced for an object, and can be incremented to get to
        * the next element in a sequence.
        * \tparam BinaryPredicate A function object defining an operation that is applied to consecutive elements in
        * the sequence.
        * \return The position of the min_element.
        *
        *
        * \details The following code example shows how to find the position of the  \p min_element of 10 numbers,
        * using the default less operator.
        * \code
        * #include <bolt/amp/min_element.h>
        *
        * int a[10] = {4, 8, 6, 1, 5, 3, 10, 2, 9, 7};
        *
        * int min_pos = bolt::amp::min_element(a, a+10, bolt::amp::less<T>());
        * // min_pos = 3
        *  \endcode
        * \sa http://www.sgi.com/tech/stl/min_element.html
        */

        template<typename ForwardIterator, typename BinaryPredicate>
        ForwardIterator min_element(bolt::amp::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op);

        template<typename ForwardIterator, typename BinaryPredicate>
        ForwardIterator min_element(ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op);


    };
};

#include <bolt/amp/detail/min_element.inl>
#endif