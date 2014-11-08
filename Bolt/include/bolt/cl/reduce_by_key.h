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

#if !defined( BOLT_CL_REDUCE_BY_KEY_H )
#define BOLT_CL_REDUCE_BY_KEY_H
#pragma once

#include "bolt/cl/device_vector.h"
#include "bolt/cl/pair.h"

/*! \file bolt/cl/reduce_by_key.h
    \brief Performs on a sequence, a reduction of each sub-sequence as defined by equivalent keys.
*/
namespace bolt
{
     namespace cl
        {

            /*! \addtogroup algorithms
            */

            /*! \addtogroup reductions
            *   \ingroup algorithms
            *   reduce_by_key performs, on a sequence, a reduction of each sub-sequence as defined by equivalent keys.
            */

            /*! \addtogroup CL-reduce_by_key
            *   \ingroup reductions
            *   \{
            */


            /*! \brief \p reduce_by_key performs, on a sequence,
            * a reduction of each sub-sequence as defined by equivalent keys;
            * the BinaryFunction in this version is plus(), and the BinaryPredicate is equal_to().
            *
            * \param ctl           \b Optional Control structure to control command-queue, debug, tuning, etc.
            *                      See bolt::cl::control.
            * \param keys_first    The first element of the key sequence.
            * \param keys_last     The last  element of the key sequence.
            * \param values_first  The first element of the value sequence.
            * \param keys_output   The first element of the key output sequence.
            * \param values_output The first element of the value output sequence.
            * \param user_code     A user-specified Optional string that is preppended to the generated OpenCL kernel.
            *
            * \tparam InputIterator1   is a model of Input Iterator.
            * \tparam InputIterator2   is a model of Input Iterator.
            * \tparam OutputIterator   is a model of Output Iterator.
            *
            * \return result+(last1-first1).
            *
            * \details Example:
            * \code
            * #include "bolt/cl/reduce_by_key.h"
            * ...
            *
            * int keys[11] = { 0, 0, 0, 2, 2, 2, -5, -5, -5, -5, 6 };
            * int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 };
            * int keys_out[11];
            * int vals_out[11];
            *
            * bolt::cl::control ctrl = control::getDefault();
            *
            * bolt::cl::reduce_by_key( ctl, keys, keys+11, vals, keys_out, vals_out );
            * // keys_out => { 0, 2, -5, 6 }
            * // vals_out => { 6, 6, 8, 2 }
            *  \endcode
            *
            */
            template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2>
                pair<OutputIterator1, OutputIterator2>
                reduce_by_key(
                control &ctl,
                InputIterator1  keys_first,
                InputIterator1  keys_last,
                InputIterator2  values_first,
                OutputIterator1  keys_output,
                OutputIterator2  values_output,
                const std::string& user_code="" );

            template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2>
                pair<OutputIterator1, OutputIterator2>
                reduce_by_key(
                InputIterator1  keys_first,
                InputIterator1  keys_last,
                InputIterator2  values_first,
                OutputIterator1  keys_output,
                OutputIterator2  values_output,
                const std::string& user_code="" );

            /*! \brief \p reduce_by_key performs, on a sequence,
            * a reduction of each sub-sequence as defined by equivalent keys;
            * the BinaryFunction in this version is plus().
			* reduce_by_key is a generalization of reduce to key-value pairs.
			* If the reduction operator is not commutative then bolt::reduce_by_key should not be used. 
            *
            * \param ctl           \b Optional Control structure to control command-queue, debug, tuning, etc.
            *                       See bolt::cl::control.
            * \param keys_first    The first element of the key sequence.
            * \param keys_last     The last  element of the key sequence.
            * \param values_first  The first element of the value sequence.
            * \param keys_output   The first element of the key output sequence.
            * \param values_output The first element of the value output sequence.
            * \param binary_pred   Binary predicate which determines if two keys are equal.
            * \param user_code     A user-specified Optional string that is preppended to the generated OpenCL kernel.
            *
            * \tparam InputIterator1   is a model of Input Iterator.
            * \tparam InputIterator2   is a model of Input Iterator.
            * \tparam OutputIterator   is a model of Output Iterator.
            * \tparam BinaryPredicate  is a model of Binary Predicate.
            *
            * \return pair(last of keys output, last of value output)
            *
            * \details Example:
            * \code
            * #include "bolt/cl/reduce_by_key.h"
            * ...
            *
            * int keys[11] = { 0, 0, 0, 2, 2, 2, -5, -5, -5, -5, 6 };
            * int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 };
            * int keys_out[11];
            * int vals_out[11];
            *
            * bolt::cl::equal_to<int> eq;
            * bolt::cl::control ctrl = control::getDefault();
            *
            * bolt::cl::reduce_by_key( ctl, keys, keys+11, vals, keys_out, vals_out, eq );
            * // keys_out => { 0, 2, -5, 6 }
            * // vals_out => { 6, 6, 8, 2 }
            *  \endcode
            *
            */
            template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2,
                typename BinaryPredicate>
                pair<OutputIterator1, OutputIterator2>
                reduce_by_key(
                control &ctl,
                InputIterator1  keys_first,
                InputIterator1  keys_last,
                InputIterator2  values_first,
                OutputIterator1  keys_output,
                OutputIterator2  values_output,
                BinaryPredicate binary_pred,
                const std::string& user_code="" );

            template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2,
                typename BinaryPredicate>
                pair<OutputIterator1, OutputIterator2>
                reduce_by_key(
                InputIterator1  keys_first,
                InputIterator1  keys_last,
                InputIterator2  values_first,
                OutputIterator1  keys_output,
                OutputIterator2  values_output,
                BinaryPredicate binary_pred,
                const std::string& user_code="" );


           /*! \brief \p reduce_by_key performs, on a sequence,
            * a reduction of each sub-sequence as defined by equivalent keys;
			* reduce_by_key is a generalization of reduce to key-value pairs.
			* If the reduction operator is not commutative then bolt::reduce_by_key should not be used. 
            *
            * \param ctl           \b Optional Control structure to control command-queue, debug, tuning, etc.
            *                      See bolt::cl::control.
            * \param keys_first    The first element of the key sequence.
            * \param keys_last     The last  element of the key sequence.
            * \param values_first  The first element of the value sequence.
            * \param keys_output   The first element of the key output sequence.
            * \param values_output The first element of the value output sequence.
            * \param binary_pred   Binary predicate which determines if two keys are equal.
            * \param binary_op     Binary function for scanning transformed elements.
            * \param user_code     A user-specified Optional tring that is preppended to the generated OpenCL kernel.
            *
            * \tparam InputIterator1   is a model of Input Iterator.
            * \tparam InputIterator2   is a model of Input Iterator.
            * \tparam OutputIterator   is a model of Output Iterator.
            * \tparam BinaryPredicate  is a model of Binary Predicate.
            * \tparam BinaryFunction   is a model of Binary Function whose return type
            *                          is convertible to \c OutputIterator's  \c value_type.
            *
            * \return pair(last of keys output, last of value output)
            *
            *\details Example:
            * \code
            * #include "bolt/cl/reduce_by_key.h"
            * ...
            *
            * int keys[11] = { 0, 0, 0, 2, 2, 2, -5, -5, -5, -5, 6 };
            * int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 };
            * int keys_out[11];
            * int vals_out[11];
            * bolt::cl::control ctl = control::getDefault();
            *
            * bolt::cl::reduce_by_key( ctl, keys, keys+11, vals, keys_out, vals_out,bolt::cl::equal_to<int>(),
            * bolt::cl::plus<int>() );
            * // keys_out => { 0, 2, -5, 6 }
            * // vals_out => { 6, 6, 8, 2 }
            *  \endcode
            *
            */
            template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2,
                typename BinaryPredicate,
                typename BinaryFunction>
                pair<OutputIterator1, OutputIterator2>
                reduce_by_key(
                control &ctl,
                InputIterator1  keys_first,
                InputIterator1  keys_last,
                InputIterator2  values_first,
                OutputIterator1  keys_output,
                OutputIterator2  values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op,
                const std::string& user_code="" );

            template<
                typename InputIterator1,
                typename InputIterator2,
                typename OutputIterator1,
                typename OutputIterator2,
                typename BinaryPredicate,
                typename BinaryFunction>
                pair<OutputIterator1, OutputIterator2>
                reduce_by_key(
                InputIterator1  keys_first,
                InputIterator1  keys_last,
                InputIterator2  values_first,
                OutputIterator1  keys_output,
                OutputIterator2  values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op,
                const std::string& user_code="" );



        };// end of bolt::cl
};// end of bolt namespace

#include <bolt/cl/detail/reduce_by_key.inl>

#endif
