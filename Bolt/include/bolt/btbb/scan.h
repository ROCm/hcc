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

/******************************************************************************
 * OpenCL Scan
 *****************************************************************************/

#if !defined( BOLT_BBTBB_SCAN_H )
#define BOLT_BBTBB_SCAN_H
#pragma once

#include "tbb/parallel_scan.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"

/*! \file bolt/cl/scan.h
    \brief Scan calculates a running sum over a range of values, inclusive or exclusive
*/


namespace bolt
{
namespace btbb
{

/*! \addtogroup algorithms
 */

/*! \addtogroup PrefixSums Prefix Sums
 *   \ingroup algorithms
 *   The sorting Algorithm for sorting the given InputIterator.
 */

/*! \addtogroup TBB-scan
 *   \ingroup PrefixSums
 *   \{
 *
 */

/*! \brief \p inclusive_scan calculates a running sum over a range of values, inclusive of the current value.
 *   The result value at iterator position \p i is the running sum of all values less than \p i in the input range.
 *

 * \param first The first iterator in the input range to be scanned.
 * \param last  The last iterator in the input range to be scanned.
 * \param result  The first iterator in the output range.
 * \tparam InputIterator An iterator signifying the range is used as input.
 * \tparam OutputIterator An iterator signifying the range is used as output.
 * \return Iterator at the end of result sequence.
 *
 * \details Example
 * \code
 * #include "bolt/btbb/scan.h"
 *
 * int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *
 * // Calculate the inclusive scan of an input range, modifying the values in-place.
 * bolt::cl::inclusive_scan( a, a+10, a );
 * // a => {1, 3, 6, 10, 15, 21, 28, 36, 45, 55}
 *  \endcode
 * \sa http://www.sgi.com/tech/stl/partial_sum.html
 */

template< typename InputIterator, typename OutputIterator >
OutputIterator
inclusive_scan(
    InputIterator first,
    InputIterator last,
    OutputIterator result);



/*! \brief \p inclusive_scan calculates a running sum over a range of values, inclusive of the current value.
 *   The result value at iterator position \p i is the running sum of all values less than \p i in the input range.
 *
 * \param first The first iterator in the input range to be scanned.
 * \param last  The last iterator in the input range to be scanned.
 * \param result  The first iterator in the output range.
 * \param binary_op A functor object specifying the operation between two elements in the input range.
 * \tparam InputIterator An iterator signifying the range is used as input.
 * \tparam OutputIterator An iterator signifying the range is used as output.
 * \return Iterator at the end of result sequence.
 *
 * \details Example
 * \code
 * #include "bolt/btbb/scan.h"
 *
 * int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *
 * // Calculate the inclusive scan of an input range, modifying the values in-place.
 * bolt::cl::inclusive_scan( a, a+10, a );
 * // a => {1, 3, 6, 10, 15, 21, 28, 36, 45, 55}
 *  \endcode
 * \sa http://www.sgi.com/tech/stl/partial_sum.html
 */
template< typename InputIterator, typename OutputIterator, typename BinaryFunction >
OutputIterator
inclusive_scan(
    InputIterator first,
    InputIterator last,
    OutputIterator result,
    BinaryFunction binary_op);


/*! \brief \p exclusive_scan calculates a running sum over a range of values, exclusive of the current value.
 *   The result value at iterator position \p i is the running sum of all values less than \p i in the input range.
 *
 * \param first The first iterator in the input range to be scanned.
 * \param last  The last iterator in the input range to be scanned.
 * \param result  The first iterator in the output range.
 * \tparam InputIterator An iterator signifying the range is used as input.
 * \tparam OutputIterator An iterator signifying the range is used as output.
 * \return An iterator pointing at the end of the result range.
 *
 * \code
 * #include "bolt/btbb/scan.h"
 *
 * int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *
 * // Calculate the exclusive scan of an input range, modifying the values in-place.
 * bolt::cl::exclusive_scan( a, a+10, a );
 * // a => {0, 1, 3, 6, 10, 15, 21, 28, 36, 45}
 *  \endcode
 * \sa http://www.sgi.com/tech/stl/partial_sum.html
 */


template< typename InputIterator, typename OutputIterator >
OutputIterator
    exclusive_scan( InputIterator first, InputIterator last, OutputIterator result );


/*! \brief \p exclusive_scan calculates a running sum over a range of values, exclusive of the current value.
 *   The result value at iterator position \p i is the running sum of all values less than \p i in the input range.
 *
 * \param first The first iterator in the input range to be scanned.
 * \param last  The last iterator in the input range to be scanned.
 * \param result  The first iterator in the output range.
 * \param init  The value used to initialize the output scan sequence.
 * \tparam InputIterator implements an input iterator.
 * \tparam OutputIterator implements an output iterator.
 * \tparam T is convertible to std::iterator_traits< OutputIterator >::value_type.
 * \return An iterator pointing at the end of the result range.
 *
 * \code
 * #include "bolt/btbb/scan.h"
 *
 * int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *
 * int init = 0;
 *
 * // Calculate the exclusive scan of an input range, modifying the values in-place.
 * bolt::cl::exclusive_scan( a, a+10, a, init );
 * // a => {0, 1, 3, 6, 10, 15, 21, 28, 36, 45}
 *  \endcode
 * \sa http://www.sgi.com/tech/stl/partial_sum.html
 */

template< typename InputIterator, typename OutputIterator, typename T >
OutputIterator
    exclusive_scan( InputIterator first, InputIterator last, OutputIterator result, T init );

/*! \brief \p exclusive_scan calculates a running sum over a range of values, exclusive of the current value.
 *   The result value at iterator position \p i is the running sum of all values less than \p i in the input range.
 *
  * \param first The first iterator in the input range to be scanned.
 * \param last  The last iterator in the input range to be scanned.
 * \param result  The first iterator in the output range.
 * \param init  The value used to initialize the output scan sequence.
 * \param binary_op A functor object specifying the operation between two elements in the input range.
  * \tparam InputIterator An iterator signifying the range is used as input.
 * \tparam OutputIterator An iterator signifying the range is used as output.
 * \tparam T is convertible to std::iterator_traits< OutputIterator >::value_type.
 * \tparam BinaryFunction implements a binary function; its result should be  {{** Is ? **}}convertible to
 *   std::iterator_traits< OutputIterator >::value_type.
 * \return An iterator pointing at the end of the result range.
 *
 * \code
 * #include "bolt/btbb/scan.h"
 *
 * int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *
 * // Calculate the exclusive scan of an input range, modifying the values in-place
 * bolt::cl::exclusive_scan( a, a+10, a, 0, bolt::cl::plus< int >( ) );
 * // a => {0, 1, 3, 6, 10, 15, 21, 28, 36, 45}
 *  \endcode
 * \sa http://www.sgi.com/tech/stl/partial_sum.html
 */
template< typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction >
OutputIterator
    exclusive_scan( InputIterator first, InputIterator last, OutputIterator result, T init, BinaryFunction binary_op);


/*!   \}  */
}// end of bolt::btbb namespace
}// end of bolt namespace


#include <bolt/btbb/detail/scan.inl>

#endif // BTBB_SCAN_H