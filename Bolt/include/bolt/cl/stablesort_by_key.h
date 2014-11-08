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

#pragma once
#if !defined( BOLT_CL_STABLESORT_BY_KEY_H )
#define BOLT_CL_STABLESORT_BY_KEY_H

#include "bolt/cl/device_vector.h"
#include "bolt/cl/functional.h"
#include "bolt/cl/copy.h"

namespace bolt {
namespace cl {
    /*! \addtogroup algorithms
        */

    /*! \addtogroup sorting
    *   \ingroup algorithms
    *   Algorithms for sorting a given iterator range, with a possible user specified sorting criteria.  These algorithms
    *   handle data where the keys and the values are split into seperate arrays.  The criteria to sort is applied to
    *   the key array, but the value array follows suite.  Fundamental or user-defined data types can be sorted.
    */

    /*! \addtogroup CL-stable_sort_by_key
    *   \ingroup sorting
    *   \{
    */

    /*! \p stable_sort_by_key returns the sorted result of all the elements in the range specified
    * given by the first and last \p RandomAccessIterator1 key iterators. This routine recieves two input ranges, the
    * first represents the range of keys to base the sort on, and the second to represent values that should identically be
    * sorted.  The permutation of elements returned in value range will be identical to the permutation of elements
    * applied to the key range.  This routine arranges the elements in ascending order assuming that an operator <
    * exists for the value_type given by the iterator.  No comparison operator needs to be provided for the value array.
    *
    * stable_sort_by_key is a stable operation with respect to the key data, in that if two elements are equivalent in
    * the key range and element X appears before element Y, then element X has to maintain that relationship and
    * appear before element Y after the sorting operation.  In general, stable sorts are usually prefered over
    * unstable sorting algorithms, but may sacrifice a little performance to maintain this relationship.

    * \param keys_first Defines the beginning of the key range to be sorted
    * \param keys_last  Defines the end of the key range to be sorted
    * \param values_first  Defines the beginning of the value range to be sorted, whose length equals
    * std::distance( keys_first, keys_last )
    * \param cl_code Optional OpenCL &trade; code to be passed to the OpenCL compiler. The cl_code is inserted first
    * in the generated code, before the cl_code traits. This can be used for any extra cl code to be passed when
    * compiling the OpenCl Kernel.
    * \return The data is sorted in place within the range [first,last)
    *
    * \tparam RandomAccessIterator1 models a random access iterator; iterator for the key range
    * \tparam RandomAccessIterator2 models a random access iterator; iterator for the value range

    * The following code example shows the use of \p stable_sort_by_key to sort the elements in ascending order
    * \code
    * #include "bolt/cl/stablesort_by_key.h"
    *
    * int   i[ 10 ] = { 2, 9, 3, 7, 5, 6, 3, 8, 9, 0 };
    * float f[ 10 ] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    *
    * bolt::cl::stable_sort_by_key( i, i + 10, f );
    *
    * \\ results i[] = { 0, 2, 3, 3, 5, 6, 7, 8, 9, 9 }
    * \\ results f[] = { 9.0f, 0.0f, 2.0f, 6.0f, 4.0f, 5.0f, 3.0f, 7.0f, 1.0f, 8.0f }
    * \\ The 3s and the 9s kept their respective ordering from the original input
    * \endcode
    * \see bolt::cl::stablesort
    * \see http://www.sgi.com/tech/stl/RandomAccessIterator.html
    */
    template< typename RandomAccessIterator1, typename RandomAccessIterator2 >
    void stable_sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, const std::string& cl_code="" );

    /*! \p stable_sort_by_key returns the sorted result of all the elements in the range specified
    * given by the first and last \p RandomAccessIterator1 key iterators. This routine recieves two input ranges, the
    * first represents the range of keys to base the sort on, and the second to represent values that should identically be
    * sorted.  The permutation of elements returned in value range will be identical to the permutation of elements
    * applied to the key range.  This routine arranges the elements in ascending order assuming that an operator <
    * exists for the value_type given by the iterator.  No comparison operator needs to be provided for the value array.
    *
    * stable_sort_by_key is a stable operation with respect to the key data, in that if two elements are equivalent in
    * the key range and element X appears before element Y, then element X has to maintain that relationship and
    * appear before element Y after the sorting operation.  In general, stable sorts are usually prefered over
    * unstable sorting algorithms, but may sacrifice a little performance to maintain this relationship.

    * \param keys_first Defines the beginning of the key range to be sorted
    * \param keys_last  Defines the end of the key range to be sorted
    * \param values_first  Defines the beginning of the value range to be sorted, whose length equals
    * std::distance( keys_first, keys_last )
    * \param comp A user defined comparison function or functor that models a strict weak < operator
    * \param cl_code Optional OpenCL &trade; code to be passed to the OpenCL compiler. The cl_code is inserted first
    * in the generated code, before the cl_code traits. This can be used for any extra cl code to be passed when
    * compiling the OpenCl Kernel.
    * \return The data is sorted in place within the range [first,last)
    *
    * \tparam RandomAccessIterator1 models a random access iterator; iterator for the key range
    * \tparam RandomAccessIterator2 models a random access iterator; iterator for the value range
    * \tparam StrictWeakOrdering models a binary predicate which returns true if the first element is 'less than' the second

    * The following code example shows the use of \p stable_sort_by_key to sort the elements in ascending order
    * \code
    * #include "bolt/cl/stablesort_by_key.h"
    *
    * int   i[ 10 ] = { 2, 9, 3, 7, 5, 6, 3, 8, 9, 0 };
    * float f[ 10 ] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    *
    * bolt::cl::stable_sort_by_key( i, i + 10, f, bolt::cl::greater< int >( ) );
    *
    * \\ results a[] = { 9, 9, 8, 7, 6, 5, 3, 3, 2, 0 }
    * \\ results f[] = { 1.0f, 8.0f, 7.0f, 3.0f, 5.0f, 4.0f, 2.0f, 6.0f, 0.0f, 9.0f }
    * \\ The 3s and the 9s kept their respective ordering from the original input
    * \endcode
    * \see bolt::cl::stablesort
    * \see http://www.sgi.com/tech/stl/RandomAccessIterator.html
    * \see http://www.sgi.com/tech/stl/StrictWeakOrdering.html
    */
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
    void stable_sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first,
        StrictWeakOrdering comp, const std::string& cl_code="");

    /*! \p stable_sort_by_key returns the sorted result of all the elements in the range specified
    * given by the first and last \p RandomAccessIterator1 key iterators. This routine recieves two input ranges, the
    * first represents the range of keys to base the sort on, and the second to represent values that should identically be
    * sorted.  The permutation of elements returned in value range will be identical to the permutation of elements
    * applied to the key range.  This routine arranges the elements in ascending order assuming that an operator <
    * exists for the value_type given by the iterator.  No comparison operator needs to be provided for the value array.
    *
    * stable_sort_by_key is a stable operation with respect to the key data, in that if two elements are equivalent in
    * the key range and element X appears before element Y, then element X has to maintain that relationship and
    * appear before element Y after the sorting operation.  In general, stable sorts are usually prefered over
    * unstable sorting algorithms, but may sacrifice a little performance to maintain this relationship.

    * \param ctl A control object passed into stable_sort_by_key used to make runtime decisions
    * \param keys_first Defines the beginning of the key range to be sorted
    * \param keys_last  Defines the end of the key range to be sorted
    * \param values_first  Defines the beginning of the value range to be sorted, whose length equals
    * std::distance( keys_first, keys_last )
    * \param cl_code Optional OpenCL &trade; code to be passed to the OpenCL compiler. The cl_code is inserted first
    * in the generated code, before the cl_code traits. This can be used for any extra cl code to be passed when
    * compiling the OpenCl Kernel.
    * \return The data is sorted in place within the range [first,last)
    *
    * \tparam RandomAccessIterator1 models a random access iterator; iterator for the key range
    * \tparam RandomAccessIterator2 models a random access iterator; iterator for the value range

    * The following code example shows the use of \p stable_sort_by_key to sort the elements in ascending order
    * \code
    * #include "bolt/cl/stablesort_by_key.h"
    *
    * int   i[ 10 ] = { 2, 9, 3, 7, 5, 6, 3, 8, 9, 0 };
    * float f[ 10 ] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    *
    * bolt::cl::stable_sort_by_key( control::getDefault( ), i, i + 10, f );
    *
    * \\ results i[] = { 0, 2, 3, 3, 5, 6, 7, 8, 9, 9 }
    * \\ results f[] = { 9.0f, 0.0f, 2.0f, 6.0f, 4.0f, 5.0f, 3.0f, 7.0f, 1.0f, 8.0f }
    * \\ The 3s and the 9s kept their respective ordering from the original input
    * \endcode
    * \see bolt::cl::stablesort
    * \see bolt::cl::control
    * \see http://www.sgi.com/tech/stl/RandomAccessIterator.html
    */
    template< typename RandomAccessIterator1, typename RandomAccessIterator2 >
    void stable_sort_by_key( bolt::cl::control &ctl, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, const std::string& cl_code="");

    /*! \p stable_sort_by_key returns the sorted result of all the elements in the range specified
    * given by the first and last \p RandomAccessIterator1 key iterators. This routine recieves two input ranges, the
    * first represents the range of keys to base the sort on, and the second to represent values that should identically be
    * sorted.  The permutation of elements returned in value range will be identical to the permutation of elements
    * applied to the key range.  This routine arranges the elements in ascending order assuming that an operator <
    * exists for the value_type given by the iterator.  No comparison operator needs to be provided for the value array.
    *
    * stable_sort_by_key is a stable operation with respect to the key data, in that if two elements are equivalent in
    * the key range and element X appears before element Y, then element X has to maintain that relationship and
    * appear before element Y after the sorting operation.  In general, stable sorts are usually prefered over
    * unstable sorting algorithms, but may sacrifice a little performance to maintain this relationship.

    * \param ctl A control object passed into stable_sort_by_key used to make runtime decisions
    * \param keys_first Defines the beginning of the key range to be sorted
    * \param keys_last  Defines the end of the key range to be sorted
    * \param values_first  Defines the beginning of the value range to be sorted, whose length equals
    * std::distance( keys_first, keys_last )
    * \param comp A user defined comparison function or functor that models a strict weak < operator
    * \param cl_code Optional OpenCL &trade; code to be passed to the OpenCL compiler. The cl_code is inserted first
    * in the generated code, before the cl_code traits. This can be used for any extra cl code to be passed when
    * compiling the OpenCl Kernel.
    * \return The data is sorted in place within the range [first,last)
    *
    * \tparam RandomAccessIterator1 models a random access iterator; iterator for the key range
    * \tparam RandomAccessIterator2 models a random access iterator; iterator for the value range
    * \tparam StrictWeakOrdering models a binary predicate which returns true if the first element is 'less than' the second

    * The following code example shows the use of \p stable_sort_by_key to sort the elements in ascending order
    * \code
    * #include "bolt/cl/stablesort_by_key.h"
    *
    * int   i[ 10 ] = { 2, 9, 3, 7, 5, 6, 3, 8, 9, 0 };
    * float f[ 10 ] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    *
    * bolt::cl::stable_sort_by_key( control::getDefault( ), i, i + 10, f, bolt::cl::greater< int >( ) );
    *
    * \\ results a[] = { 9, 9, 8, 7, 6, 5, 3, 3, 2, 0 }
    * \\ results f[] = { 1.0f, 8.0f, 7.0f, 3.0f, 5.0f, 4.0f, 2.0f, 6.0f, 0.0f, 9.0f }
    * \\ The 3s and the 9s kept their respective ordering from the original input
    * \endcode
    * \see bolt::cl::stablesort
    * \see bolt::cl::control
    * \see http://www.sgi.com/tech/stl/RandomAccessIterator.html
    */
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
    void stable_sort_by_key( bolt::cl::control &ctl, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, StrictWeakOrdering comp, const std::string& cl_code="");

    /*!   \}  */

}// end of bolt::cl namespace
}// end of bolt namespace

#include "bolt/cl/detail/stablesort_by_key.inl"
#endif