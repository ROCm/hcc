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
#if !defined( BOLT_AMP_STABLESORT_H )
#define BOLT_AMP_STABLESORT_H

#include "bolt/amp/bolt.h"
#include <string>


namespace bolt {
namespace amp {
    /*! \addtogroup algorithms
        */

    /*! \addtogroup sorting
    *   \ingroup algorithms
    *   Algorithms for sorting a given iterator range, with a possible user specified sorting criteria.
    *   Either fundamental or user-defined data types can be sorted.
    */ 

    /*! \addtogroup AMP-stable_sort
    *   \ingroup sorting
    *   \{
    */

    /*! \p Stable_sort returns the sorted result of all the elements in the range specified 
    * between the the first and last \p RandomAccessIterator iterators. This routine arranges the elements in 
    * ascending order assuming that an operator < exists for the value_type given by the iterator.
    *
    * The \p stable_sort operation is analogus to the std::stable_sort function.  It is a stable operation with respect
    * to the input data, in that if two elements are equivalent in the input range, and element X appears before element
    * Y, then element X has to maintain that relationship and appear before element Y after the sorting operation.  In 
    * general, stable sorts are usually prefered over unstable sorting algorithms, but may sacrifice a little performance 
    * to maintain this relationship.

    * \param first Defines the beginning of the range to be sorted
    * \param last  Defines the end of the range to be sorted
    * \return The data is sorted in place in the range [first,last)
    *
    * \tparam RandomAccessIterator models a random access iterator

    * The following code example shows the use of \p stable_sort to sort the elements in ascending order
    * \code
    * #include "bolt/amp/stablesort.h"
    * 
    * int   a[ 10 ] = { 2, 9, 3, 7, 5, 6, 3, 8, 9, 0 };
    * 
    * bolt::amp::stable_sort( a, a + 10 );
    * 
    * \\ results a[] = { 0, 2, 3, 3, 5, 6, 7, 8, 9, 9 }
    * \\ The 3s and the 9s kept their respective ordering from the original input
    * \endcode
    * \see http://www.sgi.com/tech/stl/stable_sort.html
    * \see http://www.sgi.com/tech/stl/RandomAccessIterator.html
    */
    template< typename RandomAccessIterator > 
    void stable_sort( RandomAccessIterator first, RandomAccessIterator last);

    /*! \p Stable_sort returns the sorted result of all the elements in the range specified 
    * between the the first and last \p RandomAccessIterator iterators. This routine arranges the elements in 
    * ascending order assuming that an operator < exists for the value_type given by the iterator.
    *
    * The \p stable_sort operation is analogus to the std::stable_sort function.  It is a stable operation with respect
    * to the input data, in that if two elements are equivalent in the input range, and element X appears before element
    * Y, then element X has to maintain that relationship and appear before element Y after the sorting operation.  In 
    * general, stable sorts are usually prefered over unstable sorting algorithms, but may sacrifice a little performance 
    * to maintain this relationship.  This overload of stable_sort accepts an additional comparator functor that allows
    * the user to specify the comparison operator to use.

    * \param first Defines the beginning of the range to be sorted
    * \param last  Defines the end of the range to be sorted
    * \param comp A user defined comparison function or functor that models a strict weak < operator
    * \return The data is sorted in place in the range [first,last)
    *
    * \tparam RandomAccessIterator models a random access iterator
    * \tparam StrictWeakOrdering models a binary predicate which returns true if the first element is 'less than' the second

    * The following code example shows the use of \p stable_sort to sort the elements in ascending order
    * \code
    * #include "bolt/amp/stablesort.h"
    * 
    * int   a[ 10 ] = { 2, 9, 3, 7, 5, 6, 3, 8, 9, 0 };
    * 
    * bolt::amp::stable_sort( a, a + 10, bolt::amp::greater< int >( ) );
    * 
    * \\ results a[] = { 9, 9, 8, 7, 6, 5, 3, 3, 2, 0 }
    * \endcode
    * \see http://www.sgi.com/tech/stl/stable_sort.html
    * \see http://www.sgi.com/tech/stl/RandomAccessIterator.html
    */
    template<typename RandomAccessIterator, typename StrictWeakOrdering> 
    void stable_sort( RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp);

    /*! \p Stable_sort returns the sorted result of all the elements in the range specified 
    * between the the first and last \p RandomAccessIterator iterators. This routine arranges the elements in 
    * ascending order assuming that an operator < exists for the value_type given by the iterator.
    *
    * The \p stable_sort operation is analogus to the std::stable_sort function.  It is a stable operation with respect
    * to the input data, in that if two elements are equivalent in the input range, and element X appears before element
    * Y, then element X has to maintain that relationship and appear before element Y after the sorting operation.  In 
    * general, stable sorts are usually prefered over unstable sorting algorithms, but may sacrifice a little performance 
    * to maintain this relationship.  This overload of stable_sort accepts an additional bolt::amp::control object that
    * allows the user to change the state that the function uses to make runtime decisions.

    * \param ctl A control object passed into stable_sort that the function uses to make runtime decisions
    * \param first Defines the beginning of the range to be sorted
    * \param last  Defines the end of the range to be sorted
    * \return The data is sorted in place in the range [first,last)
    *
    * \tparam RandomAccessIterator models a random access iterator

    * The following code example shows the use of \p stable_sort to sort the elements in ascending order
    * \code
    * #include "bolt/amp/stablesort.h"
    * 
    * int   a[ 10 ] = { 2, 9, 3, 7, 5, 6, 3, 8, 9, 0 };
    * 
    * bolt::amp::stable_sort( bolt::amp::control::getDefault( ), a, a + 10 );
    * 
    * \\ results a[] = { 9, 9, 8, 7, 6, 5, 3, 3, 2, 0 }
    * \endcode
    * \see bolt::amp::control
    * \see http://www.sgi.com/tech/stl/stable_sort.html
    * \see http://www.sgi.com/tech/stl/RandomAccessIterator.html
    */
    template<typename RandomAccessIterator> 
    void stable_sort( bolt::amp::control &ctl, RandomAccessIterator first, RandomAccessIterator last);

    /*! \p Stable_sort returns the sorted result of all the elements in the range specified 
    * between the the first and last \p RandomAccessIterator iterators. This routine arranges the elements in 
    * ascending order assuming that an operator < exists for the value_type given by the iterator.
    *
    * The \p stable_sort operation is analogus to the std::stable_sort function.  It is a stable operation with respect
    * to the input data, in that if two elements are equivalent in the input range, and element X appears before element
    * Y, then element X has to maintain that relationship and appear before element Y after the sorting operation.  In 
    * general, stable sorts are usually prefered over unstable sorting algorithms, but may sacrifice a little performance 
    * to maintain this relationship.  This overload of stable_sort accepts an additional comparator functor that allows
    * the user to specify the comparison operator to use.  This overload also accepts an additional bolt::amp::control
    * object that allows the user to change the state that the function uses to make runtime decisions.

    * \param ctl A control object passed into stable_sort that the function uses to make runtime decisions
    * \param first Defines the beginning of the range to be sorted
    * \param last  Defines the end of the range to be sorted
    * \param comp A user defined comparison function or functor that models a strict weak < operator
    * \return The data is sorted in place in the range [first,last)
    *
    * \tparam RandomAccessIterator models a random access iterator
    * \tparam StrictWeakOrdering models a binary predicate which returns true if the first element is 'less than' the second

    * The following code example shows the use of \p stable_sort to sort the elements in ascending order
    * \code
    * #include "bolt/amp/stablesort.h"
    * 
    * int   a[ 10 ] = { 2, 9, 3, 7, 5, 6, 3, 8, 9, 0 };
    * 
    * bolt::amp::stable_sort( bolt::amp::control::getDefault( ), a, a + 10, bolt::amp::greater< int >( ) );
    * 
    * \\ results a[] = { 9, 9, 8, 7, 6, 5, 3, 3, 2, 0 }
    * \endcode
    * \see bolt::amp::control
    * \see http://www.sgi.com/tech/stl/stable_sort.html
    * \see http://www.sgi.com/tech/stl/RandomAccessIterator.html
    */
    template<typename RandomAccessIterator, typename StrictWeakOrdering> 
    void stable_sort( bolt::amp::control &ctl, RandomAccessIterator first, RandomAccessIterator last,
        StrictWeakOrdering comp);

    /*!   \}  */

}// end of bolt::amp namespace
}// end of bolt namespace

#include "bolt/amp/detail/stablesort.inl"
#endif