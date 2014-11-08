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

#if !defined( BOLT_AMP_TRANSFORM_SCAN_H )
#define BOLT_AMP_TRANSFORM_SCAN_H
#pragma once

#include <bolt/amp/bolt.h>

/*! \file bolt/amp/transform_scan.h
    \brief  Performs on a sequence, the transformation defined by a unary operator, then the inclusive/exclusive scan defined by a binary operator.
*/
namespace bolt
{
namespace amp
{

/*! \addtogroup algorithms
 */

/*! \addtogroup PrefixSums Prefix Sums
*   \ingroup algorithms
*/ 

/*! \addtogroup AMPTransformedPrefixSums AMP-Transformed Prefix Sums
*   \ingroup PrefixSums
*   \{
*/

/*! \brief \p transform_inclusive_scan performs, on a sequence, the transformation defined by a unary operator,
* then the inclusive scan defined by a binary operator.
*
* \param ctl   \b Optional Bolt control object, to describe the environment under which the function runs.
* \param first The first element of the input sequence.
* \param last  The last element of the input sequence.
* \param result  The first element of the output sequence.
* \param unary_op Unary operator for transformation.
* \param binary_op Binary operator for scanning transformed elements.
*
* \tparam InputIterator is a model of Input Iterator.
* \tparam OutputIterator is a model of Output Iterator.
* \tparam UnaryFunction is a model of Unary Function which takes as input \c InputIterator's \c value_type
* and whose return type is convertible to \c BinaryFunction's \c input types.
* \tparam BinaryFunction is a model of Binary Function which takes as input two values convertible
* from \c UnaryFunction's \c return type and whose return type
* is convertible to \c OutputIterator's \c value_type.
* \return result+(last-first).
*
* \code
* #include "bolt/amp/transform_scan.h"
* ...
*
* bolt::amp::square<int> sqInt;
* bolt::amp::plus<int> plInt;
* bolt::amp::control ctrl = control::getDefault();
* ...
*
* int a[10] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
*
* bolt::amp::transform_inclusive_scan( ctrl, a, a+10, a, sqInt, plInt );
* // a => {1, 5, 14, 30, 55, 91, 140, 204, 285, 385}
*  \endcode
*
* \sa transform
* \sa inclusive_scan
* \sa http://www.sgi.com/tech/stl/transform.html
* \sa http://www.sgi.com/tech/stl/partial_sum.html
* \sa http://www.sgi.com/tech/stl/InputIterator.html
* \sa http://www.sgi.com/tech/stl/OutputIterator.html
* \sa http://www.sgi.com/tech/stl/UnaryFunction.html
* \sa http://www.sgi.com/tech/stl/BinaryFunction.html
*/
template<
    typename InputIterator,
    typename OutputIterator,
    typename UnaryFunction,
    typename BinaryFunction>
OutputIterator
transform_inclusive_scan(
    control &ctl,
    InputIterator first,
    InputIterator last,
    OutputIterator result, 
    UnaryFunction unary_op,
    BinaryFunction binary_op);

template<
    typename InputIterator,
    typename OutputIterator,
    typename UnaryFunction,
    typename BinaryFunction>
OutputIterator
transform_inclusive_scan(
    InputIterator first,
    InputIterator last,
    OutputIterator result, 
    UnaryFunction unary_op,
    BinaryFunction binary_op);



/*! \brief \p transform_exclusive_scan performs, on a sequence, the transformation defined by a unary operator,
* then the exclusive scan defined by a binary operator.
*
* \param ctl   \b Optional Bolt control object, to describe the environment under which the function runs.
* \param first The first element of the input sequence.
* \param last  The last element of the input sequence.
* \param result  The first element of the output sequence.
* \param unary_op Unary operator for transformation.
* \param init  The value used to initialize the output scan sequence.
* \param binary_op Binary operator for scanning transformed elements.
*
* \tparam InputIterator is a model of Input Iterator.
* \tparam OutputIterator is a model of Output Iterator.
* \tparam UnaryFunction is a model of Unary Function which takes as input \c InputIterator's \c value_type
* and whose return type is convertible to \c BinaryFunction's \c input types.
* \tparam T is convertible to \c OutputIterator's value_type.
* \tparam BinaryFunction is a model of Binary Function which takes as input two values convertible
* from \c UnaryFunction's \c return type and whose return type
* is convertible to \c OutputIterator's \c value_type.
* \return result+(last-first).
*
* \code
* #include "bolt/amp/transform_scan.h"
* ...
*
* bolt::amp::square<int> sqInt;
* bolt::amp::plus<int> plInt;
* bolt::amp::control ctrl = control::getDefault();
* ...
*
* int a[10] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
*
* bolt::amp::transform_exclusive_scan( ctrl, a, a+10, a, sqInt, 0, plInt );
* // a => { 0, 1, 5, 14, 30, 55, 91, 140, 204, 285}
*  \endcode
*
* \sa transform
* \sa exclusive_scan
* \sa http://www.sgi.com/tech/stl/transform.html
* \sa http://www.sgi.com/tech/stl/partial_sum.html
* \sa http://www.sgi.com/tech/stl/InputIterator.html
* \sa http://www.sgi.com/tech/stl/OutputIterator.html
* \sa http://www.sgi.com/tech/stl/UnaryFunction.html
* \sa http://www.sgi.com/tech/stl/BinaryFunction.html
*/
template<
    typename InputIterator,
    typename OutputIterator,
    typename UnaryFunction,
    typename T,
    typename BinaryFunction>
OutputIterator
transform_exclusive_scan(
    control &ctl,
    InputIterator first,
    InputIterator last,
    OutputIterator result, 
    UnaryFunction unary_op,
    T init,
    BinaryFunction binary_op);

template<
    typename InputIterator,
    typename OutputIterator,
    typename UnaryFunction,
    typename T,
    typename BinaryFunction>
OutputIterator
transform_exclusive_scan(
    InputIterator first,
    InputIterator last,
    OutputIterator result, 
    UnaryFunction unary_op,
    T init,
    BinaryFunction binary_op);


/*!   \}  */
}// end of bolt::amp namespace
}// end of bolt namespace

#include <bolt/amp/detail/transform_scan.inl>

#endif
