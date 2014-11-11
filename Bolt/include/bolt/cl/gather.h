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

#if !defined( BOLT_CL_GATHER_H )
#define BOLT_CL_GATHER_H
#pragma once

#include "bolt/cl/device_vector.h"
#include "bolt/cl/functional.h"


/*! \file bolt/cl/gather.h
    \brief gathers elements from a source array to a destination range.
*/


namespace bolt {
    namespace cl {
        
		/*! \addtogroup algorithms
         */

        /*! \addtogroup CL-gather
        *   \ingroup copying
        *   \{
        */

        /*! 
        *   \p gather APIs copy elements from a source array to a destination range (conditionally) according to a
        *   specified map. For common code between the host and device, one can take a look at the ClCode and TypeName
        *   implementations. See Bolt Tools for Split-Source for a detailed description.
        */
		
       /*! \brief This version of \p gather copies elements from a source array to a destination range according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [map_first, map_last), gather copies
         * the corresponding \p input_first[ map [ i ] ] to result[ i - map_first ]
         *
         * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc.See bolt::cl::control.
         * \param map_first The beginning of map sequence.
         * \param map_last The end of map sequence.
         * \param input The beginning of the source sequence.
         * \param result The beginning of the output sequence.
         * \param user_code Optional OpenCL&tm; code to be passed to the OpenCL compiler. The cl_code is inserted
         *   first in the generated code, before the cl_code trait.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *
         *  \details The following code snippet demonstrates how to use \p gather
         *
         *  \code
         *  #include <bolt/cl/gather.h>
         *  #include <bolt/cl/functional.h>
         *
         *  int map[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
         *  int input[10] = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};
         *  int output[10];
         *  bolt::cl::gather(map, map + 10, input, output);
         *
         *  // output is now {99, 88, 77, 66, 55, 44, 33, 22, 11, 0};
         *  \endcode
         *
         */

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename OutputIterator >
        void gather( ::bolt::cl::control &ctl,
                     InputIterator1 map_first,
                     InputIterator1 map_last,
                     InputIterator2 input_first,
                     OutputIterator result,
                     const std::string& user_code="" );

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename OutputIterator >
        void gather( InputIterator1 map_first,
                     InputIterator1 map_last,
                     InputIterator2 input_first,
                     OutputIterator result,
                     const std::string& user_code="" );


       /*! \brief This version of \p gather_if copies elements from a source array to a destination range according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [map_first, map_last), gather_if copies
         * the corresponding \p input_first[ map [ i ] ] to result[ i - map_first ] if stencil[ i - map_first ] is
         * \p true.
         *
         * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc.See bolt::cl::control.
         * \param map_first The beginning of map sequence.
         * \param map_last The end of map sequence.
         * \param stencil The beginning of the stencil sequence.
         * \param input The beginning of the source sequence.
         * \param result The beginning of the output sequence.
         * \param user_code Optional OpenCL&tm; code to be passed to the OpenCL compiler. The cl_code is inserted
         *   first in the generated code, before the cl_code trait.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam InputIterator3 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *
         *  \details The following code snippet demonstrates how to use \p gather_if
         *
         *  \code
         *  #include <bolt/cl/gather.h>
         *  #include <bolt/cl/functional.h>
         *
         *  int map[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
         *  int input[10] = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};
         *  int stencil[10] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
         *  int output[10];
         *  bolt::cl::gather_if(map, map + 10, stencil, input, output);
         *
         *  // output is now {0, 0, 0, 0, 0, 44, 33, 22, 11, 0};
         *  \endcode
         *
         */

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator >
        void gather_if( bolt::cl::control &ctl,
                         InputIterator1 map_first,
                         InputIterator1 map_last,
                         InputIterator2 stencil,
                         InputIterator3 input_first,
                         OutputIterator result,
                         const std::string& user_code="" );

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator >
        void gather_if( InputIterator1 map_first,
                         InputIterator1 map_last,
                         InputIterator2 stencil,
                         InputIterator3 input_first,
                         OutputIterator result,
                         const std::string& user_code="" );

       /*! \brief This version of \p gather_if copies elements from a source array to a destination range according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [map_first, map_last), gather_if copies
         * the corresponding \p input_first[ map [ i ] ] to result[ i - map_first ] if pred (stencil[ i - map_first ])
         * is \p true.
         *
         * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc.See bolt::cl::control.
         * \param map_first The beginning of map sequence.
         * \param map_last The end of map sequence.
         * \param stencil The beginning of the stencil sequence.
         * \param input The beginning of the source sequence.
         * \param result The beginning of the output sequence.
         * \param pred A predicate for stencil.
         * \param user_code Optional OpenCL&tm; code to be passed to the OpenCL compiler. The cl_code is inserted
         *   first in the generated code, before the cl_code trait.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam InputIterator3 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *  \tparam Predicate is a model of Predicate
         *
         *  \details The following code snippet demonstrates how to use \p gather_if
         *
         *  \code
         *  #include <bolt/cl/gather.h>
         *  #include <bolt/cl/functional.h>
         *
         *  BOLT_FUNCTOR( greater_pred,
         *    struct greater_pred{
         *        bool operator () (int x)
         *        {
         *            return ( (x > 5)?1:0 );
         *        }
         *    };
         *  );
         * 
         *  ...
         *
         *  int map[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
         *  int input[10] = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};
         *  int stencil[10] = {2, 3, 1, 4, 5, 10, 6, 8, 7, 9};
         *  int output[10];
         *
         *  greater_pred is_gt_5;
         *  bolt::cl::gather_if(map, map + 10, stencil, input, output, is_gt_5);
         *
         *  // output is now {0, 0, 0, 0, 0, 44, 33, 22, 11, 0};
         *  \endcode
         *
         */


        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator,
                  typename Predicate >
        void gather_if( bolt::cl::control &ctl,
                         InputIterator1 map_first,
                         InputIterator1 map_last,
                         InputIterator2 stencil,
                         InputIterator3 input,
                         OutputIterator result,
                         Predicate pred,
                         const std::string& user_code="" );

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator,
                  typename Predicate >
        void gather_if( InputIterator1 map_first,
                         InputIterator1 map_last,
                         InputIterator2 stencil,
                         InputIterator3 input,
                         OutputIterator result,
                         Predicate pred,
                         const std::string& user_code="" );


        /*!   \}  */
    };
};

#include <bolt/cl/detail/gather.inl>
#endif
