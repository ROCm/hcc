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

#if !defined( BOLT_CL_SCATTER_H )
#define BOLT_CL_SCATTER_H
#pragma once

#include "bolt/cl/device_vector.h"
#include "bolt/cl/functional.h"


/*! \file bolt/cl/scatter.h
    \brief scatters elements from a source range to a destination array.
*/


namespace bolt {
    namespace cl {
        
		/*! \addtogroup algorithmssc
         */

        /*! \addtogroup CL-scatter
        *   \ingroup copying
        *   \{
        */

        /*! 
        *   \p scatter APIs copy elements from a source range to a destination array (conditionally) according to a
        *   specified map. For common code between the host and device, one can take a look at the ClCode and TypeName
        *   implementations. See Bolt Tools for Split-Source for a detailed description.
        */
		
       /*! \brief This version of \p scatter copies elements from a source range to a destination array according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [first, last), scatter copies
         * the corresponding \p input_first to result[ map [ i ] ]
         *
         * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc.See bolt::cl::control.
         * \param first The beginning of input sequence.
         * \param last The end of input sequence.
         * \param map The beginning of the source sequence.
         * \param result The beginning of the output sequence.
         * \param user_code Optional OpenCL&tm; code to be passed to the OpenCL compiler. The cl_code is inserted
         *   first in the generated code, before the cl_code trait.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *
         *  \details The following code snippet demonstrates how to use \p scatter
         *
         *  \code
         *  #include <bolt/cl/scatter.h>
         *  #include <bolt/cl/functional.h>
         *
         *  int input[10] = {5, 7, 2, 3, 12, 6, 9, 8, 1, 4};
         *  int map[10] = {8, 2, 3, 9, 0, 5, 1, 7, 6, 4};
         *  int output[10];
         *  bolt::cl::scatter(input, input + 10, map, output);
         *
         *  // output is now {12, 9, 7, 2, 4, 6, 1, 8, 5, 3};
         *  \endcode
         *
         */

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename OutputIterator >
        void scatter( ::bolt::cl::control &ctl,
                      InputIterator1 first,
                      InputIterator1 last,
                      InputIterator2 map,
                      OutputIterator result,
                      const std::string& user_code="" );

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename OutputIterator >
        void scatter( InputIterator1 first,
                      InputIterator1 last1,
                      InputIterator2 map,
                      OutputIterator result,
                      const std::string& user_code="" );

       /*! \brief This version of \p scatter_if copies elements from a source range to a destination array according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [first, last), scatter_if copies
         * the corresponding \p input_first to result[ map [ i ] ] if stencil[ i - first ] is
         * \p true.
         *
         * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc.See bolt::cl::control.
         * \param first The beginning of input sequence.
         * \param last The end of input sequence.
         * \param map The beginning of the map sequence.
         * \param stencil The beginning of the stencil sequence.
         * \param result The beginning of the output sequence.
         * \param user_code Optional OpenCL&tm; code to be passed to the OpenCL compiler. The cl_code is inserted
         *   first in the generated code, before the cl_code trait.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam InputIterator3 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *
         *  \details The following code snippet demonstrates how to use \p scatter_if
         *
         *  \code
         *  #include <bolt/cl/scatter.h>
         *  #include <bolt/cl/functional.h>
         *
         *  int input[10]   = {5, 7, 2, 3, 12, 6, 9, 8, 1, 4};
         *  int map[10]     = {8, 2, 3, 9, 0, 5, 1, 7, 6, 4};
         *  int stencil[10] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
         *  int output[10];
         *  bolt::cl::scatter_if(input, input + 10, map, stencil, output);
         *
         *  // output is now {0, 9, 0, 0, 4, 6, 1, 8, 0, 0};
         *  \endcode
         *
         */


        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator >
        void scatter_if( bolt::cl::control &ctl,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 map,
                         InputIterator3 stencil,
                         OutputIterator result,
                         const std::string& user_code="" );

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator >
        void scatter_if( InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 map,
                         InputIterator3 stencil,
                         OutputIterator result,
                         const std::string& user_code="" );

       /*! \brief This version of \p scatter_if copies elements from a source range to a destination array according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [first, last), scatter_if copies
         * the corresponding \p input_first to result[ map [ i ] ] if pred (stencil[ i - first ])
         * is \p true.
         *
         * \param ctl \b Optional Control structure to control command-queue, debug, tuning, etc.See bolt::cl::control.
         * \param first The beginning of input sequence.
         * \param last The end of input sequence.
         * \param map The beginning of the map sequence.
         * \param stencil The beginning of the stencil sequence.
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
         *  \details The following code snippet demonstrates how to use \p scatter_if
         *
         *  \code
         *  #include <bolt/cl/scatter.h>
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
         *  int input[10]   = {5, 7, 2, 3, 12, 6, 9, 8, 1, 4};
         *  int map[10]     = {8, 2, 3, 9, 0, 5, 1, 7, 6, 4};
         *  int stencil[10] = {1, 3, 5, 2, 4, 6, 10, 9, 12, 22};
         *  int output[10];
         *  greater_pred is_gt_5;
         *  bolt::cl::scatter_if(input, input + 10, map, stencil, output, is_gt_5);
         *
         *  // output is now {0, 9, 0, 0, 4, 6, 1, 8, 0, 0};
         *  \endcode
         *
         */

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator,
                  typename Predicate >
        void scatter_if( bolt::cl::control &ctl,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 map,
                         InputIterator3 stencil,
                         OutputIterator result,
                         Predicate pred,
                         const std::string& user_code="" );

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator,
                  typename Predicate >
        void scatter_if( InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 map,
                         InputIterator3 stencil,
                         OutputIterator result,
                         Predicate pred,
                         const std::string& user_code="" );


        /*!   \}  */
    };
};

#include <bolt/cl/detail/scatter.inl>
#endif
