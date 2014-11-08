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
#if !defined( BOLT_BTBB_SCATTER_H )
#define BOLT_BTBB_SCATTER_H

#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

/*! \file bolt/bttb/scatter.h
    \brief scatters elements from a source range to a destination array.
*/


namespace bolt {
    namespace btbb {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup copying
        *   \ingroup algorithms
        *   \p scatter APIs copy elements from a source range to a destination array (conditionally) according to a
        *   specified map. For common code between the host and device, one can take a look at the ClCode and TypeName
        *   implementations. See Bolt Tools for Split-Source for a detailed description.
        */

        /*! \addtogroup TBB-scatter
        *   \ingroup algorithms
        *   \{
        */


       /*! \brief This version of \p scatter copies elements from a source range to a destination array according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [first, last), scatter copies
         * the corresponding \p input_first to result[ map [ i ] ]
         *
         * \param first The beginning of input sequence.
         * \param last The end of input sequence.
         * \param map The beginning of the source sequence.
         * \param result The beginning of the output sequence.
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
         *  #include <bolt/cl/control.h>
         *
         *  int input[10] = {5, 7, 2, 3, 12, 6, 9, 8, 1, 4};
         *  int map[10] = {8, 2, 3, 9, 0, 5, 1, 7, 6, 4};
         *  int output[10];
         *
         *  bolt::cl::control ctl;
         *  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
         *  bolt::cl::scatter(ctl, input, input + 10, map, output);
         *
         *  // output is now {1, 2, 3, 4, 5, 6, 7, 8, 9, 12};
         *  \endcode
         *
         */

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename OutputIterator >
        void scatter( InputIterator1 first,
                      InputIterator1 last,
                      InputIterator2 map,
                      OutputIterator result);

       /*! \brief This version of \p scatter copies elements from a source range to a destination array according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [first, last), scatter copies
         * the corresponding \p input_first to result[ map [ i ] ] if stencil[ i - first ] is
         * \p true.
         *
         * \param first The beginning of input sequence.
         * \param last The end of input sequence.
         * \param map The beginning of the map sequence.
         * \param stencil The beginning of the stencil sequence.
         * \param result The beginning of the output sequence.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam InputIterator3 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *
         *  \details The following code snippet demonstrates how to use \p scatter
         *
         *  \code
         *  #include <bolt/cl/scatter.h>
         *  #include <bolt/cl/functional.h>
         *  #include <bolt/cl/control.h>
         *
         *  int input[10]   = {5, 7, 2, 3, 12, 6, 9, 8, 1, 4};
         *  int map[10]     = {8, 2, 3, 9, 0, 5, 1, 7, 6, 4};
         *  int stencil[10] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
         *  int output[10];
         *
         *  bolt::cl::control ctl;
         *  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
         *
         *  bolt::cl::scatter(ctl, input, input + 10, map, stencil, output);
         *
         *  // output is now {0, 0, 0, 0, 0, 6, 7, 8, 9, 12};
         *  \endcode
         *
         */


       
        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator >
        void scatter_if( InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 map,
                         InputIterator3 stencil,
                         OutputIterator result);


       /*! \brief This version of \p scatter copies elements from a source range to a destination array according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [first, last), scatter copies
         * the corresponding \p input_first to result[ map [ i ] ] if pred (stencil[ i - first ])
         * is \p true.
         *
         * \param first The beginning of input sequence.
         * \param last The end of input sequence.
         * \param map The beginning of the map sequence.
         * \param stencil The beginning of the stencil sequence.
         * \param result The beginning of the output sequence.
         * \param pred A predicate for stencil.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam InputIterator3 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *  \tparam Predicate is a model of Predicate
         *
         *  \details The following code snippet demonstrates how to use \p scatter
         *
         *  \code
         *  #include <bolt/cl/scatter.h>
         *  #include <bolt/cl/functional.h>
         *  #include <bolt/cl/control.h>
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
         *
         *  bolt::cl::control ctl;
         *  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
         *
         *  bolt::cl::scatter(ctl, input, input + 10, map, stencil, output, is_gt_5);
         *
         *  // output is now {0, 0, 0, 0, 0, 6, 7, 8, 9, 12};
         *  \endcode
         *
         */

     
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
                         Predicate pred);


        /*!   \}  */
    }
}

#include <bolt/btbb/detail/scatter.inl>
#endif
