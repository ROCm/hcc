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

#if !defined( BOLT_BTBB_GATHER_H )
#define BOLT_BTBB_GATHER_H
#pragma once

#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

/*! \file bolt/btbb/gather.h
    \brief gathers elements from a source array to a destination range.
*/


namespace bolt {
    namespace btbb {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup copying
        *   \ingroup algorithms
        *   \p gather APIs copy elements from a source array to a destination range (conditionally) according to a
        *   specified map. For common code between the host and device, one can take a look at the ClCode and TypeName
        *   implementations. See Bolt Tools for Split-Source for a detailed description.
        */

        /*! \addtogroup TBB-gather
        *   \ingroup algorithms
        *   \{
        */


       /*! \brief This version of \p gather copies elements from a source array to a destination range according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [map_first, map_last), gather copies
         * the corresponding \p input_first[ map [ i ] ] to result[ i - map_first ]
         *
         * \param map_first The beginning of map sequence.
         * \param map_last The end of map sequence.
         * \param input The beginning of the source sequence.
         * \param result The beginning of the output sequence.
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
         *  #include <bolt/cl/control.h>
         *
         *  int map[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
         *  int input[10] = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};
         *  int output[10];
         *
         *  bolt::cl::control ctl;
         *  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
         *
         *  bolt::cl::gather(ctl, map, map + 10, input, output);
         *
         *  // output is now {99, 88, 77, 66, 55, 44, 33, 22, 11, 0};
         *  \endcode
         *
         */

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename OutputIterator >
         void gather( InputIterator1 mapfirst,
                      InputIterator1 maplast,
                      InputIterator2 first,
                      OutputIterator result);

       /*! \brief This version of \p gather copies elements from a source array to a destination range according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [map_first, map_last), gather copies
         * the corresponding \p input_first[ map [ i ] ] to result[ i - map_first ] if stencil[ i - map_first ] is
         * \p true.
         *
         * \param map_first The beginning of map sequence.
         * \param map_last The end of map sequence.
         * \param stencil The beginning of the stencil sequence.
         * \param input The beginning of the source sequence.
         * \param result The beginning of the output sequence.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam InputIterator3 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *
         *  \details The following code snippet demonstrates how to use \p gather
         *
         *  \code
         *  #include <bolt/cl/gather.h>
         *  #include <bolt/cl/functional.h>
         *  #include <bolt/cl/control.h>
         *
         *  int map[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
         *  int input[10] = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};
         *  int stencil[10] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
         *  int output[10];
         *
         *  bolt::cl::control ctl;
         *  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
         *
         *  bolt::cl::gather(ctl, map, map + 10, stencil, input, output);
         *
         *  // output is now {99, 88, 77, 66, 55, 0, 0, 0, 0, 0};
         *  \endcode
         *
         */

        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator >
         void gather_if( InputIterator1 mapfirst,
                         InputIterator1 maplast,
                         InputIterator2 stencil,
                         InputIterator3 first,
                         OutputIterator result);

       /*! \brief This version of \p gather copies elements from a source array to a destination range according to a
         * specified map. For each \p i in \p InputIterator1 in the range \p [map_first, map_last), gather copies
         * the corresponding \p input_first[ map [ i ] ] to result[ i - map_first ] if pred (stencil[ i - map_first ])
         * is \p true.
         *
         * \param map_first The beginning of map sequence.
         * \param map_last The end of map sequence.
         * \param stencil The beginning of the stencil sequence.
         * \param input The beginning of the source sequence.
         * \param result The beginning of the output sequence.
         * \param pred A predicate for stencil.
         *
         *  \tparam InputIterator1 is a model of InputIterator
         *  \tparam InputIterator2 is a model of InputIterator
         *  \tparam InputIterator3 is a model of InputIterator
         *  \tparam OutputIterator is a model of OutputIterator
         *  \tparam Predicate is a model of Predicate
         *
         *  \details The following code snippet demonstrates how to use \p gather
         *
         *  \code
         *  #include <bolt/cl/gather.h>
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
         *  int map[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
         *  int input[10] = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};
         *  int stencil[10] = {2, 3, 1, 4, 5, 10, 6, 8, 7, 9};
         *  int output[10];
         *
         *  bolt::cl::control ctl;
         *  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
         *
         *  greater_pred is_gt_5;
         *  bolt::cl::gather(ctl, map, map + 10, stencil, input, output, is_gt_5);
         *
         *  // output is now {99, 88, 77, 66, 55, 0, 0, 0, 0, 0};
         *  \endcode
         *
         */

     
        template< typename InputIterator1,
                  typename InputIterator2,
                  typename InputIterator3,
                  typename OutputIterator,
                  typename Predicate >
        void gather_if( InputIterator1 mapfirst,
                         InputIterator1 maplast,
                         InputIterator2 stencil,
                         InputIterator3 first,
                         OutputIterator result,
                         Predicate pred);


        /*!   \}  */
    }
}

#include <bolt/btbb/detail/gather.inl>
#endif
