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

#if !defined( BOLT_CL_GENERATE_H )
#define BOLT_CL_GENERATE_H
#pragma once

#include "bolt/cl/device_vector.h"

/*! \file bolt/cl/generate.h
    \brief Generate assigns to each element of a sequence [first,last].
*/

namespace bolt {
    namespace cl {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup transformations
        *   \ingroup algorithms
        *   \p generate assigns to each element of a sequence the value returned by a generator.
        */

        /*! \addtogroup CL-generate
        *   \ingroup transformations
        *   \{
        */

         /*! \brief \p generate assigns to each element of a sequence [first,last] the value returned by gen.
         *
         *  \param ctl      \b Optional control structure to control command-queue, debug, tuning, etc.
         *                  See bolt::cl::control.
         *  \param first    The first element of the sequence.
         *  \param last     The last element of the sequence.
         *  \param gen      A generator, functor taking no parameters.
         *  \param cl_code  Optional OpenCL(TM) code to be prepended to any OpenCL kernels used by this function.
         *
         *  \tparam ForwardIterator is a model of Forward Iterator, and \c ForwardIterator \c is mutable.
         *  \tparam Generator is a model of Generator, and \c Generator's \c result_type is convertible to \c
         *          ForwardIterator's \c value_type.
         *
         *  \details The following code snippet demonstrates how to fill a vector with a constant number.
         *
         *  \code
         *  #include <bolt/cl/generate.h>
         *  #include <bolt/cl/device_vector.h>
         *  #include <stdlib.h>
         *  ...
         *
         *  BOLT_FUNCTOR(ConstFunctor,
         *  struct ConstFunctor
         *  {
	     *      int val;
	     *      ConstFunctor(int a) : val(a) {};
         *
	     *      int operator() ()
	     *      {
		 *          return val;
	     *      };
         *  };
         *  );
         *  ...
         *
         *  ConstFunctor cf(1);
         *  std::vector<int> vec(1024);
         *  bolt::cl::generate( vec.begin(), vec.end(), cf);
         *
         *  // vec is now filled with 1
         *  \endcode
         *
         *  \sa http://www.sgi.com/tech/stl/generate.html
         */
        template<typename ForwardIterator, typename Generator>
        void generate(
            bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            Generator gen,
            const std::string& cl_code="");

        template<typename ForwardIterator, typename Generator>
        void generate(
            ForwardIterator first,
            ForwardIterator last,
            Generator gen,
            const std::string& cl_code="");

        /*! \brief \p generate_n assigns to each element of a sequence [first,first+n] the value returned by gen.
         *
         *  \param ctl      \b Optional control structure to control command-queue, debug, tuning, etc.
         *                  See bolt::cl::control.
         *  \param first    The first element of the sequence.
         *  \param n        The number of sequence elements to generate.
         *  \param gen      A generator, functor taking no parameters.
         *  \param cl_code  Optional OpenCL(TM) code to be prepended to any OpenCL kernels used by this function.
         *
         *  \tparam OutputIterator is a model of Output Iterator.
         *  \tparam Size is an integral type.
         *  \tparam Generator is a model of Generator, and \c Generator's \c result_type is convertible to
         *          \c OutputIterator's \c value_type.
         *
         *  \details The following code snippet demonstrates how to fill a vector with a constant number.
         *
         *  \code
         *  #include <bolt/cl/generate.h>
         *  #include <bolt/cl/bolt.h>
         *  #include <stdlib.h>
         *  ...
         *
         *  BOLT_FUNCTOR(ConstFunctor,
         *  struct ConstFunctor
         *  {
	     *      int val;
	     *      ConstFunctor(int a) : val(a) {};
         *
	     *      int operator() ()
	     *      {
		 *          return val;
	     *      };
         *  };
         *  );
         *  ...
         *
         *  ConstFunctor cf(1);
         *  std::vector<int> vec(1024);
         *  int n = 1024;
         *  bolt::cl::generate_n(vec.begin(), n, cf);
         *
         *  // vec is now filled with 1
         *  \endcode
         *
         *  \sa http://www.sgi.com/tech/stl/generate_n.html
         */
        template<typename OutputIterator, typename Size, typename Generator>
        OutputIterator generate_n(
            bolt::cl::control &ctl,
            OutputIterator first,
            Size n,
            Generator gen,
            const std::string& cl_code="");

        template<typename OutputIterator, typename Size, typename Generator>
        OutputIterator generate_n(
            OutputIterator first,
            Size n,
            Generator gen,
            const std::string& cl_code="");

         /*!   \}  */
    };
};

#include <bolt/cl/detail/generate.inl>
#endif
