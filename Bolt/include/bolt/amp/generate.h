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

#if !defined( BOLT_AMP_GENERATE_H )
#define BOLT_AMP_GENERATE_H
#pragma once

#include <bolt/amp/bolt.h>
#include <bolt/amp/device_vector.h>

#include <string>
#include <iostream>

/*! \file bolt/amp/generate.h
    \brief Generate assigns to each element of a sequence [first,last].
*/

namespace bolt {
    namespace amp {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup transformations
        *   \ingroup algorithms
        *   \p generate assigns to each element of a sequence the value returned by a generator.
        */

        /*! \addtogroup AMP-generate
        *   \ingroup transformations
        *   \{
        */

         /*! \brief \p generate assigns to each element of a sequence [first,last] the value returned by gen.
         *
         *  \param ctl      \b Optional Control structure to control accelerator, debug, tuning, etc.See bolt::amp::control.
         *  \param first    The first element of the sequence.
         *  \param last     The last element of the sequence.
         *  \param gen      A generator, functor taking no parameters.
         *  \tparam ForwardIterator is a model of Forward Iterator, and \c ForwardIterator \c is mutable.
         *  \tparam Generator is a model of Generator, and \c Generator's \c result_type is convertible to \c
         *          ForwardIterator's \c value_type.
         *
         *  \details The following code snippet demonstrates how to fill a vector with a generator.
         *
         *  \code
         *  #include <bolt/amp/generate.h>
         *  #include <bolt/amp/device_vector.h>
         *  #include <stdlib.h>
         *  ...
         *
         *  struct ConstFunctor
         *  {
	     *      int val;
	     *      ConstFunctor(int a) : val(a) {};
         *
	     *      int operator() () const restrict(amp,cpu)
	     *      {
		 *          return val;
	     *      };
         *  };
         *  ...
         *
         *  ConstFunctor cf(1);
         *  std::vector<int> vec(1024);
         *  bolt::amp::generate( vec.begin(), vec.end(), cf);
         *
         *  // vec is now filled with 1
         *  \endcode
         *
         *  \sa http://www.sgi.com/tech/stl/generate.html
         */
        template<typename ForwardIterator, typename Generator>
        void generate(
            bolt::amp::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const Generator & gen);

        template<typename ForwardIterator, typename Generator>
        void generate(
            ForwardIterator first,
            ForwardIterator last,
            const Generator & gen);

        /*! \brief \p generate_n assigns to each element of a sequence [first,first+n] the value returned by gen.
         *
         *  \param ctl      Optional Control structure to control accelerator, debug, tuning, etc.See bolt::amp::control.
         *  \param first    The first element of the sequence.
         *  \param n        The number of sequence elements to generate.
         *  \param gen      A generator, functor taking no parameters.
         *  \tparam OutputIterator is a model of Output Iterator.
         *  \tparam Size is an integral type.
         *  \tparam Generator is a model of Generator, and \c Generator's \c result_type is convertible to
         *          \c OutputIterator's \c value_type.
         *
         *  \details The following code snippet demonstrates how to fill a vector with a generator.
         *
         *  \code
         *  #include <bolt/amp/generate.h>
         *  #include <bolt/amp/bolt.h>
         *  #include <stdlib.h>
         *  ...
         *
         *  struct ConstFunctor
         *  {
	     *      int val;
	     *      ConstFunctor(int a) : val(a) {};
         *
	     *      int operator() () const restrict(amp,cpu)
	     *      {
		 *          return val;
	     *      };
         *  };
         *  ...
         *
         *  ConstFunctor cf(1);
         *  std::vector<int> vec(1024);
         *  int n = 1024;
         *  bolt::amp::generate_n(vec.begin(), n, cf);
         *
         *  // vec is now filled with 1
         *  \endcode
         *
         *  \sa http://www.sgi.com/tech/stl/generate_n.html
         */
        template<typename OutputIterator, typename Size, typename Generator>
        OutputIterator generate_n(
            bolt::amp::control &ctl,
            OutputIterator first,
            Size n,
            const Generator & gen);

        template<typename OutputIterator, typename Size, typename Generator>
        OutputIterator generate_n(
            OutputIterator first,
            Size n,
            const Generator & gen);

         /*!   \}  */
    };
};

#include <bolt/amp/detail/generate.inl>
#endif
