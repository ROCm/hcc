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
#if !defined( BOLT_AMP_COUNT_H )
#define BOLT_AMP_COUNT_H

#include "bolt/amp/bolt.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/iterator/iterator_traits.h"

/*! \file bolt/amp/count.h
    \brief Counts the number of elements in the specified range.
*/


namespace bolt {
    namespace amp {

        /*! \addtogroup algorithms
         */

        /*! \addtogroup reductions
        *   \ingroup algorithms
        */

        /*! \addtogroup AMP-counting
        *  \ingroup reductions
        *  \{
        *  
        */

        namespace detail 
        {
        template <typename T> 
        struct CountIfEqual {
            CountIfEqual(const T &targetValue) restrict(amp,cpu)  : _targetValue(targetValue)
            { };
            CountIfEqual() restrict(amp,cpu) {}
            bool operator()  (const T &x) const restrict(amp,cpu)
            {
                   T temp= _targetValue;
                   return x == temp;
            };

        private:
            T _targetValue;
        };
        }
        // Hui. Prior to use
        /*!
        * \brief \p count_if counts the number of elements in the specified range for which the specified \p predicate 
        *  is \p true.  
        * 
        * \param ctl \b Optional Control structure to control accelerator,debug, tuning. See bolt::amp::control.
        * \param first The first position in the sequence to be counted.
        * \param last The last position in the sequence to be counted.
        * \param predicate The count is incremented for each element which returns true when passed to 
        *  the predicate function.        
        * \returns: The number of elements for which \p predicate is true.          
        *
        *  \tparam InputIterator is a model of InputIterator
        *  \tparam OutputIterator is a model of OutputIterator
        * \details  This example returns the number of elements in the range 1-60.
        * \code
        *
        * //Bolt functor specialized for int type.
        * template<typename T>
        * // Functor for range checking.
        * struct InRange {
        *   InRange (T low, T high) {
        *     _low=low;
        *     _high=high;
        *   };
        *
        *   bool operator() (const T& value) restrict(amp,cpu) { 
        *     return (value >= _low) && (value <= _high) ; 
        *   };
        *
        *  T _low;
        *  T _high;
        * };
        *
        *    int a[14] = {0, 10, 42, 55, 13, 13, 42, 19, 42, 11, 42, 99, 13, 77};
        *    int boltCount = bolt::amp::count_if (a, a+14, InRange<int>(1,60)) ;
        *    // boltCount 11 in range 1-60.
        *  \endcode
        *
        * \details Example to show how to use UDD type for count.
        * \code
        *  struct UDD { 
        *      int a; 
        *      int b;
        *
        *      bool operator() (const int &x) restrict(amp,cpu) {
        *          return (x == a || x == b);
        *      }
        *
        *      UDD() 
        *          : a(0),b(0) { } 
        *      UDD(int _in) 
        *          : a(_in), b(_in +1)  { } 
        *          
        *  }; 
        *
        *
        *    std::vector<UDD> boltInput(SIZE);
        *    UDD myUDD;
        *    myUDD.a = 3;
        *    myUDD.b = 5;
        *    // Initialize boltInput
        *    size_t boltCount = bolt::amp::count(boltInput.begin(), boltInput.end(), myUDD);
        *
        *
        *  \endcode
        */

       template<typename InputIterator, typename Predicate> 
        typename bolt::amp::iterator_traits<InputIterator>::difference_type
            count_if(control& ctl, InputIterator first, 
            InputIterator last, 
            //Predicate predicate=bolt::amp::detail::CountIfEqual< int >());
            // Hui
            Predicate predicate=bolt::amp::detail::CountIfEqual< typename std::iterator_traits<InputIterator>::value_type >());


        template<typename InputIterator, typename Predicate> 
        typename bolt::amp::iterator_traits<InputIterator>::difference_type
            count_if(InputIterator first, 
            InputIterator last, 
            Predicate predicate);
            
        

        /*!
         * \brief \p count counts the number of elements in the specified range which compare equal to the specified 
         * \p value.
         * 

         *  \param ctl \b Optional Control structure to control accelerator,debug, tuning. See bolt::amp::control.
         *  \param first Beginning of the source copy sequence.
         *  \param last  End of the source copy sequence.
         *  \param value Equality Comparable value.
         *  \return Count of the number of occurrences of \p value.
         *
         *  \tparam InputIterator is a model of InputIterator
         *
         * \details  Example:
         * \code
         *    int a[14] = {0, 10, 42, 55, 13, 13, 42, 19, 42, 11, 42, 99, 13, 77};
         *
         *    size_t countOf42 = bolt::amp::count (A, A+14, 42);
         *    // countOf42 contains 4.
         *  \endcode
         * 
         */

        template<typename InputIterator, typename EqualityComparable> 
        typename bolt::amp::iterator_traits<InputIterator>::difference_type
            count(control& ctl, InputIterator first, 
            InputIterator last, 
            const EqualityComparable &value)
        {
            typedef typename std::iterator_traits<InputIterator>::value_type T;
            return bolt::amp::count_if(ctl, first, last, detail::CountIfEqual<T>(value));
        };

        template<typename InputIterator, typename EqualityComparable> 
        typename bolt::amp::iterator_traits<InputIterator>::difference_type
            count(InputIterator first, 
            InputIterator last, 
            const EqualityComparable &value)
        {
            typedef typename std::iterator_traits<InputIterator>::value_type T;
            return bolt::amp::count_if(first, last, detail::CountIfEqual<T>(value));
        };
        
         /*!   \}  */
    };
};


#include <bolt/amp/detail/count.inl>

#endif
