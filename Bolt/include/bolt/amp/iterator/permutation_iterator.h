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
#if !defined( BOLT_CL_PERMUTATION_ITERATOR_H )
#define BOLT_CL_PERMUTATION_ITERATOR_H
#include "bolt/amp/bolt.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include "bolt/amp/iterator/counting_iterator.h"

/*! \file bolt/amp/iterator/permutation_iterator.h
    \brief
*/


namespace bolt {
namespace amp {

  struct permutation_iterator_tag
      : public fancy_iterator_tag
      {   // identifying tag for random-access iterators
      };

      /*! \addtogroup fancy_iterators
       */

      /*! \addtogroup AMP-PermutationIterator
      *   \ingroup fancy_iterators
      *   \{
      */

      /*! permutation_iterator permutes values in a range according to keys.
       *
       *
       *
       *  \details The following example demonstrates how to use a \p permutation_iterator.
       *
       *  \code
       *  #include <bolt/amp/permutation_iterator.h>
       *  #include <bolt/amp/transform.h>
       *
       *  ...
       *
       *  // Create device_vectors
       *  bolt::amp::device_vector< int > values( 5 );
       *  bolt::amp::device_vector< int > vecEmpty( 5, 0 );
       *  bolt::amp::device_vector< int > vecDest( 5, 0 );
       *  bolt::amp::device_vector< int > index( 5 );
       *
       *  // Fill values
       *  values[ 0 ] = 10 ; values[ 1 ] = 15 ; values[ 2 ] = 20 ;
       *  values[ 3 ] = 25 ; values[ 4 ] = 30 ;
       *
       *  // Fill permutation indices
       *  index[ 0 ] = 3 ; index[ 1 ] = 2 ; index[ 2 ] = 4 ;
       *  index[ 3 ] = 0 ; index[ 4 ] = 1 ;
       *
       *  bolt::amp::control ctrl = control::getDefault( );
       *  ...
       *  bolt::amp::transform( ctrl,
       *                        bolt::amp::make_permutation_iterator( values.begin( ), index.begin( ) ),
       *                        bolt::amp::make_permutation_iterator( values.end( ), index.end( ) ),
       *                        vecEmpty.begin( ),
       *                        vecDest.begin( ),
       *                        bolt::amp::plus< int >( ) );
       *
       *  // Output:
       *  // vecDest = { 25, 20, 30, 10, 15 }
       *
       *  \endcode
       *
       */
      template< typename element_type, typename key_type >
      class permutation_iterator: public std::iterator< permutation_iterator_tag, element_type, int>
      {
        public:
         typedef typename std::iterator< permutation_iterator_tag, element_type, int>::difference_type
         difference_type;

         typedef permutation_iterator<element_type,key_type> perm_iterator;
         typedef typename iterator_traits<element_type>::value_type value_type;
         typedef typename iterator_traits<key_type>::value_type index_type;
         typedef concurrency::array_view< value_type > arrayview_type;



        // Default constructor
         permutation_iterator():key_iterator(), element_iterator(), m_Index(0) { }

        //  Basic constructor requires a reference to the container and a positional element
        permutation_iterator( element_type ivalue, key_type ikey, const control& ctl = control::getDefault( ) ):
        element_iterator ( ivalue ), key_iterator ( ikey ), m_Index( ikey.getIndex( ) ) { }

        //  This copy constructor allows an iterator to convert into a perm_iterator, but not vica versa
        template< typename OtherValType, typename OtherKeyType >
        permutation_iterator( const permutation_iterator< OtherValType, OtherKeyType >& rhs ):m_Index( rhs.m_Index ){}

        //  This copy constructor allows an iterator to convert into a perm_iterator, but not vica versa
        permutation_iterator< element_type, key_type >& operator= ( const permutation_iterator< element_type, key_type >& rhs )
        {
            if( this == &rhs )
                return *this;

            key_iterator = rhs.key_iterator;
            element_iterator = rhs.element_iterator;

            m_Index = rhs.m_Index;
            return *this;
        }
            
        permutation_iterator< element_type, key_type >& operator+= ( const  difference_type & n ) const restrict (cpu,amp)
        {
            advance( n );
            return *this;
        }
            
        const permutation_iterator< element_type, key_type > operator+ ( const difference_type & n ) const restrict (cpu,amp)
        {
            permutation_iterator< element_type, key_type > result( *this );
            result.advance( n );
            return result;
        }

        const permutation_iterator< element_type, key_type > operator- ( const difference_type & n ) const restrict (cpu,amp)
        {
            permutation_iterator< element_type, key_type > result( *this );
            result.advance( -n );
            return result;
        }
// Find no use
#ifdef _WIN32
        const concurrency::array_view<int> & getBuffer( perm_iterator itr ) const
        {
            return *value;
        }
#endif

        const permutation_iterator< element_type, key_type > & getContainer( ) const
        {
            return *this;
        }

        difference_type operator- ( const permutation_iterator< element_type, key_type >& rhs ) const
        {
            return element_iterator.getIndex() - rhs.element_iterator.getIndex();
        }

        //  Public member variables
        difference_type m_Index;

        //  Used for templatized copy constructor and the templatized equal operator
        template < typename, typename > friend class permutation_iterator;

        void advance( difference_type n ) restrict ( cpu, amp ) 
        {
            m_Index += n;
        }


        // Pre-increment
        permutation_iterator< element_type, key_type > operator++ ( )
        {
            advance( 1 );
            permutation_iterator< element_type, key_type > result( *this );
            return result;
        }

        // Post-increment
        permutation_iterator< element_type, key_type > operator++ ( int )
        {
            permutation_iterator< element_type, key_type > result( *this );
            result.advance( 1 );
            return result;
        }

        // Pre-decrement
        permutation_iterator< element_type, key_type > operator--( ) const
        {
            permutation_iterator< element_type, key_type > result( *this );
            result.advance( -1 );
            return result;
        }

        // Post-decrement
        permutation_iterator< element_type, key_type > operator--( int ) const
        {
            permutation_iterator< element_type, key_type > result( *this );
            result.advance( -1 );
            return result;
        }

        difference_type getIndex() const
        {
            return m_Index;
        }

        template< typename OtherValue, typename OtherKey >
        bool operator== ( const permutation_iterator< OtherValue, OtherKey >& rhs ) const
        {
          bool sameIter = ( rhs.m_Index == m_Index );
          return sameIter;
        }

        template< typename OtherValue, typename OtherKey >
        bool operator!= ( const permutation_iterator< OtherValue, OtherKey >& rhs ) const
        {
          bool sameIter = ( rhs.m_Index != m_Index );
          return sameIter;
        }

        template< typename OtherValue, typename OtherKey >
        bool operator< ( const permutation_iterator< OtherValue, OtherKey >& rhs ) const
        {
            bool sameIndex = (m_Index < rhs.m_Index);

            return sameIndex;
        }

        // Dereference operators
        value_type& operator*() const restrict(cpu,amp)
        {
          index_type temp_index = key_iterator[m_Index];
          return element_iterator[temp_index];
        }

        value_type& operator[](int x) restrict(cpu,amp)
        {
          index_type temp_index = key_iterator[x];
          return element_iterator[temp_index];
        }

        value_type& operator[](int x) const restrict(cpu,amp)
        {
          index_type temp_index = key_iterator[x];
          return element_iterator[temp_index];
        }

        key_type key_iterator;
        element_type element_iterator;
      };


  template< typename valueType, typename keyType >
  permutation_iterator< valueType, keyType > make_permutation_iterator( valueType value, keyType key )
  {
      permutation_iterator< valueType, keyType > tmp( value, key );
      return tmp;
  }

}
}


#endif
