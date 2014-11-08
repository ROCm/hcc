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
#if !defined( BOLT_CL_COUNTING_ITERATOR_H )
#define BOLT_CL_COUNTING_ITERATOR_H

#include "bolt/cl/device_vector.h"
#include "bolt/cl/iterator/iterator_traits.h"
#include <boost/iterator/iterator_facade.hpp>

/*! \file bolt/cl/iterator/counting_iterator.h
    \brief Return Incremented Value on dereferencing.
*/

namespace bolt {
namespace cl {

    struct counting_iterator_tag
        : public fancy_iterator_tag
        {   // identifying tag for random-access iterators
        };

        /*! \addtogroup fancy_iterators
         */

        /*! \addtogroup CL-CountingIterator
        *   \ingroup fancy_iterators
        *   \{
        */

        /*! counting_iterator iterates a range with sequential values.
         *
         *
         *
         *  \details The following demonstrates how to use a \p counting_iterator.
         *
         *  \code
         *  #include <bolt/cl/counting_iterator.h>
         *  #include <bolt/cl/transform.h>
         *  ...
         *
         *  std::vector<int> vecSrc( 5 );
         *  std::vector<int> vecDest( 5 );
         *
         *  std::fill( vecSrc.begin( ), vecSrc.end( ), 10 );
         *
         *  bolt::cl::control ctrl = control::getDefault( );
         *  ...
         *  bolt::cl::counting_iterator< int > count5( 5 );
         *  bolt::cl::transform( ctrl, vecSrc.begin( ), vecSrc.end( ), count5, vecDest.begin( ), bolt::cl::plus< int >( ) );
         *
         *  // Output:
         *  // vecDest = { 15, 16, 17, 18, 19 }
         *
         *  // counting_iterator can save bandwidth when used instead of a range of values.
         *  \endcode
         *
         */

        template< typename value_type >
        class counting_iterator: public boost::iterator_facade< counting_iterator< value_type >, value_type,
            counting_iterator_tag, value_type, int >
        {
        public:

	    typedef typename boost::iterator_facade< counting_iterator< value_type >, value_type,
            counting_iterator_tag, value_type, int >::difference_type  difference_type;
            typedef counting_iterator_tag                              iterator_category;
            typedef std::random_access_iterator_tag                    memory_system;
            typedef value_type *                                       pointer;

            struct Payload
            {
                value_type m_Value;
            };

            //  Basic constructor requires a reference to the container and a positional element
            counting_iterator( value_type init, const control& ctl = control::getDefault( ) ):
                m_initValue( init ),
                m_Index( 0 )
            {
                const ::cl::CommandQueue& m_commQueue = ctl.getCommandQueue( );

                //  We want to use the context from the passed in commandqueue to initialize our buffer
                cl_int l_Error = CL_SUCCESS;
                ::cl::Context l_Context = m_commQueue.getInfo< CL_QUEUE_CONTEXT >( &l_Error );
                V_OPENCL( l_Error, "device_vector failed to query for the context of the ::cl::CommandQueue object" );

                m_devMemory = ::cl::Buffer( l_Context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                    1 * sizeof( value_type ), &m_initValue );
            }

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
            template< typename OtherType >
            counting_iterator( const counting_iterator< OtherType >& rhs ): m_devMemory( rhs.m_devMemory ),
                m_Index( rhs.m_Index ), m_initValue( rhs.m_initValue )
            {
            }

            value_type* getPointer()
            {
                return &m_initValue;
            }

            const value_type* getPointer() const
            {
                return &m_initValue;
            }

            counting_iterator< value_type >& operator= ( const counting_iterator< value_type >& rhs )
            {
                if( this == &rhs )
                    return *this;

                m_devMemory = rhs.m_devMemory;
                m_initValue = rhs.m_initValue;
                m_Index = rhs.m_Index;
                return *this;
            }

            counting_iterator< value_type >& operator+= ( const difference_type & n )
            {
                advance( n );
                return *this;
            }

            const counting_iterator< value_type > operator+ ( const difference_type & n ) const
            {
                counting_iterator< value_type > result( *this );
                result.advance( n );
                return result;
            }

            const ::cl::Buffer& getBuffer( ) const
            {
                return m_devMemory;
            }

            const counting_iterator< value_type > & getContainer( ) const
            {
                return *this;
            }

            const counting_iterator< value_type > & base( ) const
            {
                return *this;
            }

            Payload gpuPayload( ) const
            {
                Payload payload = { m_initValue };
                return payload;
            }

            const difference_type gpuPayloadSize( ) const
            {
                return sizeof( Payload );
            }

            int setKernelBuffers(int arg_num, ::cl::Kernel &kernel) const
            {
                    const ::cl::Buffer &buffer = getContainer().getBuffer();
                    kernel.setArg(arg_num, buffer );
                    arg_num++;
                    return arg_num;
            }

            difference_type distance_to( const counting_iterator< value_type >& rhs ) const
            {
                //return static_cast< typename iterator_facade::difference_type >( 1 );
                return rhs.m_Index - m_Index;
            }

            //  Public member variables
            difference_type m_Index;

        private:
            //  Implementation detail of boost.iterator
            friend class boost::iterator_core_access;

            //  Used for templatized copy constructor and the templatized equal operator
            template < typename > friend class counting_iterator;

            //  For a counting_iterator, do nothing on an advance
            void advance(difference_type n )
            {
                m_Index += n;
            }

            void increment( )
            {
                advance( 1 );
            }

            void decrement( )
            {
                advance( -1 );
            }

            template< typename OtherType >
            bool equal( const counting_iterator< OtherType >& rhs ) const
            {
                bool sameIndex = (rhs.m_initValue == m_initValue) && (rhs.m_Index == m_Index);

                return sameIndex;
            }

            typename boost::iterator_facade< counting_iterator< value_type >, value_type, 
            counting_iterator_tag, value_type, int >::reference  dereference( ) const
            {
                return m_initValue + m_Index;
            }

            ::cl::Buffer m_devMemory;
            value_type m_initValue;
        };
    //)

    //  This string represents the device side definition of the counting_iterator template
    static std::string deviceCountingIterator =
        std::string("#if !defined(BOLT_CL_COUNTING_ITERATOR) \n") +
        STRINGIFY_CODE(
        #define BOLT_CL_COUNTING_ITERATOR \n
        namespace bolt { namespace cl { \n
        template< typename T > \n
        class counting_iterator \n
        { \n
        public: \n
            typedef int iterator_category;      // device code does not understand std:: tags \n
            typedef T value_type; \n
            typedef T base_type; \n
            typedef size_t difference_type; \n
            typedef size_t size_type; \n
            typedef T* pointer; \n
            typedef T& reference; \n

            counting_iterator( value_type init ): m_StartIndex( init ), m_Ptr( 0 ) \n
            {}; \n

            void init( global value_type* ptr ) \n
            { \n

                //m_Ptr = ptr; \n
            }; \n

            value_type operator[]( size_type threadID ) const \n
            { \n
                return m_StartIndex + threadID; \n
            } \n

            value_type operator*( ) const \n
            { \n
                return m_StartIndex + threadID; \n
            } \n

            value_type m_StartIndex; \n
        }; \n
    } } \n
    )
    +  std::string("#endif \n");


    template< typename Type >
    counting_iterator< Type > make_counting_iterator( Type constValue )
    {
        counting_iterator< Type > tmp( constValue );
        return tmp;
    }

}
}

BOLT_CREATE_TYPENAME( bolt::cl::counting_iterator< int > );
BOLT_CREATE_CLCODE( bolt::cl::counting_iterator< int >, bolt::cl::deviceCountingIterator );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::counting_iterator, int, unsigned int );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::counting_iterator, int, float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::counting_iterator, int, double );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::counting_iterator, int, cl_long );

#endif
