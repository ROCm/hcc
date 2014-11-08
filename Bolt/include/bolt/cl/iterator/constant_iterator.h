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
#if !defined( BOLT_CL_CONSTANT_ITERATOR_H )
#define BOLT_CL_CONSTANT_ITERATOR_H
#include "bolt/cl/bolt.h"
#include "bolt/cl/iterator/iterator_traits.h"
#include <boost/iterator/iterator_facade.hpp>

/*! \file bolt/cl/iterator/constant_iterator.h
    \brief Return Same Value or Constant Value on dereferencing.
*/


namespace bolt {
namespace cl {

    struct constant_iterator_tag
        : public fancy_iterator_tag
        {   // identifying tag for random-access iterators

        };


        /*! \addtogroup fancy_iterators
         */

        /*! \addtogroup CL-ConstantIterator
        *   \ingroup fancy_iterators
        *   \{
        */

        /*! constant_iterator iterates a range with a constant value.
         *
         *
         *
         *  \details The following demonstrates how to use a \p constant_iterator.
         *
         *  \code
         *  #include <bolt/cl/constant_iterator.h>
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
         *  bolt::cl::constant_iterator< int > const5( 5 );
         *  bolt::cl::transform( ctrl, vecSrc.begin( ), vecSrc.end( ), const5, vecDest.begin( ), bolt::cl::plus< int >( ) );
         *
         *  // vecDest is vector filled with the value 15, the sum of vecSrc and the constant value 5 from the iterator
         *  // constant_iterator can save bandwidth when used instead of a range of values.
         *  \endcode
         *
         */

        template< typename value_type >
        class constant_iterator: public boost::iterator_facade< constant_iterator< value_type >, value_type,
            constant_iterator_tag, value_type, int >
        {
        public:
            typedef typename boost::iterator_facade< constant_iterator< value_type >, value_type, constant_iterator_tag,
                                   value_type, int >::difference_type        difference_type;
            typedef constant_iterator_tag                                    iterator_category;
            typedef std::random_access_iterator_tag                          memory_system;
            typedef value_type *                                             pointer;

            struct Payload
            {
                value_type m_Value;
            };

            //  Basic constructor requires a reference to the container and a positional element
            constant_iterator( value_type init, const control& ctl = control::getDefault( ) ):
                m_constValue( init ), m_Index( 0 )
            {
                const ::cl::CommandQueue& m_commQueue = ctl.getCommandQueue( );

                //  We want to use the context from the passed in commandqueue to initialize our buffer
                cl_int l_Error = CL_SUCCESS;
                ::cl::Context l_Context = m_commQueue.getInfo< CL_QUEUE_CONTEXT >( &l_Error );
                V_OPENCL( l_Error, "device_vector failed to query for the context of the ::cl::CommandQueue object" );

                m_devMemory = ::cl::Buffer( l_Context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                    1 * sizeof( value_type ),const_cast<value_type *>(&m_constValue) );
            }

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
            template< typename OtherType >
            constant_iterator( const constant_iterator< OtherType >& rhs ): m_devMemory( rhs.m_devMemory ),
                m_Index( rhs.m_Index ), m_constValue( rhs.m_constValue )
            {}

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
            constant_iterator< value_type >& operator= ( const constant_iterator< value_type >& rhs )
            {
                if( this == &rhs )
                    return *this;

                m_devMemory = rhs.m_devMemory;
                m_constValue = rhs.m_constValue;
                m_Index = rhs.m_Index;
                return *this;
            }
            
            value_type* getPointer()
            {
                return &m_constValue;
            }

            const value_type* getPointer() const
            {
                return &m_constValue;
            }

            constant_iterator< value_type >& operator+= ( const  difference_type & n )
            {
                advance( n );
                return *this;
            }

            const constant_iterator< value_type > operator+ ( const difference_type & n ) const
            {
                constant_iterator< value_type > result( *this );
                result.advance( n );
                return result;
            }

            const ::cl::Buffer& getBuffer( ) const
            {
                return m_devMemory;
            }

            const constant_iterator< value_type > & getContainer( ) const
            {
                return *this;
            }

            const constant_iterator< value_type > & base( ) const
            {
                return *this;
            }

            Payload gpuPayload( ) const
            {
                Payload payload = { m_constValue };
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

            difference_type distance_to( const constant_iterator< value_type >& rhs ) const
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
            template < typename > friend class constant_iterator;

            //  For a constant_iterator, do nothing on an advance
            void advance( difference_type n )
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
            bool equal( const constant_iterator< OtherType >& rhs ) const
            {
                bool sameIndex = (rhs.m_constValue == m_constValue) && (rhs.m_Index == m_Index);
                return sameIndex;
            }

            typename boost::iterator_facade< constant_iterator< value_type >, value_type,
            constant_iterator_tag, value_type, int >::reference dereference( ) const
            {
                return m_constValue;
            }

            ::cl::Buffer m_devMemory;
            value_type m_constValue;
        };
    //)

    //  This string represents the device side definition of the constant_iterator template
    static std::string deviceConstantIterator =
        std::string("#if !defined(BOLT_CL_CONSTANT_ITERATOR) \n") +
        STRINGIFY_CODE(
        #define BOLT_CL_CONSTANT_ITERATOR \n
        namespace bolt { namespace cl { \n
        template< typename T > \n
        class constant_iterator \n
        { \n
        public: \n
            typedef int iterator_category;      // device code does not understand std:: tags \n
            typedef T value_type; \n
            typedef T base_type; \n
            typedef size_t difference_type; \n
            typedef size_t size_type; \n
            typedef T* pointer; \n
            typedef T& reference; \n

            constant_iterator( value_type init ): m_constValue( init ), m_Ptr( 0 ) \n
            { }; \n

            void init( global value_type* ptr ) \n
            { }; \n

            value_type operator[]( size_type threadID ) const \n
            { \n
                return m_constValue; \n
            } \n

            value_type operator*( ) const \n
            { \n
                return m_constValue; \n
            } \n

            value_type m_constValue; \n
        }; \n
    } } \n
    )
    +  std::string("#endif \n");

    template< typename Type >
    constant_iterator< Type > make_constant_iterator( Type constValue )
    {
        constant_iterator< Type > tmp( constValue );
        return tmp;
    }

}
}

BOLT_CREATE_TYPENAME( bolt::cl::constant_iterator< int > );
BOLT_CREATE_CLCODE( bolt::cl::constant_iterator< int >, bolt::cl::deviceConstantIterator );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::constant_iterator, int, unsigned int );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::constant_iterator, int, float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::constant_iterator, int, double );

#endif
