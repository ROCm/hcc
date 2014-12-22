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

/*! \file bolt/amp/device_vector.h
    \brief Header file for the device_container class.
*/


#pragma once
#if !defined( BOLT_AMP_DEVICE_VECTOR_H )
#define BOLT_AMP_DEVICE_VECTOR_H

#include <iterator>
#include <type_traits>
#include <numeric>
#include <amp.h>
#include <bolt/amp/control.h>
#include <exception> // For exception class
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/reverse_iterator.hpp>
#include <boost/shared_array.hpp>

// Hui: Define is_iterator type trait
#if BOOST_VERSION <= 155000 || !defined(is_iterator)
// Borrow idea from http://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error
namespace bolt {
  namespace amp {
  template <typename T>
    struct is_iterator {
      // Types "yes" and "no" are guaranteed to have different sizes,
      // specifically sizeof(yes) == 1 and sizeof(no) == 2.
      typedef char yes[1];
      typedef char no[2];

      template <typename C>
      static yes& test(typename C::iterator_category* );

      template <typename>
      static no& test(...);

      // If the "sizeof" of the result of calling test<T>(0) would be equal to sizeof(yes),
      // the first overload worked and T has a nested type named T.
      static const bool value = sizeof(test<T>(0)) == sizeof(yes);
    };
  }
}
#else
// TODO: implemented?
using boost::is_iterator;
#endif

/*! \brief Defining namespace for the Bolt project
    */
namespace bolt
{
/*! \brief Namespace containing AMP related data types and functions
 */
namespace amp
{
/*! \addtogroup Containers
 */

/*! \addtogroup AMP-Device
*   \ingroup Containers
*   Containers that guarantee random access to a flat, sequential region of memory that is performant
*   for the device.
*/

template< typename T, template < typename, int RANK = 1 > class CONT= concurrency::array_view >
class device_vector;

    struct device_vector_tag
        : public std::random_access_iterator_tag
        {   // identifying tag for random-access iterators
        };



    template <typename T>
    class create_empty_array
    {
    public:
        // Create an AV on the CPU
        static device_vector< T > getav() restrict(cpu)
        {
            return device_vector<T>();
        }
    };

    template <typename T>
    class create_empty_array_view
    {
    public:
        static concurrency::array_view<T, 1> getav() restrict(cpu)
        {
            static T type_default;
            concurrency::array<T, 1> A = concurrency::array<T>(1);
            return concurrency::array_view<T,1>(A);
        }
    };
/*! \brief This defines the AMP version of a device_vector
*   \ingroup AMP-Device
*   \details A device_vector is an abstract data type that provides random access to a flat, sequential region of memory that is performant
*   for the device.  This can imply different memories for different devices.  For discrete class graphics,
*   devices, this is most likely video memory; for APU devices, this can imply zero-copy memory; for CPU devices, this can imply
*   standard host memory.
*   \sa http://www.sgi.com/tech/stl/Vector.html
*/

// Hui. CONT is defined as above
template< typename T, template < typename, int RANK = 1 > class CONT/*= concurrency::array_view*/ >
class device_vector
{
    typedef T* naked_pointer;
    typedef const T* const_naked_pointer;

public:
    //  Useful typedefs specific to this container
    typedef T value_type;
    typedef ptrdiff_t difference_type;
    typedef difference_type distance_type;
    typedef int size_type;

    // These typedefs help define the template template parameter that represents our AMP container
    typedef concurrency::array_view< T > arrayview_type;
    typedef concurrency::array< T > array_type;
    
    typedef CONT< T > container_type;

    typedef naked_pointer pointer;
    typedef const_naked_pointer const_pointer;

    /*! \brief A writeable element of the container
    *   The location of an element of the container may not actually reside in system memory, but rather in device
    *   memory, which may be in a partitioned memory space.  Access to a reference of the container results in
    *   a mapping and unmapping operation of device memory.
    *   \note The container element reference is implemented as a proxy object.
    *   \warning Use of this class can be slow: each operation on it results in a map/unmap sequence.
    */

    template< typename Container >
    class reference_base
    {
    public:
        reference_base( Container& rhs, size_type index ): m_Container( rhs, false ), m_Index( index )
        {}

        //  Automatic type conversion operator to turn the reference object into a value_type
        operator value_type( ) const
        {
            arrayview_type av( m_Container.m_devMemory );
            value_type &result = av[static_cast< int >( m_Index )];

            return result;
        }

        reference_base< Container >& operator=( const value_type& rhs )
        {
            arrayview_type av( m_Container.m_devMemory );
            av[static_cast< int >( m_Index )] = rhs;

            return *this;
        }

        /*! \brief A get accessor function to return the encapsulated device_vector.
        */
        Container& getContainer( ) const
        {
            return m_Container;
        }

        /*! \brief A get index function to return the index of the reference object within the AMP device_vector.
        */
        size_type getIndex() const
        {
            return m_Index;
        }

    private:
        Container m_Container;
        size_type m_Index;
    };

    /*! \brief Typedef to create the non-constant reference.
    */

    typedef reference_base< device_vector< value_type, CONT > > reference;

    template< typename Container >
    class const_reference_base
    {
    public:
        const_reference_base( const Container& rhs, size_type index ): m_Container( rhs, false ), m_Index( index )
        {}

        //  Automatic type conversion operator to turn the reference object into a value_type
        operator value_type( ) const
        {
            arrayview_type av( m_Container.m_devMemory );
            value_type &result = av[static_cast< int >( m_Index )];

            return result;
        }

        const_reference_base< const Container >& operator=( const value_type& rhs )
        {
            arrayview_type av( m_Container.m_devMemory );
            av[static_cast< int >( m_Index )] = rhs;

            return *this;
        }

        /*! \brief A get accessor function to return the encapsulated device_vector.
        */
        const Container& getContainer( ) const
        {
            return m_Container;
        }

        /*! \brief A get index function to return the index of the reference object within the AMP device_vector.
        */
        size_type getIndex() const
        {
            return m_Index;
        }

    private:
        const Container m_Container;
        size_type m_Index;
    };

    /*! \brief A non-writeable copy of an element of the container.
    *   Constant references are optimized to return a value_type, since it is certain that
    *   the value will not be modified
    *   \note A const_reference actually returns a value, not a reference.
    */
    typedef const_reference_base< device_vector< value_type, CONT > > const_reference;

    //  Handy for the reference class to get at the wrapped ::cl objects
    #ifdef _WIN32
    friend class reference;
    #else
    friend reference;
    #endif

    /*! \brief Base class provided to encapsulate all the common functionality for constant
    *   and non-constant iterators.
    *   \sa http://www.sgi.com/tech/stl/Iterators.html
    *   \sa http://www.sgi.com/tech/stl/RandomAccessIterator.html
    *   \bug operator[] with device_vector iterators result in a compile-time error when accessed for reading.
    *   Writing with operator[] appears to be OK.  Workarounds: either use the operator[] on the device_vector
    *   container, or use iterator arithmetic instead, such as *(iter + 5) for reading from the iterator.
    */
    template< typename Container >
    class iterator_base: public boost::iterator_facade< iterator_base< Container >,
        value_type, device_vector_tag, typename device_vector::reference >
    {
    public:

        iterator_base( ): m_Container( create_empty_array< value_type >::getav( ) ), m_Index( 0 )
        {}

        //  Basic constructor requires a reference to the container and a positional element
        iterator_base( Container& rhs, size_type index ): m_Container( rhs, false ), m_Index( static_cast<int>(index) )
        {}

        //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
        template< typename OtherContainer >
        iterator_base( const iterator_base< OtherContainer >& rhs ):
            m_Container( rhs.m_Container, false ), m_Index( rhs.m_Index )
        {}
        iterator_base( const iterator_base& rhs ):
            m_Container( rhs.m_Container, false ), m_Index( rhs.m_Index )
        {}
        //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
        iterator_base< Container >& operator= ( const iterator_base< Container >& rhs )
        {
            m_Container = rhs.m_Container;
            m_Index = rhs.m_Index;
            return *this;
        }

        iterator_base< Container >& operator+= ( const difference_type & n )
        {
            advance( n );
            return *this;
        }

        const iterator_base< Container > operator+ ( const difference_type & n ) const
        {
            iterator_base< Container > result(*this);
            result.advance(n);
            return result;
        }

        Container& getContainer( )
        {
          return m_Container;
        }

        const Container& getContainer( ) const
        {
          return m_Container;
        }

        size_type getIndex() const
        {
            return m_Index;
        }


        difference_type distance_to( const iterator_base< Container >& rhs ) const
        {
            return ( rhs.m_Index - m_Index );
        }
        int m_Index;

        //  Implementation detail of boost.iterator
        friend class boost::iterator_core_access;

        //  Handy for the device_vector erase methods
        friend class device_vector< value_type >;

        //  Used for templatized copy constructor and the templatized equal operator
        template < typename > friend class iterator_base;

        void advance( difference_type n )
        {
            m_Index += static_cast<int>(n);
        }

        void increment( )
        {
            advance( 1 );
        }

        void decrement( )
        {
            advance( -1 );
        }


        template< typename OtherContainer >
        bool equal( const iterator_base< OtherContainer >& rhs ) const
        {
            bool sameIndex = rhs.m_Index == m_Index;
            bool sameContainer = &m_Container.m_devMemory[m_Index] == &rhs.m_Container.m_devMemory[rhs.m_Index];
            return ( sameIndex && sameContainer );
        }

        value_type& operator[](int x) const restrict(cpu,amp)
        {
            return m_Container[m_Index + x];
        }

        value_type& operator*() const restrict(cpu,amp)
        {
            return m_Container[m_Index];
        }

private:
        Container m_Container;

    };


    /*! \brief A reverse random access iterator in the classic sense
    *   \todo Need to implement base() which returns the base iterator
    *   \sa http://www.sgi.com/tech/stl/ReverseIterator.html
    *   \sa http://www.sgi.com/tech/stl/RandomAccessIterator.html
    */
    template< typename Container >
    class reverse_iterator_base: public boost::iterator_facade< reverse_iterator_base< Container >,
        value_type, std::random_access_iterator_tag, typename device_vector::reference >
    {
    public:

        reverse_iterator_base( ): m_Container( create_empty_array< value_type >::getav( ) ), m_Index( 0 )
        {}

        //  Basic constructor requires a reference to the container and a positional element
        reverse_iterator_base( Container& lhs, int index ): m_Container( lhs, false ), m_Index( static_cast<int>(index-1) )
        {}

        //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
        template< typename OtherContainer >
        reverse_iterator_base( const reverse_iterator_base< OtherContainer >& lhs ):
            m_Container( lhs.m_Container, false ), m_Index( lhs.m_Index )
        {}

		reverse_iterator_base( const reverse_iterator_base& lhs ):
            m_Container( lhs.m_Container, false ), m_Index( lhs.m_Index )
        {}
        //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
        reverse_iterator_base< Container >& operator= ( const reverse_iterator_base< Container >& lhs )
        {
            m_Container = lhs.m_Container;
            m_Index = lhs.m_Index;
            return *this;
        }

        reverse_iterator_base< Container >& operator+= ( const difference_type & n )
        {
            advance( -n );
            return *this;
        }

        const reverse_iterator_base< Container > operator+ ( const difference_type & n ) const
        {
            reverse_iterator_base< Container > result(*this);
            result.advance(-n);
            return result;
        }

        int getIndex() const
        {
            return m_Index;
        }

        Container& getContainer( ) const
        {
          return m_Container;
        }


        difference_type distance_to( const reverse_iterator_base< Container >& lhs ) const
        {
            return static_cast< difference_type >( m_Index - lhs.m_Index );
        }

        int m_Index;

        //  Implementation detail of boost.iterator
        friend class boost::iterator_core_access;

        //  Handy for the device_vector erase methods
        friend class device_vector< value_type >;

        //  Used for templatized copy constructor and the templatized equal operator
        template < typename > friend class reverse_iterator_base;

        void advance( difference_type n )
        {
            m_Index += static_cast<int>(n);
        }

        void increment( )
        {
            advance( -1 );
        }

        void decrement( )
        {
            advance( 1 );
        }

        template< typename OtherContainer >
        bool equal( const reverse_iterator_base< OtherContainer >& lhs ) const
        {
            bool sameIndex = lhs.m_Index == m_Index;
			bool sameContainer = &m_Container.m_devMemory[m_Index] == &lhs.m_Container.m_devMemory[lhs.m_Index];
            return ( sameIndex && sameContainer );
        }

        value_type& operator[](int x) const restrict(cpu,amp)
        {
            return m_Container[m_Index - x];
        }

        value_type& operator*() const restrict(cpu,amp)
        {
            return m_Container[m_Index];
        }

    private:
        Container m_Container;

    };


    /*! \brief Typedef to create the non-constant iterator
    */
    typedef iterator_base< device_vector< value_type, CONT > > iterator;

    /*! \brief Typedef to create the constant iterator
    */
    typedef iterator_base< const device_vector< value_type, CONT > > const_iterator;

    /*! \brief Typedef to create the non-constant reverse iterator
    */
    typedef reverse_iterator_base< device_vector< value_type, CONT > > reverse_iterator;

    /*! \brief Typedef to create the constant reverse iterator
    */
    typedef reverse_iterator_base< const device_vector< value_type, CONT > > const_reverse_iterator;


    /*! \brief A default constructor that creates an empty device_vector
    *   \param ctl An Bolt control class used to perform copy operations; a default is used if not supplied by the user
    *   \todo Find a way to be able to unambiguously specify memory flags for this constructor, that is not
    *   confused with the size constructor below.
    */
    device_vector( control& ctl = control::getDefault( ) )
        : m_Size( static_cast<int>(0) ), m_devMemory( create_empty_array_view<value_type>::getav() )
    { 
		
	}

    /*! \brief A constructor that creates a new device_vector with the specified number of elements,
    *   with a specified initial value.
    *   \param newSize The number of elements of the new device_vector
    *   \param value The value with which to initialize new elements.
    *   \param init Boolean value to indicate whether to initialize device memory from initValue.
    *   \param ctl A Bolt control class for copy operations; a default is used if not supplied by the user.
    *   \warning The ::cl::CommandQueue is not an STD reserve( ) parameter.
    */
    device_vector( size_type newSize, const value_type& initValue = value_type( ), bool init = true,
        control& ctl = control::getDefault( ) )
        : m_Size( static_cast<int>(newSize) ), m_devMemory( create_empty_array_view<value_type>::getav() )
    {

        if( m_Size > 0 )
        {
			concurrency::array<value_type> tmp = array_type( static_cast< int >( m_Size ), ctl.getAccelerator().get_default_view() );
            m_devMemory = tmp.view_as(tmp.get_extent());
            if( init )
            {
                arrayview_type m_devMemoryAV( m_devMemory );
                Concurrency::parallel_for_each( m_devMemoryAV.get_extent(), [=](Concurrency::index<1> idx) restrict(amp)
                {
                    m_devMemoryAV[idx] = initValue;
                }
                );
            }
        }
    }

    /*! \brief A constructor that creates a new device_vector using a range specified by the user.
    *   \param begin An iterator pointing at the beginning of the range.
	*   \param newSize The number of elements of the new device_vector
    *   \param discard Boolean value to whether the container data will be discarded for read operation.
    *   \param ctl A Bolt control class used to perform copy operations; a default is used if not supplied by the user.
    *   \note Ignore the enable_if<> parameter; it prevents this constructor from being called with integral types.
    */
    template< typename InputIterator >
    device_vector( const InputIterator begin, size_type newSize, bool discard = false, control& ctl = control::getDefault( ), 
                typename std::enable_if< !std::is_integral< InputIterator >::value &&
                                    std::is_same< arrayview_type, container_type >::value>::type* = 0 )
                                    : m_Size( static_cast<int>(newSize) ), m_devMemory( create_empty_array_view<value_type>::getav() )
    {
		if( m_Size > 0 )
        {
			concurrency::extent<1> ext( static_cast< int >( m_Size ) );
			concurrency::array_view<value_type> tmp = arrayview_type( ext, reinterpret_cast< value_type* >(&begin[ 0 ])  );
			m_devMemory = tmp.view_as(tmp.get_extent());
		}
		if(discard)
			m_devMemory.discard_data();
    };


	/*! \brief A constructor that creates a new device_vector using a range specified by the user.
    *   \param begin An iterator pointing at the beginning of the range.
	*   \param newSize The number of elements of the new device_vector
    *   \param discard Boolean value to whether the container data will be discarded for read operation.
    *   \param ctl A Bolt control class used to perform copy operations; a default is used if not supplied by the user.
    *   \note Ignore the enable_if<> parameter; it prevents this constructor from being called with integral types.
    */
    template< typename InputIterator >
    device_vector( const InputIterator begin, size_type newSize, bool discard = false, control& ctl = control::getDefault( ), 
                typename std::enable_if< !std::is_integral< InputIterator >::value &&
                                    std::is_same< array_type, container_type >::value>::type* = 0 )
                                    : m_Size( static_cast<int>(newSize) ), m_devMemory( create_empty_array_view<value_type>::getav() )
    {
		if( m_Size > 0 )
        {
			concurrency::extent<1> ext( static_cast< int >( m_Size ) );
			concurrency::array<value_type> tmp = array_type( ext, reinterpret_cast< value_type* >(&begin[ 0 ])  );
			m_devMemory = tmp.view_as(tmp.get_extent());
		}
    };


    /*! \brief A constructor that creates a new device_vector from device_vector specified by user.
    *   \param cont An device_vector object that has both .data() and .size() members
	*   \param copy Boolean value to decide whether new device_vector will be shallow copy or deep copy
	*   \param ctl A Bolt control class used to perform copy operations; a default is used if not supplied by the user.
    */
	device_vector( const device_vector<T, CONT> &cont, bool copy = true,control& ctl = control::getDefault( ) ) restrict(amp, cpu) : m_Size( cont.size( ) ),
														m_devMemory( cont.m_devMemory.view_as(cont.m_devMemory.get_extent()))
    {
		if(!copy)
			return;
		if( m_Size > 0 )
        {
			concurrency::array<value_type> tmp = array_type( m_devMemory );
			m_devMemory = tmp.view_as(tmp.get_extent());
		}
    };
	
    /*! \brief A constructor that creates a new device_vector using a pre-initialized array supplied by the user.
    *   \param cont An concurrency::array object.
    */
    device_vector( arrayview_type &cont): m_Size(cont.get_extent().size()), m_devMemory( cont.view_as(cont.get_extent()))
    {
    };


   /*! \brief A constructor that creates a new device_vector using a pre-initialized array_view supplied by the user.
    *   \param cont An concurrency::array_view object.
    */
	device_vector( array_type &cont): m_Size(cont.get_extent().size()), m_devMemory( cont.view_as(cont.get_extent()))
    {
    };

    /*! \brief A constructor that creates a new device_vector using a range specified by the user.
    *   \param begin An iterator pointing at the beginning of the range.
    *   \param end An iterator pointing at the end of the range.
	*   \param discard Boolean value to whether the container data will be discarded for read operation.
    *   \param ctl A Bolt control class used to perform copy operations; a default is used if not supplied by the user.
    *   \note concurrency::array_view specialization Ignore the enable_if<> parameter; it prevents this constructor from being called with integral types.
    */
    template< typename InputIterator >
    device_vector( const InputIterator begin, const InputIterator end, bool discard = false, control& ctl = control::getDefault( ),
        typename std::enable_if< std::is_same< arrayview_type, container_type >::value &&
                                   !std::is_integral< InputIterator >::value>::type* = 0 )
                                    : m_devMemory( create_empty_array_view<value_type>::getav() )
    {
        m_Size =  static_cast<int>(std::distance( begin, end ));

		if( m_Size > 0 )
        {
			concurrency::extent<1> ext( static_cast< int >( m_Size ) );
			concurrency::array_view<value_type> tmp = arrayview_type( ext, reinterpret_cast< value_type* >(&begin[ 0 ])  );
	        m_devMemory = tmp.view_as(tmp.get_extent());
		}
		if(discard)
			m_devMemory.discard_data();

    };


	/*! \brief A constructor that creates a new device_vector using a range specified by the user.
    *   \param begin An iterator pointing at the beginning of the range.
    *   \param end An iterator pointing at the end of the range.
	*   \param discard Boolean value to whether the container data will be discarded for read operation.
    *   \param ctl A Bolt control class used to perform copy operations; a default is used if not supplied by the user.
    *   \note concurrency::array specializationIgnore the enable_if<> parameter; it prevents this constructor from being called with integral types.
    */
	template< typename InputIterator >
    device_vector( const InputIterator begin, const InputIterator end, bool discard = false, control& ctl = control::getDefault( ),
        typename std::enable_if< std::is_same< array_type, container_type >::value &&
									!std::is_integral< InputIterator >::value>::type* = 0 )
                                    : m_devMemory( create_empty_array_view<value_type>::getav() )
    {
        m_Size =  static_cast<int>(std::distance( begin, end ));

		if( m_Size > 0 )
        {
			concurrency::extent<1> ext( static_cast< int >( m_Size ) );
			concurrency::array<value_type> tmp = array_type( ext, reinterpret_cast< value_type* >(&begin[ 0 ])  );
			m_devMemory = tmp.view_as(tmp.get_extent());
		}
    };


	    //destructor for device_vector
    ~device_vector() restrict(amp, cpu)
    {
    }

    //  Member functions

    /*! \brief A get accessor function to return the encapsulated device buffer for const objects.
    *   This member function allows access to the Buffer object, which can be retrieved through a reference or an iterator.
    *   This is necessary to allow library functions to get the encapsulated C++ AMP array object as a pass by reference argument
    *   to the C++ AMP parallel_for_each constructs.
    *   \note This get function could be implemented in the iterator, but the reference object is usually a temporary rvalue, so
    *   this location seems less intrusive to the design of the vector class.
    */
    arrayview_type getBuffer( ) const
    {
        concurrency::extent<1> ext( static_cast< int >( m_Size ) );
        return m_devMemory.view_as( ext );
    }

	/*! \brief A get accessor function to return the encapsulated device buffer for const objects based on the iterator getIndex() and size.
    *   This member function allows access to the Buffer object, which can be retrieved through a reference or an iterator.
    *   This is necessary to allow library functions to get the encapsulated C++ AMP array object as a pass by reference argument
    *   to the C++ AMP parallel_for_each constructs.
	*   \param itr An iterator pointing at the beginning of the range. 
	*   \param size Size of buffer. 
    *   \note This get function could be implemented in the iterator, but the reference object is usually a temporary rvalue, so
    *   this location seems less intrusive to the design of the vector class.
    */
	arrayview_type getBuffer( const_iterator itr, unsigned int size ) const
    {
		if(size == static_cast<unsigned int>(m_Size))
		{
			concurrency::extent<1> ext( static_cast< int >( m_Size ) );
			return m_devMemory.view_as( ext );
		}
		else
		{
			size_type offset = itr.getIndex();
			concurrency::extent<1> ext( static_cast< int >( size ) );
			return m_devMemory.section( Concurrency::index<1>(offset), ext );
		}
    }


	arrayview_type getBuffer( const_reverse_iterator itr, unsigned int size  ) const
    {
		if(size == static_cast<unsigned int>(m_Size))
		{
			concurrency::extent<1> ext( static_cast< int >( m_Size ) );
			return m_devMemory.view_as( ext );
		}
		else
		{
			size_type offset = itr.getIndex();
			concurrency::extent<1> ext( static_cast< int >( size ) );
			return m_devMemory.section( Concurrency::index<1>(offset), ext );
		}
    }

    /*! \brief Change the number of elements in device_vector to reqSize.
    *   If the new requested size is less than the original size, the data is truncated and lost.  If the
	*   new size is greater than the original
    *   size, the extra paddign will be initialized with the value specified by the user.
    *   \param reqSize The requested size of the device_vector in elements.
    *   \param val All new elements are initialized with this new value.
    *   \note capacity( ) may exceed n, but is not less than n.
    *   \warning If the device_vector must reallocate, all previous iterators, references, and pointers are invalidated.
    *   \warning The ::cl::CommandQueue is not a STD reserve( ) parameter
    */


    void resize( size_type reqSize, const value_type& val = value_type( ) )
    {
        size_type cap = capacity( );

        if( reqSize == cap )
            return;

        //TODO - Add if statement for max size allowed in array class
        array_type l_tmpArray = array_type(reqSize);
        arrayview_type l_tmpBuffer = arrayview_type(l_tmpArray);
        if( m_Size > 0 )
        {
            //1622 Arrays are logically considered to be value types in that when an array is copied to another array,
            //a deep copy is performed. Two arrays never point to the same data.
            //m_Size data elements are copied

            if( reqSize > m_Size )
            {
                m_devMemory.copy_to(l_tmpBuffer.section( 0, m_devMemory.get_extent().size() ) );
                arrayview_type l_tmpBufferSectionAV =
                l_tmpBuffer.section(m_Size, (reqSize - m_Size));
                concurrency::parallel_for_each(l_tmpBufferSectionAV.get_extent(), [=](Concurrency::index<1> idx) restrict(amp)
                {
                    l_tmpBufferSectionAV[idx] = val;
                });
            }
            else
            {
                arrayview_type l_devMemoryAV = m_devMemory.section(0, reqSize);
                l_devMemoryAV.copy_to(l_tmpBuffer);
            }
        }
        else
        {
            arrayview_type l_tmpBufferAV(l_tmpBuffer);
            Concurrency::parallel_for_each(l_tmpBufferAV.get_extent(), [=](Concurrency::index<1> idx) restrict(amp)
            {
                l_tmpBufferAV[idx] = val;
            });
        }
        //  Remember the new size
        m_Size = reqSize;
        m_devMemory = l_tmpBuffer;
    }

    /*! \brief Return the number of known elements
    *   \note size( ) differs from capacity( ), in that size( ) returns the number of elements between begin() & end()
    *   \return Number of valid elements
    */
    size_type size( void ) const restrict(amp, cpu)
    {
        return m_Size;
    }

  

    /*! \brief Request a change in the capacity of the device_vector.
    *   If reserve completes successfully,
    *   this device_vector object guarantees that the it can store the requested amount
    *   of elements without another reallocation, until the device_vector size exceeds n.
    *   \param n The requested size of the device_vector in elements
    *   \note capacity( ) may exceed n, but will not be less than n.
    *   \note Contents are preserved, and the size( ) of the vector is not affected.
    *   \warning if the device_vector must reallocate, all previous iterators, references, and pointers are invalidated.
    *   \warning The ::cl::CommandQueue is not a STD reserve( ) parameter
    *   \TODO what if reqSize < the size of the original buffer
    */

    void reserve( size_type reqSize )
    {
       if( reqSize <= capacity( ) )
           return;

		concurrency::array<value_type> tmp =  array_type( static_cast< int >( reqSize ) );
		arrayview_type l_tmpBuffer = arrayview_type( tmp );
		if( m_Size > 0 )
        {
			if( capacity() != 0 )
			{
				arrayview_type l_tmpBuffer = arrayview_type( tmp );
				m_devMemory.copy_to(l_tmpBuffer.section(0, m_devMemory.get_extent().size()));
			}
		}
		m_devMemory = l_tmpBuffer;

    }

    /*! \brief Return the maximum possible number of elements without reallocation.
    *   \note Capacity() differs from size(), in that capacity() returns the number of elements that \b could be stored
    *   in the memory currently allocated.
    *   \return The size of the memory held by device_vector, counted in elements.
    */
    size_type capacity( void ) const
    {
        Concurrency::extent<1> ext = m_devMemory.get_extent();
        return ext.size();
    }

    /*! \brief Shrink the capacity( ) of this device_vector to just fit its elements.
    *   This makes the size( ) of the vector equal to its capacity( ).
    *   \note Contents are preserved.
    *   \warning if the device_vector must reallocate, all previous iterators, references, and pointers are invalidated.
    */
    void shrink_to_fit( )
    {   
        if( m_Size == capacity( ) )
             return;

        array_type l_tmpArray = array_type( static_cast< int >( size( ) ) );
        arrayview_type l_tmpBuffer = arrayview_type(l_tmpArray);
        arrayview_type l_devMemoryAV = m_devMemory.section( 0,(int)size() );
        arrayview_type l_tmpBufferAV = l_tmpBuffer.section( 0,(int)size() );

        l_devMemoryAV.copy_to( l_tmpBufferAV );
        m_devMemory = l_tmpBuffer;
    }

    /*! \brief Retrieves the value stored at index n.
    *   \return Returns a proxy reference object, to control when device memory gets mapped.
    */
    value_type& operator[]( size_type n ) restrict(cpu,amp)
    {
        return m_devMemory[n];
    }

    /*! \brief Retrieves a constant value stored at index n.
    *   \return Returns a const_reference, which is not a proxy object.
    */
    value_type& operator[]( size_type ix ) const restrict(cpu,amp)
    {
        return m_devMemory[ix];
    }

    /*! \brief Retrieves an iterator for this container that points at the beginning element.
    *   \return A device_vector< value_type >::iterator.
    */
    iterator begin( void )
    {
        return iterator( *this, 0 );
    }

    /*! \brief Retrieves an iterator for this container that points at the beginning constant element.
    *   No operation through this iterator may modify the contents of the referenced container.
    *   \return A device_vector< value_type >::const_iterator
    */
    const_iterator begin( void ) const
    {
        return const_iterator( *this, 0 );
    }

    /*! \brief Retrieves an iterator for this container that points at the beginning constant element.
    *   No operation through this iterator may modify the contents of the referenced container.
    *   \note This method may return a constant iterator from a non-constant container.
    *   \return A device_vector< value_type >::const_iterator.
    */
    const_iterator cbegin( void ) const
    {
        return const_iterator( *this, 0 );
    }

    /*! \brief Retrieves a reverse_iterator for this container that points at the last element.
    *   \return A device_vector< value_type >::reverse_iterator.
    */

    reverse_iterator rbegin( void )
    {
        return reverse_iterator(*this,m_Size);
    }

    /*! \brief Retrieves a reverse_iterator for this container that points at the last constant element.
    *   No operation through this iterator may modify the contents of the referenced container.
    *   \return A device_vector< value_type >::const_reverse_iterator
    */

    const_reverse_iterator rbegin( void ) const
    {
        return const_reverse_iterator(*this,m_Size);
    }

    /*! \brief Retrieves an iterator for this container that points at the last constant element.
    *   No operation through this iterator may modify the contents of the referenced container.
    *   \note This method may return a constant iterator from a non-constant container.
    *   \return A device_vector< value_type >::const_reverse_iterator.
    */

    const_reverse_iterator crbegin( void ) const
    {
        return const_reverse_iterator(*this,m_Size);
    }

    /*! \brief Retrieves an iterator for this container that points at the last element.
    *   \return A device_vector< value_type >::iterator.
    */
    iterator end( void )
    {
        return iterator( *this, m_Size );
    }

    /*! \brief Retrieves an iterator for this container that points at the last constant element.
    *   No operation through this iterator may modify the contents of the referenced container.
    *   \return A device_vector< value_type >::const_iterator.
    */
    const_iterator end( void ) const
    {
        return const_iterator( *this, m_Size );
    }

    /*! \brief Retrieves an iterator for this container that points at the last constant element.
    *   No operation through this iterator may modify the contents of the referenced container.
    *   \note This method may return a constant iterator from a non-constant container.
    *   \return A device_vector< value_type >::const_iterator.
    */
    const_iterator cend( void ) const
    {
        return const_iterator( *this, m_Size );
    }

    /*! \brief Retrieves a reverse_iterator for this container that points at the beginning element.
    *   \return A device_vector< value_type >::reverse_iterator.
    */

    reverse_iterator rend( void )
    {
        return reverse_iterator( *this, 0  );
    }

    /*! \brief Retrieves a reverse_iterator for this container that points at the beginning constant element.
    *   No operation through this iterator may modify the contents of the referenced container.
    *   \return A device_vector< value_type >::const_reverse_iterator.
    */

    const_reverse_iterator rend( void ) const
    {
        return const_reverse_iterator( *this, 0  );
    }

    /*! \brief Retrieves a reverse_iterator for this container that points at the beginning constant element.
    *   No operation through this iterator may modify the contents of the referenced container.
    *   \note This method may return a constant iterator from a non-constant container.
    *   \return A device_vector< value_type >::const_reverse_iterator.
    */

    const_reverse_iterator crend( void ) const
    {
        return const_reverse_iterator( *this, 0  );
    }

    /*! \brief Retrieves the value stored at index 0.
    *   \note This returns a proxy object, to control when device memory gets mapped.
    */
    value_type& front( void )
    {
		return (*begin());
    }

    /*! \brief Retrieves the value stored at index 0.
    *   \return Returns a const_reference, which is not a proxy object.
    */
    const value_type& front( void ) const
    {
        return (*begin());
    }

    /*! \brief Retrieves the value stored at index size( ) - 1.
    *   \note This returns a proxy object, to control when device memory gets mapped.
    */
    value_type& back( void )
    {
		return (*(end() - 1));
    }

    /*! \brief Retrieves the value stored at index size( ) - 1.
    *   \return Returns a const_reference, which is not a proxy object.
    */
    const value_type& back( void ) const
    {
		return ( *(end() - 1) );
    }

    //Yes you need the shared_array object.
    //Ask kent for a better solution.
    pointer data( void )
    {
        /// \TODO need to understand what Array_view.data is returning. Who should free the pointer?
        // below av.data(). It should anyway be freed in the UnMapBufferFunctor Functor
		if(0 == size())
        {

             return NULL;
        }
        synchronize( *this );
        arrayview_type av( m_devMemory );
        return av.data( );
    }

    const_pointer data( void ) const
    {
		if(0 == size())
        {
             return NULL;
        }
		synchronize( *this );
        arrayview_type av( m_devMemory );
        return av.data( );
    }

    /*! \brief Removes all elements (makes the device_vector empty).
    *   \note All previous iterators, references and pointers are invalidated.
    */
    void clear( void )
    {   
        m_devMemory = create_empty_array_view<value_type>::getav();
        m_Size = 0;
    }

    /*! \brief Test whether the container is empty
    *   \return Returns true if size( ) == 0
    */
    bool empty( void ) const
    {
        return m_Size ? false: true;
    }

    /*! \brief Appends a copy of the value to the container
     *  \param value The element to append
    */
    void push_back( const value_type& value )
    {
        if( m_Size > capacity( ) )
            throw Concurrency::runtime_exception( "device_vector size can not be greater than capacity( )", 0);

        //  Need to grow the vector to push new value.
        //  Vectors double their capacity on push_back if the array is not big enough.
        if( m_Size == capacity( ) )
        {
            m_Size ? reserve( m_Size * 2 ) : reserve( 1 );
        }

        arrayview_type av( m_devMemory );
        //insert(end(),value);
        av[static_cast<int>( m_Size )] = value;
        ++m_Size;
    }

    /*! \brief Removes the last element, but does not return it.
    */
    void pop_back( void )
    {
        if( m_Size > 0 )
        {
            --m_Size;
        }
    }

    /*! \brief Swaps the contents of two device_vectors in an efficient manner.
     *  \param vec The device_vector to swap with.
    */
    void swap( device_vector& vec )
    {
        if( this == &vec )
            return;

        arrayview_type swapBuffer( m_devMemory );
        m_devMemory = vec.m_devMemory;
        vec.m_devMemory = swapBuffer;

        size_type sizeTmp = m_Size;
        m_Size = vec.m_Size;
        vec.m_Size = static_cast<int>(sizeTmp);
    }

    /*! \brief Removes an element.
     *  \param index The iterator position in which to remove the element.
    *   \return The iterator position after the deleted element.
    */
    iterator erase( const_iterator index )
    {
        iterator l_End = end( );
        if( index.m_Index >= l_End.m_Index )
            throw Concurrency::runtime_exception( "Iterator is pointing past the end of this container", 0);

        size_type sizeRegion = l_End.m_Index - index.m_Index;

        arrayview_type av( m_devMemory );
        naked_pointer ptrBuff = av.data();
        naked_pointer ptrBuffTemp = ptrBuff + index.m_Index;
        ::memmove( ptrBuffTemp, ptrBuffTemp + 1, (sizeRegion - 1)*sizeof( value_type ) );

        --m_Size;

        size_type newIndex = (m_Size < index.m_Index) ? m_Size : index.m_Index;
        return iterator( *this, newIndex );
    }

    /*! \brief Removes a range of elements.
     *  \param begin The iterator position signifiying the beginning of the range.
     *  \param end The iterator position signifying the end of the range (exclusive).
    *   \return The iterator position after the deleted range.
    */
    iterator erase( const_iterator first, const_iterator last )
    {
        if( last.m_Index > m_Size )
            throw Concurrency::runtime_exception( "Iterator is pointing past the end of this container" , 0);

        if( (first == begin( )) && (last == end( )) )
        {
            clear( );
            return iterator( *this, m_Size );
        }

        iterator l_End = end( );
        size_type sizeMap = l_End.m_Index - first.m_Index;

        arrayview_type av( m_devMemory );
        naked_pointer ptrBuff = av.data();
        ptrBuff = ptrBuff + first.m_Index;
        size_type sizeErase = last.m_Index - first.m_Index;
        ::memmove( ptrBuff, ptrBuff + sizeErase, (sizeMap - sizeErase)*sizeof( value_type ) );

        m_Size -= static_cast<int>(sizeErase);

        size_type newIndex = (m_Size < last.m_Index) ? m_Size : last.m_Index;
        return iterator( *this, newIndex );
    }

    /*! \brief Insert a new element into the container.
     *  \param index The iterator position to insert a copy of the element.
     *  \param value The element to insert.
    *   \return The position of the new element.
    *   \note Only iterators before the insertion point remain valid after the insertion.
    *   \note If the container must grow to contain the new value, all iterators and references are invalidated.
    */
    iterator insert( const_iterator index, const value_type& value )
    {
        if( index.m_Index > m_Size )
            throw Concurrency::runtime_exception( "Iterator is pointing past the end of this container", 0);

        if( index.m_Index == m_Size )
        {
            push_back( value );
            return iterator( *this, index.m_Index );
        }

        //  Need to grow the vector to insert a new value.
        //  TODO:  What is an appropriate growth strategy for GPU memory allocation?  Exponential growth does not seem
        //  right at first blush.
        if( m_Size == capacity( ) )
        {
            m_Size ? reserve( m_Size * 2 ) : reserve( 1 );
        }

        size_type sizeMap = (m_Size - index.m_Index) + 1;

        arrayview_type av( m_devMemory );
        naked_pointer ptrBuff = av.data();
        ptrBuff = ptrBuff + index.m_Index;

        //  Shuffle the old values 1 element down
        ::memmove( ptrBuff + 1, ptrBuff, (sizeMap - 1)*sizeof( value_type ) );

        //  Write the new value in its place
        *ptrBuff = value;

        ++m_Size;

        return iterator( *this, index.m_Index );
    }

    /*! \brief Inserts n copies of the new element into the container.
     *  \param index The iterator position to insert n copies of the element.
     *  \param n The number of copies of element.
     *  \param value The element to insert.
     *  \note Only iterators before the insertion point remain valid after the insertion.
     *  \note If the container must grow to contain the new value, all iterators and references are invalidated.
     */
    void insert( const_iterator index, size_type n, const value_type& value )
    {
        if( index.m_Index > m_Size )
            throw Concurrency::runtime_exception(  "Iterator is pointing past the end of this container" , 0);

        //  Need to grow the vector to insert n new values
        if( ( m_Size + n ) > capacity( ) )
        {
            reserve( m_Size + n );
        }

        size_type sizeMap = (m_Size - index.m_Index) + n;

        arrayview_type av( m_devMemory );
        naked_pointer ptrBuff = av.data( );
        ptrBuff = ptrBuff + index.m_Index;

        //  Shuffle the old values n element down.
        ::memmove( ptrBuff + n, ptrBuff, (sizeMap - n)*sizeof( value_type ) );

        //  Copy the new value n times in the buffer.
        for( size_type i = 0; i < n; ++i )
        {
            ptrBuff[ i ] = value;
        }

        m_Size += n;
    }

    template< typename InputIterator >
    void insert( const_iterator index, InputIterator begin, InputIterator end )
    {
        if( index.m_Index > m_Size )
            throw Concurrency::runtime_exception(  "Iterator is pointing past the end of this container", 0);

        //  Need to grow the vector to insert the range of new values
        size_type n = static_cast<int>(std::distance( begin, end ));
        if( ( m_Size + n ) > capacity( ) )
        {
            reserve( m_Size + n );
        }
        size_type sizeMap = (m_Size - index.m_Index) + n;

        arrayview_type av( m_devMemory );
        naked_pointer ptrBuff = av.data() + index.m_Index;

        //  Shuffle the old values n element down.
        ::memmove( ptrBuff + n, ptrBuff, (sizeMap - n)*sizeof( value_type ) );

#if( _WIN32 )
        std::copy( begin, end, stdext::checked_array_iterator< naked_pointer >( ptrBuff, n ) );
#else
        std::copy( begin, end, ptrBuff );
#endif

        m_Size += static_cast<int>(n);
    }

    /*! \brief Assigns newSize copies of element value.
     *  \param newSize The new size of the device_vector.
     *  \param value The value of the element that is replicated newSize times.
    *   \warning All previous iterators, references, and pointers are invalidated.
    */
    void assign( size_type newSize, const value_type& value )
    {
        if( newSize > m_Size )
        {
            reserve( newSize );
        }
        m_Size = newSize;

        arrayview_type m_devMemoryAV( m_devMemory );
        Concurrency::parallel_for_each( m_devMemoryAV.get_extent(), [=](Concurrency::index<1> idx) restrict(amp)
        {
            m_devMemoryAV[idx] = value;
        }
        );
    }

    /*! \brief Assigns a range of values to device_vector, replacing all previous elements.
     *  \param begin The iterator position signifiying the beginning of the range.
     *  \param end The iterator position signifying the end of the range (exclusive).
    *   \warning All previous iterators, references, and pointers are invalidated.
    */
	template<typename InputIterator>
#ifdef _WIN32	
    typename std::enable_if< std::_Is_iterator<InputIterator>::value, void>::type
#else
    typename std::enable_if< is_iterator<InputIterator>::value, void>::type
#endif
    assign( InputIterator begin, InputIterator end )
    {
        size_type l_Count = static_cast<int>(std::distance( begin, end ));

        if( l_Count > m_Size )
        {
            reserve( l_Count );
        }
        m_Size = static_cast<int>(l_Count);

        arrayview_type m_devMemoryAV( m_devMemory );
        naked_pointer ptrBuff = m_devMemoryAV.data();
#if( _WIN32 )
        std::copy( begin, end, stdext::checked_array_iterator< naked_pointer >( ptrBuff, m_Size ) );
#else
        std::copy( begin, end, ptrBuff );
#endif
    }

private:

    //  These private routines make sure that the data that resides in the concurrency::array* object are
    //  reflected back in the host memory.  However, the complication is that the concurrency::array object
    //  does not expose a synchronize method, whereas the concurrency::array_view does.  These routines
    //  differentiate between the two different containers
    void synchronize( device_vector< T, concurrency::array >& rhs ) const
    {
    };
    void synchronize( const device_vector< T, concurrency::array >& rhs ) const
    {
    };
    void synchronize( device_vector< T, concurrency::array_view >& rhs ) const
    {
        rhs.m_devMemory.synchronize( );
    };
    void synchronize( const device_vector< T, concurrency::array_view >& rhs ) const
    {
        rhs.m_devMemory.synchronize( );
    };

    int m_Size;
    concurrency::array_view<T, 1> m_devMemory;
};

}
}

#endif
