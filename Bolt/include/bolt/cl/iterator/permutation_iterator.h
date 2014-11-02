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

// (C) Copyright Toon Knapen    2001.
// (C) Copyright David Abrahams 2003.
// (C) Copyright Roland Richter 2003.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOLT_PERMUTATION_ITERATOR_H
#define BOLT_PERMUTATION_ITERATOR_H

//#include <iterator>
#include <type_traits>
#include <tuple>
#include <bolt/cl/bolt.h>
#include <bolt/cl/device_vector.h>
#include <bolt/cl/iterator/iterator_adaptor.h>
#include <bolt/cl/iterator/iterator_facade.h>
#include <bolt/cl/iterator/iterator_traits.h>
#include <bolt/cl/iterator/constant_iterator.h>
#include <bolt/cl/iterator/counting_iterator.h>


namespace bolt {
namespace cl {
  struct permutation_iterator_tag
      : public fancy_iterator_tag
        {  };

template< class ElementIterator
        , class IndexIterator>
class permutation_iterator
  : public iterator_adaptor< 
             permutation_iterator<ElementIterator, IndexIterator>
           , IndexIterator, typename bolt::cl::iterator_traits<ElementIterator>::value_type
           , use_default, typename bolt::cl::iterator_traits<ElementIterator>::reference
           , std::ptrdiff_t>
{
  typedef iterator_adaptor< 
            permutation_iterator<ElementIterator, IndexIterator>
          , IndexIterator, typename bolt::cl::iterator_traits<ElementIterator>::value_type
          , use_default, typename bolt::cl::iterator_traits<ElementIterator>::reference
          , std::ptrdiff_t> super_t;

  friend class iterator_core_access;

public:
    typedef std::ptrdiff_t                                           difference_type;
    typedef typename bolt::cl::iterator_traits<ElementIterator>::value_type     value_type;
    typedef typename bolt::cl::iterator_traits<ElementIterator>::value_type *   pointer;
    typedef typename bolt::cl::iterator_traits<ElementIterator>::value_type     index_type;
    typedef permutation_iterator_tag                                 iterator_category;
    typedef std::tuple<value_type *, index_type *>                   tuple;
    permutation_iterator() : m_elt_iter() {}

    explicit permutation_iterator(ElementIterator x, IndexIterator y) 
      : super_t(y), m_elt_iter(x) {}

    template<class OtherElementIterator, class OtherIndexIterator>
    permutation_iterator(
      permutation_iterator<OtherElementIterator, OtherIndexIterator> const& r
      , typename enable_if_convertible<OtherElementIterator, ElementIterator>::type* = 0
      , typename enable_if_convertible<OtherIndexIterator, IndexIterator>::type* = 0
      )
    : super_t(r.base()), m_elt_iter(r.m_elt_iter)
    {}

    index_type* getIndex_pointer ()
    {
        return &(*(this->base_reference())); 
    }

    value_type* getElement_pointer ()
    {
        return &(*(this->m_elt_iter)); 
    }

    struct Payload
    {
        int m_Index;
        int m_ElementPtr[ 3 ];  //Holds pointer to Element Iterator
        int m_IndexPtr[ 3 ];    //Holds pointer to Index Iterator 
    };

    const Payload  gpuPayload( ) const
    {
        Payload payload = { 0/*m_Index*/, { 0, 0, 0 }, { 0, 0, 0 } };
        return payload;
    }

    const difference_type gpuPayloadSize( ) const
    {
        cl_int l_Error = CL_SUCCESS;
        //::cl::Device which_device;
        //l_Error  = m_it.getContainer().m_commQueue.getInfo(CL_QUEUE_DEVICE,&which_device );	
        //TODO - fix the device bits 
        cl_uint deviceBits = 32;// = which_device.getInfo< CL_DEVICE_ADDRESS_BITS >( );
        //  Size of index and pointer
        difference_type payloadSize = sizeof( int ) + 2*( deviceBits >> 3 );

        //  64bit devices need to add padding for 8 byte aligned pointer
        if( deviceBits == 64 )
            payloadSize += 8;

        return payloadSize;
    }

    int setKernelBuffers(int arg_num, ::cl::Kernel &kernel) const
    {
        /*First set the element Iterator*/
        arg_num = m_elt_iter.setKernelBuffers(arg_num, kernel);
        /*Next set the Argument Iterator*/
        arg_num = this->base().setKernelBuffers(arg_num, kernel);
        return arg_num;
    }


private:
    typename super_t::reference dereference() const
        { return *(m_elt_iter + *this->base()); }


public:
    ElementIterator m_elt_iter;
};


template <class ElementIterator, class IndexIterator>
permutation_iterator<ElementIterator, IndexIterator> 
make_permutation_iterator( ElementIterator e, IndexIterator i )
{
    return permutation_iterator<ElementIterator, IndexIterator>( e, i );
}


   //  This string represents the device side definition of the Transform Iterator template
    static std::string devicePermutationIteratorTemplate = 

        bolt::cl::deviceVectorIteratorTemplate +
        bolt::cl::deviceConstantIterator +
        bolt::cl::deviceCountingIterator +
        std::string("#if !defined(BOLT_CL_PERMUTATION_ITERATOR) \n") +
        STRINGIFY_CODE(
            #define BOLT_CL_PERMUTATION_ITERATOR \n
            namespace bolt { namespace cl { \n
            template< typename IndexIterator, typename ElementIterator > \n
            class permutation_iterator \n
            { \n
                public:    \n
                    typedef int iterator_category;        \n
                    typedef typename ElementIterator::value_type value_type; \n
                    typedef typename IndexIterator::value_type index_type; \n
                    typedef int difference_type; \n
                    typedef int size_type; \n
                    typedef value_type* pointer; \n
                    typedef value_type& reference; \n
    
                    permutation_iterator( value_type init ): m_StartIndex( init ), m_Ptr( 0 ) \n
                    {} \n
    
                    void init( global value_type* element_ptr, global index_type* index_ptr )\n
                    { \n
                        m_ElementPtr = element_ptr; \n
                        m_IndexPtr   = index_ptr; \n
                    } \n

                    value_type operator[]( size_type threadID ) const \n
                    { \n
                       return m_ElementPtr[ m_IndexPtr[ m_StartIndex + threadID ] ]; \n
                    } \n

                    value_type operator*( ) const \n
                    { \n
                        return m_ElementPtr[m_IndexPtr[ m_StartIndex + threadID ] ]; \n
                    } \n

                    size_type m_StartIndex; \n
                    global value_type* m_ElementPtr; \n
                    global index_type* m_IndexPtr; \n
            }; \n
            } } \n
        )
        +  std::string("#endif \n"); 

} // End of namespace cl
} // End of namespace bolt
#endif
