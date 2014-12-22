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

#if !defined( BOLT_AMP_GENERATE_INL )
#define BOLT_AMP_GENERATE_INL
#define GEN_WAVEFRONT_SIZE 256

#pragma once

#include <algorithm>
#include <type_traits>
#include "bolt/amp/bolt.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include "bolt/amp/device_vector.h"
#include <amp.h>

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/generate.h"
#endif

namespace bolt {
namespace amp {

namespace detail {

/*****************************************************************************
* Enqueue
****************************************************************************/

template< typename DVForwardIterator, typename Generator >
void generate_enqueue(
    bolt::amp::control &ctl,
    const DVForwardIterator &first,
    const DVForwardIterator &last,
    const Generator &gen )
{
	        concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();

	        typedef typename std::iterator_traits<DVForwardIterator>::value_type Type;

            const int szElements =  static_cast< int >( std::distance( first, last ) );

            const unsigned int leng =  szElements + GEN_WAVEFRONT_SIZE - (szElements % GEN_WAVEFRONT_SIZE);

            concurrency::extent< 1 > inputExtent(leng);

            try
            {

                concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
                {

                    int globalId = idx[ 0 ];

                    if( globalId >= szElements)
                         return;

                    first[globalId] =(Type) gen();

                });
            }


			catch(std::exception &e)
            {
                   std::cout << "Exception while calling bolt::amp::generate parallel_for_each " ;
                   std::cout<< e.what() << std::endl;
                   throw std::exception();
            }

}; // end generate_enqueue


/*****************************************************************************
             * Pick Iterator
****************************************************************************/

/*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
               \detail This template is called by the non-detail versions of generate, it already assumes random access
             *  iterators.  This overload is called strictly for non-device_vector iterators
            */
            template<typename ForwardIterator, typename Generator>
            void generate_pick_iterator(bolt::amp::control &ctl,  const ForwardIterator &first,
                const ForwardIterator &last, const Generator &gen, std::random_access_iterator_tag)
            {
                typedef typename std::iterator_traits<ForwardIterator>::value_type Type;

                int sz = static_cast< int >(last - first);
                if (sz < 1)
                    return;

                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
				if (runMode == bolt::amp::control::Automatic)
				{
					runMode = ctl.getDefaultPathToRun();
				}
				
                if( runMode == bolt::amp::control::SerialCpu)
                {
                    std::generate(first, last, gen );
                }
                else if(runMode == bolt::amp::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB   
                        			
                           bolt::btbb::generate(first, last, gen );
                    #else
                           throw std::runtime_error("MultiCoreCPU Version of generate not Enabled! \n");
                    #endif
                }
                else
                {

   
					device_vector< Type, concurrency::array_view> range( first, last, true, ctl);

                    generate_enqueue( ctl, range.begin( ), range.end( ), gen);

                    // This should immediately map/unmap the buffer
                    range.data( );
                }
}

            // This template is called by the non-detail versions of generate,
            // it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVForwardIterator, typename Generator>
            void generate_pick_iterator(bolt::amp::control &ctl, const DVForwardIterator &first,
                const DVForwardIterator &last, const Generator &gen, bolt::amp::device_vector_tag)
            {
                typedef typename std::iterator_traits<DVForwardIterator>::value_type iType;
                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
				if (runMode == bolt::amp::control::Automatic)
				{
					runMode = ctl.getDefaultPathToRun();
				}
              
                if( runMode == bolt::amp::control::SerialCpu)
                {
                    typename bolt::amp::device_vector< iType >::pointer generateInputBuffer =  const_cast<typename bolt::amp::device_vector< iType >::pointer>(first.getContainer( ).data( ));
                    std::generate(&generateInputBuffer[first.m_Index], &generateInputBuffer[last.m_Index], gen );
                }
                else if(runMode == bolt::amp::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB	
                      typename bolt::amp::device_vector< iType >::pointer generateInputBuffer =  const_cast<typename bolt::amp::device_vector< iType >::pointer>(first.getContainer( ).data( ));
                      bolt::btbb::generate(&generateInputBuffer[first.m_Index], &generateInputBuffer[last.m_Index], gen );
                    #else
                        throw std::runtime_error("MultiCoreCPU Version of generate not Enabled! \n");
                    #endif
                }
                else
                {
                    generate_enqueue( ctl, first, last, gen);
                }
            }


            // This template is called by the non-detail versions of generate,
            // it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVForwardIterator, typename Generator>
            void generate_pick_iterator(bolt::amp::control &ctl,  const DVForwardIterator &first,
                const DVForwardIterator &last,
                const Generator &gen, bolt::amp::fancy_iterator_tag )
            {
                static_assert( std::is_same< DVForwardIterator, bolt::amp::fancy_iterator_tag  >::value, "It is not possible to generate into fancy iterators. They are not mutable! " );
            }

/*****************************************************************************
             * Random Access
****************************************************************************/

// generate, yes random-access
template<typename ForwardIterator, typename Generator>
void generate_detect_random_access( bolt::amp::control &ctrl, const ForwardIterator& first,  const ForwardIterator& last,
                        const Generator& gen, std::random_access_iterator_tag )
{
                generate_pick_iterator(ctrl, first, last, gen, 
				typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
}

// generate, not random-access
template<typename ForwardIterator, typename Generator>
void generate_detect_random_access( bolt::amp::control &ctrl, const ForwardIterator& first,  const ForwardIterator& last,
                        const Generator& gen,  std::forward_iterator_tag )
{
                static_assert(std::is_same< ForwardIterator, std::forward_iterator_tag   >::value, "Bolt only supports random access iterator types" );
}



}//End of detail namespace


// default control, start->stop
template<typename ForwardIterator, typename Generator>
void generate( ForwardIterator first, ForwardIterator last,  const Generator & gen)
{
            detail::generate_detect_random_access( bolt::amp::control::getDefault(), first, last, gen, 
            typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
}

// user specified control, start->stop
template<typename ForwardIterator, typename Generator>
void generate( bolt::amp::control &ctl, ForwardIterator first, ForwardIterator last,  const Generator & gen)
{
            detail::generate_detect_random_access( ctl, first, last, gen, 
            typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
}

// default control, start-> +n
template<typename OutputIterator, typename Size, typename Generator>
OutputIterator generate_n( OutputIterator first, Size n, const Generator & gen)
{
            detail::generate_detect_random_access( bolt::amp::control::getDefault(), first, first+static_cast< const int >( n ), gen,
            typename std::iterator_traits< OutputIterator >::iterator_category( ) );
            return (first+static_cast< const int >( n ));
}

// user specified control, start-> +n
template<typename OutputIterator, typename Size, typename Generator>
OutputIterator generate_n( bolt::amp::control &ctl, OutputIterator first, Size n, const Generator & gen)
{
            detail::generate_detect_random_access( ctl, first, first+static_cast< const int >( n ), gen,
            typename std::iterator_traits< OutputIterator >::iterator_category( ) );
            return (first+static_cast< const int >( n ));
}

}//end of amp namespace
};//end of bolt namespace



#endif
