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
#if !defined( BOLT_AMP_FILL_INL )
#define BOLT_AMP_FILL_INL
#define FILL_WAVEFRONT_SIZE 256

#include <algorithm>
#include <type_traits>
#include "bolt/amp/bolt.h"
#include "bolt/amp/device_vector.h"
#include <amp.h>


//TBB Includes
#ifdef ENABLE_TBB
#include "bolt/btbb/fill.h"
#endif


namespace bolt {
    namespace amp {


namespace detail {

            /*****************************************************************************
            * Fill Enqueue
            ****************************************************************************/

            template< typename DVForwardIterator, typename T >
            void fill_enqueue(bolt::amp::control &ctl, const DVForwardIterator &first,
            const DVForwardIterator &last, const T & val)
            {
                
				concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();

				typedef typename std::iterator_traits<DVForwardIterator>::value_type Type;
				const int szElements =  static_cast< int >( std::distance( first, last ) );
				const unsigned int leng =  szElements + FILL_WAVEFRONT_SIZE - (szElements % FILL_WAVEFRONT_SIZE);

				concurrency::extent< 1 > inputExtent(leng);

				try
				{

					concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
					{

						int globalId = idx[ 0 ];

						if( globalId >= szElements)
							 return;

						first[globalId] =(Type) val;

					});
				}


				catch(std::exception &e)
				{
				std::cout << "Exception while calling bolt::amp::fill parallel_for_each " ;
				std::cout<< e.what() << std::endl;
				throw std::exception();
				}	
									         
			     
            }; // end fill_enqueue



            /*****************************************************************************
             * Pick Iterator
             ****************************************************************************/

        /*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
                \detail This template is called by the non-detail versions of fill, it already assumes random access
             *  iterators.  This overload is called strictly for non-device_vector iterators
        */
            template<typename ForwardIterator, typename T>
			typename std::enable_if<
                   !(std::is_base_of<typename device_vector<typename std::iterator_traits<ForwardIterator>::value_type>::iterator,ForwardIterator>::value),void >::type
            fill_pick_iterator(bolt::amp::control &ctl,  const ForwardIterator &first,
                const ForwardIterator &last, const T & value)
            {


                typedef typename  std::iterator_traits<ForwardIterator>::value_type Type;

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
                     std::fill(first, last, value );
                }
                else if(runMode == bolt::amp::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB
                          bolt::btbb::fill(first, last, value);
                    #else
                          throw std::runtime_error("MultiCoreCPU Version of fill not Enabled! \n");
                    #endif
                }
                else
                {
					    std::fill(first, last, value );
					 
                }

            }

            // This template is called by the non-detail versions of fill, it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVForwardIterator, typename T>
			typename std::enable_if<
                   (std::is_base_of<typename device_vector<typename std::iterator_traits< DVForwardIterator>::value_type>::iterator, DVForwardIterator>::value),void >::type
            fill_pick_iterator( bolt::amp::control &ctl,  const DVForwardIterator &first,
                const DVForwardIterator &last,  const T & value)
            {
				int sz = static_cast< int >(std::distance( first, last ));
                if( sz == 0 )
                    return;

                typedef typename std::iterator_traits<DVForwardIterator>::value_type iType;
                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
				if (runMode == bolt::amp::control::Automatic)
				{
					runMode = ctl.getDefaultPathToRun();
				}
				
                if( runMode == bolt::amp::control::SerialCpu)
                {				
					typename bolt::amp::device_vector< iType >::pointer fillInputBuffer =  first.getContainer( ).data( );
                    std::fill(&fillInputBuffer[first.m_Index], &fillInputBuffer[last.m_Index], (iType) value );
                }
                else if(runMode == bolt::amp::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB
                        typename bolt::amp::device_vector< iType >::pointer fillInputBuffer =  first.getContainer( ).data( );
                        bolt::btbb::fill(&fillInputBuffer[first.m_Index], &fillInputBuffer[last.m_Index], (iType) value );
                    #else
                        throw std::runtime_error("MultiCoreCPU Version of fill not Enabled! \n");
                    #endif
                }
                else
                {
                    fill_enqueue( ctl, first, last, value);
					
                }
            }



            /*****************************************************************************
             * Random Access
             ****************************************************************************/

            // fill no support
            template<typename ForwardIterator, typename T>
            void fill_detect_random_access( bolt::amp::control &ctl, ForwardIterator first, ForwardIterator last,
                const T & value, std::forward_iterator_tag )
            {
                static_assert( std::is_same< ForwardIterator, std::forward_iterator_tag   >::value , "Bolt only supports random access iterator types" );
            }

            // fill random-access
            template<typename ForwardIterator, typename T>
            void fill_detect_random_access( bolt::amp::control &ctl, ForwardIterator first, ForwardIterator last,
                const T & value,  std::random_access_iterator_tag )
            {
                     fill_pick_iterator(ctl, first, last, value);  
            }


        }

        // default control, start->stop
        template< typename ForwardIterator, typename T >
        void fill( ForwardIterator first,
            ForwardIterator last,
            const T & value)
        {
            detail::fill_detect_random_access( bolt::amp::control::getDefault(), first, last, value
                ,typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        }

        // user specified control, start->stop
        template< typename ForwardIterator, typename T >
        void fill(bolt::amp::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const T & value )
        {
            detail::fill_detect_random_access( ctl, first, last, value
                ,typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        }

        // default control, start-> +n
        template< typename OutputIterator, typename Size, typename T >
        OutputIterator fill_n( OutputIterator first,
            Size n,
            const T & value)
        {
            detail::fill_detect_random_access( bolt::amp::control::getDefault(),
                first, first+static_cast< const int >( n ),
                value, typename std::iterator_traits< OutputIterator >::iterator_category( ) );
            return first+static_cast< const int >( n );
        }

        // user specified control, start-> +n
        template<typename OutputIterator, typename Size, typename T>
        OutputIterator fill_n( bolt::amp::control &ctl,
            OutputIterator first,
            Size n,
            const T & value)
        {
            detail::fill_detect_random_access( ctl, first, first+static_cast< const int >( n ), value,
                typename std::iterator_traits< OutputIterator >::iterator_category( ) );
            return (first+static_cast< const int >( n ));
        }

    }//end of amp namespace
};//end of bolt namespace

#endif
