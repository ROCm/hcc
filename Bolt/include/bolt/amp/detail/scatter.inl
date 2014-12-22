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
#if !defined( BOLT_AMP_SCATTER_INL )
#define BOLT_AMP_SCATTER_INL
#define SCATTER_WAVEFRNT_SIZE 264

#include <algorithm>
#include <type_traits>
#include "bolt/amp/bolt.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include "bolt/amp/device_vector.h"
#include <amp.h>

#ifdef ENABLE_TBB
    #include "bolt/btbb/scatter.h"
#endif

namespace bolt {
namespace amp {

namespace detail {


/* Begin-- Serial Implementation of the scatter and scatter_if routines */

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>

void gold_scatter_enqueue (InputIterator1 first1,
                           InputIterator1 last1,
                           InputIterator2 map,
                           OutputIterator result)
    {
       int numElements = static_cast< int >( std::distance( first1, last1 ) );

	   for (int iter = 0; iter<(int)numElements; iter++)
                *(result+*(map + iter)) = *(first1 + iter);
    }

 template< typename InputIterator1,
           typename InputIterator2,
           typename InputIterator3,
           typename OutputIterator >
void gold_scatter_if_enqueue (InputIterator1 first1,
                              InputIterator1 last1,
                              InputIterator2 map,
                              InputIterator3 stencil,
                              OutputIterator result)
   {
       int numElements = static_cast< int >( std::distance( first1, last1 ) );
       for(int iter = 0; iter<numElements; iter++)
        {
             if(stencil[iter] == 1)
                  result[*(map+(iter - 0))] = first1[iter];
             }
   }

 template< typename InputIterator1,
           typename InputIterator2,
           typename InputIterator3,
           typename OutputIterator,
           typename Predicate>
void gold_scatter_if_enqueue (InputIterator1 first1,
                              InputIterator1 last1,
                              InputIterator2 map,
                              InputIterator3 stencil,
                              OutputIterator result,
                              Predicate pred)
   {
       int numElements = static_cast< int >( std::distance( first1, last1 ) );
	   for (int iter = 0; iter< numElements; iter++)
        {
             if(pred(stencil[iter]) != 0)
                  result[*(map+(iter))] = first1[iter];
             }
   }

 /* End-- Serial Implementation of the scatter and scatter_if routines */
 
////////////////////////////////////////////////////////////////////
// ScatterIf enqueue
////////////////////////////////////////////////////////////////////

    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVInputIterator3,
              typename DVOutputIterator,
              typename Predicate >
    void scatter_if_enqueue( bolt::amp::control &ctl,
                             const DVInputIterator1& first1,
                             const DVInputIterator1& last1,
                             const DVInputIterator2& map,
                             const DVInputIterator3& stencil,
                             const DVOutputIterator& result,
                             const Predicate& pred )
    {
		concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();
		typedef typename std::iterator_traits< DVInputIterator1 >::value_type iType1;
		typedef typename std::iterator_traits< DVInputIterator2 >::value_type iType2;		
		typedef typename std::iterator_traits< DVInputIterator3 >::value_type iType3;
		typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        const int szElements = static_cast< int >(  std::distance(first1,last1) );
		const int leng =  szElements + SCATTER_WAVEFRNT_SIZE - (szElements % SCATTER_WAVEFRNT_SIZE);
		concurrency::extent< 1 > inputExtent(leng);
                try
                {
                    concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
                    {
                        int globalId = idx[ 0 ];

                        if( globalId >= szElements)
                        return;
						iType2 m = map[ globalId ];
						iType3 s = stencil[ globalId ];

						if (pred( s ))
						{
							result [ m ] = first1 [ globalId ] ;
						}
						});	
                }
			    catch(std::exception &e)
                {
                      std::cout << "Exception while calling bolt::amp::scatter parallel_for_each"<<e.what()<<std::endl;
                      return;
                }
		result.getContainer().getBuffer(result, szElements).synchronize();
    };

////////////////////////////////////////////////////////////////////
// Scatter enqueue
////////////////////////////////////////////////////////////////////
    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVOutputIterator >
    void scatter_enqueue( bolt::amp::control &ctl,
                          const DVInputIterator1& first1,
                          const DVInputIterator1& last1,
                          const DVInputIterator2& map,
                          const DVOutputIterator& result )
    {
		concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();
		typedef typename std::iterator_traits< DVInputIterator1 >::value_type iType1;
		typedef typename std::iterator_traits< DVInputIterator2 >::value_type iType2;
		typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        const int szElements = static_cast< int >(  std::distance(first1,last1) );
		const unsigned int leng =  szElements + SCATTER_WAVEFRNT_SIZE - (szElements % SCATTER_WAVEFRNT_SIZE);
		concurrency::extent< 1 > inputExtent(leng);
                try
                {
                    concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
                    {
                        int globalId = idx[ 0 ];

                        if( globalId >= szElements)
                        return;
						iType2 m = map[ globalId ];
						result [ m ] = first1 [ globalId ] ;
						});	
                }
			    catch(std::exception &e)
                {
                      std::cout << "Exception while calling bolt::amp::scatter parallel_for_each"<<e.what()<<std::endl;
                      return;
                }
		result.getContainer().getBuffer(result, szElements).synchronize();
	};

////////////////////////////////////////////////////////////////////
// Enqueue ends
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////
// ScatterIf pick iterator
////////////////////////////////////////////////////////////////////

// Host vectors

    /*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
        \detail This template is called by the non-detail versions of inclusive_scan, it already assumes random access
        *  iterators.  This overload is called strictly for non-device_vector iterators
    */
    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
    void scatter_if_pick_iterator( bolt::amp::control &ctl,
                                   const InputIterator1& first1,
                                   const InputIterator1& last1,
                                   const InputIterator2& map,
                                   const InputIterator3& stencil,
                                   const OutputIterator& result,
                                   const Predicate& pred,
                                   std::random_access_iterator_tag,
                                   std::random_access_iterator_tag,
                                   std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        int sz = static_cast< int >(std::distance( first1, last1 ));

        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
           runMode = ctl.getDefaultPathToRun();
        }
		
        if( runMode == bolt::amp::control::SerialCpu )
        {
            gold_scatter_if_enqueue(first1, last1, map, stencil, result, pred);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
           bolt::btbb::scatter_if(first1, last1, map, stencil, result, pred);
#else
          throw std::runtime_error( "The MultiCoreCpu version of scatter_if is not enabled to be built! \n" );

#endif

        }
        else
        {
          // Map the input iterator to a device_vector
		  device_vector< iType1, concurrency::array_view> dvInput( first1, last1, false, ctl );
		  device_vector< iType2, concurrency::array_view> dvMap( map, sz, false, ctl );
		  device_vector< iType3, concurrency::array_view> dvStencil( stencil, sz, false, ctl );

          // Map the output iterator to a device_vector	  
		  device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );
          scatter_if_enqueue( ctl,
                              dvInput.begin( ),
                              dvInput.end( ),
                              dvMap.begin( ),
                              dvStencil.begin( ),
                              dvResult.begin( ),
                              pred );

          // This should immediately map/unmap the buffer
          dvResult.data( );
        }
    }


// Stencil is a fancy iterator
    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
    void scatter_if_pick_iterator( bolt::amp::control &ctl,
                                   const InputIterator1& first1,
                                   const InputIterator1& last1,
                                   const InputIterator2& map,
                                   const InputIterator3& stencilFancyIter,
                                   const OutputIterator& result,
                                   const Predicate& pred,
                                   std::random_access_iterator_tag,
                                   std::random_access_iterator_tag,
                                   bolt::amp::fancy_iterator_tag )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast< int >(std::distance( first1, last1 ));
        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
          runMode = ctl.getDefaultPathToRun();
        }
        if( runMode == bolt::amp::control::SerialCpu )
        {
            gold_scatter_if_enqueue(first1, last1, map, stencilFancyIter, result, pred);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            bolt::btbb::scatter_if(first1, last1, map, stencilFancyIter, result, pred);
#else
            throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );

#endif
            return;
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType1, concurrency::array_view> dvInput( first1, last1, false, ctl );
		    device_vector< iType2, concurrency::array_view> dvMap( map, sz, false, ctl );

            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            scatter_if_enqueue( ctl,
                                dvInput.begin( ),
                                dvInput.end( ),
                                dvMap.begin(),
                                stencilFancyIter,
                                dvResult.begin( ),
                                pred );

            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }

// Input is a fancy iterator
    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
    void scatter_if_pick_iterator( bolt::amp::control &ctl,
                                  const InputIterator1& fancyIterfirst,
                                  const InputIterator1& fancyIterlast,
                                  const InputIterator2& map,
                                  const InputIterator3& stencil,
                                  const OutputIterator& result,
                                  const Predicate& pred,
                                  bolt::amp::fancy_iterator_tag,
                                  std::random_access_iterator_tag,
                                  std::random_access_iterator_tag )
    {

        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast< int >(std::distance( fancyIterfirst, fancyIterlast ));
        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
           runMode = ctl.getDefaultPathToRun();
        }
        if( runMode == bolt::amp::control::SerialCpu )
        {
            gold_scatter_if_enqueue(fancyIterfirst, fancyIterlast, map, stencil, result, pred);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            bolt::btbb::scatter_if(fancyIterfirst, fancyIterlast, map, stencil, result, pred);
#else
            throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );

#endif
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType2, concurrency::array_view> dvMap( map, sz, false, ctl );
		    device_vector< iType3, concurrency::array_view> dvStencil( stencil, sz, false, ctl );

            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            scatter_if_enqueue( ctl,
                               fancyIterfirst,
                               fancyIterlast,
                               dvMap.begin( ),
                               dvStencil.begin( ),
                               dvResult.begin( ),
                               pred );

            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }

// Device Vectors

    // This template is called by the non-detail versions of inclusive_scan, it already assumes random access iterators
    // This is called strictly for iterators that are derived from device_vector< T >::iterator
    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVInputIterator3,
              typename DVOutputIterator,
              typename Predicate >
    void scatter_if_pick_iterator( bolt::amp::control &ctl,
                                   const DVInputIterator1& first1,
                                   const DVInputIterator1& last1,
                                   const DVInputIterator2& map,
                                   const DVInputIterator3& stencil,
                                   const DVOutputIterator& result,
                                   const Predicate& pred,
                                   bolt::amp::device_vector_tag,
                                   bolt::amp::device_vector_tag,
                                   bolt::amp::device_vector_tag )
    {

        typedef typename std::iterator_traits< DVInputIterator1 >::value_type iType1;
        typedef typename std::iterator_traits< DVInputIterator2 >::value_type iType2;
        typedef typename std::iterator_traits< DVInputIterator3 >::value_type iType3;
        typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        int sz = static_cast< int >(std::distance( first1, last1 ));
        if( sz == 0 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
             runMode = ctl.getDefaultPathToRun();
        }
        if( runMode == bolt::amp::control::SerialCpu )
        {
			
            typename bolt::amp::device_vector< iType1 >::pointer firstPtr =  const_cast<typename bolt::amp::device_vector< iType1 >::pointer>(first1.getContainer( ).data( ));
            typename bolt::amp::device_vector< iType2 >::pointer mapPtr =  const_cast<typename bolt::amp::device_vector< iType2 >::pointer>(map.getContainer( ).data( ));
            typename bolt::amp::device_vector< iType3 >::pointer stenPtr =  const_cast<typename bolt::amp::device_vector< iType3 >::pointer>(stencil.getContainer( ).data( ));
            typename bolt::amp::device_vector< oType >::pointer resPtr =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));

#if defined( _WIN32 )
            gold_scatter_if_enqueue(&firstPtr[ first1.m_Index ], &firstPtr[ last1.m_Index ], &mapPtr[ map.m_Index ],
                 &stenPtr[ stencil.m_Index ], stdext::make_checked_array_iterator( &resPtr[ result.m_Index ], sz ), pred );

#else
            gold_scatter_if_enqueue( &firstPtr[ first1.m_Index ], &firstPtr[ last1.m_Index ],
                &mapPtr[ map.m_Index ], &stenPtr[ stencil.m_Index ], &resPtr[ result.m_Index ], pred );
#endif
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
		   	
            // Call MC
#if defined( ENABLE_TBB )
            {
			
                typename bolt::amp::device_vector< iType1 >::pointer firstPtr =  const_cast<typename bolt::amp::device_vector< iType1 >::pointer>(first1.getContainer( ).data( ));
                typename bolt::amp::device_vector< iType2 >::pointer mapPtr =  const_cast<typename bolt::amp::device_vector< iType2 >::pointer>(map.getContainer( ).data( ));
                typename bolt::amp::device_vector< iType3 >::pointer stenPtr =  const_cast<typename bolt::amp::device_vector< iType3 >::pointer>(stencil.getContainer( ).data( ));
                typename bolt::amp::device_vector< oType >::pointer resPtr =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));

                bolt::btbb::scatter_if( &firstPtr[ first1.m_Index ], &firstPtr[ last1.m_Index ],
                &mapPtr[ map.m_Index ], &stenPtr[ stencil.m_Index ], &resPtr[ result.m_Index ], pred );
            }
#else
             throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );
#endif
        }
        else
        {
			
            scatter_if_enqueue( ctl, first1, last1, map, stencil, result, pred );
        }
    }

////////////////////////////////////////////////////////////////////
// Scatter pick iterator
////////////////////////////////////////////////////////////////////

// Host vectors

    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
    void scatter_pick_iterator( bolt::amp::control &ctl,
                                const InputIterator1& first1,
                                const InputIterator1& last1,
                                const InputIterator2& map,
                                const OutputIterator& result,
                                std::random_access_iterator_tag,
                                std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        int sz = static_cast< int >(std::distance( first1, last1 ));

        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
           runMode = ctl.getDefaultPathToRun();
        }
		
        if( runMode == bolt::amp::control::SerialCpu )
        {
		    
            gold_scatter_enqueue(first1, last1, map, result);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
                bolt::btbb::scatter(first1, last1, map, result);
#else
                 throw std::runtime_error( "The MultiCoreCpu version of scatter_if is not enabled to be built! \n" );

#endif
        }

        else
        {
			
          // Map the input iterator to a device_vector
		  device_vector< iType1, concurrency::array_view> dvInput( first1, last1, false, ctl );
		  device_vector< iType2, concurrency::array_view> dvMap( map, sz, false, ctl );

          // Map the output iterator to a device_vector
		  device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );
          scatter_enqueue( ctl,
                           dvInput.begin( ),
                           dvInput.end( ),
                           dvMap.begin( ),
                           dvResult.begin( ) );

          // This should immediately map/unmap the buffer
          dvResult.data( );
        }
    }

// Input is a fancy iterator
    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
    void scatter_pick_iterator( bolt::amp::control &ctl,
                                const InputIterator1& firstFancy,
                                const InputIterator1& lastFancy,
                                const InputIterator2& map,
                                const OutputIterator& result,
                                bolt::amp::fancy_iterator_tag,
                                std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast< int >(std::distance( firstFancy, lastFancy ));
        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
          runMode = ctl.getDefaultPathToRun();
        }
		
        if( runMode == bolt::amp::control::SerialCpu )
        {
             gold_scatter_enqueue(firstFancy, lastFancy, map, result);

        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            bolt::btbb::scatter(firstFancy, lastFancy, map, result);
#else
            throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );

#endif
        }
        else
        {
		  
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType2, concurrency::array_view> dvMap( map, sz, false, ctl );
            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            scatter_enqueue( ctl,
                                firstFancy,
                                lastFancy,
                                dvMap.begin(),
                                dvResult.begin( ));

            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }


// Map is a fancy iterator
    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator>
    void scatter_pick_iterator( bolt::amp::control &ctl,
                                const InputIterator1& first1,
                                const InputIterator1& last1,
                                const InputIterator2& mapFancy,
                                const OutputIterator& result,
                                std::random_access_iterator_tag,
                                 bolt::amp::fancy_iterator_tag )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast< int >(std::distance( first1, last1 ));
        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
          runMode = ctl.getDefaultPathToRun();
        }
		
        if( runMode == bolt::amp::control::SerialCpu )
        {		    
            gold_scatter_enqueue(first1, last1, mapFancy, result);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            bolt::btbb::scatter(first1, last1, mapFancy, result);
#else
            throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );

#endif
        }
        else
        {			
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType1, concurrency::array_view> dvInput( first1, last1, false, ctl );

            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            scatter_enqueue( ctl,
                                dvInput.begin( ),
                                dvInput.end( ),
                                mapFancy,
                                dvResult.begin( ));

            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }

// Device Vectors
    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVOutputIterator >
    void scatter_pick_iterator( bolt::amp::control &ctl,
                                const DVInputIterator1& first1,
                                const DVInputIterator1& last1,
                                const DVInputIterator2& map,
                                const DVOutputIterator& result,
                                bolt::amp::device_vector_tag,
                                bolt::amp::device_vector_tag )
    {

        typedef typename std::iterator_traits< DVInputIterator1 >::value_type iType1;
        typedef typename std::iterator_traits< DVInputIterator2 >::value_type iType2;
        typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        int sz = static_cast< int >(std::distance( first1, last1 ));
        if( sz == 0 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
             runMode = ctl.getDefaultPathToRun();
        }		
        if( runMode == bolt::amp::control::SerialCpu )
        {		    		
            typename bolt::amp::device_vector< iType1 >::pointer InputBuffer  =  const_cast<typename bolt::amp::device_vector< iType1 >::pointer>(first1.getContainer( ).data( ));
            typename bolt::amp::device_vector< iType2 >::pointer MapBuffer    =  const_cast<typename bolt::amp::device_vector< iType2>::pointer>(map.getContainer( ).data( ));
            typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
            gold_scatter_enqueue(&InputBuffer[ first1.m_Index ], &InputBuffer[ last1.m_Index ], &MapBuffer[ map.m_Index ],
            &ResultBuffer[ result.m_Index ]);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {

#if defined( ENABLE_TBB )
            {
              typename bolt::amp::device_vector< iType1 >::pointer InputBuffer   =  const_cast<typename bolt::amp::device_vector< iType1 >::pointer>(first1.getContainer( ).data( ));
              typename bolt::amp::device_vector< iType2 >::pointer MapBuffer     =  const_cast<typename bolt::amp::device_vector< iType2 >::pointer>(map.getContainer( ).data( ));
              typename bolt::amp::device_vector< oType >::pointer ResultBuffer   =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
                bolt::btbb::scatter(&InputBuffer[ first1.m_Index ], &InputBuffer[ last1.m_Index ], &MapBuffer[ map.m_Index ],
                &ResultBuffer[ result.m_Index ]);
            }
#else
             throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );
#endif
         }
        else
        {
            scatter_enqueue( ctl,
                             first1,
                             last1,
                             map,
                             result );
        }
    }

// Input fancy ; Map DV
    template< typename FancyIterator,
              typename DVMapIterator,
              typename DVOutputIterator >
    void scatter_pick_iterator( bolt::amp::control &ctl,
                                const FancyIterator& firstFancy,
                                const FancyIterator& lastFancy,
                                const DVMapIterator& map,
                                const DVOutputIterator& result,
                                bolt::amp::fancy_iterator_tag,
                                bolt::amp::device_vector_tag )
    {

        typedef typename std::iterator_traits< FancyIterator >::value_type iType1;
        typedef typename std::iterator_traits< DVMapIterator >::value_type iType2;
        typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        int sz = static_cast< int >(std::distance( firstFancy, lastFancy ));
        if( sz == 0 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
             runMode = ctl.getDefaultPathToRun();
        }
        if( runMode == bolt::amp::control::SerialCpu )
        {			
            typename bolt::amp::device_vector< iType2 >::pointer MapBuffer    =  const_cast<typename bolt::amp::device_vector< iType2 >::pointer>(map.getContainer( ).data( ));
            typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
            gold_scatter_enqueue(firstFancy, lastFancy, &MapBuffer[ map.m_Index ], &ResultBuffer[ result.m_Index ]);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            {
                typename bolt::amp::device_vector< iType2 >::pointer MapBuffer =  const_cast<typename bolt::amp::device_vector< iType2 >::pointer>(map.getContainer( ).data( ));
                typename bolt::amp::device_vector< oType >::pointer ResultBuffer =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));

                bolt::btbb::scatter(firstFancy, lastFancy, &MapBuffer[ map.m_Index ],&ResultBuffer[ result.m_Index ]);
            }
#else
             throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );
#endif
        }
        else
        {			
            scatter_enqueue( ctl,
                             firstFancy,
                             lastFancy,
                             map,
                             result );
        }
    }

// Map fancy ; Input DV
    template< typename DVInputIterator,
              typename FancyMapIterator,
              typename DVOutputIterator >
    void scatter_pick_iterator( bolt::amp::control &ctl,
                                const DVInputIterator& first1,
                                const DVInputIterator& last1,
                                const FancyMapIterator& mapFancy,
                                const DVOutputIterator& result,
                                bolt::amp::device_vector_tag,
                                bolt::amp::fancy_iterator_tag )
    {
        typedef typename std::iterator_traits< DVInputIterator >::value_type iType1;
        typedef typename std::iterator_traits< FancyMapIterator >::value_type iType2;
        typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        int sz = static_cast< int >(std::distance( first1, last1 ));
        if( sz == 0 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
             runMode = ctl.getDefaultPathToRun();
        }
        if( runMode == bolt::amp::control::SerialCpu )
        {
            typename bolt::amp::device_vector< iType1 >::pointer InputBuffer    =  const_cast<typename bolt::amp::device_vector< iType1 >::pointer>(first1.getContainer( ).data( ));
            typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
            gold_scatter_enqueue( &InputBuffer[ first1.m_Index ], &InputBuffer[ last1.m_Index ], mapFancy,
                                                                         &ResultBuffer[ result.m_Index ]);

        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            {
                typename bolt::amp::device_vector< iType1 >::pointer InputBuffer    =  const_cast<typename bolt::amp::device_vector< iType1 >::pointer>(first1.getContainer( ).data( ));
                typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
                bolt::btbb::scatter( &InputBuffer[ first1.m_Index ], &InputBuffer[ last1.m_Index ], mapFancy,
                                                                            &ResultBuffer[ result.m_Index ]);
            }
#else
             throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );
#endif
        }
        else
        {
            scatter_enqueue( ctl,
                             first1,
                             last1,
                             mapFancy,
                             result );
        }
    }

// Input DV ; Map random access
    template< typename DVInputIterator,
              typename MapIterator,
              typename OutputIterator >
    void scatter_pick_iterator( bolt::amp::control &ctl,
                                const DVInputIterator& first,
                                const DVInputIterator& last,
                                const MapIterator& map,
                                const OutputIterator& result,
                                bolt::amp::device_vector_tag,
                                std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits<DVInputIterator>::value_type iType1;
        typedef typename std::iterator_traits<MapIterator>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast< int >(std::distance( first, last ));
        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
          runMode = ctl.getDefaultPathToRun();
        }
        if( runMode == bolt::amp::control::SerialCpu )
        {
            typename bolt::amp::device_vector< iType1 >::pointer InputBuffer    =  const_cast<typename bolt::amp::device_vector< iType1 >::pointer>(first.getContainer( ).data( ));
            gold_scatter_enqueue( &InputBuffer[ first.m_Index ], &InputBuffer[ last.m_Index ], map,result);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            {
                typename bolt::amp::device_vector< iType1 >::pointer InputBuffer    =  const_cast<typename bolt::amp::device_vector< iType1 >::pointer>(first.getContainer( ).data( ));
                bolt::btbb::scatter( &InputBuffer[ first.m_Index ], &InputBuffer[ last.m_Index ],map, result);
            }
#else
            throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );

#endif
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the map iterator to a device_vector
		    device_vector< iType2, concurrency::array_view> dvMap( map, sz, false, ctl );
            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            scatter_enqueue( ctl,
                             first,
                             last,
                             dvMap.begin(),
                             dvResult.begin( ) );

            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }

// DV Map ; RA Input
    template< typename InputIterator,
              typename DVMapIterator,
              typename OutputIterator>
    void scatter_pick_iterator( bolt::amp::control &ctl,
                                const InputIterator& first1,
                                const InputIterator& last1,
                                const DVMapIterator& map,
                                const OutputIterator& result,
                                std::random_access_iterator_tag,
                                bolt::amp::device_vector_tag )
    {
        typedef typename std::iterator_traits<InputIterator>::value_type iType1;
        typedef typename std::iterator_traits<DVMapIterator>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast< int >(std::distance( first1, last1 ));
        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
          runMode = ctl.getDefaultPathToRun();
        }
        if( runMode == bolt::amp::control::SerialCpu )
        {
           typename bolt::amp::device_vector< iType2 >::pointer mapBuffer    =  const_cast<typename bolt::amp::device_vector< iType2 >::pointer>(map.getContainer( ).data( ));
           gold_scatter_enqueue(first1, last1, &mapBuffer[ map.m_Index ], result);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            {
                typename bolt::amp::device_vector< iType2 >::pointer mapBuffer    =  const_cast<typename bolt::amp::device_vector< iType2 >::pointer>(map.getContainer( ).data( ));
                bolt::btbb::scatter(first1, last1, &mapBuffer[ map.m_Index ], result);
            }
#else
            throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );

#endif
        }
        else
        {				
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType1, concurrency::array_view> dvInput( first1, last1, false, ctl );
            // Map the result iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            scatter_enqueue( ctl,
                             dvInput.begin(),
                             dvInput.end(),
                             map,
                             dvResult.begin( ));

            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }






////////////////////////////////////////////////////////////////////
// ScatterIf detect random access
////////////////////////////////////////////////////////////////////


    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
    void scatter_if_detect_random_access( bolt::amp::control& ctl,
                                          const InputIterator1& first1,
                                          const InputIterator1& last1,
                                          const InputIterator2& map,
                                          const InputIterator3& stencil,
                                          const OutputIterator& result,
                                          const Predicate& pred,
                                          std::random_access_iterator_tag,
                                          std::random_access_iterator_tag,
                                          std::random_access_iterator_tag )
    {
       scatter_if_pick_iterator( ctl,
                                 first1,
                                 last1,
                                 map,
                                 stencil,
                                 result,
                                 pred,
                                 typename  std::iterator_traits< InputIterator1 >::iterator_category( ),
                                 typename  std::iterator_traits< InputIterator2 >::iterator_category( ),
                                 typename  std::iterator_traits< InputIterator3 >::iterator_category( ));
    };






////////////////////////////////////////////////////////////////////
// Scatter detect random access
////////////////////////////////////////////////////////////////////



    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
    void scatter_detect_random_access( bolt::amp::control& ctl,
                                       const InputIterator1& first1,
                                       const InputIterator1& last1,
                                       const InputIterator2& map,
                                       const OutputIterator& result,
                                       std::random_access_iterator_tag,
                                       std::random_access_iterator_tag )
    {
       scatter_pick_iterator( ctl,
                              first1,
                              last1,
                              map,
                              result,
                              typename  std::iterator_traits< InputIterator1 >::iterator_category( ),
                              typename  std::iterator_traits< InputIterator2 >::iterator_category( ) );
    };


    // Wrapper that uses default ::bolt::amp::control class, iterator interface
    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
    void scatter_if_detect_random_access( bolt::amp::control& ctl,
                                          const InputIterator1& first1,
                                          const InputIterator1& last1,
                                          const InputIterator2& map,
                                          const InputIterator3& stencil,
                                          const OutputIterator& result,
                                          const Predicate& pred,
                                          std::input_iterator_tag,
                                          std::input_iterator_tag,
                                          std::input_iterator_tag )
    {
            // TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
            // to a temporary buffer.  Should we?

            static_assert( std::is_same< InputIterator1, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
    };

    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator>
    void scatter_detect_random_access( bolt::amp::control& ctl,
                                       const InputIterator1& first1,
                                       const InputIterator1& last1,
                                       const InputIterator2& map,
                                       const OutputIterator& result,
                                       std::input_iterator_tag,
                                       std::input_iterator_tag )
    {
            static_assert( std::is_same< InputIterator1, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
    };

} //End of detail namespace

////////////////////////////////////////////////////////////////////
// Scatter APIs
////////////////////////////////////////////////////////////////////
template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
void scatter( bolt::amp::control& ctl,
              InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 map,
              OutputIterator result )
{
    detail::scatter_detect_random_access( ctl,
                                          first1,
                                          last1,
                                          map,
                                          result,
                                          typename  std::iterator_traits< InputIterator1 >::iterator_category( ),
                                          typename  std::iterator_traits< InputIterator2 >::iterator_category( ) );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
void scatter( InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 map,
              OutputIterator result )
{
    detail::scatter_detect_random_access( control::getDefault( ),
                                          first1,
                                          last1,
                                          map,
                                          result,
                                          typename  std::iterator_traits< InputIterator1 >::iterator_category( ),
                                          typename  std::iterator_traits< InputIterator2 >::iterator_category( ) );
}


////////////////////////////////////////////////////////////////////
// ScatterIf APIs
////////////////////////////////////////////////////////////////////
template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >
void scatter_if( bolt::amp::control& ctl,
                 InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 map,
                 InputIterator3 stencil,
                 OutputIterator result )
{
    typedef typename  std::iterator_traits<InputIterator3>::value_type stencilType;
    scatter_if( ctl,
                first1,
                last1,
                map,
                stencil,
                result,
                bolt::amp::identity <stencilType> ( ));
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >
void scatter_if( InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 map,
                 InputIterator3 stencil,
                 OutputIterator result)
{
    typedef typename  std::iterator_traits<InputIterator3>::value_type stencilType;
    scatter_if( first1,
                last1,
                map,
                stencil,
                result,
                bolt::amp::identity <stencilType> ( ) );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
void scatter_if( bolt::amp::control& ctl,
                 InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 map,
                 InputIterator3 stencil,
                 OutputIterator result,
                 Predicate pred )
{
    detail::scatter_if_detect_random_access( ctl,
                                             first1,
                                             last1,
                                             map,
                                             stencil,
                                             result,
                                             pred,
                                             typename  std::iterator_traits< InputIterator1 >::iterator_category( ),
                                             typename  std::iterator_traits< InputIterator2 >::iterator_category( ),
                                             typename  std::iterator_traits< InputIterator3 >::iterator_category( ) );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
void scatter_if( InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 map,
                 InputIterator3 stencil,
                 OutputIterator result,
                 Predicate pred )
{
    detail::scatter_if_detect_random_access( control::getDefault( ),
                                             first1,
                                             last1,
                                             map,
                                             stencil,
                                             result,
                                             pred,
                                             typename  std::iterator_traits< InputIterator1 >::iterator_category( ),
                                             typename  std::iterator_traits< InputIterator2 >::iterator_category( ),
                                             typename  std::iterator_traits< InputIterator3 >::iterator_category( ));
}



} //End of amp namespace
} //End of bolt namespace

#endif
