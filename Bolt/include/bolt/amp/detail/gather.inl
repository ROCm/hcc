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
#if !defined( BOLT_AMP_GATHER_INL )
#define BOLT_AMP_GATHER_INL
#define GATHER_WAVEFRNT_SIZE 64

#include <algorithm>
#include <type_traits>
#include "bolt/amp/bolt.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include "bolt/amp/device_vector.h"
#include <amp.h>

#ifdef ENABLE_TBB
    #include "bolt/btbb/gather.h"
#endif

namespace bolt {
namespace amp {

namespace detail {

/* Begin-- Serial Implementation of the gather and gather_if routines */



template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >

void serial_gather(InputIterator1 mapfirst,
                   InputIterator1 maplast,
                   InputIterator2 input,
                   OutputIterator result)
{
    //std::cout<<"Serial code path ... \n";
   int numElements = static_cast< int >( std::distance( mapfirst, maplast ) );
   typedef typename  std::iterator_traits<InputIterator1>::value_type iType1;
   iType1 temp;
   for(int iter = 0; iter < numElements; iter++)
   {
                   temp = *(mapfirst + (int)iter);
                  *(result + (int)iter) = *(input + (int)temp);
   }
}


template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >

void serial_gather_if(InputIterator1 mapfirst,
                      InputIterator1 maplast,
                      InputIterator2 stencil,
                      InputIterator3 input,
                      OutputIterator result)
{
    //std::cout<<"Serial code path ... \n";
   int numElements = static_cast< int >( std::distance( mapfirst, maplast ) );
   for(size_t iter = 0; iter < numElements; iter++)
   {
       if(stencil[(int)iter]== 1)
            result[(int)iter] = *(input + mapfirst[(int)iter]);
   }
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >

void serial_gather_if(InputIterator1 mapfirst,
                      InputIterator1 maplast,
                      InputIterator2 stencil,
                      InputIterator3 input,
                      OutputIterator result,
                      Predicate pred)
{
   //std::cout<<"Serial code path ... \n";
   int numElements = static_cast< int >( std::distance( mapfirst, maplast ) );
   for(int iter = 0; iter < numElements; iter++)
   {
        if(pred(*(stencil + (int)iter)))
             result[(int)iter] = input[mapfirst[(int)iter]];
   }
}

 /* End-- Serial Implementation of the gather and gather_if routines */

////////////////////////////////////////////////////////////////////
// GatherIf enqueue
////////////////////////////////////////////////////////////////////

    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVInputIterator3,
              typename DVOutputIterator,
              typename Predicate >
    void gather_if_enqueue( bolt::amp::control &ctl,
                            const DVInputIterator1& map_first,
                            const DVInputIterator1& map_last,
                            const DVInputIterator2& stencil,
                            const DVInputIterator3& input,
                            const DVOutputIterator& result,
                            const Predicate& pred )
    {
		concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();
        typedef typename std::iterator_traits<DVInputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<DVInputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<DVInputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;

       int szElements = static_cast< int >(std::distance( map_first, map_last));
		const int leng =  szElements + GATHER_WAVEFRNT_SIZE - (szElements % GATHER_WAVEFRNT_SIZE);
		concurrency::extent< 1 > inputExtent(leng);
		try
                {
                    concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
                    {
                        int globalId = idx[ 0 ];
                        if( globalId >= szElements)
                        return;

						iType1 m = map_first[ globalId ];
						iType2 s = stencil[ globalId ];
						if ( pred( s ) )
						{
							result [ globalId ] = input [ m ] ;
						}
						});	
                }
			    catch(std::exception &e)
                {
                      std::cout << "Exception while calling bolt::amp::gather parallel_for_each"<<e.what()<<std::endl;
					  return;
                }	
	};

////////////////////////////////////////////////////////////////////
// Gather enqueue
////////////////////////////////////////////////////////////////////
    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVOutputIterator >
    void gather_enqueue( bolt::amp::control &ctl,
                         const DVInputIterator1& map_first,
                         const DVInputIterator1& map_last,
                         const DVInputIterator2& input,
                         const DVOutputIterator& result )
    {
		concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();
        typedef typename std::iterator_traits<DVInputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<DVInputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
        int szElements = static_cast< int >(std::distance( map_first, map_last));
		const unsigned int leng =  szElements + GATHER_WAVEFRNT_SIZE - (szElements % GATHER_WAVEFRNT_SIZE);
		concurrency::extent< 1 > inputExtent(leng);
		try
                {
                    concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
                    {
                        int globalId = idx[ 0 ];

                        if( globalId >= szElements)
                        return;
						iType1 m = map_first[ globalId ];
						result [ globalId ] = input [ m ] ;
						});	
                }
			    catch(std::exception &e)
                {
                      std::cout << "Exception while calling bolt::amp::gather parallel_for_each"<<e.what()<<std::endl;
					  return;
                }
    };

////////////////////////////////////////////////////////////////////
// Enqueue ends
////////////////////////////////////////////////////////////////////
// GatherIf pick iterator

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
    void gather_if_pick_iterator( bolt::amp::control &ctl,
                                  const InputIterator1& map_first,
                                  const InputIterator1& map_last,
                                  const InputIterator2& stencil,
                                  const InputIterator3& input,
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
        int sz = static_cast< int >(std::distance( map_first, map_last));
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
            serial_gather_if(map_first, map_last, stencil, input, result, pred );
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
           bolt::btbb::gather_if(map_first, map_last, stencil, input, result, pred);
#else
          throw std::runtime_error( "The MultiCoreCpu version of gather_if is not enabled to be built! \n" );
#endif
        }
        else
		{						
          // Map the input iterator to a device_vector
		  device_vector< iType1, concurrency::array_view> dvMap( map_first, map_last, false, ctl );
		  device_vector< iType2, concurrency::array_view> dvStencil( stencil, sz, false, ctl );
		  device_vector< iType3, concurrency::array_view> dvInput( input, sz, false, ctl );

          // Map the output iterator to a device_vector
		  device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );
          gather_if_enqueue( ctl,
                              dvMap.begin( ),
                              dvMap.end( ),
                              dvStencil.begin( ),
                              dvInput.begin( ),
                              dvResult.begin( ),
                              pred);
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
    void gather_if_pick_iterator( bolt::amp::control &ctl,
                                  const InputIterator1& map_first,
                                  const InputIterator1& map_last,
                                  const InputIterator2& stencilFancyIter,
                                  const InputIterator3& input,
                                  const OutputIterator& result,
                                  const Predicate& pred,
                                  std::random_access_iterator_tag,
                                  bolt::amp::fancy_iterator_tag,
                                  std::random_access_iterator_tag
                                   )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast< int >(std::distance( map_first, map_last));
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
            serial_gather_if(map_first, map_last, stencilFancyIter, input, result, pred);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            bolt::btbb::gather_if(map_first, map_last, stencilFancyIter, input, result, pred);
#else
            throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType1, concurrency::array_view> dvMap( map_first, map_last, false, ctl );
		    device_vector< iType3, concurrency::array_view> dvInput( input, sz, false, ctl );
            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            gather_if_enqueue( ctl,
                               dvMap.begin( ),
                               dvMap.end( ),
                               stencilFancyIter,
                               dvInput.begin(),
                               dvResult.begin( ),
                               pred);
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
    void gather_if_pick_iterator( bolt::amp::control &ctl,
                                  const InputIterator1& fancymapFirst,
                                  const InputIterator1& fancymapLast,
                                  const InputIterator2& stencil,
                                  const InputIterator3& input,
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
        int sz = static_cast<int> (std::distance( fancymapFirst, fancymapLast ));
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
           serial_gather_if (fancymapFirst, fancymapLast, stencil, input, result, pred );
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            bolt::btbb::gather_if( fancymapFirst, fancymapLast, stencil, input, result, pred );
#else
            throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType2, concurrency::array_view> dvStencil( stencil, sz, false, ctl );
		    device_vector< iType3, concurrency::array_view> dvInput( input, sz, false, ctl );
            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );
            gather_if_enqueue( ctl,
                               fancymapFirst,
                               fancymapLast,
                               dvStencil.begin( ),
                               dvInput.begin( ),
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
    void gather_if_pick_iterator( bolt::amp::control &ctl,
                                  const DVInputIterator1& map_first,
                                  const DVInputIterator1& map_last,
                                  const DVInputIterator2& stencil,
                                  const DVInputIterator3& input,
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

        int sz = static_cast<int >(std::distance( map_first, map_last ));
        if( sz == 0 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
             runMode = ctl.getDefaultPathToRun();
        }
		
        if( runMode == bolt::amp::control::SerialCpu )
        {		   
            typename bolt::amp::device_vector< iType1 >::pointer mapPtr =  map_first.getContainer( ).data( );
            typename bolt::amp::device_vector< iType2 >::pointer stenPtr =  stencil.getContainer( ).data( );
            typename bolt::amp::device_vector< iType3 >::pointer inputPtr =  input.getContainer( ).data( );
            typename bolt::amp::device_vector< oType >::pointer resPtr =  result.getContainer( ).data( );

#if defined( _WIN32 )
            serial_gather_if(&mapPtr[ map_first.m_Index ], &mapPtr[ map_last.m_Index ], &stenPtr[ stencil.m_Index ],
                 &inputPtr[ input.m_Index ], stdext::make_checked_array_iterator( &resPtr[ result.m_Index ], sz ), pred );

#else
             serial_gather_if( &mapPtr[ map_first.m_Index ], &mapPtr[ map_last.m_Index ], &stenPtr[ stencil.m_Index ],
                 &inputPtr[ input.m_Index ], &resPtr[ result.m_Index ], pred );
#endif
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
          {
           typename bolt::amp::device_vector< iType1 >::pointer mapPtr =  map_first.getContainer( ).data( );
           typename bolt::amp::device_vector< iType2 >::pointer stenPtr =  stencil.getContainer( ).data( );
           typename bolt::amp::device_vector< iType3 >::pointer inputPtr =  input.getContainer( ).data( );
           typename bolt::amp::device_vector< oType >::pointer resPtr =  result.getContainer( ).data( );

           bolt::btbb::gather_if( &mapPtr[ map_first.m_Index ], &mapPtr[ map_last.m_Index ], &stenPtr[ stencil.m_Index ],
                 &inputPtr[ input.m_Index ], &resPtr[ result.m_Index ], pred );
          }
#else
             throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            gather_if_enqueue( ctl, map_first, map_last, stencil, input, result, pred );
        }
    }

////////////////////////////////////////////////////////////////////
// Gather pick iterator
////////////////////////////////////////////////////////////////////

// Host vectors

    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
    void gather_pick_iterator( bolt::amp::control &ctl,
                               const InputIterator1& map_first,
                               const InputIterator1& map_last,
                               const InputIterator2& input,
                               const OutputIterator& result,
                               std::random_access_iterator_tag,
                               std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        int sz = static_cast<int >(std::distance( map_first, map_last ));

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
            serial_gather(map_first, map_last, input, result);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
           bolt::btbb::gather(map_first, map_last, input, result);
#else
          throw std::runtime_error( "The MultiCoreCpu version of gather_if is not enabled to be built! \n" );
#endif
        }
        else
        {		  
          // Map the input iterator to a device_vector		  
		  device_vector< iType1, concurrency::array_view> dvMap( map_first, map_last, false, ctl );
		  device_vector< iType2, concurrency::array_view> dvInput( input, sz, false, ctl );
          // Map the output iterator to a device_vector
		  device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );
          gather_enqueue( ctl,
                          dvMap.begin( ),
                          dvMap.end( ),
                          dvInput.begin( ),
                          dvResult.begin( ) );
          // This should immediately map/unmap the buffer
          dvResult.data( );
        }
    }

// Map is a fancy iterator
    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
    void gather_pick_iterator( bolt::amp::control &ctl,
                               const InputIterator1& firstFancy,
                               const InputIterator1& lastFancy,
                               const InputIterator2& input,
                               const OutputIterator& result,
                               bolt::amp::fancy_iterator_tag,
                               std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast<int>(std::distance( firstFancy, lastFancy ));
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
            serial_gather( firstFancy, lastFancy, input, result);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            bolt::btbb::gather( firstFancy, lastFancy, input, result);
#else
            throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType2, concurrency::array_view> dvInput( input, sz, false, ctl );
            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );
            gather_enqueue( ctl,
                            firstFancy,
                            lastFancy,
                            dvInput.begin(),
                            dvResult.begin( ));

            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }


// Input is a fancy iterator
    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator>
    void gather_pick_iterator( bolt::amp::control &ctl,
                               const InputIterator1& map_first,
                               const InputIterator1& map_last,
                               const InputIterator2& inputFancy,
                               const OutputIterator& result,
                               std::random_access_iterator_tag,
                               bolt::amp::fancy_iterator_tag )
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast<int>(std::distance( map_first, map_last ));
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
            serial_gather(map_first, map_last, inputFancy, result);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
             bolt::btbb::gather(map_first, map_last, inputFancy, result);
#else
            throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType1, concurrency::array_view> dvMap( map_first, map_last, false, ctl );
            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );
            gather_enqueue( ctl,
                            dvMap.begin( ),
                            dvMap.end( ),
                            inputFancy,
                            dvResult.begin( ));
            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }

// Device Vectors
    template< typename DVMapIterator,
              typename DVInputIterator,
              typename DVOutputIterator >
    void gather_pick_iterator( bolt::amp::control &ctl,
                               const DVMapIterator& map_first,
                               const DVMapIterator& map_last,
                               const DVInputIterator& input,
                               const DVOutputIterator& result,
                               bolt::amp::device_vector_tag,
                               bolt::amp::device_vector_tag )
    {

        typedef typename std::iterator_traits< DVMapIterator >::value_type iType1;
        typedef typename std::iterator_traits< DVInputIterator >::value_type iType2;
        typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        int sz = static_cast<int>(std::distance( map_first, map_last ));
        if( sz == 0 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
             runMode = ctl.getDefaultPathToRun();
        }
        if( runMode == bolt::amp::control::SerialCpu )
        {
           typename bolt::amp::device_vector< iType1 >::pointer  MapBuffer  =  map_first.getContainer( ).data( );
           typename bolt::amp::device_vector< iType2 >::pointer InputBuffer    =  input.getContainer( ).data( );
           typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  result.getContainer( ).data( );
           serial_gather(&MapBuffer[ map_first.m_Index ], &MapBuffer[ map_last.m_Index ], &InputBuffer[ input.m_Index ],
           &ResultBuffer[ result.m_Index ]);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
         #if defined( ENABLE_TBB )
            {
               typename bolt::amp::device_vector< iType1 >::pointer  MapBuffer  =  map_first.getContainer( ).data( );
               typename bolt::amp::device_vector< iType2 >::pointer InputBuffer    =  input.getContainer( ).data( );
               typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  result.getContainer( ).data( );
               bolt::btbb::gather(&MapBuffer[ map_first.m_Index ], &MapBuffer[ map_last.m_Index ], &InputBuffer[ input.m_Index ],
               &ResultBuffer[ result.m_Index ]);
            }
#else
             throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            gather_enqueue( ctl,
                            map_first,
                            map_last,
                            input,
                            result );
        }
    }

// Map fancy ; Input DV
    template< typename FancyIterator,
              typename DVInputIterator,
              typename DVOutputIterator >
    void gather_pick_iterator( bolt::amp::control &ctl,
                                const FancyIterator& firstFancy,
                                const FancyIterator& lastFancy,
                                const DVInputIterator& input,
                                const DVOutputIterator& result,
                                bolt::amp::fancy_iterator_tag,
                                bolt::amp::device_vector_tag )
    {

        typedef typename std::iterator_traits< FancyIterator >::value_type iType1;
        typedef typename std::iterator_traits< DVInputIterator >::value_type iType2;
        typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        int sz = static_cast<int>(std::distance( firstFancy, lastFancy ));
        if( sz == 0 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
             runMode = ctl.getDefaultPathToRun();
        }		
        if( runMode == bolt::amp::control::SerialCpu )
        {
           typename bolt::amp::device_vector< iType2 >::pointer InputBuffer    =  input.getContainer( ).data( );
           typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  result.getContainer( ).data( );
            serial_gather(firstFancy, lastFancy, &InputBuffer[ input.m_Index ], &ResultBuffer[ result.m_Index ]);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            {
               typename bolt::amp::device_vector< iType2 >::pointer InputBuffer    =  input.getContainer( ).data( );
               typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  result.getContainer( ).data( );
                bolt::btbb::gather(firstFancy, lastFancy, &InputBuffer[ input.m_Index ], &ResultBuffer[ result.m_Index ]);
            }
#else
             throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            gather_enqueue( ctl,
                            firstFancy,
                            lastFancy,
                            input,
                            result);
        }
    }

// Input fancy ; Map DV
    template< typename DVMapIterator,
              typename FancyInput,
              typename DVOutputIterator >
    void gather_pick_iterator( bolt::amp::control &ctl,
                                const DVMapIterator& mapfirst,
                                const DVMapIterator& maplast,
                                const FancyInput& fancyInpt,
                                const DVOutputIterator& result,
                                bolt::amp::device_vector_tag,
                                bolt::amp::fancy_iterator_tag )
    {

        typedef typename std::iterator_traits< DVMapIterator >::value_type iType1;
        typedef typename std::iterator_traits< FancyInput >::value_type iType2;
        typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

        int sz = static_cast<int>(std::distance( mapfirst, maplast ));
        if( sz == 0 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::amp::control::Automatic)
        {
             runMode = ctl.getDefaultPathToRun();
        }		
        if( runMode == bolt::amp::control::SerialCpu )
        {		   
           typename bolt::amp::device_vector< iType1 >::pointer mapBuffer    =  mapfirst.getContainer( ).data( );
           typename bolt::amp::device_vector< oType >::pointer  ResultBuffer =  result.getContainer( ).data( );
            serial_gather( &mapBuffer[ mapfirst.m_Index ], &mapBuffer[ maplast.m_Index ], fancyInpt,
                                                                         &ResultBuffer[ result.m_Index ]);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            {
               typename bolt::amp::device_vector< iType1 >::pointer mapBuffer    =  mapfirst.getContainer( ).data( );
               typename  bolt::amp::device_vector< oType >::pointer  ResultBuffer =  result.getContainer( ).data( );
               bolt::btbb::gather(  &mapBuffer[ mapfirst.m_Index ], &mapBuffer[ maplast.m_Index ], fancyInpt,
                                                                         &ResultBuffer[ result.m_Index ]);
            }

#else
             throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            gather_enqueue( ctl,
                            mapfirst,
                            maplast,
                            fancyInpt,
                            result );
        }
    }

// Map DV ; Input random access
    template< typename DVInputIterator,
              typename InputIterator,
              typename OutputIterator >
    void gather_pick_iterator( bolt::amp::control &ctl,
                               const DVInputIterator& map_first,
                               const DVInputIterator& map_last,
                               const InputIterator& input,
                               const OutputIterator& result,
                               bolt::amp::device_vector_tag,
                               std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits<DVInputIterator>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast<int>(std::distance( map_first, map_last ));
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
            serial_gather(map_first, map_last, input, result );
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
             {
               typename bolt::amp::device_vector< iType1 >::pointer mapBuffer    =  map_first.getContainer( ).data( );
                bolt::btbb::gather( &mapBuffer[ map_first.m_Index ], &mapBuffer[ map_last.m_Index ], input, result);
            }
#else
            throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType2, concurrency::array_view> dvInput( input, sz, false, ctl );

            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            gather_enqueue( ctl,
                            map_first,
                            map_last,
                            dvInput.begin(),
                            dvResult.begin( ) );

            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }

// RA Map ; DV Input
    template< typename InputIterator,
              typename DVInputIterator,
              typename OutputIterator>
    void gather_pick_iterator( bolt::amp::control &ctl,
                                const InputIterator& map_first,
                                const InputIterator& map_last,
                                const DVInputIterator& input,
                                const OutputIterator& result,
                                std::random_access_iterator_tag,
                                bolt::amp::device_vector_tag )
    {
        typedef typename std::iterator_traits<InputIterator>::value_type iType1;
        typedef typename std::iterator_traits<DVInputIterator>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        int sz = static_cast<int>(std::distance( map_first, map_last ));
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
          typename bolt::amp::device_vector< iType2 >::pointer inputBuffer    =  input.getContainer( ).data( );
           serial_gather(map_first, map_last, &inputBuffer[ input.m_Index ], result);
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
           bolt::btbb::gather(map_first, map_last , input, result);
            {
               typename bolt::amp::device_vector< iType2 >::pointer inputBuffer    =  input.getContainer( ).data( );
                bolt::btbb::gather(map_first, map_last, &inputBuffer[ input.m_Index ], result);
            }
#else
            throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
            // Use host pointers memory since these arrays are only read once - no benefit to copying.
            // Map the input iterator to a device_vector
		    device_vector< iType1, concurrency::array_view> dvMap( map_first, map_last, false, ctl );
            // Map the output iterator to a device_vector
		    device_vector< oType, concurrency::array_view> dvResult( result, sz, false, ctl );

            gather_enqueue( ctl,
                            dvMap.begin(),
                            dvMap.end(),
                            input,
                            dvResult.begin( ));
            // This should immediately map/unmap the buffer
            dvResult.data( );
        }
    }



////////////////////////////////////////////////////////////////////
// GatherIf detect random access
////////////////////////////////////////////////////////////////////



    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
    void gather_if_detect_random_access( bolt::amp::control& ctl,
                                         const InputIterator1& map_first,
                                         const InputIterator1& map_last,
                                         const InputIterator2& stencil,
                                         const InputIterator3& input,
                                         const OutputIterator& result,
                                         const Predicate& pred,
                                         std::random_access_iterator_tag,
                                         std::random_access_iterator_tag,
                                         std::random_access_iterator_tag )
    {
       gather_if_pick_iterator( ctl,
                                map_first,
                                map_last,
                                stencil,
                                input,
                                result,
                                pred,
                                typename std::iterator_traits< InputIterator1 >::iterator_category( ),
                                typename std::iterator_traits< InputIterator2 >::iterator_category( ),
                                typename std::iterator_traits< InputIterator3 >::iterator_category( ) );
    };


    // Wrapper that uses default ::bolt::amp::control class, iterator interface
    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
    void gather_if_detect_random_access( bolt::amp::control& ctl,
                                         const InputIterator1& map_first,
                                         const InputIterator1& map_last,
                                         const InputIterator2& stencil,
                                         const InputIterator3& input,
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
    void gather_detect_random_access( bolt::amp::control& ctl,
                                      const InputIterator1& map_first,
                                      const InputIterator1& map_last,
                                      const InputIterator2& input,
                                      const OutputIterator& result,
                                      std::input_iterator_tag,
                                      std::input_iterator_tag )
    {
            static_assert( std::is_same< InputIterator1, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
    };


////////////////////////////////////////////////////////////////////
// Gather detect random access
////////////////////////////////////////////////////////////////////



    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
    void gather_detect_random_access( bolt::amp::control& ctl,
                                      const InputIterator1& map_first,
                                      const InputIterator1& map_last,
                                      const InputIterator2& input,
                                      const OutputIterator& result,
                                      std::random_access_iterator_tag,
                                      std::random_access_iterator_tag )
    {
       gather_pick_iterator( ctl,
                             map_first,
                             map_last,
                             input,
                             result,
                             typename std::iterator_traits< InputIterator1 >::iterator_category( ),
                             typename std::iterator_traits< InputIterator2 >::iterator_category( ) );
    };


} //End of detail namespace

////////////////////////////////////////////////////////////////////
// Gather APIs
////////////////////////////////////////////////////////////////////
template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
void gather( bolt::amp::control& ctl,
             InputIterator1 map_first,
             InputIterator1 map_last,
             InputIterator2 input,
             OutputIterator result )
{
    detail::gather_detect_random_access( ctl,
                                         map_first,
                                         map_last,
                                         input,
                                         result,
                                         typename std::iterator_traits< InputIterator1 >::iterator_category( ),
                                         typename std::iterator_traits< InputIterator2 >::iterator_category( ) );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
void gather( InputIterator1 map_first,
             InputIterator1 map_last,
             InputIterator2 input,
             OutputIterator result)
{
    detail::gather_detect_random_access( control::getDefault( ),
                                         map_first,
                                         map_last,
                                         input,
                                         result,
                                         typename std::iterator_traits< InputIterator1 >::iterator_category( ),
                                         typename std::iterator_traits< InputIterator2 >::iterator_category( ) );
}


////////////////////////////////////////////////////////////////////
// GatherIf APIs
////////////////////////////////////////////////////////////////////
template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >
void gather_if( bolt::amp::control& ctl,
                InputIterator1 map_first,
                InputIterator1 map_last,
                InputIterator2 stencil,
                InputIterator3 input,
                OutputIterator result)
{
    typedef typename std::iterator_traits<InputIterator2>::value_type stencilType;
    gather_if( ctl,
               map_first,
               map_last,
               stencil,
               input,
               result,
               bolt::amp::identity <stencilType> ( ) );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >
void gather_if( InputIterator1 map_first,
                InputIterator1 map_last,
                InputIterator2 stencil,
                InputIterator3 input,
                OutputIterator result )
{
    typedef typename std::iterator_traits<InputIterator2>::value_type stencilType;
    gather_if( map_first,
               map_last,
               stencil,
               input,
               result,
               bolt::amp::identity <stencilType> ( ) );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
void gather_if( bolt::amp::control& ctl,
                InputIterator1 map_first,
                InputIterator1 map_last,
                InputIterator2 stencil,
                InputIterator3 input,
                OutputIterator result,
                Predicate pred )
{
    detail::gather_if_detect_random_access( ctl,
                                            map_first,
                                            map_last,
                                            stencil,
                                            input,
                                            result,
                                            pred,
                                            typename std::iterator_traits< InputIterator1 >::iterator_category( ),
                                            typename std::iterator_traits< InputIterator2 >::iterator_category( ),
                                            typename std::iterator_traits< InputIterator3 >::iterator_category( ) );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
void gather_if(  InputIterator1 map_first,
                 InputIterator1 map_last,
                 InputIterator2 stencil,
                 InputIterator3 input,
                 OutputIterator result,
                 Predicate pred)
{
    detail::gather_if_detect_random_access( control::getDefault( ),
                                            map_first,
                                            map_last,
                                            stencil,
                                            input,
                                            result,
                                            pred,
                                            typename std::iterator_traits< InputIterator1 >::iterator_category( ),
                                            typename std::iterator_traits< InputIterator2 >::iterator_category( ),
                                            typename std::iterator_traits< InputIterator3 >::iterator_category( ));
}


} //End of cl namespace
} //End of bolt namespace

#endif
