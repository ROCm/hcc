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

#if !defined( BOLT_AMP_COUNT_INL )
#define BOLT_AMP_COUNT_INL
#pragma once

#include <algorithm>
#include <amp.h>
#include "bolt/amp/bolt.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/iterator/iterator_traits.h"
#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/count.h"
#endif

#define _COUNT_REDUCE_STEP(_LENGTH, _IDX, _W) \
    if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
      scratch_count[_IDX] =  scratch_count[_IDX] + scratch_count[_IDX + _W];\
    }\
    t_idx.barrier.wait();

#define COUNT_WAVEFRONT_SIZE 256

namespace bolt {
    namespace amp {
        namespace detail {

            //----
            // This is the base implementation of reduction that is called by all of the convenience wrappers below.
            // first and last must be iterators from a DeviceVector
            template<typename DVInputIterator, typename Predicate>
            unsigned int count_enqueue(bolt::amp::control &ctl,
                const DVInputIterator& first,
                const DVInputIterator& last,
                const Predicate& predicate)
            {
				typedef typename std::iterator_traits< DVInputIterator >::value_type iType;				
				const int szElements = static_cast< int >(std::distance(first, last));

				int max_ComputeUnits = 32;
				int numTiles = max_ComputeUnits*32;	/* Max no. of WG for Tahiti(32 compute Units) and 32 is the tuning factor that gives good performance*/
				int length = (COUNT_WAVEFRONT_SIZE * numTiles);
				length = szElements < length ? szElements : length;
				unsigned int residual = length % COUNT_WAVEFRONT_SIZE;
				length = residual ? (length + COUNT_WAVEFRONT_SIZE - residual): length ;
				numTiles = static_cast< int >((szElements/COUNT_WAVEFRONT_SIZE)>= numTiles?(numTiles):
									(std::ceil( static_cast< float >( szElements ) / COUNT_WAVEFRONT_SIZE) ));
				
				concurrency::array<unsigned int, 1> result(numTiles);
				concurrency::extent< 1 > inputExtent(length);
				concurrency::tiled_extent< COUNT_WAVEFRONT_SIZE > tiledExtentReduce = inputExtent.tile< COUNT_WAVEFRONT_SIZE >();

                try
                {
					concurrency::parallel_for_each(ctl.getAccelerator().get_default_view(),
                                                   tiledExtentReduce,
                                                   [ first,
                                                     szElements,
													 length,
                                                     &result,
                                                     predicate ]
					(concurrency::tiled_index<COUNT_WAVEFRONT_SIZE> t_idx) restrict(amp)
                    {
						int gx = t_idx.global[0];
						int gloId = gx;
						unsigned int tileIndex = t_idx.local[0];
                      //  Initialize local data store
                      bool stat;
					  unsigned int count = 0;

					  tile_static unsigned int scratch_count[COUNT_WAVEFRONT_SIZE];

                      //  Abort threads that are passed the end of the input vector
					  if (gloId < szElements)
                      {
                       //  Initialize the accumulator private variable with data from the input array
                       //  This essentially unrolls the loop below at least once
						  iType accumulator = first[gloId];
                       stat =  predicate(accumulator);
					   scratch_count[tileIndex]  = stat ? ++count : count;
					   gx += length;
                      }
                      t_idx.barrier.wait();


					  // Loop sequentially over chunks of input vector, reducing an arbitrary size input
					  // length into a length related to the number of workgroups
					  while (gx < szElements)
					  {
						  iType element = first[gx];
						  stat = predicate(element);
						  scratch_count[tileIndex] = stat ? ++scratch_count[tileIndex] : scratch_count[tileIndex];
						  gx += length;
					  }

					  t_idx.barrier.wait();

                      //  Tail stops the last workgroup from reading past the end of the input vector
					  unsigned int tail = szElements - (t_idx.tile[0] * t_idx.tile_dim0);
                      // Parallel reduction within a given workgroup using local data store
                      // to share values between workitems

					  _COUNT_REDUCE_STEP(tail, tileIndex, 128);
					  _COUNT_REDUCE_STEP(tail, tileIndex, 64);
                      _COUNT_REDUCE_STEP(tail, tileIndex, 32);
                      _COUNT_REDUCE_STEP(tail, tileIndex, 16);
                      _COUNT_REDUCE_STEP(tail, tileIndex,  8);
                      _COUNT_REDUCE_STEP(tail, tileIndex,  4);
                      _COUNT_REDUCE_STEP(tail, tileIndex,  2);
                      _COUNT_REDUCE_STEP(tail, tileIndex,  1);


					  //  Abort threads that are passed the end of the input vector
					  if (gloId >= szElements)
						  return;

                      //  Write only the single reduced value for the entire workgroup
                      if (tileIndex == 0)
                      {
                          result[t_idx.tile[ 0 ]] = scratch_count[0];
                      }


                    });

					std::vector<unsigned int> *cpuPointerReduce = new std::vector<unsigned int>(numTiles);
					concurrency::copy(result, (*cpuPointerReduce).begin());                  
					unsigned int count = (*cpuPointerReduce)[0];
					for (int i = 1; i < numTiles; ++i)
					{
                       count +=  (*cpuPointerReduce)[i];
                    }
					delete cpuPointerReduce;

                    return count;
                }
                catch(std::exception &e)
                {

                      std::cout << "Exception while calling bolt::amp::count parallel_for_each"<<e.what()<<std::endl;

                      return 0;
                }
            }
#ifdef _WIN32
            template<typename InputIterator, typename Predicate>
            int count_detect_random_access(bolt::amp::control &ctl,
                const InputIterator& first,
                const InputIterator& last,
                const Predicate& predicate,
                std::input_iterator_tag)
            {

                //  TODO:  It should be possible to support non-random_access_iterator_tag iterators,
                // if we copied the data
                //  to a temporary buffer.  Should we?
                static_assert( false, "Bolt only supports random access iterator types" );
            }
#endif
// Prior to use
            template<typename InputIterator, typename Predicate>
            typename bolt::amp::iterator_traits<InputIterator>::difference_type
            count_pick_iterator(bolt::amp::control &ctl,
                const InputIterator& first,
                const InputIterator& last,
                const Predicate& predicate,
                std::random_access_iterator_tag );
            template<typename DVInputIterator, typename Predicate>
            typename bolt::amp::iterator_traits<DVInputIterator>::difference_type
            count_pick_iterator( bolt::amp::control &ctl,
                                 const DVInputIterator& first,
                                 const DVInputIterator& last,
                                 const Predicate& predicate,
                                 bolt::amp::device_vector_tag );         
             template<typename DVInputIterator, typename Predicate>
            typename bolt::amp::iterator_traits<DVInputIterator>::difference_type
            count_pick_iterator( bolt::amp::control &ctl,
                                 const DVInputIterator& first,
                                 const DVInputIterator& last,
                                 const Predicate& predicate,
                                 bolt::amp::fancy_iterator_tag );
                                 
            template<typename InputIterator, typename Predicate>
            typename bolt::amp::iterator_traits<InputIterator>::difference_type
                count_detect_random_access(bolt::amp::control &ctl,
                const InputIterator& first,
                const InputIterator& last,
                const Predicate& predicate,
                std::random_access_iterator_tag)
            {
                return count_pick_iterator( ctl, first, last, predicate,
                                            typename std::iterator_traits< InputIterator >::iterator_category( ) );
            }

            // This template is called after we detect random access iterators
            // This is called strictly for any non-device_vector iterator
            template<typename InputIterator, typename Predicate>
            typename bolt::amp::iterator_traits<InputIterator>::difference_type
            count_pick_iterator(bolt::amp::control &ctl,
                const InputIterator& first,
                const InputIterator& last,
                const Predicate& predicate,
                std::random_access_iterator_tag )

            {
                /*************/
                typedef typename std::iterator_traits<InputIterator>::value_type iType;
                int szElements = static_cast< int >(last - first);
                if (szElements == 0)
                    return 0;
                /*TODO - probably the forceRunMode should be replaced by getRunMode and setRunMode*/
                // Its a dynamic choice. See the reduce Test Code
                // What should we do if the run mode is automatic. Currently it goes to the last else statement
                //How many threads we should spawn?
                //Need to look at how to control the number of threads spawned.

                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.

                if (runMode == bolt::amp::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
                if (runMode == bolt::amp::control::SerialCpu)
                {
                      return (int) std::count_if(first,last,predicate);
                }
                else if (runMode == bolt::amp::control::MultiCoreCpu)
                {
#ifdef ENABLE_TBB
                    return bolt::btbb::count_if(first,last,predicate);
#else

                    throw Concurrency::runtime_exception( "The MultiCoreCpu version of count function is not enabled to be built.", 0);
                    return 0;
#endif
                }
                else

                {
                    device_vector< iType, concurrency::array_view > dvInput( first, last, false, ctl );
                    return count_enqueue( ctl, dvInput.begin(), dvInput.end(), predicate );
                }
            };

            // This template is called after we detect random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVInputIterator, typename Predicate>
            typename bolt::amp::iterator_traits<DVInputIterator>::difference_type
            count_pick_iterator( bolt::amp::control &ctl,
                                 const DVInputIterator& first,
                                 const DVInputIterator& last,
                                 const Predicate& predicate,
                                 bolt::amp::device_vector_tag )
            {
                typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
                int szElements = static_cast< int > (last - first);
                if (szElements == 0)
                    return 0;

                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.

                if (runMode == bolt::amp::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
                if (runMode == bolt::amp::control::SerialCpu)
                {
                  typename bolt::amp::device_vector< iType >::pointer countInputBuffer = first.getContainer( ).data( );
                   return  (int) std::count_if(&countInputBuffer[first.m_Index],
                       &countInputBuffer[first.m_Index + szElements], predicate) ;

                }

                else if (runMode == bolt::amp::control::MultiCoreCpu)
                {
#ifdef ENABLE_TBB
                   
                   typename bolt::amp::device_vector< iType >::pointer countInputBuffer =  first.getContainer( ).data( );
                     return  bolt::btbb::count_if(&countInputBuffer[first.m_Index],
                         &countInputBuffer[first.m_Index + szElements] ,predicate);
#else              
                   throw Concurrency::runtime_exception( "The MultiCoreCpu version of count function is not enabled to be built.", 0);
                   
                   return 0;
#endif

                }
                else
                {
                  return  count_enqueue( ctl, first, last, predicate );
                }
            }

            // This template is called after we detect random access iterators
            // This is called strictly for iterators that are derived from fancy_iterator
            template<typename DVInputIterator, typename Predicate>
            typename bolt::amp::iterator_traits<DVInputIterator>::difference_type
            count_pick_iterator( bolt::amp::control &ctl,
                                 const DVInputIterator& first,
                                 const DVInputIterator& last,
                                 const Predicate& predicate,
                                 bolt::amp::fancy_iterator_tag )
            {
                typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
                int szElements = static_cast< int > (last - first);
                if (szElements == 0)
                    return 0;

                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.

                if (runMode == bolt::amp::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
                if (runMode == bolt::amp::control::SerialCpu)
                {
                  return  (int) std::count_if(first, last, predicate) ;

                }

                else if (runMode == bolt::amp::control::MultiCoreCpu)
                {
#ifdef ENABLE_TBB
                   
                   return  bolt::btbb::count_if(first, last,predicate);
#else              
                   throw Concurrency::runtime_exception( "The MultiCoreCpu version of count function is not enabled to be built.", 0);
                   return 0;
#endif

                }
                else
                {
                  return  count_enqueue( ctl, first, last, predicate );
                }
            }


        } //end of detail

        template<typename InputIterator, typename Predicate>
        typename bolt::amp::iterator_traits<InputIterator>::difference_type
            count_if(control& ctl, InputIterator first,
            InputIterator last,
            Predicate predicate)
        {
              return detail::count_detect_random_access(ctl, first, last, predicate,
                typename std::iterator_traits< InputIterator >::iterator_category( ) );

        }

       template<typename InputIterator, typename Predicate>
        typename bolt::amp::iterator_traits<InputIterator>::difference_type
            count_if( InputIterator first,
            InputIterator last,
            Predicate predicate)
        {

         return count_if(bolt::amp::control::getDefault(), first, last, predicate);

        }

    } //end of amp
}//end of bolt

#endif //COUNT_INL
