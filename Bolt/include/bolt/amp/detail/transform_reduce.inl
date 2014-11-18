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
#if !defined( BOLT_AMP_TRANSFORM_REDUCE_INL )
#define BOLT_AMP_TRANSFORM_REDUCE_INL

#include <cmath>
#include <string>
#include <iostream>
#include <numeric>
#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/transform_reduce.h"
#endif

#include "bolt/amp/iterator/iterator_traits.h"

#define _T_REDUCE_WAVEFRONT_SIZE 256 

#define _T_REDUCE_STEP(_LENGTH, _IDX, _W) \
    if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
      oType mine = scratch[_IDX];\
      oType other = scratch[_IDX + _W];\
      scratch[_IDX] = binary_op(mine, other); \
    }\
    t_idx.barrier.wait();

namespace bolt {

  namespace amp {

    namespace  detail {

		 template<typename DVInputIterator, typename UnaryFunction, typename oType, typename BinaryFunction>
        oType transform_reduce_enqueue( control& ctl,
                                        const DVInputIterator& first,
                                        const DVInputIterator& last,
                                        const UnaryFunction& transform_op,
                                        const oType& init,
                                        const BinaryFunction& binary_op )
        {
                typedef typename std::iterator_traits< DVInputIterator >::value_type iType;
                const int szElements = static_cast< int >( std::distance( first, last ) );




				int max_ComputeUnits = 32;
				int numTiles = max_ComputeUnits*32;	/* Max no. of WG for Tahiti(32 compute Units) and 32 is the tuning factor that gives good performance*/
				int length = (_T_REDUCE_WAVEFRONT_SIZE * numTiles);
				length = szElements < length ? szElements : length;
				unsigned int residual = length % _T_REDUCE_WAVEFRONT_SIZE;
				length = residual ? (length + _T_REDUCE_WAVEFRONT_SIZE - residual): length ;
				numTiles = static_cast< int >((szElements/_T_REDUCE_WAVEFRONT_SIZE)>= numTiles?(numTiles):
									(std::ceil( static_cast< float >( szElements ) / _T_REDUCE_WAVEFRONT_SIZE) ));
				
				concurrency::array<oType, 1> result(numTiles);
				concurrency::extent< 1 > inputExtent(length);
				concurrency::tiled_extent< _T_REDUCE_WAVEFRONT_SIZE > tiledExtentReduce = inputExtent.tile< _T_REDUCE_WAVEFRONT_SIZE >();

                // Algorithm is different from cl::reduce. We launch worksize = number of elements here.
                // AMP doesn't have APIs to get CU capacity. Launchable size is great though.

                try
                {
                    concurrency::parallel_for_each(ctl.getAccelerator().get_default_view(),
                                                   tiledExtentReduce,
                                                    [ first,
                                                    szElements,
                                                    length,
                                                    transform_op,
                                                    &result,
                                                    binary_op ]
                                                   ( concurrency::tiled_index<_T_REDUCE_WAVEFRONT_SIZE> t_idx ) restrict(amp)
                    {
						int gx = t_idx.global[0];
						int gloId = gx;
						tile_static oType scratch[_T_REDUCE_WAVEFRONT_SIZE];
						//  Initialize local data store
						unsigned int tileIndex = t_idx.local[0];

						oType accumulator;
						if (gloId < szElements)
						{
							accumulator = transform_op(first[gx]);
							gx += length;
						}
						

						// Loop sequentially over chunks of input vector, reducing an arbitrary size input
						// length into a length related to the number of workgroups
						while (gx < szElements)
						{
							oType element = transform_op(first[gx]);
							accumulator = binary_op(accumulator, element);
							gx += length;
						}
                        
						scratch[tileIndex] = accumulator;
						t_idx.barrier.wait();

						unsigned int tail = szElements - (t_idx.tile[0] * _T_REDUCE_WAVEFRONT_SIZE);

						_T_REDUCE_STEP(tail, tileIndex, 128);
						_T_REDUCE_STEP(tail, tileIndex, 64);
						_T_REDUCE_STEP(tail, tileIndex, 32);
						_T_REDUCE_STEP(tail, tileIndex, 16);
						_T_REDUCE_STEP(tail, tileIndex, 8);
						_T_REDUCE_STEP(tail, tileIndex, 4);
						_T_REDUCE_STEP(tail, tileIndex, 2);
						_T_REDUCE_STEP(tail, tileIndex, 1);


						//  Abort threads that are passed the end of the input vector
						if (gloId >= szElements)
							return;

                      //  Write only the single reduced value for the entire workgroup
                      if (tileIndex == 0)
                      {
                          result[t_idx.tile[ 0 ]] = scratch[0];
                      }

                    });
                     
					oType acc = static_cast<oType>(init);
					std::vector<oType> *cpuPointerReduce = new std::vector<oType>(numTiles);
					concurrency::copy(result, (*cpuPointerReduce).begin());
					for(int i = 0; i < numTiles; ++i)
					{
						acc = binary_op(acc, (*cpuPointerReduce)[i]);
					}
					delete cpuPointerReduce;

					return acc;
                }
                catch(std::exception &e)
                {
                      std::cout << "Exception while calling bolt::amp::reduce parallel_for_each " ;
                      std::cout<< e.what() << std::endl;
                      throw std::exception();
                }

        };


		  // This template is called by the non-detail versions of transform_reduce, it already assumes
        // random access iterators
        // This is called strictly for any non-device_vector iterator
        template< typename InputIterator,
                  typename UnaryFunction,
                  typename oType,
                  typename BinaryFunction >
        oType
        transform_reduce_pick_iterator(
            control &c,
            const InputIterator& first,
            const InputIterator& last,
            const UnaryFunction& transform_op,
            const oType& init,
            const BinaryFunction& reduce_op,
			std::random_access_iterator_tag )
        {
            typedef typename std::iterator_traits<InputIterator>::value_type iType;
            int szElements = static_cast< int >(last - first);
            if (szElements == 0)
                    return init;

            bolt::amp::control::e_RunMode runMode = c.getForceRunMode();  // could be dynamic choice some day.
			if (runMode == bolt::amp::control::Automatic)
			{
				runMode = c.getDefaultPathToRun();
			}

            if (runMode == bolt::amp::control::SerialCpu)
            {
                //Create a temporary array to store the transform result;
                //throw std::exception( "transform_reduce device_vector CPU device not implemented" );
                std::vector<oType> output(szElements);
                std::transform(first, last, output.begin(),transform_op);
                return std::accumulate(output.begin(), output.end(), init, reduce_op);
            }
            else if (runMode == bolt::amp::control::MultiCoreCpu)
            {
#ifdef ENABLE_TBB

                    return bolt::btbb::transform_reduce(first,last,transform_op,init,reduce_op);
#else
                    throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform_reduce is not enabled to be built.",0);
                    return init;
#endif
                 }
            else
            {
                // Map the input iterator to a device_vector
                device_vector< iType, concurrency::array_view > dvInput( first, last, false, c );

                return  transform_reduce_enqueue( c, dvInput.begin( ), dvInput.end( ), transform_op, init, reduce_op );
            }
        };

        // This template is called by the non-detail versions of transform_reduce,
        // it already assumes random access iterators
        // This is called strictly for iterators that are derived from device_vector< T >::iterator
        template< typename DVInputIterator,
                  typename UnaryFunction,
                  typename oType,
                  typename BinaryFunction >
        oType 
        transform_reduce_pick_iterator(
            control &c,
            const DVInputIterator& first,
            const DVInputIterator& last,
            const UnaryFunction& transform_op,
            const oType& init,
            const BinaryFunction& reduce_op,
			bolt::amp::device_vector_tag )
        {
            typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
            int szElements = static_cast< int >(last - first);
            if (szElements == 0)
                    return init;

            bolt::amp::control::e_RunMode runMode = c.getForceRunMode();  // could be dynamic choice some day.
			if (runMode == bolt::amp::control::Automatic)
			{
				runMode = c.getDefaultPathToRun();
			}
            if (runMode == bolt::amp::control::SerialCpu)
            {

			   typename bolt::amp::device_vector< iType >::pointer firstPtr = const_cast<typename bolt::amp::device_vector< iType >::pointer>(first.getContainer( ).data( ));
               std::vector< oType > result ( last.m_Index - first.m_Index );

               std::transform( &firstPtr[ first.m_Index ], &firstPtr[ last.m_Index ], result.begin(), transform_op );

               return std::accumulate(  result.begin(), result.end(), init, reduce_op );

            }
            else if (runMode == bolt::amp::control::MultiCoreCpu)
            {

#ifdef ENABLE_TBB
           
			    typename  bolt::amp::device_vector< iType >::pointer firstPtr = const_cast<typename bolt::amp::device_vector< iType >::pointer>(first.getContainer( ).data( ));

				return  bolt::btbb::transform_reduce(  &firstPtr[ first.m_Index ], &firstPtr[ last.m_Index ],
				                                       transform_op,init,reduce_op);

#else
               throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform_reduce is not enabled to be built.", 0);
               return init;
#endif
            }
            else
            {
				 return  transform_reduce_enqueue( c, first, last, transform_op, init, reduce_op );
            }
        };


		 // This template is called by the non-detail versions of transform_reduce, it already assumes
        // random access iterators
        // This is called strictly for any non-device_vector iterator
        template< typename InputIterator,
                  typename UnaryFunction,
                  typename oType,
                  typename BinaryFunction >
        oType
        transform_reduce_pick_iterator(
            control &c,
            const InputIterator& first,
            const InputIterator& last,
            const UnaryFunction& transform_op,
            const oType& init,
            const BinaryFunction& reduce_op,
			bolt::amp::fancy_iterator_tag )
        {
            typedef typename std::iterator_traits<InputIterator>::value_type iType;
            int szElements = static_cast< int >(last - first);
            if (szElements == 0)
                    return init;

            bolt::amp::control::e_RunMode runMode = c.getForceRunMode();  // could be dynamic choice some day.
			if (runMode == bolt::amp::control::Automatic)
			{
				runMode = c.getDefaultPathToRun();
			}

            if (runMode == bolt::amp::control::SerialCpu)
            {
                //Create a temporary array to store the transform result;
                //throw std::exception( "transform_reduce device_vector CPU device not implemented" );
                std::vector<oType> output(szElements);
                std::transform(first, last, output.begin(),transform_op);
                return std::accumulate(output.begin(), output.end(), init, reduce_op);
            }
            else if (runMode == bolt::amp::control::MultiCoreCpu)
            {
#ifdef ENABLE_TBB

                    return bolt::btbb::transform_reduce(first,last,transform_op,init,reduce_op);
#else
                    throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform_reduce is not enabled to be built.", 0);
                    return init;
#endif
                 }
            else
            {
				return  transform_reduce_enqueue( c, first, last, transform_op, init, reduce_op );
            }
        };
#ifdef _WIN32
		//  The following two functions disallow non-random access functions
        // Wrapper that uses default control class, iterator interface
        template< typename InputIterator,
                  typename UnaryFunction,
                  typename T,
                  typename BinaryFunction >
        T transform_reduce_detect_random_access( control &ctl,
                                                 const InputIterator& first,
                                                 const InputIterator& last,
                                                 const UnaryFunction& transform_op,
                                                 const T& init,
                                                 const BinaryFunction& reduce_op,
                                                 std::input_iterator_tag )
        {
            //  TODO:  It should be possible to support non-random_access_iterator_tag iterators,if we copied the data
            //  to a temporary buffer.  Should we?
            static_assert( false, "Bolt only supports random access iterator types" );
        };
#endif
        // Wrapper that uses default control class, iterator interface
        template< typename InputIterator,
                  typename UnaryFunction,
                  typename T,
                  typename BinaryFunction >
        T transform_reduce_detect_random_access( control& ctl,
                                                 const InputIterator& first,
                                                 const InputIterator& last,
                                                 const UnaryFunction& transform_op,
                                                 const T& init,
                                                 const BinaryFunction& reduce_op,
                                                 std::random_access_iterator_tag )
        {
            return transform_reduce_pick_iterator( ctl, first, last, transform_op, init, reduce_op, 
													typename std::iterator_traits< InputIterator >::iterator_category( ) );
        };


    };// end of namespace detail

	    // The following two functions are visible in .h file
        // Wrapper that user passes a control class
        template< typename InputIterator,
                  typename UnaryFunction,
                  typename T,
                  typename BinaryFunction >
        T transform_reduce( control& ctl,
                            InputIterator first,
                            InputIterator last,
                            UnaryFunction transform_op,
                            T init,
                            BinaryFunction reduce_op )
        {
            return detail::transform_reduce_detect_random_access( ctl, first, last, transform_op, init, reduce_op,
              typename std::iterator_traits< InputIterator >::iterator_category( ) );
        };

        // Wrapper that generates default control class
        template< typename InputIterator,
                  typename UnaryFunction,
                  typename T,
                  typename BinaryFunction >
        T transform_reduce( InputIterator first,
                            InputIterator last,
                            UnaryFunction transform_op,
                            T init,
                            BinaryFunction reduce_op )
        {
            return detail::transform_reduce_detect_random_access( control::getDefault(), first, last, transform_op,
                                                                                                    init, reduce_op,
                typename std::iterator_traits< InputIterator >::iterator_category( ) );
        };

  };// end of namespace amp
};// end of namespace bolt

#endif
