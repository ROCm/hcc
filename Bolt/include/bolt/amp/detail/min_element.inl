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
#if !defined( BOLT_AMP_MINELEMENT_INL )
#define BOLT_AMP_MINELEMENT_INL

#define MIN_MAX_WAVEFRONT_SIZE 256 
#define _REDUCE_STEP_MIN(_LENGTH, _IDX, _W) \
    if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
      iType mine = scratch[_IDX];\
      iType other = scratch[_IDX + _W];\
      bool stat = binary_op(mine, other); \
      scratch[_IDX] = stat ? mine : other ;\
      scratch_index[_IDX] = stat ? scratch_index[_IDX]:scratch_index[_IDX + _W];\
    }\
    t_idx.barrier.wait();


#define _REDUCE_STEP_MAX(_LENGTH, _IDX, _W)\
    if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
      iType mine = scratch[_IDX];\
      iType other = scratch[_IDX + _W];\
        bool stat = binary_op(other, mine);\
        scratch[_IDX] = stat ? mine : other ;\
        scratch_index[_IDX] = stat ? scratch_index[_IDX]:scratch_index[_IDX + _W];\
        }\
   t_idx.barrier.wait();


#pragma once
#include <cmath>
#include <cstring>
#include <algorithm>
#include "bolt/amp/bolt.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/iterator/iterator_traits.h"

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/min_element.h"
#endif

#include <type_traits>
#include "bolt/amp/device_vector.h"

namespace bolt {
    namespace amp {
        namespace detail {

            // This is the base implementation of reduction that is called by all of the convenience wrappers below.
            // first and last must be iterators from a DeviceVector
            template<typename DVInputIterator, typename BinaryPredicate>
            int min_element_enqueue(bolt::amp::control &ctl,
                const DVInputIterator& first,
                const DVInputIterator& last,
                const BinaryPredicate& binary_op,
                const char * min_max )
            {

				typedef typename std::iterator_traits< DVInputIterator >::value_type iType;

				const int szElements = static_cast< int >(std::distance(first, last));
				int max_ComputeUnits = 32;
				int numTiles = max_ComputeUnits*32;	/* Max no. of WG for Tahiti(32 compute Units) and 32 is the tuning factor that gives good performance*/
				int length = (MIN_MAX_WAVEFRONT_SIZE * numTiles);
				length = szElements < length ? szElements : length;
				unsigned int residual = length % MIN_MAX_WAVEFRONT_SIZE;
				length = residual ? (length + MIN_MAX_WAVEFRONT_SIZE - residual): length ;
				numTiles = static_cast< int >((szElements/MIN_MAX_WAVEFRONT_SIZE)>= numTiles?(numTiles):
									(std::ceil( static_cast< float >( szElements ) / MIN_MAX_WAVEFRONT_SIZE) ));

				concurrency::array<unsigned int, 1> result(numTiles);
				concurrency::extent< 1 > inputExtent(length);
				concurrency::tiled_extent< MIN_MAX_WAVEFRONT_SIZE > tiledExtentReduce = inputExtent.tile< MIN_MAX_WAVEFRONT_SIZE >();
                
                const char * str = "MAX_KERNEL";

                if(std::strcmp(min_max,str) != 0)
                {
                //Min Element Code
					try
					{
						concurrency::parallel_for_each(ctl.getAccelerator().get_default_view(),
													   tiledExtentReduce,
													   [ first,
														 szElements,
														 length,
														 &result,
														 binary_op ]
						(concurrency::tiled_index<MIN_MAX_WAVEFRONT_SIZE> t_idx) restrict(amp)
						{
						  int globalId = t_idx.global[ 0 ];
						  int gx = globalId;
						  unsigned int tileIndex = t_idx.local[0];

						  //  Initialize local data store
						  tile_static iType scratch[MIN_MAX_WAVEFRONT_SIZE];
						  tile_static unsigned int scratch_index[MIN_MAX_WAVEFRONT_SIZE];

						  //  Abort threads that are passed the end of the input vector
						  if (globalId < szElements)
						  {
						   //  Initialize the accumulator private variable with data from the input array
						   //  This essentially unrolls the loop below at least once
						   scratch[tileIndex] = first[globalId];
						   scratch_index[tileIndex] = gx;
						   gx += length;
						  }

						  t_idx.barrier.wait();

						  // Loop sequentially over chunks of input vector, reducing an arbitrary size input
						  // length into a length related to the number of workgroups
						  while (gx < szElements)
						  {
							  iType element = first[gx];						  
							  bool stat = binary_op(scratch[tileIndex], element);
							  scratch[tileIndex] = stat ? scratch[tileIndex] : element;						
							  scratch_index[tileIndex] = stat ? scratch_index[tileIndex] : gx;
							  gx += length;
						  }

						  t_idx.barrier.wait();

						  //  Tail stops the last workgroup from reading past the end of the input vector
						  unsigned int tail = szElements - (t_idx.tile[ 0 ] * t_idx.tile_dim0);
						  // Parallel reduction within a given workgroup using local data store
						  // to share values between workitems

						  _REDUCE_STEP_MIN(tail, tileIndex, 128);
						  _REDUCE_STEP_MIN(tail, tileIndex, 64);
						  _REDUCE_STEP_MIN(tail, tileIndex, 32);
						  _REDUCE_STEP_MIN(tail, tileIndex, 16);
						  _REDUCE_STEP_MIN(tail, tileIndex,  8);
						  _REDUCE_STEP_MIN(tail, tileIndex,  4);
						  _REDUCE_STEP_MIN(tail, tileIndex,  2);
						  _REDUCE_STEP_MIN(tail, tileIndex,  1);



						  //  Abort threads that are passed the end of the input vector
						  if (globalId >= szElements)
							  return;

						  //  Write only the single reduced value for the entire workgroup
						  if (tileIndex == 0)
						  {
							   result[t_idx.tile[ 0 ]] = scratch_index[0];
						  }


						});

						std::vector<unsigned int> *cpuPointerReduce = new std::vector<unsigned int>(numTiles);
						concurrency::copy(result, (*cpuPointerReduce).begin());

						iType minele =  first[(*cpuPointerReduce)[0]];
						unsigned int minele_indx = (*cpuPointerReduce)[0];

                
						for (int i = 0; i < numTiles; ++i)
						{
							bool stat = binary_op( minele, first[(*cpuPointerReduce)[i]]);
							minele = stat ? minele : first[(*cpuPointerReduce)[i]];
							minele_indx =  stat ? minele_indx : (*cpuPointerReduce)[i];
						}
						delete cpuPointerReduce;
						return minele_indx ;
					}

					catch(std::exception &e)
					{
						  std::cout << "Exception while calling bolt::amp::min_element parallel_for_each " ;
						  std::cout<< e.what() << std::endl;
						  throw std::exception();
					}		

                }


                else
                {
                    //Max Element Code

					try
					{
						concurrency::parallel_for_each(ctl.getAccelerator().get_default_view(),
							tiledExtentReduce,
							[first,
							szElements,
							length,
							&result,
							binary_op]
						(concurrency::tiled_index<MIN_MAX_WAVEFRONT_SIZE> t_idx) restrict(amp)
						{
							int globalId = t_idx.global[0];
							int gx = globalId;
							unsigned int tileIndex = t_idx.local[0];

							//  Initialize local data store
							tile_static iType scratch[MIN_MAX_WAVEFRONT_SIZE];
							tile_static unsigned int scratch_index[MIN_MAX_WAVEFRONT_SIZE];

							//  Abort threads that are passed the end of the input vector
							if (globalId < szElements)
							{
								//  Initialize the accumulator private variable with data from the input array
								//  This essentially unrolls the loop below at least once
								scratch[tileIndex] = first[globalId];
								scratch_index[tileIndex] = gx;
								gx += length;
							}

							t_idx.barrier.wait();

							// Loop sequentially over chunks of input vector, reducing an arbitrary size input
							// length into a length related to the number of workgroups
							while (gx < szElements)
							{
								iType element = first[gx];
								bool stat = binary_op(scratch[tileIndex], element);
								scratch[tileIndex] = stat ? scratch[tileIndex] : element;
								scratch_index[tileIndex] = stat ? scratch_index[tileIndex] : gx;
								gx += length;
							}

							t_idx.barrier.wait();

							//  Tail stops the last workgroup from reading past the end of the input vector
							unsigned int tail = szElements - (t_idx.tile[0] * t_idx.tile_dim0);
							// Parallel reduction within a given workgroup using local data store
							// to share values between workitems

							_REDUCE_STEP_MAX(tail, tileIndex, 128);
							_REDUCE_STEP_MAX(tail, tileIndex, 64);
							_REDUCE_STEP_MAX(tail, tileIndex, 32);
							_REDUCE_STEP_MAX(tail, tileIndex, 16);
							_REDUCE_STEP_MAX(tail, tileIndex, 8);
							_REDUCE_STEP_MAX(tail, tileIndex, 4);
							_REDUCE_STEP_MAX(tail, tileIndex, 2);
							_REDUCE_STEP_MAX(tail, tileIndex, 1);



							//  Abort threads that are passed the end of the input vector
							if (globalId >= szElements)
								return;

							//  Write only the single reduced value for the entire workgroup
							if (tileIndex == 0)
							{
								result[t_idx.tile[0]] = scratch_index[0];
							}


						});

                    
						std::vector<unsigned int> *cpuPointerReduce = new std::vector<unsigned int>(numTiles);
						concurrency::copy(result, (*cpuPointerReduce).begin());

						iType minele =  first[(*cpuPointerReduce)[0]];
						unsigned int minele_indx = (*cpuPointerReduce)[0];

                
						for (int i = 0; i < numTiles; ++i)
						{
							bool stat = binary_op(first[(*cpuPointerReduce)[i]], minele);
							minele = stat ? minele : first[(*cpuPointerReduce)[i]];
							minele_indx =  stat ? minele_indx : (*cpuPointerReduce)[i];
						}
						delete cpuPointerReduce;
						return minele_indx ;

					}

					catch(std::exception &e)
					{
						  std::cout << "Exception while calling bolt::amp::max_element parallel_for_each " ;
						  std::cout<< e.what() << std::endl;
						  throw std::exception();
					}							    
             
                }
            }



            // This template is called after we detect random access iterators
            // This is called strictly for any non-device_vector iterator
            template<typename ForwardIterator, typename BinaryPredicate>
            ForwardIterator min_element_pick_iterator(bolt::amp::control &ctl,
                const ForwardIterator& first,
                const ForwardIterator& last,
                const BinaryPredicate& binary_op,
                const char * min_max,
                std::random_access_iterator_tag)
            {

                typedef typename std::iterator_traits<ForwardIterator>::value_type iType;
                int szElements = static_cast< int >(last - first);
                if (szElements == 0)
                    return last;
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
            
                const char * str = "MAX_KERNEL";

                switch(runMode)
                {

                case bolt::amp::control::MultiCoreCpu:
                    #ifdef ENABLE_TBB   
                        if(std::strcmp(min_max,str) == 0)
                              return bolt::btbb::max_element(first, last, binary_op);
                        else
                              return bolt::btbb::min_element(first, last, binary_op);
                    #else
                        throw std::runtime_error( "The MultiCoreCpu version of Max-Min is not enabled to be built! \n" );
                    #endif

                case bolt::amp::control::SerialCpu:
                  {	
                    if(std::strcmp(min_max,str) == 0)
                       return std::max_element(first, last, binary_op);
                    else
                       return std::min_element(first, last, binary_op);
                  }
                default:
                  { 
                    {
                    device_vector< iType, concurrency::array_view > dvInput( first, last, false, ctl );
                    int  dvminele  =  min_element_enqueue( ctl, dvInput.begin(), dvInput.end(),  binary_op, min_max);
                    return first + dvminele;
                    }
                  }
                }

            }

            // This template is called after we detect random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
           template<typename DVInputIterator, typename BinaryPredicate>
           DVInputIterator min_element_pick_iterator(bolt::amp::control &ctl,
                const DVInputIterator& first,
                const DVInputIterator& last,
                const BinaryPredicate& binary_op,
                const char * min_max,
                bolt::amp::device_vector_tag)
            {
				typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
                int szElements = static_cast< int >(last - first);
                if (szElements == 0)
                    return last;

                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if (runMode == bolt::amp::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
         
                const char * str = "MAX_KERNEL";
            
                switch(runMode)
                {

                case bolt::amp::control::MultiCoreCpu:
					{
                    #ifdef ENABLE_TBB   
						typename bolt::amp::device_vector< iType >::pointer InputBuffer =  first.getContainer( ).data( );
						iType* stlPtr;
                        if(std::strcmp(min_max,str) == 0)
                              stlPtr = bolt::btbb::max_element(&InputBuffer[first.m_Index], &InputBuffer[last.m_Index], binary_op);
                        else
                              stlPtr = bolt::btbb::min_element(&InputBuffer[first.m_Index], &InputBuffer[last.m_Index], binary_op);
						return first+(unsigned int)(stlPtr-&InputBuffer[first.m_Index]);
                    #else
                        throw std::runtime_error( "The MultiCoreCpu version of Max-Min is not enabled to be built! \n" );
                    #endif
					}
                case bolt::amp::control::SerialCpu:
					{
						typename bolt::amp::device_vector< iType >::pointer InputBuffer =  first.getContainer( ).data( );
						iType* stlPtr;
						if(std::strcmp(min_max,str) == 0)
						   stlPtr = std::max_element(&InputBuffer[first.m_Index], &InputBuffer[last.m_Index], binary_op);
						else
						   stlPtr = std::min_element(&InputBuffer[first.m_Index], &InputBuffer[last.m_Index], binary_op);
						return first+(unsigned int)(stlPtr-&InputBuffer[first.m_Index]);
					}
                default:
                     {
						 int minele = min_element_enqueue( ctl, first, last,  binary_op, min_max);
						 return first + minele;
                     }

                }

            }


            // This template is called after we detect random access iterators
            // This is called strictly for iterators that are derived from fancy_iterators
           template<typename DVInputIterator, typename BinaryPredicate>
           DVInputIterator min_element_pick_iterator(bolt::amp::control &ctl,
                const DVInputIterator& first,
                const DVInputIterator& last,
                const BinaryPredicate& binary_op,
                const char * min_max,
                bolt::amp::fancy_iterator_tag)
            {
                int szElements = static_cast< int >(last - first);
                if (szElements == 0)
                    return last;

                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.

                if (runMode == bolt::amp::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
         
                const char * str = "MAX_KERNEL";
            
                switch(runMode)
                {

                case bolt::amp::control::MultiCoreCpu:
                    #ifdef ENABLE_TBB   
                        if(std::strcmp(min_max,str) == 0)
                              return bolt::btbb::max_element(first, last, binary_op);
                        else
                              return bolt::btbb::min_element(first, last, binary_op);
                    #else
                        throw std::runtime_error( "The MultiCoreCpu version of Max-Min is not enabled to be built! \n" );
                    #endif

                case bolt::amp::control::SerialCpu:
                    if(std::strcmp(min_max,str) == 0)
                       return std::max_element(first, last, binary_op);
                    else
                       return std::min_element(first, last, binary_op);

                default:
                     {
                     int minele = min_element_enqueue( ctl, first, last,  binary_op, min_max);
                     return first + minele;
                     }

                }

            }

            template<typename ForwardIterator, typename BinaryPredicate>
            ForwardIterator min_element_detect_random_access(bolt::amp::control &ctl,
                const ForwardIterator& first,
                const ForwardIterator& last,
                const BinaryPredicate& binary_op,
                std::input_iterator_tag)
            {
                //TODO:It should be possible to support non-random_access_iterator_tag iterators,if we copied the data
                //to a temporary buffer.  Should we?
                static_assert( std::is_same< ForwardIterator, std::forward_iterator_tag   >::value, "Bolt only supports random access iterator types" );
            }

             template<typename ForwardIterator, typename BinaryPredicate>
             ForwardIterator max_element_detect_random_access(bolt::amp::control &ctl,
                 const ForwardIterator& first,
                 const ForwardIterator& last,
                 const BinaryPredicate& binary_op,
                 std::input_iterator_tag)
             {
                 //TODO:It should be possible to support non-random_access_iterator_tag iterators,if we copied the data
                 //to a temporary buffer.  Should we?
                 static_assert( std::is_same< ForwardIterator, std::forward_iterator_tag   >::value, "Bolt only supports random access iterator types" );
             }

            template<typename ForwardIterator, typename BinaryPredicate>
            ForwardIterator min_element_detect_random_access(bolt::amp::control &ctl,
                const ForwardIterator& first,
                const ForwardIterator& last,
                const BinaryPredicate& binary_op,
                std::random_access_iterator_tag)
            {
                const char * str = "MIN_KERNEL";
                return min_element_pick_iterator( ctl, first, last,  binary_op, str,
                                                  typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
            }



            template<typename ForwardIterator, typename BinaryPredicate>
            ForwardIterator max_element_detect_random_access(bolt::amp::control &ctl,
                const ForwardIterator& first,
                const ForwardIterator& last,
                const BinaryPredicate& binary_op,
                std::random_access_iterator_tag)
            {
                const char * str = "MAX_KERNEL";
                return min_element_pick_iterator( ctl, first, last,  binary_op, str,
                                                  typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
            }


        };


        template<typename ForwardIterator>
        ForwardIterator max_element(ForwardIterator first,
            ForwardIterator last)
        {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T;
            return detail::max_element_detect_random_access(bolt::amp::control::getDefault(), first, last, bolt::amp::less<T>(),  
                typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        };


        template<typename ForwardIterator,typename BinaryPredicate>
        ForwardIterator max_element(ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op)
        {
            return detail::max_element_detect_random_access(bolt::amp::control::getDefault(), first, last, binary_op,
                typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        };



        template<typename ForwardIterator>
        ForwardIterator  max_element(bolt::amp::control &ctl,
            ForwardIterator first,
            ForwardIterator last)
        {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T;
            return detail::max_element_detect_random_access(ctl, first, last, bolt::amp::less<T>(),
                typename std::iterator_traits< ForwardIterator >::iterator_category( ));
        };

        // This template is called by all other "convenience" version of max_element.
        // It also implements the CPU-side mappings of the algorithm for SerialCpu and MultiCoreCpu
        template<typename ForwardIterator, typename BinaryPredicate>
        ForwardIterator max_element(bolt::amp::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op)
            {
                    return detail::max_element_detect_random_access(ctl, first, last, binary_op, 
                        typename std::iterator_traits< ForwardIterator >::iterator_category( ));
            }


        // This template is called by all other "convenience" version of min_element.
        // It also implements the CPU-side mappings of the algorithm for SerialCpu and MultiCoreCpu
        template<typename ForwardIterator, typename BinaryPredicate>
        ForwardIterator min_element(bolt::amp::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op)
             {
               return detail::min_element_detect_random_access(ctl, first, last, binary_op,
                   typename std::iterator_traits< ForwardIterator >::iterator_category( ));
            }


        template<typename ForwardIterator>
        ForwardIterator min_element(ForwardIterator first,
            ForwardIterator last)
        {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T;
            return min_element(bolt::amp::control::getDefault(), first, last, bolt::amp::less<T>());
        };


        template<typename ForwardIterator,typename BinaryPredicate>
        ForwardIterator min_element(ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op)
        {
            return min_element(bolt::amp::control::getDefault(), first, last, binary_op);
        };



        template<typename ForwardIterator>
        ForwardIterator  min_element(bolt::amp::control &ctl,
            ForwardIterator first,
            ForwardIterator last)
        {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T;
            return min_element(ctl, first, last, bolt::amp::less<T>());
        };

       };

};

#endif
