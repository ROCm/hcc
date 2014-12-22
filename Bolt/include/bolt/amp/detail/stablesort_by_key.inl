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
#if !defined( BOLT_AMP_STABLESORT_BY_KEY_INL )
#define BOLT_AMP_STABLESORT_BY_KEY_INL

#include <algorithm>
#include <type_traits>

#include "bolt/amp/bolt.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/sort_by_key.h"
#include "bolt/amp/stablesort_by_key.h"
#include "bolt/amp/pair.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include <amp.h>

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/stable_sort_by_key.h"
#endif

#define STABLESORT_BY_KEY_BUFFER_SIZE 256
#define STABLESORTBYKEY_TILE_MAX 65535


namespace bolt {
namespace amp {
namespace detail
{
    
    //Serial CPU code path implementation.
    //Class to hold the key value pair. This will be used to zip th ekey and value together in a vector.
    template <typename keyType, typename valueType>
    class std_stable_sort
    {
    public:
        keyType   key;
        valueType value;
    };
    //This is the functor which will sort the std_stable_sort vector.
    template <typename keyType, typename valueType, typename StrictWeakOrdering>
    class std_stable_sort_comp
    {
    public:
        typedef std_stable_sort<keyType, valueType> KeyValueType;
        std_stable_sort_comp(const StrictWeakOrdering &_swo):swo(_swo)
        {}
        StrictWeakOrdering swo;
        bool operator() (const KeyValueType &lhs, const KeyValueType &rhs) const
        {
            return swo(lhs.key, rhs.key);
        }
    };

    //The serial CPU implementation of stable_sort_by_key routine. This routines zips the key value pair and then sorts
    //using the std::stable_sort routine.
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void serialCPU_stable_sort_by_key(const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                      const RandomAccessIterator2 values_first,
                                      const StrictWeakOrdering& comp)
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< RandomAccessIterator2 >::value_type valType;
        typedef std_stable_sort<keyType, valType> KeyValuePair;
        typedef std_stable_sort_comp<keyType, valType, StrictWeakOrdering> KeyValuePairFunctor;

        int vecSize = static_cast< int >(std::distance( keys_first, keys_last ));
        std::vector<KeyValuePair> KeyValuePairVector(vecSize);
        KeyValuePairFunctor functor(comp);
        //Zip the key and values iterators into a std_stable_sort vector.
        for (int i=0; i< vecSize; i++)
        {
            KeyValuePairVector[i].key   = *(keys_first + i);
            KeyValuePairVector[i].value = *(values_first + i);
        }
        //Sort the std_stable_sort vector using std::stable_sort
        std::stable_sort(KeyValuePairVector.begin(), KeyValuePairVector.end(), functor);
        //Extract the keys and values from the KeyValuePair and fill the respective iterators.
        for (int i=0; i< vecSize; i++)
        {
            *(keys_first + i)   = KeyValuePairVector[i].key;
            *(values_first + i) = KeyValuePairVector[i].value;
        }
    }


    /**************************************************************/
    /**StableSort_by_key with keys as unsigned int specialization**/
    /**************************************************************/
    template< typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering >
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator1 >::value_type,
                                           unsigned int
                                         >::value
                           >::type   /*If enabled then this typename will be evaluated to void*/
    stablesort_by_key_enqueue( control& ctrl,
                                    const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                                    const DVRandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp )
    {
		
        bolt::amp::detail::sort_by_key_enqueue(ctrl, keys_first, keys_last, values_first, comp);
        return;    
    }

    /**************************************************************/
    /*StableSort_by_key with keys as int specialization*/
    /**************************************************************/
    template< typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering >
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator1 >::value_type,
                                           int
                                         >::value
                           >::type   /*If enabled then this typename will be evaluated to void*/
    stablesort_by_key_enqueue( control& ctrl,
                               const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                               const DVRandomAccessIterator2 values_first,
                               const StrictWeakOrdering& comp )
    {
		
        bolt::amp::detail::sort_by_key_enqueue(ctrl, keys_first, keys_last, values_first, comp);
        return;    
    }


	template< typename sType, typename Container, typename StrictWeakOrdering >
    unsigned int sort_by_key_lowerBoundBinary( Container& data, int left, int right, sType searchVal, StrictWeakOrdering& lessOp ) restrict(amp)
    {
        //  The values firstIndex and lastIndex get modified within the loop, narrowing down the potential sequence
        int firstIndex = left;
        int lastIndex = right;
        
        //  This loops through [firstIndex, lastIndex)
        //  Since firstIndex and lastIndex will be different for every thread depending on the nested branch,
        //  this while loop will be divergent within a wavefront
        while( firstIndex < lastIndex )
        {
            //  midIndex is the average of first and last, rounded down
            unsigned int midIndex = ( firstIndex + lastIndex ) / 2;
            sType midValue = data[ midIndex ];
            
            //  This branch will create divergent wavefronts
            if( lessOp( midValue, searchVal ) )
            {
                firstIndex = midIndex+1;
                 //printf( "lowerBound: lastIndex[ %i ]=%i\n", get_local_id( 0 ), lastIndex );
            }
            else
            {
                lastIndex = midIndex;
                 //printf( "lowerBound: firstIndex[ %i ]=%i\n", get_local_id( 0 ), firstIndex );
            }
        }
        //printf("sort_by_key_lowerBoundBinary: left=%d, right=%d, firstIndex=%d\n", left, right, firstIndex);
        return firstIndex;
    }
    
    //  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
    //  by a base pointer and left and right index for a particular candidate value.  The comparison operator is 
    //  passed as a functor parameter lessOp
    //  This function returns an index that is the first index whos value would be greater than the searched value
    //  If the search value is not found in the sequence, upperbound returns the same result as lowerbound
    template< typename sType, typename Container, typename StrictWeakOrdering >
    unsigned int  sort_by_key_upperBoundBinary( Container& data, unsigned int left, unsigned int right, sType searchVal, StrictWeakOrdering& lessOp ) restrict(amp)
    {
        unsigned int upperBound = sort_by_key_lowerBoundBinary( data, left, right, searchVal, lessOp );
        
         //printf( "start of sort_by_key_upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right );
        //  upperBound is always between left and right or equal to right
        //  If upperBound == right, then  searchVal was not found in the sequence.  Just return.
        if( upperBound != right )
        {
            //  While the values are equal i.e. !(x < y) && !(y < x) increment the index
            int mid = 0;
            sType upperValue = data[ upperBound ];
            //This loop is a kind of a specialized binary search. 
            //This will find the first index location which is not equal to searchVal.
            while( !lessOp( upperValue, searchVal ) && !lessOp( searchVal, upperValue) && (upperBound < right))
            {
                mid = (upperBound + right)/2;
                sType midValue = data[mid];
                if( !lessOp( midValue, searchVal ) && !lessOp( searchVal, midValue) )
                {
                    upperBound = mid + 1;
                }   
                else
                {
                    right = mid;
                    upperBound++;
                }
                upperValue = data[ upperBound ];
                //printf( "sort_by_key_upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right);
            }
        }
        //printf( "end of sort_by_key_upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right);
        return upperBound;
    }

    #define max(a,b)    (((a) > (b)) ? (a) : (b))
    #define min(a,b)    (((a) < (b)) ? (a) : (b))


    template< typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering >
    typename std::enable_if<
    !(std::is_same< typename std::iterator_traits<DVRandomAccessIterator1 >::value_type, unsigned int >::value || 
      std::is_same< typename std::iterator_traits<DVRandomAccessIterator1 >::value_type, int >::value  )
                       >::type
    stablesort_by_key_enqueue( control& ctrl,
                                    const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                                    const DVRandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp)
    {

		concurrency::accelerator_view av = ctrl.getAccelerator().get_default_view();
        int vecSize = static_cast< int >( std::distance( keys_first, keys_last ) );

        typedef typename std::iterator_traits< DVRandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< DVRandomAccessIterator2 >::value_type valueType;

        const unsigned int localRange= STABLESORT_BY_KEY_BUFFER_SIZE;
		
        //  Make sure that globalRange is a multiple of localRange
        unsigned int globalRange = vecSize;
        unsigned int modlocalRange = ( globalRange & ( localRange-1 ) );
        if( modlocalRange )
        {
            globalRange &= ~modlocalRange;
            globalRange += localRange;
        }

	    /*********************************************************************************
          Kernel 0
        *********************************************************************************/

		const unsigned int tile_limit = STABLESORTBYKEY_TILE_MAX;
		const unsigned int max_ext = (tile_limit*localRange);

		unsigned int	   tempBuffsize = globalRange; 
		unsigned int	   iteration = (globalRange-1)/max_ext; 

		for(unsigned int i=0; i<=iteration; i++)
		{
			unsigned int extent_sz =  (tempBuffsize > max_ext) ? max_ext : tempBuffsize; 
			concurrency::extent< 1 > inputExtent( extent_sz );
			concurrency::tiled_extent< localRange > tileK0 = inputExtent.tile< localRange >();
			unsigned int index = i*(tile_limit*localRange);
			unsigned int tile_index = i*tile_limit;

				try
				{

				  concurrency::parallel_for_each( av, tileK0,
				  [
					values_first,
					keys_first,
					vecSize,
					comp,
					index,
					tile_index,
					localRange
				  ] ( concurrency::tiled_index< localRange > t_idx ) restrict(amp)
				  {

					   int gloId = t_idx.global[ 0 ] + index;
					   int groId = t_idx.tile[ 0 ] + tile_index;
					   unsigned int locId = t_idx.local[ 0 ];
					   int wgSize = localRange;

					   tile_static keyType key_lds[STABLESORT_BY_KEY_BUFFER_SIZE]; 
					   tile_static keyType key_lds2[STABLESORT_BY_KEY_BUFFER_SIZE]; 
					   tile_static valueType val_lds[STABLESORT_BY_KEY_BUFFER_SIZE]; 
					   tile_static valueType val_lds2[STABLESORT_BY_KEY_BUFFER_SIZE]; 

					   //  Make a copy of the entire input array into fast local memory
					   if( gloId < vecSize)
					   {
						  key_lds[ locId ] = keys_first[ gloId ];
						  val_lds[ locId ] = values_first[ gloId ];
					   }
					   //barrier( CLK_LOCAL_MEM_FENCE );
					   unsigned int end =  wgSize;
					   if( (groId+1)*(wgSize) >= vecSize )
							  end = vecSize - (groId*wgSize);

					   unsigned int numMerges = 8;
					   unsigned int pass;

					   for( pass = 1; pass <= numMerges; ++pass )
					   {
						 unsigned int srcLogicalBlockSize = 1 << (pass-1);
						 if( gloId < vecSize)
						 {
  							  unsigned int srcBlockNum = (locId) / srcLogicalBlockSize;
							  unsigned int srcBlockIndex = (locId) % srcLogicalBlockSize;
    
							  unsigned int dstLogicalBlockSize = srcLogicalBlockSize<<1;
							  unsigned int leftBlockIndex = (locId)  & ~(dstLogicalBlockSize - 1 );

							  leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
							  leftBlockIndex = min( leftBlockIndex, end );
							  unsigned int rightBlockIndex = min( leftBlockIndex + srcLogicalBlockSize,  end  );
  							  unsigned int insertionIndex = 0;
							  if(pass%2 != 0)
							  {
								 if( (srcBlockNum & 0x1) == 0 )
								 {
				          			insertionIndex = sort_by_key_lowerBoundBinary( key_lds, leftBlockIndex, rightBlockIndex, key_lds[ locId ], comp ) - leftBlockIndex;
								 }
								 else
								 {
									insertionIndex = sort_by_key_upperBoundBinary( key_lds, leftBlockIndex, rightBlockIndex, key_lds[ locId ], comp ) - leftBlockIndex;
								 }
							 }
							 else
							 {
			          			if( (srcBlockNum & 0x1) == 0 )
								{
								   insertionIndex = sort_by_key_lowerBoundBinary( key_lds2, leftBlockIndex, rightBlockIndex, key_lds2[ locId ], comp ) - leftBlockIndex;
								}
								else
								{
								   insertionIndex = sort_by_key_upperBoundBinary( key_lds2, leftBlockIndex, rightBlockIndex, key_lds2[ locId ], comp ) - leftBlockIndex;
								} 
							}
							unsigned int dstBlockIndex = srcBlockIndex + insertionIndex;
							unsigned int dstBlockNum = srcBlockNum/2;
							if(pass%2 != 0)
							{
								  key_lds2[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = key_lds[ locId ];
								  val_lds2[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = val_lds[ locId ];
							}
							else
							{
								  key_lds[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = key_lds2[ locId ]; 
								  val_lds[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = val_lds2[ locId ];
							}
					  }
					  t_idx.barrier.wait();
					}	  
					if( gloId < vecSize)
					{
						keys_first[ gloId ] = key_lds[ locId ];
						values_first[ gloId ] = val_lds[ locId ];
					}
    
				} );

				}
 
				catch(std::exception &e)
				{
					  std::cout << "Exception while calling bolt::amp::stablesort_by_key parallel_for_each " ;
					  std::cout<< e.what() << std::endl;
					  throw std::exception();
				}	
				tempBuffsize = tempBuffsize - max_ext;
		}

        //  An odd number of elements requires an extra merge pass to sort
        unsigned int numMerges = 0;

        //  Calculate the log2 of vecSize, taking into account our block size from kernel 1 is 64
        //  this is how many merge passes we want
        unsigned int log2BlockSize = vecSize >> 8;
        for( ; log2BlockSize > 1; log2BlockSize >>= 1 )
        {
            ++numMerges;
        }

        //  Check to see if the input vector size is a power of 2, if not we will need last merge pass
        unsigned int vecPow2 = (vecSize & (vecSize-1));
        numMerges += vecPow2? 1: 0;

        //  Allocate a flipflop buffer because the merge passes are out of place
		concurrency::array< keyType >  tmpKeyBuffer( vecSize, av );
		concurrency::array< valueType >  tmpValueBuffer( vecSize, av );


		/**********************************************************************************
        *  Kernel 1
        *********************************************************************************/
    
        concurrency::extent< 1 > globalSizeK1( globalRange );

        for( unsigned int pass = 1; pass <= numMerges; ++pass )
        {

             //  For each pass, the merge window doubles
             int srcLogicalBlockSize = static_cast< int  >( localRange << (pass-1) );

             try
             {
                concurrency::parallel_for_each( av, globalSizeK1,
                [
                  values_first,
				  keys_first,
                  &tmpKeyBuffer,
				  &tmpValueBuffer,
                  vecSize,
                  comp,
                  localRange,
                  srcLogicalBlockSize,
			      pass
                ] ( concurrency::index< 1 > idx ) restrict(amp)
             {
                  int gloID = idx[ 0 ];

                  //  Abort threads that are passed the end of the input vector
                  if( gloID >= vecSize )
                     return; // on SI this doesn't mess-up barriers

                  //  For an element in sequence A, find the lowerbound index for it in sequence B
                  unsigned int srcBlockNum = gloID / srcLogicalBlockSize;
                  unsigned int srcBlockIndex = gloID % srcLogicalBlockSize;
    
                  //printf( "mergeTemplate: srcBlockNum[%i]=%i\n", srcBlockNum, srcBlockIndex );

                  //  Pairs of even-odd blocks will be merged together 
                  //  An even block should search for an insertion point in the next odd block, 
                  //  and the odd block should look for an insertion point in the corresponding previous even block
                  unsigned int dstLogicalBlockSize = srcLogicalBlockSize<<1;
                  int leftBlockIndex = gloID & ~(dstLogicalBlockSize - 1 );
                  //printf("mergeTemplate: leftBlockIndex=%d\n", leftBlockIndex );
                  leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
                  leftBlockIndex = min( leftBlockIndex, vecSize );
                  int rightBlockIndex = min( leftBlockIndex + srcLogicalBlockSize, vecSize );
                  
	              //  For a particular element in the input array, find the lowerbound index for it in the search sequence given by leftBlockIndex & rightBlockIndex
                  // uint insertionIndex = lowerBoundLinear( source_ptr, leftBlockIndex, rightBlockIndex, source_ptr[ globalID ], lessOp ) - leftBlockIndex;
                  unsigned int insertionIndex = 0;
	              
	              if( pass & 0x1 )
	              {
                    if( (srcBlockNum & 0x1) == 0 )
                    {
					  insertionIndex = sort_by_key_lowerBoundBinary( keys_first, leftBlockIndex, rightBlockIndex, keys_first[ gloID ], comp ) - leftBlockIndex;
                    }
                    else
                    {
                      insertionIndex = sort_by_key_upperBoundBinary( keys_first, leftBlockIndex, rightBlockIndex, keys_first[ gloID ], comp ) - leftBlockIndex;
                    }
                  
                    //  The index of an element in the result sequence is the summation of it's indixes in the two input 
                    //  sequences
                    unsigned int dstBlockIndex = srcBlockIndex + insertionIndex;
                    unsigned int dstBlockNum = srcBlockNum/2;
                  
                    tmpValueBuffer[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = values_first[ gloID ];
					tmpKeyBuffer[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = keys_first[ gloID ];
	              }
	              
	              else
	              {
                  
	                if( (srcBlockNum & 0x1) == 0 )
                    {
                      insertionIndex = sort_by_key_lowerBoundBinary( tmpKeyBuffer, leftBlockIndex, rightBlockIndex, tmpKeyBuffer[ gloID ], comp ) - leftBlockIndex;
                    }
                    else
                    {
                      insertionIndex = sort_by_key_upperBoundBinary( tmpKeyBuffer, leftBlockIndex, rightBlockIndex, tmpKeyBuffer[ gloID ], comp ) - leftBlockIndex;
                    }
                  
                    //  The index of an element in the result sequence is the summation of it's indixes in the two input 
                    //  sequences
                    unsigned int dstBlockIndex = srcBlockIndex + insertionIndex;
                    unsigned int dstBlockNum = srcBlockNum/2;

                    values_first[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = tmpValueBuffer[ gloID ];
					keys_first[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = tmpKeyBuffer[ gloID ];
	              
	              }  
             } );

         }

         catch(std::exception &e)
         {
             std::cout << "Exception while calling bolt::amp::stablesort_by_key parallel_for_each " ;
             std::cout<< e.what() << std::endl;
             throw std::exception();
         }	 
		}

		 //  If there are an odd number of merges, then the output data is sitting in the temp buffer.  We need to copy
         //  the results back into the input array
		if( numMerges & 1 )
	    {
		   concurrency::extent< 1 > modified_ext( vecSize );
		   tmpValueBuffer.section( modified_ext ).copy_to( values_first.getContainer().getBuffer(values_first, vecSize) );
		   tmpKeyBuffer.section( modified_ext ).copy_to( keys_first.getContainer().getBuffer(keys_first, vecSize) );
		}

        return;
    }// END of stablesort_by_key_enqueue


    //Non Device Vector specialization.
    //This implementation creates a Buffer and passes the buffer to the sort specialization whichtakes the AMP buffer as a parameter.
    //In the future, Each input buffer should be mapped to the device_vector and the specialization specific to device_vector should be called.
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_pick_iterator( control &ctl,
                                const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                const RandomAccessIterator2 values_first,
                                const StrictWeakOrdering& comp, 
                                std::random_access_iterator_tag, std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< RandomAccessIterator2 >::value_type valType;

        unsigned int vecSize = (unsigned int)std::distance( keys_first, keys_last );
        if( vecSize < 2 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();

				

        if( runMode == bolt::amp::control::Automatic )
        {
            runMode = ctl.getDefaultPathToRun( );

        }
      
        if( runMode == bolt::amp::control::SerialCpu )
        { 
		   
            serialCPU_stable_sort_by_key(keys_first, keys_last, values_first, comp);
            return;
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
            #ifdef ENABLE_TBB
                bolt::btbb::stable_sort_by_key(keys_first, keys_last, values_first, comp);
            #else
                throw std::runtime_error("MultiCoreCPU Version of stable_sort_by_key not Enabled! \n");
            #endif
            return;
        }
        else
        {
		   
			device_vector< keyType, concurrency::array_view > dvKeys(   keys_first, keys_last, false, ctl );
			device_vector< valType, concurrency::array_view > dvValues(  values_first, vecSize, false, ctl );

            //Now call the actual AMP algorithm
            stablesort_by_key_enqueue( ctl, dvKeys.begin(), dvKeys.end(), dvValues.begin( ), comp);

            //Map the buffer back to the host
            dvKeys.data( );
            dvValues.data( );
            return;
        }
    }

    //Device Vector specialization
    template< typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_pick_iterator( control &ctl,
                                    const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                                    const DVRandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp,
                                    bolt::amp::device_vector_tag, bolt::amp::device_vector_tag )
    {
        typedef typename std::iterator_traits< DVRandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< DVRandomAccessIterator2 >::value_type valueType;
        int vecSize = static_cast< int >(std::distance( keys_first, keys_last ));
        if( vecSize < 2 )
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();

        if( runMode == bolt::amp::control::Automatic )
        {
            runMode = ctl.getDefaultPathToRun( );
        }
       
		
        if( runMode == bolt::amp::control::SerialCpu )
        {
                typename bolt::amp::device_vector< keyType >::pointer   keysPtr   =  const_cast<typename bolt::amp::device_vector< keyType >::pointer>(keys_first.getContainer( ).data( ));
                typename bolt::amp::device_vector< valueType >::pointer valuesPtr =  const_cast<typename bolt::amp::device_vector< valueType >::pointer>(values_first.getContainer( ).data( ));
                serialCPU_stable_sort_by_key(&keysPtr[keys_first.m_Index], &keysPtr[keys_last.m_Index],
                                             &valuesPtr[values_first.m_Index], comp);
                return;
        }
        else if( runMode == bolt::amp::control::MultiCoreCpu )
        {
            #ifdef ENABLE_TBB	   
                typename bolt::amp::device_vector< keyType >::pointer   keysPtr   =  const_cast<typename bolt::amp::device_vector< keyType >::pointer>(keys_first.getContainer( ).data( ));
                typename bolt::amp::device_vector< valueType >::pointer valuesPtr =  const_cast<typename bolt::amp::device_vector< valueType >::pointer>(values_first.getContainer( ).data( ));
                bolt::btbb::stable_sort_by_key(&keysPtr[keys_first.m_Index], &keysPtr[keys_last.m_Index],
                                             &valuesPtr[values_first.m_Index], comp);
                return;
             #else
                throw std::runtime_error("MultiCoreCPU Version of stable_sort_by_key not Enabled! \n");
             #endif
        }
        else
        {
            stablesort_by_key_enqueue( ctl, keys_first, keys_last, values_first, comp);
        }
        return;
    }

    //Fancy iterator specialization
    template<typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering>
    void stablesort_by_key_pick_iterator( control &ctl,
                                    const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                                    const DVRandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp,  bolt::amp::fancy_iterator_tag )
    {
        static_assert(std::is_same< DVRandomAccessIterator1, bolt::amp::fancy_iterator_tag >::value  , "It is not possible to output to fancy iterators; they are not mutable! " );
    }


    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, 
                                    std::random_access_iterator_tag, std::random_access_iterator_tag )
    {
        return stablesort_by_key_pick_iterator( ctl, keys_first, keys_last, values_first,
                                    comp, 
                                    typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                    typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
    };

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, 
                                    bolt::amp::fancy_iterator_tag, std::input_iterator_tag )
    {
        static_assert(std::is_same< RandomAccessIterator1, bolt::amp::fancy_iterator_tag>::value, "It is not possible to sort fancy iterators. They are not mutable" );
        static_assert(std::is_same< RandomAccessIterator2,std::input_iterator_tag >::value  , "It is not possible to sort fancy iterators. They are not mutable" );
    }
    // Wrapper that uses default control class, iterator interface
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, 
                                    std::input_iterator_tag, std::input_iterator_tag )
    {
        //  \TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
        //  to a temporary buffer.  Should we?
        static_assert(std::is_same< RandomAccessIterator1,std::input_iterator_tag>::value  , "Bolt only supports random access iterator types" );
        static_assert(std::is_same< RandomAccessIterator2,std::input_iterator_tag>::value  , "Bolt only supports random access iterator types" );
    };


    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, 
                                    std::input_iterator_tag, bolt::amp::fancy_iterator_tag )
    {


        static_assert(std::is_same< RandomAccessIterator2, bolt::amp::fancy_iterator_tag>::value, "It is not possible to sort fancy iterators. They are not mutable" );
        static_assert(std::is_same< RandomAccessIterator1,std::input_iterator_tag >::value  , "It is not possible to sort fancy iterators. They are not mutable" );

    }



    }//namespace bolt::amp::detail


    template< typename RandomAccessIterator1, typename RandomAccessIterator2 >
    void stable_sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first)
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type T;

        detail::stablesort_by_key_detect_random_access( control::getDefault( ),
                                           keys_first, keys_last, values_first,
                                           less< T >( ), 
                                           typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                           typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
        return;
    }

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stable_sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, StrictWeakOrdering comp)
    {
        detail::stablesort_by_key_detect_random_access( control::getDefault( ),
                                           keys_first, keys_last, values_first,
                                           comp, 
                                           typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                           typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
        return;
    }

    template< typename RandomAccessIterator1, typename RandomAccessIterator2 >
    void stable_sort_by_key( control &ctl, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first)
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type T;

        detail::stablesort_by_key_detect_random_access(ctl,
                                           keys_first, keys_last, values_first,
                                          less< T >( ), 
                                           typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                           typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
        return;
    }

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stable_sort_by_key( control &ctl, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, StrictWeakOrdering comp)
    {
        detail::stablesort_by_key_detect_random_access(ctl,
                                           keys_first, keys_last, values_first,
                                          comp, 
                                           typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                           typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
        return;
    }

}//namespace bolt::amp
}//namespace bolt

#endif
