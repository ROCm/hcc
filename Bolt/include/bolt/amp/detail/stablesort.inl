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
#if !defined( BOLT_AMP_STABLESORT_INL )
#define BOLT_AMP_STABLESORT_INL
#define STABLESORT_BUFFER_SIZE 512
#define STABLESORT_TILE_MAX 65535
#include <algorithm>
#include <type_traits>

#include "bolt/amp/bolt.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include <amp.h>

#include "bolt/amp/sort.h"
#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/stable_sort.h"
#endif

namespace bolt {
namespace amp {

namespace detail
{

template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
void sort_enqueue_int_uint(bolt::amp::control &ctl,
             DVRandomAccessIterator &first, DVRandomAccessIterator &last,
             StrictWeakOrdering comp,
                         bool int_flag
                         );


template< typename DVRandomAccessIterator, typename StrictWeakOrdering >
typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,
                                       unsigned int
                                     >::value
                       >::type  /*If enabled then this typename will be evaluated to void*/
stablesort_enqueue(control &ctl,
             DVRandomAccessIterator first, DVRandomAccessIterator last,
             StrictWeakOrdering comp)
{
    bolt::amp::detail::sort_enqueue_int_uint(ctl, first, last, comp, true);
    return;
}

template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,
                                       int
                                     >::value
                       >::type  /*If enabled then this typename will be evaluated to void*/
stablesort_enqueue(control &ctl,
             DVRandomAccessIterator first, DVRandomAccessIterator last,
             StrictWeakOrdering comp)
{
    bolt::amp::detail::sort_enqueue_int_uint(ctl, first, last, comp, true);
    return;
}

// On Linux. 'max' macro is expanded in std::max. Just rename it
#define _max(a,b)    (((a) > (b)) ? (a) : (b))
#define _min(a,b)    (((a) < (b)) ? (a) : (b))

template< typename sType, typename Container, typename StrictWeakOrdering >
unsigned int sort_lowerBoundBinary( Container& data, int left, int right, sType searchVal, StrictWeakOrdering& lessOp ) restrict(amp)
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
    //printf("lowerBoundBinary: left=%d, right=%d, firstIndex=%d\n", left, right, firstIndex);
    return firstIndex;
}

//  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and left and right index for a particular candidate value.  The comparison operator is 
//  passed as a functor parameter lessOp
//  This function returns an index that is the first index whos value would be greater than the searched value
//  If the search value is not found in the sequence, upperbound returns the same result as lowerbound
template< typename sType, typename Container, typename StrictWeakOrdering >
unsigned int  sort_upperBoundBinary( Container& data, unsigned int left, unsigned int right, sType searchVal, StrictWeakOrdering& lessOp ) restrict(amp)
{
    unsigned int upperBound = sort_lowerBoundBinary( data, left, right, searchVal, lessOp );
    
     //printf( "start of sort_upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right );
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
            //printf( "sort_upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right);
        }
    }
    //printf( "end of sort_upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right);
    return upperBound;
}


template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
typename std::enable_if<
    !(std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type, unsigned int >::value || 
      std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type, int >::value  )
                       >::type
stablesort_enqueue(control& ctrl, const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
             const StrictWeakOrdering& comp)
{

    concurrency::accelerator_view av = ctrl.getAccelerator().get_default_view();
    int vecSize = static_cast< int >( std::distance( first, last ) );

    /**********************************************************************************
     * Type Names - used in KernelTemplateSpecializer
     *********************************************************************************/
    typedef typename std::iterator_traits< DVRandomAccessIterator >::value_type iType;

    const unsigned int localRange=STABLESORT_BUFFER_SIZE;

    //  Make sure that globalRange is a multiple of localRange
    unsigned int globalRange = vecSize/**localRange*/;
    unsigned int  modlocalRange = ( globalRange & ( localRange-1 ) );
    if( modlocalRange )
    {
        globalRange &= ~modlocalRange;
        globalRange += localRange;
    }

    /**********************************************************************************
     *  Kernel 0
     *********************************************************************************/

	const unsigned int tile_limit = STABLESORT_TILE_MAX;
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
					first,
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

			tile_static iType lds[STABLESORT_BUFFER_SIZE]; 
			tile_static iType lds2[STABLESORT_BUFFER_SIZE]; 

			//  Abort threads that have passed the end of the input vector
			if (gloId < vecSize) {
				lds[ locId ] = first[ gloId ];;
			}
			//  Make a copy of the entire input array into fast local memory
			t_idx.barrier.wait();
			int end =  wgSize;
			if( (groId+1)*(wgSize) >= vecSize )
				end = vecSize - (groId*wgSize);

			unsigned int numMerges = 9;
			unsigned int pass;
			for( pass = 1; pass <= numMerges; ++pass )
			{
				int srcLogicalBlockSize = 1 << (pass-1);
				if( gloId < vecSize)
				{
  					unsigned int srcBlockNum = (locId) / srcLogicalBlockSize;
					unsigned int srcBlockIndex = (locId) % srcLogicalBlockSize;
    
					unsigned int dstLogicalBlockSize = srcLogicalBlockSize<<1;
					int leftBlockIndex = (locId)  & ~(dstLogicalBlockSize - 1 );

					leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
					leftBlockIndex = _min( leftBlockIndex, end );
					int rightBlockIndex = _min( leftBlockIndex + srcLogicalBlockSize,  end  );
    
					unsigned int insertionIndex = 0;
					if(pass%2 != 0)
					{
						if( (srcBlockNum & 0x1) == 0 )
						{
							insertionIndex = sort_lowerBoundBinary( lds, leftBlockIndex, rightBlockIndex, lds[ locId ], comp ) - leftBlockIndex;
						}
						else
						{
							insertionIndex = sort_upperBoundBinary( lds, leftBlockIndex, rightBlockIndex, lds[ locId ], comp ) - leftBlockIndex;
						}
					}
					else
					{
						if( (srcBlockNum & 0x1) == 0 )
						{
							insertionIndex = sort_lowerBoundBinary( lds2, leftBlockIndex, rightBlockIndex, lds2[ locId ], comp ) - leftBlockIndex;
						}
						else
						{
							insertionIndex = sort_upperBoundBinary( lds2, leftBlockIndex, rightBlockIndex, lds2[ locId ], comp ) - leftBlockIndex;
						} 
					}
					unsigned int dstBlockIndex = srcBlockIndex + insertionIndex;
					unsigned int dstBlockNum = srcBlockNum/2;

					if(pass%2 != 0)
					   lds2[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = lds[ locId ];
					else
					   lds[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = lds2[ locId ]; 
				}
				t_idx.barrier.wait();
			}	  

			if (gloId < vecSize) {
				first[ gloId ] = lds2[ locId ];;
			}
    
			} );

		}
		catch(std::exception &e)
		{
			std::cout << "Exception while calling bolt::amp::stablesort 1st parallel_for_each " ;
			std::cout<< e.what() << std::endl;
			throw std::exception();
		}	
		tempBuffsize = tempBuffsize - max_ext;
	}
    //  An odd number of elements requires an extra merge pass to sort
    unsigned int numMerges = 0;

    //  Calculate the log2 of vecSize, taking into account our block size from kernel 1 is 64
    //  this is how many merge passes we want
    unsigned int log2BlockSize = vecSize >> 9;
    for( ; log2BlockSize > 1; log2BlockSize >>= 1 )
    {
        ++numMerges;
    }

    //  Check to see if the input vector size is a power of 2, if not we will need last merge pass
    unsigned int vecPow2 = (vecSize & (vecSize-1));
    numMerges += vecPow2? 1: 0;

    //  Allocate a flipflop buffer because the merge passes are out of place
	concurrency::array< iType >  tmpBuffer( vecSize, av );


    /**********************************************************************************
     *  Kernel 1
     *********************************************************************************/
    
    concurrency::extent< 1 > globalSizeK1( globalRange );
    for( unsigned int pass = 1; pass <= numMerges; ++pass )
    {

        //  For each pass, the merge window doubles
        int srcLogicalBlockSize = static_cast< int >( localRange << (pass-1) );

        try
        {
        concurrency::parallel_for_each( av, globalSizeK1,
        [
            first,
            &tmpBuffer,
            vecSize,
            comp,
            localRange,
            srcLogicalBlockSize,
			pass
        ] ( concurrency::index<1> idx ) restrict(amp)
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
    leftBlockIndex = _min( leftBlockIndex, vecSize );
    int rightBlockIndex = _min( leftBlockIndex + srcLogicalBlockSize, vecSize );
    
	//  For a particular element in the input array, find the lowerbound index for it in the search sequence given by leftBlockIndex & rightBlockIndex
    // uint insertionIndex = lowerBoundLinear( source_ptr, leftBlockIndex, rightBlockIndex, source_ptr[ globalID ], lessOp ) - leftBlockIndex;
    unsigned int insertionIndex = 0;

	if( pass & 0x1 )
	{
      if( (srcBlockNum & 0x1) == 0 )
      {
        insertionIndex = sort_lowerBoundBinary( first, leftBlockIndex, rightBlockIndex, first[ gloID ], comp ) - leftBlockIndex;
      }
      else
      {
        insertionIndex = sort_upperBoundBinary( first, leftBlockIndex, rightBlockIndex, first[ gloID ], comp ) - leftBlockIndex;
      }
    
      //  The index of an element in the result sequence is the summation of it's indixes in the two input 
      //  sequences
      unsigned int dstBlockIndex = srcBlockIndex + insertionIndex;
      unsigned int dstBlockNum = srcBlockNum/2;
    
      tmpBuffer[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = first[ gloID ];
	}

	else
	{
    
	  if( (srcBlockNum & 0x1) == 0 )
      {
        insertionIndex = sort_lowerBoundBinary( tmpBuffer, leftBlockIndex, rightBlockIndex, tmpBuffer[ gloID ], comp ) - leftBlockIndex;
      }
      else
      {
        insertionIndex = sort_upperBoundBinary( tmpBuffer, leftBlockIndex, rightBlockIndex, tmpBuffer[ gloID ], comp ) - leftBlockIndex;
      }
    
      //  The index of an element in the result sequence is the summation of it's indixes in the two input 
      //  sequences
      unsigned int dstBlockIndex = srcBlockIndex + insertionIndex;
      unsigned int dstBlockNum = srcBlockNum/2;
    
      first[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = tmpBuffer[ gloID ];

	}

    } );

    }

    catch(std::exception &e)
    {
        std::cout << "Exception while calling bolt::amp::stablesort 2nd parallel_for_each " ;
        std::cout<< e.what() << std::endl;
        throw std::exception();
    }	 
       
    }
     //  If there are an odd number of merges, then the output data is sitting in the temp buffer.  We need to copy
    //  the results back into the input array
    if( numMerges & 1 )
    {
	   concurrency::extent< 1 > modified_ext( vecSize );
	   tmpBuffer.section( modified_ext ).copy_to( first.getContainer().getBuffer(first, vecSize) );
    }

    return;
}// END of stablesort_enqueue


//Non Device Vector specialization.
//This implementation creates a cl::Buffer and passes the cl buffer to the sort specialization whichtakes the
//cl buffer as a parameter.
//In the future, Each input buffer should be mapped to the device_vector and the specialization specific to
//device_vector should be called.
template< typename RandomAccessIterator, typename StrictWeakOrdering >
void stablesort_pick_iterator( control &ctl, const RandomAccessIterator& first, const RandomAccessIterator& last,
                            const StrictWeakOrdering& comp, 
                            std::random_access_iterator_tag )
{

    typedef typename std::iterator_traits< RandomAccessIterator >::value_type Type;

    int vecSize = static_cast< int >(std::distance( first, last ));
    if( vecSize < 2 )
        return;

    bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode( );

    if( runMode == bolt::amp::control::Automatic )
    {
        runMode = ctl.getDefaultPathToRun();
    }
   
    if( (runMode == bolt::amp::control::SerialCpu) || (vecSize < STABLESORT_BUFFER_SIZE) )
    {
        
        std::stable_sort( first, last, comp );
        return;
    }
    else if( runMode == bolt::amp::control::MultiCoreCpu )
    {
        #ifdef ENABLE_TBB
            bolt::btbb::stable_sort( first, last, comp );
        #else
            throw std::runtime_error("MultiCoreCPU Version of stable_sort not Enabled! \n");
        #endif

        return;
    }
    else
    {
        device_vector< Type, concurrency::array_view > dvInputOutput(  first, last, false, ctl );

        //Now call the actual AMP algorithm
        stablesort_enqueue(ctl,dvInputOutput.begin(),dvInputOutput.end(),comp);

        //Map the buffer back to the host
        dvInputOutput.data( );
        return;
    }
}

//Device Vector specialization
template< typename DVRandomAccessIterator, typename StrictWeakOrdering >
void stablesort_pick_iterator( control &ctl,
                         const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
                         const StrictWeakOrdering& comp, 
                         bolt::amp::device_vector_tag )
{

    typedef typename std::iterator_traits< DVRandomAccessIterator >::value_type Type;

    int vecSize = static_cast< int >(std::distance( first, last ));
    if( vecSize < 2 )
        return;

    bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode( );

    if( runMode == bolt::amp::control::Automatic )
    {
        runMode = ctl.getDefaultPathToRun();
    }
   
    if( runMode == bolt::amp::control::SerialCpu || (vecSize < STABLESORT_BUFFER_SIZE) )
    {
        typename bolt::amp::device_vector< Type >::pointer firstPtr =  const_cast<typename bolt::amp::device_vector< Type >::pointer>(first.getContainer( ).data( ));
        std::stable_sort( &firstPtr[ first.m_Index ], &firstPtr[ last.m_Index ], comp );
        return;
    }
    else if( runMode == bolt::amp::control::MultiCoreCpu )
    {
        #ifdef ENABLE_TBB
            typename bolt::amp::device_vector< Type >::pointer firstPtr =  const_cast<typename bolt::amp::device_vector< Type >::pointer>(first.getContainer( ).data( ));
            bolt::btbb::stable_sort( &firstPtr[ first.m_Index ], &firstPtr[ last.m_Index ], comp );
        #else
            throw std::runtime_error("MultiCoreCPU Version of stable_sort not Enabled! \n");
        #endif
        return;
    }
    else
    {
        stablesort_enqueue(ctl,first,last,comp);
    }

    return;
}
#ifdef _WIN32
//Device Vector specialization
template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
void stablesort_pick_iterator( control &ctl,
                         const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
                         const StrictWeakOrdering& comp,
                         bolt::amp::fancy_iterator_tag )
{
    static_assert(std::is_same<DVRandomAccessIterator, bolt::amp::fancy_iterator_tag  >::value , "It is not possible to sort fancy iterators. They are not mutable" );
}
#endif



template<typename RandomAccessIterator, typename StrictWeakOrdering>
void stablesort_detect_random_access( control &ctl,
                                const RandomAccessIterator& first, const RandomAccessIterator& last,
                                const StrictWeakOrdering& comp, 
                                std::random_access_iterator_tag )
{
    return stablesort_pick_iterator(ctl, first, last,
                              comp,
                              typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
};
#ifdef _WIN32
// Wrapper that uses default control class, iterator interface
template<typename RandomAccessIterator, typename StrictWeakOrdering>
void stablesort_detect_random_access( control &ctl,
                                const RandomAccessIterator& first, const RandomAccessIterator& last,
                                const StrictWeakOrdering& comp,
                                std::input_iterator_tag )
{
    //  \TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
    //  to a temporary buffer.  Should we?
    static_assert( std::is_same< RandomAccessIterator, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
};
#endif


}//namespace bolt::cl::detail


    template<typename RandomAccessIterator>
    void stable_sort(RandomAccessIterator first,
              RandomAccessIterator last)
    {
        typedef typename std::iterator_traits< RandomAccessIterator >::value_type T;

        detail::stablesort_detect_random_access( control::getDefault( ),
                                           first, last,
                                           less< T >( ),
                                           typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
        return;
    }

    template<typename RandomAccessIterator, typename StrictWeakOrdering>
    void stable_sort(RandomAccessIterator first,
              RandomAccessIterator last,
              StrictWeakOrdering comp)
    {
        detail::stablesort_detect_random_access( control::getDefault( ),
                                           first, last,
                                           comp, 
                                           typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
        return;
    }

    template<typename RandomAccessIterator>
    void stable_sort(control &ctl,
              RandomAccessIterator first,
              RandomAccessIterator last)
    {
        typedef typename std::iterator_traits< RandomAccessIterator >::value_type T;

        detail::stablesort_detect_random_access(ctl,
                                          first, last,
                                          less< T >( ), 
                                          typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
        return;
    }

    template<typename RandomAccessIterator, typename StrictWeakOrdering>
    void stable_sort(control &ctl,
              RandomAccessIterator first,
              RandomAccessIterator last,
              StrictWeakOrdering comp)
    {
        detail::stablesort_detect_random_access(ctl,
                                          first, last,
                                          comp, 
                                          typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
        return;
    }

}//namespace bolt::amp
}//namespace bolt

#endif
