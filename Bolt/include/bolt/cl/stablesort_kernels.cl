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

//  Good web references to read about stable sorting:
//  Parallel Merge Sort: http://www.drdobbs.com/parallel/parallel-merge-sort/229400239
//  Parallel Merge: http://www.drdobbs.com/parallel/parallel-merge/229204454

//  Good white papers to read about stable sorting:
//  Efficient parallel merge sort for fixed and variable length keys: 
//  http://www.idav.ucdavis.edu/publications/print_pub?pub_id=1085

//  Designing Efficient sorting algorithms for ManyCore GPUs: 
//  http://www.drdobbs.com/parallel/parallel-merge/229204454

// #pragma OPENCL EXTENSION cl_amd_printf : enable


template< typename sType, typename StrictWeakOrdering >
uint lowerBoundBinarylocal( local sType* data, uint left, uint right, sType searchVal, global StrictWeakOrdering* lessOp )
{
    //  The values firstIndex and lastIndex get modified within the loop, narrowing down the potential sequence
    uint firstIndex = left;
    uint lastIndex = right;
    
    //  This loops through [firstIndex, lastIndex)
    //  Since firstIndex and lastIndex will be different for every thread depending on the nested branch,
    //  this while loop will be divergent within a wavefront
    while( firstIndex < lastIndex )
    {
        //  midIndex is the average of first and last, rounded down
        uint midIndex = ( firstIndex + lastIndex ) / 2;
        sType midValue = data[ midIndex ];
        
        //  This branch will create divergent wavefronts
        if( (*lessOp)( midValue, searchVal ) )
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
template< typename sType, typename StrictWeakOrdering >
uint upperBoundBinarylocal( local sType* data, uint left, uint right, sType searchVal, global StrictWeakOrdering* lessOp )
{
    uint upperBound = lowerBoundBinarylocal( data, left, right, searchVal, lessOp );
    
     //printf( "start of upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right );
    //  upperBound is always between left and right or equal to right
    //  If upperBound == right, then  searchVal was not found in the sequence.  Just return.
    if( upperBound != right )
    {
        //  While the values are equal i.e. !(x < y) && !(y < x) increment the index
        uint mid = 0;
        sType upperValue = data[ upperBound ];
        //This loop is a kind of a specialized binary search. 
        //This will find the first index location which is not equal to searchVal.
        while( !(*lessOp)( upperValue, searchVal ) && !(*lessOp)( searchVal, upperValue) && (upperBound < right))
        {
            mid = (upperBound + right)/2;
            sType midValue = data[mid];
            if( !(*lessOp)( midValue, searchVal ) && !(*lessOp)( searchVal, midValue) )
            {
                upperBound = mid + 1;
            }   
            else
            {
                right = mid;
                upperBound++;
            }
            upperValue = data[ upperBound ];
            //printf( "upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right);
        }
    }
    //printf( "end of upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right);
    return upperBound;
}



//  This implements a linear search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and a left and right index, with a  candidate value.  The comparison operator is 
//  passed as a functor parameter lessOp
//  This function returns an index that would be the appropriate index to use to insert the value
template< typename sType, typename StrictWeakOrdering >
uint lowerBoundLinear( global sType* data, uint left, uint right, sType searchVal, global StrictWeakOrdering* lessOp )
{
    //  The values firstIndex and lastIndex get modified within the loop, narrowing down the potential sequence
    uint firstIndex = left;
    uint lastIndex = right;
    
    //  This loops through [firstIndex, lastIndex)
    //  Since firstIndex and lastIndex will be different for every thread depending on the nested branch,
    //  this while loop will be divergent within a wavefront
    while( firstIndex < lastIndex )
    {
        sType dataVal = data[ firstIndex ];
        
        //  This branch will create divergent wavefronts
        if( (*lessOp)( dataVal, searchVal ) )
        {
            firstIndex = firstIndex+1;
        }
        else
        {
            break;
        }
    }

    return firstIndex;
}

//  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and left and right index for a particular candidate value.  The comparison operator is 
//  passed as a functor parameter lessOp
//  This function returns an index that is the first index whos value would be equal to the searched value
template< typename dIterType, typename dPtrType, typename sType, typename StrictWeakOrdering >
uint lowerBoundBinary( dIterType source_iter, dPtrType *source_ptr, uint left, uint right, sType searchVal, global StrictWeakOrdering* lessOp )
{
    //  The values firstIndex and lastIndex get modified within the loop, narrowing down the potential sequence
    uint firstIndex = left;
    uint lastIndex = right;
    source_iter.init(source_ptr);
    //  This loops through [firstIndex, lastIndex)
    //  Since firstIndex and lastIndex will be different for every thread depending on the nested branch,
    //  this while loop will be divergent within a wavefront
    while( firstIndex < lastIndex )
    {
        //  midIndex is the average of first and last, rounded down
        uint midIndex = ( firstIndex + lastIndex ) / 2;
        sType midValue = source_iter[ midIndex ];
        
        //  This branch will create divergent wavefronts
        if( (*lessOp)( midValue, searchVal ) )
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
template< typename dIterType, typename dPtrType, typename sType, typename StrictWeakOrdering >
uint upperBoundBinary( dIterType source_iter, dPtrType *source_ptr, uint left, uint right, sType searchVal, global StrictWeakOrdering* lessOp )
{
    uint upperBound = lowerBoundBinary( source_iter, source_ptr, left, right, searchVal, lessOp );
    
	source_iter.init(source_ptr);
     //printf( "start of upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right );
    //  upperBound is always between left and right or equal to right
    //  If upperBound == right, then  searchVal was not found in the sequence.  Just return.
    if( upperBound != right )
    {
        //  While the values are equal i.e. !(x < y) && !(y < x) increment the index
        uint mid = 0;
        sType upperValue = source_iter[ upperBound ];
        //This loop is a kind of a specialized binary search. 
        //This will find the first index location which is not equal to searchVal.
        while( !(*lessOp)( upperValue, searchVal ) && !(*lessOp)( searchVal, upperValue) && (upperBound < right))
        {
            mid = (upperBound + right)/2;
            sType midValue = source_iter[mid];
            if( !(*lessOp)( midValue, searchVal ) && !(*lessOp)( searchVal, midValue) )
            {
                upperBound = mid + 1;
            }   
            else
            {
                right = mid;
                upperBound++;
            }
            upperValue = source_iter[ upperBound ];
            //printf( "upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right);
        }
    }
    //printf( "end of upperBoundBinary: upperBound, left, right = [%d, %d, %d]\n", upperBound, left, right);
    return upperBound;
}

//  This kernel implements merging of blocks of sorted data.  The input to this kernel most likely is
//  the output of blockInsertionSortTemplate.  It is expected that the source array contains multiple
//  blocks, each block is independently sorted.  The goal is to write into the output buffer half as 
//  many blocks, of double the size.  The even and odd blocks are stably merged together to form
//  a new sorted block of twice the size.  The algorithm is out-of-place.
template< typename sPtrType, typename dIterType, typename StrictWeakOrdering >
kernel void mergeTemplate( 
                global sPtrType* source_ptr,
                dIterType    source_iter, 
                global sPtrType* result_ptr,
                dIterType    result_iter, 
                const uint srcVecSize,
                const uint srcLogicalBlockSize,
                local sPtrType* lds,
                global StrictWeakOrdering* lessOp
            )
{
    size_t globalID     = get_global_id( 0 );
    size_t groupID      = get_group_id( 0 );
    size_t localID      = get_local_id( 0 );
    size_t wgSize       = get_local_size( 0 );

	source_iter.init(source_ptr);
	result_iter.init(result_ptr);


    //  Abort threads that are passed the end of the input vector
    if( globalID >= srcVecSize )
        return; // on SI this doesn't mess-up barriers

    //  For an element in sequence A, find the lowerbound index for it in sequence B
    uint srcBlockNum = globalID / srcLogicalBlockSize;
    uint srcBlockIndex = globalID % srcLogicalBlockSize;

    //printf( "mergeTemplate: srcBlockNum[%i]=%i\n", srcBlockNum, srcBlockIndex );
    //  Pairs of even-odd blocks will be merged together 
    //  An even block should search for an insertion point in the next odd block, 
    //  and the odd block should look for an insertion point in the corresponding previous even block

    uint dstLogicalBlockSize = srcLogicalBlockSize<<1;
    uint leftBlockIndex = globalID & ~(dstLogicalBlockSize - 1 );

    //printf("mergeTemplate: leftBlockIndex=%d\n", leftBlockIndex );
    leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
    leftBlockIndex = min( leftBlockIndex, srcVecSize );
    uint rightBlockIndex = min( leftBlockIndex + srcLogicalBlockSize, srcVecSize );
   
    //  For a particular element in the input array, find the lowerbound index for it in the search sequence given by leftBlockIndex & rightBlockIndex
    // uint insertionIndex = lowerBoundLinear( source_iter, source_ptr, leftBlockIndex, rightBlockIndex, search_val, lessOp ) - leftBlockIndex;
    uint insertionIndex = 0;
	sPtrType search_val = source_iter[ globalID ];
    if( (srcBlockNum & 0x1) == 0 )
    {
        insertionIndex = lowerBoundBinary( source_iter, source_ptr, leftBlockIndex, rightBlockIndex, search_val, lessOp ) - leftBlockIndex;
    }
    else
    {
        insertionIndex = upperBoundBinary( source_iter, source_ptr, leftBlockIndex, rightBlockIndex, search_val, lessOp ) - leftBlockIndex;
    }
    
    //  The index of an element in the result sequence is the summation of it's indixes in the two input 
    //  sequences
    uint dstBlockIndex = srcBlockIndex + insertionIndex;
    uint dstBlockNum = srcBlockNum/2;
    
    result_iter[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = source_iter[ globalID ];

	//sPtrType val = source_iter[ globalID ];
//	if(val <= 0.0f)
	//	printf("%d %d %f\n", globalID, groupID, val);

}

template< typename dPtrType, typename dIterType, typename StrictWeakOrdering >
kernel void LocalMergeSortTemplate( 
                global dPtrType* data_ptr,
                dIterType    data_iter, 
                const uint vecSize,
                local dPtrType* lds,
				local dPtrType* lds2,
                global StrictWeakOrdering* lessOp
            )
{
    size_t gloId    = get_global_id( 0 );
    size_t groId    = get_group_id( 0 );
    size_t locId    = get_local_id( 0 );
    size_t wgSize   = get_local_size( 0 );


    data_iter.init( data_ptr );

    //  Make a copy of the entire input array into fast local memory
	dPtrType val;
	if( gloId < vecSize)
	{
	      val = data_iter[ gloId ];
		  lds[locId ] = val;
	}
    barrier( CLK_LOCAL_MEM_FENCE );
	uint end =  wgSize;
	if( (groId+1)*(wgSize) >= vecSize )
		end = vecSize - (groId*wgSize);

	uint numMerges = 8;
	uint pass;
	for( pass = 1; pass <= numMerges; ++pass )
	{
		uint srcLogicalBlockSize = 1 << (pass-1);
	    if( gloId < vecSize)
		{
  		    uint srcBlockNum = (locId) / srcLogicalBlockSize;
			uint srcBlockIndex = (locId) % srcLogicalBlockSize;
    
			uint dstLogicalBlockSize = srcLogicalBlockSize<<1;
			uint leftBlockIndex = (locId)  & ~(dstLogicalBlockSize - 1 );

		    leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
			leftBlockIndex = min( leftBlockIndex, end );
			uint rightBlockIndex = min( leftBlockIndex + srcLogicalBlockSize,  end  );
    
			uint insertionIndex = 0;
			if(pass%2 != 0)
			{
				if( (srcBlockNum & 0x1) == 0 )
				{
					insertionIndex = lowerBoundBinarylocal( lds, leftBlockIndex, rightBlockIndex, lds[ locId ], lessOp ) - leftBlockIndex;
				}
				else
				{
					insertionIndex = upperBoundBinarylocal( lds, leftBlockIndex, rightBlockIndex, lds[ locId ], lessOp ) - leftBlockIndex;
				}
			}
			else
			{
				if( (srcBlockNum & 0x1) == 0 )
				{
					insertionIndex = lowerBoundBinarylocal( lds2, leftBlockIndex, rightBlockIndex, lds2[ locId ], lessOp ) - leftBlockIndex;
				}
				else
				{
					insertionIndex = upperBoundBinarylocal( lds2, leftBlockIndex, rightBlockIndex, lds2[ locId ], lessOp ) - leftBlockIndex;
				} 
			}
			uint dstBlockIndex = srcBlockIndex + insertionIndex;
			uint dstBlockNum = srcBlockNum/2;
			if(pass%2 != 0)
			   lds2[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = lds[ locId ];
			else
			   lds[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = lds2[ locId ]; 
		}
        barrier( CLK_LOCAL_MEM_FENCE );
	}	  
	if( gloId < vecSize)
	{
	    val = lds[ locId ];
	    data_iter[ gloId ] = val;
	}
}
