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
            // printf( "lowerBound: lastIndex[ %i ]=%i\n", get_local_id( 0 ), lastIndex );
        }
        else
        {
            lastIndex = midIndex;
            // printf( "lowerBound: firstIndex[ %i ]=%i\n", get_local_id( 0 ), firstIndex );
        }
    }
    
    return firstIndex;
}

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
//  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and left and right index for a particular candidate value.  The comparison operator is 
//  passed as a functor parameter lessOp
//  This function returns an index that is the first index whos value would be equal to the searched value
template< typename keyIterType, typename keyType, typename sType, typename StrictWeakOrdering >
uint lowerBoundBinary( keyIterType keyIter, keyType *keyptr, uint left, uint right, sType searchVal, global StrictWeakOrdering* lessOp )
{
    //  The values firstIndex and lastIndex get modified within the loop, narrowing down the potential sequence
    uint firstIndex = left;
    uint lastIndex = right;
    keyIter.init( keyptr ); 
    //  This loops through [firstIndex, lastIndex)
    //  Since firstIndex and lastIndex will be different for every thread depending on the nested branch,
    //  this while loop will be divergent within a wavefront
    while( firstIndex < lastIndex )
    {
        //  midIndex is the average of first and last, rounded down
        uint midIndex = ( firstIndex + lastIndex ) / 2;
        sType midValue = keyIter[ midIndex ];
        
        //  This branch will create divergent wavefronts
        if( (*lessOp)( midValue, searchVal ) )
        {
            firstIndex = midIndex+1;
            // printf( "lowerBound: lastIndex[ %i ]=%i\n", get_local_id( 0 ), lastIndex );
        }
        else
        {
            lastIndex = midIndex;
            // printf( "lowerBound: firstIndex[ %i ]=%i\n", get_local_id( 0 ), firstIndex );
        }
    }
    
    return firstIndex;
}

//  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and left and right index for a particular candidate value.  The comparison operator is 
//  passed as a functor parameter lessOp
//  This function returns an index that is the first index whos value would be greater than the searched value
//  If the search value is not found in the sequence, upperbound returns the same result as lowerbound
template< typename keyIterType, typename keyType, typename sType, typename StrictWeakOrdering >
uint upperBoundBinary( keyIterType keyIter, keyType *keyptr, uint left, uint right, sType searchVal, global StrictWeakOrdering* lessOp )
{

    uint upperBound = lowerBoundBinary( keyIter, keyptr, left, right, searchVal, lessOp );

     keyIter.init( keyptr );    
    //  upperBound is always between left and right or equal to right
    //  If upperBound == right, then  searchVal was not found in the sequence.  Just return.
    if( upperBound != right )
    {
        //  While the values are equal i.e. !(x < y) && !(y < x) increment the index
        uint mid = 0;
        sType upperValue = keyIter[ upperBound ];
        //This loop is a kind of a specialized binary search. 
        //This will find the first index location which is not equal to searchVal.
        while( !(*lessOp)( upperValue, searchVal ) && !(*lessOp)( searchVal, upperValue) && (upperBound < right))
        {
            mid = (upperBound + right)/2;
            sType midValue = keyIter[mid];
            if( !(*lessOp)( midValue, searchVal ) && !(*lessOp)( searchVal, midValue) )
            {
                upperBound = mid + 1;
            }   
            else
            {
                right = mid;
                upperBound++;
            }
            upperValue = keyIter[ upperBound ];
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
template< typename keyType, typename keyIterType, typename valueType, typename valueIterType, 
            typename StrictWeakOrdering >
kernel void mergeTemplate( 
                global keyType*     iKey_ptr,
                keyIterType         iKey_iter, 
                global valueType*   iValue_ptr,
                valueIterType       iValue_iter, 
                global keyType*     oKey_ptr,
                keyIterType         oKey_iter, 
                global valueType*   oValue_ptr,
                valueIterType       oValue_iter, 
                const uint srcVecSize,
                const uint srcLogicalBlockSize,
                local keyType*      key_lds,
                local valueType*    val_lds,
                global StrictWeakOrdering* lessOp
            )
{
    size_t globalID     = get_global_id( 0 );
    size_t groupID      = get_group_id( 0 );
    size_t localID      = get_local_id( 0 );
    size_t wgSize       = get_local_size( 0 );


	 iKey_iter.init( iKey_ptr );
     iValue_iter.init( iValue_ptr );

	 oKey_iter.init( oKey_ptr );
     oValue_iter.init( oValue_ptr );

    //  Abort threads that are passed the end of the input vector
    if( globalID >= srcVecSize )
        return; // on SI this doesn't mess-up barriers

    //  For an element in sequence A, find the lowerbound index for it in sequence B
    uint srcBlockNum = globalID / srcLogicalBlockSize;
    uint srcBlockIndex = globalID % srcLogicalBlockSize;
    
    // printf( "mergeTemplate: srcBlockNum[%i]=%i\n", srcBlockNum, srcBlockIndex );

    //  Pairs of even-odd blocks will be merged together 
    //  An even block should search for an insertion point in the next odd block, 
    //  and the odd block should look for an insertion point in the corresponding previous even block
    uint dstLogicalBlockSize = srcLogicalBlockSize<<1;
    uint leftBlockIndex = globalID & ~((dstLogicalBlockSize) - 1 );
    leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
    leftBlockIndex = min( leftBlockIndex, srcVecSize );
    uint rightBlockIndex = min( leftBlockIndex + srcLogicalBlockSize, srcVecSize );
    
    //  For a particular element in the input array, find the lowerbound index for it in the search sequence given by leftBlockIndex & rightBlockIndex
    // uint insertionIndex = lowerBoundLinear( iKey_iter, leftBlockIndex, rightBlockIndex, iKey_iter[ globalID ], lessOp ) - leftBlockIndex;
    uint insertionIndex = 0;
	keyType searchKey = iKey_iter[ globalID ];
    if( (srcBlockNum & 0x1) == 0 )
    {
        insertionIndex = lowerBoundBinary( iKey_iter, iKey_ptr, leftBlockIndex, rightBlockIndex, searchKey, lessOp ) - leftBlockIndex;
    }
    else
    {
        insertionIndex = upperBoundBinary( iKey_iter, iKey_ptr, leftBlockIndex, rightBlockIndex, searchKey, lessOp ) - leftBlockIndex;
    }
    
    //  The index of an element in the result sequence is the summation of it's indixes in the two input 
    //  sequences
    uint dstBlockIndex = srcBlockIndex + insertionIndex;
    uint dstBlockNum = srcBlockNum/2;
    
    oKey_iter[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = iKey_iter[ globalID ];
    oValue_iter[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = iValue_iter[ globalID ];
}

template< typename keyType, typename keyIterType, typename valueType, typename valueIterType, 
            typename StrictWeakOrdering >
kernel void LocalMergeSortTemplate( 
                global keyType*     key_ptr,
                keyIterType         key_iter, 
                global valueType*   value_ptr,
                valueIterType       value_iter, 
                const uint          vecSize,
                local keyType*      key_lds,
				local keyType*      key_lds2,
                local valueType*    val_lds,
				local valueType*    val_lds2,
                global StrictWeakOrdering* lessOp
            )
{

    size_t gloId    = get_global_id( 0 );
    size_t groId    = get_group_id( 0 );
    size_t locId    = get_local_id( 0 );
    size_t wgSize   = get_local_size( 0 );


    key_iter.init( key_ptr );
    value_iter.init( value_ptr );

    //  Make a copy of the entire input array into fast local memory
	keyType key; 
	valueType val; 
	if( gloId < vecSize)
	{
	      key = key_iter[ gloId ];
		  val = value_iter[ gloId ];
		  key_lds[ locId ] = key;
		  val_lds[ locId ] = val;
	}
    barrier( CLK_LOCAL_MEM_FENCE );
	uint end =  wgSize;
	if( (groId+1)*(wgSize) >= vecSize )
	{
		end = vecSize - (groId*wgSize);
	}

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
					insertionIndex = lowerBoundBinarylocal( key_lds, leftBlockIndex, rightBlockIndex, key_lds[ locId ], lessOp ) - leftBlockIndex;
				}
				else
				{
					insertionIndex = upperBoundBinarylocal( key_lds, leftBlockIndex, rightBlockIndex, key_lds[ locId ], lessOp ) - leftBlockIndex;
				}
			}
			else
			{
				if( (srcBlockNum & 0x1) == 0 )
				{
					insertionIndex = lowerBoundBinarylocal( key_lds2, leftBlockIndex, rightBlockIndex, key_lds2[ locId ], lessOp ) - leftBlockIndex;
				}
				else
				{
					insertionIndex = upperBoundBinarylocal( key_lds2, leftBlockIndex, rightBlockIndex, key_lds2[ locId ], lessOp ) - leftBlockIndex;
				} 
			}
			uint dstBlockIndex = srcBlockIndex + insertionIndex;
			uint dstBlockNum = srcBlockNum/2;
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
        barrier( CLK_LOCAL_MEM_FENCE );
	}	  
	if( gloId < vecSize)
	{
		key = key_lds[ locId ];
		val = val_lds[ locId ];
		key_iter[ gloId ] = key;
        value_iter[ gloId ] = val;

	}

    
}
