#pragma once

namespace details {

// FIXME: stablesort algorithm implementation breaks in Clang 4.0 as of now
#if 0

#define STABLESORT_BUFFER_SIZE 512
#define STABLESORT_TILE_MAX 65535

// template<typename InputIt, typename Compare>
// void sort_enqueue_int_uint(InputIt &first, InputIt &last, Compare comp, bool int_flag);

template< typename InputIt, typename Compare >
typename std::enable_if< std::is_same< typename std::iterator_traits<InputIt >::value_type,
                                       unsigned int
                                     >::value
                       >::type  /*If enabled then this typename will be evaluated to void*/
stablesort_dispatch(InputIt first, InputIt last, Compare comp)
{
    sort_enqueue_int_uint(first, last, comp, false);
    return;
}

template<typename InputIt, typename Compare>
typename std::enable_if< std::is_same< typename std::iterator_traits<InputIt >::value_type,
                                       int
                                     >::value
                       >::type  /*If enabled then this typename will be evaluated to void*/
stablesort_dispatch(InputIt first, InputIt last, Compare comp)
{
    sort_enqueue_int_uint(first, last, comp, true);
    return;
}

// On Linux. 'max' macro is expanded in std::max. Just rename it
#define _max(a,b)    (((a) > (b)) ? (a) : (b))
#define _min(a,b)    (((a) < (b)) ? (a) : (b))

template< typename sType, typename Container, typename Compare >
unsigned int sort_lowerBoundBinary( Container& data, int left, int right, sType searchVal, Compare& lessOp ) [[hc]]
{
    int firstIndex = left;
    int lastIndex = right;
    
    while( firstIndex < lastIndex )
    {
        unsigned int midIndex = ( firstIndex + lastIndex ) / 2;
        sType midValue = data[ midIndex ];
        
        if( lessOp( midValue, searchVal ) )
            firstIndex = midIndex+1;
        else
            lastIndex = midIndex;
    }
    return firstIndex;
}

template< typename sType, typename Container, typename Compare >
unsigned int  sort_upperBoundBinary( Container& data, unsigned int left, unsigned int right, sType searchVal, Compare& lessOp ) [[hc]]
{
    unsigned int upperBound = sort_lowerBoundBinary( data, left, right, searchVal, lessOp );
    
    if( upperBound != right )
    {
        int mid = 0;
        sType upperValue = data[ upperBound ];
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
        }
    }
    return upperBound;
}


template<typename InputIt, typename Compare>
typename std::enable_if<
    !(std::is_same< typename std::iterator_traits<InputIt >::value_type, unsigned int >::value || 
      std::is_same< typename std::iterator_traits<InputIt >::value_type, int >::value  )
                       >::type
stablesort_dispatch(const InputIt& first, const InputIt& last, const Compare& comp)
{

    int vecSize = static_cast< int >( std::distance( first, last ) );

    typedef typename std::iterator_traits< InputIt >::value_type iType;

    const unsigned int localRange=STABLESORT_BUFFER_SIZE;

    unsigned int globalRange = vecSize/**localRange*/;
    unsigned int  modlocalRange = ( globalRange & ( localRange-1 ) );
    if( modlocalRange )
    {
        globalRange &= ~modlocalRange;
        globalRange += localRange;
    }

	const unsigned int tile_limit = STABLESORT_TILE_MAX;
	const unsigned int max_ext = (tile_limit*localRange);

	unsigned int	   tempBuffsize = globalRange; 
	unsigned int	   iteration = (globalRange-1)/max_ext; 

    auto f_ = utils::get_pointer(first);
    hc::array_view<iType> first_(hc::extent<1>(vecSize), f_);
    for(unsigned int i=0; i<=iteration; i++)
	{
	    unsigned int extent_sz =  (tempBuffsize > max_ext) ? max_ext : tempBuffsize; 
		unsigned int index = i*(tile_limit*localRange);
		unsigned int tile_index = i*tile_limit;

        kernel_launch( extent_sz,
				[
					first_,
					vecSize,
					comp,
					index,
					tile_index,
					localRange
				] ( hc::tiled_index< 1 > t_idx ) [[hc]]
		  {

			int gloId = t_idx.global[ 0 ] + index;
			int groId = t_idx.tile[ 0 ] + tile_index;
			unsigned int locId = t_idx.local[ 0 ];
			int wgSize = localRange;

			tile_static iType lds[STABLESORT_BUFFER_SIZE]; 
			tile_static iType lds2[STABLESORT_BUFFER_SIZE]; 

			if (gloId < vecSize)
				lds[ locId ] = first_[ gloId ];;
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
				first_[ gloId ] = lds2[ locId ];;
			}
    
			}, localRange );

		tempBuffsize = tempBuffsize - max_ext;
	}
    unsigned int numMerges = 0;

    unsigned int log2BlockSize = vecSize >> 9;
    for( ; log2BlockSize > 1; log2BlockSize >>= 1 )
    {
        ++numMerges;
    }

    unsigned int vecPow2 = (vecSize & (vecSize-1));
    numMerges += vecPow2? 1: 0;

    std::vector<iType> tmp(vecSize);
    hc::array_view<iType> tmpBuffer(hc::extent<1>(vecSize), tmp);
    tmpBuffer.discard_data();


    hc::extent< 1 > globalSizeK1( globalRange );
    for( unsigned int pass = 1; pass <= numMerges; ++pass )
    {

        //  For each pass, the merge window doubles
        int srcLogicalBlockSize = static_cast< int >( localRange << (pass-1) );

        kernel_launch( globalRange,
        [
            first_,
            tmpBuffer,
            vecSize,
            comp,
            localRange,
            srcLogicalBlockSize,
			pass
        ] ( hc::index<1> idx ) [[hc]]
   {

    int gloID = idx[ 0 ];

    if( gloID >= vecSize )
        return;

    unsigned int srcBlockNum = gloID / srcLogicalBlockSize;
    unsigned int srcBlockIndex = gloID % srcLogicalBlockSize;
    
    unsigned int dstLogicalBlockSize = srcLogicalBlockSize<<1;
    int leftBlockIndex = gloID & ~(dstLogicalBlockSize - 1 );
    leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
    leftBlockIndex = _min( leftBlockIndex, vecSize );
    int rightBlockIndex = _min( leftBlockIndex + srcLogicalBlockSize, vecSize );
    
    unsigned int insertionIndex = 0;

	if( pass & 0x1 )
	{
      if( (srcBlockNum & 0x1) == 0 )
      {
        insertionIndex = sort_lowerBoundBinary( first_, leftBlockIndex, rightBlockIndex, first_[ gloID ], comp ) - leftBlockIndex;
      }
      else
      {
        insertionIndex = sort_upperBoundBinary( first_, leftBlockIndex, rightBlockIndex, first_[ gloID ], comp ) - leftBlockIndex;
      }
    
      //  The index of an element in the result sequence is the summation of it's indixes in the two input 
      //  sequences
      unsigned int dstBlockIndex = srcBlockIndex + insertionIndex;
      unsigned int dstBlockNum = srcBlockNum/2;
    
      tmpBuffer[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = first_[ gloID ];
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
    
      first_[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = tmpBuffer[ gloID ];

	}

    } );

    }
    if( numMerges & 1 )
    {
       tmpBuffer.synchronize();
       std::copy(std::begin(tmp), std::end(tmp), first);
    }

    return;
}
#endif

template<class InputIt, class Compare>
void stablesort_impl(InputIt first, InputIt last, Compare comp, std::input_iterator_tag) {
    std::stable_sort(first, last, comp);
}
 

template<class InputIt, class Compare>
void stablesort_impl(InputIt first, InputIt last, Compare comp,
               std::random_access_iterator_tag) {
  unsigned N = std::distance(first, last);
  if (N == 0)
      return;

// FIXME: stablesort algorithm implementation breaks Clang 4.0 as of now
#if 0
  // call to std::sort when small data size
  if (N <= details::PARALLELIZE_THRESHOLD) {
      std::stable_sort(first, last, comp);
  }
  stablesort_dispatch(first, last, comp);
#endif

  std::stable_sort(first, last, comp);
}

} // namespace details
