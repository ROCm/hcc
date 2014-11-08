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
#if !defined( BOLT_AMP_SCAN_BY_KEY_INL )
#define BOLT_AMP_SCAN_BY_KEY_INL

#define SCANBYKEY_KERNELWAVES 4
#define SCANBYKEY_WAVESIZE 64
#define SCANBYKEY_TILE_MAX 65535

#include "bolt/amp/iterator/iterator_traits.h"
#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/scan_by_key.h"
#endif
#ifdef BOLT_ENABLE_PROFILING
#include "bolt/AsyncProfiler.h"
//AsyncProfiler aProfiler("transform_scan");
#endif


#if 0
#define NUM_PEEK 128
#define PEEK_AT( ... ) \
    { \
        unsigned int numPeek = NUM_PEEK; \
        numPeek = (__VA_ARGS__.get_extent().size() < numPeek) ? __VA_ARGS__.get_extent().size() : numPeek; \
        std::vector< oType > hostMem( numPeek ); \
        concurrency::array_view< oType > peekOutput( static_cast< int >( numPeek ), (oType *)&hostMem.begin()[ 0 ] ); \
        __VA_ARGS__.section( Concurrency::extent< 1 >( numPeek ) ).copy_to( peekOutput ); \
        for ( unsigned int i = 0; i < numPeek; i++) \
        { \
            std::cout << #__VA_ARGS__ << "[ " << i << " ] = " << peekOutput[ i ] << std::endl; \
        } \
    }
#else
#define PEEK_AT( ... )
#endif



namespace bolt
{
namespace amp
{

namespace detail
{
/*!
*   \internal
*   \addtogroup detail
*   \ingroup scan
*   \{
*/

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
Serial_inclusive_scan_by_key(
    InputIterator1 firstKey,
    InputIterator2 values,
    OutputIterator result,
    unsigned int  num,
    const BinaryPredicate binary_pred,
    const BinaryFunction binary_op)
{

    if ( num < 1 ) return result;

    // zeroth element
    *result = *values;

    // scan oneth element and beyond
    for ( unsigned int i = 1; i < num;  i++ )
    {
        if ( binary_pred( *( firstKey + i ), *( firstKey + ( i - 1 ) ) ) )
        {
            *( result + i ) = binary_op( *( result + ( i - 1 ) ), *( values + i ) );
        }
        else
        {
            *( result + i ) = *( values + i );
        }
    }

    return result;
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate,
    typename BinaryFunction,
    typename T>
OutputIterator
Serial_exclusive_scan_by_key(
    InputIterator1 firstKey,
    InputIterator2 values,
    OutputIterator result,
    unsigned int  num,
    const BinaryPredicate binary_pred,
    const BinaryFunction binary_op,
    const T &init)
{
    typedef typename std::iterator_traits< OutputIterator >::value_type oType;
    // do zeroeth element

    oType temp = *values;
    *result = init;
    // scan oneth element and beyond
    for ( unsigned int i= 1; i<num; i++)
    {
        // load value
        oType currentValue = temp; // convertible
        oType previousValue = result [ i - 1];

        // within segment
        if (binary_pred(firstKey [ i ],firstKey [ i - 1 ]))
        {
            temp = values [ i ];
            oType r = binary_op( previousValue, currentValue);
            result [ i ] = r;
        }
        else // new segment
        {
             temp = values [ i ];
             result [ i ] = init;
        }
    }

    return result;
}


//  All calls to scan_by_key end up here, unless an exception was thrown
//  This is the function that sets up the kernels to compile (once only) and execute
template<
    typename DVInputIterator1,
    typename DVInputIterator2,
    typename DVOutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction >
void
scan_by_key_enqueue(
    control& ctl,
    const DVInputIterator1& firstKey,
    const DVInputIterator1& lastKey,
    const DVInputIterator2& firstValue,
    const DVOutputIterator& result,
    const T& init,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_funct,
    const std::string& user_code,
    const bool& inclusive )
{
    concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();

    /**********************************************************************************
     * Type Names - used in KernelTemplateSpecializer
     *********************************************************************************/
    typedef typename std::iterator_traits< DVInputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< DVInputIterator2 >::value_type vType;
  	typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

    int exclusive = inclusive ? 0 : 1;

    int numElements = static_cast< int >( std::distance( firstKey, lastKey ) );
    const unsigned int kernel0_WgSize = SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES;
    const unsigned int kernel1_WgSize = SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES;
    const unsigned int kernel2_WgSize = SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES;

    //  Ceiling function to bump the size of input to the next whole wavefront size
    unsigned int sizeInputBuff = numElements;
    unsigned int modWgSize = (sizeInputBuff & ((kernel0_WgSize*2)-1));
    if( modWgSize )
    {
        sizeInputBuff &= ~modWgSize;
        sizeInputBuff += (kernel0_WgSize*2);
    }
    int numWorkGroupsK0 = static_cast< int >( sizeInputBuff / (kernel0_WgSize*2) );
    //  Ceiling function to bump the size of the sum array to the next whole wavefront size
    unsigned int sizeScanBuff = numWorkGroupsK0;
    modWgSize = (sizeScanBuff & ((kernel0_WgSize*2)-1));
    if( modWgSize )
    {
        sizeScanBuff &= ~modWgSize;
        sizeScanBuff += (kernel0_WgSize*2);
    }

  	concurrency::array< kType >  keySumArray( sizeScanBuff, av );
    concurrency::array< vType >  preSumArray( sizeScanBuff, av );
    concurrency::array< vType >  preSumArray1( sizeScanBuff, av );


    /**********************************************************************************
     *  Kernel 0
     *********************************************************************************/
  //	Loop to calculate the inclusive scan of each individual tile, and output the block sums of every tile
  //	This loop is inherently parallel; every tile is independant with potentially many wavefronts

	const unsigned int tile_limit = SCANBYKEY_TILE_MAX;
	const unsigned int max_ext = (tile_limit*kernel0_WgSize);
	unsigned int	   tempBuffsize = (sizeInputBuff/2); 
	unsigned int	   iteration = (tempBuffsize-1)/max_ext; 

    for(unsigned int i=0; i<=iteration; i++)
	{
	    unsigned int extent_sz =  (tempBuffsize > max_ext) ? max_ext : tempBuffsize; 
		concurrency::extent< 1 > inputExtent( extent_sz );
		concurrency::tiled_extent< kernel0_WgSize > tileK0 = inputExtent.tile< kernel0_WgSize >();
		unsigned int index = i*(tile_limit*kernel0_WgSize);
		unsigned int tile_index = i*tile_limit;

			concurrency::parallel_for_each( av, tileK0,
			[
				firstKey,
				firstValue,
				init,
				numElements,
				&keySumArray,
				&preSumArray,
				&preSumArray1,
		      	binary_pred,
				exclusive,
				index,
				tile_index,
		      	binary_funct,
				kernel0_WgSize
			] ( concurrency::tiled_index< kernel0_WgSize > t_idx ) restrict(amp)
			  {

				int gloId = t_idx.global[ 0 ] + index;
				unsigned int groId = t_idx.tile[ 0 ] + tile_index;
				unsigned int locId = t_idx.local[ 0 ];
				int wgSize = kernel0_WgSize;

				tile_static vType ldsVals[ SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES*2 ];
				tile_static kType ldsKeys[ SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES*2 ];
   				wgSize *=2;

  				unsigned int  offset = 1;
				// load input into shared memory
				int input_offset = (groId*wgSize)+locId;

				if (exclusive)
				{
				   if (gloId > 0 && input_offset < numElements)
				   {
					  kType key1 = firstKey[ input_offset ];
					  kType key2 = firstKey[ input_offset-1 ];
					  if( binary_pred( key1, key2 )  )
					  {
						 ldsVals[locId] = firstValue[ input_offset ];
					  }
					  else 
					  {
						 vType start_val = firstValue[ input_offset ]; 
						 ldsVals[locId] = binary_funct(init, start_val);
					  }
					  ldsKeys[locId] = firstKey[ input_offset ];
				   }
				   else{ 
					  vType start_val = firstValue[0];
					  ldsVals[locId] = binary_funct(init, start_val);
					  ldsKeys[locId] = firstKey[0];
				   }
				   if(input_offset + (wgSize/2) < numElements)
				   {
					  kType key1 = firstKey[ input_offset + (wgSize/2)];
					  kType key2 = firstKey[ input_offset + (wgSize/2) -1];
					  if( binary_pred( key1, key2 )  )
					  {
						 ldsVals[locId+(wgSize/2)] = firstValue[ input_offset + (wgSize/2)];
					  }
					  else 
					  {
						 vType start_val = firstValue[ input_offset + (wgSize/2)]; 
						 ldsVals[locId+(wgSize/2)] = binary_funct(init, start_val);
					  } 
					  ldsKeys[locId+(wgSize/2)] = firstKey[ input_offset + (wgSize/2)];
				   }
				}
				else
				{
				   if(input_offset < numElements)
				   {
					   ldsVals[locId] = firstValue[input_offset];
					   ldsKeys[locId] = firstKey[input_offset];
				   }
				   if(input_offset + (wgSize/2) < numElements)
				   {
					   ldsVals[locId+(wgSize/2)] = firstValue[ input_offset + (wgSize/2)];
					   ldsKeys[locId+(wgSize/2)] = firstKey[ input_offset + (wgSize/2)];
				   }
				}
				for (unsigned int start = wgSize>>1; start > 0; start >>= 1) 
				{
				   t_idx.barrier.wait();
				   if (locId < start)
				   {
					  unsigned int temp1 = offset*(2*locId+1)-1;
					  unsigned int temp2 = offset*(2*locId+2)-1;
       
					  kType key = ldsKeys[temp2]; 
					  kType key1 = ldsKeys[temp1];
					  if(binary_pred( key, key1 )) 
					  {
						 oType y = ldsVals[temp2];
						 oType y1 =ldsVals[temp1];
						 ldsVals[temp2] = binary_funct(y, y1);
					  }

				   }
				   offset *= 2;
				}
				t_idx.barrier.wait();
				if (locId == 0)
				{
					keySumArray[ groId ] = ldsKeys[ wgSize-1 ];
					preSumArray[ groId ] = ldsVals[wgSize -1];
					preSumArray1[ groId ] = ldsVals[wgSize/2 -1];
				}

		 } );
	     tempBuffsize = tempBuffsize - max_ext;
	}

	PEEK_AT( output )

   /**********************************************************************************
     *  Kernel 1
     *********************************************************************************/
    int workPerThread = static_cast< int >( sizeScanBuff / kernel1_WgSize );
    workPerThread = workPerThread ? workPerThread : 1;

	concurrency::extent< 1 > globalSizeK1( kernel1_WgSize );
    concurrency::tiled_extent< kernel1_WgSize > tileK1 = globalSizeK1.tile< kernel1_WgSize >();

    //std::cout << "Kernel 1 Launching w/" << sizeScanBuff << " threads for " << numWorkGroupsK0 << " elements. " << std::endl;
    concurrency::parallel_for_each( av, tileK1,
        [
	      	&keySumArray, 
            &preSumArray,
            numWorkGroupsK0,
            workPerThread,
	        binary_pred,
		    numElements,
            binary_funct,
		    kernel1_WgSize
        ] ( concurrency::tiled_index< kernel1_WgSize > t_idx ) restrict(amp)
  {

	unsigned int gloId = t_idx.global[ 0 ];
    unsigned int groId = t_idx.tile[ 0 ];
    int locId = t_idx.local[ 0 ];
    int wgSize = kernel1_WgSize;
    int mapId  = gloId * workPerThread;

    tile_static kType ldsKeys[ SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES];
	tile_static vType ldsVals[ SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES ];

	// do offset of zero manually
    int offset;
    kType key;
    vType workSum;
    if (mapId < numWorkGroupsK0)
    {
        kType prevKey;

        // accumulate zeroth value manually
        offset = 0;
        key = keySumArray[ mapId+offset ];
        workSum = preSumArray[ mapId+offset ];
        //  Serial accumulation
        for( offset = offset+1; offset < workPerThread; offset += 1 )
        {
            prevKey = key;
            key = keySumArray[ mapId+offset ];
            if (mapId+offset<numWorkGroupsK0 )
            {
                vType y = preSumArray[ mapId+offset ];
                if ( binary_pred(key, prevKey ) )
                {
                    workSum = binary_funct( workSum, y );
                }
                else
                {
                    workSum = y;
                }
                preSumArray[ mapId+offset ] = workSum;
            }
        }
    }
    t_idx.barrier.wait();
    vType scanSum = workSum;
    offset = 1;
    // load LDS with register sums
    ldsVals[ locId ] = workSum;
    ldsKeys[ locId ] = key;
    // scan in lds

    for( offset = offset*1; offset < wgSize; offset *= 2 )
    {
        t_idx.barrier.wait();
        if (mapId < numWorkGroupsK0)
        {
            if (locId >= offset  )
            {
                vType y = ldsVals[ locId - offset ];
                kType key1 = ldsKeys[ locId ];
                kType key2 = ldsKeys[ locId-offset ];
                if ( binary_pred( key1, key2 ) )
                {
                    scanSum = binary_funct( scanSum, y );
                }
                else
                    scanSum = ldsVals[ locId ];
            }
      
        }
        t_idx.barrier.wait();
        ldsVals[ locId ] = scanSum;
    } // for offset
    t_idx.barrier.wait();

    // write final scan from pre-scan and lds scan
    for( offset = 0; offset < workPerThread; offset += 1 )
    {
        t_idx.barrier.wait_with_global_memory_fence();

        if (mapId+offset < numWorkGroupsK0 && locId > 0)
        {
            vType y = preSumArray[ mapId+offset ];
            kType key1 = keySumArray[ mapId+offset ]; // change me
            kType key2 = ldsKeys[ locId-1 ];
            if ( binary_pred( key1, key2 ) )
            {
                vType y2 = ldsVals[locId-1];
                y = binary_funct( y, y2 );
            }
            preSumArray[ mapId+offset ] = y;
        } // thread in bounds
    } // for 

  });

   PEEK_AT( postSumArray )

 /**********************************************************************************
     *  Kernel 2
 *********************************************************************************/
 	tempBuffsize = (sizeInputBuff); 
	iteration = (tempBuffsize-1)/max_ext; 

    for(unsigned int a=0; a<=iteration ; a++)
	{
	    unsigned int extent_sz =  (tempBuffsize > max_ext) ? max_ext : tempBuffsize; 
		concurrency::extent< 1 > inputExtent( extent_sz );
		concurrency::tiled_extent< kernel2_WgSize > tileK2 = inputExtent.tile< kernel2_WgSize >();
		unsigned int index = a*(tile_limit*kernel2_WgSize);
		unsigned int tile_index = a*tile_limit;

			concurrency::parallel_for_each( av, tileK2,
				[
	      			firstKey,
					firstValue,
					result,
					&preSumArray,
					&preSumArray1,
					binary_pred,
					exclusive,
					index,
					tile_index,
					numElements,
					init,
					binary_funct,
					kernel2_WgSize
				] ( concurrency::tiled_index< kernel2_WgSize > t_idx ) restrict(amp)
		  {

			int gloId = t_idx.global[ 0 ] + index;
			unsigned int groId = t_idx.tile[ 0 ] + tile_index;
			unsigned int locId = t_idx.local[ 0 ];
			unsigned int wgSize = kernel2_WgSize;

			tile_static kType ldsKeys[ SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES ];
			tile_static vType ldsVals[ SCANBYKEY_WAVESIZE*SCANBYKEY_KERNELWAVES ];
	
				 // if exclusive, load gloId=0 w/ init, and all others shifted-1
			kType key;
			oType val;
			if (gloId < numElements){
			   if (exclusive)
			   {
				  if (gloId > 0)
				  { // thread>0
					  key = firstKey[ gloId];
					  kType key1 = firstKey[ gloId];
					  kType key2 = firstKey[ gloId-1];
					  if( binary_pred( key1, key2 )  )
						  val = firstValue[ gloId-1 ];
					  else 
						  val = init;
					  ldsKeys[ locId ] = key;
					  ldsVals[ locId ] = val;
				  }
				  else
				  { // thread=0
					  val = init;
					  ldsVals[ locId ] = val;
					  ldsKeys[ locId ] = firstKey[gloId];
					// key stays null, this thread should never try to compare it's key
					// nor should any thread compare it's key to ldsKey[ 0 ]
					// I could put another key into lds just for whatevs
					// for now ignore this
				  }
			   }
			   else
			   {
				  key = firstKey[ gloId ];
				  val = firstValue[ gloId ];
				  ldsKeys[ locId ] = key;
				  ldsVals[ locId ] = val;
			   }
			 }
	
			 // Each work item writes out its calculated scan result, relative to the beginning
			// of each work group
			vType scanResult = ldsVals[ locId ];
			vType postBlockSum, newResult;
			vType y, y1, sum;
			kType key1, key2, key3, key4;
	
			if(locId == 0 && gloId < numElements)
			{
				if(groId > 0)
				{
				   key1 = firstKey[gloId];
				   key2 = firstKey[groId*wgSize -1 ];

				   if(groId % 2 == 0)
				   {
					  postBlockSum = preSumArray[ groId/2 -1 ];
				   }
				   else if(groId == 1)
				   {
					  postBlockSum = preSumArray1[0];
				   }
				   else
				   {
					  key3 = firstKey[groId*wgSize -1 ];
					  key4 = firstKey[(groId-1)*wgSize -1];
					  if(binary_pred(key3 ,key4))
					  {
						 y = preSumArray[ groId/2 -1 ];
						 y1 = preSumArray1[groId/2];
						 postBlockSum = binary_funct(y, y1);
					  }
					  else
					  {
						 postBlockSum = preSumArray1[groId/2];
					  }
				   }
				   if (!exclusive)
				   {
					  if(binary_pred( key1, key2))
					  {
						 newResult = binary_funct( scanResult, postBlockSum );
					  }
					  else
					  {
						 newResult = scanResult;
					  }
				   }
				   else
				   {
					  if(binary_pred( key1, key2)) 
						 newResult = postBlockSum;
					  else
						 newResult = init;  
				   }
				}
				else
				{
					 newResult = scanResult;
				}
				ldsVals[ locId ] = newResult;
			}
			
			// Computes a scan within a workgroup
			// updates vals in lds but not keys
			sum = ldsVals[ locId ];
			for( unsigned int offset = 1; offset < wgSize; offset *= 2 )
			{
				t_idx.barrier.wait();
				if (locId >= offset )
				{
					kType key2 = ldsKeys[ locId - offset];
					if( binary_pred( key, key2 )  )
					{
						oType y = ldsVals[ locId - offset];
						sum = binary_funct( sum, y );
					}
					else
						sum = ldsVals[ locId];
					
				}
				t_idx.barrier.wait();
				ldsVals[ locId ] = sum;
			}
			 t_idx.barrier.wait(); // needed for large data types
			//  Abort threads that are passed the end of the input vector
			if (gloId >= numElements) return; 
			result[ gloId ] = sum;
	
	
		  } );
		  tempBuffsize = tempBuffsize - max_ext;
	}
    PEEK_AT( output )
}   //end of scan_by_key_enqueue( )


/*!
* \brief This overload is called strictly for non-device_vector iterators
* \details This template function overload is used to seperate device_vector iterators from all other iterators
*/
template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction >
OutputIterator
scan_by_key_pick_iterator(
    control& ctl,
    const InputIterator1& firstKey,
    const InputIterator1& lastKey,
    const InputIterator2& firstValue,
    const OutputIterator& result,
    const T& init,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_funct,
    const std::string& user_code,
    const bool& inclusive,
	std::random_access_iterator_tag,
	std::random_access_iterator_tag )
{
    typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< OutputIterator >::value_type oType;
    static_assert( std::is_convertible< vType, oType >::value, "InputValue and Output iterators are incompatible" );

    int numElements = static_cast< int >( std::distance( firstKey, lastKey ) );
    if( numElements < 1 )
        return result;


    bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode( );


    if( runMode == bolt::amp::control::Automatic )
    {
        runMode = ctl.getDefaultPathToRun( );
    }

    #if defined(BOLT_DEBUG_LOG)
    BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
    #endif
                
    if( runMode == bolt::amp::control::SerialCpu )
    {
          #if defined(BOLT_DEBUG_LOG)
          dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_SERIAL_CPU,"::Scan_By_Key::SERIAL_CPU");
          #endif
          if(inclusive){
          Serial_inclusive_scan_by_key( firstKey, firstValue, result, numElements, binary_pred, binary_funct);
       }
       else{
          Serial_exclusive_scan_by_key( firstKey, firstValue, result, numElements, binary_pred, binary_funct, init);
       }
       return result + numElements;
    }
    else if(runMode == bolt::amp::control::MultiCoreCpu)
    {
#ifdef ENABLE_TBB
      #if defined(BOLT_DEBUG_LOG)
      dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_MULTICORE_CPU,"::Scan_By_Key::MULTICORE_CPU");
      #endif
      if (inclusive)
        return bolt::btbb::inclusive_scan_by_key(firstKey,lastKey,firstValue,result,binary_pred,binary_funct);
      else
        return bolt::btbb::exclusive_scan_by_key(firstKey,lastKey,firstValue,result,init,binary_pred,binary_funct);

#else
        //std::cout << "The MultiCoreCpu version of Scan by key is not enabled." << std ::endl;
        throw std::runtime_error( "The MultiCoreCpu version of scan by key is not enabled to be built! \n" );

#endif
    }
    else
    {
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_OPENCL_GPU,"::Scan_By_Key::OPENCL_GPU");
        #endif
        
        device_vector< kType, concurrency::array_view > dvKeys( firstKey, lastKey, false, ctl );
        device_vector< vType, concurrency::array_view > dvValues( firstValue, numElements, false, ctl );
	    device_vector< oType, concurrency::array_view > dvOutput( result, numElements, true, ctl );


        //Now call the actual cl algorithm
        scan_by_key_enqueue( ctl, dvKeys.begin( ), dvKeys.end( ), dvValues.begin(), dvOutput.begin( ),
            init, binary_pred, binary_funct, user_code, inclusive );

        // This should immediately map/unmap the buffer
        dvOutput.data( );
      }

      return result + numElements;
}

/*!
* \brief This overload is called strictly for device_vector iterators
* \details This template function overload is used to seperate device_vector iterators from all other iterators
*/

    template<
    typename DVInputIterator1,
    typename DVInputIterator2,
    typename DVOutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction >
DVOutputIterator
scan_by_key_pick_iterator(
    control& ctl,
    const DVInputIterator1& firstKey,
    const DVInputIterator1& lastKey,
    const DVInputIterator2& firstValue,
    const DVOutputIterator& result,
    const T& init,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_funct,
    const std::string& user_code,
    const bool& inclusive,
	bolt::amp::device_vector_tag,
	bolt::amp::device_vector_tag )
{
    typedef typename std::iterator_traits< DVInputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< DVInputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;
    static_assert( std::is_convertible< vType, oType >::value, "InputValue and Output iterators are incompatible" );

    int numElements = static_cast< int >( std::distance( firstKey, lastKey ) );
    if( numElements < 1 )
        return result;

    bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode( );

    if( runMode == bolt::amp::control::Automatic )
    {
        runMode = ctl.getDefaultPathToRun( );
    }

    #if defined(BOLT_DEBUG_LOG)
    BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
    #endif
    
    if( runMode == bolt::amp::control::SerialCpu )
    {
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_SERIAL_CPU,"::Scan_By_Key::SERIAL_CPU");
        #endif
          
	  	typename bolt::amp::device_vector< kType >::pointer scanInputkey =  firstKey.getContainer( ).data( );
		typename bolt::amp::device_vector< vType >::pointer scanInputBuffer =  firstValue.getContainer( ).data( );
        typename bolt::amp::device_vector< oType >::pointer scanResultBuffer =  result.getContainer( ).data( );

        if(inclusive)
            Serial_inclusive_scan_by_key(&scanInputkey[ firstKey.m_Index ],
                                 &scanInputBuffer[ firstValue.m_Index ], &scanResultBuffer[ result.m_Index ], numElements, binary_pred, binary_funct);
        else
            Serial_exclusive_scan_by_key(&scanInputkey[ firstKey.m_Index ],
                             &scanInputBuffer[ firstValue.m_Index ], &scanResultBuffer[ result.m_Index ], numElements, binary_pred, binary_funct, init);

        return result + numElements;

    }
    else if( runMode == bolt::amp::control::MultiCoreCpu )
    {
#ifdef ENABLE_TBB

            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_MULTICORE_CPU,"::Scan_By_Key::MULTICORE_CPU");
            #endif
      
           typename  bolt::amp::device_vector< kType >::pointer scanInputkey =  firstKey.getContainer( ).data( );
		    typename bolt::amp::device_vector< vType >::pointer scanInputBuffer =  firstValue.getContainer( ).data( );
            typename bolt::amp::device_vector< oType >::pointer scanResultBuffer =  result.getContainer( ).data( );

            if (inclusive)
               bolt::btbb::inclusive_scan_by_key(&scanInputkey[ firstKey.m_Index ],&scanInputkey[ firstKey.m_Index + numElements],  &scanInputBuffer[ firstValue.m_Index ],
                                                 &scanResultBuffer[ result.m_Index ], binary_pred,binary_funct);
            else
               bolt::btbb::exclusive_scan_by_key(&scanInputkey[ firstKey.m_Index ],&scanInputkey[ firstKey.m_Index + numElements],  &scanInputBuffer[ firstValue.m_Index ],
                                                 &scanResultBuffer[ result.m_Index ],init,binary_pred,binary_funct);

            return result + numElements;
#else
                throw std::runtime_error("The MultiCoreCpu version of scan by key is not enabled to be built! \n" );

#endif

     }
     else{
     #if defined(BOLT_DEBUG_LOG)
     dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_OPENCL_GPU,"::Scan_By_Key::OPENCL_GPU");
     #endif
     //Now call the actual cl algorithm
              scan_by_key_enqueue( ctl, firstKey, lastKey, firstValue, result,
                       init, binary_pred, binary_funct, user_code, inclusive );
     }
     return result + numElements;
}


/*!
* \brief This overload is called strictly for non-device_vector iterators
* \details This template function overload is used to seperate device_vector iterators from all other iterators
*/
template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction >
OutputIterator
scan_by_key_pick_iterator(
    control& ctl,
    const InputIterator1& firstKey,
    const InputIterator1& lastKey,
    const InputIterator2& firstValue,
    const OutputIterator& result,
    const T& init,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_funct,
    const std::string& user_code,
    const bool& inclusive,
	bolt::amp::fancy_iterator_tag,
	std::random_access_iterator_tag  )
{
    typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< OutputIterator >::value_type oType;

    int numElements = static_cast< int >( std::distance( firstKey, lastKey ) );
    if( numElements < 1 )
        return result;


    bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode( );


    if( runMode == bolt::amp::control::Automatic )
    {
        runMode = ctl.getDefaultPathToRun( );
    }

    #if defined(BOLT_DEBUG_LOG)
    BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
    #endif
                
    if( runMode == bolt::amp::control::SerialCpu )
    {
          #if defined(BOLT_DEBUG_LOG)
          dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_SERIAL_CPU,"::Scan_By_Key::SERIAL_CPU");
          #endif
          if(inclusive){
          Serial_inclusive_scan_by_key( firstKey, firstValue, result, numElements, binary_pred, binary_funct);
       }
       else{
          Serial_exclusive_scan_by_key( firstKey, firstValue, result, numElements, binary_pred, binary_funct, init);
       }
       return result + numElements;
    }
    else if(runMode == bolt::amp::control::MultiCoreCpu)
    {
#ifdef ENABLE_TBB
      #if defined(BOLT_DEBUG_LOG)
      dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_MULTICORE_CPU,"::Scan_By_Key::MULTICORE_CPU");
      #endif
      if (inclusive)
        return bolt::btbb::inclusive_scan_by_key(firstKey,lastKey,firstValue,result,binary_pred,binary_funct);
      else
        return bolt::btbb::exclusive_scan_by_key(firstKey,lastKey,firstValue,result,init,binary_pred,binary_funct);

#else
        //std::cout << "The MultiCoreCpu version of Scan by key is not enabled." << std ::endl;
        throw std::runtime_error( "The MultiCoreCpu version of scan by key is not enabled to be built! \n" );

#endif
    }
    else
    {
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SCANBYKEY,BOLTLOG::BOLT_OPENCL_GPU,"::Scan_By_Key::OPENCL_GPU");
        #endif
        
            scan_by_key_enqueue( ctl, firstKey, lastKey, firstValue, result,
                       init, binary_pred, binary_funct, user_code, inclusive );
      }
      return result + numElements;
}



/***********************************************************************************************************************
 * Detect Random Access
 ********************************************************************************************************************/

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction >
OutputIterator
scan_by_key_detect_random_access(
    control& ctl,
    const InputIterator1& firstKey,
    const InputIterator1& lastKey,
    const InputIterator2& firstValue,
    const OutputIterator& result,
    const T& init,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_funct,
    const std::string& user_code,
    const bool& inclusive,
    std::random_access_iterator_tag,
	std::random_access_iterator_tag )
{
    return detail::scan_by_key_pick_iterator( ctl, firstKey, lastKey, firstValue, result, init,
        binary_pred, binary_funct, user_code, inclusive, 
		typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( ) );
}


template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction >
OutputIterator
scan_by_key_detect_random_access(
    control& ctl,
    const InputIterator1& firstKey,
    const InputIterator1& lastKey,
    const InputIterator2& firstValue,
    const OutputIterator& result,
    const T& init,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_funct,
    const std::string& user_code,
    const bool& inclusive,
    std::random_access_iterator_tag, bolt::amp::fancy_iterator_tag )
{
    //  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
    //  to a temporary buffer.  Should we?
	static_assert( std::is_same< OutputIterator, bolt::amp::fancy_iterator_tag >::value , "It is not possible to output to fancy iterators; they are not mutable" );
};


template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction >
OutputIterator
scan_by_key_detect_random_access(
    control& ctl,
    const InputIterator1& firstKey,
    const InputIterator1& lastKey,
    const InputIterator2& firstValue,
    const OutputIterator& result,
    const T& init,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_funct,
    const std::string& user_code,
    const bool& inclusive,
    std::input_iterator_tag, std::input_iterator_tag  )
{
    //  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
    //  to a temporary buffer.  Should we?
    static_assert(std::is_same< InputIterator1, std::input_iterator_tag >::value  , "Bolt only supports random access iterator types" );
};



    /*!   \}  */
} //namespace detail


/*********************************************************************************************************************
 * Inclusive Segmented Scan
 ********************************************************************************************************************/
template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
inclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    BinaryPredicate binary_pred,
    BinaryFunction  binary_funct,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    control& ctl = control::getDefault();
    oType init; memset(&init, 0, sizeof(oType) );
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        true, // inclusive
        typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate>
OutputIterator
inclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    control& ctl = control::getDefault();
    oType init; memset(&init, 0, sizeof(oType) );
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        true, // inclusive
       	typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator>
OutputIterator
inclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    control& ctl = control::getDefault();
    oType init; memset(&init, 0, sizeof(oType) );
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        true, // inclusive
        typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

///////////////////////////// CTRL ////////////////////////////////////////////

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
inclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    BinaryPredicate binary_pred,
    BinaryFunction  binary_funct,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    oType init; memset(&init, 0, sizeof(oType) );
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        true, // inclusive
       	typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate>
OutputIterator
inclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    oType init; memset(&init, 0, sizeof(oType) );
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        true, // inclusive
       	typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator>
OutputIterator
inclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    oType init; memset(&init, 0, sizeof(oType) );
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        true, // inclusive
        typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}




/*********************************************************************************************************************
 * Exclusive Segmented Scan
 ********************************************************************************************************************/

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
exclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    BinaryPredicate binary_pred,
    BinaryFunction  binary_funct,
    const std::string& user_code )
{
    control& ctl = control::getDefault();
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        false, // exclusive
       	typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate>
OutputIterator
exclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    control& ctl = control::getDefault();
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        false, // exclusive
        typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T>
OutputIterator
exclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    control& ctl = control::getDefault();
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        false, // exclusive
        typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
	);
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator>
OutputIterator
exclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    control& ctl = control::getDefault();
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        0,
        binary_pred,
        binary_funct,
        user_code,
        false, // exclusive
        typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

///////////////////////////// CTRL ////////////////////////////////////////////

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
exclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    BinaryPredicate binary_pred,
    BinaryFunction  binary_funct,
    const std::string& user_code )
{
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        false, // exclusive
       	typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate>
OutputIterator
exclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        false, // exclusive
       	typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T>
OutputIterator
exclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        user_code,
        false, // exclusive
       	typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator>
OutputIterator
exclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
    return detail::scan_by_key_detect_random_access(
        ctl,
        first1,
        last1,
        first2,
        result,
        0,
        binary_pred,
        binary_funct,
        user_code,
        false, // exclusive
		typename std::iterator_traits< InputIterator1 >::iterator_category( ),
		typename std::iterator_traits< OutputIterator >::iterator_category( )
    ); // return
}


} //namespace cl
} //namespace bolt

#endif
