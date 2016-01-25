//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace details {

#define SCAN_KERNELWAVES 2
#define SCAN_WAVESIZE 128
#define SCAN_TILE_MAX 65535

template<
    typename InputIterator,
    typename OutputIterator,
    typename T,
    typename BinaryFunction >
void scan_impl(
    const InputIterator& first,
    const InputIterator& last,
    const OutputIterator& result,
    const T& init,
    const BinaryFunction& binary_op,
    const bool& inclusive = true )
{
    /**********************************************************************************
     * Type Names - used in KernelTemplateSpecializer
     *********************************************************************************/
    typedef typename std::iterator_traits< InputIterator >::value_type iType;
    typedef typename std::iterator_traits< OutputIterator >::value_type oType;

    int exclusive = inclusive ? 0 : 1;

    int numElements = static_cast< int >( std::distance( first, last ) );
    const unsigned int kernel0_WgSize = SCAN_WAVESIZE*SCAN_KERNELWAVES;
    const unsigned int kernel1_WgSize = SCAN_WAVESIZE*SCAN_KERNELWAVES ;
    const unsigned int kernel2_WgSize = SCAN_WAVESIZE*SCAN_KERNELWAVES;

    //  Ceiling function to bump the size of input to the next whole wavefront size
    unsigned int sizeInputBuff = numElements;
    unsigned int modWgSize = (sizeInputBuff & ((kernel0_WgSize)-1));
    if( modWgSize )
    {
        sizeInputBuff &= ~modWgSize;
        sizeInputBuff += (kernel0_WgSize);
    }
    // tiles number
    int numWorkGroupsK0 = static_cast< int >( sizeInputBuff / (kernel0_WgSize) );
    //  Ceiling function to bump the size of the sum array to the next whole wavefront size
    unsigned int sizeScanBuff = numWorkGroupsK0;
    modWgSize = (sizeScanBuff & ((kernel0_WgSize)-1));
    if( modWgSize )
    {
        sizeScanBuff &= ~modWgSize;
        sizeScanBuff += (kernel0_WgSize);
    }

    hc::array_view<iType> preSumArray((hc::extent<1>(sizeScanBuff)));

    /**********************************************************************************
     *  Kernel 0
     *********************************************************************************/
  //	Loop to calculate the inclusive scan of each individual tile, and output the block sums of every tile
  //	This loop is inherently parallel; every tile is independant with potentially many wavefronts

	const unsigned int tile_limit = SCAN_TILE_MAX;
	const unsigned int max_ext = (tile_limit*kernel0_WgSize);
	unsigned int	   tempBuffsize = (sizeInputBuff); 
	unsigned int	   iteration = (tempBuffsize-1)/max_ext; 

    auto f_ = utils::get_pointer(first);
    hc::array_view<iType> first_(hc::extent<1>(numElements), f_);
    for(unsigned int i=0; i<=iteration; i++)
	{
	    unsigned int extent_sz =  (tempBuffsize > max_ext) ? max_ext : tempBuffsize; 
		unsigned int index = i*(tile_limit*kernel0_WgSize);
		unsigned int tile_index = i*tile_limit;

        auto kernel =
            [first_, init, numElements, preSumArray,
            exclusive, index, tile_index, kernel0_WgSize, binary_op]
                ( hc::tiled_index< 1 > t_idx ) [[hc]]
                {
                    unsigned int gloId = t_idx.global[ 0 ] + index;
                    unsigned int groId = t_idx.tile[ 0 ] + tile_index;
                    unsigned int locId = t_idx.local[ 0 ];
                    int wgSize = kernel0_WgSize;
                    tile_static iType lds[ SCAN_WAVESIZE*SCAN_KERNELWAVES ];

                    int input_offset = (groId*wgSize)+locId;
                    // if exclusive, load gloId=0 w/ identity, and all others shifted-1
                    if(input_offset < numElements)
                        lds[locId] = first_[input_offset];
                    if(input_offset+(wgSize/2) < numElements)
                        lds[locId+(wgSize/2)] = first_[ input_offset+(wgSize/2)];

                    // Exclusive case
                    if(exclusive && gloId == 0)
                    {
                        iType start_val = first_[0];
                        lds[locId] = binary_op(init, start_val);
                    }
                    unsigned int  offset = 1;
                    //  Computes a scan within a workgroup with two data per element

                    for (unsigned int start = wgSize>>1; start > 0; start >>= 1) 
                    {
                        t_idx.barrier.wait();
                        if (locId < start)
                        {
                            unsigned int temp1 = offset*(2*locId+1)-1;
                            unsigned int temp2 = offset*(2*locId+2)-1;
                            iType y = lds[temp2];
                            iType y1 =lds[temp1];

                            lds[temp2] = binary_op(y, y1);
                        }
                        offset *= 2;
                    }
                    t_idx.barrier.wait();
                    if (locId == 0)
                    {
                        preSumArray[ groId  ] = lds[wgSize -1];
                    }
                };

        details::kernel_launch(extent_sz, kernel, kernel0_WgSize);
        tempBuffsize = tempBuffsize - max_ext;
	}

    /**********************************************************************************
     *  Kernel 1
     *********************************************************************************/

    int workPerThread = static_cast< int >( sizeScanBuff / kernel1_WgSize );
    workPerThread = workPerThread ? workPerThread : 1;

    auto kernel1 =
        [preSumArray, numWorkGroupsK0,
        workPerThread, binary_op, kernel1_WgSize]
            ( hc::tiled_index< 1 > t_idx ) [[hc]]
            {
                unsigned int gloId = t_idx.global[ 0 ];
                int locId = t_idx.local[ 0 ];
                int wgSize = kernel1_WgSize;
                int mapId  = gloId * workPerThread;

                tile_static iType lds[ SCAN_WAVESIZE*SCAN_KERNELWAVES ];

                // do offset of zero manually
                int offset;
                iType workSum;
                if (mapId < numWorkGroupsK0)
                {
                    // accumulate zeroth value manually
                    offset = 0;
                    workSum = preSumArray[mapId+offset];
                    //  Serial accumulation
                    for( offset = offset+1; offset < workPerThread; offset += 1 )
                    {
                        if (mapId+offset<numWorkGroupsK0)
                        {
                            iType y = preSumArray[mapId+offset];
                            workSum = binary_op( workSum, y );
                        }
                    }
                }
                t_idx.barrier.wait();
                iType scanSum = workSum;
                offset = 1;
                // load LDS with register sums
                lds[ locId ] = workSum;

                // scan in lds
                for( offset = offset*1; offset < wgSize; offset *= 2 )
                {
                    t_idx.barrier.wait();
                    if (mapId < numWorkGroupsK0)
                    {
                        if (locId >= offset)
                        {
                            iType y = lds[ locId - offset ];
                            scanSum = binary_op( scanSum, y );
                        }
                    }
                    t_idx.barrier.wait();
                    lds[ locId ] = scanSum;
                } // for offset
                t_idx.barrier.wait();
                workSum = preSumArray[mapId];
                if(locId > 0){
                    iType y = lds[locId-1];
                    workSum = binary_op(workSum, y);
                    preSumArray[ mapId] = workSum;
                }

                // write final scan from pre-scan and lds scan
                for( offset = 1; offset < workPerThread; offset += 1 )
                {
                    t_idx.barrier.wait_with_global_memory_fence();
                    if (mapId+offset < numWorkGroupsK0 && locId > 0)
                    {
                        iType y  = preSumArray[ mapId + offset ] ;
                        iType y1 = binary_op(y, workSum);
                        preSumArray[ mapId + offset ] = y1;
                        workSum = y1;

                    } // thread in bounds
                    else if(mapId+offset < numWorkGroupsK0 ){
                        iType y  = preSumArray[ mapId + offset ] ;
                        preSumArray[ mapId + offset ] = binary_op(y, workSum);
                        workSum = preSumArray[ mapId + offset ];
                    }

                } // for
            };

    details::kernel_launch(kernel1_WgSize, kernel1, kernel1_WgSize);

    /**********************************************************************************
     *  Kernel 2
     *********************************************************************************/
	tempBuffsize = (sizeInputBuff); 
	iteration = (tempBuffsize-1)/max_ext; 
    auto re_ = utils::get_pointer(result);
    hc::array_view<oType> re(hc::extent<1>(numElements), re_);
    re.discard_data();

    for(unsigned int a=0; a<=iteration ; a++)
    {
        unsigned int extent_sz =  (tempBuffsize > max_ext) ? max_ext : tempBuffsize; 
        unsigned int index = a*(tile_limit*kernel2_WgSize);
        unsigned int tile_index = a*tile_limit;

        auto kernel =
            [ first_, re, preSumArray, numElements, binary_op,
            init, exclusive, index, tile_index, kernel2_WgSize]
                ( hc::tiled_index< 1 > t_idx ) [[hc]]
                {
                    int gloId = t_idx.global[ 0 ] + index;
                    unsigned int groId = t_idx.tile[ 0 ] + tile_index;
                    int locId = t_idx.local[ 0 ];
                    int wgSize = kernel2_WgSize;

                    tile_static iType lds[ SCAN_WAVESIZE*SCAN_KERNELWAVES ];
                    // if exclusive, load gloId=0 w/ identity, and all others shifted-1
                    iType val;

                    if (gloId < numElements){
                        if (exclusive)
                        {
                            if (gloId > 0)
                            { // thread>0
                                val = first_[gloId-1];
                                lds[ locId ] = val;
                            }
                            else
                            { // thread=0
                                val = init;
                                lds[ locId ] = val;
                            }
                        }
                        else
                        {
                            val = first_[gloId];
                            lds[ locId ] = val;
                        }
                    }

                    iType scanResult = lds[locId];
                    iType postBlockSum, newResult;
                    // accumulate prefix
                    iType y, y1, sum;
                    if(locId == 0 && gloId < numElements)
                    {
                        if(groId > 0) {
                            postBlockSum = preSumArray[ groId-1 ];
                            if (!exclusive)
                                newResult = binary_op( scanResult, postBlockSum );
                            else 
                                newResult =  postBlockSum;
                        }
                        else {
                            newResult = scanResult;
                        }
                        lds[ locId ] = newResult;
                    } 
                    //  Computes a scan within a workgroup
                    sum = lds[ locId ];

                    for( int offset = 1; offset < wgSize; offset *= 2 )
                    {
                        t_idx.barrier.wait();
                        if (locId >= offset)
                        {
                            iType y = lds[ locId - offset ];
                            sum = binary_op( sum, y );
                        }
                        t_idx.barrier.wait();
                        lds[ locId ] = sum;
                    }
                    t_idx.barrier.wait();
                    //  Abort threads that are passed the end of the input vector
                    if (gloId >= numElements) return; 

                    re[ gloId ] = sum;

                };

        details::kernel_launch(extent_sz, kernel, kernel2_WgSize);
        tempBuffsize = tempBuffsize - max_ext;
    }
}   //end of inclusive_scan_enqueue( )

} // namespace details

