#pragma once

namespace details
{

#define TRANSFORMSCAN_KERNELWAVES 4
#define TRANSFORMSCAN_WAVESIZE 128
#define TRANSFORMSCAN_TILE_MAX 65535

template<
    typename InputIterator,
    typename OutputIterator,
    typename UnaryFunction,
    typename T,
    typename BinaryFunction >
void
transform_scan_impl(
    const InputIterator& first,
    const InputIterator& last,
    const OutputIterator& result,
    const UnaryFunction& unary_op,
    const T& init_T,
    const BinaryFunction& binary_op,
    const bool& inclusive = true )
{

    /**********************************************************************************
     * Type Names - used in KernelTemplateSpecializer
     *********************************************************************************/
    typedef typename std::iterator_traits< InputIterator  >::value_type iType;
    typedef typename std::iterator_traits< OutputIterator >::value_type oType;
  
	int exclusive = inclusive ? 0 : 1;

    int numElements = static_cast< int >( std::distance( first, last ) );
    const unsigned int kernel0_WgSize = TRANSFORMSCAN_WAVESIZE*TRANSFORMSCAN_KERNELWAVES;
    const unsigned int kernel1_WgSize = TRANSFORMSCAN_WAVESIZE*TRANSFORMSCAN_KERNELWAVES ;
    const unsigned int kernel2_WgSize = TRANSFORMSCAN_WAVESIZE*TRANSFORMSCAN_KERNELWAVES;

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

    hc::array_view<oType> preSumArray((hc::extent<1>(sizeScanBuff)));
    hc::array_view<oType> preSumArray1((hc::extent<1>(sizeScanBuff)));

    /**********************************************************************************
     *  Kernel 0
     *********************************************************************************/

    const unsigned int tile_limit = TRANSFORMSCAN_TILE_MAX;
	const unsigned int max_ext = (tile_limit*kernel0_WgSize);
	unsigned int	   tempBuffsize = (sizeInputBuff/2); 
	unsigned int	   iteration = (tempBuffsize-1)/max_ext; 
    auto f_ = utils::get_pointer(first);
    hc::array_view<iType> first_(hc::extent<1>(numElements), f_);
 

    for(unsigned int i=0; i<=iteration; i++)
    {
        unsigned int extent_sz =  (tempBuffsize > max_ext) ? max_ext : tempBuffsize; 
		unsigned int tile_index = i*tile_limit;
        unsigned int index = i*(tile_limit*kernel0_WgSize);
        auto kernel = [first_, init_T, numElements, preSumArray, preSumArray1,
             binary_op, exclusive, index, tile_index, kernel0_WgSize, unary_op]
                 (hc::tiled_index<1> t_idx) [[hc]]
                 {
                     unsigned int gloId = t_idx.global[ 0 ];
                     unsigned int groId = t_idx.tile[ 0 ];
                     unsigned int locId = t_idx.local[ 0 ];
                     int wgSize = kernel0_WgSize;

                     tile_static oType lds[ TRANSFORMSCAN_WAVESIZE*TRANSFORMSCAN_KERNELWAVES*2 ];

                     wgSize *=2;

                     oType val;
                     int input_offset = (groId*wgSize)+locId+index;
                     // load input into shared memory
                     if(input_offset < numElements){
                         iType inVal = first_[input_offset];
                         val = unary_op(inVal);
                         lds[locId] = val;
                     }
                     if(input_offset + (wgSize/2) < numElements){
                         iType inVal = first_[ input_offset + (wgSize/2)];
                         val = unary_op(inVal);
                         lds[locId+(wgSize/2)] = val;
                     }

                     // Exclusive case
                     if(exclusive && gloId == 0)
                     {
                         iType start_val = first_[0];
                         oType val = unary_op(start_val);
                         lds[locId] = binary_op(init_T, val);
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
                             oType y = lds[temp2];
                             oType y1 =lds[temp1];
                             lds[temp2] = binary_op(y, y1);
                         }
                         offset *= 2;
                     }
                     t_idx.barrier.wait();
                     if (locId == 0)
                     {
                         preSumArray[ groId + tile_index ] = lds[wgSize -1];
                         preSumArray1[ groId + tile_index ] = lds[wgSize/2 -1];
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
        [ preSumArray, numWorkGroupsK0, workPerThread,
        binary_op, kernel1_WgSize]
            ( hc::tiled_index< 1 > t_idx ) [[hc]]
            {
                unsigned int gloId = t_idx.global[ 0 ];
                unsigned int groId = t_idx.tile[ 0 ];
                int locId = t_idx.local[ 0 ];
                int wgSize = kernel1_WgSize;
                int mapId  = gloId * workPerThread;

                tile_static oType lds[ TRANSFORMSCAN_WAVESIZE*TRANSFORMSCAN_KERNELWAVES ];

                // do offset of zero manually
                int offset;
                oType workSum;
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
                            oType y = preSumArray[mapId+offset];
                            workSum = binary_op( workSum, y );
                        }
                    }
                }
                t_idx.barrier.wait();
                oType scanSum = workSum;
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
                            oType y = lds[ locId - offset ];
                            scanSum = binary_op( scanSum, y );
                        }

                    }
                    t_idx.barrier.wait();
                    lds[ locId ] = scanSum;
                } // for offset
                t_idx.barrier.wait();
                workSum = preSumArray[mapId];
                if(locId > 0){
                    oType y = lds[locId-1];
                    workSum = binary_op(workSum, y);
                    preSumArray[ mapId] = workSum;
                }
                else{
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
    details::kernel_launch(kernel1_WgSize, kernel1 ,kernel1_WgSize);


    /**********************************************************************************
     *  Kernel 2
     *********************************************************************************/
	tempBuffsize = (sizeInputBuff); 
	iteration = (tempBuffsize-1)/max_ext; 

    auto r_ = utils::get_pointer(result);
    hc::array_view<oType> re(hc::extent<1>(numElements), r_);
    re.discard_data();
    for(unsigned int a=0; a<=iteration ; a++)
	{
	    unsigned int extent_sz =  (tempBuffsize > max_ext) ? max_ext : tempBuffsize; 
		unsigned int index = a*(tile_limit*kernel2_WgSize);
		unsigned int tile_index = a*tile_limit;
        auto kernel2 = [first_, init_T, numElements, preSumArray, preSumArray1,
             binary_op, exclusive, index, tile_index, kernel2_WgSize, unary_op, re]
                (hc::tiled_index<1> t_idx) [[hc]]
                {
                    int gloId = t_idx.global[ 0 ] + index;
                    unsigned int groId = t_idx.tile[ 0 ] + tile_index;
                    int locId = t_idx.local[ 0 ];
                    int wgSize = kernel2_WgSize;

                    tile_static oType lds[ TRANSFORMSCAN_WAVESIZE*TRANSFORMSCAN_KERNELWAVES ];
                    // if exclusive, load gloId=0 w/ identity, and all others shifted-1
                    iType val;
                    oType oval;

                    if (gloId < numElements){
                        if (exclusive)
                        {
                            if (gloId > 0)
                            { // thread>0
                                val = first_[gloId-1];
                                oval = unary_op(val);
                                lds[ locId ] = oval;
                            }
                            else
                            { // thread=0
                                oval = init_T;
                                lds[ locId ] = oval;
                            }
                        }
                        else
                        {
                            val = first_[gloId];
                            oval = unary_op(val);
                            lds[ locId ] = oval;
                        }
                    }

                    oType scanResult = lds[locId];
                    oType postBlockSum, newResult;
                    // accumulate prefix
                    oType y, y1, sum;
                    if(locId == 0 && gloId < numElements)
                    {
                        if(groId > 0) {
                            if(groId % 2 == 0)
                                postBlockSum = preSumArray[ groId/2 -1 ];
                            else if(groId == 1)
                                postBlockSum = preSumArray1[0];
                            else {
                                y = preSumArray[ groId/2 -1 ];
                                y1 = preSumArray1[groId/2];
                                postBlockSum = binary_op(y, y1);
                            }
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
                            oType y = lds[ locId - offset ];
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
        details::kernel_launch(extent_sz, kernel2, kernel2_WgSize);
	    tempBuffsize = tempBuffsize - max_ext;
	}
    //std::cout << "Kernel 2 Done" << std::endl;

}   //end of transform_scan

}
