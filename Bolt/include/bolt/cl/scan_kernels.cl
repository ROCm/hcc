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
#pragma OPENCL EXTENSION cl_amd_printf : enable
//#define USE_AMD_HSA 1

#if USE_AMD_HSA

/******************************************************************************
 *  HSA Kernel
 *****************************************************************************/
template< typename iType, typename oType, typename initType, typename BinaryFunction >
kernel void HSA_Scan(
    global oType    *output,
    global iType    *input,
    initType        init,
    const uint      numElements,
    const uint      numIterations,
    local oType     *lds,
    global BinaryFunction* binaryOp,
    global oType    *intermediateScanArray,
    global int      *dev2host,
    global int      *host2dev,
    int             exclusive)
{
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    // report P1 completion
    intermediateScanArray[ groId ] = input[ groId ];
    dev2host[ groId ] = 1;



    // wait for P2 completion
    for (size_t i = 0; i < 10000; i++ )
    {
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        //printf("DEV: interScan[%i]=%i ( %i, %i )", groId, intermediateScanArray[groId], dev2host[groId], host2dev[groId]);
        if ( host2dev[ groId] == 2 )
        { // host reported P2 completion
            // report P3 completion
            dev2host[ groId ] = 3;
            break;
        }
    }
}






/******************************************************************************
 *  Not Using HSA
 *****************************************************************************/
#else

#define NUM_ITER 16
#define MIN(X,Y) X<Y?X:Y;
#define MAX(X,Y) X>Y?X:Y;
/******************************************************************************
 *  Kernel 2
 *****************************************************************************/
template< typename iPtrType, typename iIterType, typename oPtrType, typename oIterType, typename BinaryFunction, typename initType >
kernel void perBlockAddition( 
                global oPtrType* output_ptr,
                oIterType    output_iter, 
                global iPtrType* input_ptr,
                iIterType    input_iter, 
                global iPtrType* preSumArray,
                global iPtrType* preSumArray1,
                local iPtrType* lds,
                const uint vecSize,
                global BinaryFunction* binaryOp,
                int exclusive,
                initType identity )
{
    
// 1 thread per element
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    output_iter.init( output_ptr );
    input_iter.init( input_ptr );

  // if exclusive, load gloId=0 w/ identity, and all others shifted-1
    iPtrType val;
  
    if (gloId < vecSize){
	   if (exclusive)
       {
          if (gloId > 0)
          { // thread>0
              val = input_iter[gloId-1];
              lds[ locId ] = val;
          }
          else
          { // thread=0
              val = identity;
              lds[ locId ] = val;
          }
       }
       else
       {
          val = input_iter[gloId];
          lds[ locId ] = val;
       }
    }
  
   iPtrType scanResult = lds[locId];
   iPtrType postBlockSum, newResult;
   iPtrType y, y1, sum;

   if(locId == 0 && gloId < vecSize)
   {
      if(groId > 0) {
         if(groId % 2 == 0)
             postBlockSum = preSumArray[ groId/2 -1 ];
         else if(groId == 1)
             postBlockSum = preSumArray1[0];
         else {
              y = preSumArray[ groId/2 -1 ];
              y1 = preSumArray1[groId/2];
              postBlockSum = (*binaryOp)(y, y1);
         }
         if (!exclusive)
            newResult = (*binaryOp)( scanResult, postBlockSum );
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
    for( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (locId >= offset)
        {
            iPtrType y = lds[ locId - offset ];
            sum = (*binaryOp)( sum, y );
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        lds[ locId ] = sum;
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    //  Abort threads that are passed the end of the input vector
    if (gloId >= vecSize) return; 
   
    output_iter[ gloId ] = sum;
    
}


/******************************************************************************
 *  Kernel 1
 *****************************************************************************/
template< typename iPtrType, typename initType, typename BinaryFunction >
kernel void intraBlockInclusiveScan(
                global iPtrType* preSumArray, 
                initType identity,
                const uint vecSize,
                local iPtrType* lds,
                const uint workPerThread,
                global BinaryFunction* binaryOp
                )
{
    size_t gloId = get_global_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    uint mapId  = gloId * workPerThread;
    // do offset of zero manually
    uint offset;
    iPtrType workSum;
    if (mapId < vecSize)
    {
        // accumulate zeroth value manually
        offset = 0;
        workSum = preSumArray[mapId+offset];

        //  Serial accumulation
        for( offset = offset+1; offset < workPerThread; offset += 1 )
        {
            if (mapId+offset<vecSize)
            {
                iPtrType y = preSumArray[mapId+offset];
                workSum = (*binaryOp)( workSum, y );
            }
        }
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    iPtrType scanSum = workSum;
    lds[ locId ] = workSum;
    offset = 1;
  // scan in lds
    for( offset = offset*1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (mapId < vecSize)
        {
            if (locId >= offset)
            {
                iPtrType y = lds[ locId - offset ];
                scanSum = (*binaryOp)( scanSum, y );
            }
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        lds[ locId ] = scanSum;  

    } // for offset
    barrier( CLK_LOCAL_MEM_FENCE );
    // write final scan from pre-scan and lds scan
     workSum = preSumArray[mapId];
     if(locId > 0){
        iPtrType y = lds[locId-1];
        workSum = (*binaryOp)(workSum, y);
        preSumArray[ mapId] = workSum;
     }
     else{
       preSumArray[ mapId] = workSum;
    }
    for( offset = 1; offset < workPerThread; offset += 1 )
    {
        barrier( CLK_GLOBAL_MEM_FENCE );

        if ((mapId + offset) < vecSize && locId > 0)
        {
            iPtrType y  = preSumArray[ mapId + offset ] ;
            iPtrType y1 = (*binaryOp)(y, workSum);
            preSumArray[ mapId + offset ] = y1;
            workSum = y1;

        } // thread in bounds
        else if((mapId + offset) < vecSize){
           iPtrType y  = preSumArray[ mapId + offset ] ;
           preSumArray[ mapId + offset ] = (*binaryOp)(y, workSum);
           workSum = preSumArray[ mapId + offset ];
        }

    } // for 


} // end kernel


/******************************************************************************
 *  Kernel 0
 *****************************************************************************/
template< typename iPtrType, typename iIterType, typename initType, typename BinaryFunction >
kernel void perBlockInclusiveScan(
                global iPtrType* input_ptr,
                iIterType    input_iter, 
                initType identity,
                const uint vecSize,
                local iPtrType* lds,
                global BinaryFunction* binaryOp,
                global iPtrType* preSumArray,
                global iPtrType* preSumArray1,
                int exclusive) // do exclusive scan ?
{
// 2 thread per element
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 )  ;

    wgSize *=2;
    input_iter.init( input_ptr );
    size_t offset = 1;

   // load input into shared memory
   
    uint input_offset = (groId*wgSize)+locId;

	if(input_offset < vecSize)
       lds[locId] = input_iter[input_offset];
    if(input_offset+(wgSize/2) < vecSize)
        lds[locId+(wgSize/2)] = input_iter[ input_offset+(wgSize/2)];
    
	// Exclusive case
    if(exclusive && gloId == 0)
	{
	    iPtrType start_val = input_iter[0];
		lds[locId] = (*binaryOp)(identity, start_val);
    }
	
    for (size_t start = wgSize>>1; start > 0; start >>= 1) 
    {
       barrier( CLK_LOCAL_MEM_FENCE );
       if (locId < start)
       {
          size_t temp1 = offset*(2*locId+1)-1;
          size_t temp2 = offset*(2*locId+2)-1;
          iPtrType y = lds[temp2];
          iPtrType y1 =lds[temp1];
          lds[temp2] = (*binaryOp)(y, y1);
       }
       offset *= 2;
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    if (locId == 0)
    {
        preSumArray[ groId ] = lds[wgSize -1];
        preSumArray1[ groId ] = lds[wgSize/2 -1];
    }
  
}

// not using HSA
#endif
