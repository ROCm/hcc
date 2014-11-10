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

/******************************************************************************
 *  Kernel 0
 *****************************************************************************/
template<
    typename kType,
    typename kIterType,
    typename vType,
    typename iIterType,
    typename oType,
    typename initType,
    typename BinaryPredicate,
    typename BinaryFunction >
__kernel void perBlockScanByKey(
    global kType *keys,
    kIterType    keys_iter, 
    global vType *vals,
    iIterType     vals_iter,
    initType init,
    const uint vecSize,
    local kType *ldsKeys,
    local oType *ldsVals,
    global BinaryPredicate *binaryPred,
    global BinaryFunction *binaryFunct,
    global kType *keyBuffer,
    global oType *valBuffer,
    global oType *valBuffer1,
    int exclusive) // do exclusive scan ?
{
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    wgSize *=2;
    vals_iter.init( vals );
    keys_iter.init( keys );
    size_t offset = 1;

   // load input into shared memory
    
    if (exclusive)
    {
       if (gloId > 0 && groId*wgSize+locId < vecSize)
	   {
          kType key1 = keys_iter[ groId*wgSize+locId];
          kType key2 = keys_iter[ groId*wgSize+locId-1];
		  
          if( (*binaryPred)( key1, key2 )  )
             ldsVals[locId] = vals_iter[groId*wgSize+locId];
          else 
		  {
		     vType start_val = vals_iter[groId*wgSize+locId]; 
             ldsVals[locId] = (*binaryFunct)(init, start_val);
          }
          ldsKeys[locId] = keys_iter[groId*wgSize+locId];
       }
       else{ 
	      vType start_val = vals_iter[0];
          ldsVals[locId] = (*binaryFunct)(init, start_val);
          ldsKeys[locId] = keys_iter[0];
       }
       if(groId*wgSize +locId+ (wgSize/2) < vecSize)
	   {
          kType key1 = keys_iter[ groId*wgSize +locId+ (wgSize/2)];
          kType key2 = keys_iter[groId*wgSize +locId+ (wgSize/2) -1];
		 
          if( (*binaryPred)( key1, key2 )  )
             ldsVals[locId+(wgSize/2)] = vals_iter[groId*wgSize +locId+ (wgSize/2)];
          else 
		  {
		     vType start_val = vals_iter[groId*wgSize +locId+ (wgSize/2)]; 
             ldsVals[locId+(wgSize/2)] = (*binaryFunct)(init, start_val);
          } 
          ldsKeys[locId+(wgSize/2)] = keys_iter[groId*wgSize +locId+ (wgSize/2)];
       }
     
    }
    else
    {
       if(groId*wgSize+locId < vecSize)
	   {
           ldsVals[locId] = vals_iter[groId*wgSize+locId];
           ldsKeys[locId] = keys_iter[groId*wgSize+locId];
       }
       if(groId*wgSize +locId+ (wgSize/2) < vecSize)
	   {
           ldsVals[locId+(wgSize/2)] = vals_iter[ groId*wgSize +locId+ (wgSize/2)];
           ldsKeys[locId+(wgSize/2)] = keys_iter[ groId*wgSize +locId+ (wgSize/2)];
       }
    }
    

    for (size_t start = wgSize>>1; start > 0; start >>= 1) 
    {
       barrier( CLK_LOCAL_MEM_FENCE );
       if (locId < start)
       {
          size_t temp1 = offset*(2*locId+1)-1;
          size_t temp2 = offset*(2*locId+2)-1;
       
          kType key = ldsKeys[temp2]; 
          kType key1 = ldsKeys[temp1];
          if((*binaryPred)( key, key1 )) 
		  {
             oType y = ldsVals[temp2];
             oType y1 =ldsVals[temp1];
             ldsVals[temp2] = (*binaryFunct)(y, y1);
          }
       }
       offset *= 2;
    }

    

    barrier( CLK_LOCAL_MEM_FENCE );
    if (locId == 0)
    {
        keyBuffer[ groId ] = ldsKeys[ wgSize-1 ];
        valBuffer[ groId ] = ldsVals[wgSize -1];
        valBuffer1[ groId ] = ldsVals[wgSize/2 -1];
    }
}


/******************************************************************************
 *  Kernel 1
 *****************************************************************************/
template<
    typename kType,
    typename oType,
    typename BinaryPredicate,
    typename BinaryFunction >
__kernel void intraBlockInclusiveScanByKey(
    global kType *keySumArray,
    global oType *preSumArray,
    const uint vecSize,
    local kType *ldsKeys,
    local oType *ldsVals,
    const uint workPerThread,
    global BinaryPredicate *binaryPred,
    global BinaryFunction  *binaryFunct )
{
    size_t groId = get_group_id( 0 );
    size_t gloId = get_global_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    uint mapId  = gloId * workPerThread;

    // do offset of zero manually
    uint offset;
    kType key;
    oType workSum;
    if (mapId < vecSize)
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
            if (mapId+offset<vecSize )
            {
                oType y = preSumArray[ mapId+offset ];
                if ( (*binaryPred)(key, prevKey ) )
                {
                    workSum = (*binaryFunct)( workSum, y );
                }
                else
                {
                    workSum = y;
                }
                preSumArray[ mapId+offset ] = workSum;
            }
        }
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    oType scanSum = workSum;
    offset = 1;
    // load LDS with register sums
    ldsVals[ locId ] = workSum;
    ldsKeys[ locId ] = key;
    // scan in lds
    for( offset = offset*1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (mapId < vecSize)
        {
            if (locId >= offset  )
            {
                oType y = ldsVals[ locId - offset ];
                kType key1 = ldsKeys[ locId ];
                kType key2 = ldsKeys[ locId-offset ];
                if ( (*binaryPred)( key1, key2 ) )
                {
                    scanSum = (*binaryFunct)( scanSum, y );
                }
                else
                    scanSum = ldsVals[ locId ];
            }
      
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        ldsVals[ locId ] = scanSum;
    } // for offset
    barrier( CLK_LOCAL_MEM_FENCE );
    
    // write final scan from pre-scan and lds scan
    for( offset = 0; offset < workPerThread; offset += 1 )
    {
        barrier( CLK_GLOBAL_MEM_FENCE );

        if (mapId < vecSize && locId > 0)
        {
            oType y = preSumArray[ mapId+offset ];
            kType key1 = keySumArray[ mapId+offset ]; // change me
            kType key2 = ldsKeys[ locId-1 ];
            if ( (*binaryPred)( key1, key2 ) )
            {
                oType y2 = ldsVals[locId-1];
                y = (*binaryFunct)( y, y2 );
            }
            preSumArray[ mapId+offset ] = y;
        } // thread in bounds
    } // for 
} // end kernel


/******************************************************************************
 *  Kernel 2
 *****************************************************************************/
template<
    typename kType,
    typename kIterType,
    typename vType,
    typename iIterType,
    typename oType,
    typename oIterType,
    typename initType,
    typename BinaryPredicate,
    typename BinaryFunction >
__kernel void perBlockAdditionByKey(
    global oType *preSumArray,
    global oType *preSumArray1,
    global kType *keys,
    kIterType    keys_iter, 
    global vType *vals,
    iIterType     vals_iter,
    global oType *output, // input
    oIterType     output_iter,
    local kType   *ldsKeys,
    local oType   *ldsVals,
    const uint vecSize,
    global BinaryPredicate *binaryPred,
    global BinaryFunction *binaryFunct,
    int exclusive,
    initType init)
{
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    output_iter.init( output);
    vals_iter.init( vals );
    keys_iter.init( keys );
    // if exclusive, load gloId=0 w/ init, and all others shifted-1
    kType key;
    oType val;
    if (gloId < vecSize){
       if (exclusive)
       {
          if (gloId > 0)
          { // thread>0
              key = keys_iter[ gloId];
              kType key1 = keys_iter[ gloId];
              kType key2 = keys_iter[ gloId-1];
              if( (*binaryPred)( key1, key2 )  )
                  val = vals_iter[ gloId-1 ];
              else 
                  val = init;
              ldsKeys[ locId ] = key;
              ldsVals[ locId ] = val;
          }
          else
          { // thread=0
              val = init;
              ldsVals[ locId ] = val;
              ldsKeys[ locId ] = keys_iter[gloId];
            // key stays null, this thread should never try to compare it's key
            // nor should any thread compare it's key to ldsKey[ 0 ]
            // I could put another key into lds just for whatevs
            // for now ignore this
          }
       }
       else
       {
          //vType inVal = vals[gloId];
          //val = (oType) (*unaryOp)(inVal);
          key = keys_iter[ gloId ];
          val = vals_iter[ gloId ];
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
    if(locId == 0 && gloId < vecSize)
    {
        if(groId > 0) {
           key1 = keys_iter[gloId];
           key2 = keys_iter[groId*wgSize -1 ];
           if(groId % 2 == 0)
              postBlockSum = preSumArray[ groId/2 -1 ];
           else if(groId == 1)
              postBlockSum = preSumArray1[0];
           else {
              key3 = keys_iter[groId*wgSize -1];
              key4 = keys_iter[(groId-1)*wgSize -1];
              if((*binaryPred)(key3 ,key4)){
                 y = preSumArray[ groId/2 -1 ];
                 y1 = preSumArray1[groId/2];
                 postBlockSum = (*binaryFunct)(y, y1);
              }
              else
                 postBlockSum = preSumArray1[groId/2];
           }
           if (!exclusive)
	          if((*binaryPred)( key1, key2))
                 newResult = (*binaryFunct)( scanResult, postBlockSum );
              else
		         newResult = scanResult;
	       else
	          if((*binaryPred)( key1, key2)) 
		         newResult = postBlockSum;
              else
		         newResult = init;  
        }
        else {
             newResult = scanResult;
        }
	    ldsVals[ locId ] = newResult;
    }
    // Computes a scan within a workgroup
    // updates vals in lds but not keys
    sum = ldsVals[ locId ];
    for( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (locId >= offset )
        {
            kType key2 = ldsKeys[ locId - offset];
            if( (*binaryPred)( key, key2 )  )
            {
                oType y = ldsVals[ locId - offset];
                sum = (*binaryFunct)( sum, y );
            }
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        ldsVals[ locId ] = sum;
    }
    barrier( CLK_LOCAL_MEM_FENCE ); // needed for large data types
    //  Abort threads that are passed the end of the input vector
    if (gloId >= vecSize) return; 

    output_iter[ gloId ] = sum;
    
}
