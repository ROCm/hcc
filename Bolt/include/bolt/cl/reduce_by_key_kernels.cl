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
    typename BinaryPredicate >
__kernel void OffsetCalculation(
    global kType *ikeys, //input keys
    kIterType keys,
    global int *tempArray, //offsetKeys
    const uint vecSize,
    global BinaryPredicate *binaryPred
    )
{

    keys.init( ikeys );

    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

	 if (gloId >= vecSize) return;

    typename kIterType::value_type key, prev_key;

    if(gloId > 0){
      key = keys[ gloId ];
	  prev_key = keys[ gloId - 1];
	  if((*binaryPred)(key, prev_key))
	    tempArray[ gloId ] = 0;
	  else
		tempArray[ gloId ] = 1;
	}
	else{
		 tempArray[ gloId ] = 0;
	}
}

/******************************************************************************
 *  Kernel 1
 *****************************************************************************/
template<
    typename vType,
    typename vIterType,
    typename BinaryFunction >
__kernel void perBlockScanByKey(
    global int *keys,
    global vType *ivals, //input values
    vIterType vals,
    const uint vecSize,
    local int *ldsKeys,
    local typename vIterType::value_type *ldsVals,
    global BinaryFunction *binaryFunct,
    global int   *keyBuffer,
    global typename vIterType::value_type *valBuffer)
{

    vals.init( ivals );

    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    // if exclusive, load gloId=0 w/ init, and all others shifted-1
    int key;
    typename vIterType::value_type val;

    if(gloId < vecSize){
      key = keys[ gloId ];
      val = vals[ gloId ];
      ldsKeys[ locId ] = key;
      ldsVals[ locId ] = val;
    }
    // Computes a scan within a workgroup
    // updates vals in lds but not keys
    typename vIterType::value_type sum = val;
    for( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        int key2 = ldsKeys[locId - offset];
        if (locId >= offset && key == key2)
        {
            typename vIterType::value_type y = ldsVals[ locId - offset ];
            sum = (*binaryFunct)( sum, y );
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        ldsVals[ locId ] = sum;
    }
    barrier( CLK_LOCAL_MEM_FENCE ); // needed for large data types
    //  Abort threads that are passed the end of the input vector


    if (gloId >= vecSize) return;

    if (locId == 0)
    {
        keyBuffer[ groId ] = ldsKeys[ wgSize-1 ];
        valBuffer[ groId ] = ldsVals[ wgSize-1 ];
    }
}


/******************************************************************************
 *  Kernel 2
 *****************************************************************************/
template<
    typename vType,
    typename BinaryFunction >
__kernel void intraBlockInclusiveScanByKey(
    global int *keySumArray,
    global vType *preSumArray,
    global vType *postSumArray,
    const uint vecSize,
    local int *ldsKeys,
    local vType *ldsVals,
    const uint workPerThread,
    global BinaryFunction  *binaryFunct )
{
    size_t groId = get_group_id( 0 );
    size_t gloId = get_global_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    uint mapId  = gloId * workPerThread;

    // do offset of zero manually
    uint offset;
    int key;
    vType  workSum;

    if (mapId < vecSize)
    {
        int prevKey;

        // accumulate zeroth value manually
        offset = 0;
        key = keySumArray[ mapId+offset ];
        workSum = preSumArray[ mapId+offset ];
        postSumArray[ mapId+offset ] = workSum;

        //  Serial accumulation
        for( offset = offset+1; offset < workPerThread; offset += 1 )
        {
            prevKey = key;
            key = keySumArray[ mapId+offset ];
            if (mapId+offset<vecSize )
            {
                vType  y = preSumArray[ mapId+offset ];
                if ( key == prevKey )
                {
                    workSum = (*binaryFunct)( workSum, y );
                }
                else
                {
                    workSum = y;
                }
                postSumArray[ mapId+offset ] = workSum;
            }
        }
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    vType scanSum = workSum;
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
                vType y = ldsVals[ locId - offset ];
                int key1 = ldsKeys[ locId ];
                int key2 = ldsKeys[ locId-offset ];
                if ( key1 == key2 )
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
            vType  y = postSumArray[ mapId+offset ];
            int key1 = keySumArray[ mapId+offset ]; // change me
            int key2 = ldsKeys[ locId-1 ];
            if ( key1 == key2 )
            {
                vType  y2 = ldsVals[locId-1];
                y = (*binaryFunct)( y, y2 );
            }
            postSumArray[ mapId+offset ] = y;
        } // thread in bounds
    } // for

} // end kernel


/******************************************************************************
 *  Kernel 3
 *****************************************************************************/
 template<
    typename kType,
    typename kIterType,
    typename koType,
    typename koIterType,
	typename vType,
    typename vIterType,
    typename voType,
    typename voIterType,
	 typename BinaryFunction >
__kernel void keyValueMapping(
    global kType *ikeys,
    kIterType keys,
    global koType *ikeys_output,
    koIterType keys_output,
	global vType  *ivals, //input values
    vIterType vals,
    global voType *ivals_output,
    voIterType vals_output,
	local int *ldsKeys,
    local typename vIterType::value_type  *ldsVals,
	global int *newkeys,
	global int *keySumArray, //InputBuffer
    global typename vIterType::value_type *postSumArray, //InputBuffer
    const uint vecSize,
	global BinaryFunction *binaryFunct   )
{
    keys.init( ikeys );
	keys_output.init( ikeys_output );
    vals.init( ivals );
    vals_output.init( ivals_output );

    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    // if exclusive, load gloId=0 w/ init, and all others shifted-1
    int key;
    typename vIterType::value_type  val;

    if(gloId < vecSize){
      key = newkeys[ gloId ];
      val = vals[ gloId ];
      ldsKeys[ locId ] = key;
      ldsVals[ locId ] = val;
    }
    // Computes a scan within a workgroup
    // updates vals in lds but not keys
        typename vIterType::value_type  sum = val;
    for( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        int key2 = ldsKeys[locId - offset];
        if (locId >= offset && key == key2)
        {
                typename vIterType::value_type  y = ldsVals[ locId - offset ];
            sum = (*binaryFunct)( sum, y );
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        ldsVals[ locId ] = sum;
    }
    barrier( CLK_LOCAL_MEM_FENCE ); // needed for large data types

    //  Abort threads that are passed the end of the input vector
    if (gloId >= vecSize) return;

	 // accumulate prefix
    int  key1 =  keySumArray[ groId-1 ];
    int  key2 =  newkeys[ gloId ];
    int  key3 = -1;
    if(gloId < vecSize -1 )
      key3 =  newkeys[ gloId + 1];
    if (groId > 0 && key1 == key2 && key2 != key3)
    {
        typename vIterType::value_type  scanResult = sum;
        typename vIterType::value_type  postBlockSum = postSumArray[ groId-1 ];
        typename vIterType::value_type  newResult = (*binaryFunct)( scanResult, postBlockSum );
        sum = newResult;

    }
	unsigned int count_number_of_sections = 0;		
    count_number_of_sections = newkeys[vecSize-1] + 1;
    if(gloId < (vecSize-1) && newkeys[ gloId ] != newkeys[ gloId +1])
    {
        keys_output [newkeys [ gloId ]] = keys[ gloId];
        vals_output[ newkeys [ gloId ]] = sum;
    }
	//printf("keys_output %d vals_output %d newkeys [ gloId ]: %d\n", keys_output [newkeys [ gloId ]], vals_output[ newkeys [ gloId ]], newkeys [ gloId ]);
    if( gloId == (vecSize-1) )
    {
        keys_output[ count_number_of_sections - 1 ] = keys[ gloId ]; //Copying the last key directly. Works either ways
        vals_output[ count_number_of_sections - 1 ] = sum;
        newkeys [ gloId ] = count_number_of_sections;
    }

}
