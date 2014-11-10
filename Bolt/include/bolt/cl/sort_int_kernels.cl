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

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable 
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define WG_SIZE                 256
#define ELEMENTS_PER_WORK_ITEM  4
#define RADICES                 16
#define CHECK_BOUNDARY

typedef unsigned int u32;
#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)
#define GET_GROUP_SIZE get_local_size(0)
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)

#define AtomInc(x) atom_inc(&(x))
#define AtomInc1(x, out) out = atom_inc(&(x))
#define AtomAdd(x, value) atom_add(&(x), value)
#define USE_2LEVEL_REDUCE 1
#define SELECT_UINT4( b, a, condition ) select( b,a,condition )
#define SET_HISTOGRAM(setIdx, key) ldsSortData[(setIdx)*NUM_BUCKET+key]

#define make_uint4 (uint4)
#define make_uint2 (uint2)
#define make_int2 (int2)
#define m_n             x
#define m_nWGs          y
#define m_startBit      z
#define m_nBlocksPerWG  w

#define WG_SIZE 256
//#define ELEMENTS_PER_WORK_ITEM (1024/WG_SIZE)
#define BITS_PER_PASS 4
#define NUM_BUCKET (1<<BITS_PER_PASS)

    // __attribute__((reqd_work_group_size((powers of 2),1,1)))
uint scanLocalAndReduce(uint val, __local uint* lmem, uint *totalSum, int exclusive)
{
    // Set first half of local memory to zero to make room for scanning
    int lIdx = get_local_id(0);
    int wgSize = get_local_size(0);
    lmem[lIdx] = 0;
    
    lIdx += wgSize;
    lmem[lIdx] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Now, perform Kogge-Stone scan
    uint t;
    for (int i = 1; i < wgSize; i *= 2)
    {
        t = lmem[lIdx -  i]; 
        barrier(CLK_LOCAL_MEM_FENCE);
        lmem[lIdx] += t;     
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    *totalSum = lmem[wgSize*2 - 1];
    return lmem[lIdx-exclusive];
}

uint prefixScanVectorEx( uint4* data )
{
    u32 sum = 0;
    u32 tmp = data[0].x;
    data[0].x = sum;
    sum += tmp;
    tmp = data[0].y;
    data[0].y = sum;
    sum += tmp;
    tmp = data[0].z;
    data[0].z = sum;
    sum += tmp;
    tmp = data[0].w;
    data[0].w = sum;
    sum += tmp;
    return sum;
}

//__attribute__((reqd_work_group_size(256,1,1)))
uint4 localPrefixSum256V( uint4 pData, uint lIdx, uint* totalSum, __local u32* sorterSharedMemory )
{
    u32 s4 = prefixScanVectorEx( &pData );
    u32 rank = scanLocalAndReduce( s4, sorterSharedMemory, totalSum,  1 );
    return pData + make_uint4( rank, rank, rank, rank );
}

void sort4BitsSignedAscending(u32 sortData[4], int startBit, int lIdx, __local u32* ldsSortData)
{
    //printf("before sort lid = %d - %x %x %x %x \n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);
        u32 signedints[4];

        signedints[0] = ( ( ( (sortData[0] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[0] >> startBit) & (1<<3));
        signedints[1] = ( ( ( (sortData[1] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[1] >> startBit) & (1<<3));
        signedints[2] = ( ( ( (sortData[2] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[2] >> startBit) & (1<<3));
        signedints[3] = ( ( ( (sortData[3] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[3] >> startBit) & (1<<3));
    //printf("signed sort lid = %d - %x %x %x %x \n", lIdx, signedints[0], signedints[1], signedints[2], signedints[3]);
    for(int bitIdx=0; bitIdx<BITS_PER_PASS; bitIdx++)
    {
        u32 mask = (1<<bitIdx);
        uint4 cmpResult = make_uint4( (signedints[0]) & mask, (signedints[1]) & mask, (signedints[2]) & mask, (signedints[3]) & mask );
        //printf("cmpResult        lid = %d - %x %x %x %x \n",lIdx, cmpResult.x, cmpResult.y, cmpResult.z, cmpResult.w);
        // Note that the mask and 0 is the opposite in the case of unsigned ints.
#if defined(DESCENDING) 
        uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(0,0,0,0) );
#else
        uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(mask,mask,mask,mask) );
#endif
        //printf("Before prefixSum lid = %d - %x %x %x %x \n",lIdx, prefixSum.x, prefixSum.y, prefixSum.z, prefixSum.w);
        u32 total;
        prefixSum = localPrefixSum256V( prefixSum, lIdx, &total, ldsSortData );
        //printf("total = %d\n", total);
        //printf("After  prefixSum lid = %d - %x %x %x %x \n",lIdx, prefixSum.x, prefixSum.y, prefixSum.z, prefixSum.w);
        {
            uint4 localAddr = make_uint4(lIdx*4+0,lIdx*4+1,lIdx*4+2,lIdx*4+3);
            uint4 dstAddr = localAddr - prefixSum + make_uint4( total, total, total, total );
            // Note that the mask and 0 is the opposite in the case of unsigned ints.
#if defined(DESCENDING) 
            dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(0,0,0,0) );
#else
            dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(mask,mask,mask,mask) );
#endif
        //printf("dstAddr          lid = %d - %x %x %x %x \n",lIdx, dstAddr.x, dstAddr.y, dstAddr.z, dstAddr.w);
            GROUP_LDS_BARRIER;

            ldsSortData[dstAddr.x] = sortData[0];
            ldsSortData[dstAddr.y] = sortData[1];
            ldsSortData[dstAddr.z] = sortData[2];
            ldsSortData[dstAddr.w] = sortData[3];

            GROUP_LDS_BARRIER;

            sortData[0] = ldsSortData[localAddr.x];
            sortData[1] = ldsSortData[localAddr.y];
            sortData[2] = ldsSortData[localAddr.z];
            sortData[3] = ldsSortData[localAddr.w];

            GROUP_LDS_BARRIER;

            ldsSortData[dstAddr.x] = signedints[0];
            ldsSortData[dstAddr.y] = signedints[1];
            ldsSortData[dstAddr.z] = signedints[2];
            ldsSortData[dstAddr.w] = signedints[3];

            GROUP_LDS_BARRIER;

            signedints[0] = ldsSortData[localAddr.x];
            signedints[1] = ldsSortData[localAddr.y];
            signedints[2] = ldsSortData[localAddr.z];
            signedints[3] = ldsSortData[localAddr.w];

            GROUP_LDS_BARRIER;
        }
    }
    //printf("Signedsort lid = %d - %x %x %x %x \n\n", lIdx, signedints[0], signedints[1], signedints[2], signedints[3]);
    //printf("After sort lid = %d - %x %x %x %x \n\n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);

}

__kernel
__attribute__((reqd_work_group_size(256,1,1)))
void permuteSignedAscInstantiated( __global const u32* restrict gSrc, __global const u32* rHistogram, __global u32* restrict gDst, int4  cb )
{
    __local u32 ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+WG_SIZE];
    __local u32 localHistogramToCarry[NUM_BUCKET];
    __local u32 localHistogram[NUM_BUCKET*2];

    u32 gIdx = GET_GLOBAL_IDX;
    u32 lIdx = GET_LOCAL_IDX;
    u32 wgIdx = GET_GROUP_IDX;
    u32 wgSize = GET_GROUP_SIZE;
    const int dataAlignment = 1024;
    const int n = cb.m_n;
    const int w_n = n + dataAlignment-(n%dataAlignment);

    const int nWGs = cb.m_nWGs;
    const int startBit     = cb.m_startBit;
    const int nBlocksPerWG = cb.m_nBlocksPerWG;

    if( lIdx < (NUM_BUCKET) )
    {
#if defined(DESCENDING)
        localHistogramToCarry[lIdx] = rHistogram[lIdx*nWGs + wgIdx]; 
#else
        localHistogramToCarry[lIdx] = rHistogram[lIdx*nWGs + wgIdx];
#endif
    }

    GROUP_LDS_BARRIER;

    const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;

    int nBlocks = w_n/blockSize - nBlocksPerWG*wgIdx;

    int addr = blockSize*nBlocksPerWG*wgIdx + ELEMENTS_PER_WORK_ITEM*lIdx;

    for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++, addr+=blockSize)
    {
        u32 myHistogram = 0;

        u32 sortData[ELEMENTS_PER_WORK_ITEM];
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
#if defined(CHECK_BOUNDARY)
#if defined(DESCENDING)
            sortData[i] = ( addr+i < n )? gSrc[ addr+i ] : 0x80000000;
#else
            sortData[i] = ( addr+i < n )? gSrc[ addr+i ] : 0x7fffffff;
#endif
#else
            sortData[i] = gSrc[ addr+i ];
#endif
        //printf("before sort lid = %d - %x %x %x %x \n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);
#if defined(DESCENDING)
        sort4BitsSignedDescending(sortData, startBit, lIdx, ldsSortData);
#else
        sort4BitsSignedAscending(sortData, startBit, lIdx, ldsSortData);
#endif

        ////printf("Before sort lid = %d - %x %x %x %x \n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);
        //printf("After sort lid = %d - %x %x %x %x \n", lIdx, ldsSortData[lIdx*ELEMENTS_PER_WORK_ITEM], 
        //                                                     ldsSortData[lIdx*ELEMENTS_PER_WORK_ITEM+1], 
        //                                                     ldsSortData[lIdx*ELEMENTS_PER_WORK_ITEM+2], 
        //                                                     ldsSortData[lIdx*ELEMENTS_PER_WORK_ITEM+3] );
        u32 keys[ELEMENTS_PER_WORK_ITEM];
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
        {
            //keys[i] = (sortData[i]>>startBit) & 0xf;
#if defined(DESCENDING)
            keys[i] = 0xF - (( ( ( (sortData[i] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[i] >> startBit) & (1<<3)) );
#else
            keys[i] = 0xF - (( ( ( (sortData[i] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[i] >> startBit) & (1<<3)) );
#endif
        }
        //printf("After sort lid = %d - %x %x %x %x \n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);

        {	
            u32 setIdx = lIdx/16;
            if( lIdx < NUM_BUCKET )
            {
                localHistogram[lIdx] = 0;
            }
            ldsSortData[lIdx] = 0;
            GROUP_LDS_BARRIER;

            for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
#if defined(CHECK_BOUNDARY)
                if( addr+i < n )
#endif
#if defined(NV_GPU)
                SET_HISTOGRAM( setIdx, keys[i] )++;
#else
#if defined (DESCENDING)
                AtomInc( SET_HISTOGRAM( setIdx, (NUM_BUCKET - keys[i] - 1) ) );
#else
                AtomInc( SET_HISTOGRAM( setIdx, keys[i] ) );
#endif
#endif

            GROUP_LDS_BARRIER;

            uint hIdx = NUM_BUCKET+lIdx;
            if( lIdx < NUM_BUCKET )
            {
                u32 sum = 0;
                for(int i=0; i<WG_SIZE/16; i++)
                {
                    sum += SET_HISTOGRAM( i, lIdx );
                }
                myHistogram = sum;
                localHistogram[hIdx] = sum;
            }
            GROUP_LDS_BARRIER;

#if defined(USE_2LEVEL_REDUCE)
            if( lIdx < NUM_BUCKET )
            {
                localHistogram[hIdx] = localHistogram[hIdx-1];
                GROUP_MEM_FENCE;

                u32 u0, u1, u2;
                u0 = localHistogram[hIdx-3];
                u1 = localHistogram[hIdx-2];
                u2 = localHistogram[hIdx-1];
                AtomAdd( localHistogram[hIdx], u0 + u1 + u2 );
                GROUP_MEM_FENCE;
                u0 = localHistogram[hIdx-12];
                u1 = localHistogram[hIdx-8];
                u2 = localHistogram[hIdx-4];
                AtomAdd( localHistogram[hIdx], u0 + u1 + u2 );
                GROUP_MEM_FENCE;
            }
#else
            if( lIdx < NUM_BUCKET )
            {
                localHistogram[hIdx] = localHistogram[hIdx-1];
                GROUP_MEM_FENCE;
                localHistogram[hIdx] += localHistogram[hIdx-1];
                GROUP_MEM_FENCE;
                localHistogram[hIdx] += localHistogram[hIdx-2];
                GROUP_MEM_FENCE;
                localHistogram[hIdx] += localHistogram[hIdx-4];
                GROUP_MEM_FENCE;
                localHistogram[hIdx] += localHistogram[hIdx-8];
                GROUP_MEM_FENCE;
            }
#endif
            GROUP_LDS_BARRIER;
        }

        {
            for(int ie=0; ie<ELEMENTS_PER_WORK_ITEM; ie++)
            {
                int dataIdx = ELEMENTS_PER_WORK_ITEM*lIdx+ie;
#if defined (DESCENDING)
                int binIdx = 0xF - keys[ie];
                int groupOffset = localHistogramToCarry[binIdx];
#else
                int binIdx = keys[ie];
                int groupOffset = localHistogramToCarry[binIdx];
#endif
                //printf("gid = %d , binIdx=%d,  hist = %d\n", gIdx, binIdx, groupOffset);
#if defined (DESCENDING)
                int myIdx = dataIdx - localHistogram[NUM_BUCKET + binIdx ];
#else
                int myIdx = dataIdx - localHistogram[NUM_BUCKET + binIdx];
#endif

#if defined(CHECK_BOUNDARY)
                if( addr+ie < n )
#endif
                   gDst[ groupOffset + myIdx ] = sortData[ie];
                    //printf ( "(addr+ie)=%d, sortData[ie]=%d, groupOffset = %d, binIdx=%d, myIdx=%d, dataIdx=%d, localHistogram[NUM_BUCKET+binIdx]=%d,  (groupOffset + myIdx) = %d\n", 
                    //     addr+ie, sortData[ie], groupOffset, binIdx,  myIdx, dataIdx, localHistogram[NUM_BUCKET  + binIdx ], groupOffset + myIdx );
                
            }
        }

        GROUP_LDS_BARRIER;

        if( lIdx < NUM_BUCKET )
        {
            localHistogramToCarry[lIdx] += myHistogram;
        }
        GROUP_LDS_BARRIER;
    }
}

//The code above and the code below are the same. 
//Only difference is that DESCENDING is defined here.

#define DESCENDING

void sort4BitsSignedDescending(u32 sortData[4], int startBit, int lIdx, __local u32* ldsSortData)
{
    //printf("before sort lid = %d - %x %x %x %x \n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);
        u32 signedints[4];

        signedints[0] = ( ( ( (sortData[0] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[0] >> startBit) & (1<<3));
        signedints[1] = ( ( ( (sortData[1] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[1] >> startBit) & (1<<3));
        signedints[2] = ( ( ( (sortData[2] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[2] >> startBit) & (1<<3));
        signedints[3] = ( ( ( (sortData[3] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[3] >> startBit) & (1<<3));
    //printf("signed sort lid = %d - %x %x %x %x \n", lIdx, signedints[0], signedints[1], signedints[2], signedints[3]);
    for(int bitIdx=0; bitIdx<BITS_PER_PASS; bitIdx++)
    {
        u32 mask = (1<<bitIdx);
        uint4 cmpResult = make_uint4( (signedints[0]) & mask, (signedints[1]) & mask, (signedints[2]) & mask, (signedints[3]) & mask );
        //printf("cmpResult        lid = %d - %x %x %x %x \n",lIdx, cmpResult.x, cmpResult.y, cmpResult.z, cmpResult.w);
        // Note that the mask and 0 is the opposite in the case of unsigned ints.
#if defined(DESCENDING) 
        uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(0,0,0,0) );
#else
        uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(mask,mask,mask,mask) );
#endif
        //printf("Before prefixSum lid = %d - %x %x %x %x \n",lIdx, prefixSum.x, prefixSum.y, prefixSum.z, prefixSum.w);
        u32 total;
        prefixSum = localPrefixSum256V( prefixSum, lIdx, &total, ldsSortData );
        //printf("total = %d\n", total);
        //printf("After  prefixSum lid = %d - %x %x %x %x \n",lIdx, prefixSum.x, prefixSum.y, prefixSum.z, prefixSum.w);
        {
            uint4 localAddr = make_uint4(lIdx*4+0,lIdx*4+1,lIdx*4+2,lIdx*4+3);
            uint4 dstAddr = localAddr - prefixSum + make_uint4( total, total, total, total );
            // Note that the mask and 0 is the opposite in the case of unsigned ints.
#if defined(DESCENDING) 
            dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(0,0,0,0) );
#else
            dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(mask,mask,mask,mask) );
#endif
        //printf("dstAddr          lid = %d - %x %x %x %x \n",lIdx, dstAddr.x, dstAddr.y, dstAddr.z, dstAddr.w);
            GROUP_LDS_BARRIER;

            ldsSortData[dstAddr.x] = sortData[0];
            ldsSortData[dstAddr.y] = sortData[1];
            ldsSortData[dstAddr.z] = sortData[2];
            ldsSortData[dstAddr.w] = sortData[3];

            GROUP_LDS_BARRIER;

            sortData[0] = ldsSortData[localAddr.x];
            sortData[1] = ldsSortData[localAddr.y];
            sortData[2] = ldsSortData[localAddr.z];
            sortData[3] = ldsSortData[localAddr.w];

            GROUP_LDS_BARRIER;

            ldsSortData[dstAddr.x] = signedints[0];
            ldsSortData[dstAddr.y] = signedints[1];
            ldsSortData[dstAddr.z] = signedints[2];
            ldsSortData[dstAddr.w] = signedints[3];

            GROUP_LDS_BARRIER;

            signedints[0] = ldsSortData[localAddr.x];
            signedints[1] = ldsSortData[localAddr.y];
            signedints[2] = ldsSortData[localAddr.z];
            signedints[3] = ldsSortData[localAddr.w];

            GROUP_LDS_BARRIER;
        }
    }
    //printf("Signedsort lid = %d - %x %x %x %x \n\n", lIdx, signedints[0], signedints[1], signedints[2], signedints[3]);
    //printf("After sort lid = %d - %x %x %x %x \n\n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);

}

__kernel
__attribute__((reqd_work_group_size(256,1,1)))
void permuteSignedDescInstantiated( __global const u32* restrict gSrc, __global const u32* rHistogram, __global u32* restrict gDst, int4  cb )
{
    __local u32 ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+WG_SIZE];
    __local u32 localHistogramToCarry[NUM_BUCKET];
    __local u32 localHistogram[NUM_BUCKET*2];

    u32 gIdx = GET_GLOBAL_IDX;
    u32 lIdx = GET_LOCAL_IDX;
    u32 wgIdx = GET_GROUP_IDX;
    u32 wgSize = GET_GROUP_SIZE;
    const int dataAlignment = 1024;
    const int n = cb.m_n;
    const int w_n = n + dataAlignment-(n%dataAlignment);

    const int nWGs = cb.m_nWGs;
    const int startBit     = cb.m_startBit;
    const int nBlocksPerWG = cb.m_nBlocksPerWG;

    if( lIdx < (NUM_BUCKET) )
    {
#if defined(DESCENDING)
        localHistogramToCarry[lIdx] = rHistogram[lIdx*nWGs + wgIdx]; 
#else
        localHistogramToCarry[lIdx] = rHistogram[lIdx*nWGs + wgIdx];
#endif
    }

    GROUP_LDS_BARRIER;

    const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;

    int nBlocks = w_n/blockSize - nBlocksPerWG*wgIdx;

    int addr = blockSize*nBlocksPerWG*wgIdx + ELEMENTS_PER_WORK_ITEM*lIdx;

    for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++, addr+=blockSize)
    {
        u32 myHistogram = 0;

        u32 sortData[ELEMENTS_PER_WORK_ITEM];
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
#if defined(CHECK_BOUNDARY)
#if defined(DESCENDING)
            sortData[i] = ( addr+i < n )? gSrc[ addr+i ] : 0x80000000;
#else
            sortData[i] = ( addr+i < n )? gSrc[ addr+i ] : 0x7fffffff;
#endif
#else
            sortData[i] = gSrc[ addr+i ];
#endif
        //printf("before sort lid = %d - %x %x %x %x \n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);
#if defined(DESCENDING)
        sort4BitsSignedDescending(sortData, startBit, lIdx, ldsSortData);
#else
        sort4BitsSignedAscending(sortData, startBit, lIdx, ldsSortData);
#endif

        ////printf("Before sort lid = %d - %x %x %x %x \n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);
        //printf("After sort lid = %d - %x %x %x %x \n", lIdx, ldsSortData[lIdx*ELEMENTS_PER_WORK_ITEM], 
        //                                                     ldsSortData[lIdx*ELEMENTS_PER_WORK_ITEM+1], 
        //                                                     ldsSortData[lIdx*ELEMENTS_PER_WORK_ITEM+2], 
        //                                                     ldsSortData[lIdx*ELEMENTS_PER_WORK_ITEM+3] );
        u32 keys[ELEMENTS_PER_WORK_ITEM];
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
        {
            //keys[i] = (sortData[i]>>startBit) & 0xf;
#if defined(DESCENDING)
            keys[i] = 0xF - (( ( ( (sortData[i] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[i] >> startBit) & (1<<3)) );
#else
            keys[i] = 0xF - (( ( ( (sortData[i] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[i] >> startBit) & (1<<3)) );
#endif
        }
        //printf("After sort lid = %d - %x %x %x %x \n", lIdx, sortData[0], sortData[1], sortData[2], sortData[3]);

        {	
            u32 setIdx = lIdx/16;
            if( lIdx < NUM_BUCKET )
            {
                localHistogram[lIdx] = 0;
            }
            ldsSortData[lIdx] = 0;
            GROUP_LDS_BARRIER;

            for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
#if defined(CHECK_BOUNDARY)
                if( addr+i < n )
#endif
#if defined(NV_GPU)
                SET_HISTOGRAM( setIdx, keys[i] )++;
#else
#if defined (DESCENDING)
                AtomInc( SET_HISTOGRAM( setIdx, (NUM_BUCKET - keys[i] - 1) ) );
#else
                AtomInc( SET_HISTOGRAM( setIdx, keys[i] ) );
#endif
#endif

            GROUP_LDS_BARRIER;

            uint hIdx = NUM_BUCKET+lIdx;
            if( lIdx < NUM_BUCKET )
            {
                u32 sum = 0;
                for(int i=0; i<WG_SIZE/16; i++)
                {
                    sum += SET_HISTOGRAM( i, lIdx );
                }
                myHistogram = sum;
                localHistogram[hIdx] = sum;
            }
            GROUP_LDS_BARRIER;

#if defined(USE_2LEVEL_REDUCE)
            if( lIdx < NUM_BUCKET )
            {
                localHistogram[hIdx] = localHistogram[hIdx-1];
                GROUP_MEM_FENCE;

                u32 u0, u1, u2;
                u0 = localHistogram[hIdx-3];
                u1 = localHistogram[hIdx-2];
                u2 = localHistogram[hIdx-1];
                AtomAdd( localHistogram[hIdx], u0 + u1 + u2 );
                GROUP_MEM_FENCE;
                u0 = localHistogram[hIdx-12];
                u1 = localHistogram[hIdx-8];
                u2 = localHistogram[hIdx-4];
                AtomAdd( localHistogram[hIdx], u0 + u1 + u2 );
                GROUP_MEM_FENCE;
            }
#else
            if( lIdx < NUM_BUCKET )
            {
                localHistogram[hIdx] = localHistogram[hIdx-1];
                GROUP_MEM_FENCE;
                localHistogram[hIdx] += localHistogram[hIdx-1];
                GROUP_MEM_FENCE;
                localHistogram[hIdx] += localHistogram[hIdx-2];
                GROUP_MEM_FENCE;
                localHistogram[hIdx] += localHistogram[hIdx-4];
                GROUP_MEM_FENCE;
                localHistogram[hIdx] += localHistogram[hIdx-8];
                GROUP_MEM_FENCE;
            }
#endif
            GROUP_LDS_BARRIER;
        }

        {
            for(int ie=0; ie<ELEMENTS_PER_WORK_ITEM; ie++)
            {
                int dataIdx = ELEMENTS_PER_WORK_ITEM*lIdx+ie;
#if defined (DESCENDING)
                int binIdx = 0xF - keys[ie];
                int groupOffset = localHistogramToCarry[binIdx];
#else
                int binIdx = keys[ie];
                int groupOffset = localHistogramToCarry[binIdx];
#endif
                //printf("gid = %d , binIdx=%d,  hist = %d\n", gIdx, binIdx, groupOffset);
#if defined (DESCENDING)
                int myIdx = dataIdx - localHistogram[NUM_BUCKET + binIdx ];
#else
                int myIdx = dataIdx - localHistogram[NUM_BUCKET + binIdx];
#endif

#if defined(CHECK_BOUNDARY)
                if( addr+ie < n )
#endif
                   gDst[ groupOffset + myIdx ] = sortData[ie];
                    //printf ( "(addr+ie)=%d, sortData[ie]=%d, groupOffset = %d, binIdx=%d, myIdx=%d, dataIdx=%d, localHistogram[NUM_BUCKET+binIdx]=%d,  (groupOffset + myIdx) = %d\n", 
                    //     addr+ie, sortData[ie], groupOffset, binIdx,  myIdx, dataIdx, localHistogram[NUM_BUCKET  + binIdx ], groupOffset + myIdx );
                
            }
        }

        GROUP_LDS_BARRIER;

        if( lIdx < NUM_BUCKET )
        {
            localHistogramToCarry[lIdx] += myHistogram;
        }
        GROUP_LDS_BARRIER;
    }
}
