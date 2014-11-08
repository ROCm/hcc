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
#define m_n        x
#define m_nWGs     y
#define m_startBit z
#define m_nBlocksPerWG  w

#define WG_SIZE 256
#define BITS_PER_PASS 4
#define NUM_BUCKET (1<<BITS_PER_PASS)

uint scanLocalMemAndTotal(uint val, __local uint* lmem, uint *totalSum, int exclusive)
{
    // Set first half of local memory to zero to make room for scanning
    int l_id = get_local_id(0);
    int l_size = get_local_size(0);
    lmem[l_id] = 0;
    
    l_id += l_size;
    lmem[l_id] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    uint t;
    for (int i = 1; i < l_size; i *= 2)
    {
        t = lmem[l_id -  i]; 
        barrier(CLK_LOCAL_MEM_FENCE);
        lmem[l_id] += t;     
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    *totalSum = lmem[l_size*2 - 1];
    return lmem[l_id-exclusive];
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

uint4 localPrefixSum256V( uint4 pData, uint lIdx, uint* totalSum, __local u32* sorterSharedMemory )
{
    u32 s4 = prefixScanVectorEx( &pData );
    u32 rank = scanLocalMemAndTotal( s4, sorterSharedMemory, totalSum,  1 );
    return pData + make_uint4( rank, rank, rank, rank );
}

template <typename Values>
void sort4BitsSignedKeyValueAscending(u32 sortData[4], Values sortVal[4], const int startBit, int lIdx, __local u32* ldsSortData, __local Values *ldsSortVal)
{   
  u32 signedints[4];

  signedints[0] = ( ( ( (sortData[0] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[0] >> startBit) & (1<<3));
  signedints[1] = ( ( ( (sortData[1] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[1] >> startBit) & (1<<3));
  signedints[2] = ( ( ( (sortData[2] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[2] >> startBit) & (1<<3));
  signedints[3] = ( ( ( (sortData[3] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[3] >> startBit) & (1<<3));

   for(int bitIdx=0; bitIdx<BITS_PER_PASS; bitIdx++)
   {
   u32 mask = (1<<bitIdx);
   uint4 cmpResult = make_uint4( (signedints[0]) & mask, (signedints[1]) & mask, (signedints[2]) & mask, (signedints[3]) & mask );

#if defined(DESCENDING) 
        uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(0,0,0,0) );
#else
        uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(mask,mask,mask,mask) );
#endif
  u32 total;
  prefixSum = localPrefixSum256V( prefixSum, lIdx, &total, ldsSortData );

  {
    uint4 localAddr = make_uint4(lIdx*4+0,lIdx*4+1,lIdx*4+2,lIdx*4+3);
    uint4 dstAddr = localAddr - prefixSum + make_uint4( total, total, total, total );
#if defined(DESCENDING) 
    dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(0,0,0,0) );
#else
    dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(mask,mask,mask,mask) );
#endif
    GROUP_LDS_BARRIER;
    ldsSortData[dstAddr.x] = sortData[0];
    ldsSortData[dstAddr.y] = sortData[1];
    ldsSortData[dstAddr.z] = sortData[2];
    ldsSortData[dstAddr.w] = sortData[3];
    ldsSortVal[dstAddr.x] = sortVal[0];
    ldsSortVal[dstAddr.y] = sortVal[1];
    ldsSortVal[dstAddr.z] = sortVal[2];
    ldsSortVal[dstAddr.w] = sortVal[3];
    GROUP_LDS_BARRIER;
    sortData[0] = ldsSortData[localAddr.x];
    sortData[1] = ldsSortData[localAddr.y];
    sortData[2] = ldsSortData[localAddr.z];
    sortData[3] = ldsSortData[localAddr.w];
    sortVal[0] = ldsSortVal[localAddr.x];
    sortVal[1] = ldsSortVal[localAddr.y];
    sortVal[2] = ldsSortVal[localAddr.z];
    sortVal[3] = ldsSortVal[localAddr.w];
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
}

template <typename Values>
__kernel
void permuteByKeySignedAscTemplate( __global const u32* restrict gKeys, 
          __global const Values* restrict gValues, 
          __global const u32* rHistogram, 
          __global u32* restrict gDstKeys, 
          __global Values* restrict gDstValues, 
          int4 cb)
{
    __local u32 ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+WG_SIZE];
    __local Values ldsSortVal[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];
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
        Values sortVal[ELEMENTS_PER_WORK_ITEM];
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
        {
#if defined(CHECK_BOUNDARY)
#if defined(DESCENDING)
            sortData[i] = ( addr+i < n )? gKeys[ addr+i ] : 0x80000000;
            sortVal[i]  = ( addr+i < n )? gValues[ addr+i ] : 0x80000000;
#else
            sortData[i] = ( addr+i < n )? gKeys[ addr+i ] : 0x7fffffff;
            sortVal[i]  = ( addr+i < n )? gValues[ addr+i ] : 0x7fffffff;
#endif
#else
            sortData[i] = gKeys[ addr+i ];
            sortVal[i]  = gValues[ addr+i ];
#endif
        }

        sort4BitsSignedKeyValueAscending(sortData, sortVal, startBit, lIdx, ldsSortData, ldsSortVal);
        u32 keys[ELEMENTS_PER_WORK_ITEM];
        for (int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
        {
            keys[i] = 0xF - (( ( ( (sortData[i] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[i] >> startBit) & (1<<3)) );
        }

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
                int myIdx = dataIdx - localHistogram[NUM_BUCKET + binIdx];
#if defined(CHECK_BOUNDARY)
                if( addr+ie < n )
#endif
                {   
                    if ((groupOffset + myIdx)<n)
                    {
                        gDstKeys[ groupOffset + myIdx ]   = sortData[ie];
                        gDstValues[ groupOffset + myIdx ] = sortVal[ie];
                    }
                }
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


#define DESCENDING

template <typename Values>
void sort4BitsSignedKeyValueDescending(u32 sortData[4], Values sortVal[4], const int startBit, int lIdx, __local u32* ldsSortData, __local Values *ldsSortVal)
{   
  u32 signedints[4];

  signedints[0] = ( ( ( (sortData[0] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[0] >> startBit) & (1<<3));
  signedints[1] = ( ( ( (sortData[1] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[1] >> startBit) & (1<<3));
  signedints[2] = ( ( ( (sortData[2] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[2] >> startBit) & (1<<3));
  signedints[3] = ( ( ( (sortData[3] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[3] >> startBit) & (1<<3));

   for(int bitIdx=0; bitIdx<BITS_PER_PASS; bitIdx++)
   {
   u32 mask = (1<<bitIdx);
   uint4 cmpResult = make_uint4( (signedints[0]) & mask, (signedints[1]) & mask, (signedints[2]) & mask, (signedints[3]) & mask );

#if defined(DESCENDING) 
        uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(0,0,0,0) );
#else
        uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(mask,mask,mask,mask) );
#endif
  u32 total;
  prefixSum = localPrefixSum256V( prefixSum, lIdx, &total, ldsSortData );

  {
    uint4 localAddr = make_uint4(lIdx*4+0,lIdx*4+1,lIdx*4+2,lIdx*4+3);
    uint4 dstAddr = localAddr - prefixSum + make_uint4( total, total, total, total );
#if defined(DESCENDING) 
    dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(0,0,0,0) );
#else
    dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(mask,mask,mask,mask) );
#endif
    GROUP_LDS_BARRIER;
    ldsSortData[dstAddr.x] = sortData[0];
    ldsSortData[dstAddr.y] = sortData[1];
    ldsSortData[dstAddr.z] = sortData[2];
    ldsSortData[dstAddr.w] = sortData[3];
    ldsSortVal[dstAddr.x] = sortVal[0];
    ldsSortVal[dstAddr.y] = sortVal[1];
    ldsSortVal[dstAddr.z] = sortVal[2];
    ldsSortVal[dstAddr.w] = sortVal[3];
    GROUP_LDS_BARRIER;
    sortData[0] = ldsSortData[localAddr.x];
    sortData[1] = ldsSortData[localAddr.y];
    sortData[2] = ldsSortData[localAddr.z];
    sortData[3] = ldsSortData[localAddr.w];
    sortVal[0] = ldsSortVal[localAddr.x];
    sortVal[1] = ldsSortVal[localAddr.y];
    sortVal[2] = ldsSortVal[localAddr.z];
    sortVal[3] = ldsSortVal[localAddr.w];
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
}

template <typename Values>
__kernel
void permuteByKeySignedDescTemplate( __global const u32* restrict gKeys, 
          __global const Values* restrict gValues, 
          __global const u32* rHistogram, 
          __global u32* restrict gDstKeys, 
          __global Values* restrict gDstValues, 
          int4 cb)
{
    __local u32 ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+WG_SIZE];
    __local Values ldsSortVal[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];
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
        Values sortVal[ELEMENTS_PER_WORK_ITEM];
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
        {
#if defined(CHECK_BOUNDARY)
#if defined(DESCENDING)
            sortData[i] = ( addr+i < n )? gKeys[ addr+i ] : 0x80000000;
            sortVal[i]  = ( addr+i < n )? gValues[ addr+i ] : 0x80000000;
#else
            sortData[i] = ( addr+i < n )? gKeys[ addr+i ] : 0x7fffffff;
            sortVal[i]  = ( addr+i < n )? gValues[ addr+i ] : 0x7fffffff;
#endif
#else
            sortData[i] = gKeys[ addr+i ];
            sortVal[i]  = gValues[ addr+i ];
#endif
        }

        sort4BitsSignedKeyValueDescending(sortData, sortVal, startBit, lIdx, ldsSortData, ldsSortVal);
        u32 keys[ELEMENTS_PER_WORK_ITEM];
        for (int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
        {
            keys[i] = 0xF - (( ( ( (sortData[i] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[i] >> startBit) & (1<<3)) );
        }

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
                int myIdx = dataIdx - localHistogram[NUM_BUCKET + binIdx];
#if defined(CHECK_BOUNDARY)
                if( addr+ie < n )
#endif
                {   
                    if ((groupOffset + myIdx)<n)
                    {
                        gDstKeys[ groupOffset + myIdx ]   = sortData[ie];
                        gDstValues[ groupOffset + myIdx ] = sortVal[ie];
                    }
                }
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
