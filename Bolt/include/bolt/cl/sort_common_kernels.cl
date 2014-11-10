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

#define WG_SIZE                 256
#define ELEMENTS_PER_WORK_ITEM  4
#define RADICES                 16
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)
#define m_n             x
#define m_nWGs          y
#define m_startBit      z
#define m_nBlocksPerWG  w

#define CHECK_BOUNDARY


uint scanlMemPrivData(uint val, __local uint* lmem, int exclusive)
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
    return lmem[lIdx-exclusive];
}

__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void
scanInstantiated(__global uint * isums, 
         const int n,
         __local uint * lmem)
{
    __local int s_seed; 
    s_seed = 0; barrier(CLK_LOCAL_MEM_FENCE);
    
    int last_thread = (get_local_id(0) < n &&
                      (get_local_id(0)+1) == n) ? 1 : 0;
    //printf("top_scan n = %d\n", n);
    for (int d = 0; d < 16; d++)
    {
        uint val = 0;

        if (get_local_id(0) < n)
        {
            val = isums[(n * d) + get_local_id(0)];
        }
        // Exclusive scan the counts in local memory
        uint res = scanlMemPrivData(val, lmem, 1);
        // Write scanned value out to global
        if (get_local_id(0) < n)
        {
            isums[(n * d) + get_local_id(0)] = res + s_seed;
        }
        
        if (last_thread) 
        {
            s_seed += res + val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void
histogramAscInstantiated(__global const uint * in, 
       __global uint * isums, 
       int4  cb) 
{

    __local uint lmem[WG_SIZE*RADICES];

    uint gIdx = get_global_id(0);
    uint lIdx = get_local_id(0);
    uint wgIdx = get_group_id(0);
    uint wgSize = get_local_size(0);

    const int shift = cb.m_startBit;
    const int dataAlignment = 1024;
    const int n = cb.m_n;
    const int w_n = n + dataAlignment-(n%dataAlignment);

    const int nWGs = cb.m_nWGs;
    const int nBlocksPerWG = cb.m_nBlocksPerWG;

    for(int i=0; i<RADICES; i++)
    {
        lmem[i*get_local_size(0)+ lIdx] = 0;
    }
    GROUP_LDS_BARRIER;

    const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;

    int nBlocks = (w_n)/blockSize - nBlocksPerWG*wgIdx;
    int addr = blockSize*nBlocksPerWG*wgIdx + lIdx;
    for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++)
    {
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++, addr+=WG_SIZE )
        {
#if defined(CHECK_BOUNDARY)
            if( (addr) < n)
#endif
            {
                uint local_key = (in[addr] >> shift) & 0xFU;
#if defined(DESCENDING)
                lmem[(RADICES - local_key -1)*get_local_size(0)+ lIdx]++;   
#else
                lmem[local_key*get_local_size(0)+ lIdx]++;
#endif
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if( lIdx < RADICES )
    {
        uint sum = 0;
        for(int i=0; i<get_local_size(0); i++)
        {
            sum += lmem[lIdx*get_local_size(0)+ i];
        }
        isums[lIdx * get_num_groups(0) + get_group_id(0)] = sum;
    }
}

__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void
histogramSignedAscInstantiated(__global const uint * in, 
       __global uint * isums, 
       int4  cb) 
{

    __local uint lmem[WG_SIZE*RADICES];

    uint gIdx = get_global_id(0);
    uint lIdx = get_local_id(0);
    uint wgIdx = get_group_id(0);
    uint wgSize = get_local_size(0);

    const int shift = cb.m_startBit;
    const int dataAlignment = 1024;
    const int n = cb.m_n;
    const int w_n = n + dataAlignment-(n%dataAlignment);

    const int nWGs = cb.m_nWGs;
    const int nBlocksPerWG = cb.m_nBlocksPerWG;

    for(int i=0; i<RADICES; i++)
    {
        lmem[i*get_local_size(0)+ lIdx] = 0;
    }
    GROUP_LDS_BARRIER;

    const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;

    int nBlocks = (w_n)/blockSize - nBlocksPerWG*wgIdx;
    int addr = blockSize*nBlocksPerWG*wgIdx + lIdx;
    for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++)
    {
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++, addr+=WG_SIZE )
        {
#if defined(CHECK_BOUNDARY)
            if( (addr) < n)
#endif
            {
                uint local_key = (in[addr] >> shift);
                uint signBit   = local_key & (1<<3);
#if defined(DESCENDING)
                local_key = 0xF - (( ( ( local_key & 0x7 ) ^ 0x7 ) & 0x7 ) | signBit);
#else
                local_key = 0xF - (( ( ( local_key & 0x7 ) ^ 0x7 ) & 0x7 ) | signBit);
#endif

#if defined(DESCENDING)
                lmem[(RADICES - local_key -1)*get_local_size(0)+ lIdx]++;  
#else
                lmem[local_key*get_local_size(0)+ lIdx]++;
#endif
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if( lIdx < RADICES )
    {
        uint sum = 0;
        for(int i=0; i<get_local_size(0); i++)
        {
            sum += lmem[lIdx*get_local_size(0)+ i];
        }
        isums[lIdx * get_num_groups(0) + get_group_id(0)] = sum;
    }
}



#define DESCENDING
__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void
histogramDescInstantiated(__global const uint * in, 
       __global uint * isums, 
       int4  cb) 
{

    __local uint lmem[WG_SIZE*RADICES];

    uint gIdx = get_global_id(0);
    uint lIdx = get_local_id(0);
    uint wgIdx = get_group_id(0);
    uint wgSize = get_local_size(0);

    const int shift = cb.m_startBit;
    const int dataAlignment = 1024;
    const int n = cb.m_n;
    const int w_n = n + dataAlignment-(n%dataAlignment);

    const int nWGs = cb.m_nWGs;
    const int nBlocksPerWG = cb.m_nBlocksPerWG;

    for(int i=0; i<RADICES; i++)
    {
        lmem[i*get_local_size(0)+ lIdx] = 0;
    }
    GROUP_LDS_BARRIER;

    const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;

    int nBlocks = (w_n)/blockSize - nBlocksPerWG*wgIdx;
    int addr = blockSize*nBlocksPerWG*wgIdx + lIdx;
    for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++)
    {
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++, addr+=WG_SIZE )
        {
#if defined(CHECK_BOUNDARY)
            if( (addr) < n)
#endif
            {
                uint local_key = (in[addr] >> shift) & 0xFU;
#if defined(DESCENDING)
                lmem[(RADICES - local_key -1)*get_local_size(0)+ lIdx]++;   
#else
                lmem[local_key*get_local_size(0)+ lIdx]++;
#endif
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if( lIdx < RADICES )
    {
        uint sum = 0;
        for(int i=0; i<get_local_size(0); i++)
        {
            sum += lmem[lIdx*get_local_size(0)+ i];
        }
        isums[lIdx * get_num_groups(0) + get_group_id(0)] = sum;
    }
}

__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void
histogramSignedDescInstantiated(__global const uint * in, 
       __global uint * isums, 
       int4  cb) 
{

    __local uint lmem[WG_SIZE*RADICES];

    uint gIdx = get_global_id(0);
    uint lIdx = get_local_id(0);
    uint wgIdx = get_group_id(0);
    uint wgSize = get_local_size(0);

    const int shift = cb.m_startBit;
    const int dataAlignment = 1024;
    const int n = cb.m_n;
    const int w_n = n + dataAlignment-(n%dataAlignment);

    const int nWGs = cb.m_nWGs;
    const int nBlocksPerWG = cb.m_nBlocksPerWG;

    for(int i=0; i<RADICES; i++)
    {
        lmem[i*get_local_size(0)+ lIdx] = 0;
    }
    GROUP_LDS_BARRIER;

    const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;

    int nBlocks = (w_n)/blockSize - nBlocksPerWG*wgIdx;
    int addr = blockSize*nBlocksPerWG*wgIdx + lIdx;
    for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++)
    {
        for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++, addr+=WG_SIZE )
        {
#if defined(CHECK_BOUNDARY)
            if( (addr) < n)
#endif
            {
                uint local_key = (in[addr] >> shift);
                uint signBit   = local_key & (1<<3);
#if defined(DESCENDING)
                local_key = 0xF - (( ( ( local_key & 0x7 ) ^ 0x7 ) & 0x7 ) | signBit);
#else
                local_key = 0xF - (( ( ( local_key & 0x7 ) ^ 0x7 ) & 0x7 ) | signBit);
#endif

#if defined(DESCENDING)
                lmem[(RADICES - local_key -1)*get_local_size(0)+ lIdx]++;   
#else
                lmem[local_key*get_local_size(0)+ lIdx]++;
#endif
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if( lIdx < RADICES )
    {
        uint sum = 0;
        for(int i=0; i<get_local_size(0); i++)
        {
            sum += lmem[lIdx*get_local_size(0)+ i];
        }
        isums[lIdx * get_num_groups(0) + get_group_id(0)] = sum;
    }
}


