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

#if !defined( BOLT_AMP_SORT_INL )
#define BOLT_AMP_SORT_INL
#pragma once

#include <algorithm>
#include <type_traits>


#include "bolt/amp/bolt.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/device_vector.h"
#include <amp.h>
#include "bolt/amp/detail/stablesort.inl"
#include "bolt/amp/iterator/iterator_traits.h"
#include <amp_short_vectors.h>
#include "bolt/amp/detail/stablesort.inl"

#ifdef ENABLE_TBB
#include "bolt/btbb/sort.h"

#endif


#define WG_SIZE                 256
#define RADICES                 16
#define ELEMENTS_PER_WORK_ITEM  4
#define BITS_PER_PASS			4
#define NUM_BUCKET				(1<<BITS_PER_PASS)


#define AtomInc(x) concurrency::atomic_fetch_inc(&(x))
#define AtomAdd(x, value) concurrency::atomic_fetch_add(&(x), value)
#define USE_2LEVEL_REDUCE 
#define uint_4 Concurrency::graphics::uint_4
// On Linux. Macro is expanded in std::max
#define _max(a,b)    (((a) > (b)) ? (a) : (b))
#define _min(a,b)    (((a) < (b)) ? (a) : (b))
#define make_uint4 (uint_4)
#define SET_HISTOGRAM(setIdx, key) ldsSortData[(setIdx)*NUM_BUCKET+key]

namespace bolt {
namespace amp {
namespace detail {


	static inline uint_4 SELECT_UINT4(uint_4 &a,uint_4 &b,uint_4  &condition )  restrict(amp)
	{
		uint_4 res;
		res.x = (condition.x )? b.x : a.x;
		res.y = (condition.y )? b.y : a.y;
		res.z = (condition.z )? b.z : a.z;
		res.w = (condition.w )? b.w : a.w;
		return res;

	}
	static unsigned int scanLocalMemAndTotal(unsigned int val, unsigned int* lmem, unsigned int *totalSum, int exclusive, concurrency::tiled_index< WG_SIZE > t_idx) restrict(amp)
	{
		// Set first half of local memory to zero to make room for scanning
		int l_id = t_idx.local[ 0 ];
		int l_size = WG_SIZE;
		lmem[l_id] = 0;
    
		l_id += l_size;
		lmem[l_id] = val;
		t_idx.barrier.wait();
    
		unsigned int t;
		for (int i = 1; i < l_size; i *= 2)
		{
			t = lmem[l_id -  i]; 
			t_idx.barrier.wait();
			lmem[l_id] += t;     
			t_idx.barrier.wait();
		}
		*totalSum = lmem[l_size*2 - 1];
		return lmem[l_id-exclusive];
	}
	static unsigned int prefixScanVectorEx( uint_4* data ) restrict(amp)
	{
		unsigned int sum = 0;
		unsigned int tmp = data[0].x;
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
	static uint_4 localPrefixSum256V( uint_4 pData, unsigned int lIdx, unsigned int* totalSum, unsigned int* sorterSharedMemory, concurrency::tiled_index< WG_SIZE > t_idx ) restrict(amp)
	{
		unsigned int s4 = prefixScanVectorEx( &pData );
		unsigned int rank = scanLocalMemAndTotal( s4, sorterSharedMemory, totalSum,  1, t_idx);
		return pData + make_uint4( rank, rank, rank, rank );
	}
	static void sort4BitsKeyValueAscending(unsigned int sortData[4],  const int startBit, int lIdx,  unsigned int* ldsSortData,  bool Asc_sort, concurrency::tiled_index< WG_SIZE > t_idx) restrict(amp)
	{
		for(int bitIdx=0; bitIdx<BITS_PER_PASS; bitIdx++)
		{
			unsigned int mask = (1<<bitIdx);
			uint_4 prefixSum;
			uint_4 cmpResult( (sortData[0]>>startBit) & mask, (sortData[1]>>startBit) & mask, (sortData[2]>>startBit) & mask, (sortData[3]>>startBit) & mask );
			uint_4 temp;

			if(!Asc_sort)
			{
				temp.x = (cmpResult.x != mask);
				temp.y = (cmpResult.y != mask);
				temp.z = (cmpResult.z != mask);
				temp.w = (cmpResult.w != mask);
			}
			else
			{
				temp.x = (cmpResult.x != 0);
				temp.y = (cmpResult.y != 0);
				temp.z = (cmpResult.z != 0);
				temp.w = (cmpResult.w != 0);
			}
                        uint_4 arg1 = make_uint4(1,1,1,1);
                        uint_4 arg2 = make_uint4(0,0,0,0);
			prefixSum = SELECT_UINT4( arg1, arg2, temp );//(cmpResult != make_uint4(mask,mask,mask,mask)));

			unsigned int total = 0;
			prefixSum = localPrefixSum256V( prefixSum, lIdx, &total, ldsSortData, t_idx);
			{
				uint_4 localAddr(lIdx*4+0,lIdx*4+1,lIdx*4+2,lIdx*4+3);
				uint_4 dstAddr = localAddr - prefixSum + make_uint4( total, total, total, total );
				if(!Asc_sort)
				{
					temp.x = (cmpResult.x != mask);
					temp.y = (cmpResult.y != mask);
					temp.z = (cmpResult.z != mask);
					temp.w = (cmpResult.w != mask);
					}
				else
				{
					temp.x = (cmpResult.x != 0);
					temp.y = (cmpResult.y != 0);
					temp.z = (cmpResult.z != 0);
					temp.w = (cmpResult.w != 0);
				}
				dstAddr = SELECT_UINT4( prefixSum, dstAddr, temp);

				t_idx.barrier.wait();
        
				ldsSortData[dstAddr.x] = sortData[0];
				ldsSortData[dstAddr.y] = sortData[1];
				ldsSortData[dstAddr.z] = sortData[2];
				ldsSortData[dstAddr.w] = sortData[3];

				t_idx.barrier.wait();

				sortData[0] = ldsSortData[localAddr.x];
				sortData[1] = ldsSortData[localAddr.y];
				sortData[2] = ldsSortData[localAddr.z];
				sortData[3] = ldsSortData[localAddr.w];

				t_idx.barrier.wait();
			}
		}
	}
	static void sort4BitsSignedKeyValueAscending(unsigned int sortData[4],  const int startBit, int lIdx,  unsigned int* ldsSortData, bool Asc_sort, concurrency::tiled_index< WG_SIZE > t_idx) restrict(amp)
	{
		unsigned int signedints[4];
	    signedints[0] = ( ( ( (sortData[0] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[0] >> startBit) & (1<<3));
	    signedints[1] = ( ( ( (sortData[1] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[1] >> startBit) & (1<<3));
	    signedints[2] = ( ( ( (sortData[2] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[2] >> startBit) & (1<<3));
	    signedints[3] = ( ( ( (sortData[3] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[3] >> startBit) & (1<<3));



		for(int bitIdx=0; bitIdx<BITS_PER_PASS; bitIdx++)
		{
			unsigned int mask = (1<<bitIdx);
			uint_4 prefixSum;
			uint_4 cmpResult( signedints[0] & mask, signedints[1] & mask, signedints[2] & mask, signedints[3] & mask );
			uint_4 temp;

			if(!Asc_sort)
			{
				temp.x = (cmpResult.x != 0);
				temp.y = (cmpResult.y != 0);
				temp.z = (cmpResult.z != 0);
				temp.w = (cmpResult.w != 0);
			}
			else
			{
				temp.x = (cmpResult.x != mask);
				temp.y = (cmpResult.y != mask);
				temp.z = (cmpResult.z != mask);
				temp.w = (cmpResult.w != mask);
			}
                        uint_4 arg1 = make_uint4(1,1,1,1);
                        uint_4 arg2 = make_uint4(0,0,0,0);
			prefixSum = SELECT_UINT4( arg1, arg2, temp );//(cmpResult != make_uint4(mask,mask,mask,mask)));

			unsigned int total = 0;
			prefixSum = localPrefixSum256V( prefixSum, lIdx, &total, ldsSortData, t_idx);
			{
				uint_4 localAddr(lIdx*4+0,lIdx*4+1,lIdx*4+2,lIdx*4+3);
				uint_4 dstAddr = localAddr - prefixSum + make_uint4( total, total, total, total );
				if(!Asc_sort)
				{
					temp.x = (cmpResult.x != 0);
					temp.y = (cmpResult.y != 0);
					temp.z = (cmpResult.z != 0);
					temp.w = (cmpResult.w != 0);
					}
				else
				{
					temp.x = (cmpResult.x != mask);
					temp.y = (cmpResult.y != mask);
					temp.z = (cmpResult.z != mask);
					temp.w = (cmpResult.w != mask);
				}
				dstAddr = SELECT_UINT4( prefixSum, dstAddr, temp);

				t_idx.barrier.wait();
        
				ldsSortData[dstAddr.x] = sortData[0];
				ldsSortData[dstAddr.y] = sortData[1];
				ldsSortData[dstAddr.z] = sortData[2];
				ldsSortData[dstAddr.w] = sortData[3];

				t_idx.barrier.wait();

				sortData[0] = ldsSortData[localAddr.x];
				sortData[1] = ldsSortData[localAddr.y];
				sortData[2] = ldsSortData[localAddr.z];
				sortData[3] = ldsSortData[localAddr.w];

				t_idx.barrier.wait();

				ldsSortData[dstAddr.x] = signedints[0];
				ldsSortData[dstAddr.y] = signedints[1];
				ldsSortData[dstAddr.z] = signedints[2];
				ldsSortData[dstAddr.w] = signedints[3];

				t_idx.barrier.wait();
				signedints[0] = ldsSortData[localAddr.x];
				signedints[1] = ldsSortData[localAddr.y];
				signedints[2] = ldsSortData[localAddr.z];
				signedints[3] = ldsSortData[localAddr.w];
				t_idx.barrier.wait();
			}
		}
	}


	static unsigned int scanlMemPrivData( unsigned int val,  unsigned int* lmem, int exclusive, 
	                            concurrency::tiled_index< WG_SIZE > t_idx) restrict (amp)
	{
		// Set first half of local memory to zero to make room for scanning
		unsigned int lIdx = t_idx.local[ 0 ];
		unsigned int wgSize = WG_SIZE;
		lmem[lIdx] = 0;
    
		lIdx += wgSize;
		lmem[lIdx] = val;
		t_idx.barrier.wait();
    
		// Now, perform Kogge-Stone scan
		 unsigned int t;
		for (unsigned int i = 1; i < wgSize; i *= 2)
		{
			t = lmem[lIdx -  i]; 
			t_idx.barrier.wait();
			lmem[lIdx] += t;     
			t_idx.barrier.wait();
		}
		return lmem[lIdx-exclusive];
	}

    typedef struct _b3ConstData
    {
       int m_n;
       int m_nWGs;
       int m_startBit;
       int m_nBlocksPerWG;
    } b3ConstData;


template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
void sort_enqueue_int_uint(bolt::amp::control &ctl,
             DVRandomAccessIterator &first, DVRandomAccessIterator &last,
             StrictWeakOrdering comp,
			 bool int_flag
			 )
{
  
	typedef typename std::iterator_traits< DVRandomAccessIterator >::value_type Values;
    const int RADIX = 4; //Now you cannot replace this with Radix 8 since there is a
                         //local array of 16 elements in the histogram kernel.
    int orig_szElements = static_cast<int>(std::distance(first, last));
	const unsigned int localSize  = WG_SIZE;

	unsigned int szElements = (unsigned int)orig_szElements;
    unsigned int modWgSize = (szElements & ((localSize)-1));
    if( modWgSize )
    {
        szElements &= ~modWgSize;
        szElements += (localSize);
    }
	unsigned int numGroups = (szElements/localSize)>= 32?(32*8):(szElements/localSize); // 32 is no of compute units for Tahiti
	concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();

    concurrency::array<Values, 1 > dvSwapInputValues(static_cast<int>(orig_szElements), av);

	bool Asc_sort = 0;
	if(comp(2,3))
       Asc_sort = 1;
	int swap = 0;
    unsigned int blockSize = (int)(ELEMENTS_PER_WORK_ITEM*localSize);//set at 1024
    unsigned int nBlocks = (int)(orig_szElements + blockSize-1)/(blockSize);
    b3ConstData cdata;
	cdata.m_n = (int)orig_szElements;
	cdata.m_nWGs = (int)numGroups;
	cdata.m_nBlocksPerWG = (int)(nBlocks + numGroups - 1)/numGroups;
    if(nBlocks < numGroups)
    {
		cdata.m_nBlocksPerWG = 1;
		numGroups = nBlocks;
        cdata.m_nWGs = numGroups;
	}
	concurrency::array<int, 1> dvHistogramBins(static_cast<int>(numGroups * RADICES), av);

	concurrency::extent< 1 > inputExtent( numGroups*localSize );
	concurrency::tiled_extent< localSize > tileK0 = inputExtent.tile< localSize >();
	int bits;
	for(bits = 0; bits < (sizeof(Values) * 8); bits += RADIX)
    {
          cdata.m_startBit = bits;
		  concurrency::parallel_for_each( av, tileK0, 
				[
					first,
					&dvSwapInputValues,
					&dvHistogramBins,
					cdata,
					swap,
					Asc_sort,
					int_flag,
					tileK0
				] ( concurrency::tiled_index< localSize > t_idx ) restrict(amp)
		  {
			tile_static unsigned int lmem[WG_SIZE*RADICES];
			unsigned int gIdx = t_idx.global[ 0 ];
			unsigned int lIdx = t_idx.local[ 0 ];
			unsigned int wgIdx = t_idx.tile[ 0 ];
			unsigned int localSize = tileK0.tile_dim0;
			unsigned int numGroups = tileK0[0]/tileK0.tile_dim0;

			const int shift = cdata.m_startBit;
			const int dataAlignment = 1024;
			const int n = cdata.m_n;
			const int w_n = n + dataAlignment-(n%dataAlignment);
			const int nWGs = cdata.m_nWGs;
			const int nBlocksPerWG = cdata.m_nBlocksPerWG;

			for(int i=0; i<RADICES; i++)
			{
				lmem[i*localSize+ lIdx] = 0;
			}
			t_idx.barrier.wait();
			const int blockSize = ELEMENTS_PER_WORK_ITEM*localSize;
			int nBlocks = (w_n)/blockSize - nBlocksPerWG*wgIdx;
			int addr = blockSize*nBlocksPerWG*wgIdx + lIdx;
			unsigned int local_key;
			for(int iblock=0; iblock<_min(nBlocksPerWG, nBlocks); iblock++)
			{
				for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++, addr+=localSize )
				{
					if( (addr) < n)
					{
						if(int_flag && (shift >= sizeof(Values) * 7))
						{
							 if(swap == 0)
								local_key = (first[addr] >> shift);
							 else
								local_key = (dvSwapInputValues[addr] >> shift);
				             unsigned int signBit   = local_key & (1<<3);
							 if(!Asc_sort)
									local_key = 0xF - (( ( ( local_key & 0x7 ) ^ 0x7 ) & 0x7 ) | signBit);
							 else
									local_key = 0xF - (( ( ( local_key & 0x7 ) ^ 0x7 ) & 0x7 ) | signBit);
						}
						else
						{
							if(swap == 0)
								local_key = (first[addr] >> shift) & 0xFU;
							else
								local_key = (dvSwapInputValues[addr] >> shift) & 0xFU;
						}
						if(!Asc_sort)
							lmem[(RADICES - local_key -1)*localSize+ lIdx]++;   
						else
							lmem[local_key*localSize+ lIdx]++;
					}
				}
			}
			t_idx.barrier.wait();
			if( lIdx < RADICES )
			{
				unsigned int sum = 0;
				for(unsigned int i=0; i<localSize; i++)
				{
					sum += lmem[lIdx*localSize+ i];
				}
				dvHistogramBins[lIdx * numGroups + wgIdx] = sum;
			}
		});

	
        concurrency::extent< 1 > scaninputExtent( localSize );
		concurrency::tiled_extent< localSize > tileK1 = scaninputExtent.tile< localSize >();
		concurrency::parallel_for_each( av, tileK1, 
				[
					&dvHistogramBins,
					numGroups,
					tileK1
				] ( concurrency::tiled_index< localSize > t_idx ) restrict(amp)
		  {

			unsigned int lIdx = t_idx.local[ 0 ];
			unsigned int llIdx = lIdx;
			unsigned int wgSize = tileK1.tile_dim0;

			tile_static unsigned int lmem[WG_SIZE*8];
			tile_static int s_seed; 
			s_seed = 0;
			t_idx.barrier.wait();
    
			bool last_thread = (lIdx < numGroups && (lIdx+1) == numGroups) ? 1 : 0;
			//printf("top_scan n = %d\n", n);
			for (int d = 0; d < 16; d++)
			{
				unsigned int val = 0;
				if (lIdx < numGroups)
				{
					val = dvHistogramBins[(numGroups * d) + lIdx];
				}
				// Exclusive scan the counts in local memory
				unsigned int res =  scanlMemPrivData(val, lmem,1, t_idx);
				// Write scanned value out to global
				if (lIdx < numGroups)
				{
					dvHistogramBins[(numGroups * d) + lIdx] = res + s_seed;
				}
				if (last_thread) 
				{
					s_seed += res + val;
				}
				t_idx.barrier.wait();
			}

		});
		if((bits >= sizeof(Values) * 7) && int_flag)
			break;
		concurrency::parallel_for_each( av, tileK0, 
				[
					first,
					&dvSwapInputValues,
					&dvHistogramBins,
					cdata,
					swap,
					Asc_sort,
					tileK0
				] ( concurrency::tiled_index< localSize > t_idx ) restrict(amp)
		  {

			tile_static unsigned int ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];
			tile_static unsigned int localHistogramToCarry[NUM_BUCKET];
			tile_static unsigned int localHistogram[NUM_BUCKET*2];

			unsigned int gIdx = t_idx.global[ 0 ];
			unsigned int lIdx = t_idx.local[ 0 ];
			unsigned int wgIdx = t_idx.tile[ 0 ];
			unsigned int localSize = tileK0.tile_dim0;

			const int dataAlignment = 1024;
			const int n = cdata.m_n;
			const int w_n = n + dataAlignment-(n%dataAlignment);

			const int nWGs = cdata.m_nWGs;
			const int startBit = cdata.m_startBit;
			const int nBlocksPerWG = cdata.m_nBlocksPerWG;

			if( lIdx < (NUM_BUCKET) )
			{
				if(!Asc_sort)
					localHistogramToCarry[lIdx] = dvHistogramBins[(NUM_BUCKET - lIdx -1)*nWGs + wgIdx]; 
				else
					localHistogramToCarry[lIdx] = dvHistogramBins[lIdx*nWGs + wgIdx];
			}

			t_idx.barrier.wait();
			const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;
			int nBlocks = w_n/blockSize - nBlocksPerWG*wgIdx;
			int addr = blockSize*nBlocksPerWG*wgIdx + ELEMENTS_PER_WORK_ITEM*lIdx;
			for(int iblock=0; iblock<_min(nBlocksPerWG, nBlocks); iblock++, addr+=blockSize)
			{
				unsigned int myHistogram = 0;
				unsigned int sortData[ELEMENTS_PER_WORK_ITEM];
				for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
				{
					if(!Asc_sort)
					{
						if(swap == 0)
						{
							sortData[i] = ( addr+i < n )? first[ addr+i ] : 0x0;
						}
						else
						{
							sortData[i] = ( addr+i < n )? dvSwapInputValues[ addr+i ] : 0x0;
						}
					}
					else
					{
						if(swap == 0)
						{
							sortData[i] = ( addr+i < n )? first[ addr+i ] : 0xffffffff;
						}
						else
						{
							sortData[i] = ( addr+i < n )? dvSwapInputValues[ addr+i ] : 0xffffffff;
						}
					}
				}
				sort4BitsKeyValueAscending(sortData, startBit, lIdx, ldsSortData, Asc_sort, t_idx);

				unsigned int keys[ELEMENTS_PER_WORK_ITEM];
				for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
					keys[i] = (sortData[i]>>startBit) & 0xf;
				
				{	
					unsigned int setIdx = lIdx/16;
					if( lIdx < NUM_BUCKET )
					{
						localHistogram[lIdx] = 0;
					}
					ldsSortData[lIdx] = 0;
					t_idx.barrier.wait();
					
					for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
					{
						if( addr+i < n )
						{
							if(!Asc_sort)
								AtomInc( SET_HISTOGRAM( setIdx, (NUM_BUCKET - keys[i] - 1) ) );
							else
								AtomInc( SET_HISTOGRAM( setIdx, keys[i] ) );
						}
					}
					t_idx.barrier.wait();
					unsigned int hIdx = NUM_BUCKET+lIdx;
					if( lIdx < NUM_BUCKET )
					{
						unsigned int sum = 0;
						for(int i=0; i<WG_SIZE/16; i++)
						{
							sum += SET_HISTOGRAM( i, lIdx );
						}
						myHistogram = sum;
						localHistogram[hIdx] = sum;
					}
					t_idx.barrier.wait();
					if( lIdx < NUM_BUCKET )
					{
						localHistogram[hIdx] = localHistogram[hIdx-1];//may cause race condition
					}
					t_idx.barrier.wait();
					unsigned int u0, u1, u2;
					if( lIdx < NUM_BUCKET )
					{
						u0 = localHistogram[hIdx-3];
						u1 = localHistogram[hIdx-2];
						u2 = localHistogram[hIdx-1];
					}
					t_idx.barrier.wait();
					if( lIdx < NUM_BUCKET )
						localHistogram[hIdx] += u0 + u1 + u2;
					t_idx.barrier.wait();
					if( lIdx < NUM_BUCKET )
					{
						u0 = localHistogram[hIdx-12];
						u1 = localHistogram[hIdx-8];
						u2 = localHistogram[hIdx-4];
					}
					t_idx.barrier.wait();
					if( lIdx < NUM_BUCKET )
						localHistogram[hIdx] += u0 + u1 + u2;
					t_idx.barrier.wait();
				}
				{
					for(int ie=0; ie<ELEMENTS_PER_WORK_ITEM; ie++)
					{
						int dataIdx = ELEMENTS_PER_WORK_ITEM*lIdx+ie;
						int binIdx;
						int groupOffset;
						if(!Asc_sort)
						{
							binIdx = NUM_BUCKET - keys[ie] - 1;
							groupOffset = localHistogramToCarry[NUM_BUCKET - binIdx -1];
						}
						else
						{
							binIdx = keys[ie];
							groupOffset = localHistogramToCarry[binIdx];
						}
						int myIdx = dataIdx - localHistogram[NUM_BUCKET+binIdx];
						if( addr+ie < n )
						{
							if ((groupOffset + myIdx)<n)
							{
								if(swap == 0)
								{
									dvSwapInputValues[ groupOffset + myIdx ] =  sortData[ie]; 
								}
								else
								{
									first[ groupOffset + myIdx ] = sortData[ie];
								}
							}
						}
					}
				}
				t_idx.barrier.wait();
				if( lIdx < NUM_BUCKET )
				{
					if(!Asc_sort)
						localHistogramToCarry[NUM_BUCKET - lIdx -1] += myHistogram;
					else
						localHistogramToCarry[lIdx] += myHistogram;
				}
				t_idx.barrier.wait();
			}
		 });
		 swap = swap? 0: 1;
	}
	if(int_flag)
	{
		cdata.m_startBit = bits;
		concurrency::parallel_for_each( av, tileK0, 
				[
					first,
					&dvSwapInputValues,
					&dvHistogramBins,
					cdata,
					swap,
					Asc_sort,
					tileK0
				] ( concurrency::tiled_index< localSize > t_idx ) restrict(amp)
		  {

			tile_static unsigned int ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+WG_SIZE];
			tile_static unsigned int localHistogramToCarry[NUM_BUCKET];
			tile_static unsigned int localHistogram[NUM_BUCKET*2];

			unsigned int gIdx = t_idx.global[ 0 ];
			unsigned int lIdx = t_idx.local[ 0 ];
			unsigned int wgIdx = t_idx.tile[ 0 ];
			unsigned int localSize = tileK0.tile_dim0;

			const int dataAlignment = 1024;
			const int n = cdata.m_n;
			const int w_n = n + dataAlignment-(n%dataAlignment);

			const int nWGs = cdata.m_nWGs;
			const int startBit = cdata.m_startBit;
			const int nBlocksPerWG = cdata.m_nBlocksPerWG;

			if( lIdx < (NUM_BUCKET) )
			{
				if(!Asc_sort)
					localHistogramToCarry[lIdx] = dvHistogramBins[lIdx*nWGs + wgIdx];
				else
					localHistogramToCarry[lIdx] = dvHistogramBins[lIdx*nWGs + wgIdx];
			}

			t_idx.barrier.wait();
			const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;
			int nBlocks = w_n/blockSize - nBlocksPerWG*wgIdx;
			int addr = blockSize*nBlocksPerWG*wgIdx + ELEMENTS_PER_WORK_ITEM*lIdx;

			for(int iblock=0; iblock<_min(nBlocksPerWG, nBlocks); iblock++, addr+=blockSize)
			{
				unsigned int myHistogram = 0;
				unsigned int sortData[ELEMENTS_PER_WORK_ITEM];
				for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
				{
					if(!Asc_sort)
					{
						sortData[i] = ( addr+i < n )? dvSwapInputValues[ addr+i ] : 0x80000000;
					}
					else
					{
						sortData[i] = ( addr+i < n )? dvSwapInputValues[ addr+i ] : 0x7fffffff;
					}
				}
				sort4BitsSignedKeyValueAscending(sortData, startBit, lIdx, ldsSortData, Asc_sort, t_idx);

				unsigned int keys[ELEMENTS_PER_WORK_ITEM];
				for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
					keys[i] = 0xF - (( ( ( (sortData[i] >> startBit) & 0x7 ) ^ 0x7 ) & 0x7 ) | ((sortData[i] >> startBit) & (1<<3)) );
				
				{	
					unsigned int setIdx = lIdx/16;
					if( lIdx < NUM_BUCKET )
					{
						localHistogram[lIdx] = 0;
					}
					ldsSortData[lIdx] = 0;
					t_idx.barrier.wait();
					
					for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
					{
						if( addr+i < n )
						{
							if(!Asc_sort)
								AtomInc( SET_HISTOGRAM( setIdx, (NUM_BUCKET - keys[i] - 1) ) );
							else
								AtomInc( SET_HISTOGRAM( setIdx, keys[i] ) );
						}
					}
					t_idx.barrier.wait();
					unsigned int hIdx = NUM_BUCKET+lIdx;
					if( lIdx < NUM_BUCKET )
					{
						unsigned int sum = 0;
						for(int i=0; i<WG_SIZE/16; i++)
						{
							sum += SET_HISTOGRAM( i, lIdx );
						}
						myHistogram = sum;
						localHistogram[hIdx] = sum;
					}
					t_idx.barrier.wait();
					if( lIdx < NUM_BUCKET )
					{
						localHistogram[hIdx] = localHistogram[hIdx-1];
					}
					t_idx.barrier.wait();
					unsigned int u0, u1, u2;
					if( lIdx < NUM_BUCKET )
					{
						u0 = localHistogram[hIdx-3];
						u1 = localHistogram[hIdx-2];
						u2 = localHistogram[hIdx-1];
					}
					t_idx.barrier.wait();
					if( lIdx < NUM_BUCKET )
						localHistogram[hIdx] += u0 + u1 + u2;
					t_idx.barrier.wait();
					if( lIdx < NUM_BUCKET )
					{
						u0 = localHistogram[hIdx-12];
						u1 = localHistogram[hIdx-8];
						u2 = localHistogram[hIdx-4];
					}
					t_idx.barrier.wait();
					if( lIdx < NUM_BUCKET )
						localHistogram[hIdx] += u0 + u1 + u2;
					t_idx.barrier.wait();
				}
				{
					for(int ie=0; ie<ELEMENTS_PER_WORK_ITEM; ie++)
					{
						int dataIdx = ELEMENTS_PER_WORK_ITEM*lIdx+ie;
						int binIdx;
						int groupOffset;
						if(!Asc_sort)
						{
							binIdx = 0xF - keys[ie];
							groupOffset = localHistogramToCarry[binIdx];
						}
						else
						{
							binIdx = keys[ie];
							groupOffset = localHistogramToCarry[binIdx];
						}
						int myIdx = dataIdx - localHistogram[NUM_BUCKET+binIdx];
						if( addr+ie < n )
						{
							if ((groupOffset + myIdx)<n)
							{
									first[ groupOffset + myIdx ] = sortData[ie];
							}
						}
					}
				}
				t_idx.barrier.wait();
				if( lIdx < NUM_BUCKET )
				{
					localHistogramToCarry[lIdx] += myHistogram;
				}
				t_idx.barrier.wait();
			}
		 });
	}
    return;



}

	template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,
                                           int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_enqueue( control &ctl,
                         DVRandomAccessIterator first, DVRandomAccessIterator last,
							 StrictWeakOrdering comp)
	{
		bool int_flag = 1;
		sort_enqueue_int_uint(ctl, first, last, comp, int_flag);
		return;
	}
    template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,
                                           unsigned int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_enqueue( control &ctl,
                         DVRandomAccessIterator first, DVRandomAccessIterator last,
                         StrictWeakOrdering comp)
	{
		bool int_flag = 0;
		sort_enqueue_int_uint(ctl, first, last, comp, int_flag);
		return;
	}


// Hui. Declar prior to its use
template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
 typename std::enable_if<
     !(std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type, unsigned int >::value ||
       std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,          int >::value)
                        >::type
 sort_enqueue(bolt::amp::control &ctl, const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
 const StrictWeakOrdering& comp);

//Device Vector specialization
template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
void sort_pick_iterator( bolt::amp::control &ctl,
                         const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
                         const StrictWeakOrdering& comp, bolt::amp::device_vector_tag )
{
    // User defined Data types are not supported with device_vector. Hence we have a static assert here.
    // The code here should be in compliant with the routine following this routine.
    typedef typename std::iterator_traits<DVRandomAccessIterator>::value_type T;
    int szElements = static_cast< int >( std::distance( first, last ) );
    if (szElements < 2)
        return;
    bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
	if (runMode == bolt::amp::control::Automatic)
	{
		runMode = ctl.getDefaultPathToRun();
	}

    if ((runMode == bolt::amp::control::SerialCpu)) {
        // Hui
        typename bolt::amp::device_vector< T >::pointer firstPtr =  first.getContainer( ).data( );
        std::sort(&firstPtr[ first.m_Index ], &firstPtr[ last.m_Index ], comp);
    } else if (runMode == bolt::amp::control::MultiCoreCpu) {
#ifdef ENABLE_TBB
        // Hui
        typename bolt::amp::device_vector< T >::pointer firstPtr =  first.getContainer( ).data( );
        bolt::btbb::sort(&firstPtr[ first.m_Index ], &firstPtr[ last.m_Index ], comp);
#else
        throw Concurrency::runtime_exception( "The MultiCoreCpu version of sort is not enabled to be built.", 0);
        return;
#endif
    } else {
        sort_enqueue(ctl,first,last,comp);
    }
    return;
}

// FIXME: it can't compile on Linux
#ifdef _WIN32
template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
void sort_pick_iterator( control &ctl,
                         const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
                         const StrictWeakOrdering& comp, bolt::amp::fancy_iterator_tag )
{ 
    static_assert( false, "It is not possible to sort fancy iterators. They are not mutable" );
}
#endif

//Non Device Vector specialization.
//This implementation creates a cl::Buffer and passes the cl buffer to the sort specialization whichtakes
//the cl buffer as a parameter. In the future, Each input buffer should be mapped to the device_vector
//and the specialization specific to device_vector should be called.
template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort_pick_iterator( bolt::amp::control &ctl,
                         const RandomAccessIterator& first, const RandomAccessIterator& last,
                         const StrictWeakOrdering& comp, std::random_access_iterator_tag )
{
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
    int szElements = static_cast< int >( std::distance( first, last ) );
    if (szElements < 2)
        return;

    bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
	if (runMode == bolt::amp::control::Automatic)
	{
		runMode = ctl.getDefaultPathToRun();
	}

    if ((runMode == bolt::amp::control::SerialCpu)) {
        std::sort(first, last, comp);
        return;
    } else if (runMode == bolt::amp::control::MultiCoreCpu) {
#ifdef ENABLE_TBB
        bolt::btbb::sort(first,last, comp);
#else
//      std::cout << "The MultiCoreCpu version of sort is not enabled. " << std ::endl;
        throw Concurrency::runtime_exception( "The MultiCoreCpu version of sort is not enabled to be built.", 0);
        return;
#endif
    } else {
        device_vector< T, concurrency::array_view > dvInputOutput( first, last, false, ctl );
        //Now call the actual amp algorithm
        sort_enqueue(ctl,dvInputOutput.begin(),dvInputOutput.end(),comp);
        //Map the buffer back to the host
        dvInputOutput.data( );
        return;
    }
}

// FIXME: it can't compile on Linux
#ifdef _WIN32
// Wrapper that uses default control class, iterator interface
template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort_detect_random_access( bolt::amp::control &ctl,
                                const RandomAccessIterator& first, const RandomAccessIterator& last,
                                const StrictWeakOrdering& comp, std::input_iterator_tag )
{
    //  \TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
    //  to a temporary buffer.  Should we?
    static_assert( false, "Bolt only supports random access iterator types" );
};
#endif

template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort_detect_random_access( bolt::amp::control &ctl,
                                const RandomAccessIterator& first, const RandomAccessIterator& last,
                                const StrictWeakOrdering& comp, std::random_access_iterator_tag )
{
    return sort_pick_iterator(ctl, first, last, comp,
                              typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
};


/*AMP Kernels for Signed integer sorting*/
/*!
* \brief This template function is a permutation Algorithm for one work item.
*        This is called by the the sort_enqueue for input buffers with siigned numbers routine for other than int's and unsigned int's
*
* \details
*         The idea behind sorting signed numbers is that when we sort the MSB which contains the sign bit. We should
*         sort in the signed bit in the descending order. For example in a radix 4 sort of signed numbers if we are
*         doing ascending sort. Then the bits 28:31 should be sorted in an descending order, after doing the following
*         transformation of the input bits.
*         Shift bits 31:28 ->  3:0
*               value = (value >> shiftCount);
*         Retain the sign bit in the bit location 3.
*               signBit = value & (1<<(RADIX_T-1));
*         XOR bits 2:1  with 111
*               value = ( ( ( value & MASK_T ) ^ MASK_T ) & MASK_T )
*         Or the sign bit with the xor'ed bit. This forms your index.
*               value = value | signBitl
*
*       As an Example for Ascending order sort of 4 bit signed numbers.
*       ________________________________
*       -Flip the three bits other than the sign bit
*       -sort the resultant array considering them as unsigned numbers
*       -map the corresponding flipped numbers with the corresponding values.
*       1st Example
*       1111     1000      0000     0111
*       0111     0000      0101     0010
*       1000 ->  1111 ->   0111 ->  0000
*       1010     1101      1000     1111
*       0000     0111      1101     1010
*       0010     0101      1111     1000
*
*       2nd Example
*       1111     1000     1111      1000
*       0111     0000     1101      1010
*       1101     1010     1010      1101
*       1000     1111     1000      1111
*       1010 ->  1101 ->  0111 ->   0000
*       0000     0111     0101      0010
*       0010     0101     0010      0101
*       0101     0010     0001      0110
*       0110     0001     0000      0111
*
*/

template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
typename std::enable_if<
    !(std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type, unsigned int >::value ||
      std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,          int >::value)
                       >::type
sort_enqueue(bolt::amp::control &ctl, const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
const StrictWeakOrdering& comp)
{
 
      bolt::amp::detail::stablesort_enqueue(ctl,first,last,comp);
      return;
}// END of sort_enqueue



}//namespace bolt::amp::detail

template<typename RandomAccessIterator>
void sort(RandomAccessIterator first,
          RandomAccessIterator last)
{ 
    typedef typename std::iterator_traits< RandomAccessIterator >::value_type T;

    detail::sort_detect_random_access( bolt::amp::control::getDefault( ),
                                       first, last, less< T >( ),
                                       typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
    return;
}

template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
    detail::sort_detect_random_access( bolt::amp::control::getDefault( ),
                                       first, last, comp,
                                       typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
    return;
}

template<typename RandomAccessIterator>
void sort(bolt::amp::control &ctl,
            RandomAccessIterator first,
            RandomAccessIterator last)
{
    typedef typename std::iterator_traits< RandomAccessIterator >::value_type T;
    detail::sort_detect_random_access(ctl,
                                        first, last, less< T >( ),
                                        typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
    return;
}

template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort(bolt::amp::control &ctl,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
    detail::sort_detect_random_access(ctl,
                                        first, last, comp,
                                        typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
    return;
}

}//namespace bolt::amp
}//namespace bolt

#endif
