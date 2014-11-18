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

/***************************************************************************
* The Radix sort algorithm implementation in BOLT library is a derived work from 
* the radix sort sample which is provided in the Book. "Heterogeneous Computing with OpenCL"
* Link: http://www.heterogeneouscompute.org/?page_id=7
* The original Authors are: Takahiro Harada and Lee Howes. A detailed explanation of 
* the algorithm is given in the publication linked here. 
* http://www.heterogeneouscompute.org/wordpress/wp-content/uploads/2011/06/RadixSort.pdf
* 
* The derived work adds support for descending sort and signed integers. 
* Performance optimizations were provided for the AMD GCN architecture. 
* 
*  Besides this following publications were referred: 
*  1. "Parallel Scan For Stream Architectures"  
*     Technical Report CS2009-14Department of Computer Science, University of Virginia. 
*     Duane Merrill and Andrew Grimshaw
*    https://sites.google.com/site/duanemerrill/ScanTR2.pdf
*  2. "Revisiting Sorting for GPGPU Stream Architectures" 
*     Duane Merrill and Andrew Grimshaw
*    https://sites.google.com/site/duanemerrill/RadixSortTR.pdf
*  3. The SHOC Benchmark Suite 
*     https://github.com/vetter/shoc
*
***************************************************************************/

#pragma once
#if !defined( BOLT_AMP_SORT_BY_KEY_INL )
#define BOLT_AMP_SORT_BY_KEY_INL

#include <algorithm>
#include <type_traits>
#include <amp.h>
#include <amp_short_vectors.h>
#include "bolt/amp/pair.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/iterator/iterator_traits.h"


#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/sort_by_key.h"
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
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))
#define make_uint4 (uint_4)
#define SET_HISTOGRAM(setIdx, key) ldsSortData[(setIdx)*NUM_BUCKET+key]

namespace bolt {
namespace amp {

namespace detail {
  // Hui. Adding declarations prior to use
        template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
							 StrictWeakOrdering comp);
    template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           unsigned int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
                         StrictWeakOrdering comp);

     template< typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if<
        !( std::is_same< typename std::iterator_traits<DVKeys >::value_type, unsigned int >::value ||
           std::is_same< typename std::iterator_traits<DVKeys >::value_type, int >::value 
         )
                           >::type
    sort_by_key_enqueue(control &ctl, const DVKeys& keys_first,
                        const DVKeys& keys_last, const DVValues& values_first,
                        const StrictWeakOrdering& comp);
                        
	static inline uint_4 SELECT_UINT4_FOR_KEY(uint_4 &a,uint_4 &b,uint_4  &condition )  restrict(amp)
	{
		uint_4 res;
		res.x = (condition.x )? b.x : a.x;
		res.y = (condition.y )? b.y : a.y;
		res.z = (condition.z )? b.z : a.z;
		res.w = (condition.w )? b.w : a.w;
		return res;

	}
    //Serial CPU code path implementation.
    //Class to hold the key value pair. This will be used to zip th ekey and value together in a vector.
    template <typename keyType, typename valueType>
    class std_sort
    {
    public:
        keyType   key;
        valueType value;
    };
    
    //This is the functor which will sort the std_sort vector.
    template <typename keyType, typename valueType, typename StrictWeakOrdering>
    class std_sort_comp
    {
    public:
        typedef std_sort<keyType, valueType> KeyValueType;
        std_sort_comp(const StrictWeakOrdering &_swo):swo(_swo)
        {}
        StrictWeakOrdering swo;
        bool operator() (const KeyValueType &lhs, const KeyValueType &rhs) const
        {
            return swo(lhs.key, rhs.key);
        }
    };

    //The serial CPU implementation of sort_by_key routine. This routines zips the key value pair and then sorts
    //using the std::sort routine.
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    static void serialCPU_sort_by_key( const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                const RandomAccessIterator2 values_first,
                                const StrictWeakOrdering& comp)
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< RandomAccessIterator2 >::value_type valType;
        typedef std_sort<keyType, valType> KeyValuePair;
        typedef std_sort_comp<keyType, valType, StrictWeakOrdering> KeyValuePairFunctor;

        int vecSize = static_cast< int >(std::distance( keys_first, keys_last ));
        std::vector<KeyValuePair> KeyValuePairVector(vecSize);
        KeyValuePairFunctor functor(comp);
        //Zip the key and values iterators into a std_sort vector.
        for (int i=0; i< vecSize; i++)
        {
            KeyValuePairVector[i].key   = *(keys_first + i);
            KeyValuePairVector[i].value = *(values_first + i);
        }
        //Sort the std_sort vector using std::sort
        std::sort(KeyValuePairVector.begin(), KeyValuePairVector.end(), functor);
        //Extract the keys and values from the KeyValuePair and fill the respective iterators.
        for (int i=0; i< vecSize; i++)
        {
            *(keys_first + i)   = KeyValuePairVector[i].key;
            *(values_first + i) = KeyValuePairVector[i].value;
        }
    }
	static unsigned int scanLocalMemAndTotal_for_key(unsigned int val, unsigned int* lmem, unsigned int *totalSum, int exclusive, concurrency::tiled_index< WG_SIZE > t_idx) restrict(amp)
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
	static unsigned int prefixScanVectorEx_for_key( uint_4* data ) restrict(amp)
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
	static uint_4 localPrefixSum256V_for_key( uint_4 pData, unsigned int lIdx, unsigned int* totalSum, unsigned int* sorterSharedMemory, concurrency::tiled_index< WG_SIZE > t_idx ) restrict(amp)
	{
		unsigned int s4 = prefixScanVectorEx_for_key( &pData );
		unsigned int rank = scanLocalMemAndTotal_for_key( s4, sorterSharedMemory, totalSum,  1, t_idx);
		return pData + make_uint4( rank, rank, rank, rank );
	}
	template<typename Values>
	static void sort4BitsKeyValueAscending_for_key(unsigned int sortData[4],  Values sortVal[4], const int startBit, int lIdx,  unsigned int* ldsSortData,  Values *ldsSortVal, bool Asc_sort, concurrency::tiled_index< WG_SIZE > t_idx) restrict(amp)
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
			prefixSum = SELECT_UINT4_FOR_KEY( arg1, arg2, temp );//(cmpResult != make_uint4(mask,mask,mask,mask)));

			unsigned int total = 0;
			prefixSum = localPrefixSum256V_for_key( prefixSum, lIdx, &total, ldsSortData, t_idx);
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
				dstAddr = SELECT_UINT4_FOR_KEY( prefixSum, dstAddr, temp);

				t_idx.barrier.wait();
        
				ldsSortData[dstAddr.x] = sortData[0];
				ldsSortData[dstAddr.y] = sortData[1];
				ldsSortData[dstAddr.z] = sortData[2];
				ldsSortData[dstAddr.w] = sortData[3];

				ldsSortVal[dstAddr.x] = sortVal[0];
				ldsSortVal[dstAddr.y] = sortVal[1];
				ldsSortVal[dstAddr.z] = sortVal[2];
				ldsSortVal[dstAddr.w] = sortVal[3];

				t_idx.barrier.wait();

				sortData[0] = ldsSortData[localAddr.x];
				sortData[1] = ldsSortData[localAddr.y];
				sortData[2] = ldsSortData[localAddr.z];
				sortData[3] = ldsSortData[localAddr.w];

				sortVal[0] = ldsSortVal[localAddr.x];
				sortVal[1] = ldsSortVal[localAddr.y];
				sortVal[2] = ldsSortVal[localAddr.z];
				sortVal[3] = ldsSortVal[localAddr.w];

				t_idx.barrier.wait();
			}
		}
	}
	template<typename Values>
	static void sort4BitsSignedKeyValueAscending_for_key(unsigned int sortData[4],  Values sortVal[4], const int startBit, int lIdx,  unsigned int* ldsSortData,  Values *ldsSortVal, bool Asc_sort, concurrency::tiled_index< WG_SIZE > t_idx) restrict(amp)
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
			prefixSum = SELECT_UINT4_FOR_KEY( arg1, arg2, temp );//(cmpResult != make_uint4(mask,mask,mask,mask)));

			unsigned int total = 0;
			prefixSum = localPrefixSum256V_for_key( prefixSum, lIdx, &total, ldsSortData, t_idx);
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
				dstAddr = SELECT_UINT4_FOR_KEY( prefixSum, dstAddr, temp);

				t_idx.barrier.wait();
        
				ldsSortData[dstAddr.x] = sortData[0];
				ldsSortData[dstAddr.y] = sortData[1];
				ldsSortData[dstAddr.z] = sortData[2];
				ldsSortData[dstAddr.w] = sortData[3];

				ldsSortVal[dstAddr.x] = sortVal[0];
				ldsSortVal[dstAddr.y] = sortVal[1];
				ldsSortVal[dstAddr.z] = sortVal[2];
				ldsSortVal[dstAddr.w] = sortVal[3];

				t_idx.barrier.wait();

				sortData[0] = ldsSortData[localAddr.x];
				sortData[1] = ldsSortData[localAddr.y];
				sortData[2] = ldsSortData[localAddr.z];
				sortData[3] = ldsSortData[localAddr.w];

				sortVal[0] = ldsSortVal[localAddr.x];
				sortVal[1] = ldsSortVal[localAddr.y];
				sortVal[2] = ldsSortVal[localAddr.z];
				sortVal[3] = ldsSortVal[localAddr.w];

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


	static unsigned int scanlMemPrivData_for_key( unsigned int val,  unsigned int* lmem, int exclusive, 
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

struct b3ConstData
{
   int m_n;
   int m_nWGs;
   int m_startBit;
   int m_nBlocksPerWG;
};

template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
void sort_by_key_enqueue_int_uint( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
                         StrictWeakOrdering comp,
						 bool int_flag)
{

	typedef typename std::iterator_traits< DVKeys >::value_type Keys;
    typedef typename std::iterator_traits< DVValues >::value_type Values;
    const int RADIX = 4; //Now you cannot replace this with Radix 8 since there is a
                         //local array_view of 16 elements in the histogram kernel.
    int orig_szElements = static_cast<int>(std::distance(keys_first, keys_last));
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

	device_vector< Keys, concurrency::array_view > dvSwapInputKeys(static_cast<int>(orig_szElements), 0);
    device_vector< Values, concurrency::array_view > dvSwapInputValues(static_cast<int>(orig_szElements), 0);

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

	device_vector< int, concurrency::array_view > dvHistogramBins(static_cast<int>(numGroups * RADICES), int(0) );

	concurrency::extent< 1 > inputExtent( numGroups*localSize );
	concurrency::tiled_extent< localSize > tileK0 = inputExtent.tile< localSize >();
	int bits;
	for(bits = 0; bits < (sizeof(Keys) * 8); bits += RADIX)
    {
          cdata.m_startBit = bits;
		  concurrency::parallel_for_each( av, tileK0, 
				[
					keys_first,
					dvSwapInputKeys,
					dvHistogramBins,
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
			for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++)
			{
				for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++, addr+=localSize )
				{
					if( (addr) < n)
					{
						if(int_flag && (shift >= sizeof(Keys) * 7))
						{
							 if(swap == 0)
								local_key = (keys_first[addr] >> shift);
							 else
								local_key = (dvSwapInputKeys[addr] >> shift);
				             unsigned int signBit   = local_key & (1<<3);
							 if(!Asc_sort)
									local_key = 0xF - (( ( ( local_key & 0x7 ) ^ 0x7 ) & 0x7 ) | signBit);
							 else
									local_key = 0xF - (( ( ( local_key & 0x7 ) ^ 0x7 ) & 0x7 ) | signBit);
						}
						else
						{
							if(swap == 0)
								local_key = (keys_first[addr] >> shift) & 0xFU;
							else
								local_key = (dvSwapInputKeys[addr] >> shift) & 0xFU;
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
					dvHistogramBins,
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
				unsigned int res =  scanlMemPrivData_for_key(val, lmem,1, t_idx);
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
		if((bits >= sizeof(Keys) * 7) && int_flag)
			break;
		concurrency::parallel_for_each( av, tileK0, 
				[
					keys_first,
					values_first,
					dvSwapInputKeys,
					dvSwapInputValues,
					dvHistogramBins,
					cdata,
					swap,
					Asc_sort,
					tileK0
				] ( concurrency::tiled_index< localSize > t_idx ) restrict(amp)
		  {

			tile_static unsigned int ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];
			tile_static Values ldsSortVal[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];
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
			for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++, addr+=blockSize)
			{
				unsigned int myHistogram = 0;
				unsigned int sortData[ELEMENTS_PER_WORK_ITEM];
				Values sortVal[ELEMENTS_PER_WORK_ITEM];
				for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
				{
					if(!Asc_sort)
					{
						if(swap == 0)
						{
							sortData[i] = ( addr+i < n )? keys_first[ addr+i ] : 0x0;
							sortVal[i]  = ( addr+i < n )? values_first[ addr+i ] : 0x0;
						}
						else
						{
							sortData[i] = ( addr+i < n )? dvSwapInputKeys[ addr+i ] : 0x0;
							sortVal[i]  = ( addr+i < n )? dvSwapInputValues[ addr+i ] : 0x0;
						}
					}
					else
					{
						if(swap == 0)
						{
							sortData[i] = ( addr+i < n )? keys_first[ addr+i ] : 0xffffffff;
							sortVal[i]  = ( addr+i < n )? values_first[ addr+i ] : 0xffffffff;
						}
						else
						{
							sortData[i] = ( addr+i < n )? dvSwapInputKeys[ addr+i ] : 0xffffffff;
							sortVal[i]  = ( addr+i < n )? dvSwapInputValues[ addr+i ] : 0xffffffff;
						}
					}
				}
				sort4BitsKeyValueAscending_for_key(sortData, sortVal, startBit, lIdx, ldsSortData, ldsSortVal, Asc_sort, t_idx);

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
									dvSwapInputKeys[ groupOffset + myIdx ] =  sortData[ie]; 
									dvSwapInputValues[ groupOffset + myIdx ] =  sortVal[ie]; 
								}
								else
								{
									keys_first[ groupOffset + myIdx ] = sortData[ie];
									values_first[ groupOffset + myIdx ] = sortVal[ie];
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
					keys_first,
					values_first,
					dvSwapInputKeys,
					dvSwapInputValues,
					dvHistogramBins,
					cdata,
					swap,
					Asc_sort,
					tileK0
				] ( concurrency::tiled_index< localSize > t_idx ) restrict(amp)
		  {

			tile_static unsigned int ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+WG_SIZE];
			tile_static Values ldsSortVal[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];
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

			for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++, addr+=blockSize)
			{
				unsigned int myHistogram = 0;
				unsigned int sortData[ELEMENTS_PER_WORK_ITEM];
				Values sortVal[ELEMENTS_PER_WORK_ITEM];
				for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
				{
					if(!Asc_sort)
					{
						sortData[i] = ( addr+i < n )? dvSwapInputKeys[ addr+i ] : 0x80000000;
						sortVal[i]  = ( addr+i < n )? dvSwapInputValues[ addr+i ] : 0x80000000;
					}
					else
					{
						sortData[i] = ( addr+i < n )? dvSwapInputKeys[ addr+i ] : 0x7fffffff;
						sortVal[i]  = ( addr+i < n )? dvSwapInputValues[ addr+i ] : 0x7fffffff;
					}
				}
				sort4BitsSignedKeyValueAscending_for_key(sortData, sortVal, startBit, lIdx, ldsSortData, ldsSortVal, Asc_sort, t_idx);

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
									keys_first[ groupOffset + myIdx ] = sortData[ie];
									values_first[ groupOffset + myIdx ] = sortVal[ie];
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


    template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
							 StrictWeakOrdering comp)
	{
		bool int_flag = true;
		sort_by_key_enqueue_int_uint(ctl, keys_first, keys_last, values_first, comp, int_flag);
		return;
	}
    template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           unsigned int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
                         StrictWeakOrdering comp)
	{
		bool int_flag = false;
		sort_by_key_enqueue_int_uint(ctl, keys_first, keys_last, values_first, comp, int_flag);
		return;
	}

	template< typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if<
        !( std::is_same< typename std::iterator_traits<DVKeys >::value_type, unsigned int >::value ||
           std::is_same< typename std::iterator_traits<DVKeys >::value_type, int >::value 
         )
                           >::type
    sort_by_key_enqueue(control &ctl, const DVKeys& keys_first,
                        const DVKeys& keys_last, const DVValues& values_first,
                        const StrictWeakOrdering& comp)
    {
        // FIXME : actualy use stablesort_by_key_enqueue
        //stablesort_by_key_enqueue(ctl, keys_first, keys_last, values_first, comp);
        return;
    }// END of sort_by_key_enqueue



    //Fancy iterator specialization
    template<typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering>
    void sort_by_key_pick_iterator(control &ctl, DVRandomAccessIterator1 keys_first,
                                   DVRandomAccessIterator1 keys_last, DVRandomAccessIterator2 values_first,
                                   StrictWeakOrdering comp,  bolt::amp::fancy_iterator_tag )
    {
        static_assert( std::is_same<DVRandomAccessIterator1, bolt::amp::fancy_iterator_tag  >::value, "It is not possible to output to fancy iterators; they are not mutable! " );
    }

    //Device Vector specialization
    template<typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering>
    void sort_by_key_pick_iterator(control &ctl, DVRandomAccessIterator1 keys_first,
                                   DVRandomAccessIterator1 keys_last, DVRandomAccessIterator2 values_first,
                                   StrictWeakOrdering comp,
                                   bolt::amp::device_vector_tag, bolt::amp::device_vector_tag )
    {
        typedef typename std::iterator_traits< DVRandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< DVRandomAccessIterator2 >::value_type valueType;
        // User defined Data types are not supported with device_vector. Hence we have a static assert here.
        // The code here should be in compliant with the routine following this routine.
        unsigned int szElements = (unsigned int)(keys_last - keys_first);
        if (szElements < 2 )
                return;
        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode( );
        if( runMode == bolt::amp::control::Automatic )
        {
            runMode = ctl.getDefaultPathToRun( );
        }
		
        if (runMode == bolt::amp::control::SerialCpu) 
		{
		   			
            typename bolt::amp::device_vector< keyType >::pointer   keysPtr   =  const_cast<typename bolt::amp::device_vector< keyType >::pointer>(keys_first.getContainer( ).data( ));
            typename bolt::amp::device_vector< valueType >::pointer valuesPtr =  const_cast<typename bolt::amp::device_vector< valueType >::pointer>(values_first.getContainer( ).data( ));
            serialCPU_sort_by_key(&keysPtr[keys_first.m_Index], &keysPtr[keys_last.m_Index],
                                            &valuesPtr[values_first.m_Index], comp);
            return;
        } 
		else if (runMode == bolt::amp::control::MultiCoreCpu) 
		{

            #ifdef ENABLE_TBB
                typename bolt::amp::device_vector< keyType >::pointer   keysPtr   =  const_cast<typename bolt::amp::device_vector< keyType >::pointer>(keys_first.getContainer( ).data( ));
                typename bolt::amp::device_vector< valueType >::pointer valuesPtr =  const_cast<typename bolt::amp::device_vector< valueType >::pointer>(values_first.getContainer( ).data( ));
                bolt::btbb::sort_by_key(&keysPtr[keys_first.m_Index], &keysPtr[keys_last.m_Index],
                                                &valuesPtr[values_first.m_Index], comp);
                return;
            #else
               throw std::runtime_error( "The MultiCoreCpu version of Sort_by_key is not enabled to be built with TBB!\n");
            #endif
        }

        else 
		{   
            sort_by_key_enqueue(ctl, keys_first, keys_last, values_first, comp);
        }
        return;
    }

    //Non Device Vector specialization.
    //This implementation creates a Buffer and passes the AMP buffer to the
    //sort specialization whichtakes the AMP buffer as a parameter.
    //In the future, Each input buffer should be mapped to the device_vector and the
    //specialization specific to device_vector should be called.
    template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
    void sort_by_key_pick_iterator(control &ctl, RandomAccessIterator1 keys_first,
                                   RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first,
                                   StrictWeakOrdering comp, 
                                   std::random_access_iterator_tag, std::random_access_iterator_tag )
    {

        typedef typename std::iterator_traits<RandomAccessIterator1>::value_type T_keys;
        typedef typename std::iterator_traits<RandomAccessIterator2>::value_type T_values;
        unsigned int szElements = (unsigned int)(keys_last - keys_first);
        if (szElements < 2)
            return;

        bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode( );
        if( runMode == bolt::amp::control::Automatic )
        {
            runMode = ctl.getDefaultPathToRun( );
        }
	    
        if (runMode == bolt::amp::control::SerialCpu /*|| (szElements < WGSIZE) */)
		{   
            serialCPU_sort_by_key(keys_first, keys_last, values_first, comp);
            return;
        } 
		else if (runMode == bolt::amp::control::MultiCoreCpu) 
		{
            #ifdef ENABLE_TBB
                serialCPU_sort_by_key(keys_first, keys_last, values_first, comp);
                return;
            #else
                throw std::runtime_error("The MultiCoreCpu Version of Sort_by_key is not enabled to be built with TBB!\n");
            #endif
        } 
		else 
		{
            
			device_vector< T_keys, concurrency::array_view > dvInputKeys(   keys_first, keys_last, false, ctl );
			device_vector<  T_values, concurrency::array_view > dvInputValues(  values_first, szElements, false, ctl );

            //Now call the actual AMP algorithm
            sort_by_key_enqueue(ctl,dvInputKeys.begin(),dvInputKeys.end(), dvInputValues.begin(), comp);
            //Map the buffer back to the host
            dvInputValues.data( );
            dvInputKeys.data( );
            return;
        }
    }


    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void sort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, 
                                    std::random_access_iterator_tag, std::random_access_iterator_tag )
    {
        return sort_by_key_pick_iterator( ctl, keys_first, keys_last, values_first,
                                    comp, 
                                    typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                    typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
    };

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void sort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, 
                                    std::input_iterator_tag, std::input_iterator_tag )
    {
        //  \TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
        //  to a temporary buffer.  Should we?
        static_assert(std::is_same< RandomAccessIterator1, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
        static_assert(std::is_same< RandomAccessIterator2, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
    };

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void sort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp,
                                    bolt::amp::fancy_iterator_tag, std::input_iterator_tag )
    {
        static_assert( std::is_same< RandomAccessIterator1, bolt::amp::fancy_iterator_tag >::value , "It is not possible to sort fancy iterators. They are not mutable" );
        static_assert( std::is_same< RandomAccessIterator2, std::input_iterator_tag >::value , "It is not possible to sort fancy iterators. They are not mutable" );
    }

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void sort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, 
                                    std::input_iterator_tag, bolt::amp::fancy_iterator_tag )
    {


        static_assert( std::is_same< RandomAccessIterator2, bolt::amp::fancy_iterator_tag >::value , "It is not possible to sort fancy iterators. They are not mutable" );
        static_assert( std::is_same< RandomAccessIterator1, std::input_iterator_tag >::value , "It is not possible to sort fancy iterators. They are not mutable" );

    }


}//namespace bolt::amp::detail


        template<typename RandomAccessIterator1 , typename RandomAccessIterator2>
        void sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first)
        {
            typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keys_T;

            detail::sort_by_key_detect_random_access( control::getDefault( ),
                                       keys_first, keys_last,
                                       values_first,
                                       less< keys_T >( ),
                                       typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                       typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
            return;
        }

        template<typename RandomAccessIterator1 , typename RandomAccessIterator2, typename StrictWeakOrdering>
        void sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         StrictWeakOrdering comp)
        {
            typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keys_T;

            detail::sort_by_key_detect_random_access( control::getDefault( ),
                                       keys_first, keys_last,
                                       values_first,
                                       comp,
                                       typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                       typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
            return;
        }

        template<typename RandomAccessIterator1 , typename RandomAccessIterator2>
        void sort_by_key(control &ctl,
                         RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first)
        {
            typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keys_T;

            detail::sort_by_key_detect_random_access( ctl,
                                       keys_first, keys_last,
                                       values_first,
                                       less< keys_T >( ),
                                       typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                       typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
            return;
        }

        template<typename RandomAccessIterator1 , typename RandomAccessIterator2, typename StrictWeakOrdering>
        void sort_by_key(control &ctl,
                         RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         StrictWeakOrdering comp)
        {
            typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keys_T;

            detail::sort_by_key_detect_random_access( ctl,
                                       keys_first, keys_last,
                                       values_first,
                                       comp,
                                       typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                       typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
            return;
        }

    }
};

#endif
