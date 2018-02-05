#pragma once

namespace details {

// FIXME sort algorithm implementation breaks Clang 4.0 as of now
#if 0

#define WG_SIZE                 256
#define BITONIC_SORT_WGSIZE     64
#define RADICES                 16
#define ELEMENTS_PER_WORK_ITEM  4
#define BITS_PER_PASS			4
#define NUM_BUCKET				(1<<BITS_PER_PASS)


#define AtomInc(x) hc::atomic_fetch_inc(&(x))
#define AtomAdd(x, value) hc::atomic_fetch_add(&(x), value)
#define USE_2LEVEL_REDUCE 
#define _max(a,b)    (((a) > (b)) ? (a) : (b))
#define _min(a,b)    (((a) < (b)) ? (a) : (b))
#define SET_HISTOGRAM(setIdx, key) ldsSortData[(setIdx)*NUM_BUCKET+key]

struct uint_4
{
    uint_4() [[hc]] : x(0), y(0), z(0), w(0) {}
    uint_4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) [[hc]]
        : x(x), y(y), z(z), w(w) {}
    unsigned int x, y, z, w;
    uint_4 operator+=(const uint_4& other) [[hc]] {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }
    uint_4 operator-=(const uint_4& other) [[hc]] {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }
};
uint_4 operator+(const uint_4& a, const uint_4& b) [[hc]]
{
    uint_4 ret(a);
    ret += b;
    return ret;
}
uint_4 operator-(const uint_4& a, const uint_4& b) [[hc]]
{
    uint_4 ret(a);
    ret -= b;
    return ret;
}
#define make_uint4 uint_4

static inline uint_4 SELECT_UINT4(uint_4 &a,uint_4 &b,uint_4  &condition )  [[hc]]
{
	uint_4 res;
	res.x = (condition.x )? b.x : a.x;
	res.y = (condition.y )? b.y : a.y;
	res.z = (condition.z )? b.z : a.z;
	res.w = (condition.w )? b.w : a.w;
	return res;

}
static unsigned int scanLocalMemAndTotal(unsigned int val, unsigned int* lmem, unsigned int *totalSum, int exclusive, hc::tiled_index< 1 > t_idx) [[hc]]
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
static unsigned int prefixScanVectorEx( uint_4* data ) [[hc]]
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
static uint_4 localPrefixSum256V( uint_4 pData, unsigned int lIdx, unsigned int* totalSum, unsigned int* sorterSharedMemory, hc::tiled_index< 1 > t_idx ) [[hc]]
{
	unsigned int s4 = prefixScanVectorEx( &pData );
	unsigned int rank = scanLocalMemAndTotal( s4, sorterSharedMemory, totalSum,  1, t_idx);
	return pData + make_uint4( rank, rank, rank, rank );
}
static void sort4BitsKeyValueAscending(unsigned int sortData[4],  const int startBit, int lIdx,  unsigned int* ldsSortData,  bool Asc_sort, hc::tiled_index< 1 > t_idx) [[hc]]
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

static void sort4BitsSignedKeyValueAscending(unsigned int sortData[4],  const int startBit, int lIdx,  unsigned int* ldsSortData, bool Asc_sort, hc::tiled_index< 1 > t_idx) [[hc]]
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
                            hc::tiled_index< 1 > t_idx) restrict (amp)
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

template<typename InputIt, typename Compare>
void sort_enqueue_int_uint(InputIt &first, InputIt &last, Compare comp, bool int_flag)
{
  
	typedef typename std::iterator_traits< InputIt >::value_type Values;
    const int RADIX = 4;
    int orig_szElements = static_cast<int>(std::distance(first, last));
	const unsigned int localSize  = WG_SIZE;

	unsigned int szElements = (unsigned int)orig_szElements;
    unsigned int modWgSize = (szElements & ((localSize)-1));
    if( modWgSize )
    {
        szElements &= ~modWgSize;
        szElements += (localSize);
    }
	unsigned int numGroups = (szElements/localSize)>= 32?(32*8):(szElements/localSize);

    hc::array_view<Values> dvSwapInputValues((hc::extent<1>(orig_szElements)));
    hc::array_view<int> dvHistogramBins(hc::extent<1>(numGroups * RADICES));

	bool Asc_sort = 0;
	if(comp(2,3))
       Asc_sort = 1;
	int swap = 0;
    unsigned int blockSize = (int)(ELEMENTS_PER_WORK_ITEM*localSize);
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

	int bits;
    auto f_ = utils::get_pointer(first);
    hc::array_view<Values> first_(hc::extent<1>(orig_szElements), f_);
	for(bits = 0; bits < (sizeof(Values) * 8); bits += RADIX)
    {
          cdata.m_startBit = bits;
		  kernel_launch( numGroups * localSize, 
				[
					first_,
					dvSwapInputValues,
					dvHistogramBins,
					cdata,
					swap,
					Asc_sort,
					int_flag,
                    numGroups
				] ( hc::tiled_index< 1 > t_idx ) [[hc]]
		  {
			tile_static unsigned int lmem[WG_SIZE*RADICES];
			unsigned int gIdx = t_idx.global[ 0 ];
			unsigned int lIdx = t_idx.local[ 0 ];
			unsigned int wgIdx = t_idx.tile[ 0 ];
			unsigned int localSize = t_idx.tile_dim[ 0 ];

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
								local_key = (first_[addr] >> shift);
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
								local_key = (first_[addr] >> shift) & 0xFU;
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
		}, localSize);

	
		kernel_launch( numGroups * localSize, 
				[
					dvHistogramBins,
					numGroups
				] ( hc::tiled_index< 1 > t_idx ) [[hc]]
		  {

			unsigned int lIdx = t_idx.local[ 0 ];
			unsigned int llIdx = lIdx;
			unsigned int wgSize = t_idx.tile_dim[ 0 ];

			tile_static unsigned int lmem[WG_SIZE*8];
			tile_static int s_seed; 
			s_seed = 0;
			t_idx.barrier.wait();
    
			bool last_thread = (lIdx < numGroups && (lIdx+1) == numGroups) ? 1 : 0;
			for (int d = 0; d < 16; d++)
			{
				unsigned int val = 0;
				if (lIdx < numGroups)
					val = dvHistogramBins[(numGroups * d) + lIdx];
				unsigned int res =  scanlMemPrivData(val, lmem,1, t_idx);
				if (lIdx < numGroups)
					dvHistogramBins[(numGroups * d) + lIdx] = res + s_seed;
				if (last_thread) 
					s_seed += res + val;
				t_idx.barrier.wait();
			}

		}, localSize);
		if((bits >= sizeof(Values) * 7) && int_flag)
			break;
		kernel_launch( localSize, 
				[
					first_,
					dvSwapInputValues,
					dvHistogramBins,
					cdata,
					swap,
					Asc_sort
				] ( hc::tiled_index< 1 > t_idx ) [[hc]]
		  {

			tile_static unsigned int ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];
			tile_static unsigned int localHistogramToCarry[NUM_BUCKET];
			tile_static unsigned int localHistogram[NUM_BUCKET*2];

			unsigned int gIdx = t_idx.global[ 0 ];
			unsigned int lIdx = t_idx.local[ 0 ];
			unsigned int wgIdx = t_idx.tile[ 0 ];
			unsigned int localSize = t_idx.tile_dim[ 0 ];

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
							sortData[i] = ( addr+i < n )? first_[ addr+i ] : 0x0;
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
							sortData[i] = ( addr+i < n )? first_[ addr+i ] : 0xffffffff;
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
									first_[ groupOffset + myIdx ] = sortData[ie];
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
		 }, localSize);
		 swap = swap? 0: 1;
	}
	if(int_flag)
	{
		cdata.m_startBit = bits;
		kernel_launch( numGroups * localSize, 
				[
					first_,
					dvSwapInputValues,
					dvHistogramBins,
					cdata,
					swap,
					Asc_sort
				] ( hc::tiled_index< 1 > t_idx ) [[hc]]
		  {

			tile_static unsigned int ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+WG_SIZE];
			tile_static unsigned int localHistogramToCarry[NUM_BUCKET];
			tile_static unsigned int localHistogram[NUM_BUCKET*2];

			unsigned int gIdx = t_idx.global[ 0 ];
			unsigned int lIdx = t_idx.local[ 0 ];
			unsigned int wgIdx = t_idx.tile[ 0 ];
			unsigned int localSize = t_idx.tile_dim[0];

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
									first_[ groupOffset + myIdx ] = sortData[ie];
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
		 }, localSize);
	}
    return;
}


template<typename InputIt, typename Compare>
typename std::enable_if<
std::is_same<
typename std::iterator_traits<InputIt >::value_type,
         unsigned int >::value>::type
sort_dispatch(InputIt first, InputIt last, Compare comp)
{
    sort_enqueue_int_uint(first, last, comp, 0);
}
 
template<typename InputIt, typename Compare>
typename std::enable_if<std::is_same<typename std::iterator_traits<InputIt>::value_type, int>::value>::type
sort_dispatch(InputIt first, InputIt last, Compare comp)
{
    sort_enqueue_int_uint(first, last, comp, 1);
}


template<typename InputIt, typename Compare>
typename std::enable_if<
    !(std::is_same< typename std::iterator_traits<InputIt >::value_type, unsigned int >::value
   || std::is_same< typename std::iterator_traits<InputIt >::value_type,          int >::value
    )
    >::type
sort_dispatch(const InputIt& first, const InputIt& last, const Compare& comp)
{
    typedef typename std::iterator_traits< InputIt >::value_type T;
    size_t szElements = static_cast< size_t >( std::distance( first, last ) );

    size_t wgSize  = BITONIC_SORT_WGSIZE;
    if((szElements/2) < BITONIC_SORT_WGSIZE)
        wgSize = (int)szElements/2;
    unsigned int stage, passOfStage, numStages = 0;
    for(size_t temp = szElements; temp > 1; temp >>= 1)
        ++numStages;

    auto f_ = utils::get_pointer(first);
    hc::array_view<T> first_(hc::extent<1>(szElements), f_);
    for(stage = 0; stage < numStages; ++stage)
    {
        for(passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            auto ker = [first_, szElements, comp, stage, passOfStage]
                (hc::tiled_index<1> tidx) [[hc]]
            {
                int threadId = tidx.global[0];
                int pairDistance = 1 << (stage - passOfStage);
                int blockWidth = 2 * pairDistance;
                int leftId = (threadId & (pairDistance - 1))
                    + (threadId >> (stage - passOfStage) ) * blockWidth;


                int rightId = leftId + pairDistance;
                T leftElement = first_[leftId];
                T rightElement = first_[rightId];
                unsigned int sameDirectionBlockWidth = threadId >> stage;
                unsigned int sameDirection = sameDirectionBlockWidth & 0x1;


                int temp    = sameDirection?rightId:temp;
                rightId = sameDirection?leftId:rightId;
                leftId  = sameDirection?temp:leftId;

                bool compareResult = comp(leftElement, rightElement);


                T greater = compareResult?rightElement:leftElement;
                T lesser  = compareResult?leftElement:rightElement;
                first_[leftId] = lesser;
                first_[rightId] = greater;
            };
            kernel_launch(szElements / 2, ker, wgSize);
        }
    }
}
#endif
 
template<class InputIt, class Compare>
void sort_impl(InputIt first, InputIt last, Compare comp, std::input_iterator_tag) {
    std::sort(first, last, comp);
}
 

template<class InputIt, class Compare>
void sort_impl(InputIt first, InputIt last, Compare comp,
               std::random_access_iterator_tag) {
  unsigned N = std::distance(first, last);
  if (N == 0)
      return;

  // FIXME sort algorithm implementation breaks Clang 4.0 as of now
#if 0
  // call to std::sort when small data size
  if (N <= details::PARALLELIZE_THRESHOLD) {
      std::sort(first, last, comp);
  }
  sort_dispatch(first, last, comp);
#endif

  std::sort(first, last, comp);
}


} // namespace details
