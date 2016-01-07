#pragma once

namespace details {

#define BOLT_UINT_MAX 0xFFFFFFFFU
#define BOLT_UINT_MIN 0x0U
#define BOLT_INT_MAX 0x7FFFFFFF
#define BOLT_INT_MIN 0x80000000

#define BITONIC_SORT_WGSIZE 64
/* \brief - SORT_CPU_THRESHOLD should be atleast 2 times the BITONIC_SORT_WGSIZE*/
#define SORT_CPU_THRESHOLD 128

    /*
template<typename InputIt, typename Compare>
typename std::enable_if<
std::is_same<
typename std::iterator_traits<InputIt >::value_type,
         unsigned int >::value>::type
sort_dispatch(InputIt first, InputIt last, Compare comp)
{

    typedef typename std::iterator_traits< InputIt >::value_type T;
    const int RADIX = 4;
    const int RADICES = (1 << RADIX);
    int szElements = static_cast<int>(std::distance(first, last));
    int computeUnits = 32;

    auto radix_common_kts;
    auto radix_uint_kts;

    int localSize  = 256;
    int wavefronts = 8;
    int numGroups = computeUnits * wavefronts;

    if(comp(2,3)) {
    } else {
    }
}
 
template<typename InputIt, typename Compare>
typename std::enable_if<std::is_same<typename std::iterator_traits<InputIt>::value_type, int>::value>::type
sort_dispatch(InputIt first, InputIt last, Compare comp)
{
    typedef typename std::iterator_traits< InputIt >::value_type T;
    const int RADIX = 4;
    const int RADICES = (1 << RADIX);
    int szElements = static_cast<int>(std::distance(first, last));
    int computeUnits     = 32;

    int localSize  = 256;
    int wavefronts = 8;
    int numGroups = computeUnits * wavefronts;

    T* dvSwapInputData = new T[szElements];
    T* dvHistogramBins = new T[localSize * RADICES];
    auto histogramAsc =
        [first_]
        (hc::tiled_index<1> tidx) [[hc]]
        {
            tile_static T lmem[WG_SIZE*RADICES];
            int gIdx = tidx.global[0];
            int lIdx = tidx.local[0];
            int wgIdx = tidx.group[0];
            int wgSize = 123;
            const int shift = cb.m_startBit;
            const int dataAlignment = 1024;
            const int n = cb.m_n;
            const int w_n = n + dataAlignment-(n%dataAlignment);

            const int nWGs = cb.m_nWGs;
            const int nBlocksPerWG = cb.m_nBlocksPerWG;

            for(int i=0; i<RADICES; i++)
                lmem[i*get_local_size(0)+ lIdx] = 0;
            tidx.barrier.wait();
            const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;
            int nBlocks = (w_n)/blockSize - nBlocksPerWG*wgIdx;
            int addr = blockSize*nBlocksPerWG*wgIdx + lIdx;
            for(int iblock=0; iblock<min(nBlocksPerWG, nBlocks); iblock++)
            {
                for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++, addr+=WG_SIZE )
                {
#if defined(DESCENDING)
                    lmem[(RADICES - local_key -1)*get_local_size(0)+ lIdx]++;   
#else
                    lmem[local_key*get_local_size(0)+ lIdx]++;
#endif
                }
            }
            tidx.barrier.wait();
            if (lIdx < RADICES)
            {
                uint sum = 0;
                for(int i=0; i<get_local_size(0); i++)
                {
                    sum += lmem[lIdx*get_local_size(0)+ i];
                }
                isums[lIdx * get_num_groups(0) + get_group_id(0)] = sum;

            }
        };
    if(comp(2,3))
    {
    }
    else
    {
    }

    int swap = 0;
    const int ELEMENTS_PER_WORK_ITEM = 4;
    int blockSize = (int)(ELEMENTS_PER_WORK_ITEM*localSize);
    int nBlocks = (int)(szElements + blockSize-1)/(blockSize);

    struct b3ConstData
    {
        int m_n;
        int m_nWGs;
        int m_startBit;
        int m_nBlocksPerWG;
    };
    b3ConstData cdata;

    cdata.m_n = (int)szElements;
    cdata.m_nWGs = (int)numGroups;
    cdata.m_nBlocksPerWG = (int)(nBlocks + numGroups - 1)/numGroups;
    if(nBlocks < numGroups)
    {
        cdata.m_nBlocksPerWG = 1;
        numGroups = nBlocks;
        cdata.m_nWGs = numGroups;
    }

    int bits = 0;
    for(bits = 0; bits < (sizeof(T) * 7); bits += RADIX)
    {
        cdata.m_startBit = bits;
        if (swap == 0)
        {
        }
        else
        {
        }
        swap = swap? 0: 1;
    }
}
*/


template<typename InputIt, typename Compare>
// typename std::enable_if<
//     !(std::is_same< typename std::iterator_traits<InputIt >::value_type, unsigned int >::value
//    || std::is_same< typename std::iterator_traits<InputIt >::value_type,          int >::value
//     )
//                        >::type
void sort_dispatch(const InputIt& first, const InputIt& last, const Compare& comp)
{
    typedef typename std::iterator_traits< InputIt >::value_type T;
    size_t szElements = static_cast< size_t >( std::distance( first, last ) );

    size_t wgSize  = BITONIC_SORT_WGSIZE;
    if((szElements/2) < BITONIC_SORT_WGSIZE)
        wgSize = (int)szElements/2;
    unsigned int stage, passOfStage, numStages = 0;
    for(size_t temp = szElements; temp > 1; temp >>= 1)
        ++numStages;

    auto first_ = utils::get_pointer(first);
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

  // call to std::sort when small data size
  if (N <= details::PARALLELIZE_THRESHOLD) {
      std::sort(first, last, comp);
  }
  sort_dispatch(first, last, comp);
}

} // namespace details
