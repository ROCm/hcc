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
#if !defined( BOLT_CL_SORT_BY_KEY_INL )
#define BOLT_CL_SORT_BY_KEY_INL


#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/sort_by_key.h"
#endif

#include "bolt/cl/stablesort_by_key.h"

#define BITONIC_SORT_WGSIZE 64
#define DEBUG 1
namespace bolt {
namespace cl {

namespace detail {
 

	 template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
                         StrictWeakOrdering comp, const std::string& cl_code);

	 template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           unsigned int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
                         StrictWeakOrdering comp, const std::string& cl_code);

  template< typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if<
        !( std::is_same< typename std::iterator_traits<DVKeys >::value_type, unsigned int >::value ||
           std::is_same< typename std::iterator_traits<DVKeys >::value_type, int >::value 
         )
                           >::type
    sort_by_key_enqueue(control &ctl, const DVKeys& keys_first,
                        const DVKeys& keys_last, const DVValues& values_first,
                        const StrictWeakOrdering& comp, const std::string& cl_code);

	


enum sortByKeyTypes {sort_by_key_keyValueType, sort_by_key_keyIterType,
                     sort_by_key_valueValueType, sort_by_key_valueIterType,
                     sort_by_key_StrictWeakOrdering, sort_by_key_end };

    /*Radix Sort Kernel Template specializers*/
    class RadixSortByKey_Int_KernelTemplateSpecializer : public KernelTemplateSpecializer
    {
    private:

    public:
        RadixSortByKey_Int_KernelTemplateSpecializer() : KernelTemplateSpecializer()
        {
            addKernelName("permuteByKeySignedAsc");
            addKernelName("permuteByKeySignedDesc");
        }

        const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
        {
            const std::string templateSpecializationString = 
                "// Host generates this instantiation string with user-specified value type and functor\n"
                "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
                "         __attribute__((reqd_work_group_size(WG_SIZE,1,1))) \n"
                "kernel void permuteByKeySignedAscTemplate( __global const u32* restrict gKeys, \n"
                "                                                 __global const " + typeNames[sort_by_key_valueValueType] + " * restrict gValues, \n"
                "                                                 __global const u32* rHistogram, \n"
                "                                                 __global u32* restrict gDstKeys, \n"
                "                                                 __global " + typeNames[sort_by_key_valueValueType] + " * restrict gDstValues, \n"
                "                                                 int4 cb); \n\n"
                "template __attribute__((mangled_name(" + name(1) + "Instantiated)))\n"
                "         __attribute__((reqd_work_group_size(WG_SIZE,1,1))) \n"
                "kernel void permuteByKeySignedDescTemplate( __global const u32* restrict gKeys, \n"
                "                                                 __global const " + typeNames[sort_by_key_valueValueType] + " * restrict gValues, \n"
                "                                                 __global const u32* rHistogram, \n"
                "                                                 __global u32* restrict gDstKeys, \n"
                "                                                 __global " + typeNames[sort_by_key_valueValueType] + " * restrict gDstValues, \n"
                "                                                 int4 cb); \n\n";
            return templateSpecializationString;
        }
    };

    class RadixSortByKey_Uint_KernelTemplateSpecializer : public KernelTemplateSpecializer
    {
    private:
        int _radix;
    public:
        RadixSortByKey_Uint_KernelTemplateSpecializer() : KernelTemplateSpecializer()
        {
            addKernelName("permuteByKeyAsc");
            addKernelName("permuteByKeyDesc");
        }

        const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
        {
            const std::string templateSpecializationString =

                "// Host generates this instantiation string with user-specified value type and functor\n"
                "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
                "         __attribute__((reqd_work_group_size(WG_SIZE,1,1))) \n"
                "kernel void permuteByKeyAscTemplate( __global const u32* restrict gKeys, \n"
                "                                     __global const " + typeNames[sort_by_key_valueValueType] + " * restrict gValues, \n"
                "                                     __global const u32* rHistogram, \n"
                "                                     __global u32* restrict gDstKeys, \n"
                "                                     __global " + typeNames[sort_by_key_valueValueType] + " * restrict gDstValues, \n"
                "                                     int4 cb); \n\n"
                "template __attribute__((mangled_name(" + name(1) + "Instantiated)))\n"
                "         __attribute__((reqd_work_group_size(WG_SIZE,1,1))) \n"
                "kernel void permuteByKeyDescTemplate( __global const u32* restrict gKeys, \n"
                "                                      __global const " + typeNames[sort_by_key_valueValueType] + " * restrict gValues, \n"
                "                                      __global const u32* rHistogram, \n"
                "                                      __global u32* restrict gDstKeys, \n"
                "                                      __global " + typeNames[sort_by_key_valueValueType] + " * restrict gDstValues, \n"
                "                                      int4 cb); \n\n";
            return templateSpecializationString;


        }
    };

    class RadixSortByKey_Common_KernelTemplateSpecializer : public KernelTemplateSpecializer
    {
    private:
    public:
        RadixSortByKey_Common_KernelTemplateSpecializer() : KernelTemplateSpecializer()
        {
            addKernelName("histogramAsc");
            addKernelName("histogramDesc");
            addKernelName("histogramSignedAsc");
            addKernelName("histogramSignedDesc");
            addKernelName("scan");
        }

        const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
        {
            const std::string templateSpecializationString = "\n //RadixSortByKey_Common_KernelTemplateSpecializer\n";
            return templateSpecializationString;
        }
    };

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
    void serialCPU_sort_by_key( const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                const RandomAccessIterator2 values_first,
                                const StrictWeakOrdering& comp)
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< RandomAccessIterator2 >::value_type valType;
        typedef std_sort<keyType, valType> KeyValuePair;
        typedef std_sort_comp<keyType, valType, StrictWeakOrdering> KeyValuePairFunctor;

        size_t vecSize = std::distance( keys_first, keys_last );
        std::vector<KeyValuePair> KeyValuePairVector(vecSize);
        KeyValuePairFunctor functor(comp);
        //Zip the key and values iterators into a std_sort vector.
        for (size_t i=0; i< vecSize; i++)
        {
            KeyValuePairVector[i].key   = *(keys_first + i);
            KeyValuePairVector[i].value = *(values_first + i);
        }
        //Sort the std_sort vector using std::sort
        std::sort(KeyValuePairVector.begin(), KeyValuePairVector.end(), functor);
        //Extract the keys and values from the KeyValuePair and fill the respective iterators.
        for (size_t i=0; i< vecSize; i++)
        {
            *(keys_first + i)   = KeyValuePairVector[i].key;
            *(values_first + i) = KeyValuePairVector[i].value;
        }
    }

 template< typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if<
        !( std::is_same< typename std::iterator_traits<DVKeys >::value_type, unsigned int >::value ||
           std::is_same< typename std::iterator_traits<DVKeys >::value_type, int >::value 
         )
                           >::type
    sort_by_key_enqueue(control &ctl, const DVKeys& keys_first,
                        const DVKeys& keys_last, const DVValues& values_first,
                        const StrictWeakOrdering& comp, const std::string& cl_code)
    {
        stablesort_by_key_enqueue(ctl, keys_first, keys_last, values_first, comp, cl_code);
        return;
    }// END of sort_by_key_enqueue

    template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           unsigned int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
                         StrictWeakOrdering comp, const std::string& cl_code)
{
    typedef typename std::iterator_traits< DVKeys >::value_type Keys;
    typedef typename std::iterator_traits< DVValues >::value_type Values;
    const int RADIX = 4; //Now you cannot replace this with Radix 8 since there is a
                         //local array of 16 elements in the histogram kernel.

    const int RADICES = (1 << RADIX); //Values handeled by each work-item?

    int orig_szElements = static_cast<int>(std::distance(keys_first, keys_last));
    int szElements = orig_szElements;

    int computeUnits     = ctl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    cl_int l_Error = CL_SUCCESS;

    std::vector<std::string> typeNames( sort_by_key_end );
    typeNames[sort_by_key_keyValueType]         = TypeName< Keys >::get( );
    //typeNames[sort_by_key_keyIterType]          = TypeName< DVKeys >::get( );
    typeNames[sort_by_key_valueValueType]         = TypeName< Values >::get( );
    //typeNames[sort_by_key_valueIterType]          = TypeName< DVValues >::get( );
    typeNames[sort_by_key_StrictWeakOrdering] = TypeName< StrictWeakOrdering >::get();

    std::vector<std::string> typeDefinitions;
    PUSH_BACK_UNIQUE( typeDefinitions, cl_code )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< Keys >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< Values >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< StrictWeakOrdering  >::get() )

    bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
    /*\TODO - Do CPU specific kernel work group size selection here*/

    std::string compileOptions;
    RadixSortByKey_Common_KernelTemplateSpecializer radix_common_kts;
    std::vector< ::cl::Kernel > commonKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_common_kts,
        typeDefinitions,
        sort_common_kernels,
        compileOptions);

    RadixSortByKey_Uint_KernelTemplateSpecializer radix_uint_kts;
    std::vector< ::cl::Kernel > uintKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_uint_kts,
        typeDefinitions,
        sort_by_key_uint_kernels,
        compileOptions);

    int localSize  = 256;
    int wavefronts = 8;
    int numGroups = computeUnits * wavefronts;

    device_vector< Keys >           dvSwapInputKeys( szElements, 0, CL_MEM_READ_WRITE, false, ctl);
    device_vector< Values >         dvSwapInputValues( szElements, 0, CL_MEM_READ_WRITE, false, ctl);
    device_vector< unsigned int >   dvHistogramBins( (localSize * RADICES), 0, CL_MEM_READ_WRITE, false, ctl);

    ::cl::Buffer clInputKeys   = keys_first.getContainer( ).getBuffer();
    ::cl::Buffer clInputValues = values_first.getContainer( ).getBuffer();
    ::cl::Buffer clSwapKeys    = dvSwapInputKeys.begin( ).getContainer().getBuffer();
    ::cl::Buffer clSwapValues  = dvSwapInputValues.begin( ).getContainer().getBuffer();
    ::cl::Buffer clHistData    = dvHistogramBins.begin( ).getContainer().getBuffer();

    ::cl::Kernel histKernel;
    ::cl::Kernel permuteKernel;
    ::cl::Kernel scanLocalKernel;
    if(comp(2,3))
    {
        /*Ascending Sort*/
        histKernel = commonKernels[0];
        scanLocalKernel = commonKernels[4];
        permuteKernel = uintKernels[0];
    }
    else
    {
        /*Descending Sort*/
        histKernel = commonKernels[1];
        scanLocalKernel = commonKernels[4];
        permuteKernel = uintKernels[1];
    }

        int swap = 0;
        const int ELEMENTS_PER_WORK_ITEM = 4;
        int blockSize = (int)(ELEMENTS_PER_WORK_ITEM*localSize);//set at 1024
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
		//cdata.m_startBit = shift; //Shift value is set inside the for loop.
		cdata.m_nBlocksPerWG = (int)(nBlocks + numGroups - 1)/numGroups;
        if(nBlocks < numGroups)
        {
			cdata.m_nBlocksPerWG = 1;
			numGroups = nBlocks;
            cdata.m_nWGs = numGroups;
        }

    //Set Histogram kernel arguments
    V_OPENCL( histKernel.setArg(1, clHistData), "Error setting a kernel argument" );

    //Set Scan kernel arguments
    V_OPENCL( scanLocalKernel.setArg(0, clHistData), "Error setting a kernel argument" );
    V_OPENCL( scanLocalKernel.setArg(1, (int)numGroups), "Error setting a kernel argument" );
    V_OPENCL( scanLocalKernel.setArg(2, localSize * 2 * sizeof(Keys),NULL), "Error setting a kernel argument" );
    
    //Set Permute kernel arguments
    V_OPENCL( permuteKernel.setArg(2, clHistData), "Error setting a kernel argument" );

    for(int bits = 0; bits < (sizeof(Keys) * 8)/*Bits per Byte*/; bits += RADIX)
    {
        //Launch Kernel
        cdata.m_startBit = bits;
        //Histogram Kernel
        V_OPENCL( histKernel.setArg(2, cdata), "Error setting a kernel argument" );
        if (swap == 0)
            V_OPENCL( histKernel.setArg(0, clInputKeys), "Error setting a kernel argument" );
        else
            V_OPENCL( histKernel.setArg(0, clSwapKeys), "Error setting a kernel argument" );

        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            histKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize), //This mul will be removed when permute is optimized
                            NULL,
                            NULL);
#if defined(DEBUG_ENABLED)
        {
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            printf("histogramAscending Kernel global_wsize=%d, local_wsize=%d\n", localSize * numGroups, localSize);            

            unsigned int * temp = dvHistogramBins.data().get();
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            //DEBUG LOOP
            printf("Un-Scanned result\n");
            for (int jj=0;jj<(numGroups* RADICES);jj++)
            {
                printf(" %d", temp[jj] );
            }
            printf("\n\n"); 
        }

#endif

        //Launch Local Scan Kernel
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            scanLocalKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(localSize),
                            ::cl::NDRange(localSize), //This mul will be removed when permute is optimized
                            NULL,
                            NULL);

        //Launch Permute Kernel
#if defined(DEBUG_ENABLED)
        {
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            printf("histogramAscending Kernel global_wsize=%d, local_wsize=%d\n", localSize * numGroups, localSize);            

            unsigned int * temp = dvHistogramBins.data().get();
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            //DEBUG LOOP
            printf("Scanned result\n");
            for (int jj=0;jj<(numGroups* RADICES);jj++)
            {
                printf(" %d", temp[jj] );
            }
            printf("\n\n");
        }

#endif
        //void permuteAscendingRadixNTemplateInstantiated( __global const u32* restrict gKeys, __global const Values* restrict gValues, 
        //                     __global const u32* rHistogram, __global u32* restrict gDstKeys, __global const Values* restrict gDstValues, int4 cb);
        V_OPENCL( permuteKernel.setArg( 5, cdata), "Error setting a kernel argument" );        
        if (swap == 0)
        {
            V_OPENCL( permuteKernel.setArg(0, clInputKeys), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(3, clSwapKeys), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(1, clInputValues), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(4, clSwapValues), "Error setting kernel argument" );
        }
        else
        {
            V_OPENCL( permuteKernel.setArg(0, clSwapKeys), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(3, clInputKeys), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(1, clSwapValues), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(4, clInputValues), "Error setting kernel argument" );
        }
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            permuteKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize),
                            NULL,
                            NULL);
        /*For swapping the buffers*/
        swap = swap? 0: 1;
    }

    V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
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
                         StrictWeakOrdering comp, const std::string& cl_code)
{
    typedef typename std::iterator_traits< DVKeys >::value_type Keys;
    typedef typename std::iterator_traits< DVValues >::value_type Values;
    const int RADIX = 4; //Now you cannot replace this with Radix 8 since there is a
                         //local array of 16 elements in the histogram kernel.

    const int RADICES = (1 << RADIX); //Values handeled by each work-item?

    int orig_szElements = static_cast<int>(std::distance(keys_first, keys_last));
    int szElements = orig_szElements;

    int computeUnits     = ctl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    cl_int l_Error = CL_SUCCESS;

    std::vector<std::string> typeNames( sort_by_key_end );
    typeNames[sort_by_key_keyValueType]         = TypeName< Keys >::get( );
    //typeNames[sort_by_key_keyIterType]          = TypeName< DVKeys >::get( );
    typeNames[sort_by_key_valueValueType]         = TypeName< Values >::get( );
    //typeNames[sort_by_key_valueIterType]          = TypeName< DVValues >::get( );
    typeNames[sort_by_key_StrictWeakOrdering] = TypeName< StrictWeakOrdering >::get();

    std::vector<std::string> typeDefinitions;
    PUSH_BACK_UNIQUE( typeDefinitions, cl_code )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< Keys >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< Values >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< StrictWeakOrdering  >::get() )

    bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
    /*\TODO - Do CPU specific kernel work group size selection here*/

    std::string compileOptions;
    RadixSortByKey_Common_KernelTemplateSpecializer radix_common_kts;
    std::vector< ::cl::Kernel > commonKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_common_kts,
        typeDefinitions,
        sort_common_kernels,
        compileOptions);

    RadixSortByKey_Uint_KernelTemplateSpecializer radix_uint_kts;
    std::vector< ::cl::Kernel > uintKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_uint_kts,
        typeDefinitions,
        sort_by_key_uint_kernels,
        compileOptions);

    RadixSortByKey_Int_KernelTemplateSpecializer radix_int_kts;
    std::vector< ::cl::Kernel > intKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_int_kts,
        typeDefinitions,
        sort_by_key_int_kernels,
        compileOptions);

    int localSize  = 256;
    int wavefronts = 8;
    int numGroups = computeUnits * wavefronts;

    device_vector< Keys >           dvSwapInputKeys( szElements, 0, CL_MEM_READ_WRITE, false, ctl);
    device_vector< Values >         dvSwapInputValues( szElements, 0, CL_MEM_READ_WRITE, false, ctl);
    device_vector< unsigned int >   dvHistogramBins( (localSize * RADICES), 0, CL_MEM_READ_WRITE, false, ctl);

    ::cl::Buffer clInputKeys   = keys_first.getContainer( ).getBuffer();
    ::cl::Buffer clInputValues = values_first.getContainer( ).getBuffer();
    ::cl::Buffer clSwapKeys    = dvSwapInputKeys.begin( ).getContainer().getBuffer();
    ::cl::Buffer clSwapValues  = dvSwapInputValues.begin( ).getContainer().getBuffer();
    ::cl::Buffer clHistData    = dvHistogramBins.begin( ).getContainer().getBuffer();

    ::cl::Kernel histKernel;
    ::cl::Kernel histSignedKernel;
    ::cl::Kernel permuteKernel;
    ::cl::Kernel permuteSignedKernel;
    ::cl::Kernel scanLocalKernel;
    if(comp(2,3))
    {
        /*Ascending Sort*/
        histKernel = commonKernels[0];
        histSignedKernel = commonKernels[2]; 
        scanLocalKernel = commonKernels[4];
        permuteKernel = uintKernels[0];
        permuteSignedKernel = intKernels[0];
    }
    else
    {
        /*Descending Sort*/
        histKernel = commonKernels[1];
        histSignedKernel = commonKernels[3];
        scanLocalKernel = commonKernels[4];
        permuteKernel = uintKernels[1];
        permuteSignedKernel = intKernels[1];
    }

        int swap = 0;
        const int ELEMENTS_PER_WORK_ITEM = 4;
        int blockSize = (int)(ELEMENTS_PER_WORK_ITEM*localSize);//set at 1024
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
		//cdata.m_startBit = shift; //Shift value is set inside the for loop.
		cdata.m_nBlocksPerWG = (int)(nBlocks + numGroups - 1)/numGroups;
        if(nBlocks < numGroups)
        {
			cdata.m_nBlocksPerWG = 1;
			numGroups = nBlocks;
            cdata.m_nWGs = numGroups;
        }

    //Set Histogram kernel arguments
    V_OPENCL( histKernel.setArg(1, clHistData), "Error setting a kernel argument" );

    //Set Scan kernel arguments
    V_OPENCL( scanLocalKernel.setArg(0, clHistData), "Error setting a kernel argument" );
    V_OPENCL( scanLocalKernel.setArg(1, (int)numGroups), "Error setting a kernel argument" );
    V_OPENCL( scanLocalKernel.setArg(2, localSize * 2 * sizeof(Keys),NULL), "Error setting a kernel argument" );
    
    //Set Permute kernel arguments
    V_OPENCL( permuteKernel.setArg(2, clHistData), "Error setting a kernel argument" );
    int bits = 0;
    for(bits = 0; bits < (sizeof(Keys) * 7)/*Bits per Byte*/; bits += RADIX)
    {
        //Launch Kernel
        cdata.m_startBit = bits;
        //Histogram Kernel
        V_OPENCL( histKernel.setArg(2, cdata), "Error setting a kernel argument" );
        if (swap == 0)
            V_OPENCL( histKernel.setArg(0, clInputKeys), "Error setting a kernel argument" );
        else
            V_OPENCL( histKernel.setArg(0, clSwapKeys), "Error setting a kernel argument" );

        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            histKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize), //This mul will be removed when permute is optimized
                            NULL,
                            NULL);
#if defined(DEBUG_ENABLED)
        {
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            printf("histogramAscending Kernel global_wsize=%d, local_wsize=%d\n", localSize * numGroups, localSize);            

            unsigned int * temp = dvHistogramBins.data().get();
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            //DEBUG LOOP
            printf("Un-Scanned result\n");
            for (int jj=0;jj<(numGroups* RADICES);jj++)
            {
                printf(" %d", temp[jj] );
            }
            printf("\n\n"); 
        }

#endif

        //Launch Local Scan Kernel
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            scanLocalKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(localSize),
                            ::cl::NDRange(localSize), //This mul will be removed when permute is optimized
                            NULL,
                            NULL);

        //Launch Permute Kernel
#if defined(DEBUG_ENABLED)
        {
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            printf("histogramAscending Kernel global_wsize=%d, local_wsize=%d\n", localSize * numGroups, localSize);            

            unsigned int * temp = dvHistogramBins.data().get();
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            //DEBUG LOOP
            printf("Scanned result\n");
            for (int jj=0;jj<(numGroups* RADICES);jj++)
            {
                printf(" %d", temp[jj] );
            }
            printf("\n\n");
        }

#endif
        //void permuteAscendingRadixNTemplateInstantiated( __global const u32* restrict gKeys, __global const Values* restrict gValues, 
        //                     __global const u32* rHistogram, __global u32* restrict gDstKeys, __global const Values* restrict gDstValues, int4 cb);
        V_OPENCL( permuteKernel.setArg( 5, cdata), "Error setting a kernel argument" );        
        if (swap == 0)
        {
            V_OPENCL( permuteKernel.setArg(0, clInputKeys), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(3, clSwapKeys), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(1, clInputValues), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(4, clSwapValues), "Error setting kernel argument" );
        }
        else
        {
            V_OPENCL( permuteKernel.setArg(0, clSwapKeys), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(3, clInputKeys), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(1, clSwapValues), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(4, clInputValues), "Error setting kernel argument" );
        }
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            permuteKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize),
                            NULL,
                            NULL);
        /*For swapping the buffers*/
        swap = swap? 0: 1;
    }
    //Perform Signed nibble radix sort operations here operations here
    { 
        //Histogram Kernel
        cdata.m_startBit = bits;
        V_OPENCL( histSignedKernel.setArg(0, clSwapKeys), "Error setting a kernel argument" );
        V_OPENCL( histSignedKernel.setArg(1, clHistData), "Error setting a kernel argument" );
        V_OPENCL( histSignedKernel.setArg(2, cdata), "Error setting a kernel argument" );

        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            histSignedKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize), //This mul will be removed when permute is optimized
                            NULL,
                            NULL);
#if defined(DEBUG_ENABLED)
        {
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            printf("histogramAscending Kernel global_wsize=%d, local_wsize=%d\n", localSize * numGroups, localSize);            

            unsigned int * temp = dvHistogramBins.data().get();
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            //DEBUG LOOP
            printf("Un-Scanned result\n");
            for (int jj=0;jj<(numGroups* RADICES);jj++)
            {
                printf(" %d", temp[jj] );
            }
            printf("\n\n"); 
        }

#endif

        //Launch Local Scan Kernel
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            scanLocalKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(localSize),
                            ::cl::NDRange(localSize), //This mul will be removed when permute is optimized
                            NULL,
                            NULL);

        //Launch Permute Kernel
#if defined(DEBUG_ENABLED)
        {
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            printf("histogramAscending Kernel global_wsize=%d, local_wsize=%d\n", localSize * numGroups, localSize);            

            unsigned int * temp = dvHistogramBins.data().get();
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            //DEBUG LOOP
            printf("Scanned result\n");
            for (int jj=0;jj<(numGroups* RADICES);jj++)
            {
                printf(" %d", temp[jj] );
            }
            printf("\n\n");
        }

#endif


        V_OPENCL( permuteSignedKernel.setArg(0, clSwapKeys), "Error setting kernel argument" );
        V_OPENCL( permuteSignedKernel.setArg(1, clSwapValues), "Error setting kernel argument" );
        V_OPENCL( permuteSignedKernel.setArg(2, clHistData), "Error setting a kernel argument" );
        V_OPENCL( permuteSignedKernel.setArg(3, clInputKeys), "Error setting kernel argument" );
        V_OPENCL( permuteSignedKernel.setArg(4, clInputValues), "Error setting kernel argument" );
        V_OPENCL( permuteSignedKernel.setArg(5, cdata), "Error setting a kernel argument" );        

        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            permuteSignedKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize),
                            NULL,
                            NULL);
    }

    V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
    return;
}

    //Fancy iterator specialization
    template<typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering>
    void sort_by_key_pick_iterator(control &ctl, DVRandomAccessIterator1 keys_first,
                                   DVRandomAccessIterator1 keys_last, DVRandomAccessIterator2 values_first,
                                   StrictWeakOrdering comp, const std::string& cl_code, bolt::cl::fancy_iterator_tag )
    {
        static_assert( std::is_same<DVRandomAccessIterator1, bolt::cl::fancy_iterator_tag  >::value, "It is not possible to output to fancy iterators; they are not mutable! " );
    }

    //Device Vector specialization
    template<typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering>
    void sort_by_key_pick_iterator(control &ctl, DVRandomAccessIterator1 keys_first,
                                   DVRandomAccessIterator1 keys_last, DVRandomAccessIterator2 values_first,
                                   StrictWeakOrdering comp, const std::string& cl_code,
                                   bolt::cl::device_vector_tag, bolt::cl::device_vector_tag )
    {
        typedef typename std::iterator_traits< DVRandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< DVRandomAccessIterator2 >::value_type valueType;
        // User defined Data types are not supported with device_vector. Hence we have a static assert here.
        // The code here should be in compliant with the routine following this routine.
        size_t szElements = (size_t)(keys_last - keys_first);
        if (szElements == 0 )
                return;
        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode( );
        if( runMode == bolt::cl::control::Automatic )
        {
            runMode = ctl.getDefaultPathToRun( );
        }
		#if defined(BOLT_DEBUG_LOG)
        BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
        #endif
        if (runMode == bolt::cl::control::SerialCpu) {
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_SORTBYKEY,BOLTLOG::BOLT_SERIAL_CPU,"::Sort_By_Key::SERIAL_CPU");
            #endif
						
            typename bolt::cl::device_vector< keyType >::pointer   keysPtr   =  keys_first.getContainer( ).data( );
            typename bolt::cl::device_vector< valueType >::pointer valuesPtr =  values_first.getContainer( ).data( );
            serialCPU_sort_by_key(&keysPtr[keys_first.m_Index], &keysPtr[keys_last.m_Index],
                                            &valuesPtr[values_first.m_Index], comp);
            return;
        } else if (runMode == bolt::cl::control::MultiCoreCpu) {

            #ifdef ENABLE_TBB
			    #if defined(BOLT_DEBUG_LOG)
                dblog->CodePathTaken(BOLTLOG::BOLT_SORTBYKEY,BOLTLOG::BOLT_MULTICORE_CPU,"::Sort_By_Key::MULTICORE_CPU");
                #endif
                typename bolt::cl::device_vector< keyType >::pointer   keysPtr   =  keys_first.getContainer( ).data( );
                typename bolt::cl::device_vector< valueType >::pointer valuesPtr =  values_first.getContainer( ).data( );
                bolt::btbb::sort_by_key(&keysPtr[keys_first.m_Index], &keysPtr[keys_last.m_Index],
                                                &valuesPtr[values_first.m_Index], comp);
                return;
            #else
               throw std::runtime_error( "The MultiCoreCpu version of Sort_by_key is not enabled to be built with TBB!\n");
            #endif
        }

        else {
            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_SORTBYKEY,BOLTLOG::BOLT_OPENCL_GPU,"::Sort_By_Key::OPENCL_GPU");
            #endif
            sort_by_key_enqueue(ctl, keys_first, keys_last, values_first, comp, cl_code);
        }
        return;
    }

    //Non Device Vector specialization.
    //This implementation creates a cl::Buffer and passes the cl buffer to the
    //sort specialization whichtakes the cl buffer as a parameter.
    //In the future, Each input buffer should be mapped to the device_vector and the
    //specialization specific to device_vector should be called.
    template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
    void sort_by_key_pick_iterator(control &ctl, RandomAccessIterator1 keys_first,
                                   RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first,
                                   StrictWeakOrdering comp, const std::string& cl_code,
                                   std::random_access_iterator_tag, std::random_access_iterator_tag )
    {

        typedef typename std::iterator_traits<RandomAccessIterator1>::value_type T_keys;
        typedef typename std::iterator_traits<RandomAccessIterator2>::value_type T_values;
        int szElements = static_cast<int>(keys_last - keys_first);
        if (szElements == 0)
            return;

        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode( );
        if( runMode == bolt::cl::control::Automatic )
        {
            runMode = ctl.getDefaultPathToRun( );
        }
	    #if defined(BOLT_DEBUG_LOG)
        BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
        #endif
        if ((runMode == bolt::cl::control::SerialCpu) /*|| (szElements < WGSIZE) */) {
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_SORTBYKEY,BOLTLOG::BOLT_SERIAL_CPU,"::Sort_By_Key::SERIAL_CPU");
            #endif
            serialCPU_sort_by_key(keys_first, keys_last, values_first, comp);
            return;
        } else if (runMode == bolt::cl::control::MultiCoreCpu) {

            #ifdef ENABLE_TBB
			    #if defined(BOLT_DEBUG_LOG)
                dblog->CodePathTaken(BOLTLOG::BOLT_SORTBYKEY,BOLTLOG::BOLT_MULTICORE_CPU,"::Sort_By_Key::MULTICORE_CPU");
                #endif
                serialCPU_sort_by_key(keys_first, keys_last, values_first, comp);
                return;
            #else
                throw std::runtime_error("The MultiCoreCpu Version of Sort_by_key is not enabled to be built with TBB!\n");
            #endif
        } else {
            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_SORTBYKEY,BOLTLOG::BOLT_OPENCL_GPU,"::Sort_By_Key::OPENCL_GPU");
            #endif
			
            device_vector< T_values > dvInputValues( values_first, szElements,
                                                     CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
            device_vector< T_keys > dvInputKeys( keys_first, keys_last,
                                                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, ctl );
            //Now call the actual cl algorithm
            sort_by_key_enqueue(ctl,dvInputKeys.begin(),dvInputKeys.end(), dvInputValues.begin(), comp, cl_code);
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
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
                                    std::random_access_iterator_tag, std::random_access_iterator_tag )
    {
        return sort_by_key_pick_iterator( ctl, keys_first, keys_last, values_first,
                                    comp, cl_code,
                                    typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                    typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
    };

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void sort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
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
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
                                    bolt::cl::fancy_iterator_tag, std::input_iterator_tag )
    {
        static_assert( std::is_same< RandomAccessIterator1, bolt::cl::fancy_iterator_tag >::value , "It is not possible to sort fancy iterators. They are not mutable" );
        static_assert( std::is_same< RandomAccessIterator2, std::input_iterator_tag >::value , "It is not possible to sort fancy iterators. They are not mutable" );
    }

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void sort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
                                    std::input_iterator_tag, bolt::cl::fancy_iterator_tag )
    {


        static_assert( std::is_same< RandomAccessIterator2, bolt::cl::fancy_iterator_tag >::value , "It is not possible to sort fancy iterators. They are not mutable" );
        static_assert( std::is_same< RandomAccessIterator1, std::input_iterator_tag >::value , "It is not possible to sort fancy iterators. They are not mutable" );

    }


}//namespace bolt::cl::detail


        template<typename RandomAccessIterator1 , typename RandomAccessIterator2>
        void sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         const std::string& cl_code)
        {
            typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keys_T;

            detail::sort_by_key_detect_random_access( control::getDefault( ),
                                       keys_first, keys_last,
                                       values_first,
                                       less< keys_T >( ),
                                       cl_code,
                                       typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                       typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
            return;
        }

        template<typename RandomAccessIterator1 , typename RandomAccessIterator2, typename StrictWeakOrdering>
        void sort_by_key(RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         StrictWeakOrdering comp,
                         const std::string& cl_code)
        {
            typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keys_T;

            detail::sort_by_key_detect_random_access( control::getDefault( ),
                                       keys_first, keys_last,
                                       values_first,
                                       comp,
                                       cl_code,
                                       typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                       typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
            return;
        }

        template<typename RandomAccessIterator1 , typename RandomAccessIterator2>
        void sort_by_key(control &ctl,
                         RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         const std::string& cl_code)
        {
            typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keys_T;

            detail::sort_by_key_detect_random_access( ctl,
                                       keys_first, keys_last,
                                       values_first,
                                       less< keys_T >( ),
                                       cl_code,
                                       typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                       typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
            return;
        }

        template<typename RandomAccessIterator1 , typename RandomAccessIterator2, typename StrictWeakOrdering>
        void sort_by_key(control &ctl,
                         RandomAccessIterator1 keys_first,
                         RandomAccessIterator1 keys_last,
                         RandomAccessIterator2 values_first,
                         StrictWeakOrdering comp,
                         const std::string& cl_code)
        {
            typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keys_T;

            detail::sort_by_key_detect_random_access( ctl,
                                       keys_first, keys_last,
                                       values_first,
                                       comp,
                                       cl_code,
                                       typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                       typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
            return;
        }

    }
};



#endif
