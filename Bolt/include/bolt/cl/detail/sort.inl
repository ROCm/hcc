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


#if !defined( BOLT_CL_SORT_INL )
#define BOLT_CL_SORT_INL
#pragma once

#ifdef ENABLE_TBB
#include "bolt/btbb/sort.h"
#endif

#include "bolt/cl/stablesort.h"
#define BOLT_UINT_MAX 0xFFFFFFFFU
#define BOLT_UINT_MIN 0x0U
#define BOLT_INT_MAX 0x7FFFFFFF
#define BOLT_INT_MIN 0x80000000

#define BITONIC_SORT_WGSIZE 64
/* \brief - SORT_CPU_THRESHOLD should be atleast 2 times the BITONIC_SORT_WGSIZE*/
#define SORT_CPU_THRESHOLD 128

namespace bolt {
namespace cl {

namespace detail {

template< typename DVRandomAccessIterator, typename StrictWeakOrdering >
typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,
                                       unsigned int
                                     >::value
                       >::type  /*If enabled then this typename will be evaluated to void*/
stablesort_enqueue(control &ctl,
             DVRandomAccessIterator first, DVRandomAccessIterator last,
             StrictWeakOrdering comp, const std::string& cl_code);

template< typename DVRandomAccessIterator, typename StrictWeakOrdering >
typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,
                                       int
                                     >::value
                       >::type  /*If enabled then this typename will be evaluated to void*/
stablesort_enqueue(control &ctl,
             DVRandomAccessIterator first, DVRandomAccessIterator last,
             StrictWeakOrdering comp, const std::string& cl_code);

template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
typename std::enable_if<
    !(std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type, unsigned int >::value || 
      std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type, int >::value  )
                       >::type
stablesort_enqueue(control& ctrl, const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
             const StrictWeakOrdering& comp, const std::string& cl_code);

enum sortTypes {sort_iValueType, sort_iIterType, sort_StrictWeakOrdering, sort_end };

class BitonicSort_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
public:
    BitonicSort_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
        addKernelName("BitonicSortTemplate");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string >& typeNames ) const
    {
        const std::string templateSpecializationString =

            "// Host generates this instantiation string with user-specified value type and functor\n"
            "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
            "kernel void BitonicSortTemplate(\n"
            "global " + typeNames[sort_iValueType] + "* A,\n"
            ""        + typeNames[sort_iIterType]  + " input_iter,\n"
            "const uint stage,\n"
            "const uint passOfStage,\n"
            "global " + typeNames[sort_StrictWeakOrdering] + " * userComp\n"
            ");\n\n";
            return templateSpecializationString;
        }
};

class RadixSort_Int_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
private:

public:
    RadixSort_Int_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
        addKernelName("permuteSignedAsc");
        addKernelName("permuteSignedDesc");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
    {
        const std::string templateSpecializationString = "\n //RadixSort_Int_KernelTemplateSpecializer\n";
        return templateSpecializationString;
    }
};

class RadixSort_Uint_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
private:
    int _radix;
public:
    RadixSort_Uint_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
        addKernelName("permuteAsc");
        addKernelName("permuteDesc");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
    {
        const std::string templateSpecializationString = "\n //RadixSort_Uint_KernelTemplateSpecializer\n";
        return templateSpecializationString;
    }
};

class RadixSort_Common_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
private:
public:
    RadixSort_Common_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
        addKernelName("histogramAsc");
        addKernelName("histogramDesc");
        addKernelName("histogramSignedAsc");
        addKernelName("histogramSignedDesc");
        addKernelName("scan");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
    {
        const std::string templateSpecializationString = "\n //RadixSort_Common_KernelTemplateSpecializer\n";
        return templateSpecializationString;
    }
};

template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
void sort_enqueue_non_powerOf2(control &ctl,
                               const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
                               const StrictWeakOrdering& comp, const std::string& cl_code)
{
    /*The selection sort algorithm is not good for GPUs Hence calling the stablesort routines.
     *For future call a combination of selection sort and bitonic sort. To improve performance of floats
     * doubles and UDDs*/
    bolt::cl::detail::stablesort_enqueue(ctl, first, last, comp, cl_code);
    return;
}// END of sort_enqueue_non_powerOf2

/*********************************************************************
 * RADIX SORT ALGORITHM FOR unsigned integers.
 *********************************************************************/

template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,
                                       unsigned int
                                     >::value
                       >::type  /*If enabled then this typename will be evaluated to void*/
sort_enqueue(control &ctl,
             DVRandomAccessIterator first, DVRandomAccessIterator last,
             StrictWeakOrdering comp, const std::string& cl_code)
{
    typedef typename std::iterator_traits< DVRandomAccessIterator >::value_type T;
    const int RADIX = 4; //Now you cannot replace this with Radix 8 since there is a
                         //local array of 16 elements in the histogram kernel.

    const int RADICES = (1 << RADIX); //Values handeled by each work-item?

    int szElements = static_cast<int>(std::distance(first, last));

    int computeUnits     = ctl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    if (computeUnits > 32 )
        computeUnits = 32;
    cl_int l_Error = CL_SUCCESS;

    //static std::vector< ::cl::Kernel > radixSortUintKernels;
    //static std::vector< ::cl::Kernel > radixSortCommonKernels;
    std::vector<std::string> typeNames( sort_end );
    typeNames[sort_iValueType]         = TypeName< T >::get( );
    typeNames[sort_iIterType]          = TypeName< DVRandomAccessIterator >::get( );
    typeNames[sort_StrictWeakOrdering] = TypeName< StrictWeakOrdering >::get();

    std::vector<std::string> typeDefinitions;
    PUSH_BACK_UNIQUE( typeDefinitions, cl_code )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< T >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVRandomAccessIterator >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< StrictWeakOrdering  >::get() )

    bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
    /*\TODO - Do CPU specific kernel work group size selection here*/

    std::string compileOptions;
    //std::ostringstream oss;
    RadixSort_Common_KernelTemplateSpecializer radix_common_kts;
    std::vector< ::cl::Kernel > commonKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_common_kts,
        typeDefinitions,
        sort_common_kernels,
        compileOptions);

    RadixSort_Uint_KernelTemplateSpecializer radix_uint_kts;
    std::vector< ::cl::Kernel > uintKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_uint_kts,
        typeDefinitions,
        sort_uint_kernels,
        compileOptions);

    int localSize  = 256;
    int wavefronts = 8;
    int numGroups = computeUnits * wavefronts;


    device_vector< T > dvSwapInputData( szElements, 0, CL_MEM_READ_WRITE, false, ctl);
    device_vector< T > dvHistogramBins( (localSize * RADICES), 0, CL_MEM_READ_WRITE, false, ctl);

    ::cl::Buffer clInputData = first.getContainer().getBuffer();
    ::cl::Buffer clSwapData = dvSwapInputData.begin( ).getContainer().getBuffer();
    ::cl::Buffer clHistData = dvHistogramBins.begin( ).getContainer().getBuffer();

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
    V_OPENCL( scanLocalKernel.setArg(2, localSize * 2 * sizeof(T),NULL), "Error setting a kernel argument" );
    
    //Set Permute kernel arguments
    V_OPENCL( permuteKernel.setArg(1, clHistData), "Error setting a kernel argument" );

    for(int bits = 0; bits < (sizeof(T) * 8); bits += RADIX)
    {
        //Launch Kernel
        cdata.m_startBit = bits;
        //Histogram Kernel
        V_OPENCL( histKernel.setArg(2, cdata), "Error setting a kernel argument" );
        if (swap == 0)
            V_OPENCL( histKernel.setArg(0, clInputData), "Error setting a kernel argument" );
        else
            V_OPENCL( histKernel.setArg(0, clSwapData), "Error setting a kernel argument" );
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            histKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize), //This mul will be removed when permute is optimized
                            NULL,
                            NULL);
//#define DEBUG_ENABLED
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
        V_OPENCL( permuteKernel.setArg(3, cdata), "Error setting a kernel argument" );        
        if (swap == 0)
        {
            V_OPENCL( permuteKernel.setArg(0, clInputData), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(2, clSwapData), "Error setting kernel argument" );
        }
        else
        {
            V_OPENCL( permuteKernel.setArg(0, clSwapData), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(2, clInputData), "Error setting kernel argument" );
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


/*********************************************************************
 * RADIX SORT ALGORITHM FOR signed integers.
 *********************************************************************/
template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,
                                       int
                                     >::value
                       >::type   /*If enabled then this typename will be evaluated to void*/
sort_enqueue(control &ctl,
             DVRandomAccessIterator first, DVRandomAccessIterator last,
             StrictWeakOrdering comp, const std::string& cl_code)
{
    typedef typename std::iterator_traits< DVRandomAccessIterator >::value_type T;
    const int RADIX = 4; //Now you cannot replace this with Radix 8 since there is a
                         //local array of 16 elements in the histogram kernel.

    const int RADICES = (1 << RADIX); //Values handeled by each work-item?

    int szElements = static_cast<int>(std::distance(first, last));

    int computeUnits     = ctl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    if (computeUnits > 32 )
        computeUnits = 32;
    cl_int l_Error = CL_SUCCESS;

    //static std::vector< ::cl::Kernel > radixSortUintKernels;
    //static std::vector< ::cl::Kernel > radixSortCommonKernels;
    std::vector<std::string> typeNames( sort_end );
    typeNames[sort_iValueType]         = TypeName< T >::get( );
    typeNames[sort_iIterType]          = TypeName< DVRandomAccessIterator >::get( );
    typeNames[sort_StrictWeakOrdering] = TypeName< StrictWeakOrdering >::get();

    std::vector<std::string> typeDefinitions;
    PUSH_BACK_UNIQUE( typeDefinitions, cl_code )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< T >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVRandomAccessIterator >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< StrictWeakOrdering  >::get() )

    bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
    /*\TODO - Do CPU specific kernel work group size selection here*/

    std::string compileOptions;
    //std::ostringstream oss;
    RadixSort_Common_KernelTemplateSpecializer radix_common_kts;
    std::vector< ::cl::Kernel > commonKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_common_kts,
        typeDefinitions,
        sort_common_kernels,
        compileOptions);

    RadixSort_Int_KernelTemplateSpecializer radix_int_kts;
    std::vector< ::cl::Kernel > intKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_int_kts,
        typeDefinitions,
        sort_int_kernels,
        compileOptions);

    RadixSort_Uint_KernelTemplateSpecializer radix_uint_kts;
    std::vector< ::cl::Kernel > uintKernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &radix_uint_kts,
        typeDefinitions,
        sort_uint_kernels,
        compileOptions);

    int localSize  = 256;
    int wavefronts = 8;
    int numGroups = computeUnits * wavefronts;

    device_vector< T > dvSwapInputData( szElements, 0, CL_MEM_READ_WRITE, false, ctl);
    device_vector< T > dvHistogramBins( (localSize * RADICES), 0, CL_MEM_READ_WRITE, false, ctl);

    ::cl::Buffer clInputData = first.getContainer().getBuffer();
    ::cl::Buffer clSwapData = dvSwapInputData.begin( ).getContainer().getBuffer();
    ::cl::Buffer clHistData = dvHistogramBins.begin( ).getContainer().getBuffer();

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
    V_OPENCL( scanLocalKernel.setArg(2, localSize * 2 * sizeof(T),NULL), "Error setting a kernel argument" );
    
    //Set Permute kernel arguments
    V_OPENCL( permuteKernel.setArg(1, clHistData), "Error setting a kernel argument" );
    int bits = 0;
    for(bits = 0; bits < (sizeof(T) * 7); bits += RADIX)
    {
        //Launch Kernel
        cdata.m_startBit = bits;
        //Histogram Kernel
        V_OPENCL( histKernel.setArg(2, cdata), "Error setting a kernel argument" );
        if (swap == 0)
            V_OPENCL( histKernel.setArg(0, clInputData), "Error setting a kernel argument" );
        else
            V_OPENCL( histKernel.setArg(0, clSwapData), "Error setting a kernel argument" );
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            histKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize), //This mul will be removed when permute is optimized
                            NULL,
                            NULL);
//#define DEBUG_ENABLED
#if defined(DEBUG_ENABLED)
        {
            V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
            printf("histogramAscending Kernel global_wsize=%d, local_wsize=%d\n", localSize * numGroups, localSize);            

            T * temp = dvHistogramBins.data().get();
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

            T * temp = dvHistogramBins.data().get();
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
        V_OPENCL( permuteKernel.setArg(3, cdata), "Error setting a kernel argument" );        
        if (swap == 0)
        {
            V_OPENCL( permuteKernel.setArg(0, clInputData), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(2, clSwapData), "Error setting kernel argument" );
        }
        else
        {
            V_OPENCL( permuteKernel.setArg(0, clSwapData), "Error setting kernel argument" );
            V_OPENCL( permuteKernel.setArg(2, clInputData), "Error setting kernel argument" );
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
        //Launch Kernel
        cdata.m_startBit = bits;

        //Set Histogram Signed kernel arguments
        V_OPENCL( histSignedKernel.setArg(0, clSwapData), "Error setting a kernel argument" );
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

            T* temp = dvHistogramBins.data().get();
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

            T * temp = dvHistogramBins.data().get();
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
        //Set Permute Signed kernel arguments

        V_OPENCL( permuteSignedKernel.setArg(0, clSwapData), "Error setting kernel argument" );
        V_OPENCL( permuteSignedKernel.setArg(1, clHistData), "Error setting a kernel argument" );
        V_OPENCL( permuteSignedKernel.setArg(2, clInputData), "Error setting kernel argument" );
        V_OPENCL( permuteSignedKernel.setArg(3, cdata), "Error setting a kernel argument" );
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                            permuteSignedKernel,
                            ::cl::NullRange,
                            ::cl::NDRange(numGroups*localSize),
                            ::cl::NDRange(localSize),
                            NULL,
                            NULL);

    }//End of signed integer sorting
    
    V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
    return;
}


template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
typename std::enable_if<
    !(std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type, unsigned int >::value
   || std::is_same< typename std::iterator_traits<DVRandomAccessIterator >::value_type,          int >::value
    )
                       >::type
sort_enqueue(control &ctl,
             const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
             const StrictWeakOrdering& comp, const std::string& cl_code)
{
    cl_int l_Error = CL_SUCCESS;
    typedef typename std::iterator_traits< DVRandomAccessIterator >::value_type T;
    size_t szElements = static_cast< size_t >( std::distance( first, last ) );
    if(((szElements-1) & (szElements)) != 0)
    {
        sort_enqueue_non_powerOf2(ctl,first,last,comp,cl_code);
        return;
    }

    std::vector<std::string> typeNames( sort_end );
    typeNames[sort_iValueType] = TypeName< T >::get( );
    typeNames[sort_iIterType] = TypeName< DVRandomAccessIterator >::get( );
    typeNames[sort_StrictWeakOrdering] = TypeName< StrictWeakOrdering >::get();

    std::vector<std::string> typeDefinitions;
    PUSH_BACK_UNIQUE( typeDefinitions, cl_code )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< T >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVRandomAccessIterator >::get() )
    PUSH_BACK_UNIQUE( typeDefinitions, ClCode< StrictWeakOrdering  >::get() )

    bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
    /*\TODO - Do CPU specific kernel work group size selection here*/
    //const size_t kernel0_WgSize = (cpuDevice) ? 1 : WAVESIZE*KERNEL02WAVES;
    std::string compileOptions;
    //std::ostringstream oss;
    //oss << " -DKERNEL0WORKGROUPSIZE=" << kernel0_WgSize;

    size_t temp;

    BitonicSort_KernelTemplateSpecializer ts_kts;
    std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &ts_kts,
        typeDefinitions,
        sort_kernels,
        compileOptions);
    //Power of 2 buffer size
    // For user-defined types, the user must create a TypeName trait which returns the name of the class -
    // Note use of TypeName<>::get to retreive the name here.


    size_t wgSize  = BITONIC_SORT_WGSIZE;

    if((szElements/2) < BITONIC_SORT_WGSIZE)
    {
        wgSize = (int)szElements/2;
    }
    unsigned int stage,passOfStage;
    unsigned int numStages = 0;
    for(temp = szElements; temp > 1; temp >>= 1)
        ++numStages;

    //::cl::Buffer A = first.getContainer().getBuffer();
    ALIGNED( 256 ) StrictWeakOrdering aligned_comp( comp );
    control::buffPointer userFunctor = ctl.acquireBuffer( sizeof( aligned_comp ),
                                                          CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_comp );
   typename DVRandomAccessIterator::Payload first_payload = first.gpuPayload( );

    V_OPENCL( kernels[0].setArg(0, first.getContainer().getBuffer()), "Error setting 0th kernel argument" );
    V_OPENCL( kernels[0].setArg(1, first.gpuPayloadSize( ),&first_payload ),
                                                "Error setting 1st kernel argument" );

    V_OPENCL( kernels[0].setArg(4, *userFunctor), "Error setting 4th kernel argument" );
    for(stage = 0; stage < numStages; ++stage)
    {
        // stage of the algorithm
        V_OPENCL( kernels[0].setArg(2, stage), "Error setting 2nd kernel argument" );
        // Every stage has stage + 1 passes
        for(passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            // pass of the current stage
            V_OPENCL( kernels[0].setArg(3, passOfStage), "Error setting 3rd kernel argument" );
            /*
             * Enqueue a kernel run call.
             * Each thread writes a sorted pair.
             * So, the number of  threads (global) should be half the length of the input buffer.
             */
            l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                                            kernels[0],
                                            ::cl::NullRange,
                                            ::cl::NDRange(szElements/2),
                                            ::cl::NDRange(wgSize),
                                            NULL,
                                            NULL);

            V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for sort() kernel" );
            //V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
        }//end of for passStage = 0:stage-1
    }//end of for stage = 0:numStage-1

    //TODO this is a bug in APP SDK cl.hpp file The header file is non compliant with the khronos cl.hpp.
    //     Hence a finish function is added to wait for all the tasks to complete.
    /*::cl::Event bitonicSortEvent;
    V_OPENCL( ctl.getCommandQueue().clEnqueueBarrierWithWaitList(NULL, &bitonicSortEvent) ,
                        "Error calling clEnqueueBarrierWithWaitList on the command queue" );
    l_Error = bitonicSortEvent.wait( );
    V_OPENCL( l_Error, "bitonicSortEvent failed to wait" );*/
    V_OPENCL( ctl.getCommandQueue().finish(), "Error calling finish on the command queue" );
    return;
}// END of sort_enqueue

//Device Vector specialization
template<typename DVRandomAccessIterator, typename StrictWeakOrdering>
void sort_pick_iterator( control &ctl,
                         const DVRandomAccessIterator& first, const DVRandomAccessIterator& last,
                         const StrictWeakOrdering& comp, const std::string& cl_code,
                         bolt::cl::device_vector_tag )
{
    // User defined Data types are not supported with device_vector. Hence we have a static assert here.
    // The code here should be in compliant with the routine following this routine.
    typedef typename std::iterator_traits<DVRandomAccessIterator>::value_type T;
    size_t szElements = static_cast< size_t >( std::distance( first, last ) );
    if( szElements < 2 )
        return;
    bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
    if(runMode == bolt::cl::control::Automatic)
    {
        runMode = ctl.getDefaultPathToRun();
    }
    #if defined(BOLT_DEBUG_LOG)
    BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
    #endif
    
    if ((runMode == bolt::cl::control::SerialCpu) || (szElements < SORT_CPU_THRESHOLD)) {
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SORT,BOLTLOG::BOLT_SERIAL_CPU,"::Sort::SERIAL_CPU");
        #endif
        typename bolt::cl::device_vector< T >::pointer firstPtr =  first.getContainer( ).data( );
        std::sort( &firstPtr[ first.m_Index ], &firstPtr[ last.m_Index ], comp );
        return;
    } else if (runMode == bolt::cl::control::MultiCoreCpu) {
#ifdef ENABLE_TBB
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SORT,BOLTLOG::BOLT_MULTICORE_CPU,"::Sort::MULTICORE_CPU");
        #endif
        typename bolt::cl::device_vector< T >::pointer firstPtr =  first.getContainer( ).data( );
        //Compute parallel sort using TBB
        bolt::btbb::sort(&firstPtr[ first.m_Index ], &firstPtr[ last.m_Index ],comp);
        return;
#else
        //std::cout << "The MultiCoreCpu version of sort is not enabled. " << std ::endl;
        throw std::runtime_error( "The MultiCoreCpu version of sort is not enabled to be built! \n" );
#endif

    } else {
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SORT,BOLTLOG::BOLT_OPENCL_GPU,"::Sort::OPENCL_GPU");
        #endif
        sort_enqueue(ctl,first,last,comp,cl_code);
    }
    return;
}


//Non Device Vector specialization.
//This implementation creates a cl::Buffer and passes the cl buffer to the sort specialization
//whichtakes the cl buffer as a parameter. In the future, Each input buffer should be mapped to the device_vector
//and the specialization specific to device_vector should be called.
template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort_pick_iterator( control &ctl,
                         const RandomAccessIterator& first, const RandomAccessIterator& last,
                         const StrictWeakOrdering& comp, const std::string& cl_code,
                         std::random_access_iterator_tag )
{
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
    size_t szElements = (size_t)(last - first);
    if( szElements < 2 )
        return;

    bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
    if(runMode == bolt::cl::control::Automatic)
    {
        runMode = ctl.getDefaultPathToRun();
    }
    #if defined(BOLT_DEBUG_LOG)
    BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
    #endif
    
    if ((runMode == bolt::cl::control::SerialCpu) || (szElements < BITONIC_SORT_WGSIZE)) {
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SORT,BOLTLOG::BOLT_SERIAL_CPU,"::Sort::SERIAL_CPU");
        #endif
        std::sort(first, last, comp);
        return;
    } else if (runMode == bolt::cl::control::MultiCoreCpu) {
#ifdef ENABLE_TBB
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SORT,BOLTLOG::BOLT_MULTICORE_CPU,"::Sort::MULTICORE_CPU");
        #endif
        bolt::btbb::sort(first,last, comp);
#else
        throw std::runtime_error( "The MultiCoreCpu version of sort is not enabled to be built! \n" );
#endif
    } else {
        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_SORT,BOLTLOG::BOLT_OPENCL_GPU,"::Sort::OPENCL_GPU");
        #endif
        
        device_vector< T > dvInputOutput( first, last, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, ctl );
        //Now call the actual cl algorithm
        sort_enqueue(ctl,dvInputOutput.begin(),dvInputOutput.end(),comp,cl_code);
        //Map the buffer back to the host
        dvInputOutput.data( );
        return;
    }
}


template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort_detect_random_access( control &ctl,
                                const RandomAccessIterator& first, const RandomAccessIterator& last,
                                const StrictWeakOrdering& comp, const std::string& cl_code,
                                std::random_access_iterator_tag )
{
    return sort_pick_iterator(ctl, first, last,
                              comp, cl_code,
                             typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
};

// Wrapper that uses default control class, iterator interface
template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort_detect_random_access( control &ctl,
                                const RandomAccessIterator& first, const RandomAccessIterator& last,
                                const StrictWeakOrdering& comp, const std::string& cl_code,
                                std::input_iterator_tag )
{
    //  \TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
    //  to a temporary buffer.  Should we?
    static_assert( std::is_same< RandomAccessIterator, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
};

// Wrapper that uses default control class, iterator interface
template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort_detect_random_access( control &ctl,
                                const RandomAccessIterator& first, const RandomAccessIterator& last,
                                const StrictWeakOrdering& comp, const std::string& cl_code,
                                bolt::cl::fancy_iterator_tag )
{
    static_assert(std::is_same< RandomAccessIterator, bolt::cl::fancy_iterator_tag >::value  , "Bolt only supports random access iterator types. And does not support Fancy Iterator Tags" );
};

}//namespace bolt::cl::detail

template<typename RandomAccessIterator>
void sort(RandomAccessIterator first,
          RandomAccessIterator last,
          const std::string& cl_code)
{
    typedef typename std::iterator_traits< RandomAccessIterator >::value_type T;

    detail::sort_detect_random_access( control::getDefault( ),
                                       first, last,
                                       less< T >( ), cl_code,
                                       typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
    return;
}

template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort(RandomAccessIterator first,
          RandomAccessIterator last,
          StrictWeakOrdering comp,
          const std::string& cl_code)
{
    detail::sort_detect_random_access( control::getDefault( ),
                                       first, last,
                                       comp, cl_code,
                                       typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
    return;
}

template<typename RandomAccessIterator>
void sort(control &ctl,
          RandomAccessIterator first,
          RandomAccessIterator last,
          const std::string& cl_code)
{
    typedef typename std::iterator_traits< RandomAccessIterator >::value_type T;

    detail::sort_detect_random_access(ctl,
                                      first, last,
                                      less< T >( ), cl_code,
                                      typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
    return;
}

template<typename RandomAccessIterator, typename StrictWeakOrdering>
void sort(control &ctl,
          RandomAccessIterator first,
          RandomAccessIterator last,
          StrictWeakOrdering comp,
          const std::string& cl_code)
{
    detail::sort_detect_random_access(ctl,
                                      first, last,
                                      comp, cl_code,
                                      typename std::iterator_traits< RandomAccessIterator >::iterator_category( ) );
    return;
}
}
};



#endif
