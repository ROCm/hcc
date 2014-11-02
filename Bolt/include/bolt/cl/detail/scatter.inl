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

#pragma once
#if !defined( BOLT_CL_SCATTER_INL )
#define BOLT_CL_SCATTER_INL
#define WAVEFRONT_SIZE 64


#ifdef ENABLE_TBB
    #include "bolt/btbb/scatter.h"
#endif


#include <algorithm>
#include <type_traits>

#include "bolt/cl/bolt.h"
#include "bolt/cl/device_vector.h"
#include "bolt/cl/distance.h"
#include "bolt/cl/iterator/iterator_traits.h"
#include "bolt/cl/iterator/transform_iterator.h"
#include "bolt/cl/iterator/addressof.h"


namespace bolt {
namespace cl {

namespace detail {

namespace serial{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
scatter (bolt::cl::control &ctl, InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 map,
         OutputIterator result)
{
    typename InputIterator1::difference_type sz = (last1 - first1);
    if (sz == 0)
        return;
    typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
    typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer first1Buffer = first1.base().getContainer( ).getBuffer( );
    ::cl::Buffer first2Buffer = map.base().getContainer( ).getBuffer( );
    ::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
    /*Get The size of each OpenCL buffer*/
    size_t first1_sz = first1Buffer.getInfo<CL_MEM_SIZE>();
    size_t first2_sz = first2Buffer.getInfo<CL_MEM_SIZE>();
    size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();
    
    cl_int map_err;
    iType1 *first1Ptr = (iType1*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
                                                                     first1_sz, NULL, NULL, &map_err);
    iType2 *first2Ptr = (iType2*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
                                                                     first2_sz, NULL, NULL, &map_err);
    oType *resultPtr  = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
                                                                     result_sz, NULL, NULL, &map_err);
    auto mapped_first1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator1>::iterator_category(), 
                                                   ctl, first1, first1Ptr);
    auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                   ctl, map, first2Ptr);
    auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                   ctl, result, resultPtr);

	for (int iter = 0; iter<(int)sz; iter++)
                *(mapped_result_itr +*(mapped_first2_itr + iter)) = (oType) *(mapped_first1_itr + iter);


    ::cl::Event unmap_event[3];
    ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
    ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
    ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[2] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait();
    return;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
scatter (bolt::cl::control &ctl, InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 map,
              OutputIterator result)
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;

    size_t numElements = static_cast<  size_t >( std::distance( first1, last1 ) );

	for (int iter = 0; iter<(int)numElements; iter++)
                *(result+*(map + iter)) = (oType) *(first1 + iter);
}


template< typename InputIterator1,
           typename InputIterator2,
           typename InputIterator3,
           typename OutputIterator,
           typename Predicate>
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
scatter_if( bolt::cl::control &ctl, 
            InputIterator1 first1,
            InputIterator1 last1,
            InputIterator2 map,
            InputIterator3 stencil,
            OutputIterator result,
            Predicate pred)
{
    typename InputIterator1::difference_type sz = (last1 - first1);
    if (sz == 0)
        return;
    typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
    typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
	typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer first1Buffer = first1.base().getContainer( ).getBuffer( );
    ::cl::Buffer first2Buffer = map.base().getContainer( ).getBuffer( );
	::cl::Buffer first3Buffer = stencil.base().getContainer( ).getBuffer( );
    ::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
    /*Get The size of each OpenCL buffer*/
    size_t first1_sz = first1Buffer.getInfo<CL_MEM_SIZE>();
    size_t first2_sz = first2Buffer.getInfo<CL_MEM_SIZE>();
	size_t first3_sz = first3Buffer.getInfo<CL_MEM_SIZE>();
    size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();
    
    cl_int map_err;
    iType1 *first1Ptr = (iType1*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
                                                                     first1_sz, NULL, NULL, &map_err);
    iType2 *first2Ptr = (iType2*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
                                                                     first2_sz, NULL, NULL, &map_err);
	iType3 *first3Ptr = (iType3*)ctl.getCommandQueue().enqueueMapBuffer(first3Buffer, true, CL_MAP_READ, 0, 
                                                                     first3_sz, NULL, NULL, &map_err);
    oType *resultPtr  = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
                                                                     result_sz, NULL, NULL, &map_err);
    auto mapped_first1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator1>::iterator_category(), 
                                                   ctl, first1, first1Ptr);
    auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                   ctl, map, first2Ptr);
	auto mapped_first3_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator3>::iterator_category(), 
                                                   ctl, stencil, first3Ptr);
    auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                   ctl, result, resultPtr);

	for(int iter = 0; iter< (int)sz; iter++)
    {
          if(pred(*(mapped_first3_itr + iter) ) != 0)
               //result[*(map+(iter - 0))] = first1[iter];
		       *(mapped_result_itr + *(mapped_first2_itr + (iter -0))) = (oType) *(mapped_first1_itr + iter);
    }

    ::cl::Event unmap_event[4];
    ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
    ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
	ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first3Ptr, NULL, &unmap_event[2] );
    ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[3] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait(); unmap_event[3].wait();
    return;

}



template< typename InputIterator1,
           typename InputIterator2,
           typename InputIterator3,
           typename OutputIterator,
           typename Predicate>
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
scatter_if( bolt::cl::control &ctl, 
            InputIterator1 first1,
            InputIterator1 last1,
            InputIterator2 map,
            InputIterator3 stencil,
            OutputIterator result,
            Predicate pred)
{
    size_t numElements = static_cast< size_t >( std::distance( first1, last1 ) );
	for (int iter = 0; iter< (int)numElements; iter++)
    {
          if(pred(stencil[iter]) != 0)
               result[*(map+(iter))] = first1[iter];
    }
}


} // end of serial namespace

#ifdef ENABLE_TBB
namespace btbb{

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
scatter (bolt::cl::control &ctl, InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 map,
              OutputIterator result)
{
    typename InputIterator1::difference_type sz = (last1 - first1);
    if (sz == 0)
        return;
    typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
    typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer first1Buffer = first1.base().getContainer( ).getBuffer( );
    ::cl::Buffer first2Buffer = map.base().getContainer( ).getBuffer( );
    ::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
    /*Get The size of each OpenCL buffer*/
    size_t first1_sz = first1Buffer.getInfo<CL_MEM_SIZE>();
    size_t first2_sz = first2Buffer.getInfo<CL_MEM_SIZE>();
    size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();
    
    cl_int map_err;
    iType1 *first1Ptr = (iType1*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
                                                                     first1_sz, NULL, NULL, &map_err);
    iType2 *first2Ptr = (iType2*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
                                                                     first2_sz, NULL, NULL, &map_err);
    oType *resultPtr  = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
                                                                     result_sz, NULL, NULL, &map_err);
    auto mapped_first1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator1>::iterator_category(), 
                                                   ctl, first1, first1Ptr);
    auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                   ctl, map, first2Ptr);
    auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                   ctl, result, resultPtr);

	bolt::btbb::scatter(mapped_first1_itr, mapped_first1_itr + sz, mapped_first2_itr, mapped_result_itr);

    ::cl::Event unmap_event[3];
    ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
    ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
    ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[2] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait();
    return;
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
scatter (bolt::cl::control &ctl, InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 map,
              OutputIterator result)
{
    bolt::btbb::scatter(first1, last1, map, result);
}


template< typename InputIterator1,
           typename InputIterator2,
           typename InputIterator3,
           typename OutputIterator,
           typename Predicate>
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
scatter_if( bolt::cl::control &ctl, 
            InputIterator1 first1,
            InputIterator1 last1,
            InputIterator2 map,
            InputIterator3 stencil,
            OutputIterator result,
            Predicate pred)
{
    typename InputIterator1::difference_type sz = (last1 - first1);
    if (sz == 0)
        return;
    typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
    typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
	typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer first1Buffer = first1.base().getContainer( ).getBuffer( );
    ::cl::Buffer first2Buffer = map.base().getContainer( ).getBuffer( );
	::cl::Buffer first3Buffer = stencil.base().getContainer( ).getBuffer( );
    ::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
    /*Get The size of each OpenCL buffer*/
    size_t first1_sz = first1Buffer.getInfo<CL_MEM_SIZE>();
    size_t first2_sz = first2Buffer.getInfo<CL_MEM_SIZE>();
	size_t first3_sz = first3Buffer.getInfo<CL_MEM_SIZE>();
    size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();
    
    cl_int map_err;
    iType1 *first1Ptr = (iType1*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
                                                                     first1_sz, NULL, NULL, &map_err);
    iType2 *first2Ptr = (iType2*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
                                                                     first2_sz, NULL, NULL, &map_err);
	iType3 *first3Ptr = (iType3*)ctl.getCommandQueue().enqueueMapBuffer(first3Buffer, true, CL_MAP_READ, 0, 
                                                                     first3_sz, NULL, NULL, &map_err);
    oType *resultPtr  = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
                                                                     result_sz, NULL, NULL, &map_err);
    auto mapped_first1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator1>::iterator_category(), 
                                                   ctl, first1, first1Ptr);
    auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                   ctl, map, first2Ptr);
	auto mapped_first3_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator3>::iterator_category(), 
                                                   ctl, stencil, first3Ptr);
    auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                   ctl, result, resultPtr);

	bolt::btbb::scatter_if(mapped_first1_itr, mapped_first1_itr + sz, mapped_first2_itr, mapped_first3_itr, mapped_result_itr, pred);

    ::cl::Event unmap_event[4];
    ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
    ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
	ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first3Ptr, NULL, &unmap_event[2] );
    ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[3] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait(); unmap_event[3].wait();
    return;

}




template< typename InputIterator1,
           typename InputIterator2,
           typename InputIterator3,
           typename OutputIterator,
           typename Predicate>
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
scatter_if( bolt::cl::control &ctl, 
            InputIterator1 first1,
            InputIterator1 last1,
            InputIterator2 map,
            InputIterator3 stencil,
            OutputIterator result,
            Predicate pred)
{
    bolt::btbb::scatter_if(first1, last1, map, stencil, result, pred);
}



}// end of btbb namespace
#endif

////////////////////////////////////////////////////////////////////
// ScatterIf KTS
////////////////////////////////////////////////////////////////////

 namespace cl{

  enum ScatterIfTypes { scatter_if_iType, scatter_if_DVInputIterator,
                        scatter_if_mapType, scatter_if_DVMapType,
                        scatter_if_stencilType, scatter_if_DVStencilType,
                        scatter_if_resultType, scatter_if_DVResultType,
                        scatter_if_Predicate, scatter_if_endB };

class ScatterIf_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
public:
    ScatterIf_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
       addKernelName("scatterIfTemplate");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& scatterIfKernels ) const
    {
      const std::string templateSpecializationString =
        "// Host generates this instantiation string with user-specified value type and functor\n"
        "template __attribute__((mangled_name("+name(0)+"Instantiated)))\n"
        "kernel void "+name(0)+"(\n"
        "global " + scatterIfKernels[scatter_if_iType] + "* input, \n"
        + scatterIfKernels[scatter_if_DVInputIterator] + " inputIter, \n"
        "global " + scatterIfKernels[scatter_if_mapType] + "* map, \n"
        + scatterIfKernels[scatter_if_DVMapType] + " mapIter, \n"
        "global " + scatterIfKernels[scatter_if_stencilType] + "* stencil, \n"
        + scatterIfKernels[scatter_if_DVStencilType] + " stencilIter, \n"
        "global " + scatterIfKernels[scatter_if_resultType] + "* result, \n"
        + scatterIfKernels[scatter_if_DVResultType] + " resultIter, \n"
        "const uint length, \n"
        "global " + scatterIfKernels[scatter_if_Predicate] + "* functor);\n\n";

        return templateSpecializationString;
    }
};

////////////////////////////////////////////////////////////////////
// Scatter KTS
////////////////////////////////////////////////////////////////////

  enum ScatterTypes { scatter_iType, scatter_DVInputIterator,
                      scatter_mapType, scatter_DVMapType,
                      scatter_resultType, scatter_DVResultType,
                      scatter_endB };

class ScatterKernelTemplateSpecializer : public KernelTemplateSpecializer
{
public:
    ScatterKernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
       addKernelName("scatterTemplate");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& scatterKernels ) const
    {
      const std::string templateSpecializationString =
        "// Host generates this instantiation string with user-specified value type and functor\n"
        "template __attribute__((mangled_name("+name(0)+"Instantiated)))\n"
        "kernel void "+name(0)+"(\n"
        "global " + scatterKernels[scatter_iType] + "* input, \n"
        + scatterKernels[scatter_DVInputIterator] + " inputIter, \n"
        "global " + scatterKernels[scatter_mapType] + "* map, \n"
        + scatterKernels[scatter_DVMapType] + " mapIter, \n"
        "global " + scatterKernels[scatter_resultType] + "* result, \n"
        + scatterKernels[scatter_DVResultType] + " resultIter, \n"
        "const uint length ); \n";

        return templateSpecializationString;
    }
};



    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVInputIterator3,
              typename DVOutputIterator,
              typename Predicate >
    typename std::enable_if< std::is_same< typename std::iterator_traits< DVOutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value
                       >::type
    scatter_if( bolt::cl::control &ctl,
                const DVInputIterator1& first1,
                const DVInputIterator1& last1,
                const DVInputIterator2& map,
                const DVInputIterator3& stencil,
                const DVOutputIterator& result,
                const Predicate& pred,
                const std::string& cl_code )
    {
        typedef typename std::iterator_traits<DVInputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<DVInputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<DVInputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;

        cl_uint distVec = static_cast< cl_uint >( std::distance( first1, last1 ) );
        if( distVec == 0 )
            return;

        const size_t numComputeUnits = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
        const size_t numWorkGroupsPerComputeUnit = ctl.getWGPerComputeUnit( );
        size_t numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;

        /**********************************************************************************
         * Type Names - used in KernelTemplateSpecializer
         *********************************************************************************/
        std::vector<std::string> scatterIfKernels(scatter_if_endB);
        scatterIfKernels[scatter_if_iType] = TypeName< iType1 >::get( );
        scatterIfKernels[scatter_if_mapType] = TypeName< iType2 >::get( );
        scatterIfKernels[scatter_if_stencilType] = TypeName< iType3 >::get( );
        scatterIfKernels[scatter_if_DVInputIterator] = TypeName< DVInputIterator1 >::get( );
        scatterIfKernels[scatter_if_DVMapType] = TypeName< DVInputIterator2 >::get( );
        scatterIfKernels[scatter_if_DVStencilType] = TypeName< DVInputIterator3 >::get( );
        scatterIfKernels[scatter_if_resultType] = TypeName< oType >::get( );
        scatterIfKernels[scatter_if_DVResultType] = TypeName< DVOutputIterator >::get( );
        scatterIfKernels[scatter_if_Predicate] = TypeName< Predicate >::get();

       /**********************************************************************************
        * Type Definitions - directrly concatenated into kernel string
        *********************************************************************************/

        // For user-defined types, the user must create a TypeName trait which returns the name of the
        //class - note use of TypeName<>::get to retrieve the name here.
        std::vector<std::string> typeDefinitions;
        PUSH_BACK_UNIQUE( typeDefinitions, cl_code)
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType1 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType2 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType3 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVInputIterator1 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVInputIterator2 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVInputIterator3 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< oType >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVOutputIterator >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< Predicate >::get() )
        /**********************************************************************************
         * Calculate WG Size
         *********************************************************************************/

        cl_int l_Error = CL_SUCCESS;
        const size_t wgSize  = WAVEFRONT_SIZE;
        V_OPENCL( l_Error, "Error querying kernel for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" );
        assert( (wgSize & (wgSize-1) ) == 0 ); // The bitwise &,~ logic below requires wgSize to be a power of 2

        int boundsCheck = 0;
        size_t wgMultiple = distVec;
        size_t lowerBits = ( distVec & (wgSize-1) );
        if( lowerBits )
        {
            //  Bump the workitem count to the next multiple of wgSize
            wgMultiple &= ~lowerBits;
            wgMultiple += wgSize;
        }
     
        /**********************************************************************************
         * Compile Options
         *********************************************************************************/
        bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
        //std::cout << "Device is CPU: " << (cpuDevice?"TRUE":"FALSE") << std::endl;
        const size_t kernel_WgSize = (cpuDevice) ? 1 : wgSize;
        std::string compileOptions;
        std::ostringstream oss;
        oss << " -DKERNELWORKGROUPSIZE=" << kernel_WgSize;
        compileOptions = oss.str();

        /**********************************************************************************
          * Request Compiled Kernels
          *********************************************************************************/
         ScatterIf_KernelTemplateSpecializer s_if_kts;
         std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
             ctl,
             scatterIfKernels,
             &s_if_kts,
             typeDefinitions,
             scatter_kernels,
             compileOptions);
         // kernels returned in same order as added in KernelTemplaceSpecializer constructor

        ALIGNED( 256 ) Predicate aligned_binary( pred );
        control::buffPointer userPredicate = ctl.acquireBuffer( sizeof( aligned_binary ),
                                                                CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                                                                &aligned_binary );

        typename DVInputIterator1::Payload first1_payload = first1.gpuPayload( );
        typename DVInputIterator2::Payload map_payload = map.gpuPayload( );
        typename DVInputIterator3::Payload stencil_payload = stencil.gpuPayload( );
        typename DVOutputIterator::Payload result_payload = result.gpuPayload( );

        kernels[boundsCheck].setArg( 0, first1.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 1, first1.gpuPayloadSize( ),&first1_payload);
        kernels[boundsCheck].setArg( 2, map.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 3, map.gpuPayloadSize( ),&map_payload );
        kernels[boundsCheck].setArg( 4, stencil.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 5, stencil.gpuPayloadSize( ),&stencil_payload  );
        kernels[boundsCheck].setArg( 6, result.getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 7, result.gpuPayloadSize( ),&result_payload );
        kernels[boundsCheck].setArg( 8, distVec );
        kernels[boundsCheck].setArg( 9, *userPredicate );

        ::cl::Event scatterIfEvent;
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
            kernels[boundsCheck],
            ::cl::NullRange,
            ::cl::NDRange(wgMultiple), // numWorkGroups*wgSize
            ::cl::NDRange(wgSize),
            NULL,
            &scatterIfEvent );
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for scatter_if() kernel" );

        ::bolt::cl::wait(ctl, scatterIfEvent);

    };


    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVOutputIterator >
    typename std::enable_if< std::is_same< typename std::iterator_traits< DVOutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value
                       >::type
    scatter( bolt::cl::control &ctl,
             const DVInputIterator1& first1,
             const DVInputIterator1& last1,
             const DVInputIterator2& map,
             const DVOutputIterator& result,
             const std::string& cl_code )
    {
        typedef typename std::iterator_traits<DVInputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<DVInputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;

		cl_uint distVec = static_cast< cl_uint >( std::distance( first1, last1 ) );
        if( distVec == 0 )
            return;

        const int numComputeUnits = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
        const int numWorkGroupsPerComputeUnit = ctl.getWGPerComputeUnit( );
        int numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;

        /**********************************************************************************
         * Type Names - used in KernelTemplateSpecializer
         *********************************************************************************/
        std::vector<std::string> scatterKernels(scatter_endB);
        scatterKernels[scatter_iType] = TypeName< iType1 >::get( );
        scatterKernels[scatter_mapType] = TypeName< iType2 >::get( );
        scatterKernels[scatter_DVInputIterator] = TypeName< DVInputIterator1 >::get( );
        scatterKernels[scatter_DVMapType] = TypeName< DVInputIterator2 >::get( );
        scatterKernels[scatter_resultType] = TypeName< oType >::get( );
        scatterKernels[scatter_DVResultType] = TypeName< DVOutputIterator >::get( );

       /**********************************************************************************
        * Type Definitions - directly concatenated into kernel string
        *********************************************************************************/

        // For user-defined types, the user must create a TypeName trait which returns the name of the
        //class - note use of TypeName<>::get to retrieve the name here.
        std::vector<std::string> typeDefinitions;
        PUSH_BACK_UNIQUE( typeDefinitions, cl_code)
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType1 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType2 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVInputIterator1 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVInputIterator2 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< oType >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVOutputIterator >::get() )
        /**********************************************************************************
         * Calculate WG Size
         *********************************************************************************/

        cl_int l_Error = CL_SUCCESS;
        const int wgSize  = WAVEFRONT_SIZE;
        V_OPENCL( l_Error, "Error querying kernel for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" );
        assert( (wgSize & (wgSize-1) ) == 0 ); // The bitwise &,~ logic below requires wgSize to be a power of 2

        int boundsCheck = 0;
        int wgMultiple = distVec;
        int lowerBits = ( distVec & (wgSize-1) );
        if( lowerBits )
        {
            //  Bump the workitem count to the next multiple of wgSize
            wgMultiple &= ~lowerBits;
            wgMultiple += wgSize;
        }

        /**********************************************************************************
         * Compile Options
         *********************************************************************************/
        bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
        //std::cout << "Device is CPU: " << (cpuDevice?"TRUE":"FALSE") << std::endl;
        const int kernel_WgSize = (cpuDevice) ? 1 : wgSize;
        std::string compileOptions;
        std::ostringstream oss;
        oss << " -DKERNELWORKGROUPSIZE=" << kernel_WgSize;
        compileOptions = oss.str();

        /**********************************************************************************
          * Request Compiled Kernels
          *********************************************************************************/
         ScatterKernelTemplateSpecializer s_kts;
         std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
             ctl,
             scatterKernels,
             &s_kts,
             typeDefinitions,
             scatter_kernels,
             compileOptions);
         // kernels returned in same order as added in KernelTemplaceSpecializer constructor
        typename DVInputIterator1::Payload first11_payload = first1.gpuPayload( );
        typename DVInputIterator2::Payload map1_payload = map.gpuPayload( ) ;
        typename DVOutputIterator::Payload result1_payload = result.gpuPayload( );

        kernels[boundsCheck].setArg( 0, first1.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 1, first1.gpuPayloadSize( ),&first11_payload );
        kernels[boundsCheck].setArg( 2, map.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 3, map.gpuPayloadSize( ), &map1_payload);
        kernels[boundsCheck].setArg( 4, result.getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 5, result.gpuPayloadSize( ),&result1_payload );
        kernels[boundsCheck].setArg( 6, distVec );

        ::cl::Event scatterEvent;
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
            kernels[boundsCheck],
            ::cl::NullRange,
            ::cl::NDRange(wgMultiple), // numWorkGroups*wgSize
            ::cl::NDRange(wgSize),
            NULL,
            &scatterEvent );
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for scatter_if() kernel" );

        ::bolt::cl::wait(ctl, scatterEvent);

    };

    template< typename InputIterator1,
              typename MapIterator,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
	typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value
                       >::type
    scatter_if( bolt::cl::control &ctl,
                const InputIterator1& first1,
                const InputIterator1& last1,
                const MapIterator& map,
                const InputIterator3& stencil,
                const OutputIterator& result,
                const Predicate& pred,
                const std::string& user_code)
    {
  
		typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<MapIterator>::value_type iType2;
		typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        int sz = static_cast<int>( std::distance( first1, last1 ) );

        device_vector< oType > dvResult( result, sz, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY, false, ctl );

		
	    // Map the input iterator to a device_vector
	    typedef typename std::iterator_traits<InputIterator1>::pointer pointer;
        typedef typename std::iterator_traits<MapIterator>::pointer map_pointer;
		typedef typename std::iterator_traits<InputIterator3>::pointer ip_pointer;
        pointer first_pointer = bolt::cl::addressof(first1) ;
	    map_pointer map_pointer1 = bolt::cl::addressof(map) ;
		ip_pointer stencil_pointer = bolt::cl::addressof(stencil) ;
	    
        device_vector< iType1 > dvInput( first_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl ); 
        device_vector< iType2 > dvMap( map_pointer1, sz, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, true, ctl );
		device_vector< iType3 > dvStencil( stencil_pointer, sz, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, true, ctl );
        auto device_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                         first1, dvInput.begin() );
        auto device_iterator_last  = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                         last1, dvInput.end() );
		auto map_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< MapIterator >::iterator_category( ), 
                                         map, dvMap.begin() );
		auto stencil_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator3>::iterator_category( ), 
                                         stencil, dvStencil.begin() );

		cl::scatter_if( ctl,
                        device_iterator_first,
                        device_iterator_last,
                        map_iterator_first,
					    stencil_iterator_first,
                        dvResult.begin( ),
					    pred,
                        user_code
			 );
		
        // This should immediately map/unmap the buffer
        dvResult.data( );
       
    }




	template< typename InputIterator,
              typename MapIterator,
              typename OutputIterator>
	typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value
                       >::type
    scatter( bolt::cl::control &ctl,
             const InputIterator& first1,
             const InputIterator& last1,
             const MapIterator& map,
             const OutputIterator& result,
             const std::string& user_code)
    {
  
		typedef typename std::iterator_traits<InputIterator>::value_type iType1;
        typedef typename std::iterator_traits<MapIterator>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        int sz = static_cast<int>( std::distance( first1, last1 ) );

        device_vector< oType > dvResult( result, sz, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY, false, ctl );

		
	    // Map the input iterator to a device_vector
	    typedef typename std::iterator_traits<InputIterator>::pointer pointer;
        typedef typename std::iterator_traits<MapIterator>::pointer map_pointer;
        pointer first_pointer = bolt::cl::addressof(first1) ;
	    map_pointer map_pointer1 = bolt::cl::addressof(map) ;
	    
        device_vector< iType1 > dvInput( first_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl ); 
        device_vector<iType2  > dvMap( map_pointer1, sz, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, true, ctl );
        auto device_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                         first1, dvInput.begin() );
        auto device_iterator_last  = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                         last1, dvInput.end() );
		auto map_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< MapIterator >::iterator_category( ), 
                                         map, dvMap.begin() );

		cl::scatter( ctl,
                     device_iterator_first,
                     device_iterator_last,
                     map_iterator_first,
                     dvResult.begin( ),
                     user_code
			 );
		
        // This should immediately map/unmap the buffer
        dvResult.data( );
       
    }


}// end of namespace cl


    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
	          typename Predicate>
    typename std::enable_if< 
               !(std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               void
                           >::type
    scatter_if( bolt::cl::control& ctl,
                const InputIterator1& first1,
                const InputIterator1& last1,
                const InputIterator2& map,
                const InputIterator3& stencil,
                const OutputIterator& result,
                const Predicate& pred,
                const std::string& user_code )
    {   
		int sz = static_cast<int>( std::distance( first1, last1 ) );
        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::cl::control::Automatic)
        {
          runMode = ctl.getDefaultPathToRun();
        }
	    #if defined(BOLT_DEBUG_LOG)
        BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
        #endif
        if( runMode == bolt::cl::control::SerialCpu )
        {
		   #if defined(BOLT_DEBUG_LOG)
           dblog->CodePathTaken(BOLTLOG::BOLT_SCATTER,BOLTLOG::BOLT_SERIAL_CPU,"::Scatter::SERIAL_CPU");
           #endif
		   serial::scatter_if(ctl, first1, last1, map, stencil, result, pred);
        }
        else if( runMode == bolt::cl::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            {
			    #if defined(BOLT_DEBUG_LOG)
                dblog->CodePathTaken(BOLTLOG::BOLT_SCATTER,BOLTLOG::BOLT_MULTICORE_CPU,"::Scatter::MULTICORE_CPU");
                #endif
                btbb::scatter_if(ctl, first1, last1, map, stencil, result, pred);
            }
#else
            throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );

#endif
        }
        else
        {
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_SCATTER,BOLTLOG::BOLT_OPENCL_GPU,"::Scatter::OPENCL_GPU");
            #endif
				
            cl::scatter_if(ctl, first1, last1, map, stencil, result, pred, user_code);
        }

       
    };


	template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
	          typename Predicate>
    // Wrapper that uses default ::bolt::cl::control class, iterator interface
    typename std::enable_if< 
               (std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               void
                           >::type
    scatter_if( bolt::cl::control& ctl,
                     const InputIterator1& first1,
                     const InputIterator1& last1,
                     const InputIterator2& map,
                     const InputIterator3& stencil,
                     const OutputIterator& result,
                     const Predicate& pred,
                     const std::string& user_code )
    {
		//TODO: map cannot be a constant iterator! Throw compilation error for such a case.
        // TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
        // to a temporary buffer.  Should we?

        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     std::input_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of the type input_iterator_tag" );
        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     bolt::cl::fancy_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of type fancy_iterator_tag" );
    };


	template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator>
	typename std::enable_if< 
               !(std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               void
                           >::type
    scatter( bolt::cl::control& ctl,
             const InputIterator1& first1,
             const InputIterator1& last1,
             const InputIterator2& map,
             const OutputIterator& result,
             const std::string& user_code )
    {
       	
        int sz = static_cast<int>( std::distance( first1, last1 ) );
        if (sz == 0)
            return;

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::cl::control::Automatic)
        {
          runMode = ctl.getDefaultPathToRun();
        }
	    #if defined(BOLT_DEBUG_LOG)
        BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
        #endif
        if( runMode == bolt::cl::control::SerialCpu )
        {
		   #if defined(BOLT_DEBUG_LOG)
           dblog->CodePathTaken(BOLTLOG::BOLT_SCATTER,BOLTLOG::BOLT_SERIAL_CPU,"::Scatter::SERIAL_CPU");
           #endif
		   serial::scatter(ctl, first1, last1, map, result);
        }
        else if( runMode == bolt::cl::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            {
			    #if defined(BOLT_DEBUG_LOG)
                dblog->CodePathTaken(BOLTLOG::BOLT_SCATTER,BOLTLOG::BOLT_MULTICORE_CPU,"::Scatter::MULTICORE_CPU");
                #endif
                btbb::scatter(ctl, first1, last1, map, result);
            }
#else
            throw std::runtime_error( "The MultiCoreCpu version of scatter is not enabled to be built! \n" );

#endif
        }
        else
        {
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_SCATTER,BOLTLOG::BOLT_OPENCL_GPU,"::Scatter::OPENCL_GPU");
            #endif
				
            cl::scatter(ctl, first1, last1, map, result, user_code);
        }

    }



    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator>
	typename std::enable_if< 
               (std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               void
                           >::type
    scatter( bolt::cl::control& ctl,
             const InputIterator1& first1,
             const InputIterator1& last1,
             const InputIterator2& map,
             const OutputIterator& result,
             const std::string& user_code )
    {
		//TODO: map cannot be a constant iterator! Throw compilation error for such a case.

        //static_assert( std::is_same< InputIterator1, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
		static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     std::input_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of the type input_iterator_tag" );
        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     bolt::cl::fancy_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of type fancy_iterator_tag" );
    }

} //End of detail namespace

////////////////////////////////////////////////////////////////////
// Scatter APIs
////////////////////////////////////////////////////////////////////
template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
void scatter( bolt::cl::control& ctl,
              InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 map,
              OutputIterator result,
              const std::string& user_code )
{
    detail::scatter( ctl,
                     first1,
                     last1,
                     map,
                     result,
                     user_code);
}

template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
void scatter( InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 map,
              OutputIterator result,
              const std::string& user_code )
{
    scatter( control::getDefault( ),
             first1,
             last1,
             map,
             result,
             user_code);
}


////////////////////////////////////////////////////////////////////
// ScatterIf APIs
////////////////////////////////////////////////////////////////////
template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >
void scatter_if( bolt::cl::control& ctl,
                 InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 map,
                 InputIterator3 stencil,
                 OutputIterator result,
                 const std::string& user_code )
{
    typedef typename  std::iterator_traits<InputIterator3>::value_type stencilType;
    detail::scatter_if( ctl,
                        first1,
                        last1,
                        map,
                        stencil,
                        result,
                        bolt::cl::identity <stencilType> ( ),
                        user_code );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >
void scatter_if( InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 map,
                 InputIterator3 stencil,
                 OutputIterator result,
                 const std::string& user_code )
{
    scatter_if( control::getDefault( ),
		        first1,
                last1,
                map,
                stencil,
                result,
                user_code );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
void scatter_if( bolt::cl::control& ctl,
                 InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 map,
                 InputIterator3 stencil,
                 OutputIterator result,
                 Predicate pred,
                 const std::string& user_code )
{
    detail::scatter_if( ctl,
                        first1,
                        last1,
                        map,
                        stencil,
                        result,
                        pred,
                        user_code);
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
void scatter_if( InputIterator1 first1,
                 InputIterator1 last1,
                 InputIterator2 map,
                 InputIterator3 stencil,
                 OutputIterator result,
                 Predicate pred,
                 const std::string& user_code )
{
    scatter_if( control::getDefault( ),
                first1,
                last1,
                map,
                stencil,
                result,
                pred,
                user_code);
}



} //End of cl namespace
} //End of bolt namespace

#endif
