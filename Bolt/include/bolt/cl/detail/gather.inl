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
#if !defined( BOLT_CL_GATHER_INL )
#define BOLT_CL_GATHER_INL
#define WAVEFRONT_SIZE 64


#ifdef ENABLE_TBB
    #include "bolt/btbb/gather.h"
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

template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
gather(bolt::cl::control &ctl, 
       InputIterator1 mapfirst,
       InputIterator1 maplast,
       InputIterator2 input,
       OutputIterator result)
{
   int numElements = static_cast< int >( std::distance( mapfirst, maplast ) );
   typedef typename  std::iterator_traits<InputIterator1>::value_type iType1;
   iType1 temp;
   for(int iter = 0; iter < numElements; iter++)
   {
                   temp = *(mapfirst + (int)iter);
                  *(result + (int)iter) = *(input + (int)temp);
   }
}


template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
gather(bolt::cl::control &ctl, 
       InputIterator1 mapfirst,
       InputIterator1 maplast,
       InputIterator2 input,
       OutputIterator result)
{
    typename InputIterator1::difference_type sz = (maplast - mapfirst);
    if (sz == 0)
        return;
    typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
    typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer first1Buffer = mapfirst.base().getContainer( ).getBuffer( );
    ::cl::Buffer first2Buffer = input.base().getContainer( ).getBuffer( );
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
                                                   ctl, mapfirst, first1Ptr);
    auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                   ctl, input, first2Ptr);
    auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                   ctl, result, resultPtr);

	iType1 temp;
    for(int iter = 0; iter < sz; iter++)
    {
           temp = *(mapped_first1_itr + (int)iter);
           *(mapped_result_itr + (int)iter) = *(mapped_first2_itr+ (int)temp);
    }

    ::cl::Event unmap_event[3];
    ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
    ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
    ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[2] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait();
    return;
}


template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
gather_if(bolt::cl::control &ctl,
          InputIterator1 mapfirst,
          InputIterator1 maplast,
          InputIterator2 stencil,
          InputIterator3 input,
          OutputIterator result,
          Predicate pred)
{

   unsigned int numElements = static_cast< unsigned int >( std::distance( mapfirst, maplast ) );
   for(unsigned int iter = 0; iter < numElements; iter++)
   {
        if(pred(*(stencil + (int)iter)))
             result[(int)iter] = input[mapfirst[(int)iter]];
   }
}


template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
gather_if(bolt::cl::control &ctl, 
          InputIterator1 mapfirst,
          InputIterator1 maplast,
          InputIterator2 stencil,
          InputIterator3 input,
          OutputIterator result,
          Predicate pred)
{
	typename InputIterator1::difference_type sz = (maplast - mapfirst);
    if (sz == 0)
        return;
    typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
    typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
	typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer first1Buffer = mapfirst.base().getContainer( ).getBuffer( );
    ::cl::Buffer first2Buffer = stencil.base().getContainer( ).getBuffer( );
	::cl::Buffer first3Buffer = input.base().getContainer( ).getBuffer( );
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
                                                   ctl, mapfirst, first1Ptr);
    auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                   ctl, stencil, first2Ptr);
	auto mapped_first3_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator3>::iterator_category(), 
                                                   ctl, input, first3Ptr);
    auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                   ctl, result, resultPtr);

	for(int iter = 0; iter < sz; iter++)
    {
        if(pred(*(mapped_first2_itr + (int)iter)))
             mapped_result_itr[(int)iter] = mapped_first3_itr[mapped_first1_itr[(int)iter]];
    }


    ::cl::Event unmap_event[4];
    ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
    ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
	ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first3Ptr, NULL, &unmap_event[2] );
    ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[3] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait(); unmap_event[3].wait();
    return;
   
}

}// end of namespace serial


#ifdef ENABLE_TBB
namespace btbb{

template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
gather(bolt::cl::control &ctl, 
       InputIterator1 mapfirst,
       InputIterator1 maplast,
       InputIterator2 input,
       OutputIterator result)
{
   bolt::btbb::gather(mapfirst, maplast, input, result);
}


template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
gather(bolt::cl::control &ctl, 
       InputIterator1 mapfirst,
       InputIterator1 maplast,
       InputIterator2 input,
       OutputIterator result)
{
    typename InputIterator1::difference_type sz = (maplast - mapfirst);
    if (sz == 0)
        return;
    typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
    typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer first1Buffer = mapfirst.base().getContainer( ).getBuffer( );
    ::cl::Buffer first2Buffer = input.base().getContainer( ).getBuffer( );
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
                                                   ctl, mapfirst, first1Ptr);
    auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                   ctl, input, first2Ptr);
    auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                   ctl, result, resultPtr);

	bolt::btbb::gather(mapped_first1_itr, mapped_first1_itr + sz, mapped_first2_itr, mapped_result_itr);

    ::cl::Event unmap_event[3];
    ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
    ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
    ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[2] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait();
    return;
}


template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
gather_if(bolt::cl::control &ctl,
          InputIterator1 mapfirst,
          InputIterator1 maplast,
          InputIterator2 stencil,
          InputIterator3 input,
          OutputIterator result,
          Predicate pred)
{

    bolt::btbb::gather_if(mapfirst, maplast, stencil, input, result, pred);
}


template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
gather_if(bolt::cl::control &ctl, 
          InputIterator1 mapfirst,
          InputIterator1 maplast,
          InputIterator2 stencil,
          InputIterator3 input,
          OutputIterator result,
          Predicate pred)
{
	typename InputIterator1::difference_type sz = (maplast - mapfirst);
    if (sz == 0)
        return;
    typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
    typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
	typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer first1Buffer = mapfirst.base().getContainer( ).getBuffer( );
    ::cl::Buffer first2Buffer = stencil.base().getContainer( ).getBuffer( );
	::cl::Buffer first3Buffer = input.base().getContainer( ).getBuffer( );
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
                                                   ctl, mapfirst, first1Ptr);
    auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                   ctl, stencil, first2Ptr);
	auto mapped_first3_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator3>::iterator_category(), 
                                                   ctl, input, first3Ptr);
    auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                   ctl, result, resultPtr);

	bolt::btbb::gather_if(mapped_first1_itr, mapped_first1_itr + sz, mapped_first2_itr, mapped_first3_itr, mapped_result_itr, pred);


    ::cl::Event unmap_event[4];
    ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
    ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
	ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first3Ptr, NULL, &unmap_event[2] );
    ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[3] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait(); unmap_event[3].wait();
    return;
   
}

}// end of namespace btbb
#endif
namespace cl{
////////////////////////////////////////////////////////////////////
// GatherIf KTS
////////////////////////////////////////////////////////////////////
enum GatherIfTypes { gather_if_mapType, gather_if_DVMapType,
                       gather_if_stencilType, gather_if_DVStencilType,
                       gather_if_iType, gather_if_DVInputIterator,
                       gather_if_resultType, gather_if_DVResultType,
                       gather_if_Predicate, gather_if_endB };

class GatherIf_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
public:
    GatherIf_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
       addKernelName("gatherIfTemplate");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& gatherIfKernels ) const
    {
      const std::string templateSpecializationString =
        "// Host generates this instantiation string with user-specified value type and functor\n"
        "template __attribute__((mangled_name("+name(0)+"Instantiated)))\n"
        "kernel void "+name(0)+"(\n"
        "global " + gatherIfKernels[gather_if_mapType] + "* map, \n"
        + gatherIfKernels[gather_if_DVMapType] + " mapIter, \n"
        "global " + gatherIfKernels[gather_if_stencilType] + "* stencil, \n"
        + gatherIfKernels[gather_if_DVStencilType] + " stencilIter, \n"
        "global " + gatherIfKernels[gather_if_iType] + "* input, \n"
        + gatherIfKernels[gather_if_DVInputIterator] + " inputIter, \n"
        "global " + gatherIfKernels[gather_if_resultType] + "* result, \n"
        + gatherIfKernels[gather_if_DVResultType] + " resultIter, \n"
        "const uint length, \n"
        "global " + gatherIfKernels[gather_if_Predicate] + "* functor);\n\n";

        return templateSpecializationString;
    }
};

////////////////////////////////////////////////////////////////////
// Gather KTS
////////////////////////////////////////////////////////////////////

enum GatherTypes { gather_mapType, gather_DVMapType,
                     gather_iType, gather_DVInputIterator,
                     gather_resultType, gather_DVResultType,
                     gather_endB };

class GatherKernelTemplateSpecializer : public KernelTemplateSpecializer
{
public:
    GatherKernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
       addKernelName("gatherTemplate");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& gatherKernels ) const
    {
      const std::string templateSpecializationString =
        "// Host generates this instantiation string with user-specified value type and functor\n"
        "template __attribute__((mangled_name("+name(0)+"Instantiated)))\n"
        "kernel void "+name(0)+"(\n"
        "global " + gatherKernels[gather_mapType] + "* map, \n"
        + gatherKernels[gather_DVMapType] + " mapIter, \n"
        "global " + gatherKernels[gather_iType] + "* input, \n"
        + gatherKernels[gather_DVInputIterator] + " inputIter, \n"
        "global " + gatherKernels[gather_resultType] + "* result, \n"
        + gatherKernels[gather_DVResultType] + " resultIter, \n"
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
    gather_if( bolt::cl::control &ctl,
                            const DVInputIterator1& map_first,
                            const DVInputIterator1& map_last,
                            const DVInputIterator2& stencil,
                            const DVInputIterator3& input,
                            const DVOutputIterator& result,
                            const Predicate& pred,
                            const std::string& cl_code )
    {
        typedef typename std::iterator_traits<DVInputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<DVInputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<DVInputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;

        cl_uint distVec = static_cast< cl_uint >( std::distance( map_first, map_last ) );
        if( distVec == 0 )
            return;

        const int numComputeUnits = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
        const int numWorkGroupsPerComputeUnit = ctl.getWGPerComputeUnit( );
        int numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;

        /**********************************************************************************
         * Type Names - used in KernelTemplateSpecializer
         *********************************************************************************/

        std::vector<std::string> gatherIfKernels(gather_if_endB);
        gatherIfKernels[gather_if_mapType] = TypeName< iType1 >::get( );
        gatherIfKernels[gather_if_stencilType] = TypeName< iType2 >::get( );
        gatherIfKernels[gather_if_iType] = TypeName< iType3 >::get( );
        gatherIfKernels[gather_if_DVMapType] = TypeName< DVInputIterator1 >::get( );
        gatherIfKernels[gather_if_DVStencilType] = TypeName< DVInputIterator2 >::get( );
        gatherIfKernels[gather_if_DVInputIterator] = TypeName< DVInputIterator3 >::get( );
        gatherIfKernels[gather_if_resultType] = TypeName< oType >::get( );
        gatherIfKernels[gather_if_DVResultType] = TypeName< DVOutputIterator >::get( );
        gatherIfKernels[gather_if_Predicate] = TypeName< Predicate >::get();

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

        //if (wgMultiple/wgSize < numWorkGroups)
        //    numWorkGroups = wgMultiple/wgSize;

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
         GatherIf_KernelTemplateSpecializer s_if_kts;
         std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
             ctl,
             gatherIfKernels,
             &s_if_kts,
             typeDefinitions,
             gather_kernels,
             compileOptions);
         // kernels returned in same order as added in KernelTemplaceSpecializer constructor

        ALIGNED( 256 ) Predicate aligned_binary( pred );
        control::buffPointer userPredicate = ctl.acquireBuffer( sizeof( aligned_binary ),
                                                                CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,
                                                                &aligned_binary );
       typename DVInputIterator1::Payload   map_payload = map_first.gpuPayload( );
       typename DVInputIterator2::Payload   stencil_payload = stencil.gpuPayload( );
       typename DVInputIterator3::Payload   input_payload = input.gpuPayload( );
       typename DVOutputIterator::Payload   result_payload = result.gpuPayload( );

        kernels[boundsCheck].setArg( 0, map_first.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 1, map_first.gpuPayloadSize( ), &map_payload);
        kernels[boundsCheck].setArg( 2, stencil.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 3, stencil.gpuPayloadSize( ),&stencil_payload );
        kernels[boundsCheck].setArg( 4, input.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 5, input.gpuPayloadSize( ), &input_payload );
        kernels[boundsCheck].setArg( 6, result.getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 7, result.gpuPayloadSize( ),&result_payload );
        kernels[boundsCheck].setArg( 8, distVec );
        kernels[boundsCheck].setArg( 9, *userPredicate );

        ::cl::Event gatherIfEvent;
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
            kernels[boundsCheck],
            ::cl::NullRange,
            ::cl::NDRange(wgMultiple), // numWorkGroups*wgSize
            ::cl::NDRange(wgSize),
            NULL,
            &gatherIfEvent );
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for gather_if() kernel" );

        ::bolt::cl::wait(ctl, gatherIfEvent);

    };


	template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
	typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value
                       >::type
    gather_if( bolt::cl::control &ctl,
                            const InputIterator1& map_first,
                            const InputIterator1& map_last,
                            const InputIterator2& stencil,
                            const InputIterator3& input,
                            const OutputIterator& result,
                            const Predicate& pred,
                            const std::string& cl_code )
    {
		typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
		typedef typename std::iterator_traits<InputIterator3>::value_type iType3;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        int sz = static_cast<int>( std::distance( map_first, map_last ) );

        device_vector< oType > dvResult( result, sz, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY, false, ctl );

		
	    // Map the input iterator to a device_vector
	    typedef typename std::iterator_traits<InputIterator3>::pointer pointer;
        typedef typename std::iterator_traits<InputIterator1>::pointer map_pointer;
		typedef typename std::iterator_traits<InputIterator2>::pointer ip_pointer;
        pointer input_pointer = bolt::cl::addressof(input) ;
	    map_pointer map_pointer1 = bolt::cl::addressof(map_first) ;
		ip_pointer stencil_pointer = bolt::cl::addressof(stencil) ;
	    
        device_vector< iType3 > dvInput( input_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl ); 
        device_vector< iType1 > dvMap( map_pointer1, sz, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, true, ctl );
		device_vector< iType2 > dvStencil( stencil_pointer, sz, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, true, ctl );
        auto map_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                         map_first, dvMap.begin() );
        auto map_iterator_last  = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                         map_last, dvMap.end() );
		auto device_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator3 >::iterator_category( ), 
                                         input, dvInput.begin() );
		auto stencil_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator2>::iterator_category( ), 
                                         stencil, dvStencil.begin() );

		cl::gather_if( ctl,
                        map_iterator_first,
                        map_iterator_last,
					    stencil_iterator_first,
						device_iterator_first,
                        dvResult.begin( ),
					    pred,
                        cl_code
			 );
		
        // This should immediately map/unmap the buffer
        dvResult.data( );

	}


    template< typename DVInputIterator1,
              typename DVInputIterator2,
              typename DVOutputIterator >
	typename std::enable_if< std::is_same< typename std::iterator_traits< DVOutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value
                       >::type
    gather( bolt::cl::control &ctl,
                         const DVInputIterator1& map_first,
                         const DVInputIterator1& map_last,
                         const DVInputIterator2& input,
                         const DVOutputIterator& result,
                         const std::string& cl_code )
    {
        typedef typename std::iterator_traits<DVInputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<DVInputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;

        cl_uint distVec = static_cast< cl_uint >( std::distance( map_first, map_last ) );
        if( distVec == 0 )
            return;

        const int numComputeUnits = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
        const int numWorkGroupsPerComputeUnit = ctl.getWGPerComputeUnit( );
        int numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;

        /**********************************************************************************
         * Type Names - used in KernelTemplateSpecializer
         *********************************************************************************/
        std::vector<std::string> gatherKernels(gather_endB);
        gatherKernels[gather_mapType] = TypeName< iType1 >::get( );
        gatherKernels[gather_iType] = TypeName< iType2 >::get( );
        gatherKernels[gather_DVMapType] = TypeName< DVInputIterator1 >::get( );
        gatherKernels[gather_DVInputIterator] = TypeName< DVInputIterator2 >::get( );
        gatherKernels[gather_resultType] = TypeName< oType >::get( );
        gatherKernels[gather_DVResultType] = TypeName< DVOutputIterator >::get( );

       /**********************************************************************************
        * Type Definitions - directrly concatenated into kernel string
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

        //if (wgMultiple/wgSize < numWorkGroups)
        //    numWorkGroups = wgMultiple/wgSize;

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
         GatherKernelTemplateSpecializer s_kts;
         std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
             ctl,
             gatherKernels,
             &s_kts,
             typeDefinitions,
             gather_kernels,
             compileOptions);
         // kernels returned in same order as added in KernelTemplaceSpecializer constructor

        typename DVInputIterator1::Payload   map_payload = map_first.gpuPayload( );
        typename DVInputIterator2::Payload   input_payload = input.gpuPayload( );
        typename DVOutputIterator::Payload   result_payload = result.gpuPayload( );

        kernels[boundsCheck].setArg( 0, map_first.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 1, map_first.gpuPayloadSize( ),&map_payload );
        kernels[boundsCheck].setArg( 2, input.base().getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 3, input.gpuPayloadSize( ),&input_payload );
        kernels[boundsCheck].setArg( 4, result.getContainer().getBuffer() );
        kernels[boundsCheck].setArg( 5, result.gpuPayloadSize( ), &result_payload );
        kernels[boundsCheck].setArg( 6, distVec );

        ::cl::Event gatherEvent;
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
            kernels[boundsCheck],
            ::cl::NullRange,
            ::cl::NDRange(wgMultiple), // numWorkGroups*wgSize
            ::cl::NDRange(wgSize),
            NULL,
            &gatherEvent );
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for gather_if() kernel" );

        ::bolt::cl::wait(ctl, gatherEvent);

    };


	template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
	typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value
                       >::type
    gather( bolt::cl::control &ctl,
                         const InputIterator1& map_first,
                         const InputIterator1& map_last,
                         const InputIterator2& input,
                         const OutputIterator& result,
                         const std::string& cl_code )
    {

		typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        int sz = static_cast<int>( std::distance( map_first, map_last ) );

        device_vector< oType > dvResult( result, sz, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY, false, ctl );

		
	    // Map the input iterator to a device_vector
	    typedef typename std::iterator_traits<InputIterator1>::pointer map_pointer;
        typedef typename std::iterator_traits<InputIterator2>::pointer pointer;
        map_pointer map_pointer1 = bolt::cl::addressof(map_first) ;
	    pointer first_pointer = bolt::cl::addressof(input) ;
	    
        device_vector< iType1 > dvMap( map_pointer1, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, true, ctl ); 
        device_vector<iType2  > dvInput( first_pointer, sz, CL_MEM_USE_HOST_PTR|CL_MEM_READ_WRITE, true, ctl );
        auto map_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                         map_first, dvMap.begin() );
        auto map_iterator_last  = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                         map_last, dvMap.end() );
		auto device_iterator_first = bolt::cl::create_device_itr(
                                         typename bolt::cl::iterator_traits< InputIterator2 >::iterator_category( ), 
                                         input, dvInput.begin() );

		cl::gather( ctl,
                     map_iterator_first,
                     map_iterator_last,
                     device_iterator_first,
                     dvResult.begin( ),
                     cl_code
			 );
		
        // This should immediately map/unmap the buffer
        dvResult.data( );

	}

} // end of cl namespace

	template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
	typename std::enable_if< 
               !(std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               void
                           >::type
    gather_if( bolt::cl::control& ctl,
               const InputIterator1& map_first,
               const InputIterator1& map_last,
               const InputIterator2& stencil,
               const InputIterator3& input,
               const OutputIterator& result,
               const Predicate& pred,
               const std::string& user_code )
    {
        
		int sz = static_cast<int>( std::distance( map_first, map_last ) );
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
          dblog->CodePathTaken(BOLTLOG::BOLT_GATHER,BOLTLOG::BOLT_SERIAL_CPU,"::Gather_If::SERIAL_CPU");
          #endif
			
		  serial::gather_if(ctl, map_first, map_last, stencil, input, result, pred);
 
        }
        else if( runMode == bolt::cl::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
           #if defined(BOLT_DEBUG_LOG)
           dblog->CodePathTaken(BOLTLOG::BOLT_GATHER,BOLTLOG::BOLT_MULTICORE_CPU,"::Gather_If::MULTICORE_CPU");
           #endif
           btbb::gather_if(ctl, map_first, map_last, stencil, input, result, pred);
#else
           throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_GATHER,BOLTLOG::BOLT_OPENCL_GPU,"::Gather_If::OPENCL_GPU");
            #endif
			
            cl::gather_if(ctl, map_first, map_last, stencil, input, result, pred, user_code );
        }
    }


    template< typename InputIterator1,
              typename InputIterator2,
              typename InputIterator3,
              typename OutputIterator,
              typename Predicate >
	typename std::enable_if< 
               (std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               void
                           >::type
    gather_if( bolt::cl::control& ctl,
                                         const InputIterator1& map_first,
                                         const InputIterator1& map_last,
                                         const InputIterator2& stencil,
                                         const InputIterator3& input,
                                         const OutputIterator& result,
                                         const Predicate& pred,
                                         const std::string& user_code )
    {
        //static_assert( std::is_same< InputIterator1, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
		static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     std::input_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of the type input_iterator_tag" );
        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     bolt::cl::fancy_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of type fancy_iterator_tag" );
    };


	template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
	typename std::enable_if< 
               !(std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               void
                           >::type
    gather( bolt::cl::control& ctl,
            const InputIterator1& map_first,
            const InputIterator1& map_last,
            const InputIterator2& input,
            const OutputIterator& result,
            const std::string& user_code)
    {
        
        int sz = static_cast<int>( std::distance( map_first, map_last ) );
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
          dblog->CodePathTaken(BOLTLOG::BOLT_GATHER,BOLTLOG::BOLT_SERIAL_CPU,"::Gather::SERIAL_CPU");
          #endif
			
		  serial::gather(ctl, map_first, map_last, input, result);
 
        }
        else if( runMode == bolt::cl::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
           #if defined(BOLT_DEBUG_LOG)
           dblog->CodePathTaken(BOLTLOG::BOLT_GATHER,BOLTLOG::BOLT_MULTICORE_CPU,"::Gather::MULTICORE_CPU");
           #endif
           btbb::gather(ctl, map_first, map_last , input, result);
#else
           throw std::runtime_error( "The MultiCoreCpu version of gather is not enabled to be built! \n" );
#endif
        }
        else
        {
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_GATHER,BOLTLOG::BOLT_OPENCL_GPU,"::Gather::OPENCL_GPU");
            #endif
			
            cl::gather(ctl, map_first, map_last, input, result, user_code);
        }


    }


    template< typename InputIterator1,
              typename InputIterator2,
              typename OutputIterator >
	typename std::enable_if< 
               (std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               void
                           >::type
    gather( bolt::cl::control& ctl,
            const InputIterator1& map_first,
            const InputIterator1& map_last,
            const InputIterator2& input,
            const OutputIterator& result,
            const std::string& user_code)
    {
        //static_assert( std::is_same< InputIterator1, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
		static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     std::input_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of the type input_iterator_tag" );
        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     bolt::cl::fancy_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of type fancy_iterator_tag" );
    };


} //End of detail namespace


////////////////////////////////////////////////////////////////////
// Gather APIs
////////////////////////////////////////////////////////////////////
template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
void gather( bolt::cl::control& ctl,
             InputIterator1 map_first,
             InputIterator1 map_last,
             InputIterator2 input,
             OutputIterator result,
             const std::string& user_code )
{
    detail::gather( ctl,
                    map_first,
                    map_last,
                    input,
                    result,
                    user_code);
}

template< typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator >
void gather( InputIterator1 map_first,
             InputIterator1 map_last,
             InputIterator2 input,
             OutputIterator result,
             const std::string& user_code )
{
    gather( control::getDefault( ),
    map_first,
    map_last,
    input,
    result,
    user_code);
}


////////////////////////////////////////////////////////////////////
// GatherIf APIs
////////////////////////////////////////////////////////////////////
template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >
void gather_if( bolt::cl::control& ctl,
                InputIterator1 map_first,
                InputIterator1 map_last,
                InputIterator2 stencil,
                InputIterator3 input,
                OutputIterator result,
                const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator2>::value_type stencilType;
    detail::gather_if( ctl,
               map_first,
               map_last,
               stencil,
               input,
               result,
               bolt::cl::identity <stencilType> ( ),
               user_code );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator >
void gather_if( InputIterator1 map_first,
                InputIterator1 map_last,
                InputIterator2 stencil,
                InputIterator3 input,
                OutputIterator result,
                const std::string& user_code )
{
    gather_if( control::getDefault( ),
		       map_first,
               map_last,
               stencil,
               input,
               result,
               user_code );
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
void gather_if( bolt::cl::control& ctl,
                InputIterator1 map_first,
                InputIterator1 map_last,
                InputIterator2 stencil,
                InputIterator3 input,
                OutputIterator result,
                Predicate pred,
                const std::string& user_code )
{
    detail::gather_if( ctl,
                       map_first,
                       map_last,
                       stencil,
                       input,
                       result,
                       pred,
                       user_code);
}

template< typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename Predicate >
void gather_if(  InputIterator1 map_first,
                 InputIterator1 map_last,
                 InputIterator2 stencil,
                 InputIterator3 input,
                 OutputIterator result,
                 Predicate pred,
                 const std::string& user_code )
{
    gather_if( control::getDefault( ),
               map_first,
               map_last,
               stencil,
               input,
               result,
               pred,
               user_code);
}


} //End of cl namespace
} //End of bolt namespace

#endif
