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


#if !defined( BOLT_CL_REDUCE_BY_KEY_INL )
#define BOLT_CL_REDUCE_BY_KEY_INL

#define KERNEL02WAVES 4
#define KERNEL1WAVES 4
#define WAVESIZE 64

#define LENGTH_TEST 10
#define ENABLE_PRINTS 0

#include "bolt/cl/scan.h"

#include <bolt/cl/iterator/iterator_traits.h>
#include <bolt/cl/iterator/addressof.h>
#include "bolt/cl/device_vector.h"
#include "bolt/cl/distance.h"
#include "bolt/cl/iterator/transform_iterator.h"

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/reduce_by_key.h"
#endif

namespace bolt
{

namespace cl
{

namespace detail
{
/*!
*   \internal
*   \addtogroup detail
*   \ingroup reduction
*   \{
*/

namespace serial{


template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction>
typename std::enable_if< (std::is_same< typename std::iterator_traits< OutputIterator1 >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value &&
						  std::is_same< typename std::iterator_traits< OutputIterator2 >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value), unsigned int
                           >::type
reduce_by_key( ::bolt::cl::control &ctl, 
               InputIterator1 keys_first,
               InputIterator1 keys_last,
               InputIterator2 values_first,
               OutputIterator1 keys_output,
               OutputIterator2 values_output,
               BinaryPredicate binary_pred,
               BinaryFunction binary_op )
{

    typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< OutputIterator1 >::value_type koType;
    typedef typename std::iterator_traits< OutputIterator2 >::value_type voType;

    unsigned int numElements = static_cast< unsigned int >( std::distance( keys_first, keys_last ) );

    // do zeroeth element
    *values_output = *values_first;
    *keys_output = *keys_first;
    unsigned int count = 1;
    // rbk oneth element and beyond

    values_first++;
    for ( InputIterator1 key = (keys_first+1); key != keys_last; key++)
    {
        // load keys
        //kType currentKey  = *(key);
        //kType previousKey = *(key-1);

        // load value
        voType currentValue = *values_first;
        voType previousValue = *values_output;

        previousValue = *values_output;
        // within segment
        if (binary_pred(*(key), *(key-1)))
        {
            voType r = binary_op( previousValue, currentValue);
            *values_output = r;
            *keys_output = *(key);

        }
        else // new segment
        {
            values_output++;
            keys_output++;
            *values_output = currentValue;
            *keys_output = *(key);
            count++; //To count the number of elements in the output array
        }
        values_first++;
    }

    return count;
}



template<
    typename DVInputIterator1,
    typename DVInputIterator2,
    typename DVOutputIterator1,
    typename DVOutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction >
typename std::enable_if< (std::is_same< typename std::iterator_traits< DVOutputIterator1 >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value &&
                         std::is_same< typename std::iterator_traits< DVOutputIterator1 >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value),
					     unsigned int
                       >::type
reduce_by_key(
    ::bolt::cl::control &ctl, 
    DVInputIterator1& keys_first,
    DVInputIterator1& keys_last,
    DVInputIterator2& values_first,
    DVOutputIterator1& keys_output,
    DVOutputIterator2& values_output,
    BinaryPredicate& binary_pred,
    BinaryFunction& binary_op)
{

	typedef typename std::iterator_traits< DVInputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< DVInputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< DVOutputIterator1 >::value_type koType;
    typedef typename std::iterator_traits< DVOutputIterator2 >::value_type voType;

    unsigned int sz = static_cast< unsigned int >( std::distance( keys_first, keys_last ) );

    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer keyfirstBuffer  = keys_first.base().getContainer( ).getBuffer( );
	::cl::Buffer valfirstBuffer  = values_first.base().getContainer( ).getBuffer( );
    ::cl::Buffer keyresultBuffer = keys_output.getContainer( ).getBuffer( );
	::cl::Buffer valresultBuffer = values_output.getContainer( ).getBuffer( );

    /*Get The size of each OpenCL buffer*/
    size_t keyfirst_sz  = keyfirstBuffer.getInfo<CL_MEM_SIZE>();
	size_t valfirst_sz = valfirstBuffer.getInfo<CL_MEM_SIZE>();
    size_t keyresult_sz = keyresultBuffer.getInfo<CL_MEM_SIZE>();
	size_t valresult_sz = valresultBuffer.getInfo<CL_MEM_SIZE>();

    cl_int map_err;
    kType *keyfirstPtr  = (kType*)ctl.getCommandQueue().enqueueMapBuffer(keyfirstBuffer, true, CL_MAP_READ, 0, 
                                                                        keyfirst_sz, NULL, NULL, &map_err);
	vType *valfirstPtr  = (vType*)ctl.getCommandQueue().enqueueMapBuffer(valfirstBuffer, true, CL_MAP_READ, 0, 
                                                                        valfirst_sz, NULL, NULL, &map_err);
    koType *keyresultPtr = (koType*)ctl.getCommandQueue().enqueueMapBuffer(keyresultBuffer, true, CL_MAP_WRITE, 0, 
                                                                        keyresult_sz, NULL, NULL, &map_err);
	voType *valresultPtr = (voType*)ctl.getCommandQueue().enqueueMapBuffer(valresultBuffer, true, CL_MAP_WRITE, 0, 
                                                                        valresult_sz, NULL, NULL, &map_err);

    auto mapped_keyfirst_itr = create_mapped_iterator(typename std::iterator_traits<DVInputIterator1>::
	                                          iterator_category(), 
                                                    ctl, keys_first, keyfirstPtr);
	auto mapped_valfirst_itr = create_mapped_iterator(typename std::iterator_traits<DVInputIterator2>::
	                                          iterator_category(), 
                                                    ctl, values_first,valfirstPtr);
    auto mapped_keyresult_itr = create_mapped_iterator(typename std::iterator_traits<DVOutputIterator1>::
	                                          iterator_category(), 
                                                    ctl, keys_output, keyresultPtr);
	auto mapped_valresult_itr = create_mapped_iterator(typename std::iterator_traits<DVOutputIterator2>::
	                                          iterator_category(), 
                                                    ctl, values_output, valresultPtr);

	// do zeroeth element
    mapped_valresult_itr[0] = mapped_valfirst_itr[0];
    mapped_keyresult_itr[0] = mapped_keyfirst_itr[0];
    unsigned int count = 1;
    // rbk oneth element and beyond

    unsigned int vi=1, vo=0, ko=0;
    for ( unsigned int i = 1; i < sz; i++)
    {
        // load keys
        //kType currentKey  = mapped_keyfirst_itr[i];
        //kType previousKey = mapped_keyfirst_itr[i-1];

        // load value
        voType currentValue = mapped_valfirst_itr[vi];
        voType previousValue = mapped_valresult_itr[vo];

        previousValue = mapped_valresult_itr[vo];
        // within segment
        if (binary_pred(mapped_keyfirst_itr[i], mapped_keyfirst_itr[i-1]))
        {
            voType r = binary_op( previousValue, currentValue);
            mapped_valresult_itr[vo] = r;
            mapped_keyresult_itr[ko] = mapped_keyfirst_itr[i];

        }
        else // new segment
        {
			vo++;
			ko++;
            mapped_valresult_itr[vo] = currentValue;
            mapped_keyresult_itr[ko] = mapped_keyfirst_itr[i];
            count++; //To count the number of elements in the output array
        }
		vi++;
    }

    ::cl::Event unmap_event[4];
    ctl.getCommandQueue().enqueueUnmapMemObject(keyfirstBuffer, keyfirstPtr, NULL, &unmap_event[0] );
	ctl.getCommandQueue().enqueueUnmapMemObject(valfirstBuffer, valfirstPtr, NULL, &unmap_event[1] );
    ctl.getCommandQueue().enqueueUnmapMemObject(keyresultBuffer, keyresultPtr, NULL, &unmap_event[2] );
	ctl.getCommandQueue().enqueueUnmapMemObject(valresultBuffer, valresultPtr, NULL, &unmap_event[3] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait(); unmap_event[3].wait(); 

    return count;

}


} // end of namespace serial

#ifdef ENABLE_TBB
namespace btbb{


template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction>
typename std::enable_if< (std::is_same< typename std::iterator_traits< OutputIterator1 >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value &&
						  std::is_same< typename std::iterator_traits< OutputIterator2 >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value), unsigned int
                           >::type
reduce_by_key( ::bolt::cl::control &ctl, 
               InputIterator1 keys_first,
               InputIterator1 keys_last,
               InputIterator2 values_first,
               OutputIterator1 keys_output,
               OutputIterator2 values_output,
               BinaryPredicate binary_pred,
               BinaryFunction binary_op )
{
    return bolt::btbb::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}



template<
    typename DVInputIterator1,
    typename DVInputIterator2,
    typename DVOutputIterator1,
    typename DVOutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction >
typename std::enable_if< (std::is_same< typename std::iterator_traits< DVOutputIterator1 >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value &&
                         std::is_same< typename std::iterator_traits< DVOutputIterator1 >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value),
					     unsigned int
					  >::type
reduce_by_key(
    ::bolt::cl::control &ctl, 
    DVInputIterator1& keys_first,
    DVInputIterator1& keys_last,
    DVInputIterator2& values_first,
    DVOutputIterator1& keys_output,
    DVOutputIterator2& values_output,
    BinaryPredicate& binary_pred,
    BinaryFunction& binary_op)
{
	typedef typename std::iterator_traits< DVInputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< DVInputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< DVOutputIterator1 >::value_type koType;
    typedef typename std::iterator_traits< DVOutputIterator2 >::value_type voType;

    unsigned int sz = static_cast< unsigned int >( std::distance( keys_first, keys_last ) );

    /*Get The associated OpenCL buffer for each of the iterators*/
    ::cl::Buffer keyfirstBuffer  = keys_first.base().getContainer( ).getBuffer( );
	::cl::Buffer valfirstBuffer  = values_first.base().getContainer( ).getBuffer( );
    ::cl::Buffer keyresultBuffer = keys_output.getContainer( ).getBuffer( );
	::cl::Buffer valresultBuffer = values_output.getContainer( ).getBuffer( );

    /*Get The size of each OpenCL buffer*/
    size_t keyfirst_sz  = keyfirstBuffer.getInfo<CL_MEM_SIZE>();
	size_t valfirst_sz = valfirstBuffer.getInfo<CL_MEM_SIZE>();
    size_t keyresult_sz = keyresultBuffer.getInfo<CL_MEM_SIZE>();
	size_t valresult_sz = valresultBuffer.getInfo<CL_MEM_SIZE>();

    cl_int map_err;
    kType *keyfirstPtr  = (kType*)ctl.getCommandQueue().enqueueMapBuffer(keyfirstBuffer, true, CL_MAP_READ, 0, 
                                                                        keyfirst_sz, NULL, NULL, &map_err);
	vType *valfirstPtr  = (vType*)ctl.getCommandQueue().enqueueMapBuffer(valfirstBuffer, true, CL_MAP_READ, 0, 
                                                                        valfirst_sz, NULL, NULL, &map_err);
    koType *keyresultPtr = (koType*)ctl.getCommandQueue().enqueueMapBuffer(keyresultBuffer, true, CL_MAP_WRITE, 0, 
                                                                        keyresult_sz, NULL, NULL, &map_err);
	voType *valresultPtr = (voType*)ctl.getCommandQueue().enqueueMapBuffer(valresultBuffer, true, CL_MAP_WRITE, 0, 
                                                                        valresult_sz, NULL, NULL, &map_err);

    auto mapped_keyfirst_itr = create_mapped_iterator(typename std::iterator_traits<DVInputIterator1>::
	                                          iterator_category(), 
                                                    ctl, keys_first, keyfirstPtr);
	auto mapped_valfirst_itr = create_mapped_iterator(typename std::iterator_traits<DVInputIterator2>::
	                                          iterator_category(), 
                                                    ctl, values_first,valfirstPtr);
    auto mapped_keyresult_itr = create_mapped_iterator(typename std::iterator_traits<DVOutputIterator1>::
	                                          iterator_category(), 
                                                    ctl, keys_output, keyresultPtr);
	auto mapped_valresult_itr = create_mapped_iterator(typename std::iterator_traits<DVOutputIterator2>::
	                                          iterator_category(), 
                                                    ctl, values_output, valresultPtr);

	unsigned int count = bolt::btbb::reduce_by_key( mapped_keyfirst_itr,  mapped_keyfirst_itr + sz, mapped_valfirst_itr, 
		mapped_keyresult_itr, mapped_valresult_itr, binary_pred, binary_op);

    ::cl::Event unmap_event[4];
    ctl.getCommandQueue().enqueueUnmapMemObject(keyfirstBuffer, keyfirstPtr, NULL, &unmap_event[0] );
	ctl.getCommandQueue().enqueueUnmapMemObject(valfirstBuffer, valfirstPtr, NULL, &unmap_event[1] );
    ctl.getCommandQueue().enqueueUnmapMemObject(keyresultBuffer, keyresultPtr, NULL, &unmap_event[2] );
	ctl.getCommandQueue().enqueueUnmapMemObject(valresultBuffer, valresultPtr, NULL, &unmap_event[3] );
    unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait(); unmap_event[3].wait(); 

    return count;

}

}//end of namespace btbb
#endif

namespace cl{
    enum Reduce_By_Key_Types {  e_kType, e_kIterType,
                     e_vType, e_vIterType,
                     e_koType, e_koIterType,
                     e_voType, e_voIterType ,
                     e_BinaryPredicate, e_BinaryFunction,
                     e_end };



    /* "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(3) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(KERNEL2WORKGROUPSIZE,1,1)))\n"
            "__kernel void " + name(3) + "(\n"
            "global  int *h_result\n"
            ");\n\n"*/


/*********************************************************************************************************************
 * Kernel Template Specializer
 *********************************************************************************************************************/
class ReduceByKey_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
    public:

    ReduceByKey_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
		addKernelName("OffsetCalculation");
        addKernelName("perBlockScanByKey");
        addKernelName("intraBlockInclusiveScanByKey");
        addKernelName("keyValueMapping");
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
    {
        const std::string templateSpecializationString =
            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(KERNEL0WORKGROUPSIZE,1,1)))\n"
            "__kernel void " + name(0) + "(\n"
            "global " + typeNames[e_kType] + "* ikeys,\n"
            + typeNames[e_kIterType] + " keys,\n"
            "global int *tempArray,\n"
            "const uint vecSize,\n"
            "global " + typeNames[e_BinaryPredicate] + "* binaryPred\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(1) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(KERNEL0WORKGROUPSIZE,1,1)))\n"
            "__kernel void " + name(1) + "(\n"
            "global int *keys,\n"
            "global " + typeNames[e_vType] + "* ivals,\n"
            + typeNames[e_vIterType] + " vals,\n"
            "const uint vecSize,\n"
            "local int * ldsKeys,\n"
            "local "  + typeNames[e_vIterType] + "::value_type * ldsVals,\n"
            "global " + typeNames[e_BinaryFunction]  + "* binaryFunct,\n"
            "global int * keyBuffer,\n"
            "global " + typeNames[e_vIterType] + "::value_type * valBuffer\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(2) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(KERNEL1WORKGROUPSIZE,1,1)))\n"
            "__kernel void " + name(2) + "(\n"
            "global int * keySumArray,\n"
            "global " + typeNames[e_vIterType] + "::value_type * preSumArray,\n"
            "global " + typeNames[e_vIterType] + "::value_type * postSumArray,\n"
            "const uint vecSize,\n"
            "local int * ldsKeys,\n"
            "local " + typeNames[e_vIterType] + "::value_type * ldsVals,\n"
            "const uint workPerThread,\n"
            "global " + typeNames[e_BinaryFunction] + "* binaryFunct\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(3) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(KERNEL0WORKGROUPSIZE,1,1)))\n"
            "__kernel void " + name(3) + "(\n"
            "global " + typeNames[e_kType] + "*ikeys,\n"
            + typeNames[e_kIterType] + " keys,\n"
            "global " + typeNames[e_koType] + "*ikeys_output,\n"
            + typeNames[e_koIterType] + " keys_output,\n"
			"global " + typeNames[e_vType] + "* ivals,\n"
            + typeNames[e_vIterType] + " vals,\n"
            "global " + typeNames[e_voType] + "*ivals_output,\n"
			+ typeNames[e_voIterType] + " vals_output,\n"
			"local int * ldsKeys,\n"
            "local " + typeNames[e_vIterType] + "::value_type * ldsVals,\n"
            "global int *newkeys,\n"
			"global int * keySumArray,\n"
            "global " + typeNames[e_vIterType] + "::value_type * postSumArray,\n"
            "const uint vecSize, \n"
			"global " + typeNames[e_BinaryFunction] + "* binaryFunct\n"
            ");\n\n";

        return templateSpecializationString;
    }
};


//  All calls to reduce_by_key end up here, unless an exception was thrown
//  This is the function that sets up the kernels to compile (once only) and execute
template<
    typename DVInputIterator1,
    typename DVInputIterator2,
    typename DVOutputIterator1,
    typename DVOutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction >
typename std::enable_if< (std::is_same< typename std::iterator_traits< DVOutputIterator1 >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value &&
						  std::is_same< typename std::iterator_traits< DVOutputIterator2 >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value), unsigned int
                           >::type
reduce_by_key(
    control& ctl,
    const DVInputIterator1& keys_first,
    const DVInputIterator1& keys_last,
    const DVInputIterator2& values_first,
    const DVOutputIterator1& keys_output,
    const DVOutputIterator2& values_output,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_op,
    const std::string& user_code)
{

    cl_int l_Error;

    /**********************************************************************************
     * Type Names - used in KernelTemplateSpecializer
     *********************************************************************************/
    typedef typename std::iterator_traits< DVInputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< DVInputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< DVOutputIterator1 >::value_type koType;
    typedef typename std::iterator_traits< DVOutputIterator2 >::value_type voType;
    std::vector<std::string> typeNames(e_end);
    typeNames[e_kType] = TypeName< kType >::get( );
    typeNames[e_kIterType] = TypeName< DVInputIterator1 >::get( );
    typeNames[e_vType] = TypeName< vType >::get( );
    typeNames[e_vIterType] = TypeName< DVInputIterator2 >::get( );
    typeNames[e_koType] = TypeName< koType >::get( );
    typeNames[e_koIterType] = TypeName< DVOutputIterator1 >::get( );
    typeNames[e_voType] = TypeName< voType >::get( );
    typeNames[e_voIterType] = TypeName< DVOutputIterator2 >::get( );
    typeNames[e_BinaryPredicate] = TypeName< BinaryPredicate >::get( );
    typeNames[e_BinaryFunction]  = TypeName< BinaryFunction >::get( );

    /**********************************************************************************
     * Type Definitions - directly concatenated into kernel string
     *********************************************************************************/
    
    std::vector<std::string> typeDefs; // typeDefs must be unique and order does matter
    PUSH_BACK_UNIQUE( typeDefs, ClCode< kType >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< DVInputIterator1 >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< vType >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< DVInputIterator2 >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< koType >::get() )
	PUSH_BACK_UNIQUE( typeDefs, ClCode< DVOutputIterator1 >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< voType >::get() )
	PUSH_BACK_UNIQUE( typeDefs, ClCode< DVOutputIterator2 >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< BinaryPredicate >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< BinaryFunction  >::get() )

    /**********************************************************************************
     * Compile Options
     *********************************************************************************/
    bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
    //std::cout << "Device is CPU: " << (cpuDevice?"TRUE":"FALSE") << std::endl;
    const int kernel0_WgSize = (cpuDevice) ? 1 : WAVESIZE*KERNEL02WAVES;
    const int kernel1_WgSize = (cpuDevice) ? 1 : WAVESIZE*KERNEL1WAVES;
    const int kernel2_WgSize = (cpuDevice) ? 1 : WAVESIZE*KERNEL02WAVES;
    std::string compileOptions;
    std::ostringstream oss;
    oss << " -DKERNEL0WORKGROUPSIZE=" << kernel0_WgSize;
    oss << " -DKERNEL1WORKGROUPSIZE=" << kernel1_WgSize;
    oss << " -DKERNEL2WORKGROUPSIZE=" << kernel2_WgSize;
    compileOptions = oss.str();

    /**********************************************************************************
     * Request Compiled Kernels
     *********************************************************************************/
    ReduceByKey_KernelTemplateSpecializer ts_kts;
    std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
        ctl,
        typeNames,
        &ts_kts,
        typeDefs,
        reduce_by_key_kernels,
        compileOptions);
    // kernels returned in same order as added in KernelTemplaceSpecializer constructor

    // for profiling
    ::cl::Event kernel0Event, kernel1Event, kernel2Event, kernelAEvent, kernel3Event;

    // Set up shape of launch grid and buffers:
    int computeUnits     = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
    int wgPerComputeUnit =  ctl.getWGPerComputeUnit( );
    int resultCnt = computeUnits * wgPerComputeUnit;

    //  Ceiling function to bump the size of input to the next whole wavefront size
    cl_uint numElements = static_cast< cl_uint >( std::distance( keys_first, keys_last ) );
    typename device_vector< kType >::size_type sizeInputBuff = numElements;
    int modWgSize = (sizeInputBuff & (kernel0_WgSize-1));
    if( modWgSize )
    {
        sizeInputBuff &= ~modWgSize;
        sizeInputBuff += kernel0_WgSize;
    }
    cl_uint numWorkGroupsK0 = static_cast< cl_uint >( sizeInputBuff / kernel0_WgSize );

    //  Ceiling function to bump the size of the sum array to the next whole wavefront size
    typename device_vector< kType >::size_type sizeScanBuff = numWorkGroupsK0;
    modWgSize = (sizeScanBuff & (kernel0_WgSize-1));
    if( modWgSize )
    {
        sizeScanBuff &= ~modWgSize;
        sizeScanBuff += kernel0_WgSize;
    }

    // Create buffer wrappers so we can access the host functors, for read or writing in the kernel

    ALIGNED( 256 ) BinaryPredicate aligned_binary_pred( binary_pred );
    control::buffPointer binaryPredicateBuffer = ctl.acquireBuffer( sizeof( aligned_binary_pred ),
        CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_binary_pred );
     ALIGNED( 256 ) BinaryFunction aligned_binary_op( binary_op );
    control::buffPointer binaryFunctionBuffer = ctl.acquireBuffer( sizeof( aligned_binary_op ),
        CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_binary_op );


    device_vector< int > tempArray( numElements, 0, CL_MEM_READ_WRITE, false, ctl);
    ::cl::Buffer tempArrayVec = tempArray.begin( ).base().getContainer().getBuffer();



    /**********************************************************************************
     *  Kernel 0
     *********************************************************************************/
    typename DVInputIterator1::Payload keys_first_payload = keys_first.gpuPayload( );
    try
    {
    V_OPENCL( kernels[0].setArg( 0, keys_first.base().getContainer().getBuffer()), "Error setArg kernels[ 0 ]" ); // Input keys
    V_OPENCL( kernels[0].setArg( 1, keys_first.gpuPayloadSize( ),&keys_first_payload ), "Error setArg kernels[ 0 ]" );
    V_OPENCL( kernels[0].setArg( 2, tempArrayVec ), "Error setArg kernels[ 0 ]" ); // Output keys
    V_OPENCL( kernels[0].setArg( 3, numElements ), "Error setArg kernels[ 0 ]" ); // vecSize
    V_OPENCL( kernels[0].setArg( 4, *binaryPredicateBuffer),"Error setArg kernels[ 0 ]" ); // User provided functor

    l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
        kernels[0],
        ::cl::NullRange,
        ::cl::NDRange( sizeInputBuff ),
        ::cl::NDRange( kernel0_WgSize ),
        NULL,
        &kernel0Event);
    V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel[0]" );
    }
    catch( const ::cl::Error& e)
    {
        std::cerr << "::cl::enqueueNDRangeKernel( 0 ) in bolt::cl::reduce_by_key_enqueue()" << std::endl;
        std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
        std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
        std::cerr << "Error String: " << e.what() << std::endl;
    }
    // wait for results
    l_Error = kernel0Event.wait( );
    V_OPENCL( l_Error, "post-kernel[0] failed wait" );


    bolt::cl::detail::cl::scan(ctl, tempArray.begin(), tempArray.end(), tempArray.begin(), 0, true, plus< int >( ), user_code);

    control::buffPointer keySumArray  = ctl.acquireBuffer( sizeScanBuff*sizeof( int ) );
    control::buffPointer preSumArray  = ctl.acquireBuffer( sizeScanBuff*sizeof( vType ) );
    control::buffPointer postSumArray = ctl.acquireBuffer( sizeScanBuff*sizeof( vType ) );


    /**********************************************************************************
     *  Kernel 1
     *********************************************************************************/
    cl_uint ldsKeySize, ldsValueSize;
    typename DVInputIterator2::Payload values_first_payload = values_first.gpuPayload( );
    try
    {
    ldsKeySize   = static_cast< cl_uint >( kernel0_WgSize * sizeof( int ) );
    ldsValueSize = static_cast< cl_uint >( kernel0_WgSize * sizeof( voType ) );
    V_OPENCL( kernels[1].setArg( 0, tempArrayVec), "Error setArg kernels[ 1 ]" ); // Input keys
    V_OPENCL( kernels[1].setArg( 1, values_first.base().getContainer().getBuffer()),"Error setArg kernels[ 1 ]" ); // Input values
    V_OPENCL( kernels[1].setArg( 2, values_first.gpuPayloadSize( ),&values_first_payload ), "Error setArg kernels[ 1 ]" ); // Input values
    V_OPENCL( kernels[1].setArg( 3, numElements ), "Error setArg kernels[ 1 ]" ); // vecSize
    V_OPENCL( kernels[1].setArg( 4, ldsKeySize, NULL ),     "Error setArg kernels[ 1 ]" ); // Scratch buffer
    V_OPENCL( kernels[1].setArg( 5, ldsValueSize, NULL ),   "Error setArg kernels[ 1 ]" ); // Scratch buffer
    V_OPENCL( kernels[1].setArg( 6, *binaryFunctionBuffer ),"Error setArg kernels[ 1 ]" ); // User provided functor
    V_OPENCL( kernels[1].setArg( 7, *keySumArray ),         "Error setArg kernels[ 1 ]" ); // Output per block sum
    V_OPENCL( kernels[1].setArg( 8, *preSumArray ),         "Error setArg kernels[ 1 ]" ); // Output per block sum

    l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
        kernels[1],
        ::cl::NullRange,
        ::cl::NDRange( sizeInputBuff ),
        ::cl::NDRange( kernel0_WgSize ),
        NULL,
        &kernel1Event);
    V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel[1]" );
    }
    catch( const ::cl::Error& e)
    {
        std::cerr << "::cl::enqueueNDRangeKernel( 1 ) in bolt::cl::reduce_by_key_enqueue()" << std::endl;
        std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
        std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
        std::cerr << "Error String: " << e.what() << std::endl;
    }
    /**********************************************************************************
     *  Kernel 2
     *********************************************************************************/
    cl_uint workPerThread = static_cast< cl_uint >( sizeScanBuff / kernel1_WgSize );
    V_OPENCL( kernels[2].setArg( 0, *keySumArray ),         "Error setArg kernels[ 2 ]" ); // Input keys
    V_OPENCL( kernels[2].setArg( 1, *preSumArray ),         "Error setArg kernels[ 2 ]" ); // Input buffer
    V_OPENCL( kernels[2].setArg( 2, *postSumArray ),        "Error setArg kernels[ 2 ]" ); // Output buffer
    V_OPENCL( kernels[2].setArg( 3, numWorkGroupsK0 ),      "Error setArg kernels[ 2 ]" ); // Size of scratch buffer
    V_OPENCL( kernels[2].setArg( 4, ldsKeySize, NULL ),     "Error setArg kernels[ 2 ]" ); // Scratch buffer
    V_OPENCL( kernels[2].setArg( 5, ldsValueSize, NULL ),   "Error setArg kernels[ 2 ]" ); // Scratch buffer
    V_OPENCL( kernels[2].setArg( 6, workPerThread ),        "Error setArg kernels[ 2 ]" ); // Work Per Thread
    V_OPENCL( kernels[2].setArg( 7, *binaryFunctionBuffer ),"Error setArg kernels[ 2 ]" ); // User provided functor

    try
    {
    l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
        kernels[2],
        ::cl::NullRange,
        ::cl::NDRange( kernel1_WgSize ), // only 1 work-group
        ::cl::NDRange( kernel1_WgSize ),
        NULL,
        &kernel2Event);
    V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel[2]" );
    }
    catch( const ::cl::Error& e)
    {
        std::cerr << "::cl::enqueueNDRangeKernel( 2 ) in bolt::cl::reduce_by_key_enqueue()" << std::endl;
        std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
        std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
        std::cerr << "Error String: " << e.what() << std::endl;
    }

    /**********************************************************************************
     *  Kernel 3
     *********************************************************************************/
    typename DVInputIterator1::Payload keys_first1_payload = keys_first.gpuPayload( );
    typename DVOutputIterator1::Payload keys_output_payload = keys_output.gpuPayload( );
    typename DVInputIterator2::Payload  value_first1_payload = values_first.gpuPayload( );
    typename DVOutputIterator2::Payload values_output_payload = values_output.gpuPayload( );

    V_OPENCL( kernels[3].setArg( 0, keys_first.base().getContainer().getBuffer()),			 "Error setArg kernels[ 3 ]" ); // Input buffer
    V_OPENCL( kernels[3].setArg( 1, keys_first.gpuPayloadSize( ), &keys_first1_payload),	 "Error setArg kernels[ 3 ]" );
    V_OPENCL( kernels[3].setArg( 2, keys_output.getContainer().getBuffer() ),				 "Error setArg kernels[ 3 ]" ); // Output buffer
    V_OPENCL( kernels[3].setArg( 3, keys_output.gpuPayloadSize( ),&keys_output_payload ),	 "Error setArg kernels[ 3 ]" );
	V_OPENCL( kernels[3].setArg( 4, values_first.base().getContainer().getBuffer()),		 "Error setArg kernels[ 3 ]" ); // Input values
    V_OPENCL( kernels[3].setArg( 5, values_first.gpuPayloadSize( ),&value_first1_payload ),  "Error setArg kernels[ 3 ]" ); // Input values
    V_OPENCL( kernels[3].setArg( 6, values_output.getContainer().getBuffer()),				 "Error setArg kernels[ 3 ]" ); // Output buffer
    V_OPENCL( kernels[3].setArg( 7, values_output.gpuPayloadSize( ),&values_output_payload ),"Error setArg kernels[ 3 ]" );
	V_OPENCL( kernels[3].setArg( 8, ldsKeySize, NULL ),										 "Error setArg kernels[ 3 ]" ); // Scratch buffer
    V_OPENCL( kernels[3].setArg( 9, ldsValueSize, NULL ),									 "Error setArg kernels[ 3 ]" ); // Scratch buffer
    V_OPENCL( kernels[3].setArg( 10, tempArrayVec),											 "Error setArg kernels[ 3 ]" ); // Input keys
	V_OPENCL( kernels[3].setArg( 11, *keySumArray ),										 "Error setArg kernels[ 3 ]" ); // Input buffer
    V_OPENCL( kernels[3].setArg( 12, *postSumArray ),										 "Error setArg kernels[ 3 ]" ); // Input buffer
    V_OPENCL( kernels[3].setArg( 13, numElements ),											 "Error setArg kernels[ 3 ]" ); // Size of scratch buffer
    V_OPENCL( kernels[3].setArg( 14, *binaryFunctionBuffer),								 "Error setArg kernels[ 3 ]" ); // User provided functor

    try
    {
    l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
        kernels[3],
        ::cl::NullRange,
        ::cl::NDRange( sizeInputBuff ),
        ::cl::NDRange( kernel0_WgSize ),
        NULL,
        &kernel3Event );
    V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel[3]" );
    }
    catch( const ::cl::Error& e)
    {
        std::cerr << "::cl::enqueueNDRangeKernel( 3 ) in bolt::cl::reduce_by_key_enqueue()" << std::endl;
        std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
        std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
        std::cerr << "Error String: " << e.what() << std::endl;
    }
    // wait for results
    l_Error = kernel3Event.wait( );
    V_OPENCL( l_Error, "post-kernel[3] failed wait" );

    ::cl::Event l_mapEvent;
    int *h_result = (int*)ctl.getCommandQueue().enqueueMapBuffer( tempArrayVec,
                                                                    false,
                                                                    CL_MAP_READ | CL_MAP_WRITE,
                                                                    (numElements-1)*sizeof(int),
                                                                    sizeof(int)*1,
                                                                    NULL,
                                                                    &l_mapEvent,
                                                                    &l_Error );
    V_OPENCL( l_Error, "Error calling map on the result buffer" );

    bolt::cl::wait(ctl, l_mapEvent);

    unsigned int count_number_of_sections = *(h_result);
	



#if ENABLE_PRINTS
    //delete this code -start
    ::cl::Event l_mapEvent3;
    voType *v_result2 = (voType*)ctl.commandQueue().enqueueMapBuffer( *offsetValArray,
                                                                    false,
                                                                    CL_MAP_READ,
                                                                    0,
                                                                    sizeof(voType)*numElements,
                                                                    NULL,
                                                                    &l_mapEvent3,
                                                                    &l_Error );
    V_OPENCL( l_Error, "Error calling map on the result buffer" );
    std::cout<<"Myval-------------------------starts"<<std::endl;
    std::ofstream result_val_after_launch("result_val_after_launch.txt");
    for(unsigned int i = 0; i < LENGTH_TEST ; i++)
    {
        result_val_after_launch<<v_result2[i]<<std::endl;
    }
    result_val_after_launch.close();
    std::cout<<"Myval-------------------------ends"<<std::endl;
    bolt::cl::wait(ctl, l_mapEvent3);
    //delete this code -end

#endif
    return count_number_of_sections;
}   //end of reduce_by_key



template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction >
 typename std::enable_if< (std::is_same< typename std::iterator_traits< OutputIterator1 >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value &&
						  std::is_same< typename std::iterator_traits< OutputIterator2 >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value), unsigned int
                           >::type
reduce_by_key(
    control& ctl,
    const InputIterator1& keys_first,
    const InputIterator1& keys_last,
    const InputIterator2& values_first,
    const OutputIterator1& keys_output,
    const OutputIterator2& values_output,
    const BinaryPredicate& binary_pred,
    const BinaryFunction& binary_op,
    const std::string& user_code)
{

	int sz = static_cast<int>(keys_last - keys_first);
    if (sz == 1)
        return 1;

	typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< OutputIterator1 >::value_type koType;
    typedef typename std::iterator_traits< OutputIterator2 >::value_type voType;
    
    typedef typename std::iterator_traits<InputIterator1>::pointer key_pointer;
	typedef typename std::iterator_traits<InputIterator2>::pointer val_pointer;
	typedef typename std::iterator_traits<OutputIterator1>::pointer key_out_pointer;
	typedef typename std::iterator_traits<OutputIterator2>::pointer val_out_pointer;
    
    key_pointer keyfirst_pointer = bolt::cl::addressof(keys_first) ;
	val_pointer valfirst_pointer = bolt::cl::addressof(values_first) ;
	key_out_pointer keyout_pointer = bolt::cl::addressof(keys_output) ;
	val_out_pointer valout_pointer = bolt::cl::addressof(values_output) ;

    device_vector< kType > dvKeysInput( keyfirst_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
	device_vector< vType > dvValInput( valfirst_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
	device_vector< koType > dvKeysOutput( keyout_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, true, ctl );
	device_vector< voType > dvvalOutput( valout_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, true, ctl );
    
    auto device_iterator_keyfirst  = bolt::cl::create_device_itr(
                                        typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                        keys_first, dvKeysInput.begin());
    auto device_iterator_keylast   = bolt::cl::create_device_itr(
                                        typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                        keys_last, dvKeysInput.end());
    auto device_iterator_valfirst  = bolt::cl::create_device_itr(
                                        typename bolt::cl::iterator_traits< InputIterator2 >::iterator_category( ), 
                                        values_first, dvValInput.begin());
	auto device_iterator_keyout  = bolt::cl::create_device_itr(
                                        typename bolt::cl::iterator_traits< OutputIterator1 >::iterator_category( ), 
                                        keys_output, dvKeysOutput.begin());
	auto device_iterator_valout  = bolt::cl::create_device_itr(
                                        typename bolt::cl::iterator_traits< OutputIterator2 >::iterator_category( ), 
                                        values_output, dvvalOutput.begin());

    return cl::reduce_by_key(ctl, device_iterator_keyfirst,device_iterator_keylast, device_iterator_valfirst,
		device_iterator_keyout, device_iterator_valout,  binary_pred, binary_op, user_code);

}


}//end of cl namespace

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction
    >
typename std::enable_if< 
               !(std::is_same< typename std::iterator_traits< InputIterator1>::iterator_category, 
                             std::input_iterator_tag 
                           >::value),
               bolt::cl::pair<OutputIterator1, OutputIterator2>
                           >::type
reduce_by_key(
    control &ctl,
    InputIterator1  keys_first,
    InputIterator1  keys_last,
    InputIterator2  values_first,
    OutputIterator1  keys_output,
    OutputIterator2  values_output,
    BinaryPredicate binary_pred,
    BinaryFunction binary_op,
    const std::string& user_code)
{

    typename std::iterator_traits<InputIterator1>::difference_type numElements = bolt::cl::distance(keys_first, keys_last);

    if( (numElements == 1) || (numElements == 0) )
        return bolt::cl::make_pair( keys_output+(int)numElements, values_output+(int)numElements );// keys_last, values_first+numElements );

    bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
    if(runMode == bolt::cl::control::Automatic) {
        runMode = ctl.getDefaultPathToRun();
    }
	#if defined(BOLT_DEBUG_LOG)
    BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
    #endif
				
    if (runMode == bolt::cl::control::SerialCpu) {
            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_REDUCEBYKEY,BOLTLOG::BOLT_SERIAL_CPU,"::Reduce_By_Key::SERIAL_CPU");
            #endif
            int sizeOfOut = serial::reduce_by_key(ctl, keys_first, keys_last, values_first,keys_output, values_output, binary_pred, binary_op);
			return bolt::cl::make_pair(keys_output+sizeOfOut, values_output+sizeOfOut);
		
    } 
	else if (runMode == bolt::cl::control::MultiCoreCpu) {

        #ifdef ENABLE_TBB
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_REDUCEBYKEY,BOLTLOG::BOLT_MULTICORE_CPU,"::Reduce_By_Key::MULTICORE_CPU");
            #endif
            unsigned int sizeOfOut = btbb::reduce_by_key(ctl, keys_first, keys_last, values_first,keys_output, values_output, binary_pred, binary_op);
			return bolt::cl::make_pair(keys_output+sizeOfOut, values_output+sizeOfOut);
        #else
            throw std::runtime_error("MultiCoreCPU Version of ReduceByKey not Enabled! \n");
        #endif
    }
    else {

        #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_REDUCEBYKEY,BOLTLOG::BOLT_OPENCL_GPU,"::Reduce_By_Key::OPENCL_GPU");
        #endif
	    
	    unsigned int sizeOfOut = cl::reduce_by_key(ctl, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op, user_code);
	    return bolt::cl::make_pair(keys_output+sizeOfOut, values_output+sizeOfOut);

	}
 
}



template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction>
typename std::enable_if< 
               (std::is_same< typename std::iterator_traits< InputIterator1>::iterator_category, 
                             std::input_iterator_tag 
                           >::value),
               bolt::cl::pair<OutputIterator1, OutputIterator2>
                           >::type
reduce_by_key(
    control &ctl,
    InputIterator1  keys_first,
    InputIterator1  keys_last,
    InputIterator2  values_first,
    OutputIterator1  keys_output,
    OutputIterator2  values_output,
    BinaryPredicate binary_pred,
    BinaryFunction binary_op,
    const std::string& user_code)
{
    //  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
    //  to a temporary buffer.  Should we?
    static_assert( std::is_same< InputIterator1, std::input_iterator_tag >::value , "Bolt only supports random access iterator types" );
};


    /*!   \}  */
} //namespace detail



/**********************************************************************************************************************
 * REDUCE BY KEY
 *********************************************************************************************************************/
template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction>
bolt::cl::pair<OutputIterator1, OutputIterator2>
reduce_by_key(
    InputIterator1  keys_first,
    InputIterator1  keys_last,
    InputIterator2  values_first,
    OutputIterator1  keys_output,
    OutputIterator2  values_output,
    BinaryPredicate binary_pred,
    BinaryFunction binary_op,
    const std::string& user_code )
{
    control& ctl = control::getDefault();
    return detail::reduce_by_key(
        ctl,
        keys_first,
        keys_last,
        values_first,
        keys_output,
        values_output,
        binary_pred,
        binary_op,
        user_code
    ); 
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate>
bolt::cl::pair<OutputIterator1, OutputIterator2>
reduce_by_key(
    InputIterator1  keys_first,
    InputIterator1  keys_last,
    InputIterator2  values_first,
    OutputIterator1  keys_output,
    OutputIterator2  values_output,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator2>::value_type ValOType;
    control& ctl = control::getDefault();
    plus<ValOType> binary_op;
    return reduce_by_key(
        ctl,
        keys_first,
        keys_last,
        values_first,
        keys_output,
        values_output,
        binary_pred,
        binary_op,
        user_code
    ); 
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2>
bolt::cl::pair<OutputIterator1, OutputIterator2>
reduce_by_key(
    InputIterator1  keys_first,
    InputIterator1  keys_last,
    InputIterator2  values_first,
    OutputIterator1  keys_output,
    OutputIterator2  values_output,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator2>::value_type ValOType;
    control& ctl = control::getDefault();
    equal_to <kType> binary_pred;
    return reduce_by_key(
        ctl,
        keys_first,
        keys_last,
        values_first,
        keys_output,
        values_output,
        binary_pred,
        user_code
    ); 
}


///////////////////////////// CTRL ////////////////////////////////////////////

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate,
    typename BinaryFunction>
bolt::cl::pair<OutputIterator1, OutputIterator2>
reduce_by_key(
    control &ctl,
    InputIterator1  keys_first,
    InputIterator1  keys_last,
    InputIterator2  values_first,
    OutputIterator1  keys_output,
    OutputIterator2  values_output,
    BinaryPredicate binary_pred,
    BinaryFunction binary_op,
    const std::string& user_code )
{
    return detail::reduce_by_key(
        ctl,
        keys_first,
        keys_last,
        values_first,
        keys_output,
        values_output,
        binary_pred,
        binary_op,
        user_code
    ); 
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryPredicate>
bolt::cl::pair<OutputIterator1, OutputIterator2>
reduce_by_key(
    control &ctl,
    InputIterator1  keys_first,
    InputIterator1  keys_last,
    InputIterator2  values_first,
    OutputIterator1  keys_output,
    OutputIterator2  values_output,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator2>::value_type ValOType;
    plus<ValOType> binary_op;
    return reduce_by_key(
        ctl,
        keys_first,
        keys_last,
        values_first,
        keys_output,
        values_output,
        binary_pred,
        binary_op,
        user_code
    ); 
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2>
bolt::cl::pair<OutputIterator1, OutputIterator2>
reduce_by_key(
    control &ctl,
    InputIterator1  keys_first,
    InputIterator1  keys_last,
    InputIterator2  values_first,
    OutputIterator1  keys_output,
    OutputIterator2  values_output,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator2>::value_type ValOType;

    equal_to <kType> binary_pred;
    return reduce_by_key(
        ctl,
        keys_first,
        keys_last,
        values_first,
        keys_output,
        values_output,
        binary_pred,
        user_code
    ); 
}



} //namespace cl
} //namespace bolt

#endif
