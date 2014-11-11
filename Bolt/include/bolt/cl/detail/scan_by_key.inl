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
#if !defined( BOLT_CL_SCAN_BY_KEY_INL )
#define BOLT_CL_SCAN_BY_KEY_INL

#define KERNEL02WAVES 4
#define KERNEL1WAVES 4
#define WAVESIZE 64

#include <type_traits>
#include <bolt/cl/scan_by_key.h>
#include "bolt/cl/bolt.h"
#include "bolt/cl/functional.h"
#include "bolt/cl/device_vector.h"
#include "bolt/cl/distance.h"
#include "bolt/cl/iterator/iterator_traits.h"
#include "bolt/cl/iterator/transform_iterator.h"
#include "bolt/cl/iterator/addressof.h"


#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/scan_by_key.h"

#endif
#ifdef BOLT_ENABLE_PROFILING
#include "bolt/AsyncProfiler.h"
//AsyncProfiler aProfiler("scan_be_key");
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
*   \ingroup scan
*   \{
*/


	namespace serial{
		template<
			typename InputIterator1,
			typename InputIterator2,
			typename OutputIterator,
			typename T,
			typename BinaryPredicate,
			typename BinaryFunction >
			typename std::enable_if< (std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value), void
			>::type

			scan_by_key(
			::bolt::cl::control &ctl, 
			const InputIterator1& first1,
			const InputIterator1& last1, 
			const InputIterator2& first2,
			const OutputIterator& result,
			const T& init,
			const BinaryPredicate& binary_pred,
			const BinaryFunction& binary_op,
			const bool& inclusive)
			{
	
				size_t sz = (last1 - first1);
				if (sz == 0)
					return; 
				
				typedef typename std::iterator_traits<InputIterator1>::value_type kType;
				typedef typename std::iterator_traits<InputIterator2>::value_type iType;
				typedef typename std::iterator_traits<OutputIterator>::value_type oType;
				/*Get The associated OpenCL buffer for each of the iterators*/
				::cl::Buffer first1Buffer  = first1.base().getContainer( ).getBuffer( );
				::cl::Buffer first2Buffer  = first2.base().getContainer( ).getBuffer( );
				::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
				/*Get The size of each OpenCL buffer*/
				size_t first1_sz  = first1Buffer.getInfo<CL_MEM_SIZE>();
				size_t first2_sz  = first2Buffer.getInfo<CL_MEM_SIZE>();
				size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();

				cl_int map_err;
				kType *first1Ptr = (kType*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
																					first1_sz, NULL, NULL, &map_err);
				iType *first2Ptr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
																					first2_sz, NULL, NULL, &map_err);
				oType *resultPtr = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
																					result_sz, NULL, NULL, &map_err);
				auto mapped_fst1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator1>::iterator_category(), 
																ctl, first1, first1Ptr);
				auto mapped_fst2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
																ctl, first2, first2Ptr);
				auto mapped_res_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
																ctl, result, resultPtr);

				if(inclusive)
				{					
					// do zeroeth element
					*mapped_res_itr = static_cast<iType>(*mapped_fst2_itr); // assign value
					// scan oneth element and beyond
					for ( unsigned int i=1; i< sz;  i++)
					{
						// load keys
						kType currentKey  = static_cast<kType>( *(mapped_fst1_itr + i));
						kType previousKey = static_cast<kType>(*(mapped_fst1_itr + i -1 ));
						// load value
						oType currentValue = static_cast<iType>( *(mapped_fst2_itr +i )); // convertible
						oType previousValue = static_cast<oType>( *(mapped_res_itr + i -1 ));

						// within segment
						if (binary_pred(currentKey, previousKey))
						{
							oType r = binary_op( previousValue, currentValue);
							*(mapped_res_itr + i) = r;
						}
						else // new segment
						{
							*(mapped_res_itr + i) = currentValue;
						}
					}
				}
				else
				{
					oType temp = static_cast<iType>(*mapped_fst2_itr);
					*mapped_res_itr = static_cast<oType>( init );
					// scan oneth element and beyond
					for ( unsigned int i= 1; i<sz; i++)
					{
						// load keys
						kType currentKey  = static_cast<kType>( *(mapped_fst1_itr + i));
						kType previousKey = static_cast<kType>(*(mapped_fst1_itr + i -1));

						// load value
						oType currentValue = temp; // convertible
						oType previousValue = static_cast<oType>( *(mapped_res_itr + i -1 ));

						// within segment
						if (binary_pred(currentKey,previousKey))
						{
							temp = static_cast<iType>(*(mapped_fst2_itr + i));
							oType r = binary_op( previousValue, currentValue);
							*(mapped_res_itr + i) = r;
						}
						else // new segment
						{
							 temp = static_cast<iType>(*(mapped_fst2_itr + i));
							*(mapped_res_itr + i) =  static_cast<oType>(init);
						}
				
				    }		
			    }
				 
				::cl::Event unmap_event[3];
				ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
				ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
				ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[2] );
				unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait(); 
				return ;			
	
	         }

			template<
			typename InputIterator1,
			typename InputIterator2,
			typename OutputIterator,
			typename T,
			typename BinaryPredicate,
			typename BinaryFunction >
			typename std::enable_if< (std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value), void
            >::type

			scan_by_key(
			::bolt::cl::control &ctl, 
			const InputIterator1& first1,
			const InputIterator1& last1, 
			const InputIterator2& first2,
			const OutputIterator& result,
			const T& init,
			const BinaryPredicate& binary_pred,
			const BinaryFunction& binary_op,
			const bool& inclusive)
			{			
	
				size_t sz = (last1 - first1);
				if (sz == 0)
					return; 
				
				typedef typename std::iterator_traits<InputIterator1>::value_type kType;
				typedef typename std::iterator_traits<InputIterator2>::value_type iType;
				typedef typename std::iterator_traits<OutputIterator>::value_type oType;

				if(inclusive)
				{
					// do zeroeth element
					*result = static_cast<iType>(*first2); // assign value
					// scan oneth element and beyond
					for ( unsigned int i=1; i< sz;  i++)
					{
						// load keys
						kType currentKey  = static_cast<kType>(*(first1 + i));
						kType previousKey = static_cast<kType>(*(first1 + i - 1));
						// load value
						oType currentValue = static_cast<iType>(*(first2 + i)); // convertible
						oType previousValue = static_cast<oType>(*(result + i - 1));

						// within segment
						if (binary_pred(currentKey, previousKey))
						{
							oType r = binary_op( previousValue, currentValue);
							*(result+i) = r;
						}
						else // new segment
						{
							*(result + i) = currentValue;
						}
					}
				
				}
				else
				{
					// do zeroeth element
					//*result = *values; // assign value
					oType temp = static_cast<iType>(*first2);
					*result = static_cast<oType>(init);
					// scan oneth element and beyond
					for ( unsigned int i= 1; i<sz; i++)
					{
						// load keys
						kType currentKey  = *(first1 + i);
						kType previousKey = *(first1+ i -1 );

						// load value
						oType currentValue = temp; // convertible
						oType previousValue = static_cast<oType>(*(result + i -1 ));

						// within segment
						if (binary_pred(currentKey,previousKey))
						{
							temp = static_cast<iType>(*(first2 + i));
							oType r = binary_op( previousValue, currentValue);
							*(result + i) = r;
						}
						else // new segment
						{
							 temp = *(first2 + i);
							 *(result + i) = static_cast<oType>(init);
						}
					}
				
				}
				return;
			}

		}
#ifdef ENABLE_TBB
	namespace btbb{
		template<
				typename InputIterator1,
				typename InputIterator2,
				typename OutputIterator,
				typename T,
				typename BinaryPredicate,
				typename BinaryFunction >
				typename std::enable_if< (std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
										   bolt::cl::device_vector_tag
										 >::value), void
				>::type

				scan_by_key(
				::bolt::cl::control &ctl, 
				const InputIterator1& first1,
				const InputIterator1& last1, 
				const InputIterator2& first2,
				const OutputIterator& result,
				const T& init,
				const BinaryPredicate& binary_pred,
				const BinaryFunction& binary_op,
				const bool& inclusive)
				{	
					size_t sz = (last1 - first1);
					if (sz == 0)
						return; 
				
				typedef typename std::iterator_traits<InputIterator1>::value_type kType;
				typedef typename std::iterator_traits<InputIterator2>::value_type iType;
				typedef typename std::iterator_traits<OutputIterator>::value_type oType;
				/*Get The associated OpenCL buffer for each of the iterators*/
				::cl::Buffer first1Buffer  = first1.base().getContainer( ).getBuffer( );
				::cl::Buffer first2Buffer  = first2.base().getContainer( ).getBuffer( );
				::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
				/*Get The size of each OpenCL buffer*/
				size_t first1_sz  = first1Buffer.getInfo<CL_MEM_SIZE>();
				size_t first2_sz  = first2Buffer.getInfo<CL_MEM_SIZE>();
				size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();

				cl_int map_err;
				kType *first1Ptr = (kType*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
																					first1_sz, NULL, NULL, &map_err);
				iType *first2Ptr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
																					first2_sz, NULL, NULL, &map_err);
				oType *resultPtr = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
																					result_sz, NULL, NULL, &map_err);
				auto mapped_fst1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator1>::iterator_category(), 
																ctl, first1, first1Ptr);
				auto mapped_fst2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
																ctl, first2, first2Ptr);
				auto mapped_res_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(),
																ctl, result, resultPtr);
				if(inclusive)
					bolt::btbb::inclusive_scan_by_key(mapped_fst1_itr, mapped_fst1_itr + (int)sz, mapped_fst2_itr, mapped_res_itr, binary_pred, binary_op );
				else
					bolt::btbb::exclusive_scan_by_key(mapped_fst1_itr, mapped_fst1_itr + (int)sz, mapped_fst2_itr, mapped_res_itr, init, binary_pred, binary_op );
				
				::cl::Event unmap_event[3];
				ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
				ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
				ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[2] );
				unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait(); 
				return ;		
		
				}

		template<
				typename InputIterator1,
				typename InputIterator2,
				typename OutputIterator,
				typename T,
				typename BinaryPredicate,
				typename BinaryFunction >
			typename std::enable_if< (std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value), void
           >::type
		   scan_by_key(
				::bolt::cl::control &ctl, 
				const InputIterator1& first1,
				const InputIterator1& last1, 
				const InputIterator2& first2,
				const OutputIterator& result,
				const T& init,
				const BinaryPredicate& binary_pred,
				const BinaryFunction& binary_op,
				const bool& inclusive)
				{
						
					size_t sz = (last1 - first1);
					if (sz == 0)
						return; 
					if(inclusive)
						bolt::btbb::inclusive_scan_by_key(first1, last1, first2, result, binary_pred, binary_op );
					else
						bolt::btbb::exclusive_scan_by_key(first1, last1, first2, result,  init, binary_pred, binary_op );
					return;
				}
	}
#endif

	namespace cl{
		enum scanByKeyTypes  {scanByKey_kType, scanByKey_kIterType, scanByKey_vType, scanByKey_iIterType, scanByKey_oType, scanByKey_oIterType,
                scanByKey_initType, scanByKey_BinaryPredicate, scanByKey_BinaryFunction, scanbykey_end};
		
		/*********************************************************************************************************************
		 * Kernel Template Specializer
		 ********************************************************************************************************************/
		class ScanByKey_KernelTemplateSpecializer : public KernelTemplateSpecializer
		{
			public:

			ScanByKey_KernelTemplateSpecializer() : KernelTemplateSpecializer()
			{
				addKernelName("perBlockScanByKey");
				addKernelName("intraBlockInclusiveScanByKey");
				addKernelName("perBlockAdditionByKey");
			}

			const ::std::string operator() ( const ::std::vector< ::std::string >& typeNames ) const
			{
				const std::string templateSpecializationString =
					"// Dynamic specialization of generic template definition, using user supplied types\n"
					"template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
					"__attribute__((reqd_work_group_size(KERNEL0WORKGROUPSIZE,1,1)))\n"
					"__kernel void " + name(0) + "(\n"
					"global " + typeNames[scanByKey_kType] + "* keys,\n"
					""        + typeNames[scanByKey_kIterType] + " keys_iter,\n"
					"global " + typeNames[scanByKey_vType] + "* vals,\n"
					""        + typeNames[scanByKey_iIterType] + " vals_iter,\n"
					""        + typeNames[scanByKey_initType] + " init,\n"
					"const uint vecSize,\n"
					"local "  + typeNames[scanByKey_kType] + "* ldsKeys,\n"
					"local "  + typeNames[scanByKey_oType] + "* ldsVals,\n"
					"global " + typeNames[scanByKey_BinaryPredicate] + "* binaryPred,\n"
					"global " + typeNames[scanByKey_BinaryFunction]  + "* binaryFunct,\n"
					"global " + typeNames[scanByKey_kType] + "* keyBuffer,\n"
					"global " + typeNames[scanByKey_oType] + "* valBuffer,\n"
					"global " + typeNames[scanByKey_oType] + "* valBuffer1,\n"
					"int exclusive\n"
					");\n\n"


					"// Dynamic specialization of generic template definition, using user supplied types\n"
					"template __attribute__((mangled_name(" + name(1) + "Instantiated)))\n"
					"__attribute__((reqd_work_group_size(KERNEL1WORKGROUPSIZE,1,1)))\n"
					"__kernel void " + name(1) + "(\n"
					"global " + typeNames[scanByKey_kType] + "* keySumArray,\n"
					"global " + typeNames[scanByKey_oType] + "* preSumArray,\n"
					"const uint vecSize,\n"
					"local "  + typeNames[scanByKey_kType] + "* ldsKeys,\n"
					"local "  + typeNames[scanByKey_oType] + "* ldsVals,\n"
					"const uint workPerThread,\n"
					"global " + typeNames[scanByKey_BinaryPredicate] + "* binaryPred,\n"
					"global " + typeNames[scanByKey_BinaryFunction] + "* binaryFunct\n"
					");\n\n"


					"// Dynamic specialization of generic template definition, using user supplied types\n"
					"template __attribute__((mangled_name(" + name(2) + "Instantiated)))\n"
					"__attribute__((reqd_work_group_size(KERNEL2WORKGROUPSIZE,1,1)))\n"
					"__kernel void " + name(2) + "(\n"
					"global " + typeNames[scanByKey_oType] + "* preSumArray,\n"
					"global " + typeNames[scanByKey_oType] + "* preSumArray1,\n"
					"global " + typeNames[scanByKey_kType] + "* keys,\n"
					""        + typeNames[scanByKey_kIterType] + " keys_iter,\n"
					"global " + typeNames[scanByKey_vType] + "* vals,\n"
					""        + typeNames[scanByKey_iIterType] + " vals_iter,\n"
					"global " + typeNames[scanByKey_oType] + "* output,\n"
					""        + typeNames[scanByKey_oIterType] + " output_iter,\n"
					"local "  + typeNames[scanByKey_kType] + "* ldsKeys,\n"
					"local "  + typeNames[scanByKey_oType] + "* ldsVals,\n"
					"const uint vecSize,\n"
					"global " + typeNames[scanByKey_BinaryPredicate] + "* binaryPred,\n"
					"global " + typeNames[scanByKey_BinaryFunction] + "* binaryFunct,\n"
					"int exclusive,\n"
					""        + typeNames[scanByKey_initType] + " identity\n"
					");\n\n";

				return templateSpecializationString;
			}
		};

		//  All calls to scan_by_key end up here, unless an exception was thrown
		//  This is the function that sets up the kernels to compile (once only) and execute
		
		template<
		typename InputIterator1,
		typename InputIterator2,
		typename OutputIterator,
		typename T,
		typename BinaryPredicate,
		typename BinaryFunction >
		typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
									bolt::cl::device_vector_tag
									>::value
					>::type
		scan_by_key(
		control& ctl,
		const InputIterator1& firstKey,
		const InputIterator1& lastKey,
		const InputIterator2& firstValue,
		const OutputIterator& result,
		const T& init,
		const BinaryPredicate& binary_pred,
		const BinaryFunction& binary_funct,
		const bool& inclusive, 
		const std::string& user_code)
		{
			cl_int l_Error;
			#ifdef BOLT_ENABLE_PROFILING
			aProfiler.setName("scan_by_key");
			aProfiler.startTrial();
			aProfiler.setStepName("Setup");
			aProfiler.set(AsyncProfiler::device, control::SerialCpu);

			size_t k0_stepNum, k1_stepNum, k2_stepNum;
			#endif

				/**********************************************************************************
				 * Type Names - used in KernelTemplateSpecializer
				 *********************************************************************************/
				typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
				typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
				typedef typename std::iterator_traits< OutputIterator >::value_type oType;
				std::vector<std::string> typeNames(scanbykey_end);

				typeNames[scanByKey_kType] = TypeName< kType >::get( );
				typeNames[scanByKey_kIterType] = TypeName< InputIterator1 >::get( );
				typeNames[scanByKey_vType] = TypeName< vType >::get( );
				typeNames[scanByKey_iIterType] = TypeName< InputIterator2 >::get( );
				typeNames[scanByKey_oType] = TypeName< oType >::get( );
				typeNames[scanByKey_oIterType] = TypeName< OutputIterator >::get( );
				typeNames[scanByKey_initType] = TypeName< T >::get( );
				typeNames[scanByKey_BinaryPredicate] = TypeName< BinaryPredicate >::get( );
				typeNames[scanByKey_BinaryFunction]  = TypeName< BinaryFunction >::get( );

				/**********************************************************************************
				 * Type Definitions - directly concatenated into kernel string
				 *********************************************************************************/
				std::vector<std::string> typeDefs; // typeDefs must be unique and order does matter
				PUSH_BACK_UNIQUE( typeDefs, ClCode< kType >::get() )
				PUSH_BACK_UNIQUE( typeDefs, ClCode< InputIterator1 >::get() )
				PUSH_BACK_UNIQUE( typeDefs, ClCode< vType >::get() )
				PUSH_BACK_UNIQUE( typeDefs, ClCode< InputIterator2 >::get() )
				PUSH_BACK_UNIQUE( typeDefs, ClCode< oType >::get() )
				PUSH_BACK_UNIQUE( typeDefs, ClCode< OutputIterator >::get() )
				PUSH_BACK_UNIQUE( typeDefs, ClCode< T >::get() )
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
				ScanByKey_KernelTemplateSpecializer ts_kts;
				std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
					ctl,
					typeNames,
					&ts_kts,
					typeDefs,
					scan_by_key_kernels,
					compileOptions);
				// kernels returned in same order as added in KernelTemplaceSpecializer constructor

				// for profiling
				::cl::Event kernel0Event, kernel1Event, kernel2Event, kernelAEvent;
				cl_uint doExclusiveScan = inclusive ? 0 : 1;
				// Set up shape of launch grid and buffers:
				int computeUnits     = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
				int wgPerComputeUnit =  ctl.getWGPerComputeUnit( );
				int resultCnt = computeUnits * wgPerComputeUnit;

				//  Ceiling function to bump the size of input to the next whole wavefront size
				cl_uint numElements = static_cast< cl_uint >( std::distance( firstKey, lastKey ) );
				typename device_vector< kType >::size_type sizeInputBuff = numElements;

				int modWgSize = (sizeInputBuff & ((kernel0_WgSize*2)-1));
				if( modWgSize )
				{
					sizeInputBuff &= ~modWgSize;
					sizeInputBuff += (kernel0_WgSize*2);
				}
				cl_uint numWorkGroupsK0 = static_cast< cl_uint >( sizeInputBuff / (kernel0_WgSize*2) );

				//  Ceiling function to bump the size of the sum array to the next whole wavefront size
				typename device_vector< kType >::size_type sizeScanBuff = numWorkGroupsK0;
				modWgSize = (sizeScanBuff & ((kernel0_WgSize*2)-1));
				if( modWgSize )
				{
					sizeScanBuff &= ~modWgSize;
					sizeScanBuff += (kernel0_WgSize*2);
				}

				// Create buffer wrappers so we can access the host functors, for read or writing in the kernel

				ALIGNED( 256 ) BinaryPredicate aligned_binary_pred( binary_pred );
				control::buffPointer binaryPredicateBuffer = ctl.acquireBuffer( sizeof( aligned_binary_pred ),
					CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_binary_pred );
				 ALIGNED( 256 ) BinaryFunction aligned_binary_funct( binary_funct );
				control::buffPointer binaryFunctionBuffer = ctl.acquireBuffer( sizeof( aligned_binary_funct ),
					CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_binary_funct );

				control::buffPointer keySumArray  = ctl.acquireBuffer( sizeScanBuff*sizeof( kType ) );
				control::buffPointer preSumArray  = ctl.acquireBuffer( sizeScanBuff*sizeof( vType ) );
				control::buffPointer preSumArray1  = ctl.acquireBuffer( sizeScanBuff*sizeof( vType ) );
				cl_uint ldsKeySize, ldsValueSize;


				/**********************************************************************************
				 *  Kernel 0
				 *********************************************************************************/
			#ifdef BOLT_ENABLE_PROFILING
			aProfiler.nextStep();
			aProfiler.setStepName("Setup Kernel 0");
			aProfiler.set(AsyncProfiler::getDevice, control::SerialCpu);
			#endif
				typename InputIterator1::Payload firstKey_payload = firstKey.gpuPayload( );
				typename InputIterator2::Payload firstValue_payload = firstValue.gpuPayload( );
				try
				{
				ldsKeySize   = static_cast< cl_uint >( (kernel0_WgSize*2) * sizeof( kType ) );
				ldsValueSize = static_cast< cl_uint >( (kernel0_WgSize*2) * sizeof( vType ) );
				V_OPENCL( kernels[0].setArg( 0, firstKey.base().getContainer().getBuffer()), "Error setArg kernels[ 0 ]" ); // Input keys
				V_OPENCL( kernels[0].setArg( 1, firstKey.gpuPayloadSize( ), &firstKey_payload ), "Error setting a kernel argument" );
				V_OPENCL( kernels[0].setArg( 2, firstValue.base().getContainer().getBuffer()),"Error setArg kernels[ 0 ]" ); // Input buffer
				V_OPENCL( kernels[0].setArg( 3, firstValue.gpuPayloadSize( ), &firstValue_payload ), "Error setting a kernel argument" );
				V_OPENCL( kernels[0].setArg( 4, init ),                 "Error setArg kernels[ 0 ]" ); // Initial value exclusive
				V_OPENCL( kernels[0].setArg( 5, numElements ),          "Error setArg kernels[ 0 ]" ); // Size of scratch buffer
				V_OPENCL( kernels[0].setArg( 6, ldsKeySize, NULL ),     "Error setArg kernels[ 0 ]" ); // Scratch buffer
				V_OPENCL( kernels[0].setArg( 7, ldsValueSize, NULL ),   "Error setArg kernels[ 0 ]" ); // Scratch buffer
				V_OPENCL( kernels[0].setArg( 8, *binaryPredicateBuffer),"Error setArg kernels[ 0 ]" ); // User provided functor
				V_OPENCL( kernels[0].setArg( 9, *binaryFunctionBuffer ),"Error setArg kernels[ 0 ]" ); // User provided functor
				V_OPENCL( kernels[0].setArg(10, *keySumArray ),         "Error setArg kernels[ 0 ]" ); // Output per block sum
				V_OPENCL( kernels[0].setArg(11, *preSumArray ),         "Error setArg kernels[ 0 ]" ); // Output per block sum
				V_OPENCL( kernels[0].setArg(12, *preSumArray1 ),         "Error setArg kernels[ 0 ]" ); // Output per block sum
				V_OPENCL( kernels[0].setArg(13, doExclusiveScan ),      "Error setArg kernels[ 0 ]" ); // Exclusive scan?

			#ifdef BOLT_ENABLE_PROFILING
			aProfiler.nextStep();
			k0_stepNum = aProfiler.getStepNum();
			aProfiler.setStepName("Kernel 0");
			aProfiler.set(AsyncProfiler::getDevice, ctl.forceRunMode());
			aProfiler.set(AsyncProfiler::flops, 2*numElements);
			aProfiler.set(AsyncProfiler::memory,2*numElements*(sizeof(vType)+sizeof(kType)) +
											  1*sizeScanBuff*(sizeof(vType)+sizeof(kType)) );
			#endif

				l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
					kernels[0],
					::cl::NullRange,
					::cl::NDRange( sizeInputBuff/2 ),
					::cl::NDRange( kernel0_WgSize ),
					NULL,
					&kernel0Event);
				V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel[0]" );
				}
				catch( const ::cl::Error& e)
				{
					std::cerr << "::cl::enqueueNDRangeKernel( 0 ) in bolt::cl::scan_by_key_enqueue()" << std::endl;
					std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
					std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
					std::cerr << "Error String: " << e.what() << std::endl;
				}

				/**********************************************************************************
				 *  Kernel 1
				 *********************************************************************************/
			#ifdef BOLT_ENABLE_PROFILING
			aProfiler.nextStep();
			aProfiler.setStepName("Setup Kernel 1");
			aProfiler.set(AsyncProfiler::device, control::SerialCpu);
			#endif
				ldsKeySize   = static_cast< cl_uint >( (kernel0_WgSize) * sizeof( kType ) );
				ldsValueSize = static_cast< cl_uint >( (kernel0_WgSize) * sizeof( vType ) );
				cl_uint workPerThread = static_cast< cl_uint >( sizeScanBuff / kernel1_WgSize );
				workPerThread = workPerThread ? workPerThread : 1;

				V_OPENCL( kernels[1].setArg( 0, *keySumArray ),         "Error setArg kernels[ 1 ]" ); // Input keys
				V_OPENCL( kernels[1].setArg( 1, *preSumArray ),         "Error setArg kernels[ 1 ]" ); // Input buffer
				V_OPENCL( kernels[1].setArg( 2, numWorkGroupsK0 ),      "Error setArg kernels[ 1 ]" ); // Size of scratch buffer
				V_OPENCL( kernels[1].setArg( 3, ldsKeySize, NULL ),     "Error setArg kernels[ 1 ]" ); // Scratch buffer
				V_OPENCL( kernels[1].setArg( 4, ldsValueSize, NULL ),   "Error setArg kernels[ 1 ]" ); // Scratch buffer
				V_OPENCL( kernels[1].setArg( 5, workPerThread ),        "Error setArg kernels[ 1 ]" ); // User provided functor
				V_OPENCL( kernels[1].setArg( 6, *binaryPredicateBuffer ),"Error setArg kernels[ 1 ]" ); // User provided functor
				V_OPENCL( kernels[1].setArg( 7, *binaryFunctionBuffer ),"Error setArg kernels[ 1 ]" ); // User provided functor

			#ifdef BOLT_ENABLE_PROFILING
			aProfiler.nextStep();
			k1_stepNum = aProfiler.getStepNum();
			aProfiler.setStepName("Kernel 1");
			aProfiler.set(AsyncProfiler::device, ctl.forceRunMode());
			aProfiler.set(AsyncProfiler::flops, 2*sizeScanBuff);
			aProfiler.set(AsyncProfiler::memory, 4*sizeScanBuff*(sizeof(kType)+sizeof(vType)));
			#endif

				try
				{
				l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
					kernels[1],
					::cl::NullRange,
					::cl::NDRange( kernel1_WgSize ), // only 1 work-group
					::cl::NDRange( kernel1_WgSize ),
					NULL,
					&kernel1Event);
				V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel[1]" );
				}
				catch( const ::cl::Error& e)
				{
					std::cerr << "::cl::enqueueNDRangeKernel( 1 ) in bolt::cl::scan_by_key_enqueue()" << std::endl;
					std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
					std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
					std::cerr << "Error String: " << e.what() << std::endl;
				}

				/**********************************************************************************
				 *  Kernel 2
				 *********************************************************************************/
			#ifdef BOLT_ENABLE_PROFILING
			aProfiler.nextStep();
			aProfiler.setStepName("Setup Kernel 2");
			aProfiler.set(AsyncProfiler::device, control::SerialCpu);
			#endif
				typename InputIterator1::Payload firstKey1_payload = firstKey.gpuPayload( );
				typename InputIterator2::Payload firstValue1_payload = firstValue.gpuPayload( );
				typename OutputIterator::Payload result1_payload = result.gpuPayload( );


				V_OPENCL( kernels[2].setArg( 0, *preSumArray ),        "Error setArg kernels[ 2 ]" ); // Input buffer
				V_OPENCL( kernels[2].setArg( 1, *preSumArray1 ),        "Error setArg kernels[ 2 ]" ); // Input buffer
				V_OPENCL( kernels[2].setArg( 2, firstKey.base().getContainer().getBuffer()), "Error setArg kernels[ 2 ]" ); // Input keys
				V_OPENCL( kernels[2].setArg( 3, firstKey.gpuPayloadSize( ),&firstKey1_payload ), "Error setting a kernel argument" );
				V_OPENCL( kernels[2].setArg( 4, firstValue.base().getContainer().getBuffer()),"Error setArg kernels[ 2 ]" ); // Input buffer
				V_OPENCL( kernels[2].setArg( 5, firstValue.gpuPayloadSize( ),&firstValue1_payload  ), "Error setting a kernel argument" );
				V_OPENCL( kernels[2].setArg( 6, result.getContainer().getBuffer() ), "Error setArg kernels[ 2 ]" ); // Output buffer
				V_OPENCL( kernels[2].setArg( 7, result.gpuPayloadSize( ), &result1_payload ), "Error setting a kernel argument" );
				V_OPENCL( kernels[2].setArg( 8, ldsKeySize, NULL ),     "Error setArg kernels[ 2 ]" ); // Scratch buffer
				V_OPENCL( kernels[2].setArg( 9, ldsValueSize, NULL ),   "Error setArg kernels[ 2 ]" ); // Scratch buffer
				V_OPENCL( kernels[2].setArg(10, numElements ),          "Error setArg kernels[ 2 ]" ); // Size of scratch buffer
				V_OPENCL( kernels[2].setArg(11, *binaryPredicateBuffer ),"Error setArg kernels[ 2 ]" ); // User provided functor
				V_OPENCL( kernels[2].setArg(12, *binaryFunctionBuffer ),"Error setArg kernels[ 2 ]" ); // User provided functor
				V_OPENCL( kernels[2].setArg(13, doExclusiveScan ),      "Error setArg kernels[ 2 ]" ); // Exclusive scan?
				V_OPENCL( kernels[2].setArg(14, init ),                 "Error setArg kernels[ 2 ]" ); // Initial value exclusive

			#ifdef BOLT_ENABLE_PROFILING
			aProfiler.nextStep();
			k2_stepNum = aProfiler.getStepNum();
			aProfiler.setStepName("Kernel 2");
			aProfiler.set(AsyncProfiler::device, ctl.forceRunMode());
			aProfiler.set(AsyncProfiler::flops, numElements);
			aProfiler.set(
				AsyncProfiler::memory,
				2*numElements*sizeof(vType)+numElements*sizeof(kType)+1*sizeScanBuff*(sizeof(kType)+sizeof(vType)));
			#endif

				try
				{
				l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
					kernels[2],
					::cl::NullRange,
					::cl::NDRange( sizeInputBuff ),
					::cl::NDRange( kernel2_WgSize ),
					NULL,
					&kernel2Event );
				V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel[2]" );
				}
				catch( const ::cl::Error& e)
				{
					std::cerr << "::cl::enqueueNDRangeKernel( 2 ) in bolt::cl::scan_by_key_enqueue()" << std::endl;
					std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
					std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
					std::cerr << "Error String: " << e.what() << std::endl;
				}

				// wait for results
				l_Error = kernel2Event.wait( );
				V_OPENCL( l_Error, "post-kernel[2] failed wait" );

			#ifdef BOLT_ENABLE_PROFILING
			aProfiler.nextStep();
			aProfiler.setStepName("Querying Kernel Times");
			aProfiler.set(AsyncProfiler::device, control::SerialCpu);

			aProfiler.setDataSize(numElements*sizeof(oType));
			std::string strDeviceName = ctl.getDevice().getInfo< CL_DEVICE_NAME >( &l_Error );
			bolt::cl::V_OPENCL( l_Error, "Device::getInfo< CL_DEVICE_NAME > failed" );
			aProfiler.setArchitecture(strDeviceName);

				try
				{
					cl_ulong k0_start, k0_stop, k1_stop, k2_stop;

					l_Error = kernel0Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &k0_start);
					V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()");
					l_Error = kernel0Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &k0_stop);
					V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");

					//l_Error = kernel1Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &k1_start);
					//V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_START>()");
					l_Error = kernel1Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &k1_stop);
					V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");

					//l_Error = kernel2Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &k2_start);
					//V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_START>()");
					l_Error = kernel2Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &k2_stop);
					V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");

					size_t k0_start_cpu = aProfiler.get(k0_stepNum, AsyncProfiler::startTime);
					size_t shift = k0_start - k0_start_cpu;
					//size_t shift = k0_start_cpu - k0_start;

					//std::cout << "setting step " << k0_stepNum << " attribute " << AsyncProfiler::stopTime;
					//std::cout << " to " << k0_stop-shift << std::endl;
					aProfiler.set(k0_stepNum, AsyncProfiler::stopTime,  static_cast<size_t>(k0_stop-shift) );

					aProfiler.set(k1_stepNum, AsyncProfiler::startTime, static_cast<size_t>(k0_stop-shift) );
					aProfiler.set(k1_stepNum, AsyncProfiler::stopTime,  static_cast<size_t>(k1_stop-shift) );

					aProfiler.set(k2_stepNum, AsyncProfiler::startTime, static_cast<size_t>(k1_stop-shift) );
					aProfiler.set(k2_stepNum, AsyncProfiler::stopTime,  static_cast<size_t>(k2_stop-shift) );

				}
				catch( ::cl::Error& e )
				{
					std::cout << ( "Scan Benchmark error condition reported:" ) << std::endl << e.what() << std::endl;
					return;
				}

			aProfiler.stopTrial();

			#endif

		
		}

		template<
		typename InputIterator1,
		typename InputIterator2,
		typename OutputIterator,
		typename T,
		typename BinaryPredicate,
		typename BinaryFunction >
		typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                std::random_access_iterator_tag
                                >::value
                >::type
		scan_by_key(
		control& ctl,
		const InputIterator1& first1,
		const InputIterator1& last1,
		const InputIterator2& first2,
		const OutputIterator& result,
		const T& init,
		const BinaryPredicate& binary_pred,
		const BinaryFunction& binary_funct,
		const bool& inclusive, 
		const std::string& user_code)
		{
				typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
				typedef typename std::iterator_traits< InputIterator2 >::value_type iType;
				typedef typename std::iterator_traits< OutputIterator >::value_type oType;	    
	    
				int numElements = static_cast< int >( std::distance( first1, last1 ) );
				if( numElements == 0 )
					return;
	    
				typedef typename bolt::cl::iterator_traits<InputIterator1>::pointer pointer1;	    
				typedef typename bolt::cl::iterator_traits<InputIterator2>::pointer pointer2;
				pointer1 first1_pointer = bolt::cl::addressof(first1) ;
				pointer2 first2_pointer = bolt::cl::addressof(first2) ;

				device_vector< kType > dvInput1( first1_pointer, numElements, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
				device_vector< iType > dvInput2( first2_pointer, numElements, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
				device_vector< oType > dvOutput( result, numElements, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, false, ctl );
				auto device_iterator_first1 = bolt::cl::create_device_itr(
													typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
													first1, dvInput1.begin() );
				auto device_iterator_last1  = bolt::cl::create_device_itr(
													typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
													last1, dvInput1.end() );
				auto device_iterator_first2 = bolt::cl::create_device_itr(
													typename bolt::cl::iterator_traits< InputIterator2 >::iterator_category( ), 
													first2, dvInput2.begin() );
				cl::scan_by_key(ctl, device_iterator_first1, device_iterator_last1, device_iterator_first2, 
					                            dvOutput.begin(), init, binary_pred, binary_funct, inclusive, user_code);
				dvOutput.data( );
	    
				return ; 
		}
		
	} //end of namespace cl

	template<
		typename InputIterator1,
		typename InputIterator2,
		typename OutputIterator,
		typename T,
		typename BinaryPredicate,
		typename BinaryFunction >
		
		typename std::enable_if< 
				   !(std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
								 std::input_iterator_tag 
							   >::value ||
				   std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
								 bolt::cl::fancy_iterator_tag >::value ),
				   OutputIterator
							   >::type
		scan_by_key(
		control& ctl,
		const InputIterator1& first1,
		const InputIterator1& last1,
		const InputIterator2& first2,
		const OutputIterator& result,
		const T& init,
		const BinaryPredicate& binary_pred,
		const BinaryFunction& binary_funct,
		const bool& inclusive, 
		const std::string& user_code)
		{
			
			typedef typename std::iterator_traits< InputIterator1 >::value_type kType;			
			typedef typename std::iterator_traits< InputIterator2 >::value_type iType;
			typedef typename std::iterator_traits< OutputIterator >::value_type oType;

			unsigned int numElements = static_cast< unsigned int >( std::distance( first1, last1 ) );
			if( numElements == 0 )
				return result;

			bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode( );

			if( runMode == bolt::cl::control::Automatic )
			{
				runMode = ctl.getDefaultPathToRun();
			}
			#if defined(BOLT_DEBUG_LOG)
			BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
			#endif
	    
			if( runMode == bolt::cl::control::SerialCpu )
			{
				#if defined(BOLT_DEBUG_LOG)
					dblog->CodePathTaken(BOLTLOG::BOLT_SCAN_BY_KEY,BOLTLOG::BOLT_SERIAL_CPU,"::Scan_by_key::SERIAL_CPU");
				#endif
	    		serial::scan_by_key(ctl, first1, last1, first2, result, init, binary_pred, binary_funct, inclusive );
				return result + numElements;
			}
			else if( runMode == bolt::cl::control::MultiCoreCpu )
			{
				#ifdef ENABLE_TBB
	    			#if defined(BOLT_DEBUG_LOG)
						dblog->CodePathTaken(BOLTLOG::BOLT_SCAN_BY_KEY,BOLTLOG::BOLT_MULTICORE_CPU,
	    					"::Scan_by_key::MULTICORE_CPU");
					#endif
	    			btbb::scan_by_key(ctl, first1, last1, first2, result, init, binary_pred, binary_funct, inclusive );
				#else
						throw std::runtime_error("The MultiCoreCpu version of scan_by_key is not enabled to be built! \n");
				#endif
	    
				return result + numElements;
	    
			}
			else
			{
				#if defined(BOLT_DEBUG_LOG)
					dblog->CodePathTaken(BOLTLOG::BOLT_SCAN_BY_KEY,BOLTLOG::BOLT_OPENCL_GPU,"::Scan_by_key::OPENCL_GPU");
				#endif
	    	
	    		cl::scan_by_key(ctl, first1, last1, first2, result, init, binary_pred, binary_funct, inclusive, user_code );
			}
				return result + numElements;
	
		}
    
	template<
		typename InputIterator1,
		typename InputIterator2,
		typename OutputIterator,
		typename T,
		typename BinaryPredicate,
		typename BinaryFunction >
		
		typename std::enable_if< 
               (std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value ),
               OutputIterator
                           >::type
		scan_by_key(
		control& ctl,
		const InputIterator1& first1,
		const InputIterator1& last1,
		const InputIterator2& first2,
		const OutputIterator& result,
		const T& init,
		const BinaryPredicate& binary_pred,
		const BinaryFunction& binary_funct,
		const bool& inclusive, 
		const std::string& user_code)
	    {	
			//  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
			//  to a temporary buffer.  Should we?
			static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
											std::input_iterator_tag >::value , 
							"Output vector should be a mutable vector. It cannot be of the type input_iterator_tag" );
			static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
											bolt::cl::fancy_iterator_tag >::value , 
							"Output vector should be a mutable vector. It cannot be of type fancy_iterator_tag" );
	
		}

	} //namespace detail


/*********************************************************************************************************************
 * Inclusive Segmented Scan
 ********************************************************************************************************************/

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
inclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    BinaryPredicate binary_pred,
    BinaryFunction  binary_funct,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    oType init; memset(&init, 0, sizeof(oType) );
	using bolt::cl::detail::scan_by_key;
    return detail::scan_by_key(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        true, // inclusive
        user_code); 
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate>
OutputIterator
inclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    oType init; memset(&init, 0, sizeof(oType) );
    plus<oType> binary_funct;
	using bolt::cl::detail::scan_by_key;
    return detail::scan_by_key(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        true, // inclusive
        user_code );
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator>
OutputIterator
inclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    oType init; memset(&init, 0, sizeof(oType) );
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
	using bolt::cl::detail::scan_by_key;
    return detail::scan_by_key(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        true, // inclusive
        user_code);
}


template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
inclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    BinaryPredicate binary_pred,
    BinaryFunction  binary_funct,
    const std::string& user_code )
{
    using bolt::cl::inclusive_scan_by_key;
	return inclusive_scan_by_key(
           control::getDefault( ), 
		   first1, 
		   last1,  
		   first2,
		   result,
           binary_pred,
	       binary_funct,
		   user_code );
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryPredicate>
OutputIterator
inclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{	
    using bolt::cl::inclusive_scan_by_key;
	return inclusive_scan_by_key(
           control::getDefault( ), 
		   first1, 
		   last1,  
		   first2,
		   result,
           binary_pred,
		   user_code );
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator>
OutputIterator
inclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    const std::string& user_code )
{	
    using bolt::cl::inclusive_scan_by_key;
	return inclusive_scan_by_key(
           control::getDefault( ), 
		   first1, 
		   last1,  
		   first2,
		   result,
		   user_code );
}



/*********************************************************************************************************************
 * Exclusive Segmented Scan
 ********************************************************************************************************************/

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
exclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    BinaryPredicate binary_pred,
    BinaryFunction  binary_funct,
    const std::string& user_code )
{
	using bolt::cl::detail::scan_by_key;
    return detail::scan_by_key(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        false, // exclusive
        user_code ); 
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate>
OutputIterator
exclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    plus<oType> binary_funct;
	using bolt::cl::detail::scan_by_key;
    return detail::scan_by_key(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        false, // exclusive
        user_code ); 
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T>
OutputIterator
exclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
	using bolt::cl::detail::scan_by_key;
    return detail::scan_by_key(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        false, // exclusive
        user_code); 
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator>
OutputIterator
exclusive_scan_by_key(
    control &ctl,
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    const std::string& user_code )
{
    typedef typename std::iterator_traits<InputIterator1>::value_type kType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;
	oType init; memset(&init, 0, sizeof(oType) );
    equal_to<kType> binary_pred;
    plus<oType> binary_funct;
	using bolt::cl::detail::scan_by_key;
    return detail::scan_by_key(
        ctl,
        first1,
        last1,
        first2,
        result,
        init,
        binary_pred,
        binary_funct,
        false, // exclusive
        user_code); 
}


template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIterator
exclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    BinaryPredicate binary_pred,
    BinaryFunction  binary_funct,
    const std::string& user_code )
{
    using bolt::cl::exclusive_scan_by_key;
	return exclusive_scan_by_key(
           control::getDefault( ), 
		   first1, 
		   last1,  
		   first2,
		   result,
		   init,
		   binary_pred,
		   binary_funct,
		   user_code );
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T,
    typename BinaryPredicate>
OutputIterator
exclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    BinaryPredicate binary_pred,
    const std::string& user_code )
{
    using bolt::cl::exclusive_scan_by_key;
	return exclusive_scan_by_key(
           control::getDefault( ), 
		   first1, 
		   last1,  
		   first2,
		   result,
		   init,
		   binary_pred,
		   user_code );
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename T>
OutputIterator
exclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    T               init,
    const std::string& user_code )
{
    using bolt::cl::exclusive_scan_by_key;
	return exclusive_scan_by_key(
           control::getDefault( ), 
		   first1, 
		   last1,  
		   first2,
		   result,
		   init,
		   user_code );
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator>
OutputIterator
exclusive_scan_by_key(
    InputIterator1  first1,
    InputIterator1  last1,
    InputIterator2  first2,
    OutputIterator  result,
    const std::string& user_code )
{
    using bolt::cl::exclusive_scan_by_key;
	return exclusive_scan_by_key(
           control::getDefault( ), 
		   first1, 
		   last1,  
		   first2,
		   result,
		   user_code );
}


} //namespace cl
} //namespace bolt

#endif
