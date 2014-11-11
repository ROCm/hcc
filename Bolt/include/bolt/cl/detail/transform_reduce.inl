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

#if !defined( BOLT_CL_TRANSFORM_REDUCE_INL )
#define BOLT_CL_TRANSFORM_REDUCE_INL
#pragma once

#define WAVEFRONT_SIZE_REDUCE 256


#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/transform_reduce.h"
#endif

#include "bolt/cl/bolt.h"
#include "bolt/cl/distance.h"
#include "bolt/cl/iterator/iterator_traits.h"
#include "bolt/cl/iterator/transform_iterator.h"
#include "bolt/cl/iterator/addressof.h"
#include "bolt/cl/device_vector.h"
#include "bolt/cl/transform.h"
#include "bolt/cl/reduce.h"

namespace bolt {
namespace cl {


namespace  detail {

namespace serial{

	template<typename InputIterator, typename UnaryFunction, typename oType, typename BinaryFunction>
    oType transform_reduce(control& ctl,
            const InputIterator& first,
            const InputIterator& last,
            const UnaryFunction& transform_op,
            const oType& init,
            const BinaryFunction& reduce_op,
            const std::string& user_code,
			bolt::cl::device_vector_tag)
    {


		          size_t n = (last - first);

                  typedef typename std::iterator_traits< InputIterator >::value_type iType;
		          
	              /*Get The associated OpenCL buffer for each of the iterators*/
                  ::cl::Buffer inputBuffer = first.base().getContainer( ).getBuffer( );
                  /*Get The size of each OpenCL buffer*/
                  size_t input_sz = inputBuffer.getInfo<CL_MEM_SIZE>();
	              
                  cl_int map_err;
                  iType *inputPtr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(inputBuffer, true, CL_MAP_READ, 0, 
                                                                                      input_sz, NULL, NULL, &map_err);
                  auto mapped_ip_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>
					                                            ::iterator_category() ,ctl, first, inputPtr); 
				  //Create a temporary array to store the transform result;
				  std::vector<oType> output_vector(n);

				  std::transform(mapped_ip_itr, mapped_ip_itr + n, output_vector.begin(),transform_op);
	              oType output = std::accumulate(output_vector.begin(), output_vector.end(), init, reduce_op);
		          
	              ::cl::Event unmap_event[1];
                  ctl.getCommandQueue().enqueueUnmapMemObject(inputBuffer, inputPtr, NULL, &unmap_event[0] );
                  unmap_event[0].wait();  
		          	
		          	
		          return output;
    }



	template<typename InputIterator, typename UnaryFunction, typename oType, typename BinaryFunction>
    oType transform_reduce(control& ctl,
           const InputIterator& first,
           const InputIterator& last,
           const UnaryFunction& transform_op,
           const oType& init,
           const BinaryFunction& reduce_op,
           const std::string& user_code,
		   std::random_access_iterator_tag)
    {
		          size_t szElements = (last - first);

	              //Create a temporary array to store the transform result;
                  std::vector<oType> output(szElements);
                  std::transform(first, last, output.begin(),transform_op);
                  return std::accumulate(output.begin(), output.end(), init, reduce_op);
    }

} // end of serial


#ifdef ENABLE_TBB
namespace btbb{

	
	template<typename InputIterator, typename UnaryFunction, typename oType, typename BinaryFunction>
    oType transform_reduce(control& ctl,
            const InputIterator& first,
            const InputIterator& last,
            const UnaryFunction& transform_op,
            const oType& init,
            const BinaryFunction& reduce_op,
            const std::string& user_code,
			bolt::cl::device_vector_tag)
    {


		          size_t n = (last - first);

                  typedef typename std::iterator_traits< InputIterator >::value_type iType;
		          
	              /*Get The associated OpenCL buffer for each of the iterators*/
                  ::cl::Buffer inputBuffer = first.base().getContainer( ).getBuffer( );
                  /*Get The size of each OpenCL buffer*/
                  size_t input_sz = inputBuffer.getInfo<CL_MEM_SIZE>();
	              
                  cl_int map_err;
                  iType *inputPtr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(inputBuffer, true, CL_MAP_READ, 0, 
                                                                                      input_sz, NULL, NULL, &map_err);
                  auto mapped_ip_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>
					                                          ::iterator_category(), ctl, first, inputPtr); 
				  
	              oType output = bolt::btbb::transform_reduce(mapped_ip_itr, mapped_ip_itr + n, transform_op,
					  init, reduce_op);
		          
	              ::cl::Event unmap_event[1];
                  ctl.getCommandQueue().enqueueUnmapMemObject(inputBuffer, inputPtr, NULL, &unmap_event[0] );
                  unmap_event[0].wait();  
		          	
		          	
		          return output;

    }



	template<typename InputIterator, typename UnaryFunction, typename oType, typename BinaryFunction>
    oType transform_reduce(control& ctl,
           const InputIterator& first,
           const InputIterator& last,
           const UnaryFunction& transform_op,
           const oType& init,
           const BinaryFunction& reduce_op,
           const std::string& user_code,
		   std::random_access_iterator_tag)
    {
		          return bolt::btbb::transform_reduce(first,last,transform_op,init,reduce_op);
    }

}//end of namespace btbb 
#endif

namespace cl{

    enum transformReduceTypes {tr_iType, tr_iIterType, tr_oType, tr_UnaryFunction,
    tr_BinaryFunction, tr_end };

    class TransformReduce_KernelTemplateSpecializer : public KernelTemplateSpecializer
    {
    public:
       TransformReduce_KernelTemplateSpecializer() : KernelTemplateSpecializer()
        {
            addKernelName("transform_reduceTemplate");
        }

        const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
        {

            const std::string templateSpecializationString =
                "// Host generates this instantiation string with user-specified value type and functor\n"
                "template __attribute__((mangled_name("+name(0)+"Instantiated)))\n"
                "__attribute__((reqd_work_group_size(256,1,1)))\n"
                "kernel void "+name(0)+"(\n"
                "global " + typeNames[tr_iType] + "* input_ptr,\n"
                + typeNames[tr_iIterType] + " iIter,\n"
                "const int length,\n"
                "global " + typeNames[tr_UnaryFunction] + "* transformFunctor,\n"
                "const " + typeNames[tr_oType] + " init,\n"
                "global " + typeNames[tr_BinaryFunction] + "* reduceFunctor,\n"
                "global " + typeNames[tr_oType] + "* result,\n"
                "local " + typeNames[tr_oType] + "* scratch\n"
                ");\n\n";
                return templateSpecializationString;
        }
    };

	template<typename InputIterator, typename UnaryFunction, typename oType, typename BinaryFunction>
    oType transform_reduce(control& ctl,
        const InputIterator& first,
        const InputIterator& last,
        const UnaryFunction& transform_op,
        const oType& init,
        const BinaryFunction& reduce_op,
        const std::string& user_code,
		bolt::cl::device_vector_tag)
    {
        unsigned debugMode = 0; //FIXME, use control

        typedef typename std::iterator_traits< InputIterator  >::value_type iType;

        /**********************************************************************************
            * Type Names - used in KernelTemplateSpecializer
            *********************************************************************************/
        std::vector<std::string> typeNames( tr_end );
        typeNames[tr_iType] = TypeName< iType >::get( );
        typeNames[tr_iIterType] = TypeName< InputIterator >::get( );
        typeNames[tr_oType] = TypeName< oType >::get( );
        typeNames[tr_UnaryFunction] = TypeName< UnaryFunction >::get( );
        typeNames[tr_BinaryFunction] = TypeName< BinaryFunction >::get();

        /**********************************************************************************
            * Type Definitions - directrly concatenated into kernel string
            *********************************************************************************/
        std::vector<std::string> typeDefinitions;
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< InputIterator >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< oType >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< UnaryFunction >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< BinaryFunction  >::get() )

        /**********************************************************************************
            * Calculate Work Size
            *********************************************************************************/

        // Set up shape of launch grid and buffers:
        // FIXME, read from device attributes.

        int computeUnits     = ctl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();// round up if we don't know.
        int wgPerComputeUnit =  64;//ctl.getWGPerComputeUnit();

        int numWG = computeUnits * wgPerComputeUnit;

        cl_int l_Error = CL_SUCCESS;
		const size_t wgSize = WAVEFRONT_SIZE_REDUCE;
        V_OPENCL( l_Error, "Error querying kernel for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" );

        /**********************************************************************************
            * Compile Options
            *********************************************************************************/
        bool cpuDevice = ctl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
        const size_t kernel_WgSize = (cpuDevice) ? 1 : wgSize;
        std::string compileOptions;
        std::ostringstream oss;
        oss << " -DKERNELWORKGROUPSIZE=" << kernel_WgSize;
        compileOptions = oss.str();

        /**********************************************************************************
            * Request Compiled Kernels
            *********************************************************************************/
        TransformReduce_KernelTemplateSpecializer ts_kts;
        std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
            ctl,
            typeNames,
            &ts_kts,
            typeDefinitions,
            transform_reduce_kernels,
            compileOptions);
        // kernels returned in same order as added in KernelTemplaceSpecializer constructor


        // Create Buffer wrappers so we can access the host functors, for read or writing in the kernel
        ALIGNED( 256 ) UnaryFunction aligned_unary( transform_op );
        ALIGNED( 256 ) BinaryFunction aligned_binary( reduce_op );

        control::buffPointer transformFunctor = ctl.acquireBuffer( sizeof( aligned_unary ),
                                    CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_unary );
        control::buffPointer reduceFunctor = ctl.acquireBuffer( sizeof( aligned_binary ),
                                    CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_binary );
        control::buffPointer result = ctl.acquireBuffer( sizeof( oType ) * numWG,
                                                CL_MEM_ALLOC_HOST_PTR|CL_MEM_WRITE_ONLY );

        cl_uint szElements = static_cast< cl_uint >( std::distance( first, last ) );

        /***** This is a temporaray fix *****/

        /*What if  requiredWorkGroups > numWG? Do you want to loop or increase the work group size
        or increase the per item processing?*/

        int requiredWorkGroups = (int)ceil((float)szElements/wgSize);
        if (requiredWorkGroups < numWG)
            numWG = requiredWorkGroups;
        /**********************/

        typename  InputIterator::Payload first_payload = first.gpuPayload( ) ;

        V_OPENCL( kernels[0].setArg( 0, first.base().getContainer().getBuffer() ), "Error setting kernel argument" );
        V_OPENCL( kernels[0].setArg( 1, first.gpuPayloadSize( ),&first_payload),
                                                        "Error setting kernel argument" );

        V_OPENCL( kernels[0].setArg( 2, szElements), "Error setting kernel argument" );
        V_OPENCL( kernels[0].setArg( 3, *transformFunctor), "Error setting kernel argument" );
        V_OPENCL( kernels[0].setArg( 4, init), "Error setting kernel argument" );
        V_OPENCL( kernels[0].setArg( 5, *reduceFunctor), "Error setting kernel argument" );
        V_OPENCL( kernels[0].setArg( 6, *result), "Error setting kernel argument" );

        ::cl::LocalSpaceArg loc;
        loc.size_ = wgSize*sizeof(oType);
        V_OPENCL( kernels[0].setArg( 7, loc ), "Error setting kernel argument" );

        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
            kernels[0],
            ::cl::NullRange,
            ::cl::NDRange(numWG * wgSize),
            ::cl::NDRange(wgSize) );
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for transform_reduce() kernel" );

        ::cl::Event l_mapEvent;
        oType *h_result = (oType*)ctl.getCommandQueue().enqueueMapBuffer(*result, false, CL_MAP_READ, 0,
                                                    sizeof(oType)*numWG, NULL, &l_mapEvent, &l_Error );
        V_OPENCL( l_Error, "Error calling map on the result buffer" );

        //  Finish the tail end of the reduction on host side; the compute device reduces within the workgroups,
        // with one result per workgroup
        size_t ceilNumWG = static_cast< size_t >( std::ceil( static_cast< float >( szElements ) / wgSize) );
        bolt::cl::minimum< size_t >  min_size_t;
        size_t numTailReduce = min_size_t( ceilNumWG, numWG );

        bolt::cl::wait(ctl, l_mapEvent);

        oType acc = static_cast< oType >( init );
        for(unsigned int i = 0; i < numTailReduce; ++i)
        {
            acc = reduce_op( acc, h_result[ i ] );
        }


		::cl::Event unmapEvent;

		V_OPENCL( ctl.getCommandQueue().enqueueUnmapMemObject(*result,  h_result, NULL, &unmapEvent ),
			"shared_ptr failed to unmap host memory back to device memory" );
		V_OPENCL( unmapEvent.wait( ), "failed to wait for unmap event" );

        return acc;
    }



	template<typename InputIterator, typename UnaryFunction, typename oType, typename BinaryFunction>
    oType transform_reduce(control& ctl,
        const InputIterator& first,
        const InputIterator& last,
        const UnaryFunction& transform_op,
        const oType& init,
        const BinaryFunction& reduce_op,
        const std::string& user_code,
		std::random_access_iterator_tag)
    {
        int sz = static_cast<int>(last - first);
        if (sz == 0)
            return init;
        typedef typename std::iterator_traits<InputIterator>::value_type  iType;
       	          
        typedef typename bolt::cl::iterator_traits<InputIterator>::pointer pointer;
       	          
        pointer first_pointer = bolt::cl::addressof(first) ;
	          
        device_vector< iType > dvInput( first_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
                  
        auto device_iterator_first  = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                            first, dvInput.begin());
        auto device_iterator_last   = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                            last, dvInput.end());

		// Map the input iterator to a device_vector
        return  transform_reduce( ctl, device_iterator_first, device_iterator_last, 
			transform_op, init, reduce_op, user_code, typename bolt::cl::device_vector_tag() );

    }



	template<typename InputIterator, typename UnaryFunction, typename oType, typename BinaryFunction>
    oType transform_reduce(control& ctl,
        const InputIterator& first,
        const InputIterator& last,
        const UnaryFunction& transform_op,
        const oType& init,
        const BinaryFunction& reduce_op,
        const std::string& user_code,
			bolt::cl::fancy_iterator_tag)
    {
        return transform_reduce(ctl, first, last, transform_op, init, reduce_op, user_code,
                                typename bolt::cl::memory_system<InputIterator>::type() );  
    }

} // end of namespace cl

    // Wrapper that uses default control class, iterator interface
    template<typename InputIterator, typename UnaryFunction, typename T, typename BinaryFunction>
	typename std::enable_if< 
            !(std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                            std::input_iterator_tag 
                        >::value), T
                        >::type
    transform_reduce( control& ctl, const InputIterator& first, const InputIterator& last,
        const UnaryFunction& transform_op,
        const T& init,const BinaryFunction& reduce_op,const std::string& user_code)
    {
                typedef typename std::iterator_traits<InputIterator>::value_type iType;
                size_t szElements = static_cast<size_t>(std::distance(first, last) );
                if (szElements == 0)
                        return init;
			      
                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if(runMode == bolt::cl::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
			    #if defined(BOLT_DEBUG_LOG)
                BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
                #endif
                if (runMode == bolt::cl::control::SerialCpu)
                {
			        #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORMREDUCE,
						BOLTLOG::BOLT_SERIAL_CPU,"::Transform_Reduce::SERIAL_CPU");
                    #endif
			      	
                    return serial::transform_reduce(ctl,  first, last, transform_op, init, reduce_op, user_code,
						typename std::iterator_traits<InputIterator>::iterator_category() );
                }
                else if (runMode == bolt::cl::control::MultiCoreCpu)
                {
#ifdef ENABLE_TBB
                    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORMREDUCE,BOLTLOG::BOLT_MULTICORE_CPU,"
						::Transform_Reduce::MULTICORE_CPU");
                    #endif
				      
				    return  btbb::transform_reduce( ctl, first, last, transform_op, init, reduce_op, user_code,
						typename std::iterator_traits<InputIterator>::iterator_category() );
#else

                    throw std::runtime_error( "The MultiCoreCpu version of transform_reduce function is not enabled to be built! \n");
				    return init;

#endif
                }
                #if defined(BOLT_DEBUG_LOG)
                dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORMREDUCE,BOLTLOG::BOLT_OPENCL_GPU,
					"::Transform_Reduce::OPENCL_GPU");
                #endif
                return  cl::transform_reduce( ctl, first, last, transform_op, init, reduce_op, user_code,
					typename std::iterator_traits<InputIterator>::iterator_category() );
    };



	template<typename InputIterator, typename UnaryFunction, typename T, typename BinaryFunction>
    typename std::enable_if< 
            (std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                            std::input_iterator_tag 
                        >::value), T
                        >::type
    transform_reduce(control &ctl, const InputIterator& first, const InputIterator& last,
        const UnaryFunction& transform_op,
        const T& init, const BinaryFunction& reduce_op, const std::string& user_code )
    {
                //TODO - Shouldn't we support transform for input_iterator_tag also. 
                static_assert( std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                                            std::input_iterator_tag >::value , 
                                "Input vector cannot be of the type input_iterator_tag" );
    }



}// end of namespace detail



    // The following two functions are visible in .h file
    // Wrapper that user passes a control class
    template<typename InputIterator, typename UnaryFunction, typename T, typename BinaryFunction>
    T transform_reduce( control& ctl, InputIterator first, InputIterator last,
        UnaryFunction transform_op,
        T init,  BinaryFunction reduce_op, const std::string& user_code )
    {
        return detail::transform_reduce( ctl, first, last, transform_op, init, reduce_op, user_code);
    };

    // Wrapper that generates default control class
    template<typename InputIterator, typename UnaryFunction, typename T, typename BinaryFunction>
    T transform_reduce(InputIterator first, InputIterator last,
        UnaryFunction transform_op,
        T init,  BinaryFunction reduce_op, const std::string& user_code )
    {
        return transform_reduce( control::getDefault(), first, last, transform_op, init, reduce_op, user_code);
    };


}// end of namespace cl
}// end of namespace bolt

#endif
