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

#if !defined( BOLT_CL_COUNT_INL )
#define BOLT_CL_COUNT_INL
#pragma once

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/count.h"
#endif


#include "bolt/cl/bolt.h"
#include "bolt/cl/device_vector.h"
#include "bolt/cl/distance.h"
#include "bolt/cl/iterator/iterator_traits.h"
#include "bolt/cl/iterator/transform_iterator.h"
#include "bolt/cl/iterator/addressof.h"

namespace bolt{
namespace cl{


namespace detail{


namespace serial{

    template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		std::random_access_iterator_tag)
    {

		return std::count_if(first,last,predicate);

	}


	template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		bolt::cl::device_vector_tag)
    {

		size_t n = (last - first);

        typedef typename std::iterator_traits< InputIterator >::value_type iType;
		typedef typename bolt::cl::iterator_traits<InputIterator>::difference_type rType;

	    /*Get The associated OpenCL buffer for each of the iterators*/
        ::cl::Buffer inputBuffer = first.base().getContainer( ).getBuffer( );
        /*Get The size of each OpenCL buffer*/
        size_t input_sz = inputBuffer.getInfo<CL_MEM_SIZE>();
	    
        cl_int map_err;
        iType *inputPtr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(inputBuffer, true, CL_MAP_READ, 0, 
                                                                            input_sz, NULL, NULL, &map_err);
        auto mapped_ip_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>::iterator_category(),
                                                        ctl, first, inputPtr); 
		

	    std::iterator_traits<std::vector<int>::iterator>::difference_type output = std::count_if(mapped_ip_itr,
			mapped_ip_itr + (int)n, predicate);
		
	    ::cl::Event unmap_event[1];
        ctl.getCommandQueue().enqueueUnmapMemObject(inputBuffer, inputPtr, NULL, &unmap_event[0] );
        unmap_event[0].wait(); 		


		return (rType)output;

	}


	template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		bolt::cl::fancy_iterator_tag)
    {

		return std::count_if(first,last,predicate);

	}

} // end of namespace serial


#ifdef ENABLE_TBB
namespace btbb{

	template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		std::random_access_iterator_tag)
    {

		return bolt::btbb::count_if(first,last,predicate);

	}


	template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		bolt::cl::device_vector_tag)
    {

		size_t n = (last - first);

        typedef typename std::iterator_traits< InputIterator >::value_type iType;
		typedef typename bolt::cl::iterator_traits<InputIterator>::difference_type rType;

	    /*Get The associated OpenCL buffer for each of the iterators*/
        ::cl::Buffer inputBuffer = first.base().getContainer( ).getBuffer( );
        /*Get The size of each OpenCL buffer*/
        size_t input_sz = inputBuffer.getInfo<CL_MEM_SIZE>();
	    
        cl_int map_err;
        iType *inputPtr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(inputBuffer, true, CL_MAP_READ, 0, 
                                                                            input_sz, NULL, NULL, &map_err);
        auto mapped_ip_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                                        ctl, first, inputPtr); 
		

	    std::iterator_traits<std::vector<int>::iterator>::difference_type output = bolt::btbb::count_if(mapped_ip_itr,
			mapped_ip_itr + (int)n, predicate);
		
	    ::cl::Event unmap_event[1];
        ctl.getCommandQueue().enqueueUnmapMemObject(inputBuffer, inputPtr, NULL, &unmap_event[0] );
        unmap_event[0].wait();


		return (rType)output;

	}


	template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		bolt::cl::fancy_iterator_tag)
    {

		return bolt::btbb::count_if(first,last,predicate);

	}


}// end of namespace btbb
#endif

namespace cl{
    enum CountTypes {count_iValueType, count_iIterType, count_predicate, count_end };

    ///////////////////////////////////////////////////////////////////////
    //Kernel Template Specializer
    ///////////////////////////////////////////////////////////////////////
    class Count_KernelTemplateSpecializer : public KernelTemplateSpecializer
    {
        public:

        Count_KernelTemplateSpecializer() : KernelTemplateSpecializer()
            {
                addKernelName( "count_Template" );
            }

        const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
        {
            const std::string templateSpecializationString =
                    "// Host generates this instantiation string with user-specified value type and functor\n"
                    "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
                    "__attribute__((reqd_work_group_size(256,1,1)))\n"
                    "kernel void " + name(0) + "(\n"
                    "global " + typeNames[count_iValueType] + "* input_ptr,\n"
                     + typeNames[count_iIterType] + " output_iter,\n"
                    "const int length,\n"
                    "global " + typeNames[count_predicate] + "* userFunctor,\n"
                    "global int *result,\n"
                    "local int *scratch_index\n"
                    ");\n\n";

            return templateSpecializationString;
        }
    };


    //----
    // This is the base implementation of reduction that is called by all of the convenience wrappers below.
    // first and last must be iterators from a DeviceVector
    template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		bolt::cl::device_vector_tag)
    {
        typedef typename std::iterator_traits< InputIterator >::value_type iType;
        typedef typename bolt::cl::iterator_traits<InputIterator>::difference_type rType;
        std::vector<std::string> typeNames( count_end);
        typeNames[count_iValueType] = TypeName< iType >::get( );
        typeNames[count_iIterType] = TypeName< InputIterator >::get( );
        typeNames[count_predicate] = TypeName< Predicate >::get();

        std::vector<std::string> typeDefinitions;
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< InputIterator >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< Predicate  >::get() )

        //bool cpuDevice = ctl.device().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
        /*\TODO - Do CPU specific kernel work group size selection here*/
        //const size_t kernel0_WgSize = (cpuDevice) ? 1 : WAVESIZE*KERNEL02WAVES;
        std::string compileOptions;

        Count_KernelTemplateSpecializer ts_kts;
        std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
            ctl,
            typeNames,
            &ts_kts,
            typeDefinitions,
            count_kernels,
            compileOptions);


        // Set up shape of launch grid and buffers:
        cl_uint computeUnits     = ctl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        int wgPerComputeUnit =  64; //ctl.getWGPerComputeUnit();
        size_t numWG = computeUnits * wgPerComputeUnit;

        cl_int l_Error = CL_SUCCESS;
        const size_t wgSize  = 256; 
        V_OPENCL( l_Error, "Error querying kernel for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" );

        // Create buffer wrappers so we can access the host functors, for read or writing in the kernel
        ALIGNED( 256 ) Predicate aligned_count( predicate );

        control::buffPointer userFunctor = ctl.acquireBuffer( sizeof( aligned_count ),
            CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_count );


        control::buffPointer result = ctl.acquireBuffer( sizeof( int ) * numWG,
            CL_MEM_ALLOC_HOST_PTR|CL_MEM_WRITE_ONLY );

        cl_uint szElements = static_cast< cl_uint >( std::distance( first, last ) ); 
         typename InputIterator::Payload  first_payload = first.gpuPayload();
        V_OPENCL( kernels[0].setArg(0, first.base().getContainer().getBuffer() ), "Error setting kernel argument" );

        V_OPENCL( kernels[0].setArg(1, first.gpuPayloadSize( ), &first_payload), "Error setting a kernel argument" );

        V_OPENCL( kernels[0].setArg(2, szElements), "Error setting kernel argument" );
        V_OPENCL( kernels[0].setArg(3, *userFunctor), "Error setting kernel argument" );
        V_OPENCL( kernels[0].setArg(4, *result), "Error setting kernel argument" );

        ::cl::LocalSpaceArg loc2;
        loc2.size_ = wgSize*sizeof(int);;
        V_OPENCL( kernels[0].setArg(5, loc2), "Error setting kernel argument" );


        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
            kernels[0],
            ::cl::NullRange,
            ::cl::NDRange(numWG * wgSize),
            ::cl::NDRange(wgSize));
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for count() kernel" );

        ::cl::Event l_mapEvent;
        int *h_result = (int*)ctl.getCommandQueue().enqueueMapBuffer(*result, false, CL_MAP_READ, 0,
            sizeof(int)*numWG, NULL, &l_mapEvent, &l_Error );
        V_OPENCL( l_Error, "Error calling map on the result buffer" );

        //  Finish the tail end of the reduction on host side; the compute device counts within the workgroups,
        //  with one result per workgroup
        size_t ceilNumWG = static_cast< size_t >( std::ceil( static_cast< float >( szElements ) / wgSize) );
        bolt::cl::minimum<size_t>  count_size_t;
        size_t numTailReduce = count_size_t( ceilNumWG, numWG );

        bolt::cl::wait(ctl, l_mapEvent);

        rType count =  h_result[0] ;
        for(unsigned int i = 1; i < numTailReduce; ++i)
        {

           count +=  h_result[i];

        }


	::cl::Event unmapEvent;

	V_OPENCL( ctl.getCommandQueue().enqueueUnmapMemObject(*result,  h_result, NULL, &unmapEvent ),
		"shared_ptr failed to unmap host memory back to device memory" );
    V_OPENCL( unmapEvent.wait( ), "failed to wait for unmap event" );    
	return count;

    }


	template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		std::random_access_iterator_tag)
    {

		 int sz = static_cast<int>(last - first);

         typedef typename std::iterator_traits<InputIterator>::value_type  iType;
       	 
         typedef typename std::iterator_traits<InputIterator>::pointer pointer;
       	 
         pointer first_pointer = bolt::cl::addressof(first);
		 
         device_vector< iType > dvInput( first_pointer, sz, CL_MEM_USE_HOST_PTR |  CL_MEM_READ_ONLY, true, ctl );
         
		 // Map the input iterator to a device_vector
         auto device_iterator_first  = bolt::cl::create_device_itr(
                                             typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                             first, dvInput.begin());
         auto device_iterator_last   = bolt::cl::create_device_itr(
                                             typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                             last, dvInput.end());

         return  count( ctl, device_iterator_first, device_iterator_last, predicate, cl_code, 
			 bolt::cl::device_vector_tag());


	}


	template<typename InputIterator, typename Predicate>
    typename bolt::cl::iterator_traits<InputIterator>::difference_type
        count(bolt::cl::control &ctl,
        const InputIterator& first,
        const InputIterator& last,
        const Predicate& predicate,
        const std::string& cl_code,
		bolt::cl::fancy_iterator_tag)
    {
		 return count(ctl, first, last, predicate, cl_code, typename bolt::cl::memory_system<InputIterator>::type() );

	}
	

} // end of namespace cl


	template<typename InputIterator, typename Predicate>
    typename std::enable_if< 
           !(std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                         std::input_iterator_tag 
                       >::value), typename bolt::cl::iterator_traits<InputIterator>::difference_type
                       >::type
    count( bolt::cl::control &ctl,
                const InputIterator& first,
                const InputIterator& last,
                const Predicate& predicate,
                const std::string& cl_code)
    {

        typedef typename bolt::cl::iterator_traits<InputIterator>::difference_type rType;
	    
        size_t szElements = (size_t)(last - first);
        if (szElements == 0)
            return 0;
	    
        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
        if(runMode == bolt::cl::control::Automatic)
        {
            runMode = ctl.getDefaultPathToRun();
        }
        #if defined(BOLT_DEBUG_LOG)
        BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
        #endif
	    
        switch(runMode)
        {
        case bolt::cl::control::OpenCL :
	    #if defined(BOLT_DEBUG_LOG)
              dblog->CodePathTaken(BOLTLOG::BOLT_COUNT,BOLTLOG::BOLT_OPENCL_GPU,"::Count::OPENCL_GPU");
              #endif 
              return  cl::count( ctl, first, last,  predicate, cl_code, 
				  typename std::iterator_traits< InputIterator >::iterator_category( ) );
	    
        case bolt::cl::control::MultiCoreCpu:
        #ifdef ENABLE_TBB
        {
              #if defined(BOLT_DEBUG_LOG)
              dblog->CodePathTaken(BOLTLOG::BOLT_COUNT,BOLTLOG::BOLT_MULTICORE_CPU,"::Count::MULTICORE_CPU");
              #endif
              
              return btbb::count( ctl, first, last,  predicate, cl_code,
				  typename std::iterator_traits< InputIterator >::iterator_category( ) );
	    
        }
        #else
        {
              throw std::runtime_error( "The MultiCoreCpu version of reduce is not enabled to be built! \n" );
        }
        #endif
	    
        case bolt::cl::control::SerialCpu:
        {
              #if defined(BOLT_DEBUG_LOG)
              dblog->CodePathTaken(BOLTLOG::BOLT_COUNT,BOLTLOG::BOLT_SERIAL_CPU,"::Count::SERIAL_CPU");
              #endif
	    
              return  serial::count( ctl, first, last,  predicate, cl_code, 
				  typename std::iterator_traits< InputIterator >::iterator_category( ) );
	    
        }
	   
        default: /* Incase of runMode not set/corrupted */
        {
             #if defined(BOLT_DEBUG_LOG)
             dblog->CodePathTaken(BOLTLOG::BOLT_COUNT,BOLTLOG::BOLT_SERIAL_CPU,"::Count::SERIAL_CPU");
             #endif	
	   
             return  serial::count( ctl, first, last,  predicate, cl_code, 
				 typename std::iterator_traits< InputIterator >::iterator_category( ) );
        }
	   
       }

    }



	template<typename InputIterator, typename Predicate>
    typename std::enable_if< 
           (std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                         std::input_iterator_tag 
                       >::value), typename bolt::cl::iterator_traits<InputIterator>::difference_type
                       >::type
    count( bolt::cl::control &ctl,
                const InputIterator& first,
                const InputIterator& last,
                const Predicate& predicate,
                const std::string& cl_code)
    {
		      //  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied
              //   the data to a temporary buffer.  Should we?
              static_assert( std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                                           std::input_iterator_tag >::value , 
                             "Input vector cannot be of the type input_iterator_tag" );
    }

} //namespace detail


template<typename InputIterator, typename Predicate>
typename bolt::cl::iterator_traits<InputIterator>::difference_type
    count_if(control& ctl, InputIterator first,
    InputIterator last,
    Predicate predicate,
    const std::string& cl_code)
    {
        return detail::count(ctl, first, last, predicate, cl_code);
    }

template<typename InputIterator, typename Predicate>
typename bolt::cl::iterator_traits<InputIterator>::difference_type
    count_if( InputIterator first,
    InputIterator last,
    Predicate predicate,
    const std::string& cl_code)
    {

        return count_if(bolt::cl::control::getDefault(), first, last, predicate, cl_code);

    }

} //namespace cl

}; //namespace bolt

#endif //COUNT_INL
