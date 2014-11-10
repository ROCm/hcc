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

#if !defined( BOLT_CL_GENERATE_INL )
#define BOLT_CL_GENERATE_INL
#pragma once

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/generate.h"
#endif
#define BURST 1

namespace bolt {
namespace cl {

namespace detail {

enum GenerateTypes { gen_oType, gen_genType, generate_DVInputIterator, generate_end };

/**********************************************************************************************************************
 * Kernel Template Specializer
 *********************************************************************************************************************/
class Generate_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
    public:

    Generate_KernelTemplateSpecializer() : KernelTemplateSpecializer()
    {
        addKernelName( "generate_I"   );
        addKernelName( "generate_II"  );
        addKernelName( "generate_III" );
    }

    const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
    {
        const std::string templateSpecializationString =
                        "// Host generates this instantiation string with user-specified value type and generator\n"
                "template __attribute__((mangled_name("+name(0)+"Instantiated)))\n"
                "kernel void "+name(0)+"(\n"
                "global " + typeNames[gen_oType] + " * restrict dst,\n"
                 + typeNames[generate_DVInputIterator] + " input_iter,\n"
                "const int numElements,\n"
                "global " + typeNames[gen_genType] + " * restrict genPtr);\n\n"

                        "// Host generates this instantiation string with user-specified value type and generator\n"
                "template __attribute__((mangled_name("+name(1)+"Instantiated)))\n"
                "kernel void "+name(1)+"(\n"
                "global " + typeNames[gen_oType] + " * restrict dst,\n"
                "const int numElements,\n"
                "global " + typeNames[gen_genType] + " * restrict genPtr);\n\n"

                        "// Host generates this instantiation string with user-specified value type and generator\n"
                "template __attribute__((mangled_name("+name(2)+"Instantiated)))\n"
                "kernel void "+name(2)+"(\n"
                "global " + typeNames[gen_oType] + " * restrict dst,\n"
                "const int numElements,\n"
                "global " + typeNames[gen_genType] + " * restrict genPtr);\n\n"
                ;

        return templateSpecializationString;
    }
};


/*****************************************************************************
* Enqueue
****************************************************************************/

template< typename DVForwardIterator, typename Generator >
void generate_enqueue(
    bolt::cl::control &ctrl,
    const DVForwardIterator &first,
    const DVForwardIterator &last,
    const Generator &gen,
    const std::string& cl_code )
{
#ifdef BOLT_ENABLE_PROFILING
aProfiler.setName("generate");
aProfiler.startTrial();
aProfiler.setStepName("Acquire Kernel");
aProfiler.set(AsyncProfiler::device, control::SerialCpu);
#endif
    cl_int l_Error;

    /**********************************************************************************
     * Type Names - used in KernelTemplateSpecializer
     *********************************************************************************/
     typedef typename std::iterator_traits<DVForwardIterator>::value_type oType;
    std::vector<std::string> typeNames(generate_end);
    typeNames[gen_oType] = TypeName< oType >::get( );
    typeNames[gen_genType] = TypeName< Generator >::get( );
    typeNames[generate_DVInputIterator] = TypeName< DVForwardIterator >::get( );

    /**********************************************************************************
     * Type Definitions - directly concatenated into kernel string (order may matter)
     *********************************************************************************/
    std::vector<std::string> typeDefs;
    PUSH_BACK_UNIQUE( typeDefs, ClCode< oType >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< Generator >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< DVForwardIterator >::get() )

    /**********************************************************************************
     * Number of Threads
     *********************************************************************************/
    const cl_uint numElements = static_cast< cl_uint >( std::distance( first, last ) );
    if (numElements < 1) return;
    const int workGroupSize  = 256;
    const int numComputeUnits = static_cast<int>( ctrl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( ) ); // = 28
    const int numWorkGroupsPerComputeUnit = ctrl.getWGPerComputeUnit( );
    const int numWorkGroupsIdeal = numComputeUnits * numWorkGroupsPerComputeUnit;
    const cl_uint numThreadsIdeal = static_cast<cl_uint>( numWorkGroupsIdeal * workGroupSize );
    int doBoundaryCheck = 0;
    cl_uint numThreadsRUP = numElements;
    int mod = (numElements & (workGroupSize-1));
    if( mod )
            {
        numThreadsRUP &= ~mod;
        numThreadsRUP += workGroupSize;
        doBoundaryCheck = 1;
    }

    /**********************************************************************************
     * Compile Options
     *********************************************************************************/
    std::string compileOptions;
    std::ostringstream oss;
    oss << " -DBURST=" << BURST;
    oss << " -DBOUNDARY_CHECK=" << doBoundaryCheck;
    compileOptions = oss.str();

    /**********************************************************************************
     * Request Compiled Kernels
     *********************************************************************************/
    Generate_KernelTemplateSpecializer kts;
    std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
        ctrl,
        typeNames,
        &kts,
        typeDefs,
        generate_kernels,
        compileOptions);

#ifdef BOLT_ENABLE_PROFILING
aProfiler.nextStep();
aProfiler.setStepName("Acquire Intermediate Buffers");
aProfiler.set(AsyncProfiler::device, control::SerialCpu);
#endif

    /**********************************************************************************
     * Temporary Buffers
     *********************************************************************************/
                ALIGNED( 256 ) Generator aligned_generator( gen );
                // ::cl::Buffer userGenerator(ctl.context(), CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                //  sizeof( aligned_generator ), const_cast< Generator* >( &aligned_generator ) );
                control::buffPointer userGenerator = ctrl.acquireBuffer( sizeof( aligned_generator ),
                    CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, &aligned_generator );

#ifdef BOLT_ENABLE_PROFILING
aProfiler.nextStep();
aProfiler.setStepName("Kernel 0 Setup");
aProfiler.set(AsyncProfiler::device, control::SerialCpu);
#endif

    int whichKernel = 0;
    cl_uint numThreadsChosen;
    cl_uint workGroupSizeChosen = workGroupSize;
    switch( whichKernel )
    {
    case 0: // I: thread per element
        numThreadsChosen = numThreadsRUP;
        break;
    case 1: // II: ideal threads
        numThreadsChosen = numThreadsIdeal;
        break;
    case 2: // III:   ideal threads, BURST
        numThreadsChosen = numThreadsIdeal;
        break;
    } // switch kernel

    typename DVForwardIterator::Payload first_payload = first.gpuPayload( ) ;
    V_OPENCL( kernels[whichKernel].setArg( 0, first.getContainer().getBuffer()),"Error setArg kernels[0]");//I/P Buffer
    V_OPENCL( kernels[whichKernel].setArg( 1, first.gpuPayloadSize( ),&first_payload),
        "Error setting a kernel argument" );
    V_OPENCL( kernels[whichKernel].setArg( 2, numElements),         "Error setArg kernels[ 0 ]" ); // Size of buffer
    V_OPENCL( kernels[whichKernel].setArg( 3, *userGenerator ),     "Error setArg kernels[ 0 ]" ); // Generator

#ifdef BOLT_ENABLE_PROFILING
aProfiler.nextStep();
aProfiler.setStepName("Kernel 0 Enqueue");
aProfiler.set(AsyncProfiler::device, ctrl.forceRunMode());
aProfiler.nextStep();
aProfiler.setStepName("Kernel 0");
aProfiler.set(AsyncProfiler::device, ctrl.forceRunMode());
aProfiler.set(AsyncProfiler::flops, 1*numElements);
aProfiler.set(AsyncProfiler::memory, 1*numElements*sizeof(oType)
        + (numThreadsChosen/workGroupSizeChosen)*sizeof(Generator));
#endif

                // enqueue kernel
                ::cl::Event generateEvent;
    l_Error = ctrl.getCommandQueue().enqueueNDRangeKernel(
        kernels[whichKernel],
                    ::cl::NullRange,
        ::cl::NDRange(numThreadsChosen),
        ::cl::NDRange(workGroupSizeChosen),
                    NULL,
                    &generateEvent );
                V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for generate() kernel" );

                // wait to kernel completion
    bolt::cl::wait(ctrl, generateEvent);
#if 0
#ifdef BOLT_ENABLE_PROFILING
aProfiler.nextStep();
aProfiler.setStepName("Querying Kernel Times");
aProfiler.set(AsyncProfiler::device, control::SerialCpu);

aProfiler.setDataSize(numElements*sizeof(iType));
std::string strDeviceName = ctrl.device().getInfo< CL_DEVICE_NAME >( &l_Error );
bolt::cl::V_OPENCL( l_Error, "Device::getInfo< CL_DEVICE_NAME > failed" );
aProfiler.setArchitecture(strDeviceName);

    try
    {
        cl_ulong k0_start, k0_stop, k1_stop, k2_stop;
        cl_ulong k1_start, k2_start;

        l_Error = kernel0Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &k0_start);
        V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()");
        l_Error = kernel0Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &k0_stop);
        V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");

        l_Error = kernel1Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &k1_start);
        V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_START>()");
        l_Error = kernel1Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &k1_stop);
        V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");

        l_Error = kernel2Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &k2_start);
        V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_START>()");
        l_Error = kernel2Event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &k2_stop);
        V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");

        size_t k0_start_cpu = aProfiler.get(k0_stepNum, AsyncProfiler::startTime);
        size_t shift = k0_start - k0_start_cpu;
        //size_t shift = k0_start_cpu - k0_start;

        //std::cout << "setting step " << k0_stepNum << " attribute " << AsyncProfiler::stopTime
        //<< " to " << k0_stop-shift << std::endl;
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
#endif // ENABLE_PROFILING
#endif
    // profiling
    cl_command_queue_properties queueProperties;
    l_Error = ctrl.getCommandQueue().getInfo<cl_command_queue_properties>(CL_QUEUE_PROPERTIES, &queueProperties);
    unsigned int profilingEnabled = queueProperties&CL_QUEUE_PROFILING_ENABLE;
    if (1 && profilingEnabled) {
        cl_ulong start_time, stop_time;
        l_Error = generateEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start_time);
        V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_START>()");
        l_Error = generateEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &stop_time);
        V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");
        cl_ulong time = stop_time - start_time;
        double gb = (numElements*sizeof(oType))/1024.0/1024.0/1024.0;
        double sec = time/1000000000.0;
        std::cout << "Global Memory Bandwidth: " << ( gb / sec) << " ( "
          << time/1000000.0 << " ms)" << std::endl;
    }
}; // end generate_enqueue


/*****************************************************************************
             * Pick Iterator
****************************************************************************/

/*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
               \detail This template is called by the non-detail versions of generate, it already assumes random access
             *  iterators.  This overload is called strictly for non-device_vector iterators
            */
            template<typename ForwardIterator, typename Generator>
            void generate_pick_iterator(bolt::cl::control &ctl,  const ForwardIterator &first,
                const ForwardIterator &last,
                const Generator &gen, const std::string &user_code, std::random_access_iterator_tag )
            {
                typedef typename std::iterator_traits<ForwardIterator>::value_type Type;

                int sz = static_cast<int>(last - first);
                if (sz < 1)
                    return;

                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if(runMode == bolt::cl::control::Automatic)
                {
                     runMode = ctl.getDefaultPathToRun();
                }
                #if defined(BOLT_DEBUG_LOG)
                BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
                #endif
				
                if( runMode == bolt::cl::control::SerialCpu)
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_GENERATE,BOLTLOG::BOLT_SERIAL_CPU,"::Generate::SERIAL_CPU");
                    #endif
						
                    std::generate(first, last, gen );
                }
                else if(runMode == bolt::cl::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB   
                           #if defined(BOLT_DEBUG_LOG)
                           dblog->CodePathTaken(BOLTLOG::BOLT_GENERATE,BOLTLOG::BOLT_MULTICORE_CPU,"::Generate::MULTICORE_CPU");
                           #endif				
                           bolt::btbb::generate(first, last, gen );
                    #else
                           throw std::runtime_error("MultiCoreCPU Version of generate not Enabled! \n");
                    #endif
                }
                else
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_GENERATE,BOLTLOG::BOLT_OPENCL_GPU,"::Generate::OPENCL_GPU");
                    #endif
						
                    // Use host pointers memory since these arrays are only write once - no benefit to copying.
                    // Map the forward iterator to a device_vector
                    device_vector< Type > range( first, sz, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, false, ctl );

                    generate_enqueue( ctl, range.begin( ), range.end( ), gen, user_code );

                    // This should immediately map/unmap the buffer
                    range.data( );
                }
}

            // This template is called by the non-detail versions of generate,
            // it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVForwardIterator, typename Generator>
            void generate_pick_iterator(bolt::cl::control &ctl, const DVForwardIterator &first,
                const DVForwardIterator &last,
                const Generator &gen, const std::string& user_code, bolt::cl::device_vector_tag )
            {
                typedef typename std::iterator_traits<DVForwardIterator>::value_type iType;
                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if(runMode == bolt::cl::control::Automatic)
                {
                     runMode = ctl.getDefaultPathToRun();
                }
                #if defined(BOLT_DEBUG_LOG)
                BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
                #endif
				
                if( runMode == bolt::cl::control::SerialCpu)
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_GENERATE,BOLTLOG::BOLT_SERIAL_CPU,"::Generate::SERIAL_CPU");
                    #endif
                    typename bolt::cl::device_vector< iType >::pointer generateInputBuffer =  first.getContainer( ).data( );
                    std::generate(&generateInputBuffer[first.m_Index], &generateInputBuffer[last.m_Index], gen );
                }
                else if(runMode == bolt::cl::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB
					  #if defined(BOLT_DEBUG_LOG)
                      dblog->CodePathTaken(BOLTLOG::BOLT_GENERATE,BOLTLOG::BOLT_MULTICORE_CPU,"::Generate::MULTICORE_CPU");
                      #endif	
                      typename bolt::cl::device_vector< iType >::pointer generateInputBuffer =  first.getContainer( ).data( );
                      bolt::btbb::generate(&generateInputBuffer[first.m_Index], &generateInputBuffer[last.m_Index], gen );
                    #else
                        throw std::runtime_error("MultiCoreCPU Version of generate not Enabled! \n");
                    #endif
                }
                else
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_GENERATE,BOLTLOG::BOLT_OPENCL_GPU,"::Generate::OPENCL_GPU");
                    #endif
                    generate_enqueue( ctl, first, last, gen, user_code );
                }
            }

            // This template is called by the non-detail versions of generate,
            // it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVForwardIterator, typename Generator>
            void generate_pick_iterator(bolt::cl::control &ctl,  const DVForwardIterator &first,
                const DVForwardIterator &last,
                const Generator &gen, const std::string& user_code, bolt::cl::fancy_iterator_tag )
            {
                static_assert( std::is_same< DVForwardIterator, bolt::cl::fancy_iterator_tag  >::value, "It is not possible to generate into fancy iterators. They are not mutable! " );
            }





/*****************************************************************************
             * Random Access
****************************************************************************/

// generate, yes random-access
template<typename ForwardIterator, typename Generator>
void generate_detect_random_access( bolt::cl::control &ctrl, const ForwardIterator& first, const ForwardIterator& last,
                        const Generator& gen, const std::string &cl_code, std::random_access_iterator_tag )
{
                generate_pick_iterator(ctrl, first, last, gen, cl_code,
                typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
}

// generate, not random-access
template<typename ForwardIterator, typename Generator>
void generate_detect_random_access( bolt::cl::control &ctrl, const ForwardIterator& first, const ForwardIterator& last,
                        const Generator& gen, const std::string &cl_code, std::forward_iterator_tag )
{
                static_assert(std::is_same< ForwardIterator, std::forward_iterator_tag   >::value, "Bolt only supports random access iterator types" );
}



}//End of detail namespace


// default control, start->stop
template<typename ForwardIterator, typename Generator>
void generate( ForwardIterator first, ForwardIterator last, Generator gen, const std::string& cl_code)
{
            detail::generate_detect_random_access( bolt::cl::control::getDefault(), first, last, gen, cl_code,
            typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
}

// user specified control, start->stop
template<typename ForwardIterator, typename Generator>
void generate( bolt::cl::control &ctl, ForwardIterator first, ForwardIterator last, Generator gen,
              const std::string& cl_code)
{
            detail::generate_detect_random_access( ctl, first, last, gen, cl_code,
            typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
}

// default control, start-> +n
template<typename OutputIterator, typename Size, typename Generator>
OutputIterator generate_n( OutputIterator first, Size n, Generator gen, const std::string& cl_code)
{
            detail::generate_detect_random_access( bolt::cl::control::getDefault(), first, first+static_cast< const int >( n ), gen, cl_code,
            typename std::iterator_traits< OutputIterator >::iterator_category( ) );
            return (first+static_cast< const int >( n ));
}

// user specified control, start-> +n
template<typename OutputIterator, typename Size, typename Generator>
OutputIterator generate_n( bolt::cl::control &ctl, OutputIterator first, Size n, Generator gen,
                          const std::string& cl_code)
{
            detail::generate_detect_random_access( ctl, first, first+static_cast< const int >( n ), gen, cl_code,
            typename std::iterator_traits< OutputIterator >::iterator_category( ) );
            return (first+static_cast< const int >( n ));
}

}//end of cl namespace
};//end of bolt namespace



#endif
