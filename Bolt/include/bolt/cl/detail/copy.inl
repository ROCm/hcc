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

#if !defined( BOLT_CL_COPY_INL )
#define BOLT_CL_COPY_INL
#pragma once

#ifndef BURST_SIZE
#define BURST_SIZE 4
#endif

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/copy.h"
#endif

// bumps dividend up (if needed) to be evenly divisible by divisor
// returns whether dividend changed
// makeDivisible(9,4) -> 12,true
// makeDivisible(9,3) -> 9, false
template< typename Type1, typename Type2 >
bool makeDivisible( Type1& dividend, Type2 divisor)
{
    size_t lowerBits = static_cast<size_t>( dividend & (divisor-1) );
    if( lowerBits )
    { // bump it up
        dividend &= ~lowerBits;
        dividend += divisor;
        return true;
    }
    else
    { // already evenly divisible
      return false;
    }
}

// bumps dividend up (if needed) to be evenly divisible by divisor
// returns whether dividend changed
// roundUpDivide(9,4,?)  -> 12,4,3,true
// roundUpDivide(10,2,?) -> 10,2,5,false
template< typename Type1, typename Type2, typename Type3 >
bool roundUpDivide( Type1& dividend, Type2 divisor, Type3& quotient)
{
    size_t lowerBits = static_cast<size_t>( dividend & (divisor-1) );
    if( lowerBits )
    { // bump it up
        dividend &= ~lowerBits;
        dividend += divisor;
        quotient = dividend / divisor;
        return true;
    }
    else
    { // already evenly divisible
      quotient = dividend / divisor;
      return false;
    }
}

namespace bolt {
namespace cl {


namespace detail {

enum copyTypeName { copy_iType, copy_DVInputIterator, copy_oType, copy_DVOutputIterator, end_copy };

/**********************************************************************************************************************
 * Kernel Template Specializer
 *********************************************************************************************************************/
class Copy_KernelTemplateSpecializer : public KernelTemplateSpecializer
{
    public:

    Copy_KernelTemplateSpecializer() : KernelTemplateSpecializer()
        {
        addKernelName( "copy_I"     );
        addKernelName( "copy_II"    );
        addKernelName( "copy_III"   );
        addKernelName( "copy_IV"    );
        // addKernelName( "copy_V"     );
        }

    const ::std::string operator() ( const ::std::vector< ::std::string >& typeNames ) const
    {
        const std::string templateSpecializationString =
             "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(0) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
             + typeNames[copy_DVInputIterator] + " input_iter,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
             + typeNames[copy_DVOutputIterator] + " output_iter,\n"
            "const uint numElements"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(1) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(1) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(2) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(2) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"

            "// Dynamic specialization of generic template definition, using user supplied types\n"
            "template __attribute__((mangled_name(" + name(3) + "Instantiated)))\n"
            "__attribute__((reqd_work_group_size(256,1,1)))\n"
            "__kernel void " + name(3) + "(\n"
            "global " + typeNames[copy_iType] + " * restrict src,\n"
            "global " + typeNames[copy_oType] + " * restrict dst,\n"
            "const uint numElements\n"
            ");\n\n"
            ;

        return templateSpecializationString;
    }
};


template< typename DVInputIterator, typename Size, typename DVOutputIterator >
void copy_enqueue(const bolt::cl::control &ctrl, const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result, const std::string& cl_code="")
{

    /**********************************************************************************
     * Type Names - used in KernelTemplateSpecializer
     *********************************************************************************/
    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
    std::vector<std::string> typeNames(end_copy);
    typeNames[copy_iType] = TypeName< iType >::get( );
    typeNames[copy_DVInputIterator] = TypeName< DVInputIterator >::get( );
    typeNames[copy_oType] = TypeName< oType >::get( );
    typeNames[copy_DVOutputIterator] = TypeName< DVOutputIterator >::get( );

    /**********************************************************************************
     * Type Definitions - directly concatenated into kernel string (order may matter)
     *********************************************************************************/
    std::vector<std::string> typeDefs;
    PUSH_BACK_UNIQUE( typeDefs, ClCode< iType >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< DVInputIterator >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< oType >::get() )
    PUSH_BACK_UNIQUE( typeDefs, ClCode< DVOutputIterator >::get() )


    //kernelWithBoundsCheck.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( ctrl.device( ), &l_Error )
    const size_t workGroupSize  = 256;
    const size_t numComputeUnits = 40; //ctrl.device( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( ); // = 28
    const size_t numWorkGroupsPerComputeUnit = 10; //ctrl.wgPerComputeUnit( );
    const size_t numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;

    const cl_uint numThreadsIdeal = static_cast<cl_uint>( numWorkGroups * workGroupSize );
    cl_uint numElementsPerThread = n / numThreadsIdeal;
    cl_uint numThreadsRUP = n;
    size_t mod = (n & (workGroupSize-1));
    int doBoundaryCheck = 0;
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
    oss << " -DBURST_SIZE=" << BURST_SIZE;
    oss << " -DBOUNDARY_CHECK=" << doBoundaryCheck;
    compileOptions = oss.str();

    /**********************************************************************************
     * Request Compiled Kernels
     *********************************************************************************/
    Copy_KernelTemplateSpecializer c_kts;
    std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
        ctrl,
        typeNames,
        &c_kts,
        typeDefs,
        copy_kernels,
        compileOptions);

    /**********************************************************************************
     *  Kernel
     *********************************************************************************/
    ::cl::Event kernelEvent;
    cl_int l_Error;
    try
    {
        int whichKernel = 0;
        cl_uint numThreadsChosen;
        cl_uint workGroupSizeChosen = workGroupSize;
        switch( whichKernel )
            {
        case 0: // I: 1 thread per element
            numThreadsChosen = numThreadsRUP;
            break;
        case 1: // II: 1 element per thread / BURST_SIZE
            numThreadsChosen = numThreadsRUP / BURST_SIZE;
            break;
        case 2: // III: ideal threads
        case 3: // IV: ideal threads w/ BURST
        case 4: // V: ideal threads unrolled BURST
            numThreadsChosen = numThreadsIdeal;
            break;
        } // switch


        //std::cout << "NumElem: " << n << "; NumThreads: " << numThreadsChosen << ";
        //NumWorkGroups: " << numThreadsChosen/workGroupSizeChosen << std::endl;

        // Input buffer
         typename DVInputIterator::Payload first_payload = first.gpuPayload( );
         typename DVOutputIterator::Payload  result_payload = result.gpuPayload( );
        V_OPENCL( kernels[whichKernel].setArg( 0, first.getContainer().getBuffer()), "Error setArg kernels[ 0 ]" );
        V_OPENCL( kernels[whichKernel].setArg( 1, first.gpuPayloadSize( ),&first_payload), "Error setting a kernel argument" );
        // Output buffer
        V_OPENCL( kernels[whichKernel].setArg( 2, result.getContainer().getBuffer()),"Error setArg kernels[ 0 ]" );
        V_OPENCL( kernels[whichKernel].setArg( 3, result.gpuPayloadSize( ),&result_payload  ), "Error setting a kernel argument" );
        //Buffer Size
        V_OPENCL( kernels[whichKernel].setArg( 4, static_cast<cl_uint>( n ) ),"Error setArg kernels[0]" );


        l_Error = ctrl.getCommandQueue( ).enqueueNDRangeKernel(
            kernels[whichKernel],
            ::cl::NullRange,
            ::cl::NDRange( numThreadsChosen ),
            ::cl::NDRange( workGroupSizeChosen ),
            NULL,
            &kernelEvent);
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel" );
    }

    catch( const ::cl::Error& e)
    {
        std::cerr << "::cl::enqueueNDRangeKernel( ) in bolt::cl::copy_enqueue()" << std::endl;
        std::cerr << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
        std::cerr << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
        std::cerr << "Error String: " << e.what() << std::endl;
    }

    // wait for results
    bolt::cl::wait(ctrl, kernelEvent);

    // profiling
    cl_command_queue_properties queueProperties;
    l_Error = ctrl.getCommandQueue().getInfo<cl_command_queue_properties>(CL_QUEUE_PROPERTIES, &queueProperties);
    unsigned int profilingEnabled = queueProperties&CL_QUEUE_PROFILING_ENABLE;
    if ( profilingEnabled ) {
        cl_ulong start_time, stop_time;

        V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start_time),
            "failed on getProfilingInfo<CL_PROFILING_COMMAND_START>()");
        V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &stop_time),
            "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");
        cl_ulong time = stop_time -  start_time;
        double gb = (n*(sizeof(iType)+sizeof(oType))/1024.0/1024.0/1024.0);
        double sec = time/1000000000.0;
        std::cout << "Global Memory Bandwidth: " << ( gb / sec) << " ( "
          << time/1000000.0 << " ms)" << std::endl;
    }
}


/*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
                \detail This template is called by the non-detail versions of inclusive_scan, it already assumes
             *  random access iterators.  This overload is called strictly for non-device_vector iterators
            */
template<typename InputIterator, typename Size, typename OutputIterator>
void copy_pick_iterator(const bolt::cl::control &ctrl,  const InputIterator& first, const Size& n,
        const OutputIterator& result, const std::string& user_code, std::random_access_iterator_tag,
        std::random_access_iterator_tag )
{

    typedef typename  std::iterator_traits<InputIterator>::value_type iType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;


     bolt::cl::control::e_RunMode runMode = ctrl.getForceRunMode( );

     if( runMode == bolt::cl::control::Automatic )
     {
                runMode = ctrl.getDefaultPathToRun( );
     }
     #if defined(BOLT_DEBUG_LOG)
     BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
     #endif
				
     if( runMode == bolt::cl::control::SerialCpu )
     {
	     #if defined(BOLT_DEBUG_LOG)
         dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_SERIAL_CPU,"::Copy::SERIAL_CPU");
         #endif
         #if defined( _WIN32 )
           std::copy_n( first, n, stdext::checked_array_iterator<oType*>(&(*result), n ) );
         #else
           std::copy_n( first, n, result );
         #endif
     }
     else if( runMode == bolt::cl::control::MultiCoreCpu )
     {
        #ifdef ENABLE_TBB
		   #if defined(BOLT_DEBUG_LOG)
           dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_MULTICORE_CPU,"::Copy::MULTICORE_CPU");
           #endif
           bolt::btbb::copy_n(first, n, &(*result));
        #else
            throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
        #endif
     }
     else
     {
	    #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_OPENCL_GPU,"::Copy::OPENCL_GPU");
        #endif
		
        // A host 2 host copy operation, just fallback on the optimized std:: implementation
        #if defined( _WIN32 )
          std::copy_n( first, n, stdext::checked_array_iterator<oType*>(&(*result), n ) );
        #else
          std::copy_n( first, n, result );
        #endif
     }
}

/*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
                \detail This template is called by the non-detail versions of inclusive_scan, it already assumes
             *   random access iterators.  This overload is called strictly for non-device_vector iterators
            */
template<typename InputIterator, typename Size, typename OutputIterator>
void copy_pick_iterator(const bolt::cl::control &ctrl,  const InputIterator& first, const Size& n,
        const OutputIterator& result, const std::string& user_code, bolt::cl::fancy_iterator_tag,
        std::random_access_iterator_tag )
{

    typedef typename std::iterator_traits<InputIterator>::value_type iType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;

     bolt::cl::control::e_RunMode runMode = ctrl.getForceRunMode( );

     if( runMode == bolt::cl::control::Automatic )
     {
         runMode = ctrl.getDefaultPathToRun( );
     }
     #if defined(BOLT_DEBUG_LOG)
     BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
     #endif
	 
     if( runMode == bolt::cl::control::SerialCpu )
     {
	      #if defined(BOLT_DEBUG_LOG)
          dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_SERIAL_CPU,"::Copy::SERIAL_CPU");
          #endif
		 
          #if defined( _WIN32 )
           std::copy_n( first, n, stdext::checked_array_iterator<oType*>(&(*result), n ) );
          #else
           std::copy_n( first, n, result );
          #endif
     }
     else if( runMode == bolt::cl::control::MultiCoreCpu )
     {

         #ifdef ENABLE_TBB
		     #if defined(BOLT_DEBUG_LOG)
             dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_MULTICORE_CPU,"::Copy::MULTICORE_CPU");
             #endif
             bolt::btbb::copy_n(first, n, &(*result) );
         #else
               throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
         #endif
     }
     else
     {
	    #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_OPENCL_GPU,"::Copy::OPENCL_GPU");
        #endif
		
        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        // Map the output iterator to a device_vector
        device_vector< oType > dvOutput( result, n, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, false, ctrl );
        copy_enqueue( ctrl, first, n, dvOutput.begin( ), user_code );
        dvOutput.data();
     }
}

// This template is called by the non-detail versions of inclusive_scan, it already assumes random access iterators
// This is called strictly for iterators that are derived from device_vector< T >::iterator
template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator(const bolt::cl::control &ctrl,  const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result, const std::string& user_code, bolt::cl::device_vector_tag,
    bolt::cl::device_vector_tag )
{
    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
     bolt::cl::control::e_RunMode runMode = ctrl.getForceRunMode( );

     if( runMode == bolt::cl::control::Automatic )
     {
               runMode = ctrl.getDefaultPathToRun( );
     }

	 #if defined(BOLT_DEBUG_LOG)
     BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
     #endif
	 
     if( runMode == bolt::cl::control::SerialCpu )
     {
	        #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_SERIAL_CPU,"::Copy::SERIAL_CPU");
            #endif
		  
            typename bolt::cl::device_vector< iType >::pointer copySrc =  first.getContainer( ).data( );
            typename bolt::cl::device_vector< oType >::pointer copyDest =  result.getContainer( ).data( );
#if defined( _WIN32 )
            std::copy_n( &copySrc[first.m_Index], n, stdext::make_checked_array_iterator( &copyDest[result.m_Index], n) );
#else
            std::copy_n( &copySrc[first.m_Index], n, &copyDest[result.m_Index] );
#endif
            return;
     }
     else if( runMode == bolt::cl::control::MultiCoreCpu )
     {

         #ifdef ENABLE_TBB
		     #if defined(BOLT_DEBUG_LOG)
             dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_MULTICORE_CPU,"::Copy::MULTICORE_CPU");
             #endif
             typename bolt::cl::device_vector< iType >::pointer copySrc =  first.getContainer( ).data( );
             typename bolt::cl::device_vector< oType >::pointer copyDest =  result.getContainer( ).data( );
             bolt::btbb::copy_n( &copySrc[first.m_Index], n, &copyDest[result.m_Index] );
            return;
         #else
                throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
         #endif
     }
     else
     {
	     #if defined(BOLT_DEBUG_LOG)
         dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_OPENCL_GPU,"::Copy::OPENCL_GPU");
         #endif
		 
         copy_enqueue( ctrl, first, n, result, user_code );
     }
}

// This template is called by the non-detail versions of inclusive_scan, it already assumes random access iterators
// This is called strictly for iterators that are derived from device_vector< T >::iterator
template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator(const bolt::cl::control &ctrl,  const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result, const std::string& user_code, std::random_access_iterator_tag,
    bolt::cl::device_vector_tag )
{
    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
     bolt::cl::control::e_RunMode runMode = ctrl.getForceRunMode( );

     if( runMode == bolt::cl::control::Automatic )
     {
               runMode = ctrl.getDefaultPathToRun( );
     }

	 #if defined(BOLT_DEBUG_LOG)
     BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
     #endif
	 
     if( runMode == bolt::cl::control::SerialCpu )
     {
	        #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_SERIAL_CPU,"::Copy::SERIAL_CPU");
            #endif
			
            typename bolt::cl::device_vector< oType >::pointer copyDest =  result.getContainer( ).data( );
#if defined( _WIN32 )
            std::copy_n( first, n, stdext::make_checked_array_iterator( &copyDest[result.m_Index], n) );
#else
            std::copy_n( first, n, &copyDest[result.m_Index] );
#endif
            return;
     }
     else if( runMode == bolt::cl::control::MultiCoreCpu )
     {

         #ifdef ENABLE_TBB
		      #if defined(BOLT_DEBUG_LOG)
              dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_MULTICORE_CPU,"::Copy::MULTICORE_CPU");
              #endif
			 
              typename bolt::cl::device_vector< oType >::pointer copyDest =  result.getContainer( ).data( );
              bolt::btbb::copy_n( first, n, &copyDest[result.m_Index] );
            return;
         #else
                throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
         #endif
     }
     else
     {
	    #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_OPENCL_GPU,"::Copy::OPENCL_GPU");
        #endif
		 
        device_vector< iType > dvInput( first, n, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctrl );
        //Now call the actual cl algorithm
        copy_enqueue( ctrl, dvInput.begin(), n, result, user_code );
        //Map the buffer back to the host
        dvInput.data( );
     }
}

// This template is called by the non-detail versions of inclusive_scan, it already assumes random access iterators
// This is called strictly for iterators that are derived from device_vector< T >::iterator
template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator(const bolt::cl::control &ctrl,  const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result, const std::string& user_code, bolt::cl::device_vector_tag,
    std::random_access_iterator_tag)
{
    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
     bolt::cl::control::e_RunMode runMode = ctrl.getForceRunMode( );

     if( runMode == bolt::cl::control::Automatic )
     {
               runMode = ctrl.getDefaultPathToRun( );
     }

	 #if defined(BOLT_DEBUG_LOG)
     BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
     #endif
	 
     if( runMode == bolt::cl::control::SerialCpu )
     {
	     #if defined(BOLT_DEBUG_LOG)
         dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_SERIAL_CPU,"::Copy::SERIAL_CPU");
         #endif
			
         #if defined( _WIN32 )
           std::copy_n( first, n, stdext::checked_array_iterator<oType*>(&(*result), n ) );
         #else
           std::copy_n( first, n, result );
         #endif
           return;
     }
     else if( runMode == bolt::cl::control::MultiCoreCpu )
     {

           #ifdef ENABLE_TBB
		       #if defined(BOLT_DEBUG_LOG)
               dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_MULTICORE_CPU,"::Copy::MULTICORE_CPU");
               #endif
               typename bolt::cl::device_vector< iType >::pointer copySrc =  first.getContainer( ).data( );
               bolt::btbb::copy_n( &copySrc[first.m_Index], n, result );
           #else
                throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
           #endif
     }
     else
     {
	    #if defined(BOLT_DEBUG_LOG)
        dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_OPENCL_GPU,"::Copy::OPENCL_GPU");
        #endif
		
        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        // Map the output iterator to a device_vector
        device_vector< oType > dvOutput( result, n, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, false, ctrl );
        copy_enqueue( ctrl, first, n, dvOutput.begin( ), user_code );
        dvOutput.data();
     }
}

// This template is called by the non-detail versions of inclusive_scan, it already assumes random access iterators
// This is called strictly for iterators that are derived from device_vector< T >::iterator
template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator(const bolt::cl::control &ctrl,  const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result, const std::string& user_code, bolt::cl::fancy_iterator_tag,
    bolt::cl::device_vector_tag )
{
     typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
     typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
     bolt::cl::control::e_RunMode runMode = ctrl.getForceRunMode( );

     if( runMode == bolt::cl::control::Automatic )
     {
         runMode = ctrl.getDefaultPathToRun( );
     }
     
	 #if defined(BOLT_DEBUG_LOG)
     BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
     #endif
	 
     if( runMode == bolt::cl::control::SerialCpu )
     {
	     #if defined(BOLT_DEBUG_LOG)
         dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_SERIAL_CPU,"::Copy::SERIAL_CPU");
         #endif
		 
         typename bolt::cl::device_vector< oType >::pointer copyDest =  result.getContainer( ).data( );
#if defined( _WIN32 )
         std::copy_n( first, n, stdext::make_checked_array_iterator(&copyDest[result.m_Index], n) );
#else
         std::copy_n( first, n, &copyDest[result.m_Index] );
#endif
     }
     else if( runMode == bolt::cl::control::MultiCoreCpu )
     {
        #ifdef ENABLE_TBB
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_MULTICORE_CPU,"::Copy::MULTICORE_CPU");
            #endif
			   
            typename bolt::cl::device_vector< oType >::pointer copyDest =  result.getContainer( ).data( );
            bolt::btbb::copy_n( first, n, &copyDest[result.m_Index] );
            return;
        #else
              throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
        #endif
     }
     else
     {
	          #if defined(BOLT_DEBUG_LOG)
              dblog->CodePathTaken(BOLTLOG::BOLT_COPY,BOLTLOG::BOLT_OPENCL_GPU,"::Copy::OPENCL_GPU");
              #endif
		
              copy_enqueue( ctrl, first, n, result, user_code );
     }
}

template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator(const bolt::cl::control &ctl,  const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result, const std::string& user_code, std::random_access_iterator_tag,
    bolt::cl::fancy_iterator_tag )
{
    static_assert( std::is_same< DVInputIterator, bolt::cl::fancy_iterator_tag  >::value, "It is not possible to copy into fancy iterators. They are not mutable" );
}

template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator(const bolt::cl::control &ctl,  const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result, const std::string& user_code, bolt::cl::device_vector_tag,
    bolt::cl::fancy_iterator_tag )
{
    static_assert(std::is_same< DVInputIterator, bolt::cl::fancy_iterator_tag  >::value , "It is not possible to copy into fancy iterators. They are not mutable" );
}



template<typename InputIterator, typename Size, typename OutputIterator >
OutputIterator copy_detect_random_access( const bolt::cl::control& ctrl, const InputIterator& first, const Size& n,
                const OutputIterator& result, const std::string& user_code, std::random_access_iterator_tag )
{
    if (n < 0)
    {
        std::cout<<"\n Number of elements to copy cannot be negative! "<< std::endl;
    }
    if (n > 0)
    {
        copy_pick_iterator( ctrl, first, n, result, user_code,
            typename std::iterator_traits< InputIterator >::iterator_category( ),
            typename std::iterator_traits< OutputIterator >::iterator_category( ) );
    }
    return (result+n);
};

template<typename InputIterator, typename Size, typename OutputIterator >
OutputIterator copy_detect_random_access( const bolt::cl::control& ctrl, const InputIterator& first, const Size& n,
                const OutputIterator& result, const std::string& user_code, bolt::cl::device_vector_tag )
{
    if (n < 0)
    {
        std::cout<<"\n Number of elements to copy cannot be negative! "<< std::endl;
    }
    if (n > 0)
    {
        copy_pick_iterator( ctrl, first, n, result, user_code,
             typename std::iterator_traits< InputIterator >::iterator_category( ),
             typename std::iterator_traits< OutputIterator >::iterator_category( ) );
    }
    return (result+n);
};

// Wrapper that uses default control class, iterator interface
template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_detect_random_access( const bolt::cl::control& ctrl, const InputIterator& first, const Size& n,
                const OutputIterator& result, const std::string& user_code, std::input_iterator_tag )
{
    static_assert( std::is_same< InputIterator, bolt::cl::input_iterator_tag  >::value, "Bolt only supports random access iterator types" );
    return NULL;
};

}//End OF detail namespace



// user control
template<typename InputIterator, typename OutputIterator>
OutputIterator copy(const bolt::cl::control &ctrl,  InputIterator first, InputIterator last, OutputIterator result,
            const std::string& user_code)
{
    int n = static_cast<int>( std::distance( first, last ) );
    return detail::copy_detect_random_access( ctrl, first, n, result, user_code,
         typename std::iterator_traits< InputIterator >::iterator_category( ) );
}

// default control
template<typename InputIterator, typename OutputIterator>
OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result,
            const std::string& user_code)
{
    int n = static_cast<int>( std::distance( first, last ) );
            return detail::copy_detect_random_access( control::getDefault(), first, n, result, user_code,
                typename std::iterator_traits< InputIterator >::iterator_category( ) );
}

// default control
template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(InputIterator first, Size n, OutputIterator result,
            const std::string& user_code)
{
            return detail::copy_detect_random_access( control::getDefault(), first, n, result, user_code,
                typename std::iterator_traits< InputIterator >::iterator_category( ) );
}

// user control
template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(const bolt::cl::control &ctrl, InputIterator first, Size n, OutputIterator result,
            const std::string& user_code)
{
    return detail::copy_detect_random_access( ctrl, first, n, result, user_code,
                    typename std::iterator_traits< InputIterator >::iterator_category( ) );
}

}//end of cl namespace
};//end of bolt namespace






#endif
