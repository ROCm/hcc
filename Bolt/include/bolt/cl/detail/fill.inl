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
#if !defined( BOLT_CL_FILL_INL )
#define BOLT_CL_FILL_INL
#define WAVEFRONT_SIZE 64

//TBB Includes
#ifdef ENABLE_TBB
#include "bolt/btbb/fill.h"
#endif

namespace bolt {
    namespace cl {


namespace detail {
        enum fillTypeName { fill_T, fill_Type, fill_DVInputIterator, fill_end };

        ///////////////////////////////////////////////////////////////////////
        //Kernel Template Specializer
        ///////////////////////////////////////////////////////////////////////
        class Fill_KernelTemplateSpecializer : public KernelTemplateSpecializer
        {
            public:

            Fill_KernelTemplateSpecializer() : KernelTemplateSpecializer()
                {
                addKernelName( "fill_kernel" );
                }

            const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
            {
                const std::string templateSpecializationString =
                    "// Dynamic specialization of generic template definition, using user supplied types\n"
                    "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
                    "__attribute__((reqd_work_group_size(64,1,1)))\n"
                    "__kernel void " + name(0) + "(\n"
                    "const " + typeNames[fill_T] + " src,\n"
                    "global " + typeNames[fill_Type] + " * dst,\n"
                     + typeNames[fill_DVInputIterator] + " input_iter,\n"
                    "const uint numElements\n"
                    ");\n\n";

                return templateSpecializationString;
            }
        };

            /*****************************************************************************
             * Fill Enqueue
             ****************************************************************************/

            template< typename DVForwardIterator, typename T >
            void fill_enqueue(const bolt::cl::control &ctl, const DVForwardIterator &first,
                const DVForwardIterator &last, const T & val, const std::string& cl_code)
            {
                // how many elements to fill
                cl_uint sz = static_cast< cl_uint >( std::distance( first, last ) );
                if (sz < 1)
                    return;

                /**********************************************************************************
                 * Type Names - used in KernelTemplateSpecializer
                 *********************************************************************************/
                typedef typename std::iterator_traits<DVForwardIterator>::value_type Type;
                typedef T iType;
                std::vector<std::string> typeNames(fill_end);
                typeNames[fill_T] = TypeName< T >::get( );
                typeNames[fill_Type] = TypeName< Type >::get( );
                typeNames[fill_DVInputIterator] = TypeName< DVForwardIterator >::get( );

                /**********************************************************************************
                 * Type Definitions - directrly concatenated into kernel string (order may matter)
                 *********************************************************************************/
                std::vector<std::string> typeDefs;
                PUSH_BACK_UNIQUE( typeDefs, ClCode< iType >::get() )
                PUSH_BACK_UNIQUE( typeDefs, ClCode< Type >::get() )
                PUSH_BACK_UNIQUE( typeDefs, ClCode< DVForwardIterator >::get() )

                cl_int l_Error = CL_SUCCESS;
                const size_t workGroupSize  = WAVEFRONT_SIZE;
                const size_t numComputeUnits = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( ); // = 28
                const size_t numWorkGroupsPerComputeUnit = ctl.getWGPerComputeUnit( );
                const size_t numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;

                const cl_uint numThreadsIdeal = static_cast<cl_uint>( numWorkGroups * workGroupSize );
                cl_uint numElementsPerThread = sz/ numThreadsIdeal;
                cl_uint numThreadsRUP = sz;
                size_t mod = (sz& (workGroupSize-1));
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
                oss << " -DBOUNDARY_CHECK=" << doBoundaryCheck;
                compileOptions = oss.str();

                /**********************************************************************************
                 * Request Compiled Kernels
                 *********************************************************************************/
                Fill_KernelTemplateSpecializer c_kts;
                std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
                    ctl,
                    typeNames,
                    &c_kts,
                    typeDefs,
                    fill_kernels,
                    compileOptions);

                /**********************************************************************************
                 *  Kernel
                 *********************************************************************************/
                ::cl::Event kernelEvent;
                typename DVForwardIterator::Payload  first_payload = first.gpuPayload( );
                try
                {
                    cl_uint numThreadsChosen;
                    cl_uint workGroupSizeChosen = workGroupSize;
                    numThreadsChosen = numThreadsRUP;

                    //std::cout << "NumElem: " << sz<< "; NumThreads: " << numThreadsChosen << ";
                    //NumWorkGroups: " << numThreadsChosen/workGroupSizeChosen << std::endl;

                    // Input Value
                    V_OPENCL( kernels[0].setArg( 0, val), "Error setArg kernels[ 0 ]" );
                    // Fill buffer
                    V_OPENCL( kernels[0].setArg( 1, first.getContainer().getBuffer()),"Error setArg kernels[ 0 ]" );
                    // Input Iterator
                    V_OPENCL( kernels[0].setArg( 2, first.gpuPayloadSize( ),&first_payload ),
                        "Error setting a kernel argument" );
                    // Size of buffer
                    V_OPENCL( kernels[0].setArg( 3, static_cast<cl_uint>( sz) ), "Error setArg kernels[ 0 ]" );

                    l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
                        kernels[0],
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
                bolt::cl::wait(ctl, kernelEvent);


                // profiling
                cl_command_queue_properties queueProperties;
                l_Error = ctl.getCommandQueue().getInfo<cl_command_queue_properties>(CL_QUEUE_PROPERTIES,
                    &queueProperties);
                unsigned int profilingEnabled = queueProperties&CL_QUEUE_PROFILING_ENABLE;
                if ( profilingEnabled ) {
                    cl_ulong start_time, stop_time;

                    V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start_time),
                        "failed on getProfilingInfo<CL_PROFILING_COMMAND_START>()");
                    V_OPENCL( kernelEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END,    &stop_time),
                        "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");
                    cl_ulong time =stop_time - start_time;
                    double gb = (sz*(sizeof(T)+sizeof(Type))/1024.0/1024.0/1024.0);
                    double sec = time/1000000000.0;
                    std::cout << "Global Memory Bandwidth: " << ( gb / sec) << " ( "
                      << time/1000000.0 << " ms)" << std::endl;
                }

            }; // end fill_enqueue



            /*****************************************************************************
             * Pick Iterator
             ****************************************************************************/

        /*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
                \detail This template is called by the non-detail versions of fill, it already assumes random access
             *  iterators.  This overload is called strictly for non-device_vector iterators
        */
            template<typename ForwardIterator, typename T>
            void fill_pick_iterator(const bolt::cl::control &ctl,  const ForwardIterator &first,
                const ForwardIterator &last, const T & value, const std::string &user_code,
                std::random_access_iterator_tag )
            {


                typedef typename  std::iterator_traits<ForwardIterator>::value_type Type;

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
                     dblog->CodePathTaken(BOLTLOG::BOLT_FILL,BOLTLOG::BOLT_SERIAL_CPU,"::Fill::SERIAL_CPU");
                     #endif
						
                     std::fill(first, last, value );
                }
                else if(runMode == bolt::cl::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB
					      #if defined(BOLT_DEBUG_LOG)
                          dblog->CodePathTaken(BOLTLOG::BOLT_FILL,BOLTLOG::BOLT_MULTICORE_CPU,"::Fill::MULTICORE_CPU");
                          #endif
                          bolt::btbb::fill(first, last, value);
                    #else
                          throw std::runtime_error("MultiCoreCPU Version of fill not Enabled! \n");
                    #endif
                }
                else
                {
				        #if defined(BOLT_DEBUG_LOG)
                        dblog->CodePathTaken(BOLTLOG::BOLT_FILL,BOLTLOG::BOLT_OPENCL_GPU,"::Fill::OPENCL_GPU");
                        #endif
                        // Use host pointers memory since these arrays are only write once - no benefit to copying.
                        // Map the forward iterator to a device_vector
                        device_vector< Type > range( first, sz, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, false, ctl );

                        fill_enqueue( ctl, range.begin( ), range.end( ), value, user_code );

                        range.data( );
                }

            }

            // This template is called by the non-detail versions of fill, it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVForwardIterator, typename T>
            void fill_pick_iterator(const bolt::cl::control &ctl,  const DVForwardIterator &first,
                const DVForwardIterator &last,  const T & value, const std::string& user_code,
                bolt::cl::device_vector_tag )
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
                    dblog->CodePathTaken(BOLTLOG::BOLT_FILL,BOLTLOG::BOLT_SERIAL_CPU,"::Fill::SERIAL_CPU");
                    #endif
                    typename bolt::cl::device_vector< iType >::pointer fillInputBuffer =  first.getContainer( ).data( );
                    std::fill(&fillInputBuffer[first.m_Index], &fillInputBuffer[last.m_Index], value );
                }
                else if(runMode == bolt::cl::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB
					    #if defined(BOLT_DEBUG_LOG)
                        dblog->CodePathTaken(BOLTLOG::BOLT_FILL,BOLTLOG::BOLT_MULTICORE_CPU,"::Fill::MULTICORE_CPU");
                        #endif
                        typename bolt::cl::device_vector< iType >::pointer fillInputBuffer =  first.getContainer( ).data( );
                        bolt::btbb::fill(&fillInputBuffer[first.m_Index], &fillInputBuffer[last.m_Index], value );
                    #else
                           throw std::runtime_error("MultiCoreCPU Version of fill not Enabled! \n");
                    #endif
                }
                else
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_FILL,BOLTLOG::BOLT_OPENCL_GPU,"::Fill::OPENCL_GPU");
                    #endif
                    fill_enqueue( ctl, first, last, value, user_code );
                }
            }

            // This template is called by the non-detail versions of fill, it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVForwardIterator, typename T>
            void fill_pick_iterator(const bolt::cl::control &ctl,  const DVForwardIterator &first,
                const DVForwardIterator &last,  const T & value, const std::string& user_code,
                bolt::cl::fancy_iterator_tag )
            {

                static_assert( std::is_same< DVForwardIterator, bolt::cl::fancy_iterator_tag  >::value, "It is not possible to fill into fancy iterators. They are not mutable! \n" );
            }



            /*****************************************************************************
             * Random Access
             ****************************************************************************/

            // fill no support
            template<typename ForwardIterator, typename T>
            void fill_detect_random_access( const bolt::cl::control &ctl, ForwardIterator first, ForwardIterator last,
                const T & value, const std::string &cl_code, std::forward_iterator_tag )
            {
                static_assert( std::is_same< ForwardIterator, std::forward_iterator_tag   >::value , "Bolt only supports random access iterator types" );
            }

            // fill random-access
            template<typename ForwardIterator, typename T>
            void fill_detect_random_access( const bolt::cl::control &ctl, ForwardIterator first, ForwardIterator last,
                const T & value, const std::string &cl_code, std::random_access_iterator_tag )
            {
                     fill_pick_iterator(ctl, first, last, value, cl_code,
                    typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
            }


        }

        // default control, start->stop
        template< typename ForwardIterator, typename T >
        void fill( ForwardIterator first,
            ForwardIterator last,
            const T & value,
            const std::string& cl_code )
        {
            detail::fill_detect_random_access( bolt::cl::control::getDefault(), first, last, value, cl_code,
                typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        }

        // user specified control, start->stop
        template< typename ForwardIterator, typename T >
        void fill( const bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const T & value,
            const std::string& cl_code )
        {
            detail::fill_detect_random_access( ctl, first, last, value, cl_code,
                typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        }

        // default control, start-> +n
        template< typename OutputIterator, typename Size, typename T >
        OutputIterator fill_n( OutputIterator first,
            Size n,
            const T & value,
            const std::string& cl_code )
        {
            detail::fill_detect_random_access( bolt::cl::control::getDefault(),
                first, first+static_cast< const int >( n ),
                value, cl_code, typename std::iterator_traits< OutputIterator >::iterator_category( ) );
            return first+static_cast< const int >( n );
        }

        // user specified control, start-> +n
        template<typename OutputIterator, typename Size, typename T>
        OutputIterator fill_n( const bolt::cl::control &ctl,
            OutputIterator first,
            Size n,
            const T & value,
            const std::string& cl_code )
        {
            detail::fill_detect_random_access( ctl, first, first+static_cast< const int >( n ), value, cl_code,
                typename std::iterator_traits< OutputIterator >::iterator_category( ) );
            return (first+static_cast< const int >( n ));
        }

    }//end of cl namespace
};//end of bolt namespace


#endif
