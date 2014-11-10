/***************************************************************************
* Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

***************************************************************************/

#pragma once
#if !defined( BOLT_CL_BINARY_SEARCH_INL )
#define BOLT_CL_BINARY_SEARCH_INL
#define BINARY_SEARCH_WAVEFRONT_SIZE 64
//#define BINARY_SEARCH_THRESHOLD 16

//TBB Includes
#if defined(ENABLE_TBB)
#include "bolt/btbb/binary_search.h"
#endif

namespace bolt {
    namespace cl {

        namespace detail {

        enum binarysearchTypeName { bs_iType, bs_T, bs_DVForwardIterator, bs_StrictWeakOrdering, bs_end };

        ///////////////////////////////////////////////////////////////////////
        //Kernel Template Specializer
        ///////////////////////////////////////////////////////////////////////

        class BinarySearch_KernelTemplateSpecializer : public KernelTemplateSpecializer
        {
            public:

            BinarySearch_KernelTemplateSpecializer() : KernelTemplateSpecializer()
            {
                addKernelName( "binarysearch_kernel" );
            }

            const ::std::string operator() ( const ::std::vector< ::std::string>& typeNames ) const
            {
                const std::string templateSpecializationString =
                    "// Dynamic specialization of generic template definition, using user supplied types\n"
                    "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
                    "__kernel void " + name(0) + "(\n"
                    "global " + typeNames[bs_iType] + " * src,\n"
                    + typeNames[bs_DVForwardIterator] + " input_iter,\n"
                    "const " + typeNames[bs_T] + " val,\n"
                    "const uint numElements,\n"
                    "global " + typeNames[bs_StrictWeakOrdering] + " * comp,\n"
                    "global int * result, "
                    "const uint startIndex, "
                    " const uint endIndex"
                    ");\n\n";

                return templateSpecializationString;
            }
        };



            /*****************************************************************************
             * BS Enqueue
             ****************************************************************************/
            template< typename DVForwardIterator, typename T , typename StrictWeakOrdering>
            bool binary_search_enqueue( bolt::cl::control &ctl, const DVForwardIterator &first,
                const DVForwardIterator &last, const T & val, StrictWeakOrdering comp, const std::string& cl_code)
            {

                // No. of I/P elements
                int numElementsProcessedperWI = 16; //For now choose 16
                int numOfWGs;
                int globalThreads;
                int localThreads;
                int residueGlobalThreads;
                int residueLocalThreads;
                int szElements = static_cast< cl_int >( std::distance( first, last ) );

                if(szElements < (BINARY_SEARCH_WAVEFRONT_SIZE*numElementsProcessedperWI) )
                {
                    numOfWGs = 1;
                    residueGlobalThreads = (szElements/numElementsProcessedperWI);
                    residueGlobalThreads = residueGlobalThreads + ( (szElements & (numElementsProcessedperWI-1) )? 1: 0 );
                    residueLocalThreads = residueGlobalThreads; //Because only 1 WG will be spawned
                    globalThreads = 0;
                    localThreads = 0;
                }
                else
                {
                    //Here you will definitely spawn more than BINARY_SEARCH_WAVEFRONT_SIZE work items.
                    globalThreads = (szElements/numElementsProcessedperWI);
                    globalThreads = globalThreads + ( (szElements & (numElementsProcessedperWI-1) )? 1: 0 ); //Create One extra thread if some buffer residue is left.
                    localThreads = BINARY_SEARCH_WAVEFRONT_SIZE;
                    residueGlobalThreads = globalThreads % localThreads;
                    residueLocalThreads = residueGlobalThreads;
                    globalThreads = globalThreads - residueGlobalThreads; // This makes globalThreads multiple of BINARY_SEARCH_WAVEFRONT_SIZE
                }
                /**********************************************************************************
                 * Type Names - used in KernelTemplateSpecializer
                 *********************************************************************************/
                typedef typename std::iterator_traits<DVForwardIterator>::value_type iType;

                std::vector<std::string> typeNames(bs_end);
				typeNames[bs_iType] = TypeName< iType >::get( );
                typeNames[bs_T] = TypeName< T >::get( );
                typeNames[bs_DVForwardIterator] = TypeName< DVForwardIterator >::get( );
                typeNames[bs_StrictWeakOrdering] = TypeName< StrictWeakOrdering >::get( );

                /**********************************************************************************
                 * Type Definitions - directly concatenated into kernel string (order may matter)
                 *********************************************************************************/
                std::vector<std::string> typeDefs;
                PUSH_BACK_UNIQUE( typeDefs, ClCode< iType >::get() )
				PUSH_BACK_UNIQUE( typeDefs, ClCode< T >::get() )
                PUSH_BACK_UNIQUE( typeDefs, ClCode< DVForwardIterator >::get() )
                PUSH_BACK_UNIQUE( typeDefs, ClCode< StrictWeakOrdering >::get() )

                cl_int l_Error = CL_SUCCESS;
                //--------------------------------------------------------------------------
                //Compile the Kernel
                std::string compileOptions;
                std::ostringstream oss;
                compileOptions = oss.str();

                BinarySearch_KernelTemplateSpecializer c_kts;
                std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
                    ctl,
                    typeNames,
                    &c_kts,
                    typeDefs,
                    binary_search_kernels,
                    compileOptions);
                int totalThreads = globalThreads+residueGlobalThreads;

                control::buffPointer result = ctl.acquireBuffer( sizeof( int ) * totalThreads,
                                                                CL_MEM_ALLOC_HOST_PTR|CL_MEM_WRITE_ONLY );
                ALIGNED( 256 ) StrictWeakOrdering aligned_comp( comp );

                control::buffPointer userFunctor = ctl.acquireBuffer( sizeof( aligned_comp ),
                                                          CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_comp );

                ::cl::Event kernelEvent, residueKernelEvent;
                try
                {
                    // Input buffer
                    int startIndex = 0;
                    int endIndex = static_cast< cl_uint >( numElementsProcessedperWI*globalThreads );
                    typename DVForwardIterator::Payload  bin_payload = first.gpuPayload( );
                    if(globalThreads != 0 )
                    {

                        V_OPENCL( kernels[0].setArg( 0, first.getContainer().getBuffer()),"Error setArg kernels[ 0 ]" );
                        // Input Iterator
                        V_OPENCL( kernels[0].setArg( 1, first.gpuPayloadSize( ),&bin_payload  ),
                            "Error setting a kernel argument" );
                        V_OPENCL( kernels[0].setArg( 2, val), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 3, static_cast<cl_uint>(numElementsProcessedperWI) ), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 4, *userFunctor), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 5, *result), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 6, startIndex), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 7, endIndex), "Error setArg kernels[ 0 ]" );
                        l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
                            kernels[0],
                            ::cl::NullRange,
                            ::cl::NDRange( globalThreads ),
                            ::cl::NDRange( localThreads ),
                            NULL,
                            &kernelEvent);
                        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel" );
                        bolt::cl::wait(ctl, kernelEvent);
                    }


                    startIndex = globalThreads*numElementsProcessedperWI;
                    endIndex = szElements;
                    if(residueGlobalThreads !=0)
                    {
                        V_OPENCL( kernels[0].setArg( 0, first.getContainer().getBuffer()),"Error setArg kernels[ 0 ]" );
                        // Input Iterator
                        V_OPENCL( kernels[0].setArg( 1, first.gpuPayloadSize( ), &bin_payload ),
                            "Error setting a kernel argument" );
                        V_OPENCL( kernels[0].setArg( 2, val), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 3, static_cast<cl_uint>(numElementsProcessedperWI) ), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 4, *userFunctor), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 5, *result), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 6, startIndex), "Error setArg kernels[ 0 ]" );
                        V_OPENCL( kernels[0].setArg( 7, endIndex), "Error setArg kernels[ 0 ]" );
                        l_Error = ctl.getCommandQueue( ).enqueueNDRangeKernel(
                            kernels[0],
                            ::cl::NullRange,
                            ::cl::NDRange( residueGlobalThreads ),
                            ::cl::NDRange( residueLocalThreads ),
                            NULL,
                            &residueKernelEvent);
                        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for kernel" );
                        bolt::cl::wait(ctl, residueKernelEvent);
                    }
                }
                catch( const ::cl::Error& e)
                {
                    std::cerr << "::cl::enqueueNDRangeKernel( ) in bolt::cl::binary_search_enqueue()" << std::endl;
                    std::cerr << "Error Code: " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
                    std::cerr << "File: " << __FILE__ << ", line " << __LINE__ << std::endl;
                    std::cerr << "Error String: " << e.what() << std::endl;
                }

                // wait for results
                ::cl::Event l_mapEvent;
                int *h_result = (int*)ctl.getCommandQueue().enqueueMapBuffer(*result, false, CL_MAP_READ, 0,
                    sizeof(int)* totalThreads, NULL, &l_mapEvent, &l_Error );
                V_OPENCL( l_Error, "Error calling map on the result buffer" );
                bolt::cl::wait(ctl, l_mapEvent);

                bool r = false;
                for(int i=0; i<totalThreads; i++)
                {
                    if(h_result[i] == 1)
                    {
                        r = true;
                        break;
                    }
                }

                ::cl::Event unmapEvent;
                V_OPENCL( ctl.getCommandQueue().enqueueUnmapMemObject(*result, h_result, NULL, &unmapEvent ),
                "shared_ptr failed to unmap host memory back to device memory" );
                V_OPENCL( unmapEvent.wait( ), "failed to wait for unmap event" );

                return r;

            }; // end binary_search_enqueue


            /*****************************************************************************
             * Pick Iterator
             ****************************************************************************/

        /*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
                \detail This template is called by the non-detail versions of binary_search, it already assumes random access
             * iterators. This overload is called strictly for non-device_vector iterators
        */

            template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
            bool binary_search_pick_iterator( bolt::cl::control &ctl, const ForwardIterator &first,
                const ForwardIterator &last, const T & value, StrictWeakOrdering comp, const std::string &user_code,
                std::random_access_iterator_tag )
            {

                typedef typename std::iterator_traits<ForwardIterator>::value_type Type;
                int sz = static_cast<int>(last - first);
                if (sz < 1)
                     return false;

                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode(); // could be dynamic choice some day.
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
                     dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_SERIAL_CPU,"::Binary_Search::SERIAL_CPU");
                     #endif
                     return std::binary_search(first, last, value, comp );
                }
                else if(runMode == bolt::cl::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB
					      #if defined(BOLT_DEBUG_LOG)
                          dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_MULTICORE_CPU,"::Binary_Search::MULTICORE_CPU");
                          #endif
                          return bolt::btbb::binary_search(first, last, value, comp);
                    #else
                          throw std::runtime_error("MultiCoreCPU Version of Binary Search not Enabled! \n");
                    #endif
                }
                else
                {
				        #if defined(BOLT_DEBUG_LOG)
                        dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_OPENCL_GPU,"::Binary_Search::OPENCL_GPU");
                        #endif
                        // Use host pointers memory since these arrays are only write once - no benefit to copying.
                        // Map the forward iterator to a device_vector
                        device_vector< Type > range( first, sz, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, false, ctl );

                        return binary_search_enqueue( ctl, range.begin( ), range.end( ), value, comp, user_code );

                }

            }

            // This template is called by the non-detail versions of BS, it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator

            template<typename DVForwardIterator, typename T , typename StrictWeakOrdering>
            bool binary_search_pick_iterator(bolt::cl::control &ctl, const DVForwardIterator &first,
                const DVForwardIterator &last, const T & value, StrictWeakOrdering comp, const std::string& user_code,
                bolt::cl::device_vector_tag )
            {
                typedef typename std::iterator_traits<DVForwardIterator>::value_type iType;
                int szElements = static_cast<int>(std::distance(first, last) );
                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode(); // could be dynamic choice some day.
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
                     dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_SERIAL_CPU,"::Binary_Search::SERIAL_CPU");
                     #endif
                     typename bolt::cl::device_vector< iType >::pointer bsInputBuffer = first.getContainer( ).data( );
                     return std::binary_search(&bsInputBuffer[first.m_Index], &bsInputBuffer[last.m_Index], value, comp );
                }
                else if(runMode == bolt::cl::control::MultiCoreCpu)
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_MULTICORE_CPU,"::Binary_Search::MULTICORE_CPU");
                    #endif
                    #ifdef ENABLE_TBB
                        typename bolt::cl::device_vector< iType >::pointer bsInputBuffer = first.getContainer( ).data( );
                        return bolt::btbb::binary_search(&bsInputBuffer[first.m_Index], &bsInputBuffer[last.m_Index], value, comp );
                    #else
                        throw std::runtime_error("MultiCoreCPU Version of Binary Search not Enabled! \n");
                    #endif
                }
                else
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_OPENCL_GPU,"::Binary_Search::OPENCL_GPU");
                    #endif
                    return binary_search_enqueue( ctl, first, last, value, comp, user_code );
                }
            }

            // This template is called by the non-detail versions of binary_search, it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator

            template<typename DVForwardIterator, typename T, typename StrictWeakOrdering>
            bool binary_search_pick_iterator(bolt::cl::control &ctl, const DVForwardIterator &first,
                const DVForwardIterator &last, const T & value, StrictWeakOrdering comp, const std::string& user_code,
                bolt::cl::fancy_iterator_tag )
            {

                typedef typename std::iterator_traits<DVForwardIterator>::value_type iType;
                int szElements = static_cast<int>(std::distance(first, last) );
                if (szElements == 0)
                    return false;

                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode(); // could be dynamic choice some day.
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
                     dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_SERIAL_CPU,"::Binary_Search::SERIAL_CPU");
                     #endif
                     return std::binary_search(first, last, value, comp);
                }
                else if(runMode == bolt::cl::control::MultiCoreCpu)
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_MULTICORE_CPU,"::Binary_Search::MULTICORE_CPU");
                    #endif
					
                    #ifdef ENABLE_TBB
                        return bolt::btbb::binary_search(first, last, value, comp);
                        //return std::binary_search(first, last, value, comp);
                    #else
                        throw std::runtime_error("MultiCoreCPU Version of Binary Search not Enabled! \n");
                    #endif
                }
                else
                {
				    #if defined(BOLT_DEBUG_LOG)
                    dblog->CodePathTaken(BOLTLOG::BOLT_BINARYSEARCH,BOLTLOG::BOLT_OPENCL_GPU,"::Binary_Search::OPENCL_GPU");
                    #endif
                    return binary_search_enqueue( ctl, first, last, value, comp, user_code );
                }

            }

            /*****************************************************************************
             * Random Access
             ****************************************************************************/


            // Random-access
            template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
            bool binary_search_detect_random_access( bolt::cl::control &ctl, ForwardIterator first,
                ForwardIterator last,
                const T & value, StrictWeakOrdering comp, const std::string &cl_code, std::random_access_iterator_tag )
            {
                 return binary_search_pick_iterator(ctl, first, last, value, comp, cl_code,
                 typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
            }


            // No support for non random access iterators
            template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
            bool binary_search_detect_random_access( bolt::cl::control &ctl, ForwardIterator first,
                ForwardIterator last,
                const T & value, StrictWeakOrdering comp, const std::string &cl_code, std::forward_iterator_tag )
            {
                static_assert( std::is_same< ForwardIterator, std::forward_iterator_tag   >::value, "Bolt only supports random access iterator types" );
            }

        }//End of detail namespace


        //Default control
        template<typename ForwardIterator, typename T>
        bool binary_search( ForwardIterator first,
            ForwardIterator last,
            const T & value,
            const std::string& cl_code)
        {
            return detail::binary_search_detect_random_access( bolt::cl::control::getDefault(), first, last, value,
                bolt::cl::less< T >( ), cl_code, typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        }

        //User specified control
        template< typename ForwardIterator, typename T >
        bool binary_search( bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const T & value,
            const std::string& cl_code)
        {
            return detail::binary_search_detect_random_access( ctl, first, last, value, bolt::cl::less< T >( ), cl_code,
                typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        }

        //Default control
        template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
        bool binary_search(ForwardIterator first,
            ForwardIterator last,
            const T & value,
            StrictWeakOrdering comp,
            const std::string& cl_code)
        {
            return detail::binary_search_detect_random_access( bolt::cl::control::getDefault(), first, last, value,
                comp, cl_code, typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        }

        //User specified control
        template<typename ForwardIterator, typename T, typename StrictWeakOrdering>
        bool binary_search(bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const T & value,
            StrictWeakOrdering comp,
            const std::string& cl_code)
        {
            return detail::binary_search_detect_random_access( ctl, first, last, value, comp, cl_code,
                typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        }

    }//end of cl namespace
};//end of bolt namespace



#endif
