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


#if !defined( BOLT_CL_MINELEMENT_INL )
#define BOLT_CL_MINELEMENT_INL
#pragma once

#include "bolt/cl/functional.h"

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/min_element.h"
#endif

namespace bolt {
    namespace cl {
        namespace detail {



        enum MineleTypes {min_iValueType, min_iIterType, min_BinaryPredicate, min_end };

        ///////////////////////////////////////////////////////////////////////
        //Kernel Template Specializer
        ///////////////////////////////////////////////////////////////////////
        class Min_KernelTemplateSpecializer : public KernelTemplateSpecializer
            {
            public:

            Min_KernelTemplateSpecializer() : KernelTemplateSpecializer()
                {
                    addKernelName( "min_elementTemplate" );
                }

            const ::std::string operator() ( const ::std::vector< ::std::string >& typeNames ) const
            {
                const std::string templateSpecializationString =
                        "// Host generates this instantiation string with user-specified value type and functor\n"
                        "template __attribute__((mangled_name(" + name(0) + "Instantiated)))\n"
                        "__attribute__((reqd_work_group_size(256,1,1)))\n"
                        "kernel void " + name(0) + "(\n"
                        "global " + typeNames[min_iValueType] + "* input_ptr,\n"
                         + typeNames[min_iIterType] + " output_iter,\n"
                        "const int length,\n"
                        "global " + typeNames[min_BinaryPredicate] + "* userFunctor,\n"
                        "global int *result,\n"
                        "local " + typeNames[min_iValueType] + "* scratch,\n"
                        "local int *scratch_index\n"
                        ");\n\n";

                return templateSpecializationString;
            }
            };

            //----
            // This is the base implementation of reduction that is called by all of the convenience wrappers below.
            // first and last must be iterators from a DeviceVector
            template<typename DVInputIterator, typename BinaryPredicate>
            int min_element_enqueue(bolt::cl::control &ctl,
                const DVInputIterator& first,
                const DVInputIterator& last,
                const BinaryPredicate& binary_op,
                const std::string& cl_code,
                const char * min_max )
            {
                typedef typename std::iterator_traits< DVInputIterator >::value_type iType;

                std::vector<std::string> typeNames( min_end);
                typeNames[min_iValueType] = TypeName< iType >::get( );
                typeNames[min_iIterType] = TypeName< DVInputIterator >::get( );
                typeNames[min_BinaryPredicate] = TypeName< BinaryPredicate >::get();

                std::vector<std::string> typeDefinitions;
                PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType >::get() )
                PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVInputIterator >::get() )
                PUSH_BACK_UNIQUE( typeDefinitions, ClCode< BinaryPredicate  >::get() )

                //bool cpuDevice = ctl.device().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
                /*\TODO - Do CPU specific kernel work group size selection here*/
                //const size_t kernel0_WgSize = (cpuDevice) ? 1 : WAVESIZE*KERNEL02WAVES;
                std::string compileOptions="";

                const char * str = "MAX_KERNEL";

                if(std::strcmp(min_max,str) == 0)
                    compileOptions = "-D_IS_MAX_KERNEL";
                else
                    compileOptions = "-D_IS_MIN_KERNEL";

                //std::ostringstream oss;
                //oss << " -DKERNEL0WORKGROUPSIZE=" << kernel0_WgSize;

                Min_KernelTemplateSpecializer ts_kts;
                std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
                    ctl,
                    typeNames,
                    &ts_kts,
                    typeDefinitions,
                    min_element_kernels,
                    compileOptions);


                // Set up shape of launch grid and buffers:
                cl_uint computeUnits     = ctl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                int wgPerComputeUnit =  64 ;//ctl.getWGPerComputeUnit();
                size_t numWG = computeUnits * wgPerComputeUnit;

                cl_int l_Error = CL_SUCCESS;

                const size_t wgSize  = 256;//kernels[0].getWorkGroupInfo< CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >(
                    //ctl.getDevice( ), &l_Error );

                V_OPENCL( l_Error, "Error querying kernel for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" );

                // Create buffer wrappers so we can access the host functors, for read or writing in the kernel
                ALIGNED( 256 ) BinaryPredicate aligned_reduce( binary_op );
                //::cl::Buffer userFunctor(ctl.context(), CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,sizeof(aligned_reduce),
                //  &aligned_reduce );
                control::buffPointer userFunctor = ctl.acquireBuffer( sizeof( aligned_reduce ),
                    CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_reduce );

                // ::cl::Buffer result(ctl.context(), CL_MEM_ALLOC_HOST_PTR|CL_MEM_WRITE_ONLY, sizeof( iType )*numWG);
                control::buffPointer result = ctl.acquireBuffer( sizeof( int ) * numWG,
                    CL_MEM_ALLOC_HOST_PTR|CL_MEM_WRITE_ONLY );

                cl_uint szElements = static_cast< cl_uint >( first.distance_to(last ) );
                typename DVInputIterator::Payload first_payload = first.gpuPayload();

                V_OPENCL( kernels[0].setArg(0, first.getContainer().getBuffer() ), "Error setting kernel argument" );
                V_OPENCL( kernels[0].setArg(1,first.gpuPayloadSize(),&first_payload),"Error setting a kernel argument");

                V_OPENCL( kernels[0].setArg(2, szElements), "Error setting kernel argument" );
                V_OPENCL( kernels[0].setArg(3, *userFunctor), "Error setting kernel argument" );
                V_OPENCL( kernels[0].setArg(4, *result), "Error setting kernel argument" );

                ::cl::LocalSpaceArg loc,loc2;
                loc.size_ = wgSize*sizeof(iType);
                loc2.size_ = wgSize*sizeof(int);;
                V_OPENCL( kernels[0].setArg(5, loc), "Error setting kernel argument" );
                V_OPENCL( kernels[0].setArg(6, loc2), "Error setting kernel argument" );


                l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
                    kernels[0],
                    ::cl::NullRange,
                    ::cl::NDRange(numWG * wgSize),
                    ::cl::NDRange(wgSize));
                V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for reduce() kernel" );

                ::cl::Event l_mapEvent;
                int *h_result = (int*)ctl.getCommandQueue().enqueueMapBuffer(*result, false, CL_MAP_READ, 0,
                    sizeof(int)*numWG, NULL, &l_mapEvent, &l_Error );
                V_OPENCL( l_Error, "Error calling map on the result buffer" );

                // Finish the tail end of the reduction on host side; the compute device reduces within the workgroups,
                // with one result per workgroup
                size_t ceilNumWG = static_cast< size_t >( std::ceil( static_cast< float >( szElements ) / wgSize) );
                bolt::cl::minimum<size_t>  min_size_t;
                size_t numTailReduce = min_size_t( ceilNumWG, numWG );

                bolt::cl::wait(ctl, l_mapEvent);

                int minele_indx =  h_result[0] ;
                iType minele =  *(first + h_result[0]) ;


				if(std::strcmp(min_max,str) == 0)
                {
					for(int i = 1; i < (int)numTailReduce; ++i){
                       bool stat = binary_op(*(first + h_result[i]), minele);
                       minele = stat ? minele : *(first + h_result[i]);
                       minele_indx =  stat ? minele_indx : h_result[i];
					}
                }
                else
                {
					for(int i = 1; i < (int)numTailReduce; ++i){
                       bool stat = binary_op(minele,*(first + h_result[i]));
                       minele = stat ? minele : *(first + h_result[i]);
                       minele_indx =  stat ? minele_indx : h_result[i];
					}
                }

				::cl::Event unmapEvent;

				V_OPENCL( ctl.getCommandQueue().enqueueUnmapMemObject(*result,  h_result, NULL, &unmapEvent ),
					"shared_ptr failed to unmap host memory back to device memory" );
				V_OPENCL( unmapEvent.wait( ), "failed to wait for unmap event" );


                return minele_indx;
            }



            // This template is called after we detect random access iterators
            // This is called strictly for any non-device_vector iterator
            template<typename ForwardIterator, typename BinaryPredicate>
            ForwardIterator min_element_pick_iterator(bolt::cl::control &ctl,
                const ForwardIterator& first,
                const ForwardIterator& last,
                const BinaryPredicate& binary_op,
                const std::string& cl_code,
                std::random_access_iterator_tag,
                const char * min_max)
            {
                /*************/

                typedef typename std::iterator_traits<ForwardIterator>::value_type iType;
                size_t szElements = (size_t)(last - first);
                if (szElements == 0)
                    return last;
                /*TODO - probably the forceRunMode should be replaced by getRunMode and setRunMode*/
                // Its a dynamic choice. See the reduce Test Code
                // What should we do if the run mode is automatic. Currently it goes to the last else statement
                //How many threads we should spawn?
                //Need to look at how to control the number of threads spawned.

                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if(runMode == bolt::cl::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }

                const char * str = "MAX_KERNEL";

				#if defined(BOLT_DEBUG_LOG)
                BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
                #endif
				
                switch(runMode)
                {
                case bolt::cl::control::OpenCL :
                    {
					    #if defined(BOLT_DEBUG_LOG)
						if(std::strcmp(min_max,str) == 0)
                           dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_OPENCL_GPU,"::Max_Element::OPENCL_GPU");
						else
						   dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_OPENCL_GPU,"::Min_Element::OPENCL_GPU");
                        #endif
						
                        device_vector< iType > dvInput( first, last, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, ctl );
                        int  dvminele = min_element_enqueue( ctl, dvInput.begin(), dvInput.end(), binary_op, cl_code, min_max);
                        return first + dvminele ;
                    }

                case bolt::cl::control::MultiCoreCpu:
                    #ifdef ENABLE_TBB
					    #if defined(BOLT_DEBUG_LOG)
						if(std::strcmp(min_max,str) == 0)
                          dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_MULTICORE_CPU,"::Max_Element::MULTICORE_CPU");
						else
						  dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_MULTICORE_CPU,"::Min_Element::MULTICORE_CPU");
                        #endif
                        if(std::strcmp(min_max,str) == 0)
                              return bolt::btbb::max_element(first, last, binary_op);
                        else
                              return bolt::btbb::min_element(first, last, binary_op);
                    #else
                        throw std::runtime_error( "The MultiCoreCpu version of Max-Min is not enabled to be built! \n" );
                    #endif

                case bolt::cl::control::SerialCpu:
				    #if defined(BOLT_DEBUG_LOG)
					if(std::strcmp(min_max,str) == 0)
					  dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Max_Element::SERIAL_CPU");
					else
                      dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Min_Element::SERIAL_CPU");
                    #endif
                    if(std::strcmp(min_max,str) == 0)
                       return std::max_element(first, last, binary_op);
                    else
                    return std::min_element(first, last, binary_op);

                default:
                    #if defined(BOLT_DEBUG_LOG)
					if(std::strcmp(min_max,str) == 0)
					  dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Max_Element::SERIAL_CPU");
					else
                      dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Min_Element::SERIAL_CPU");
                    #endif
                    if(std::strcmp(min_max,str) == 0)
                       return std::max_element(first, last, binary_op);
                    else
                    return std::min_element(first, last, binary_op);

                }

            };

            // This template is called after we detect random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVInputIterator, typename BinaryPredicate>
            DVInputIterator min_element_pick_iterator(bolt::cl::control &ctl,
                const DVInputIterator& first,
                const DVInputIterator& last,
                const BinaryPredicate& binary_op,
                const std::string& cl_code,
                bolt::cl::device_vector_tag,
                const char * min_max )
            {
                typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
				size_t szElements = (size_t)(last - first);
                if (szElements == 0)
                    return last;

                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if(runMode == bolt::cl::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }

                const char * str = "MAX_KERNEL";
            
			    #if defined(BOLT_DEBUG_LOG)
                BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
                #endif
				
                switch(runMode)
                {
                case bolt::cl::control::OpenCL :
                    {
					    #if defined(BOLT_DEBUG_LOG)
						if(std::strcmp(min_max,str) == 0)
                           dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_OPENCL_GPU,"::Max_Element::OPENCL_GPU");
						else
						   dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_OPENCL_GPU,"::Min_Element::OPENCL_GPU");
                        #endif
						
                        int pos =  min_element_enqueue( ctl, first, last,  binary_op, cl_code, min_max);
                        return first+pos;
                    }

                case bolt::cl::control::MultiCoreCpu:
					{
                    #ifdef ENABLE_TBB
					    #if defined(BOLT_DEBUG_LOG)
						if(std::strcmp(min_max,str) == 0)
                          dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_MULTICORE_CPU,"::Max_Element::MULTICORE_CPU");
						else
						  dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_MULTICORE_CPU,"::Min_Element::MULTICORE_CPU");
                        #endif
						typename bolt::cl::device_vector< iType >::pointer InputBuffer =  first.getContainer( ).data( );
						iType* stlPtr;
                        if(std::strcmp(min_max,str) == 0)
                              stlPtr = bolt::btbb::max_element(&InputBuffer[first.m_Index], &InputBuffer[last.m_Index], binary_op);
                        else
                              stlPtr = bolt::btbb::min_element(&InputBuffer[first.m_Index], &InputBuffer[last.m_Index], binary_op);
						return first+(unsigned int)(stlPtr-&InputBuffer[first.m_Index]);
                    #else
                        throw std::runtime_error( "The MultiCoreCpu version of Max-Min is not enabled to be built! \n" );
                    #endif
					}
                 case bolt::cl::control::SerialCpu:
					 {
				    #if defined(BOLT_DEBUG_LOG)
						if(std::strcmp(min_max,str) == 0)
						  dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Max_Element::SERIAL_CPU");
						else
						  dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Min_Element::SERIAL_CPU");
					#endif
						typename bolt::cl::device_vector< iType >::pointer InputBuffer =  first.getContainer( ).data( );
						iType* stlPtr;
						if(std::strcmp(min_max,str) == 0)
						   stlPtr = std::max_element(&InputBuffer[first.m_Index], &InputBuffer[last.m_Index], binary_op);
						else
						   stlPtr = std::min_element(&InputBuffer[first.m_Index], &InputBuffer[last.m_Index], binary_op);
						return first+(unsigned int)(stlPtr-&InputBuffer[first.m_Index]);
					 }
                default:
					{
					#if defined(BOLT_DEBUG_LOG)
						if(std::strcmp(min_max,str) == 0)
						  dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Max_Element::SERIAL_CPU");
						else
						  dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Min_Element::SERIAL_CPU");
					#endif
						if(std::strcmp(min_max,str) == 0)
						   return std::max_element(first, last, binary_op);
						else
						return std::min_element(first, last, binary_op);
					}
                }

            }

            // This template is called after we detect random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVInputIterator, typename BinaryPredicate>
            DVInputIterator min_element_pick_iterator(bolt::cl::control &ctl,
                const DVInputIterator& first,
                const DVInputIterator& last,
                const BinaryPredicate& binary_op,
                const std::string& cl_code,
                bolt::cl::fancy_iterator_tag,
                const char * min_max )
            {
                bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if(runMode == bolt::cl::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }

                const char * str = "MAX_KERNEL";

				#if defined(BOLT_DEBUG_LOG)
                BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
                #endif
				
                switch(runMode)
                {
                case bolt::cl::control::OpenCL :
                    {
					    #if defined(BOLT_DEBUG_LOG)
						if(std::strcmp(min_max,str) == 0)
                           dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_OPENCL_GPU,"::Max_Element::OPENCL_GPU");
						else
						   dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_OPENCL_GPU,"::Min_Element::OPENCL_GPU");
                        #endif
						
                        int pos =  min_element_enqueue( ctl, first, last,  binary_op, cl_code, min_max);
                        return first+pos;
                    }

                case bolt::cl::control::MultiCoreCpu:
                    #ifdef ENABLE_TBB
					    #if defined(BOLT_DEBUG_LOG)
						if(std::strcmp(min_max,str) == 0)
                          dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_MULTICORE_CPU,"::Max_Element::MULTICORE_CPU");
						else
						  dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_MULTICORE_CPU,"::Min_Element::MULTICORE_CPU");
                        #endif
                        if(std::strcmp(min_max,str) == 0)
                              return bolt::btbb::max_element(first, last, binary_op);
                        else
                              return bolt::btbb::min_element(first, last, binary_op);
                    #else
                        throw std::runtime_error( "The MultiCoreCpu version of Max-Min is not enabled to be built! \n" );
                    #endif

                case bolt::cl::control::SerialCpu:
				    #if defined(BOLT_DEBUG_LOG)
					if(std::strcmp(min_max,str) == 0)
					  dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Max_Element::SERIAL_CPU");
					else
                      dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Min_Element::SERIAL_CPU");
                    #endif
                    if(std::strcmp(min_max,str) == 0)
                       return std::max_element(first, last, binary_op);
                    else
                    return std::min_element(first, last, binary_op);

                default:
				    #if defined(BOLT_DEBUG_LOG)
					if(std::strcmp(min_max,str) == 0)
					  dblog->CodePathTaken(BOLTLOG::BOLT_MAXELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Max_Element::SERIAL_CPU");
					else
                      dblog->CodePathTaken(BOLTLOG::BOLT_MINELEMENT,BOLTLOG::BOLT_SERIAL_CPU,"::Min_Element::SERIAL_CPU");
                    #endif
                    if(std::strcmp(min_max,str) == 0)
                       return std::max_element(first, last, binary_op);
                    else
                    return std::min_element(first, last, binary_op);

                }

           }

            template<typename ForwardIterator, typename BinaryPredicate>
            ForwardIterator min_element_detect_random_access(bolt::cl::control &ctl,
                const ForwardIterator& first,
                const ForwardIterator& last,
                const BinaryPredicate& binary_op,
                const std::string& cl_code,
                std::input_iterator_tag)
            {
                //TODO:It should be possible to support non-random_access_iterator_tag iterators,if we copied the data
                //to a temporary buffer.  Should we?
                static_assert( std::is_same< ForwardIterator, std::forward_iterator_tag   >::value, "Bolt only supports random access iterator types" );
            }

			 template<typename ForwardIterator, typename BinaryPredicate>
             ForwardIterator max_element_detect_random_access(bolt::cl::control &ctl,
                 const ForwardIterator& first,
                 const ForwardIterator& last,
                 const BinaryPredicate& binary_op,
                 const std::string& cl_code,
                 std::input_iterator_tag)
             {
                 //TODO:It should be possible to support non-random_access_iterator_tag iterators,if we copied the data
                 //to a temporary buffer.  Should we?
                 static_assert( std::is_same< ForwardIterator, std::forward_iterator_tag   >::value, "Bolt only supports random access iterator types" );
             }

            template<typename ForwardIterator, typename BinaryPredicate>
            ForwardIterator min_element_detect_random_access(bolt::cl::control &ctl,
                const ForwardIterator& first,
                const ForwardIterator& last,
                const BinaryPredicate& binary_op,
                const std::string& cl_code,
                std::random_access_iterator_tag)
            {
                const char * str = "MIN_KERNEL";
                return min_element_pick_iterator( ctl, first, last,  binary_op, cl_code,
                  typename  std::iterator_traits< ForwardIterator >::iterator_category( ), str );
            }



            template<typename ForwardIterator, typename BinaryPredicate>
            ForwardIterator max_element_detect_random_access(bolt::cl::control &ctl,
                const ForwardIterator& first,
                const ForwardIterator& last,
                const BinaryPredicate& binary_op,
                const std::string& cl_code,
                std::random_access_iterator_tag)
            {
                const char * str = "MAX_KERNEL";
                return min_element_pick_iterator( ctl, first, last,  binary_op, cl_code,
                   typename std::iterator_traits< ForwardIterator >::iterator_category( ), str);
            }


        };


        template<typename ForwardIterator>
        ForwardIterator max_element(ForwardIterator first,
            ForwardIterator last,
            const std::string& cl_code)
        {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T;
            return detail::max_element_detect_random_access(bolt::cl::control::getDefault(), first, last, bolt::cl::less<T>(), cl_code,
                   typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        };


        template<typename ForwardIterator,typename BinaryPredicate>
        ForwardIterator max_element(ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op,
            const std::string& cl_code)
        {
            return detail::max_element_detect_random_access(bolt::cl::control::getDefault(), first, last, binary_op, cl_code,
                    typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        };



        template<typename ForwardIterator>
        ForwardIterator  max_element(bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const std::string& cl_code)
        {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T;
            return detail::max_element_detect_random_access(ctl, first, last, bolt::cl::less<T>(), cl_code,
                   typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
        };

        // This template is called by all other "convenience" version of max_element.
        // It also implements the CPU-side mappings of the algorithm for SerialCpu and MultiCoreCpu
        template<typename ForwardIterator, typename BinaryPredicate>
        ForwardIterator max_element(bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op,
            const std::string& cl_code)
            {
                    return detail::max_element_detect_random_access(ctl, first, last, binary_op, cl_code,
                    typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
            }


        // This template is called by all other "convenience" version of min_element.
        // It also implements the CPU-side mappings of the algorithm for SerialCpu and MultiCoreCpu
        template<typename ForwardIterator, typename BinaryPredicate>
        ForwardIterator min_element(bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op,
            const std::string& cl_code)
             {
               return detail::min_element_detect_random_access(ctl, first, last, binary_op, cl_code,
               typename std::iterator_traits< ForwardIterator >::iterator_category( ) );
            }


        template<typename ForwardIterator>
      ForwardIterator min_element(ForwardIterator first,
            ForwardIterator last,
            const std::string& cl_code)
        {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T;
            return min_element(bolt::cl::control::getDefault(), first, last, bolt::cl::less<T>(), cl_code);
        };


        template<typename ForwardIterator,typename BinaryPredicate>
        ForwardIterator min_element(ForwardIterator first,
            ForwardIterator last,
            BinaryPredicate binary_op,
            const std::string& cl_code)
        {
            return min_element(bolt::cl::control::getDefault(), first, last, binary_op, cl_code);
        };



        template<typename ForwardIterator>
        ForwardIterator  min_element(bolt::cl::control &ctl,
            ForwardIterator first,
            ForwardIterator last,
            const std::string& cl_code)
        {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T;
            return min_element(ctl, first, last, bolt::cl::less<T>(),cl_code);
        };

       };

};



#endif
