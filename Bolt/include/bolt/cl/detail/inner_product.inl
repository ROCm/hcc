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

/*
TODO:
1. Optimize the Kernel. In this version transform and reduce are called directly which performs better.
2. Found a caveat in Multi-GPU scenario (Evergreen+Tahiti). Which basically applies to most of the routines.
*/

#if !defined( BOLT_CL_INNERPRODUCT_INL )
#define BOLT_CL_INNERPRODUCT_INL
#define USE_KERNEL 0
#pragma once
#include "bolt/cl/transform.h"
#include "bolt/cl/reduce.h"

#include <bolt/cl/iterator/iterator_traits.h>
#include <bolt/cl/iterator/addressof.h>


//TBB Includes
#ifdef ENABLE_TBB
#include "bolt/btbb/inner_product.h"
#endif

namespace bolt {
    namespace cl {


namespace detail {


namespace serial{

    template< typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl, const InputIterator& first1,
        const InputIterator& last1, const InputIterator& first2, const OutputType& init,
        const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& cl_code, 
		bolt::cl::device_vector_tag)
    {

         typename InputIterator::difference_type sz = (last1 - first1);

         typedef typename std::iterator_traits<InputIterator>::value_type iType;

         /*Get The associated OpenCL buffer for each of the iterators*/
         ::cl::Buffer first1Buffer = first1.base().getContainer( ).getBuffer( );
         ::cl::Buffer first2Buffer = first2.base().getContainer( ).getBuffer( );

         /*Get The size of each OpenCL buffer*/
         size_t first1_sz = first1Buffer.getInfo<CL_MEM_SIZE>();
         size_t first2_sz = first2Buffer.getInfo<CL_MEM_SIZE>();
 
         
         cl_int map_err;
         iType *first1Ptr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
                                                                          first1_sz, NULL, NULL, &map_err);
         iType *first2Ptr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
                                                                          first2_sz, NULL, NULL, &map_err);

         auto mapped_first1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                                        ctl, first1, first1Ptr);
         auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                                        ctl, first2, first2Ptr);

		 OutputType output = init;

		 std::vector<OutputType> result(sz);
         for(int index=0; index < (int)(sz); index++)
         {
             result[index] = (OutputType)  f2( *(first1+index), *(first2+index) );	
         }
		 for(int index=0; index < (int)(sz); index++)
         {
             output = (OutputType) f1( output, result[index] );	
         }

         ::cl::Event unmap_event[2];
         ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
         ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );

         unmap_event[0].wait(); unmap_event[1].wait();

         return output;

    }

	template<typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl,  const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& user_code,
                bolt::cl::fancy_iterator_tag )
    {
		typedef typename std::iterator_traits<InputIterator>::value_type iType;
		OutputType res = init;

		size_t sz = (last1 - first1);
		std::vector<OutputType> result(sz);
        for(int index=0; index < (int)(sz); index++)
        {
            result[index] = (OutputType)  f2( *(first1+index), *(first2+index) );	
        }
		for(int index=0; index < (int)(sz); index++)
        {
            res = (OutputType) f1( res, result[index] );	
        }
		return res;
	}


	template<typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl,  const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& user_code,
                std::random_access_iterator_tag )
    {	
		typedef typename std::iterator_traits<InputIterator>::value_type iType;
		OutputType res = init;

		size_t sz = (last1 - first1);
		std::vector<OutputType> result(sz);
        for(int index=0; index < (int)(sz); index++)
        {
            result[index] = (OutputType) f2( *(first1+index), *(first2+index) );	
        }
		for(int index=0; index < (int)(sz); index++)
        {
            res = (OutputType)  f1( res, result[index] );	
        }
		return res;
	}


}// end of namespace serial

#ifdef ENABLE_TBB
namespace btbb{

    template< typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl, const InputIterator& first1,
        const InputIterator& last1, const InputIterator& first2, const OutputType& init,
        const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& cl_code, 
		bolt::cl::device_vector_tag)
    {

         typename InputIterator::difference_type sz = (last1 - first1);

         typedef typename std::iterator_traits<InputIterator>::value_type iType;

         /*Get The associated OpenCL buffer for each of the iterators*/
         ::cl::Buffer first1Buffer = first1.base().getContainer( ).getBuffer( );
         ::cl::Buffer first2Buffer = first2.base().getContainer( ).getBuffer( );

         /*Get The size of each OpenCL buffer*/
         size_t first1_sz = first1Buffer.getInfo<CL_MEM_SIZE>();
         size_t first2_sz = first2Buffer.getInfo<CL_MEM_SIZE>();
 
         
         cl_int map_err;
         iType *first1Ptr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
                                                                          first1_sz, NULL, NULL, &map_err);
         iType *first2Ptr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
                                                                          first2_sz, NULL, NULL, &map_err);

         auto mapped_first1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                                        ctl, first1, first1Ptr);
         auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                                        ctl, first2, first2Ptr);

         OutputType output = bolt::btbb::inner_product(  mapped_first1_itr, mapped_first1_itr + sz, mapped_first2_itr, init, f1, f2  );

         ::cl::Event unmap_event[2];
         ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
         ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );

         unmap_event[0].wait(); unmap_event[1].wait();

         return output;

    }

	template<typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl,  const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& user_code,
                bolt::cl::fancy_iterator_tag )
    {
		return  bolt::btbb::inner_product(  first1, last1, first2, init, f1, f2  );
	}


	template<typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl,  const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& user_code,
                std::random_access_iterator_tag )
    {
		
		return  bolt::btbb::inner_product(  first1, last1, first2, init, f1, f2  );
	}


}// end of namespace btbb
#endif

namespace cl{

#if USE_KERNEL
    struct CallCompiler_InnerProduct {
       static void constructAndCompile(::cl::Kernel *masterKernel,  std::string cl_code,
           std::string valueTypeName, std::string functorTypeName1, std::string functorTypeName2,
           const control *ctl) {

           const std::string instantiationString =
               "// Host generates this instantiation string with user-specified value type and functor\n"
               "template __attribute__((mangled_name(innerProductInstantiated)))\n"
               "__attribute__((reqd_work_group_size(64,1,1)))\n"
               "kernel void innerProductTemplate(\n"
               "global " + valueTypeName + "* A,\n"
               "global " + valueTypeName + "* B,\n"
               "const int length,\n"
               "global " + functorTypeName1 + "* userFunctor,\n"
               "global " + functorTypeName2 + "* userFunctor2,\n"
               "global " + valueTypeName + "* result,\n"
               "local " + valueTypeName + "* scratch\n"
               ");\n\n";

           std::string functorTypeNames = functorTypeName1 + " , " + functorTypeName2;

           bolt::cl::constructAndCompileString( masterKernel, "innerProduct", inner_product_kernels,
               instantiationString, cl_code, valueTypeName, functorTypeNames, *ctl);

       };
    };
#endif


    template< typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl, const InputIterator& first1,
        const InputIterator& last1, const InputIterator& first2, const OutputType& init,
        const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& cl_code, 
		bolt::cl::device_vector_tag)
    {

        //Should we directly call transform and reduce routines or launch a separate kernel?
        typedef typename std::iterator_traits<InputIterator>::value_type iType;

        cl_uint distVec = static_cast< cl_uint >( std::distance( first1, last1 ) );
        if( distVec == 0 )
            return init;

        device_vector< OutputType > tempDV( distVec, OutputType() , CL_MEM_READ_WRITE, false, ctl );
        detail::cl::binary_transform( ctl, first1, last1, first2, tempDV.begin(), f2,cl_code);
        return detail::reduce( ctl, tempDV.begin(), tempDV.end(), init, f1, cl_code);
    };



    template<typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl,  const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& user_code,
                std::random_access_iterator_tag )
    {
		
		int sz = static_cast<int>(last1 - first1);

        typedef typename std::iterator_traits<InputIterator>::value_type  iType;
              
        typedef typename std::iterator_traits<InputIterator>::pointer pointer;
        
        pointer first_pointer1 = bolt::cl::addressof(first1) ;
        pointer first_pointer2 = bolt::cl::addressof(first2) ;

        device_vector< iType > dvInput1( first_pointer1, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, true, ctl );
        device_vector< iType > dvInput2( first_pointer2, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, true, ctl );

        auto device_iterator_first1  = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator>::iterator_category( ), 
                                            first1, dvInput1.begin());
        auto device_iterator_last1   = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                            last1, dvInput1.end());
        auto device_iterator_first2  = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                            first2, dvInput2.begin());
        return inner_product(ctl, device_iterator_first1, device_iterator_last1, device_iterator_first2, 
                              init, f1, f2, user_code,  bolt::cl::device_vector_tag() );

	}

 template<typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
    OutputType inner_product(bolt::cl::control &ctl,  const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& user_code,
                bolt::cl::fancy_iterator_tag )
    {
		return inner_product( ctl, first1, last1, first2, init, f1, f2, user_code, typename bolt::cl::memory_system<InputIterator>::type()  );
    }


} //end of namespace cl



    template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
    typename std::enable_if< 
           !(std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                         std::input_iterator_tag 
                       >::value), OutputType
                       >::type
    inner_product( bolt::cl::control& ctl, const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& user_code )
    {
        typedef typename std::iterator_traits<InputIterator>::value_type iType;
        int sz = static_cast<int>( std::distance( first1, last1 ) );

        if( sz == 0 )
            return init;

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
            dblog->CodePathTaken(BOLTLOG::BOLT_INNERPRODUCT,BOLTLOG::BOLT_SERIAL_CPU,"::Inner_Product::SERIAL_CPU");
            #endif
            
            return serial::inner_product( ctl, first1, last1, first2, init,
				f1, f2, user_code, typename std::iterator_traits<InputIterator>::iterator_category()  );

        }
        else if(runMode == bolt::cl::control::MultiCoreCpu)
        {
            #ifdef ENABLE_TBB
                   #if defined(BOLT_DEBUG_LOG)
                   dblog->CodePathTaken(BOLTLOG::BOLT_INNERPRODUCT,BOLTLOG::BOLT_MULTICORE_CPU,"::Inner_Product::MULTICORE_CPU");
                   #endif
                   return btbb::inner_product(ctl, first1, last1, first2,
					   init, f1, f2, user_code, typename std::iterator_traits<InputIterator>::iterator_category());
            #else
                   throw std::runtime_error("MultiCoreCPU Version of inner_product is not Enabled! \n");
            #endif
        }
        else
        {
            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_INNERPRODUCT,BOLTLOG::BOLT_OPENCL_GPU,"::Inner_Product::OPENCL_GPU");
            #endif
            return cl::inner_product( ctl, first1, last1, first2, init,
				f1, f2, user_code, typename std::iterator_traits<InputIterator>::iterator_category() );
        } 
    }
    


	template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
    typename std::enable_if< 
           (std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                         std::input_iterator_tag 
                       >::value), OutputType
                       >::type
    inner_product( bolt::cl::control& ctl, const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2, const std::string& user_code )
    {
         //TODO - Shouldn't we support inner_product for input_iterator_tag also. 
         static_assert( std::is_same< typename std::iterator_traits< InputIterator>::iterator_category, 
                                           std::input_iterator_tag >::value , 
                             "Input vector cannot be of the type input_iterator_tag" );
    }



}//End OF detail namespace


        // default control, two-input transform, std:: iterator
        template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
        OutputType inner_product(bolt::cl::control& ctl, InputIterator first1, InputIterator last1,
        InputIterator first2, OutputType init, BinaryFunction1 f1, BinaryFunction2 f2,
        const std::string& user_code )
        {
            return detail::inner_product( ctl, first1, last1, first2, init, f1, f2, user_code);
        }

        // default control, two-input transform, std:: iterator
        template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
        OutputType inner_product( InputIterator first1, InputIterator last1, InputIterator first2, OutputType init,
            BinaryFunction1 f1, BinaryFunction2 f2, const std::string& user_code )
        {
            return inner_product( control::getDefault(), first1, last1, first2, init, f1, f2, user_code );
        }
        template<typename InputIterator, typename OutputType>
        OutputType inner_product(bolt::cl::control& ctl,InputIterator first1,InputIterator last1,InputIterator first2,
            OutputType init, const std::string& user_code )
        {
            typedef typename std::iterator_traits<InputIterator>::value_type iType;
            return detail::inner_product(ctl, first1,last1,first2,init,bolt::cl::plus< iType >( ),
                bolt::cl::multiplies< iType >( ), user_code);
        }

        // default control, two-input transform, std:: iterator
        template<typename InputIterator, typename OutputType>
        OutputType inner_product( InputIterator first1, InputIterator last1, InputIterator first2, OutputType init,
            const std::string& user_code )
        {
            typedef typename std::iterator_traits<InputIterator>::value_type iType;
            return inner_product( control::getDefault(), first1, last1, first2, init,
                bolt::cl::plus< iType >( ), bolt::cl::multiplies< iType >( ), user_code);
        }


    }//end of cl namespace
};//end of bolt namespace



#endif
