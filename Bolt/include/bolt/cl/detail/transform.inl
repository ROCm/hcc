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
#if !defined( BOLT_CL_TRANSFORM_INL )
#define BOLT_CL_TRANSFORM_INL
#define WAVEFRONT_SIZE 64
#define TRANSFORM_ENABLE_PROFILING 0

#include <type_traits>

#ifdef ENABLE_TBB
    #include "bolt/btbb/transform.h"
#endif

#include "bolt/cl/bolt.h"
#include "bolt/cl/device_vector.h"
#include "bolt/cl/distance.h"
#include "bolt/cl/iterator/iterator_traits.h"
#include "bolt/cl/iterator/transform_iterator.h"
#include "bolt/cl/iterator/permutation_iterator.h"
#include "bolt/cl/iterator/addressof.h"

namespace bolt {
namespace cl {

namespace detail {

namespace serial{

    template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
    typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
    binary_transform( ::bolt::cl::control &ctl, const InputIterator1& first1, const InputIterator1& last1,
                      const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f)
    {
            typename InputIterator1::difference_type sz = (last1 - first1);
            if (sz == 0)
                return;
            typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
            typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
            typedef typename std::iterator_traits<OutputIterator>::value_type oType;
            /*Get The associated OpenCL buffer for each of the iterators*/
            ::cl::Buffer first1Buffer = first1.base().getContainer( ).getBuffer( );
            ::cl::Buffer first2Buffer = first2.base().getContainer( ).getBuffer( );
            ::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
            /*Get The size of each OpenCL buffer*/
            size_t first1_sz = first1Buffer.getInfo<CL_MEM_SIZE>();
            size_t first2_sz = first2Buffer.getInfo<CL_MEM_SIZE>();
            size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();
            
            cl_int map_err;
            iType1 *first1Ptr = (iType1*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
                                                                             first1_sz, NULL, NULL, &map_err);
            iType2 *first2Ptr = (iType2*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
                                                                             first2_sz, NULL, NULL, &map_err);
            oType *resultPtr  = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
                                                                             result_sz, NULL, NULL, &map_err);
            auto mapped_first1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator1>::iterator_category(), 
                                                           ctl, first1, first1Ptr);
            auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                           ctl, first2, first2Ptr);
            auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                           ctl, result, resultPtr);
            for(int index=0; index < (int)(sz); index++)
            {
                *(mapped_result_itr + index) = f( *(mapped_first1_itr+index), *(mapped_first2_itr+index) );
            }
            ::cl::Event unmap_event[3];
            ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
            ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
            ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[2] );
            unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait();
            return;
    }
    
    template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
    typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
    binary_transform( ::bolt::cl::control &ctl, const InputIterator1& first1, const InputIterator1& last1,
                      const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f)
    {
        size_t sz = (last1 - first1);
        if (sz == 0)
            return;
        for(int index=0; index < (int)(sz); index++)
        {
            *(result + index) = f( *(first1+index), *(first2+index) );
        }
    }

    template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
    unary_transform( ::bolt::cl::control &ctl, InputIterator& first, InputIterator& last,
                    OutputIterator& result, UnaryFunction& f)
    {
        size_t sz = (last - first);
        if (sz == 0)
            return;

        typedef typename std::iterator_traits<InputIterator>::value_type iType;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        /*Get The associated OpenCL buffer for each of the iterators*/
        ::cl::Buffer firstBuffer  = first.base().getContainer( ).getBuffer( );
        ::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
        /*Get The size of each OpenCL buffer*/
        size_t first_sz  = firstBuffer.getInfo<CL_MEM_SIZE>();
        size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();

        cl_int map_err;
        iType *firstPtr = (iType*)ctl.getCommandQueue().enqueueMapBuffer(firstBuffer, true, CL_MAP_READ, 0, 
                                                                            first_sz, NULL, NULL, &map_err);
        oType *resultPtr = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
                                                                            result_sz, NULL, NULL, &map_err);
        auto mapped_first_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                                        ctl, first, firstPtr);
        auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                        ctl, result, resultPtr);
        for(int index=0; index < (int)(sz); index++)
        {
            *(mapped_result_itr + index) = f( *(mapped_first_itr+index) );
        }
        ::cl::Event unmap_event[2];
        ctl.getCommandQueue().enqueueUnmapMemObject(firstBuffer, firstPtr, NULL, &unmap_event[0] );
        ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[1] );
        unmap_event[0].wait(); unmap_event[1].wait(); 
        return ;

    }
    
    template<typename Iterator, typename OutputIterator, typename UnaryFunction>
    typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value
                           >::type
    unary_transform( ::bolt::cl::control &ctl, Iterator& first, Iterator& last,
                    OutputIterator& result, UnaryFunction& f )
    {
        size_t sz = (last - first);
        if (sz == 0)
            return;
        for(int index=0; index < (int)(sz); index++)
        {
            *(result + index) = f( *(first+index) );
        }
        
        return;
    }
}

#ifdef ENABLE_TBB
namespace btbb{

    template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
    typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value
                       >::type
    binary_transform( ::bolt::cl::control &ctl, const InputIterator1& first1, const InputIterator1& last1,
               const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f)
    {
        typename InputIterator1::difference_type sz = (last1 - first1);
        if (sz == 0)
            return;
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        /*Get The associated OpenCL buffer for each of the iterators*/
        ::cl::Buffer first1Buffer = first1.base().getContainer( ).getBuffer( );
        ::cl::Buffer first2Buffer = first2.base().getContainer( ).getBuffer( );
        ::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
        /*Get The size of each OpenCL buffer*/
        size_t first1_sz = first1Buffer.getInfo<CL_MEM_SIZE>();
        size_t first2_sz = first2Buffer.getInfo<CL_MEM_SIZE>();
        size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();

        //typename bolt::cl::device_vector< iType >::pointer firstPtr = first.base().getContainer( ).data( ); 
        cl_int map_err;
        iType1 *first1Ptr = (iType1*)ctl.getCommandQueue().enqueueMapBuffer(first1Buffer, true, CL_MAP_READ, 0, 
                                                                            first1_sz, NULL, NULL, &map_err);
        iType2 *first2Ptr = (iType2*)ctl.getCommandQueue().enqueueMapBuffer(first2Buffer, true, CL_MAP_READ, 0, 
                                                                            first2_sz, NULL, NULL, &map_err);
        oType *resultPtr  = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
                                                                            result_sz, NULL, NULL, &map_err);
        auto mapped_first1_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator1>::iterator_category(), 
                                                        ctl, first1, first1Ptr);
        auto mapped_first2_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                                        ctl, first2, first2Ptr);
        auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                        ctl, result, resultPtr);
        bolt::btbb::transform(mapped_first1_itr, mapped_first1_itr+(int)sz, mapped_first2_itr, mapped_result_itr, f);

        ::cl::Event unmap_event[3];
        ctl.getCommandQueue().enqueueUnmapMemObject(first1Buffer, first1Ptr, NULL, &unmap_event[0] );
        ctl.getCommandQueue().enqueueUnmapMemObject(first2Buffer, first2Ptr, NULL, &unmap_event[1] );
        ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[2] );
        unmap_event[0].wait(); unmap_event[1].wait(); unmap_event[2].wait();
        return;
    }
    
    template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
    typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value
                           >::type
    binary_transform( ::bolt::cl::control &ctl, const InputIterator1& first1, const InputIterator1& last1,
               const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f)

    {
        bolt::btbb::transform(first1, last1, first2, result, f);
        return;
    }


    template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
    unary_transform( ::bolt::cl::control &ctl, const InputIterator& first, const InputIterator& last,
    const OutputIterator& result, const UnaryFunction& f)
    {
        size_t sz = (last - first);
        if (sz == 0)
            return;

        typedef typename std::iterator_traits<InputIterator>::value_type iType;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        /*Get The associated OpenCL buffer for each of the iterators*/
        ::cl::Buffer firstBuffer  = first.base().getContainer( ).getBuffer( );
        ::cl::Buffer resultBuffer = result.getContainer( ).getBuffer( );
        /*Get The size of each OpenCL buffer*/
        size_t first_sz  = firstBuffer.getInfo<CL_MEM_SIZE>();
        size_t result_sz = resultBuffer.getInfo<CL_MEM_SIZE>();

        //typename bolt::cl::device_vector< iType >::pointer firstPtr = first.base().getContainer( ).data( ); 
        cl_int map_err;
        iType *firstPtr  = (iType*)ctl.getCommandQueue().enqueueMapBuffer(firstBuffer, true, CL_MAP_READ, 0, 
                                                                            first_sz, NULL, NULL, &map_err);
        oType *resultPtr = (oType*)ctl.getCommandQueue().enqueueMapBuffer(resultBuffer, true, CL_MAP_WRITE, 0, 
                                                                            result_sz, NULL, NULL, &map_err);
        auto mapped_first_itr = create_mapped_iterator(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                                        ctl, first, firstPtr);
        auto mapped_result_itr = create_mapped_iterator(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                                        ctl, result, resultPtr);
        bolt::btbb::transform(mapped_first_itr, mapped_first_itr + (int)sz, mapped_result_itr, f);

        ::cl::Event unmap_event[2];
        ctl.getCommandQueue().enqueueUnmapMemObject(firstBuffer, firstPtr, NULL, &unmap_event[0] );
        ctl.getCommandQueue().enqueueUnmapMemObject(resultBuffer, resultPtr, NULL, &unmap_event[1] );
        unmap_event[0].wait(); unmap_event[1].wait(); 
        
        return;
    }
    
    template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                           >::value
                           >::type
    unary_transform( ::bolt::cl::control &ctl, const InputIterator& first, const InputIterator& last,
    const OutputIterator& result, const UnaryFunction& f )
    {
        // TODO - Add tbb host vector code.
        bolt::btbb::transform(first, last, result, f);
        return;
    }
}
#endif
namespace cl{

    enum TransformTypes {transform_iType1, transform_DVInputIterator1, transform_iType2, transform_DVInputIterator2,
                           transform_oTypeB,transform_DVOutputIteratorB, transform_BinaryFunction, transform_endB };
    enum TransformUnaryTypes {transform_iType, transform_DVInputIterator, transform_oTypeU,
                                transform_DVOutputIteratorU, transform_UnaryFunction, transform_endU };



    class KernelParameterStrings
    {
    private:
        std::string toString(int num) const
        {
            std::ostringstream oss;
            oss << num;
            return oss.str();
        }
        public:
            std::string getInputIteratorString(bolt::cl::permutation_iterator_tag, const ::std::string& itrStr, int itr_num ) const 
            {
                return "global " + itrStr  + "::base_type* in" + toString(itr_num) + "_ptr_0,\n"
                       "global " + itrStr  + "::index_type* in" + toString(itr_num) + "_ptr_1,\n"
                       + itrStr + " input" + toString(itr_num) + "_iter,\n";
            }
            std::string getInputIteratorString(bolt::cl::counting_iterator_tag, const ::std::string& itrStr, int itr_num ) const 
            {
                return "global " + itrStr  + "::base_type* in" + toString(itr_num) + "_ptr_0,\n"
                       + itrStr + " input" + toString(itr_num) + "_iter,\n";
            }
            std::string getInputIteratorString(bolt::cl::constant_iterator_tag, const ::std::string& itrStr, int itr_num )       const  
            {
                return "global " + itrStr  + "::base_type* in" + toString(itr_num) + "_ptr_0,\n"
                       + itrStr + " input" + toString(itr_num) + "_iter,\n";
            }
            std::string getInputIteratorString(bolt::cl::device_vector_tag, const ::std::string& itrStr, int itr_num ) const 
            {
                return "global " + itrStr  + "::base_type* in" + toString(itr_num) + "_ptr_0,\n"
                       + itrStr + " input" + toString(itr_num) + "_iter,\n";
            }
            std::string getInputIteratorString(bolt::cl::transform_iterator_tag, const ::std::string& itrStr, int itr_num ) const 
            {
                return "global " + itrStr  + "::base_type* in" + toString(itr_num) + "_ptr_0,\n"
                       + itrStr + " input" + toString(itr_num) + "_iter,\n";    
            }

            std::string getOutputIteratorString(bolt::cl::device_vector_tag, const ::std::string& itrStr ) const 
            {
                return "global " + itrStr  + "::base_type* out_ptr_0,\n"
                       + itrStr + " output_iter,\n";    
            }
    };


    template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
    class Transform_KernelTemplateSpecializer : public KernelTemplateSpecializer
    {
        KernelParameterStrings kps;
        public:
        Transform_KernelTemplateSpecializer() : KernelTemplateSpecializer()
        {
            addKernelName("transformTemplate");
            addKernelName("transformNoBoundsCheckTemplate");
        }

        const ::std::string operator() ( const ::std::vector< ::std::string>& binaryTransformKernels ) const
            {
                const std::string templateSpecializationString =
                "// Host generates this instantiation string with user-specified value type and functor\n"
                "template __attribute__((mangled_name("+name( 0 )+"Instantiated)))\n"
                "kernel void "+name(0)+"(\n"
                + kps.getInputIteratorString(typename std::iterator_traits<InputIterator1>::iterator_category(), 
                                         binaryTransformKernels[transform_DVInputIterator1], 1 )
                + kps.getInputIteratorString(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                         binaryTransformKernels[transform_DVInputIterator2], 2 )
                + kps.getOutputIteratorString(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                          binaryTransformKernels[transform_DVOutputIteratorB] )
                + "const uint length,\n"
                "global " + binaryTransformKernels[transform_BinaryFunction] + "* userFunctor);\n\n"

                "// Host generates this instantiation string with user-specified value type and functor\n"
                "template __attribute__((mangled_name("+name(1)+"Instantiated)))\n"
                "kernel void "+name(1)+"(\n"
                + kps.getInputIteratorString(typename std::iterator_traits<InputIterator1>::iterator_category(), 
                                         binaryTransformKernels[transform_DVInputIterator1], 1)
                + kps.getInputIteratorString(typename std::iterator_traits<InputIterator2>::iterator_category(), 
                                         binaryTransformKernels[transform_DVInputIterator2], 2)
                + kps.getOutputIteratorString(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                          binaryTransformKernels[transform_DVOutputIteratorB])
                + "const uint length,\n"
                "global " + binaryTransformKernels[transform_BinaryFunction] + "* userFunctor);\n\n";

                return templateSpecializationString;

            }

        const ::std::string getBinaryNoBoundsKernelPrototype (  ) 
            {
                std::string return_string = 
                "template <typename iIterType1, typename iIterType2, typename oIterType, typename unary_function > \n"
                "kernel \n"
                "void transformNoBoundsCheckTemplate( \n"
                "    global typename iIterType1::base_type* in1_ptr_0, \n"; 
                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator1>::iterator_category(), typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "    global typename iIterType1::index_type* in1_ptr_1, \n";
                return_string += 
                "    iIterType1 in1_iter,\n"
                "    global typename iIterType2::base_type* in2_ptr_0, \n"; 
                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator2>::iterator_category(), typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "    global typename iIterType2::index_type* in2_ptr_1, \n";
                return_string += 
                "    iIterType2 in2_iter,\n"
                "    global typename oIterType::base_type* out_ptr_0,\n"
                "    oIterType Z_iter,\n"
			    "    const uint length,\n"
                "    global unary_function* userFunctor)\n"
                "{\n"
                "\n";

                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator1>::iterator_category(), typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "in1_iter.init( in1_ptr_0, in1_ptr_1 );\n";
                else
                    return_string += "in1_iter.init( in1_ptr_0);\n";

                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator2>::iterator_category(), typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "in2_iter.init( in2_ptr_0, in2_ptr_1 );\n";
                else
                    return_string += "in2_iter.init( in2_ptr_0);\n";
                return_string += 
    
                "    Z_iter.init( out_ptr_0 ); \n"
                "    int gx = get_global_id( 0 ); \n"
                "    typename iIterType1::value_type aa = in1_iter[ gx ];\n"
                "    typename iIterType2::value_type bb = in2_iter[ gx ];\n"
                "    Z_iter[ gx ] = (*userFunctor)( aa, bb );\n"
                "}\n";
                return return_string;
            }

        const ::std::string getBinaryBoundsKernelPrototype (  ) 
            {
                std::string return_string = 
                "template <typename iIterType1, typename iIterType2, typename oIterType, typename unary_function > \n"
                "kernel \n"
                "void transformTemplate( \n"
                "    global typename iIterType1::base_type* in1_ptr_0, \n"; 
                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator1>::iterator_category(), typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "    global typename iIterType1::index_type* in1_ptr_1, \n";
                return_string += 
                "    iIterType1 in1_iter,\n"
                "    global typename iIterType2::base_type* in2_ptr_0, \n"; 
                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator2>::iterator_category, typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "    global typename iIterType2::index_type* in2_ptr_1, \n";
                return_string += 
                "    iIterType2 in2_iter,\n"
                "    global typename oIterType::base_type* out_ptr_0,\n"
                "    oIterType Z_iter,\n"
			    "    const uint length,\n"
                "    global unary_function* userFunctor)\n"
                "{\n"
                "\n";

                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator1>::iterator_category, typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "in1_iter.init( in1_ptr_0, in1_ptr_1 );\n";
                else
                    return_string += "in1_iter.init( in1_ptr_0);\n";

                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator2>::iterator_category, typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "in2_iter.init( in2_ptr_0, in2_ptr_1 );\n";
                else
                    return_string += "in2_iter.init( in2_ptr_0);\n";
                return_string += 
    
                "    Z_iter.init( out_ptr_0 ); \n"
                "    int gx = get_global_id( 0 ); \n"
                "    if (gx >= length) \n"
                "       return; \n"
                "    typename iIterType1::value_type aa = in1_iter[ gx ];\n"
                "    typename iIterType2::value_type bb = in2_iter[ gx ];\n"
                "    Z_iter[ gx ] = (*userFunctor)( aa, bb );\n"
                "}\n";
                return return_string;
            }
    };
    
    template <typename InputIterator, typename OutputIterator>
    class TransformUnary_KernelTemplateSpecializer : public KernelTemplateSpecializer, public  KernelParameterStrings
    {
        KernelParameterStrings kps;
        public:
        TransformUnary_KernelTemplateSpecializer() : KernelTemplateSpecializer()
        {
            addKernelName("unaryTransformTemplate");
            addKernelName("unaryTransformNoBoundsCheckTemplate");
        }
        
        const ::std::string operator() ( const ::std::vector< ::std::string>& unaryTransformKernels ) const
            {

                const std::string templateSpecializationString =
                "// Host generates this instantiation string with user-specified value type and functor\n"
                "template __attribute__((mangled_name("+name( 0 )+"Instantiated)))\n"
                "kernel void unaryTransformTemplate(\n"
                + kps.getInputIteratorString(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                         unaryTransformKernels[transform_DVInputIterator], 1 )
                + kps.getOutputIteratorString(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                          unaryTransformKernels[transform_DVOutputIteratorU] )
                + "const uint length,\n"
                "global " + unaryTransformKernels[transform_UnaryFunction] + "* userFunctor);\n\n"

                "// Host generates this instantiation string with user-specified value type and functor\n"
                "template __attribute__((mangled_name("+name(1)+"Instantiated)))\n"
                "kernel void unaryTransformNoBoundsCheckTemplate(\n"
                + kps.getInputIteratorString(typename std::iterator_traits<InputIterator>::iterator_category(), 
                                         unaryTransformKernels[transform_DVInputIterator], 1)
                + kps.getOutputIteratorString(typename std::iterator_traits<OutputIterator>::iterator_category(), 
                                          unaryTransformKernels[transform_DVOutputIteratorU])
                + "const uint length,\n"
                "global " +unaryTransformKernels[transform_UnaryFunction] + "* userFunctor);\n\n";

                return templateSpecializationString;
            }

        const ::std::string getUnaryNoBoundsKernelPrototype (  ) 
            {
                std::string return_string = 
                "template <typename iIterType, typename oIterType, typename unary_function > \n"
                "kernel \n"
                "void unaryTransformNoBoundsCheckTemplate( \n"
                "    global typename iIterType::base_type* in0_ptr_0, \n"; 
                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator>::iterator_category, typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "    global typename iIterType::index_type* in0_ptr_1, \n";
                return_string += 
                "    iIterType A_iter,\n"
                "    global typename oIterType::base_type* out_ptr_0,\n"
                "    oIterType Z_iter,\n"
			    "    const uint length,\n"
                "    global unary_function* userFunctor)\n"
                "{\n"
                "\n";

                if( std::is_same<typename bolt::cl::iterator_traits<InputIterator>::iterator_category, typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "A_iter.init( in0_ptr_0, in0_ptr_1 );\n";
                else
                    return_string += "A_iter.init( in0_ptr_0);\n";
                return_string += 
                "    Z_iter.init( out_ptr_0 ); \n"
                "    int gx = get_global_id( 0 );   printf(\"%d , gx  \");\n"
                "    typename iIterType::value_type aa = A_iter[ gx ];\n"
                "    Z_iter[ gx ] = (*userFunctor)( aa );\n"
                "}\n";
                return return_string;
            }

        const ::std::string getUnaryBoundsKernelPrototype (  ) 
            {
                std::string return_string = 
                "template <typename iIterType, typename oIterType, typename unary_function > \n"
                "kernel \n"
                "void unaryTransformTemplate( \n"
                "    global typename iIterType::base_type* in0_ptr_0, \n"; 
                if( std::is_same<typename std::iterator_traits<InputIterator>::iterator_category, typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "    global typename iIterType::index_type* in0_ptr_1, \n";
                return_string += 
                "    iIterType A_iter,\n"
                "    global typename oIterType::base_type* out_ptr_0,\n"
                "    oIterType Z_iter,\n"
			    "    const uint length,\n"
                "    global unary_function* userFunctor)\n"
                "{\n"
                "\n";
                if(std::is_same<typename std::iterator_traits<InputIterator>::iterator_category, typename bolt::cl::permutation_iterator_tag>::value == true)
                    return_string += "A_iter.init( in0_ptr_0, in0_ptr_1 );\n";
                else
                    return_string += "A_iter.init( in0_ptr_0);\n";
                return_string += 
                "    Z_iter.init( out_ptr_0 ); \n"
                "    int gx = get_global_id( 0 );\n"
	            "    if (gx >= length)\n"
		        "        return;\n"
                "    typename iIterType::value_type aa = A_iter[ gx ];\n"
                "    Z_iter[ gx ] = (*userFunctor)( aa );\n"
                "}\n";
                return return_string;
            }
    };

    /*! \brief This template function overload is used strictly for device_vector and OpenCL implementations. 
        \detail 
    */
    template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
    typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                           >::value
                           >::type
    binary_transform( ::bolt::cl::control &ctl, const InputIterator1& first1, const InputIterator1& last1,
                      const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f, 
                      const std::string& user_code)
    {
        typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        typename InputIterator1::difference_type distVec = last1 - first1;
        if( distVec == 0 )
            return;

        const size_t numComputeUnits = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
        const size_t numWorkGroupsPerComputeUnit = ctl.getWGPerComputeUnit( );
        size_t numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;

        /**********************************************************************************
         * Type Names - used in KernelTemplateSpecializer
         *********************************************************************************/

        std::vector<std::string> binaryTransformKernels(transform_endB);
        binaryTransformKernels[transform_iType1] = TypeName< iType1 >::get( );
        binaryTransformKernels[transform_iType2] = TypeName< iType2 >::get( );
        binaryTransformKernels[transform_DVInputIterator1]  = TypeName< InputIterator1 >::get( );
        binaryTransformKernels[transform_DVInputIterator2]  = TypeName< InputIterator2 >::get( );
        binaryTransformKernels[transform_oTypeB]            = TypeName< oType >::get( );
        binaryTransformKernels[transform_DVOutputIteratorB] = TypeName< OutputIterator >::get( );
        binaryTransformKernels[transform_BinaryFunction]    = TypeName< BinaryFunction >::get();

       /**********************************************************************************
        * Type Definitions - directrly concatenated into kernel string
        *********************************************************************************/

        // For user-defined types, the user must create a TypeName trait which returns the name of the
        //class - note use of TypeName<>::get to retrieve the name here.
        std::vector<std::string> typeDefinitions;
        PUSH_BACK_UNIQUE( typeDefinitions, user_code)
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< OutputIterator >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< oType >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType1 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType2 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< InputIterator1 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< InputIterator2 >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< BinaryFunction  >::get() )
        /**********************************************************************************
         * Calculate WG Size
         *********************************************************************************/

        cl_int l_Error = CL_SUCCESS;
        const size_t wgSize  = WAVEFRONT_SIZE;
        V_OPENCL( l_Error, "Error querying kernel for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" );
        assert( (wgSize & (wgSize-1) ) == 0 ); // The bitwise &,~ logic below requires wgSize to be a power of 2

        int boundsCheck = 0;
        size_t wgMultiple = distVec;
        size_t lowerBits = ( distVec & (wgSize-1) );
        if( lowerBits )
        {
            //  Bump the workitem count to the next multiple of wgSize
            wgMultiple &= ~lowerBits;
            wgMultiple += wgSize;
        }
        else
        {
            boundsCheck = 1;
        }
        if (wgMultiple/wgSize < numWorkGroups)
            numWorkGroups = wgMultiple/wgSize;

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
         Transform_KernelTemplateSpecializer<InputIterator1, InputIterator2, OutputIterator> ts_kts;
         std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
             ctl,
             binaryTransformKernels,
             &ts_kts,
             typeDefinitions,
             /*transform_kernels*/ts_kts.getBinaryNoBoundsKernelPrototype() + ts_kts.getBinaryBoundsKernelPrototype(),
             compileOptions);
         // kernels returned in same order as added in KernelTemplaceSpecializer constructor


        ALIGNED( 256 ) BinaryFunction aligned_binary( f );
        control::buffPointer userFunctor = ctl.acquireBuffer( sizeof( aligned_binary ),
        CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_binary );

        typename InputIterator1::Payload first1_payload = first1.gpuPayload( );
        typename InputIterator2::Payload first2_payload = first2.gpuPayload( );
        typename OutputIterator::Payload result_payload = result.gpuPayload( );

        int arg_num = 0;
        /*Set the arguments for the kernel for InputIterator. Note that this takes input as the argument 
          number to start setting the values. The return value is the number of the argument to begin setting 
          the next Kernel arguments. 
          Once the cl::Buffer arguments are set the GPU Payload arguments are also passed to the kernel*/
        arg_num = first1.setKernelBuffers(arg_num, kernels[boundsCheck]);
        kernels[boundsCheck].setArg(arg_num, first1.gpuPayloadSize( ),&first1_payload);
        arg_num++;

        arg_num = first2.setKernelBuffers(arg_num, kernels[boundsCheck]);
        kernels[boundsCheck].setArg(arg_num, first2.gpuPayloadSize( ),&first2_payload);
        arg_num++;

        /*Do the same for OutputIterator*/
        arg_num = result.setKernelBuffers(arg_num, kernels[boundsCheck]);
        kernels[boundsCheck].setArg(arg_num, result.gpuPayloadSize( ),&result_payload);
        arg_num++;

        //The type cast to int is required because sz is of type size_t
        kernels[boundsCheck].setArg(arg_num, (int)distVec );
        kernels[boundsCheck].setArg(arg_num+1, *userFunctor);


        ::cl::Event transformEvent;
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
          kernels[boundsCheck],
            ::cl::NullRange,
            ::cl::NDRange(wgMultiple), // numWorkGroups*wgSize
            ::cl::NDRange(wgSize),
            NULL,
            &transformEvent );
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for transform() kernel" );

        ::bolt::cl::wait(ctl, transformEvent);

#if TRANSFORM_ENABLE_PROFILING
        if( 0 )
        {
          cl_ulong start_time, stop_time;

          l_Error = transformEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start_time);
          V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()");
          l_Error = transformEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &stop_time);
          V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");
          size_t time = stop_time - start_time;
          std::cout << "Global Memory Bandwidth: "<<((distVec*(2.0*sizeof(iType1)+sizeof(oType)))/time)<<std::endl;
        }
#endif // BOLT_ENABLE_PROFILING

    }

    /*! \brief This template function overload is used strictly std random access vectors and OpenCL implementations. 
        \detail 
    */
    template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
    typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value
                           >::type
    binary_transform( ::bolt::cl::control &ctl, const InputIterator1& first1, const InputIterator1& last1,
                      const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f, 
                      const std::string& user_code )
    {
        int sz = static_cast<int>(last1 - first1);
        if (sz == 0)
            return;
        typedef typename std::iterator_traits<InputIterator1>::value_type  iType1;
        typedef typename std::iterator_traits<InputIterator2>::value_type  iType2;
        typedef typename std::iterator_traits<OutputIterator>::value_type  oType;
       
        typedef typename std::iterator_traits<InputIterator1>::pointer pointer1;
        typedef typename std::iterator_traits<InputIterator2>::pointer pointer2;
        
        pointer1 first_pointer1 = bolt::cl::addressof(first1) ;
        pointer2 first_pointer2 = bolt::cl::addressof(first2) ;

        device_vector< iType1 > dvInput1( first_pointer1, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
        device_vector< iType2 > dvInput2( first_pointer2, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
        device_vector< oType >  dvOutput( result, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, false, ctl );

        auto device_iterator_first1  = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                            first1, dvInput1.begin());
        auto device_iterator_last1   = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator1 >::iterator_category( ), 
                                            last1, dvInput1.end());
        auto device_iterator_first2  = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator2 >::iterator_category( ), 
                                            first2, dvInput2.begin());
        cl::binary_transform(ctl, device_iterator_first1, device_iterator_last1, device_iterator_first2, 
                             dvOutput.begin(), f, user_code);
        dvOutput.data( );
        return;
    }


    /********************************Unary Transform ********************************************/

    /*! \brief This template function overload is used strictly for device_vector and OpenCL implementations. 
        \detail 
    */
    template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       bolt::cl::device_vector_tag
                                     >::value
                       >::type
    unary_transform( ::bolt::cl::control &ctl, const InputIterator& first, const InputIterator& last,
    const OutputIterator& result, const UnaryFunction& f, const std::string& user_code)
    {
        typename InputIterator::difference_type sz = bolt::cl::distance(first, last);
        if (sz == 0)
            return;
        typedef typename std::iterator_traits<InputIterator>::value_type  iType;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;

        const size_t numComputeUnits = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
        const size_t numWorkGroupsPerComputeUnit = ctl.getWGPerComputeUnit( );
        const size_t numWorkGroups = numComputeUnits * numWorkGroupsPerComputeUnit;

        /**********************************************************************************
         * Type Names - used in KernelTemplateSpecializer
         *********************************************************************************/
        
        std::vector<std::string> unaryTransformKernels( transform_endU );
        unaryTransformKernels[transform_iType] = TypeName< iType >::get( );
        unaryTransformKernels[transform_DVInputIterator] =  TypeName< InputIterator >::get( );
        unaryTransformKernels[transform_oTypeU] = TypeName< oType >::get( );
        unaryTransformKernels[transform_DVOutputIteratorU] = TypeName< OutputIterator >::get( );
        unaryTransformKernels[transform_UnaryFunction] = TypeName< UnaryFunction >::get( );

        /**********************************************************************************
         * Type Definitions - directrly concatenated into kernel string
         *********************************************************************************/
        std::vector<std::string> typeDefinitions;
        PUSH_BACK_UNIQUE( typeDefinitions, user_code)
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< OutputIterator >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< iType >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< InputIterator >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< oType >::get() )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< UnaryFunction  >::get() )


        /**********************************************************************************
         * Calculate WG Size
         *********************************************************************************/
        cl_int l_Error = CL_SUCCESS;
        const size_t wgSize  = WAVEFRONT_SIZE;
        int boundsCheck = 0;

        V_OPENCL( l_Error, "Error querying kernel for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" );
        assert( (wgSize & (wgSize-1) ) == 0 ); // The bitwise &,~ logic below requires wgSize to be a power of 2

        size_t wgMultiple = sz;
        size_t lowerBits = ( sz & (wgSize-1) );
        if( lowerBits )
        {
            //  Bump the workitem count to the next multiple of wgSize
            wgMultiple &= ~lowerBits;
            wgMultiple += wgSize;
        }
        else
        {
            boundsCheck = 1;
        }

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
        TransformUnary_KernelTemplateSpecializer<InputIterator, OutputIterator> ts_kts;
        
        std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
            ctl,
            unaryTransformKernels,
            &ts_kts,
            typeDefinitions,
            /*transform_kernels + */ts_kts.getUnaryNoBoundsKernelPrototype() + ts_kts.getUnaryBoundsKernelPrototype(),
            compileOptions);
        // kernels returned in same order as added in KernelTemplaceSpecializer constructor

        // Create buffer wrappers so we can access the host functors, for read or writing in the kernel
        ALIGNED( 256 ) UnaryFunction aligned_binary( f );
        control::buffPointer userFunctor = ctl.acquireBuffer( sizeof( aligned_binary ),
        CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, &aligned_binary );

        typename InputIterator::Payload  first_payload  = first.gpuPayload( );
        typename OutputIterator::Payload result_payload = result.gpuPayload( );

        int arg_num = 0;
        /*Set the arguments for the kernel for InputIterator. Note that this takes input as the argument 
          number to start setting the values. The return value is the number of the argument to begin setting 
          the next Kernel arguments. 
          Once the cl::Buffer arguments are set the GPU Payload arguments are also passed to the kernel*/
        arg_num = first.setKernelBuffers(arg_num, kernels[boundsCheck]);
        kernels[boundsCheck].setArg(arg_num, first.gpuPayloadSize( ),&first_payload);
        arg_num++;

        /*Do the same for OutputIterator*/
        arg_num = result.setKernelBuffers(arg_num, kernels[boundsCheck]);
        kernels[boundsCheck].setArg(arg_num, result.gpuPayloadSize( ),&result_payload);
        arg_num++;

        //The type cast to int is required because sz is of type size_t
        kernels[boundsCheck].setArg(arg_num, (int)sz );
        kernels[boundsCheck].setArg(arg_num+1, *userFunctor);


        ::cl::Event transformEvent;
        l_Error = ctl.getCommandQueue().enqueueNDRangeKernel(
            kernels[boundsCheck],
            ::cl::NullRange,
            ::cl::NDRange( wgMultiple ), // numThreads
            ::cl::NDRange( wgSize ),
            NULL,
            &transformEvent );
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for transform() kernel" );

        ::bolt::cl::wait(ctl, transformEvent);
   
#if TRANSFORM_ENABLE_PROFILING
        if( 0 )
        {
          cl_ulong start_time, stop_time;

          l_Error = transformEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start_time);
          V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()");
          l_Error = transformEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &stop_time);
          V_OPENCL( l_Error, "failed on getProfilingInfo<CL_PROFILING_COMMAND_END>()");
          size_t time = stop_time - start_time;
          //std::cout << "Global Memory Bandwidth: "<<((distVec*(1.0*sizeof(iType)+sizeof(oType)))/time)<< std::endl;

        }
#endif // BOLT_ENABLE_PROFILING

        return;
    }
    
    /*! \brief This template function overload is used strictly std random access vectors and OpenCL implementations. 
        \detail 
    */
    template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    typename std::enable_if< std::is_same< typename std::iterator_traits< OutputIterator >::iterator_category ,
                                       std::random_access_iterator_tag
                                     >::value
                           >::type
    unary_transform( ::bolt::cl::control &ctl, const InputIterator& first, const InputIterator& last,
    const OutputIterator& result, const UnaryFunction& f, const std::string& user_code )
    {
        //size_t sz = bolt::cl::distance(first, last);
        int sz = static_cast<int>(last - first);
        if (sz == 0)
            return;
        typedef typename std::iterator_traits<InputIterator>::value_type  iType;
        typedef typename std::iterator_traits<OutputIterator>::value_type oType;
        
        typedef typename InputIterator::pointer pointer;
        
        pointer first_pointer = bolt::cl::addressof(first) ;

        device_vector< iType > dvInput( first_pointer, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, true, ctl );
        device_vector< oType > dvOutput( result, sz, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, false, ctl );
        auto device_iterator_first = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                            first, dvInput.begin() );
        auto device_iterator_last  = bolt::cl::create_device_itr(
                                            typename bolt::cl::iterator_traits< InputIterator >::iterator_category( ), 
                                            last, dvInput.end() );
        cl::unary_transform(ctl, device_iterator_first, device_iterator_last, dvOutput.begin(), f, user_code);
        dvOutput.data( );
        return;
    }
} // namespace cl


    /*! \brief This template function overload is used strictly for device vectors and std random access vectors. 
        \detail Here we branch out into the SerialCpu, MultiCore TBB or The OpenCL code paths. 
    */
    template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
    typename std::enable_if< 
             !(std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value) 
                           >::type
    binary_transform(::bolt::cl::control& ctl, const InputIterator1& first1, const InputIterator1& last1, 
                     const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f,
                     const std::string& user_code)
    {
        typename InputIterator1::difference_type sz = bolt::cl::distance(first1, last1 );
        if (sz == 0)
            return;

        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
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
            dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORM,BOLTLOG::BOLT_SERIAL_CPU,"::Transform::SERIAL_CPU");
            #endif
            serial::binary_transform(ctl, first1, last1, first2, result, f );
            return;
        }
        else if( runMode == bolt::cl::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORM,BOLTLOG::BOLT_MULTICORE_CPU,"::Transform::MULTICORE_CPU");
            #endif
            btbb::binary_transform(ctl, first1, last1, first2, result, f);
#else
            throw std::runtime_error( "The MultiCoreCpu version of transform is not enabled to be built! \n" );
#endif
            return;
        }
        else
        {
            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORM,BOLTLOG::BOLT_OPENCL_GPU,"::Transform::OPENCL_GPU");
            #endif
            cl::binary_transform( ctl, first1, last1, first2, result, f, user_code );
            return;
        }       
        return;
    }
    

    /*! \brief This template function overload is used to seperate input_iterator and fancy_iterator as 
               destination iterators from all other iterators
        \detail This template function overload is used to seperate input_iterator and fancy_iterator as 
                destination iterators from all other iterators. We enable this overload and should result 
                in a compilation failure.
    */
    // TODO - test the below code path
    template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
    typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value 
                           >::type
    binary_transform(::bolt::cl::control& ctl, const InputIterator1& first1, const InputIterator1& last1, 
                     const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f,
                     const std::string& user_code)
    {
        //TODO - Shouldn't we support transform for input_iterator_tag also. 
        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     std::input_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of the type input_iterator_tag" );
        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     bolt::cl::fancy_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of type fancy_iterator_tag" );
    }

    /*! \brief This template function overload is used strictly for device vectors and std random access vectors. 
        \detail Here we branch out into the SerialCpu, MultiCore TBB or The OpenCL code paths. 
    */
    template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    typename std::enable_if< 
             !(std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value) 
                           >::type
    unary_transform(::bolt::cl::control& ctl, InputIterator& first,
         InputIterator& last,  OutputIterator& result,  UnaryFunction& f,
        const std::string& user_code)
    {
        typename InputIterator::difference_type sz = bolt::cl::distance(first, last );
        if (sz == 0)
            return;

        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
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
            dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORM,BOLTLOG::BOLT_SERIAL_CPU,"::Transform::SERIAL_CPU");
            #endif
            serial::unary_transform(ctl, first, last, result, f );
            return;
        }
        else if( runMode == bolt::cl::control::MultiCoreCpu )
        {
#if defined( ENABLE_TBB )
            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORM,BOLTLOG::BOLT_MULTICORE_CPU,"::Transform::MULTICORE_CPU");
            #endif
            btbb::unary_transform(ctl, first, last, result, f);
#else
            throw std::runtime_error( "The MultiCoreCpu version of transform is not enabled to be built! \n" );
#endif
            return;
        }
        else
        {
            #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_TRANSFORM,BOLTLOG::BOLT_OPENCL_GPU,"::Transform::OPENCL_GPU");
            #endif
            cl::unary_transform( ctl, first, last, result, f, user_code );
            return;
        }       
        return;
    }
    

    /*! \brief This template function overload is used to seperate input_iterator and fancy_iterator as destination iterators from all other iterators
        \detail This template function overload is used to seperate input_iterator and fancy_iterator as destination iterators from all other iterators. 
                We enable this overload and should result in a compilation failure.
    */
    // TODO - test the below code path
    template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
    typename std::enable_if< 
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             std::input_iterator_tag 
                           >::value ||
               std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                             bolt::cl::fancy_iterator_tag >::value 
                           >::type
    unary_transform(::bolt::cl::control& ctl, const InputIterator& first1,
        const InputIterator& last1, const OutputIterator& result, const UnaryFunction& f,
        const std::string& user_code)
    {
        //TODO - Shouldn't we support transform for input_iterator_tag also. 
        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     std::input_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of the type input_iterator_tag" );
        static_assert( std::is_same< typename std::iterator_traits< OutputIterator>::iterator_category, 
                                     bolt::cl::fancy_iterator_tag >::value , 
                       "Output vector should be a mutable vector. It cannot be of type fancy_iterator_tag" );
    }

} //End of detail namespace

// two-input transform, std:: iterator
template< typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction >
void transform( bolt::cl::control& ctl, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                OutputIterator result, BinaryFunction f, const std::string& user_code )
{
    detail::binary_transform( ctl, first1, last1, first2, result, f, user_code );
}

// default ::bolt::cl::control, two-input transform, std:: iterator
template< typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction >
void transform( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result,
                BinaryFunction f, const std::string& user_code )
{
    using bolt::cl::transform;
    transform( control::getDefault(), first1, last1, first2, result, f, user_code );
}

// one-input transform, std:: iterator
template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
void transform( ::bolt::cl::control& ctl, InputIterator first1, InputIterator last1, OutputIterator result,
                UnaryFunction f, const std::string& user_code )
{
    using bolt::cl::detail::unary_transform;
    detail::unary_transform( ctl, first1, last1, result, f, user_code );
}

// default control, one-input transform, std:: iterator
template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
void transform( InputIterator first1, InputIterator last1, OutputIterator result,
                UnaryFunction f, const std::string& user_code )
{
    using bolt::cl::transform;
    transform( control::getDefault(), first1, last1, result, f, user_code );
}

} //End of cl namespace
} //End of bolt namespace

#endif
