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

#define TEST_DOUBLE 1
#define TEST_DEVICE_VECTOR 1
#define TEST_CPU_DEVICE 0
#define GOOGLE_TEST 1
#define LARGE_SIZE 0
#if (GOOGLE_TEST == 1)

#include "common/stdafx.h"
#include "common/myocl.h"

#include <bolt/cl/transform.h>
#include <bolt/cl/functional.h>
#include <bolt/cl/iterator/constant_iterator.h>
#include <bolt/cl/iterator/counting_iterator.h>
#include <bolt/miniDump.h>

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
//  This is a compare routine for naked pointers.
template< typename T >
::testing::AssertionResult cmpArrays( const T ref, const T calc, size_t N )
{
    for( size_t i = 0; i < N; ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

template< typename T, size_t N >
::testing::AssertionResult cmpArrays( const T (&ref)[N], const T (&calc)[N] )
{
    for( size_t i = 0; i < N; ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

//  Primary class template for std::array types
//  The struct wrapper is necessary to partially specialize the member function
template< typename T, size_t N >
struct cmpStdArray
{
    static ::testing::AssertionResult cmpArrays( const std::array< T, N >& ref, const std::array< T, N >& calc )
    {
        for( size_t i = 0; i < N; ++i )
        {
            EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );
    }
};

//  A very generic template that takes two container, and compares their values assuming a vector interface
template< typename S, typename B >
::testing::AssertionResult cmpArrays( const S& ref, const B& calc )
{
    for( int i = 0; i < static_cast<int>(ref.size( ) ); ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process type parameterized tests

//  This class creates a C++ 'TYPE' out of a size_t value
template< size_t N >
class TypeValue
{
public:
    static const size_t value = N;
};


//  Test fixture class, used for the Type-parameterized tests
//  Namely, the tests that use std::array and TYPED_TEST_P macros
template< typename ArrayTuple >
class TransformArrayTest: public ::testing::Test
{
public:
    TransformArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        stdOutput = stdInput;
        boltInput = stdInput;
        boltOutput = stdInput;
    };

    virtual void TearDown( )
    {};

    virtual ~TransformArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType, ArraySize > stdInput, boltInput, stdOutput, boltOutput;
    int m_Errors;
};

//  Test fixture class, used for the Type-parameterized tests
//  Namely, the tests that use std::array and TYPED_TEST_P macros
template< typename ArrayTuple >
class UnaryTransformArrayTest: public ::testing::Test
{
public:
    UnaryTransformArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        boltInput = stdInput;
        boltOutput = stdInput;
        stdOutput = stdInput;
    };

    virtual void TearDown( )
    {};

    virtual ~UnaryTransformArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize =  std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType,ArraySize > stdInput, boltInput, stdOutput, boltOutput;
    int m_Errors;
};

TYPED_TEST_CASE_P( TransformArrayTest );

TYPED_TEST_P( TransformArrayTest, Normal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ),
                    bolt::cl::plus<ArrayType>());
    bolt::cl::transform( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                    bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    
}

TYPED_TEST_P( TransformArrayTest, GPU_DeviceNormal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel
    // MyOclContext oclgpu = initOcl(CL_DEVICE_TYPE_GPU, 0);
    //bolt::cl::control c_gpu(oclgpu._queue);  // construct control structure from the queue.

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ),
                    bolt::cl::plus<ArrayType>());
    bolt::cl::transform( c_gpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceNormal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ),
                    bolt::cl::plus<ArrayType>());
    bolt::cl::transform( c_cpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#endif

TYPED_TEST_P( TransformArrayTest, MultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), 
                    bolt::cl::multiplies<ArrayType>());
    bolt::cl::transform( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::multiplies<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    
}

TYPED_TEST_P( TransformArrayTest, Serial_MultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), 
                    bolt::cl::multiplies<ArrayType>());
    bolt::cl::transform( ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::multiplies<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    
}

TYPED_TEST_P( TransformArrayTest, MultiCore_MultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), 
                    bolt::cl::multiplies<ArrayType>());
    bolt::cl::transform( ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::multiplies<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    
}


TYPED_TEST_P( TransformArrayTest, GPU_DeviceMultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel
    // MyOclContext oclgpu = initOcl(CL_DEVICE_TYPE_GPU, 0);
    //bolt::cl::control c_gpu(oclgpu._queue);  // construct control structure from the queue.

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ),
                    bolt::cl::multiplies<ArrayType>());
    bolt::cl::transform( c_gpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), bolt::cl::multiplies<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceMultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ),
                    bolt::cl::multiplies<ArrayType>());
    bolt::cl::transform( c_cpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), 
                         bolt::cl::multiplies<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#endif
TYPED_TEST_P( TransformArrayTest, MinusFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ),
                    bolt::cl::minus<ArrayType>());
    bolt::cl::transform( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::minus<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );

}

TYPED_TEST_P( TransformArrayTest, GPU_DeviceMinusFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel
    // MyOclContext oclgpu = initOcl(CL_DEVICE_TYPE_GPU, 0);
    //bolt::cl::control c_gpu(oclgpu._queue);  // construct control structure from the queue.

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), 
                    bolt::cl::minus<ArrayType>());
    bolt::cl::transform( c_gpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::minus<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceMinusFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), 
                    bolt::cl::minus<ArrayType>());
    bolt::cl::transform( c_cpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                     bolt::cl::minus<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#endif

TYPED_TEST_CASE_P( UnaryTransformArrayTest );

TYPED_TEST_P( UnaryTransformArrayTest, Normal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::negate<ArrayType>());
    bolt::cl::transform( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), bolt::cl::negate<ArrayType>());    

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
}

TYPED_TEST_P( UnaryTransformArrayTest, Serial )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::negate<ArrayType>());
    bolt::cl::transform(ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), bolt::cl::negate<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
}

TYPED_TEST_P( UnaryTransformArrayTest, MultiCoreCPU )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::negate<ArrayType>());
    bolt::cl::transform(ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), bolt::cl::negate<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
}


TYPED_TEST_P( UnaryTransformArrayTest, GPU_DeviceNormal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel
    // MyOclContext oclgpu = initOcl(CL_DEVICE_TYPE_GPU, 0);
    //bolt::cl::control c_gpu(oclgpu._queue);  // construct control structure from the queue.

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::negate<ArrayType>());
    bolt::cl::transform( c_gpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),  TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::negate<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}


#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( UnaryTransformArrayTest, CPU_DeviceNormal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::negate<ArrayType>());
    bolt::cl::transform( c_cpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::negate<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#endif

TYPED_TEST_P( UnaryTransformArrayTest, MultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::square<ArrayType>());
    bolt::cl::transform( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), bolt::cl::square<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
}

TYPED_TEST_P( UnaryTransformArrayTest, GPU_DeviceMultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel
    // MyOclContext oclgpu = initOcl(CL_DEVICE_TYPE_GPU, 0);
    //bolt::cl::control c_gpu(oclgpu._queue);  // construct control structure from the queue.

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::square<ArrayType>());
    bolt::cl::transform( c_gpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::square<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( UnaryTransformArrayTest, CPU_DeviceMultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::square<ArrayType>());
    bolt::cl::transform( c_cpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::square<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#endif
TYPED_TEST_P( UnaryTransformArrayTest, MinusFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::square<ArrayType>());
    bolt::cl::transform( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), bolt::cl::square<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );

}

TYPED_TEST_P( UnaryTransformArrayTest, GPU_DeviceMinusFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel
    // MyOclContext oclgpu = initOcl(CL_DEVICE_TYPE_GPU, 0);
    //bolt::cl::control c_gpu(oclgpu._queue);  // construct control structure from the queue.

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::square<ArrayType>());
    bolt::cl::transform( c_gpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), 
                         bolt::cl::square<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( UnaryTransformArrayTest, CPU_DeviceMinusFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::transform( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end( ), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), bolt::cl::square<ArrayType>());
    bolt::cl::transform( c_cpu, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ), TransformArrayTest< gtest_TypeParam_ >::boltOutput.begin( ),
                         bolt::cl::square<ArrayType>());

    typename  ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdOutput, TransformArrayTest< gtest_TypeParam_ >::boltOutput );
    // FIXME - releaseOcl(ocl);
}
#endif

#if (TEST_CPU_DEVICE == 1)
REGISTER_TYPED_TEST_CASE_P( TransformArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction,
                                           MinusFunction, GPU_DeviceMinusFunction, CPU_DeviceNormal,
                                           CPU_DeviceMultipliesFunction, CPU_DeviceMinusFunction, 
                                           Serial_MultipliesFunction,
                                           MultiCore_MultipliesFunction);
REGISTER_TYPED_TEST_CASE_P( UnaryTransformArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction,
                                           MinusFunction, GPU_DeviceMinusFunction, CPU_DeviceNormal, 
                                           CPU_DeviceMultipliesFunction, CPU_DeviceMinusFunction);
#else
REGISTER_TYPED_TEST_CASE_P( TransformArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction,
                                           MinusFunction, GPU_DeviceMinusFunction, Serial_MultipliesFunction,
                                           MultiCore_MultipliesFunction );
REGISTER_TYPED_TEST_CASE_P( UnaryTransformArrayTest, Normal,Serial, MultiCoreCPU, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction,
                                           MinusFunction, GPU_DeviceMinusFunction );
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class TransformIntegerVector: public ::testing::TestWithParam< int >
{
public:

    TransformIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                               stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);

        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< int > stdInput, boltInput, stdOutput, boltOutput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformFloatVector: public ::testing::TestWithParam< int >
{
public:
    TransformFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                             stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< float > stdInput, boltInput, stdOutput, boltOutput;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleVector: public ::testing::TestWithParam< int >
{
public:
    TransformDoubleVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                              stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< double > stdInput, boltInput, stdOutput, boltOutput;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformIntegerDeviceVector( ): stdInput( GetParam( ) ),
                                     stdOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            //boltInput[i] = stdInput[i];
            //boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    }

protected:
    std::vector< int > stdInput, stdOutput;
    //bolt::cl::device_vector< int > boltInput, boltOutput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformFloatDeviceVector( ): stdInput( GetParam( ) ), 
                                   stdOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            //boltInput[i] = stdInput[i];
            //boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    }

protected:
    std::vector< float > stdInput, stdOutput;
    //bolt::cl::device_vector< float > boltInput, boltOutput;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformDoubleDeviceVector( ): stdInput( GetParam( ) ),
		                            stdOutput( GetParam( ) ) 
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            //boltInput[i] = stdInput[i];
            //boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    }

protected:
    std::vector< double > stdInput, stdOutput;
    //bolt::cl::device_vector< double > boltInput, boltOutput;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformIntegerNakedPointer( ): stdInput( new int[ GetParam( ) ] ), boltInput( new int[ GetParam( ) ] ), 
                                     stdOutput( new int[ GetParam( ) ] ), boltOutput( new int[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, rand);
        for (unsigned int i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
};

protected:
     int* stdInput;
     int* boltInput;
     int* stdOutput;
     int* boltOutput;

};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformFloatNakedPointer( ): stdInput( new float[ GetParam( ) ] ), boltInput( new float[ GetParam( ) ] ), 
                                   stdOutput( new float[ GetParam( ) ] ), boltOutput( new float[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, rand);
        for (unsigned int i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
    };

protected:
     float* stdInput;
     float* boltInput;
     float* stdOutput;
     float* boltOutput;
};

#if (TEST_DOUBLE ==1 )
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformDoubleNakedPointer( ): stdInput( new double[ GetParam( ) ] ), boltInput( new double[ GetParam( ) ] ),
                                    stdOutput( new double[ GetParam( ) ] ), boltOutput( new double[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, rand);

		for (size_t i = 0; i < size; i++)
        {
            boltInput[ i ] = stdInput[ i ];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
    };

protected:
     double* stdInput;
     double* boltInput;
     double* stdOutput;
     double* boltOutput;
};
#endif

TEST_P( TransformIntegerVector, Normal )
{
    
    //typedef std::iterator_traits<std::vector<int>::iterator>::value_type T;
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin( ), bolt::cl::plus<int>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<int>());

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<int>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<int>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

TEST_P( TransformIntegerVector, Serial )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //typedef std::iterator_traits<std::vector<int>::iterator>::value_type T;
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin( ), bolt::cl::plus<int>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<int>());

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<int>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<int>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

TEST_P( TransformIntegerVector, MultiCoreCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //typedef std::iterator_traits<std::vector<int>::iterator>::value_type T;
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin( ), bolt::cl::plus<int>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<int>());

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<int>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<int>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

TEST_P( TransformFloatVector, Normal )
{
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin(),bolt::cl::plus<float>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<float>());

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<float>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<float>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

}

TEST_P( TransformFloatVector, Serial )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin(),bolt::cl::plus<float>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<float>());

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<float>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<float>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

}

TEST_P( TransformFloatVector, MultiCoreCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin(),bolt::cl::plus<float>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<float>());

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<float>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<float>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

}

#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleVector, Inplace )
{
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin( ),
                    bolt::cl::plus<double>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ),
                         bolt::cl::plus<double>());

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),
                                                                                     stdInput.end( ) );
    std::vector< double >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),
                                                                                      boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<double>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<double>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

}
#endif
#if (TEST_DEVICE_VECTOR == 1)
TEST_P( TransformIntegerDeviceVector, Inplace )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< int > boltOutput(stdOutput.begin(), stdOutput.end());

    
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin( ), bolt::cl::plus<int>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<int>());

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    //UNARY TRANSFORM Test
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<int>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<int>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

TEST_P( TransformIntegerDeviceVector, Serial )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< int > boltOutput(stdOutput.begin(), stdOutput.end());
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin( ), bolt::cl::plus<int>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<int>());

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    //UNARY TRANSFORM Test
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<int>());
    bolt::cl::transform(ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<int>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

TEST_P( TransformIntegerDeviceVector, MultiCoreCPU )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< int > boltOutput(stdOutput.begin(), stdOutput.end());
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin( ), bolt::cl::plus<int>());
    bolt::cl::transform( ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<int>());

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    //UNARY TRANSFORM Test
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<int>());
    bolt::cl::transform(ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<int>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

TEST_P( TransformFloatDeviceVector, Inplace )
{
    bolt::cl::device_vector< float > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< float > boltOutput(stdOutput.begin(), stdOutput.end());
    
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin(),bolt::cl::plus<float>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<float>());

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    //UNARY TRANSFORM Test
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<float>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<float>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

TEST_P( TransformFloatDeviceVector, Serial )
{
    bolt::cl::device_vector< float > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< float > boltOutput(stdOutput.begin(), stdOutput.end());
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin(),bolt::cl::plus<float>());
    bolt::cl::transform(ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<float>());

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    //UNARY TRANSFORM Test
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<float>());
    bolt::cl::transform(ctl,  boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<float>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

TEST_P( TransformFloatDeviceVector, MultiCoreCPU )
{
    bolt::cl::device_vector< float > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< float > boltOutput(stdOutput.begin(), stdOutput.end());
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), stdOutput.begin(),bolt::cl::plus<float>());
    bolt::cl::transform(ctl, boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ), 
                         bolt::cl::plus<float>());

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    //UNARY TRANSFORM Test
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<float>());
    bolt::cl::transform(ctl,  boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<float>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleDeviceVector, Inplace )
{
    bolt::cl::device_vector< double > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< double > boltOutput(stdOutput.begin(), stdOutput.end());
    
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ),stdOutput.begin(),bolt::cl::plus<double>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), boltOutput.begin( ),
                         bolt::cl::plus<double>());

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector<double>::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );

    //UNARY TRANSFORM Test
    //  Calling the actual functions under test
    std::transform( stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::negate<double>());
    bolt::cl::transform( boltInput.begin( ), boltInput.end( ), boltOutput.begin( ), bolt::cl::negate<double>());

    stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
    boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}
#endif
#endif
#if defined(_WIN32)
TEST_P( TransformIntegerNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltOutput( boltOutput, endIndex );

    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, wrapStdOutput, bolt::cl::plus<int>());
    bolt::cl::transform( wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, wrapBoltOutput, bolt::cl::plus<int>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
    
    //UNARY TRANSFORM Test
    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<int>());
    bolt::cl::transform( wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, bolt::cl::negate<int>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
}

TEST_P( TransformIntegerNakedPointer, Serial )
{
    size_t endIndex = GetParam( );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltOutput( boltOutput, endIndex );

    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, wrapStdOutput, bolt::cl::plus<int>());
    bolt::cl::transform( wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, wrapBoltOutput, bolt::cl::plus<int>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
    
    //UNARY TRANSFORM Test
    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<int>());
    bolt::cl::transform( ctl, wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, bolt::cl::negate<int>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
}

TEST_P( TransformIntegerNakedPointer, MultiCoreCPU )
{
    size_t endIndex = GetParam( );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltOutput( boltOutput, endIndex );

    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, wrapStdOutput, bolt::cl::plus<int>());
    bolt::cl::transform( wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, wrapBoltOutput, bolt::cl::plus<int>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
    
    //UNARY TRANSFORM Test
    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<int>());
    bolt::cl::transform( ctl, wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, bolt::cl::negate<int>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
}

TEST_P( TransformFloatNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltOutput( boltOutput, endIndex );

    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, wrapStdOutput, bolt::cl::plus<float>());
    bolt::cl::transform( wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput,wrapBoltOutput,bolt::cl::plus<float>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );

    //UNARY TRANSFORM Test
    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<float>());
    bolt::cl::transform( wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, bolt::cl::negate<float>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
}

TEST_P( TransformFloatNakedPointer, Serial)
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltOutput( boltOutput, endIndex );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, wrapStdOutput, bolt::cl::plus<float>());
    bolt::cl::transform(ctl,  wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput,wrapBoltOutput,
                                                                        bolt::cl::plus<float>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );

    //UNARY TRANSFORM Test
    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<float>());
    bolt::cl::transform(ctl,  wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, bolt::cl::negate<float>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
}

TEST_P( TransformFloatNakedPointer, MultiCoreCPU)
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltOutput( boltOutput, endIndex );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, wrapStdOutput, bolt::cl::plus<float>());
    bolt::cl::transform(ctl,  wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput,wrapBoltOutput,
                                                                          bolt::cl::plus<float>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );

    //UNARY TRANSFORM Test
    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<float>());
    bolt::cl::transform(ctl,  wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, bolt::cl::negate<float>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
}

#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< double* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< double* > wrapBoltOutput( boltOutput, endIndex );

    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, wrapStdOutput, bolt::cl::plus<double>());
    bolt::cl::transform( wrapBoltInput, wrapBoltInput+endIndex,wrapBoltOutput,wrapBoltOutput,bolt::cl::plus<double>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );

    //UNARY TRANSFORM Test
    std::transform( wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<double>());
    bolt::cl::transform( wrapBoltInput, wrapBoltInput+endIndex, wrapBoltOutput, bolt::cl::negate<double>());

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput, endIndex );
}

#endif
#endif

std::array<int, 10> TestValues = {2,4,8,16,32,64,128,256,512,1024};
std::array<int, 5> TestValues2 = {2048,4096,8192,16384,32768};
//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                         TestValues.end() ) );

INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatVector, ::testing::Range(4096, 65536, 555 ) ); //2^12 to 2^16
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatVector, ::testing::ValuesIn( TestValues.begin(), 
                         TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( TransformValues, TransformDoubleVector, ::testing::ValuesIn( TestValues.begin(), 
                         TestValues.end() ) );
//#if LARGE
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformDoubleVector, ::testing::ValuesIn( TestValues2.begin(), 
                         TestValues2.end() ) );
//#endif
#endif
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerDeviceVector, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                         TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                         TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformDoubleDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                         TestValues.end() ) );
//#if LARGE
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformDoubleDeviceVector, ::testing::ValuesIn( TestValues2.begin(),
                         TestValues2.end() ) );
//#endif
#endif
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                         TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatNakedPointer, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                         TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleNakedPointer, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( Transform, TransformDoubleNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                         TestValues.end() ) );
//#if LARGE
INSTANTIATE_TEST_CASE_P( Transform2, TransformDoubleNakedPointer, ::testing::ValuesIn( TestValues2.begin(), 
                         TestValues2.end() ) );
//#endif
#endif

//#if LARGE
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformIntegerVector, ::testing::ValuesIn( TestValues2.begin(),
                         TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformFloatVector, ::testing::ValuesIn( TestValues2.begin(), 
                         TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformIntegerDeviceVector, ::testing::ValuesIn( TestValues2.begin(),
                         TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformFloatDeviceVector, ::testing::ValuesIn( TestValues2.begin(), 
                         TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformIntegerNakedPointer, ::testing::ValuesIn( TestValues2.begin(), 
                         TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformFloatNakedPointer, ::testing::ValuesIn( TestValues2.begin(),
                         TestValues2.end() ) );
//#endif

typedef ::testing::Types< 
    std::tuple< int, TypeValue< 1 > >,
    std::tuple< int, TypeValue< 31 > >,
    std::tuple< int, TypeValue< 32 > >,
    std::tuple< int, TypeValue< 63 > >,
    std::tuple< int, TypeValue< 64 > >,
    std::tuple< int, TypeValue< 127 > >,
    std::tuple< int, TypeValue< 128 > >,
    std::tuple< int, TypeValue< 129 > >,
    std::tuple< int, TypeValue< 1000 > >,
    std::tuple< int, TypeValue< 1053 > >,
    std::tuple< int, TypeValue< 4096 > >,
    std::tuple< int, TypeValue< 4097 > >,
    std::tuple< int, TypeValue< 65535 > >,
    std::tuple< int, TypeValue< 65536 > >
> IntegerTests;

typedef ::testing::Types< 
    std::tuple< float, TypeValue< 1 > >,
    std::tuple< float, TypeValue< 31 > >,
    std::tuple< float, TypeValue< 32 > >,
    std::tuple< float, TypeValue< 63 > >,
    std::tuple< float, TypeValue< 64 > >,
    std::tuple< float, TypeValue< 127 > >,
    std::tuple< float, TypeValue< 128 > >,
    std::tuple< float, TypeValue< 129 > >,
    std::tuple< float, TypeValue< 1000 > >,
    std::tuple< float, TypeValue< 1053 > >,
    std::tuple< float, TypeValue< 4096 > >,
    std::tuple< float, TypeValue< 4097 > >,
    std::tuple< float, TypeValue< 65535 > >,
    std::tuple< float, TypeValue< 65536 > >
> FloatTests;

typedef ::testing::Types< 
    std::tuple< long, TypeValue< 1 > >,
    std::tuple< long, TypeValue< 31 > >,
    std::tuple< long, TypeValue< 32 > >,
    std::tuple< long, TypeValue< 63 > >,
    std::tuple< long, TypeValue< 64 > >,
    std::tuple< long, TypeValue< 127 > >,
    std::tuple< long, TypeValue< 128 > >,
    std::tuple< long, TypeValue< 129 > >,
    std::tuple< long, TypeValue< 1000 > >,
    std::tuple< long, TypeValue< 1053 > >,
    std::tuple< long, TypeValue< 4096 > >,
    std::tuple< long, TypeValue< 4097 > >
> LongTests;


// todo: Run long with larger sizes

typedef ::testing::Types< 
    std::tuple< unsigned long, TypeValue< 1 > >,
    std::tuple< unsigned long, TypeValue< 31 > >,
    std::tuple< unsigned long, TypeValue< 32 > >,
    std::tuple< unsigned long, TypeValue< 63 > >,
    std::tuple< unsigned long, TypeValue< 64 > >,
    std::tuple< unsigned long, TypeValue< 127 > >,
    std::tuple< unsigned long, TypeValue< 128 > >,
    std::tuple< unsigned long, TypeValue< 129 > >,
    std::tuple< unsigned long, TypeValue< 1000 > >,
    std::tuple< unsigned long, TypeValue< 1053 > >,
    std::tuple< unsigned long, TypeValue< 4096 > >,
    std::tuple< unsigned long, TypeValue< 4097 > >
> ULongTests;

// todo: Run ulong with larger sizes

typedef ::testing::Types< 
    std::tuple< short, TypeValue< 1 > >,
    std::tuple< short, TypeValue< 31 > >,
    std::tuple< short, TypeValue< 32 > >,
    std::tuple< short, TypeValue< 63 > >,
    std::tuple< short, TypeValue< 64 > >,
    std::tuple< short, TypeValue< 127 > >,
    std::tuple< short, TypeValue< 128 > >,
    std::tuple< short, TypeValue< 129 > >,
    std::tuple< short, TypeValue< 1000 > >,
    std::tuple< short, TypeValue< 1053 > >,
    std::tuple< short, TypeValue< 4096 > >,
    std::tuple< short, TypeValue< 4097 > >,
    std::tuple< short, TypeValue< 65535 > >,
    std::tuple< short, TypeValue< 65536 > >
> ShortTests;

typedef ::testing::Types< 
    std::tuple< unsigned short, TypeValue< 1 > >,
    std::tuple< unsigned short, TypeValue< 31 > >,
    std::tuple< unsigned short, TypeValue< 32 > >,
    std::tuple< unsigned short, TypeValue< 63 > >,
    std::tuple< unsigned short, TypeValue< 64 > >,
    std::tuple< unsigned short, TypeValue< 127 > >,
    std::tuple< unsigned short, TypeValue< 128 > >,
    std::tuple< unsigned short, TypeValue< 129 > >,
    std::tuple< unsigned short, TypeValue< 1000 > >,
    std::tuple< unsigned short, TypeValue< 1053 > >,
    std::tuple< unsigned short, TypeValue< 4096 > >,
    std::tuple< unsigned short, TypeValue< 4097 > >,
    std::tuple< unsigned short, TypeValue< 65535 > >,
    std::tuple< unsigned short, TypeValue< 65536 > >
> UShortTests;


#if (TEST_DOUBLE == 1)
typedef ::testing::Types< 
    std::tuple< double, TypeValue< 1 > >,
    std::tuple< double, TypeValue< 31 > >,
    std::tuple< double, TypeValue< 32 > >,
    std::tuple< double, TypeValue< 63 > >,
    std::tuple< double, TypeValue< 64 > >,
    std::tuple< double, TypeValue< 127 > >,
    std::tuple< double, TypeValue< 128 > >,
    std::tuple< double, TypeValue< 129 > >,
    std::tuple< double, TypeValue< 1000 > >,
    std::tuple< double, TypeValue< 1053 > >,
    std::tuple< double, TypeValue< 4096 > >,
    std::tuple< double, TypeValue< 4097 > >,
    std::tuple< double, TypeValue< 65535 > >,
    std::tuple< double, TypeValue< 65536 > >
> DoubleTests;
#endif 
BOLT_FUNCTOR(UDD,
struct UDD
{
    int a; 
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) const {
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    } 
    bool operator < (const UDD& other) const {
        return ((a+b) < (other.a+other.b));
    }
    bool operator > (const UDD& other) const {
        return ((a+b) > (other.a+other.b));
    }
    bool operator == (const UDD& other) const {
        return ((a+b) == (other.a+other.b));
    }

    UDD operator + (const UDD &rhs) const
    {
      UDD _result;
      _result.a = a + rhs.a;
      _result.b = b + rhs.b;
      return _result;
    }

    UDD()
        : a(0),b(0) { }
    UDD(int _in)
        : a(_in), b(_in +1)  { }
};
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< UDD >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< UDD >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::plus, int, UDD );

typedef ::testing::Types< 
    std::tuple< UDD, TypeValue< 1 > >,
    std::tuple< UDD, TypeValue< 31 > >,
    std::tuple< UDD, TypeValue< 32 > >,
    std::tuple< UDD, TypeValue< 63 > >,
    std::tuple< UDD, TypeValue< 64 > >,
    std::tuple< UDD, TypeValue< 127 > >,
    std::tuple< UDD, TypeValue< 128 > >,
    std::tuple< UDD, TypeValue< 129 > >,
    std::tuple< UDD, TypeValue< 1000 > >,
    std::tuple< UDD, TypeValue< 1053 > >,
    std::tuple< UDD, TypeValue< 4096 > >,
    std::tuple< UDD, TypeValue< 4097 > >,
    std::tuple< UDD, TypeValue< 65535 > >,
    std::tuple< UDD, TypeValue< 65536 > >
> UDDTests;
/*
INSTANTIATE_TYPED_TEST_CASE_P( Integer, TransformArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, TransformArrayTest, FloatTests );
//INSTANTIATE_TYPED_TEST_CASE_P( Long, TransformArrayTest, LongTests );
//INSTANTIATE_TYPED_TEST_CASE_P( ULong, TransformArrayTest, ULongTests );
INSTANTIATE_TYPED_TEST_CASE_P( Short, TransformArrayTest, ShortTests );
//INSTANTIATE_TYPED_TEST_CASE_P( UShort, TransformArrayTest, UShortTests );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TYPED_TEST_CASE_P( Double, TransformArrayTest, DoubleTests );
#endif 
//INSTANTIATE_TYPED_TEST_CASE_P( UDDTest, SortArrayTest, UDDTests );

INSTANTIATE_TYPED_TEST_CASE_P( Integer, UnaryTransformArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, UnaryTransformArrayTest, FloatTests );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TYPED_TEST_CASE_P( Double, UnaryTransformArrayTest, DoubleTests );
#endif 
//INSTANTIATE_TYPED_TEST_CASE_P( UDDTest, SortArrayTest, UDDTests );
*/
//Teststotesttheconstantiterator
template <int N>
int gen_constant()
{
    return N;
}
TEST(simple,constant)
{
    bolt::cl::constant_iterator<int> iter( 1 );
    bolt::cl::constant_iterator<int> iter2 = iter + 1024;
    std::vector<int> input1(1024);
    std::vector<int> input2(1024);
    std::vector<int> stdOutput(1024);
     std::vector<int> boltOutput(1024);
    std::generate(input1.begin(),input1.end(),gen_constant<1>);
    input2 = input1;
    std::transform( input1.begin(), input1.end(), input2.begin(), stdOutput.begin(), bolt::cl::plus<int>());
    bolt::cl::transform(iter,iter2,input1.begin(),boltOutput.begin(),bolt::cl::plus<int>());
    cmpArrays( stdOutput, boltOutput, 1024 );
}

TEST(simple,Serial)
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::constant_iterator<int> iter( 1 );
    bolt::cl::constant_iterator<int> iter2 = iter + 1024;
    std::vector<int> input1(1024);
    std::vector<int> input2(1024);
    std::vector<int> stdOutput(1024);
     std::vector<int> boltOutput(1024);
    std::generate(input1.begin(),input1.end(),gen_constant<1>);
    input2 = input1;
    std::transform( input1.begin(), input1.end(), input2.begin(), stdOutput.begin(), bolt::cl::plus<int>());
    bolt::cl::transform(ctl, iter,iter2,input1.begin(),boltOutput.begin(),bolt::cl::plus<int>());
    cmpArrays( stdOutput, boltOutput, 1024 );
}

TEST(simple,MultiCoreCPU)
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::constant_iterator<int> iter( 1 );
    bolt::cl::constant_iterator<int> iter2 = iter + 1024;
    std::vector<int> input1(1024);
    std::vector<int> input2(1024);
    std::vector<int> stdOutput(1024);
     std::vector<int> boltOutput(1024);
    std::generate(input1.begin(),input1.end(),gen_constant<1>);
    input2 = input1;
    std::transform( input1.begin(), input1.end(), input2.begin(), stdOutput.begin(), bolt::cl::plus<int>());
    bolt::cl::transform(ctl, iter,iter2,input1.begin(),boltOutput.begin(),bolt::cl::plus<int>());
    cmpArrays( stdOutput, boltOutput, 1024 );
}

//Teststotestthecountingiterator
TEST(simple1,counting)
{
    bolt::cl::counting_iterator<int> iter(0);
    bolt::cl::counting_iterator<int> iter2=iter+1024;
    std::vector<int> input1(1024);
    std::vector<int> input2(1024);
    std::vector<int> stdOutput(1024);
     std::vector<int> boltOutput(1024);
     for(int i=0 ; i< 1024;i++)
     {
         input1[i] = i;
     }
    input2 = input1;
    std::transform( input1.begin(), input1.end(), input2.begin(), stdOutput.begin(), bolt::cl::plus<int>());
    bolt::cl::transform(iter,iter2,input1.begin(),boltOutput.begin(),bolt::cl::plus<int>());
    cmpArrays( stdOutput, boltOutput, 1024 );
}

TEST(simple1,Serial_counting)
{
    bolt::cl::counting_iterator<int> iter(0);
    bolt::cl::counting_iterator<int> iter2=iter+1024;
    std::vector<int> input1(1024);
    std::vector<int> input2(1024);
    std::vector<int> stdOutput(1024);
     std::vector<int> boltOutput(1024);
     for(int i=0 ; i< 1024;i++)
     {
         input1[i] = i;
     }
    input2 = input1;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    std::transform( input1.begin(), input1.end(), input2.begin(), stdOutput.begin(), bolt::cl::plus<int>());
    bolt::cl::transform(ctl, iter,iter2,input1.begin(),boltOutput.begin(),bolt::cl::plus<int>());
    cmpArrays( stdOutput, boltOutput, 1024 );
}

TEST(simple1,MultiCore_counting)
{
    bolt::cl::counting_iterator<int> iter(0);
    bolt::cl::counting_iterator<int> iter2=iter+1024;
    std::vector<int> input1(1024);
    std::vector<int> input2(1024);
    std::vector<int> stdOutput(1024);
     std::vector<int> boltOutput(1024);
     for(int i=0 ; i< 1024;i++)
     {
         input1[i] = i;
     }
    input2 = input1;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    std::transform( input1.begin(), input1.end(), input2.begin(), stdOutput.begin(), bolt::cl::plus<int>());
    bolt::cl::transform(ctl, iter,iter2,input1.begin(),boltOutput.begin(),bolt::cl::plus<int>());
    cmpArrays( stdOutput, boltOutput, 1024 );
}


TEST(cl_const_iter_transformBoltClVectFloat, addIterFloatValues){
    int size = 10;
    float myConstValueF = 1.125f;
    bolt::cl::plus<float> addKaro;
    std::vector<float> stdVect(size);
    
    for (int i = 0; i < size; i++){
        stdVect[i] = (float)i + 1.0f;
    }
    bolt::cl::device_vector<float> myDevVect(stdVect.begin(), stdVect.end());
    
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myDevVect[i]<<" ";
    }

    bolt::cl::constant_iterator< float > constIter( myConstValueF );

    bolt::cl::transform(myDevVect.begin(), myDevVect.end(), constIter, myDevVect.begin(), addKaro);
    
    std::cout<<std::endl<<std::endl;
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myDevVect[i]<<" ";
    }
}


TEST(cl_const_iter_transformBoltClVectFloat, SerialaddIterFloatValues){
    int size = 10;
    float myConstValueF = 1.125f;
    bolt::cl::plus<float> addKaro;
    std::vector<float> stdVect(size);
    
    for (int i = 0; i < size; i++){
        stdVect[i] = (float)i + 1.0f;
    }
    bolt::cl::device_vector<float> myDevVect(stdVect.begin(), stdVect.end());
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myDevVect[i]<<" ";
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::constant_iterator< float > constIter( myConstValueF );

    bolt::cl::transform(ctl, myDevVect.begin(), myDevVect.end(), constIter, myDevVect.begin(), addKaro);
    
    std::cout<<std::endl<<std::endl;
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myDevVect[i]<<" ";
    }
}


TEST(cl_const_iter_transformBoltClVectFloat, MultiCoreaddIterFloatValues){
    int size = 10;
    float myConstValueF = 1.125f;
    bolt::cl::plus<float> addKaro;
    std::vector<float> stdVect(size);
    
    for (int i = 0; i < size; i++){
        stdVect[i] = (float)i + 1.0f;
    }
    bolt::cl::device_vector<float> myDevVect(stdVect.begin(), stdVect.end());
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myDevVect[i]<<" ";
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::constant_iterator< float > constIter( myConstValueF );

    bolt::cl::transform(ctl, myDevVect.begin(), myDevVect.end(), constIter, myDevVect.begin(), addKaro);
    
    std::cout<<std::endl<<std::endl;
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myDevVect[i]<<" ";
    }
}

TEST(cl_const_iter_transformBoltClFloat, addIterFloatValues){
    int size = 10;
    float myConstValueF = 1.125f;
    bolt::cl::plus<float> addKaro;
    std::vector<float> myVect(size);
    
    for (int i = 0; i < size; i++){
        myVect[i] = (float)i + 1.0f;
    }
    
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myVect[i]<<" ";
    }

    bolt::cl::constant_iterator< float > constIter( myConstValueF );

    bolt::cl::transform(myVect.begin(), myVect.end(), constIter, myVect.begin(), addKaro);
    
    std::cout<<std::endl<<std::endl;
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myVect[i]<<" ";
    }
}

TEST(cl_const_iter_transformBoltClFloat, SerialaddIterFloatValues){
    int size = 10;
    float myConstValueF = 1.125f;
    bolt::cl::plus<float> addKaro;
    std::vector<float> myVect(size);
    
    for (int i = 0; i < size; i++){
        myVect[i] = (float)i + 1.0f;
    }
    
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myVect[i]<<" ";
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::constant_iterator< float > constIter( myConstValueF );

    bolt::cl::transform(ctl, myVect.begin(), myVect.end(), constIter, myVect.begin(), addKaro);
    
    std::cout<<std::endl<<std::endl;
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myVect[i]<<" ";
    }
}

TEST(cl_const_iter_transformBoltClFloat, MultiCoreaddIterFloatValues){
    int size = 10;
    float myConstValueF = 1.125f;
    bolt::cl::plus<float> addKaro;
    std::vector<float> myVect(size);
    
    for (int i = 0; i < size; i++){
        myVect[i] = (float)i + 1.0f;
    }
    
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myVect[i]<<" ";
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::constant_iterator< float > constIter( myConstValueF );

    bolt::cl::transform(ctl, myVect.begin(), myVect.end(), constIter, myVect.begin(), addKaro);
    
    std::cout<<std::endl<<std::endl;
    for (int i = 0; i<size; ++i){
        std::cout.setf(std::ios::fixed);
        std::cout<<std::setprecision(3)<<myVect[i]<<" ";
    }
}

TEST( TransformDeviceVector, OutOfPlaceTransform)
{
  int length = 1<<8;
  std::vector<int> hVectorA( length ), hVectorB( length ), hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 0 );

  bolt::cl::device_vector<int> dVectorAAV(hVectorA.begin(), hVectorA.end()),
    dVectorBAV(hVectorB.begin(), hVectorB.end()),
    dVectorOAV(hVectorO.begin(), hVectorO.end());

  std::transform( hVectorA.begin(), hVectorA.end(), hVectorB.begin(), hVectorO.begin(), std::plus< int >( ) );
  bolt::cl::transform( dVectorAAV.begin(),
    dVectorAAV.end(),
    dVectorBAV.begin(),
    dVectorOAV.begin(),
    bolt::cl::plus< int >( ) );
}

TEST( TransformDeviceVector, SerialOutOfPlaceTransform)
{
  int length = 1<<8;
  std::vector<int> hVectorA( length ), hVectorB( length ), hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 0 );

  bolt::cl::device_vector<int> dVectorAAV(hVectorA.begin(), hVectorA.end()),
  dVectorBAV(hVectorB.begin(), hVectorB.end()),
  dVectorOAV(hVectorO.begin(), hVectorO.end());

  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::SerialCpu);

  std::transform( hVectorA.begin(), hVectorA.end(), hVectorB.begin(), hVectorO.begin(), std::plus< int >( ) );
  bolt::cl::transform(ctl,  dVectorAAV.begin(),
    dVectorAAV.end(),
    dVectorBAV.begin(),
    dVectorOAV.begin(),
    bolt::cl::plus< int >( ) );
}


BOLT_FUNCTOR(StringFunctor,
struct StringFunctor
{
    char operator() ( const char& in )
    {
        return in + 1;
    };
};
);

BOLT_FUNCTOR(UnsignedStringFunctor,
struct UnsignedStringFunctor
{
    unsigned char operator() ( const unsigned char& in )
    {
        return in + 1;
    };
};
);


TEST(TransformStdVect, BoltVideo)
{
    // Initialize and print an input string
    std::string inputString( "GdkknVnqkc" );

    // Create input and ourput buffers
    std::vector< char > inputVec( inputString.begin(), inputString.end() );
    std::vector< char > outputVec( inputString.size() );

    // This calls into OpenCL to do the work
    bolt::cl::transform( inputVec.begin(), inputVec.end(), outputVec.begin(), StringFunctor() );

    // transfer data to host memory from device memory.
    std::string outputString( outputVec.data(), outputVec.size() );
    std::string expectedString("HelloWorld");
    EXPECT_TRUE(outputString==expectedString);


}

TEST(TransformStdVect, BoltVideoUnsignedChar)
{
    // Initialize and print an input string
    std::string inputString( "GdkknVnqkc" );

    // Create input and ourput buffers
    std::vector< unsigned char > inputVec( inputString.begin(), inputString.end() );
    std::vector< unsigned char > outputVec( inputString.size() );

    // This calls into OpenCL to do the work
    bolt::cl::transform( inputVec.begin(), inputVec.end(), outputVec.begin(), UnsignedStringFunctor() );

    // transfer data to host memory from device memory.
    std::string outputString(reinterpret_cast<const char *>(outputVec.data()), outputVec.size()  );
    std::string expectedString("HelloWorld");
    EXPECT_TRUE(outputString==expectedString);


}

TEST( TransformUDD, UDDTestStdVector)
{

#if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 32768; //2^15
#endif
  std::vector<UDD> hVectorA( length ), hVectorB( length ), hVectorO( length ), hVectorBoltO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );
  std::fill( hVectorBoltO.begin(), hVectorBoltO.end(), 0 );

  bolt::cl::plus<UDD> pl;
  std::transform( hVectorA.begin(), hVectorA.end(), hVectorB.begin(), hVectorO.begin(), std::plus< UDD >( ) );
  bolt::cl::transform( hVectorA.begin(),
    hVectorA.end(),
    hVectorB.begin(),
    hVectorBoltO.begin(), pl );

  cmpArrays(hVectorO, hVectorBoltO);

}


TEST( TransformUDD, UDDTestDeviceVector)
{

#if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 32768; //2^15
#endif
  std::vector<UDD> hVectorA( length ),
                   hVectorB( length ),
                   hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::plus<UDD> pl;
  bolt::cl::device_vector<UDD> dVectorA( hVectorA.begin(), hVectorA.end() ),
                               dVectorB( hVectorB.begin(), hVectorB.end() ),
                               dVectorO( hVectorO.begin(), hVectorO.end() );
  std::transform( hVectorA.begin(), hVectorA.end(), hVectorB.begin(), hVectorO.begin(), std::plus< UDD >( ) );
  bolt::cl::transform( dVectorA.begin(),
    dVectorA.end(),
    dVectorB.begin(),
    dVectorO.begin(), pl );

  cmpArrays(hVectorO, dVectorO);

}


TEST( TransformLong, LongTests)
{

#if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 32768; //2^15
#endif
  std::vector<cl_long> hVectorA( length ),
                   hVectorB( length ),
                   hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::plus<cl_long> pl;
  bolt::cl::device_vector<cl_long> dVectorA( hVectorA.begin(), hVectorA.end() ),
                               dVectorB( hVectorB.begin(), hVectorB.end() ),
                               dVectorO( hVectorO.begin(), hVectorO.end() );
  std::transform( hVectorA.begin(), hVectorA.end(), hVectorB.begin(), hVectorO.begin(), std::plus< cl_long >( ) );
  bolt::cl::transform( dVectorA.begin(),
    dVectorA.end(),
    dVectorB.begin(),
    dVectorO.begin(), pl );

  cmpArrays(hVectorO, dVectorO);

}

TEST( TransformULong, ULongTests)
{

#if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 32768; //2^15
#endif
  std::vector<cl_ulong> hVectorA( length ),
                   hVectorB( length ),
                   hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::plus<cl_ulong> pl;
  bolt::cl::device_vector<cl_ulong> dVectorA( hVectorA.begin(), hVectorA.end() ),
                               dVectorB( hVectorB.begin(), hVectorB.end() ),
                               dVectorO( hVectorO.begin(), hVectorO.end() );
  std::transform( hVectorA.begin(), hVectorA.end(), hVectorB.begin(), hVectorO.begin(), std::plus< cl_ulong >( ) );
  bolt::cl::transform( dVectorA.begin(),
    dVectorA.end(),
    dVectorB.begin(),
    dVectorO.begin(), pl );

  cmpArrays(hVectorO, dVectorO);

}

TEST(TransformStd, OffsetTest)
{
  #if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 32768; //2^15
#endif
  std::vector<int> hVectorA( length ), hVectorB( length ), hVectorO( length ), hVectorBoltO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );
  std::fill( hVectorBoltO.begin(), hVectorBoltO.end(), 0 );

  std::transform( hVectorA.begin()+70, hVectorA.begin()+135, hVectorB.begin()+85, hVectorO.begin()+150, std::plus< int >( ) );
  bolt::cl::transform( hVectorA.begin()+70,
    hVectorA.begin()+135,
    hVectorB.begin()+85,
    hVectorBoltO.begin()+150, bolt::cl::plus<int>( ) );

  cmpArrays(hVectorO, hVectorBoltO);


}

TEST(TransformStd, OffsetTestDeviceVector)
{
  #if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 32768; //2^15
#endif
  std::vector<int> hVectorA( length ), hVectorB( length ), hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::device_vector<int> dVectorA( hVectorA.begin(), hVectorA.end() ),
                                         dVectorB( hVectorB.begin(), hVectorB.end() ),
                                         dVectorO( hVectorO.begin(), hVectorO.end() );

  std::transform( hVectorA.begin()+70, hVectorA.begin()+135, hVectorB.begin()+85, hVectorO.begin()+150, std::plus< int >( ) );
  bolt::cl::transform( dVectorA.begin()+70,
    dVectorA.begin()+135,
    dVectorB.begin()+85,
    dVectorO.begin()+150, bolt::cl::plus<int>( ) );

  cmpArrays(hVectorO, dVectorO);


}

TEST(TransformStd, OffsetTestDeviceVectorSerialCPU)
{
  #if LARGE_SIZE
    int length = 33554432; //2^25
  #else
    int length = 32768; //2^15
  #endif

  std::vector<int> hVectorA( length ), hVectorB( length ), hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::device_vector<int> dVectorA( hVectorA.begin(), hVectorA.end() ),
                                         dVectorB( hVectorB.begin(), hVectorB.end() ),
                                         dVectorO( hVectorO.begin(), hVectorO.end() );

  bolt::cl::control ctl;
  ctl.setForceRunMode(bolt::cl::control::SerialCpu);
  std::transform( hVectorA.begin()+70, hVectorA.begin()+135, hVectorB.begin()+85, hVectorO.begin()+150, std::plus< int >( ) );
  bolt::cl::transform( ctl,
                       dVectorA.begin()+70,
                       dVectorA.begin()+135,
                       dVectorB.begin()+85,
                       dVectorO.begin()+150, bolt::cl::plus<int>( ) );

  cmpArrays(hVectorO, dVectorO);


}

TEST(TransformStd, OffsetTestDeviceVectorMultiCoreCPU)
{
  #if LARGE_SIZE
    int length = 33554432; //2^25
  #else
    int length = 32768; //2^15
  #endif
  std::vector<int> hVectorA( length ), hVectorB( length ), hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::device_vector<int> dVectorA( hVectorA.begin(), hVectorA.end() ),
                                         dVectorB( hVectorB.begin(), hVectorB.end() ),
                                         dVectorO( hVectorO.begin(), hVectorO.end() );

  bolt::cl::control ctl;
  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
  std::transform( hVectorA.begin()+70, hVectorA.begin()+135, hVectorB.begin()+85, hVectorO.begin()+150, std::plus< int >( ) );
  bolt::cl::transform( ctl,
                       dVectorA.begin()+70,
                       dVectorA.begin()+135,
                       dVectorB.begin()+85,
                       dVectorO.begin()+150, bolt::cl::plus<int>( ) );

  cmpArrays(hVectorO, dVectorO);


}


TEST(TransformStd, OffsetTestMultiCoreCPU)
{
  #if LARGE_SIZE
    int length = 33554432; //2^25
  #else
    int length = 32768; //2^15
  #endif
  std::vector<cl_long> hVectorA( length ), hVectorB( length ), hVectorO( length ), hVectorDO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 1024 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1024 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );
  std::fill( hVectorDO.begin(), hVectorDO.end(), 0 );


  bolt::cl::control ctl;
  //ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
  std::transform( hVectorA.begin()+70, hVectorA.begin()+135, hVectorB.begin()+85, hVectorO.begin()+150, std::plus< cl_long >( ) );
  bolt::cl::transform( ctl,
                       hVectorA.begin()+70,
                       hVectorA.begin()+135,
                       hVectorB.begin()+85,
                       hVectorDO.begin()+150, bolt::cl::plus<cl_long>( ) );

  cmpArrays(hVectorO, hVectorDO);


}



//  Temporarily disabling this test because we have a known issue running on the CPU device with our 
//  Bolt iterators
/*TEST( Transformint , KcacheTest )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > refInput( length );
    std::vector< int > refOutput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      refInput[i] = i;
      refInput[i] = i+1;
      refOutput[i] = 0;
    }

    //Call reduce with GPU device because the default is a GPU device
    bolt::cl::control ctrl = bolt::cl::control::getDefault();
    bolt::cl::device_vector< int >gpuInput( refInput.begin(), refInput.end() );
    bolt::cl::device_vector< int >gpuOutput( refOutput.begin(), refOutput.end() );

    //Call reference code
    std::transform( refInput.begin(), refInput.end(), refOutput.begin(), std::negate<int>());

    bolt::cl::transform( ctrl, gpuInput.begin(), gpuInput.end(), gpuOutput.begin(), bolt::cl::negate<int>() );
    cmpArrays( refOutput, gpuOutput );

    //Call reduce with CPU device
    ::cl::Context cpuContext(CL_DEVICE_TYPE_CPU);
    std::vector< cl::Device > devices = cpuContext.getInfo< CL_CONTEXT_DEVICES >();
    cl::Device selectedDevice;

    for(std::vector< cl::Device >::iterator iter = devices.begin();iter < devices.end(); iter++)
    {
        if(iter->getInfo<CL_DEVICE_TYPE> ( ) == CL_DEVICE_TYPE_CPU)
        {
            selectedDevice = *iter;
            break;
        }
    }
    ::cl::CommandQueue myQueue( cpuContext, selectedDevice );
    bolt::cl::control cpu_ctrl( myQueue );  // construct control structure from the queue.
    bolt::cl::device_vector< int > cpuInput( refInput.begin( ), refInput.end( ), CL_MEM_READ_ONLY, cpu_ctrl );
    bolt::cl::device_vector< int > cpuOutput( refOutput.begin( ), refOutput.end( ), CL_MEM_WRITE_ONLY, cpu_ctrl );

    bolt::cl::transform( cpu_ctrl, cpuInput.begin(), cpuInput.end(), cpuOutput.begin() , bolt::cl::negate<int>());

    cmpArrays( refOutput, cpuOutput );
}*/

TEST( DebuggingUShort, ushortbintransfrom)
{

#if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 32768; //2^15
#endif
  std::vector<unsigned short> hVectorA( length ),
                   hVectorB( length ),
                   hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 54443 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1000 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::plus<unsigned short> pl;
  bolt::cl::device_vector<unsigned short> dVectorA( hVectorA.begin(), hVectorA.end() ),
                               dVectorB( hVectorB.begin(), hVectorB.end() ),
                               dVectorO( hVectorO.begin(), hVectorO.end() );
  std::transform( hVectorA.begin(),
                  hVectorA.end(),
                  hVectorB.begin(),
                  hVectorO.begin(),
                  std::plus< unsigned short >( ) );
  bolt::cl::transform( dVectorA.begin(),
    dVectorA.end(),
    dVectorB.begin(),
    dVectorO.begin(), pl );

  cmpArrays(hVectorO, dVectorO);


}

TEST( IntegerTests64, cl_ulongintransfrom)
{

#if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 32768; //2^15
#endif
  std::vector<cl_ulong> hVectorA( length ),
                   hVectorB( length ),
                   hVectorO( length );
  std::fill( hVectorA.begin(), hVectorA.end(), 54443 );
  std::fill( hVectorB.begin(), hVectorB.end(), 1000 );
  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::plus<cl_ulong> pl;
  bolt::cl::device_vector<cl_ulong> dVectorA( hVectorA.begin(), hVectorA.end() ),
                               dVectorB( hVectorB.begin(), hVectorB.end() ),
                               dVectorO( hVectorO.begin(), hVectorO.end() );
  std::transform( hVectorA.begin(),
                  hVectorA.end(),
                  hVectorB.begin(),
                  hVectorO.begin(),
                  std::plus< cl_ulong >( ) );
  bolt::cl::transform( dVectorA.begin(),
    dVectorA.end(),
    dVectorB.begin(),
    dVectorO.begin(), pl );

  cmpArrays(hVectorO, dVectorO);


}



int main(int argc, char* argv[])
{
    //  Register our minidump generating logic
    //bolt::miniDumpSingleton::enableMiniDumps( );

    // Define MEMORYREPORT on windows platfroms to enable debug memory heap checking
#if defined( MEMORYREPORT ) && defined( _WIN32 )
    TCHAR logPath[ MAX_PATH ];
    ::GetCurrentDirectory( MAX_PATH, logPath );
    ::_tcscat_s( logPath, _T( "\\MemoryReport.txt") );

    // We leak the handle to this file, on purpose, so that the ::_CrtSetReportFile() can output it's memory 
    // statistics on app shutdown
    HANDLE hLogFile;
    hLogFile = ::CreateFile( logPath, GENERIC_WRITE, 
        FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );

    ::_CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
    ::_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
    ::_CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );

    ::_CrtSetReportFile( _CRT_ASSERT, hLogFile );
    ::_CrtSetReportFile( _CRT_ERROR, hLogFile );
    ::_CrtSetReportFile( _CRT_WARN, hLogFile );

    int tmp = ::_CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
    tmp |= _CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF;
    ::_CrtSetDbgFlag( tmp );

    // By looking at the memory leak report that is generated by this debug heap, there is a number with 
    // {} brackets that indicates the incremental allocation number of that block.  If you wish to set
    // a breakpoint on that allocation number, put it in the _CrtSetBreakAlloc() call below, and the heap
    // will issue a bp on the request, allowing you to look at the call stack
    // ::_CrtSetBreakAlloc( 1833 );

#endif /* MEMORYREPORT */

    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Set the standard OpenCL wait behavior to help debugging
    //bolt::cl::control& myControl = bolt::cl::control::getDefault( );
    // myControl.waitMode( bolt::cl::control::NiceWait );
    //myControl.forceRunMode( bolt::cl::control::MultiCoreCpu );  // choose tbb

    int retVal = RUN_ALL_TESTS( );

    //  Reflection code to inspect how many tests failed in gTest
    ::testing::UnitTest& unitTest = *::testing::UnitTest::GetInstance( );

    unsigned int failedTests = 0;
    for( int i = 0; i < unitTest.total_test_case_count( ); ++i )
    {
        const ::testing::TestCase& testCase = *unitTest.GetTestCase( i );
        for( int j = 0; j < testCase.total_test_count( ); ++j )
        {
            const ::testing::TestInfo& testInfo = *testCase.GetTestInfo( j );
            if( testInfo.result( )->Failed( ) )
                ++failedTests;
        }
    }

    //  Print helpful message at termination if we detect errors, to help users figure out what to do next
    if( failedTests )
    {
        bolt::tout << _T( "\nFailed tests detected in test pass; please run test again with:" ) << std::endl;
        bolt::tout << _T( "\t--gtest_filter=<XXX> to select a specific failing test of interest" ) << std::endl;
        bolt::tout << _T( "\t--gtest_catch_exceptions=0 to generate minidump of failing test, or" ) << std::endl;
        bolt::tout << _T( "\t--gtest_break_on_failure to debug interactively with debugger" ) << std::endl;
        bolt::tout << _T( "\t    (only on googletest assertion failures, not SEH exceptions)" ) << std::endl;
    }

    return retVal;
}
#else
// TransformTest.cpp : Defines the entry point for the console application.
//

#include "common/stdafx.h"
#include <bolt/cl/transform.h>
#include <bolt/cl/functional.h>

#include <iostream>
#include <algorithm>  // for testing against STL functions.

#include <boost/thread.hpp>

#include "utils.h"

extern void readFromFileTest();





// Test of a body operator which is constructed with a template argument
// Do this using the low-level macros that require manually creating the typename
// We can't use the ClCode trait because that requires a fully instantiated type, and 
// we want to pass the code for a templated myplus<T>.
std::string myplusStr = BOLT_CODE_STRING(
    template<typename T>
struct myplus  
{
    T operator()(const T &lhs, const T &rhs) const {return lhs + rhs;}
};
);
BOLT_CREATE_TYPENAME(myplus<float>);
BOLT_CREATE_TYPENAME(myplus<int>);
BOLT_CREATE_TYPENAME(myplus<double>);



void simpleTransform1(int aSize)
{
    std::string fName = __FUNCTION__ ;
    fName += ":";

    std::vector<float> A(aSize), B(aSize);
    for (int i=0; i<aSize; i++) {
        A[i] = float(i);
        B[i] = 10000.0f + (float)i;
    }


    {
        // Test1: Test case where a user creates a templatized functor "myplus<float>" and passes that
        // to customize the transformation:

        std::vector<float> Z0(aSize), Z1(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), myplus<float>());

        bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), myplus<float>(), myplusStr);
        checkResults(fName + "myplus<float>", Z0.begin(), Z0.end(), Z1.begin());
    }

    {
        //Test2:  Use a  templatized function from the provided bolt::cl functional header.  "bolt::cl::plus<float>"   
        std::vector<float> Z0(aSize), Z1(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), bolt::cl::plus<float>());

        bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), bolt::cl::plus<float>());  
        checkResults(fName + "bolt::cl::plus<float>", Z0.begin(), Z0.end(), Z1.begin());
    }


#if 0
    {
        // Test4 : try use of a simple binary function "op_sum" created by user.
        // This doesn't work - OpenCL generates an error that "op_sum isn't a type name.  Maybe need to create an
        // function signature rather than "op_sum".
        std::vector<float> Z0(aSize), Z1(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), op_sum);

        bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), op_sum, boltCode1, "op_sum");
        checkResults(fName + "Inline Binary Function", Z0.begin(), Z0.end(), Z1.begin());
    }
#endif


};




//---
// SimpleTransform2 Demonstrates:
// How to create a C++ functor object and use this to customize the transform library call.

BOLT_FUNCTOR(SaxpyFunctor,
struct SaxpyFunctor
{
    float _a;
    SaxpyFunctor(float a) : _a(a) {};

    float operator() (const float &xx, const float &yy) 
    {
        return _a * xx + yy;
    };
};
);  // end BOLT_FUNCTOR



void transformSaxpy(int aSize)
{
    std::string fName = __FUNCTION__ ;
    fName += ":";
    std::cout << fName << "(" << aSize << ")" << std::endl;

    std::vector<float> A(aSize), B(aSize), Z1(aSize), Z0(aSize);

    for (int i=0; i<aSize; i++) {
        A[i] = float(i);
        B[i] = 10000.0f + (float)i;
    }

    SaxpyFunctor sb(10.0);


    std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), sb);
    bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), sb);  

    checkResults(fName, Z0.begin(), Z0.end(), Z1.begin());

};

void transformSaxpyDeviceVector(int aSize)
{
    std::string fName = __FUNCTION__ ;
    fName += ":";
    std::cout << fName << "(" << aSize << ")" << std::endl;

    std::vector<float> A(aSize), B(aSize), Z0(aSize);
    bolt::cl::device_vector< float > dvA(aSize), dvB(aSize), dvZ(aSize);

    for (int i=0; i<aSize; i++) {
        A[i] = float(i);
        B[i] = 10000.0f + (float)i;
        dvA[i] = float(i);
        dvB[i] = 10000.0f + (float)i;
    }

    SaxpyFunctor sb(10.0);

    std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), sb);
    bolt::cl::transform(dvA.begin(), dvA.end(), dvB.begin(), dvZ.begin(), sb);

    checkResults(fName, Z0.begin(), Z0.end(), dvZ.begin());

};


void singleThreadReduction(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> *Zbolt, 
                           int aSize) 
{
    bolt::cl::transform(A.begin(), A.end(), B.begin(), Zbolt->begin(), bolt::cl::multiplies<float>());
};


void multiThreadReductions(int aSize, int iters)
{
    std::string fName = __FUNCTION__  ;
    fName += ":";

    std::vector<float> A(aSize), B(aSize);
    for (int i=0; i<aSize; i++) {
        A[i] = float(i);
        B[i] = 10000.0f + (float)i;
    }


    {
        std::vector<float> Z0(aSize), Z1(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), bolt::cl::minus<float>()); // golden answer:

        // show we only compile it once:
        for (int i=0; i<iters; i++) {
            bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), bolt::cl::minus<float>());
            checkResults(fName + "MultiIteration - bolt::cl::minus<float>", Z0.begin(), Z0.end(), Z1.begin());
        };
    }

    // Now try multiple threads:
    // FIXME - multi-thread doesn't work until we fix up the kernel to be per-thread.
    if (0) {
        static const int threadCount = 4;
        std::vector<float> Z0(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), bolt::cl::multiplies<float>()); // golden answer:

        std::vector<float> ZBolt [threadCount];
        for (int i=0; i< threadCount; i++){
            ZBolt[i].resize(aSize);
        };

        boost::thread t0(singleThreadReduction, A, B, &ZBolt[0], aSize);
        boost::thread t1(singleThreadReduction,  A, B, &ZBolt[1], aSize);
        boost::thread t2(singleThreadReduction,  A, B, &ZBolt[2], aSize);
        boost::thread t3(singleThreadReduction, A, B,  &ZBolt[3], aSize);


        t0.join();
        t1.join();
        t2.join();
        t3.join();

        for (int i=0; i< threadCount; i++){
            checkResults(fName + "MultiThread", Z0.begin(), Z0.end(), ZBolt[i].begin());
        };
    }
};


//void oclTransform(int aSize)
//{
// std::vector<float> A(aSize), B(aSize);
// for (int i=0; i<aSize; i++) {
//   A[i] = float(i);
//   B[i] = 1000.0f + (float)i;
// }
// std::vector<float> Z0(aSize);
// std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), bolt::cl::plus<float>()); // golden answer:              
//
// size_t bufSize = aSize * sizeof(float);
// cl::Buffer bufferA(CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, bufSize, A.data());
// //cl::Buffer bufferA(begin(A), end(A), true);
// cl::Buffer bufferB(CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, bufSize, B.data());
// cl::Buffer bufferZ(CL_MEM_WRITE_ONLY, sizeof(float) * aSize);
//
// bolt::cl::transform<float>(bufferA, bufferB, bufferZ, bolt::cl::plus<float>());
//
// float * zMapped = static_cast<float*> (cl::CommandQueue::getDefault().enqueueMapBuffer(bufferZ, true,              
//                                         CL_MAP_READ | CL_MAP_WRITE, 0/*offset*/, bufSize)); 
//
// std::string fName = __FUNCTION__ ;
// fName += ":";
//
// checkResults(fName + "oclBuffers", Z0.begin(), Z0.end(), zMapped);
//};



int _tmain(int argc, _TCHAR* argv[])
{
    simpleTransform1(256); 
    simpleTransform1(254); 
    simpleTransform1(23); 
    transformSaxpy(256);
    transformSaxpyDeviceVector(256);
    transformSaxpy(1024);
    transformSaxpyDeviceVector( 1024 );

    //multiThreadReductions(1024, 10);

    //oclTransform(1024);
    getchar();
    return 0;
}

#endif
