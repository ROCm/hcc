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
#define OCL_CONTEXT_BUG_WORKAROUND 1
#define TEST_LARGE_BUFFERS 0

#if (GOOGLE_TEST == 1)

#include "common/stdafx.h"
#include "common/myocl.h"

#include <bolt/cl/transform_reduce.h>
#include <bolt/cl/functional.h>
#include <bolt/miniDump.h>

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
//This is a compare routine for naked pointers.
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


template <typename T>
T generateRandom()
{
    double value = rand();
    static bool negate = true;
    if (negate)
    {
        negate = false;
        return -(T)fmod(value, 10.0);
    }
    else
    {
        negate = true;
        return (T)fmod(value, 10.0);
    }
}



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
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<ArrayType>);
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

TYPED_TEST_CASE_P( TransformArrayTest );

TYPED_TEST_P( TransformArrayTest, Normal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::square<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::square<ArrayType>(), init,
                                                       bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( TransformArrayTest, Serial )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::square<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce(ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::square<ArrayType>(), init,
                                                       bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( TransformArrayTest, MultiCoreCPU )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::square<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce(ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::square<ArrayType>(), init,
                                                       bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );
}


TYPED_TEST_P( TransformArrayTest, GPU_DeviceNormal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
#if OCL_CONTEXT_BUG_WORKAROUND
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control c_gpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));  
#else
    MyOclContext oclgpu = initOcl(CL_DEVICE_TYPE_GPU, 0);
    bolt::cl::control c_gpu(oclgpu._queue);  // construct control structure from the queue.
#endif

    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::square<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce( c_gpu,TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::square<ArrayType>(), init,
                                                       bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );

}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceNormal )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

#if OCL_CONTEXT_BUG_WORKAROUND
  ::cl::Context myContext = bolt::cl::control::getDefault( ).context( );
    bolt::cl::control c_cpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_CPU, 0 ));  
#else
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.
#endif

    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::square<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce( c_cpu,TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::square<ArrayType>(), init,
                                                       bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

TYPED_TEST_P( TransformArrayTest, MultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::negate<ArrayType>( ), init,
                                                       bolt::cl::plus<ArrayType>( ));

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}

TYPED_TEST_P( TransformArrayTest, SerialMultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    ArrayType init(0);
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce( ctl,TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::negate<ArrayType>( ), init,
                                                       bolt::cl::plus<ArrayType>( ));

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}

TYPED_TEST_P( TransformArrayTest, MulticoreMultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    ArrayType init(0);
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce( ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::negate<ArrayType>( ), init,
                                                       bolt::cl::plus<ArrayType>( ));

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}


TYPED_TEST_P( TransformArrayTest, GPU_DeviceMultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
#if OCL_CONTEXT_BUG_WORKAROUND
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control c_gpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));  
#else
    MyOclContext oclgpu = initOcl(CL_DEVICE_TYPE_GPU, 0);
    bolt::cl::control c_gpu(oclgpu._queue);  // construct control structure from the queue.
#endif


    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce( c_gpu,TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::negate<ArrayType>(), init,
                                                       bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceMultipliesFunction )
{
        typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
#if OCL_CONTEXT_BUG_WORKAROUND
  ::cl::Context myContext = bolt::cl::control::getDefault( ).context( );
    bolt::cl::control c_cpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_CPU, 0 ));  
#else
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.
#endif

    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::cl::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::transform_reduce( c_cpu,TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::negate<ArrayType>(), init,
                                                       bolt::cl::plus<ArrayType>());

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

#if (TEST_CPU_DEVICE == 1)
REGISTER_TYPED_TEST_CASE_P( TransformArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction,
                                           CPU_DeviceNormal, CPU_DeviceMultipliesFunction);
#else
REGISTER_TYPED_TEST_CASE_P( TransformArrayTest, Normal,Serial, MultiCoreCPU, GPU_DeviceNormal, 
                                                 MultipliesFunction, SerialMultipliesFunction,
                                  MulticoreMultipliesFunction, GPU_DeviceMultipliesFunction );
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
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
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
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< float > stdInput, boltInput, stdOutput, boltOutput;
};

#if(TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleVector: public ::testing::TestWithParam< int >
{
public:
    TransformDoubleVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                              stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<double>);
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
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
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
    TransformFloatDeviceVector( ): stdInput( GetParam( ) )
                                 
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        stdOutput = stdInput;

        //FIXME - The above should work but the below loop is used. 
        /*for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
        }*/
    }

protected:
    std::vector< float > stdInput, stdOutput;
    //bolt::cl::device_vector< float > boltInput, boltOutput;
};

#if(TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformDoubleDeviceVector( ): stdInput( GetParam( ) )
                                                            
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<double>);
        stdOutput = stdInput;

        //FIXME - The above should work but the below loop is used. 
        /*for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
        }*/
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

        std::generate(stdInput, stdInput + size, generateRandom<int>);
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

        std::generate(stdInput, stdInput + size, generateRandom<float>);
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


class transformReduceStdVectWithInit :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    transformReduceStdVectWithInit():mySize(GetParam()){
    }
};

TEST_P( transformReduceStdVectWithInit, withIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int init = 10;
     //there is no std::square available
    std::transform(stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::square<int>( ) ); 
    int stlTransformReduce = std::accumulate(stdOutput.begin( ), stdOutput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                     bolt::cl::square<int>(), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST_P( transformReduceStdVectWithInit, SerialwithIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    int init = 10;
     //there is no std::square available
    std::transform(stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::square<int>( ) ); 
    int stlTransformReduce = std::accumulate(stdOutput.begin( ), stdOutput.end( ), init,
                                                                bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::transform_reduce( ctl, boltInput.begin( ), boltInput.end( ), 
                                         bolt::cl::square<int>( ), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST_P( transformReduceStdVectWithInit, MultiCorewithIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    int init = 10;
      //there is no std::square available
    std::transform(stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), bolt::cl::square<int>( ) );
    int stlTransformReduce = std::accumulate(stdOutput.begin( ), stdOutput.end( ),
                                                    init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::transform_reduce( ctl, boltInput.begin( ), boltInput.end( ), 
                                        bolt::cl::square<int>( ), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST_P( transformReduceStdVectWithInit, withIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::square<int>());
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                        bolt::cl::square<int>(), init, bolt::cl::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

TEST_P( transformReduceStdVectWithInit, SerialwithIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::square<int>());
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::cl::transform_reduce( ctl, boltInput.begin( ), boltInput.end( ),
                                                bolt::cl::square<int>(), init, bolt::cl::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

TEST_P( transformReduceStdVectWithInit, MultiCorewithIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::square<int>());
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::cl::transform_reduce( ctl, boltInput.begin( ), boltInput.end( ),
                                            bolt::cl::square<int>(), init, bolt::cl::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

TEST_P( transformReduceStdVectWithInit, withIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::square<int>());
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltTransformReduce= bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                        bolt::cl::square<int>(), init, bolt::cl::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

TEST_P( transformReduceStdVectWithInit, SerialwithIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::square<int>());
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltTransformReduce= bolt::cl::transform_reduce(ctl,  boltInput.begin( ), boltInput.end( ), 
                                            bolt::cl::square<int>(), init, bolt::cl::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

TEST_P( transformReduceStdVectWithInit, MultiCorewithIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::square<int>());
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltTransformReduce= bolt::cl::transform_reduce(ctl,  boltInput.begin( ), boltInput.end( ),
                                            bolt::cl::square<int>(), init, bolt::cl::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
INSTANTIATE_TEST_CASE_P( withIntWithInitValue, transformReduceStdVectWithInit, ::testing::Range(1, 100, 1) );

class transformReduceTestMultFloat: public ::testing::TestWithParam<int>{
protected:
    int arraySize;
public:
    transformReduceTestMultFloat( ):arraySize( GetParam( ) )
    {}
};

TEST_P (transformReduceTestMultFloat, multiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
    }

#if defined (_WIN32 )
    std::transform( myArray, (float *)(myArray + arraySize), stdext::make_checked_array_iterator(myArray2, arraySize),
                                                                                            std::negate<float>( ) );
#else
    std::transform( myArray, (float *)(myArray + arraySize), myArray2, std::negate<float>( ) );
#endif

    float stlTransformReduce = std::accumulate(myArray2, myArray2 + arraySize, 1.0f, std::multiplies<float>());
    float boltTransformReduce = bolt::cl::transform_reduce(myBoltArray, myBoltArray + arraySize, 
                                bolt::cl::negate<float>(), 1.0f, bolt::cl::multiplies<float>());

    EXPECT_FLOAT_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}

TEST_P( transformReduceTestMultFloat, serialFloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    
    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    std::transform(A.begin(), A.end(), B.begin(), std::negate<float>());
    float stdTransformReduceValue = std::accumulate(B.begin(), B.end(), 0.0f, std::plus<float>());
    float boltClTransformReduce = bolt::cl::transform_reduce(boltVect.begin(), boltVect.end(),
                                    bolt::cl::negate<float>(), 0.0f, bolt::cl::plus<float>());
    //compare these results with each other
    EXPECT_FLOAT_EQ( stdTransformReduceValue, boltClTransformReduce );
}

INSTANTIATE_TEST_CASE_P(serialValues, transformReduceTestMultFloat, ::testing::Range(1, 100, 10));
INSTANTIATE_TEST_CASE_P(multiplyWithFloatPredicate, transformReduceTestMultFloat, ::testing::Range(1, 20, 5));
//end of new 2

#if(TEST_DOUBLE == 1)
class transformReduceTestMultDouble: public ::testing::TestWithParam<int>{
protected:
        int arraySize;
public:
    transformReduceTestMultDouble():arraySize(GetParam()){
    }
};

TEST_P (transformReduceTestMultDouble, multiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myBoltArray[i] = myArray[i];
    }

#if defined (_WIN32 )
    std::transform( myArray, myArray + arraySize, stdext::make_checked_array_iterator( myArray2, arraySize ), 
        std::negate<double>( ) );
#else
    std::transform(myArray, myArray + arraySize, myArray2, std::negate<double>());
#endif

    double stlTransformReduce = std::accumulate(myArray2, myArray2 + arraySize, 1.0, std::multiplies<double>());

    double boltTransformReduce = bolt::cl::transform_reduce(myBoltArray, myBoltArray + arraySize,
                                bolt::cl::negate<double>(), 1.0, bolt::cl::multiplies<double>());
    
    EXPECT_DOUBLE_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}


INSTANTIATE_TEST_CASE_P( multiplyWithDoublePredicate, transformReduceTestMultDouble, ::testing::Range(1, 20, 1) );


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

        std::generate(stdInput, stdInput + size, generateRandom<double>);

        for(unsigned int i=0; i < size; i++ )
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

    int init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( TransformIntegerVector, Serial )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    int init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::cl::transform_reduce( ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( TransformIntegerVector, MultiCore )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    int init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::cl::transform_reduce( ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( TransformFloatVector, Normal )
{
    float init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( TransformFloatVector, Serial )
{
    float init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::cl::transform_reduce( ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( TransformFloatVector, MultiCore )
{
    float init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::cl::transform_reduce( ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}


#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleVector, Inplace )
{
    double init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<double>());
    double stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    double boltReduce = bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<double>(), init,
                                                       bolt::cl::plus<double>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
#if (TEST_DEVICE_VECTOR == 1)
TEST_P( TransformIntegerDeviceVector, Inplace )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< int > boltOutput(stdOutput.begin(), stdOutput.end());
    
    int init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( TransformIntegerDeviceVector, SerialInplace )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< int > boltOutput(stdOutput.begin(), stdOutput.end());
    
    int init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::cl::transform_reduce(ctl,  boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( TransformIntegerDeviceVector, MultiCoreInplace )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< int > boltOutput(stdOutput.begin(), stdOutput.end());
    
    int init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::cl::transform_reduce(ctl,  boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}


TEST_P( TransformFloatDeviceVector, Inplace )
{
    bolt::cl::device_vector< float > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< float > boltOutput(stdOutput.begin(), stdOutput.end());
    
    float init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( TransformFloatDeviceVector, SerialInplace )
{
    bolt::cl::device_vector< float > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< float > boltOutput(stdOutput.begin(), stdOutput.end());
    
    float init(0);
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::cl::transform_reduce(ctl,  boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( TransformFloatDeviceVector, MultiCoreInplace )
{
    bolt::cl::device_vector< float > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< float > boltOutput(stdOutput.begin(), stdOutput.end());
    
    float init(0);
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::cl::transform_reduce(ctl,  boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleDeviceVector, Inplace )
{
    bolt::cl::device_vector< double > boltInput(stdInput.begin(), stdInput.end());
    bolt::cl::device_vector< double > boltOutput(stdOutput.begin(), stdOutput.end());
    
    double init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::cl::negate<double>());
    double stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    double boltReduce = bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::cl::negate<double>(), init,
                                                       bolt::cl::plus<double>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
#endif
#if defined (_WIN32)
TEST_P( TransformIntegerNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );

    int init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<int>());
    int stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    int boltReduce = bolt::cl::transform_reduce( wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( TransformIntegerNakedPointer, SerialInplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    int init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<int>());
    int stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    int boltReduce = bolt::cl::transform_reduce(ctl,  wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( TransformIntegerNakedPointer, MultiCoreInplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    int init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<int>());
    int stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    int boltReduce = bolt::cl::transform_reduce(ctl,  wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::cl::negate<int>(), init,
                                                       bolt::cl::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}


TEST_P( TransformFloatNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );

    float init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<float>());
    float stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    float boltReduce = bolt::cl::transform_reduce( wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( TransformFloatNakedPointer, SerialInplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );

    float init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<float>());
    float stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    float boltReduce = bolt::cl::transform_reduce( ctl, wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( TransformFloatNakedPointer, MultiCoreInplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );

    float init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<float>());
    float stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    float boltReduce = bolt::cl::transform_reduce( ctl, wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::cl::negate<float>(), init,
                                                       bolt::cl::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}



#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< double* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );

    double init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<double>());
    double stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    double boltReduce = bolt::cl::transform_reduce( wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::cl::negate<double>(), init,
                                                       bolt::cl::plus<double>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif
#endif
std::array<int, 10> TestValues = {2,4,8,16,32,64,128,256,512,1024};
std::array<int, 5> TestValues2 = {2048,4096,8192,16384,32768};
//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16	
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( TransformValues, TransformDoubleVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformDoubleVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                    TestValues2.end() ) );
//#endif
#endif
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                            TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                        TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformDoubleDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                        TestValues.end() ) );
//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( TransformValues2, TransformDoubleDeviceVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                        TestValues2.end() ) );
//#endif
#endif
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerNakedPointer, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                                            TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                                        TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( Transform, TransformDoubleNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( Transform2, TransformDoubleNakedPointer, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                    TestValues2.end() ) );
//#endif
#endif

//#if TEST_LARGE_BUFFERS
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
struct UDD { 
    int a; 
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) { 
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

    UDD operator + (const UDD &rhs) const {
                UDD tmp = *this;
                tmp.a = tmp.a + rhs.a;
                tmp.b = tmp.b + rhs.b;
                return tmp;
    }
    
    UDD() 
        : a(0),b(0) { } 
    UDD(int _in) 
        : a(_in), b(_in +1)  { } 
        
}; 
);


BOLT_FUNCTOR(tbbUDD,
struct tbbUDD { 
    float a; 
    double b;

    tbbUDD operator + (const tbbUDD &rhs) const {
                tbbUDD tmp = *this;
                tmp.a = tmp.a + rhs.a;
                tmp.b = tmp.b + rhs.b;
                return tmp;
    }
    
     bool operator() (const tbbUDD& lhs, const tbbUDD& rhs) { 
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    } 
    tbbUDD() 
        : a(0.0f),b(0.0) { } 
    tbbUDD(int _in) 
        : a(float(_in)), b( double(_in +1))  { } 
    bool operator == (const tbbUDD& other) const { 
        return ((double)(a+b) == (double)(other.a+other.b));
    }
}; 
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< UDD >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< UDD >::iterator, bolt::cl::deviceVectorIteratorTemplate );
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< tbbUDD >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< tbbUDD >::iterator, bolt::cl::deviceVectorIteratorTemplate );
BOLT_CREATE_TYPENAME(bolt::cl::plus<tbbUDD>);
BOLT_FUNCTOR(DivUDD,
struct DivUDD
{
    float operator()(const UDD &rhs) const
    {
        float _result = (1.f*rhs.a) / (1.f*rhs.a+rhs.b); //  a/(a+b)
        return _result;
    };
}; 
);

BOLT_FUNCTOR(negateUDD,
struct negateUDD
{
    UDD operator()(const UDD &rhs) const
    {
       UDD temp;
       temp.a = -rhs.a;
       temp.b = -rhs.b;
       return temp;
    };
}; 
);
BOLT_FUNCTOR(negatetbbUDD,
struct negatetbbUDD
{
    tbbUDD operator()(const tbbUDD &rhs) const
    {
       tbbUDD temp;
       temp.a = -rhs.a;
       temp.b = -rhs.b;
       return temp;
    };
}; 
);

/**********************************************************
 * mixed unary operator - dtanner
 *********************************************************/
/*
TEST( MixedTransform, OutOfPlace )
{
    //size_t length = GetParam( );

    //setup containers
    int length = (1<<16)+23;
//    bolt::cl::negate< uddtI2 > nI2;
    UDD initial(2);
    //UDD identity();
    bolt::cl::device_vector< UDD >    input( length, initial, CL_MEM_READ_WRITE, true  );
    bolt::cl::device_vector< float > output( length,     0.f, CL_MEM_READ_WRITE, false );
    std::vector< UDD > refInput( length, initial );
    std::vector< float > refIntermediate( length, 0.f );
    std::vector< float > refOutput(       length, 0.f );

    //    T stlReduce = std::accumulate(Z.begin(), Z.end(), init);

    //T boltReduce = bolt::cl::transform_reduce(A.begin(), A.end(), SquareMe<T>(), init, 
    //                                          bolt::cl::plus<T>(), squareMeCode);

    // call transform_reduce
    DivUDD ddd;
    bolt::cl::plus<float> add;
    float boldReduce = bolt::cl::transform_reduce( input.begin(), input.end(),  ddd, 0.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0.f); // out-of-place scan

    // compare results
    cmpArrays(refOutput, output);
}
*/
BOLT_FUNCTOR( point,
     class point
     {
     public:
           int xPoint;
           int yPoint;

           point(){
           }
           point(int x, int y){
                xPoint = x;
                yPoint = y;
           }
           point operator + (const point &rhs) const {
                point tmp = *this;
                tmp.xPoint = tmp.xPoint + rhs.xPoint;
                tmp.yPoint = tmp.yPoint + rhs.yPoint;
                return tmp;
          }

           point operator - () const {
                point tmp = *this;
                tmp.xPoint = -1 * (tmp.xPoint);
                tmp.yPoint = -1 * (tmp.yPoint);
                return tmp;
           }
    };
);

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, point );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::negate, int, point );
BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, int, point );

TEST( outputTypeMismatch, pointVsInt )
{
     point pt1(12, 3);
     point pt2(2, 5);
     point pt(0,0);

     std::vector< point > my_input_vector( 2 );

     //I am using this as we are not use bolt::cl::device_vector to contain “point” type, 
     // I’m not sure this  behavior is expected!
     my_input_vector[0].xPoint = 12;
     my_input_vector[0].yPoint = 3;

     my_input_vector[1].xPoint = 2;
     my_input_vector[1].yPoint = 5;

     point newPt = bolt::cl::transform_reduce(my_input_vector.begin(),my_input_vector.end(),bolt::cl::negate<point>(),
                                                                pt, bolt::cl::plus<point>( ) );
}

BOLT_CREATE_TYPENAME(bolt::cl::less<UDD>);
BOLT_CREATE_TYPENAME(bolt::cl::greater<UDD>);
BOLT_CREATE_TYPENAME(bolt::cl::plus<UDD>);

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


INSTANTIATE_TYPED_TEST_CASE_P( Integer, TransformArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, TransformArrayTest, FloatTests );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TYPED_TEST_CASE_P( Double, TransformArrayTest, DoubleTests );
#endif 
//INSTANTIATE_TYPED_TEST_CASE_P( UDDTest, SortArrayTest, UDDTests );


//BUG 377596 reproducable program
BOLT_FUNCTOR(PointT,
struct PointT {
float x;
float y;

PointT(float x, float y):x(x),y(y) {};
PointT() {};
PointT operator+ (const PointT& point1)
{
return PointT( (x + point1.x), (y + point1.y) );
}
};
);

BOLT_TEMPLATE_REGISTER_NEW_ITERATOR(bolt::cl::device_vector, int, PointT);

BOLT_FUNCTOR(isInsideCircleFunctor,
struct isInsideCircleFunctor {
isInsideCircleFunctor(float _radius):radius(_radius) { };
int operator() (const PointT& point) {
float tx = point.x;
float ty = point.y;
float t = sqrt( (tx*tx) + (ty*ty) );
return (t <= radius)? 1: 0;
};

private:
float radius;
};
);

//BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, PointT );


TEST( TestBug377596, TestBug377596 )
{
   
bolt::cl::device_vector<PointT> boltInputPoints;

#define RADIUS 1.0f

int pointsInCircle = bolt::cl::transform_reduce(boltInputPoints.begin(), boltInputPoints.end(), isInsideCircleFunctor(RADIUS), 0, bolt::cl::plus<int>());

}






TEST(TransformReduce, Float)
{
#ifdef LARGE_SIZE
     int length = 1<<24;
#else
     int length = 1<<16;
#endif
     std::vector< float > input( length );
     std::vector< float > refInput( length);
     std::vector< float > refIntermediate( length );
     for(int i=0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
 
    // call transform_reduce
    //  DivUDD ddd;
    bolt::cl::negate<float> ddd;
    bolt::cl::plus<float> add;
    float boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 4.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.f, add); //out-of-place scan

  //  printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
} 

TEST(TransformReduce, SerialFloat)
{
#ifdef LARGE_SIZE
     int length = 1<<24;
#else
     int length = 1<<16;
#endif
     std::vector< float > input( length );
     std::vector< float > refInput( length);
     std::vector< float > refIntermediate( length );
     for(int i=0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call transform_reduce
    //  DivUDD ddd;
    bolt::cl::negate<float> ddd;
    bolt::cl::plus<float> add;
    float boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 4.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.f, add); //out-of-place scan

  //  printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
} 

TEST(TransformReduce, MultiCoreFloat)
{
#ifdef LARGE_SIZE
     int length = 1<<24;
#else
     int length = 1<<16;
#endif
     std::vector< float > input( length );
     std::vector< float > refInput( length);
     std::vector< float > refIntermediate( length );
     for(int i=0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call transform_reduce
    //  DivUDD ddd;
    bolt::cl::negate<float> ddd;
    bolt::cl::plus<float> add;
    float boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 4.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.f, add); //out-of-place scan

  //  printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
} 


#if (TEST_DOUBLE == 1)

TEST(TransformReduce, MultiCoreDouble)
{
     int length = 1<<24;
     std::vector< double > input( length );
     std::vector< double > refInput( length);
     std::vector< double > refIntermediate( length );
     for(int i=0; i<length; i++) {
        input[i] = 2.0;
        refInput[i] = 2.0;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call transform_reduce
    //  DivUDD ddd;
    bolt::cl::negate<double> ddd;
    bolt::cl::plus<double> add;
    double boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 4.0, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    double stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.0, add);//out-of-place scan

    //printf("%d %lf %lf\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_DOUBLE_EQ( stdReduce, boldReduce );
  
} 
#endif

/* TEST(TransformReduce, DefaultUDD)
{
    int length = 1<<24;
    UDD initial;
    initial.a = 2;
    initial.b = 2;
    std::vector< UDD > input( length, initial );
    std::vector< UDD > refInput( length, initial );
    std::vector< UDD > refIntermediate( length);
     for(int i=0; i<length; i++) {
        input[i].a = 2;
        refInput[i].a = 2;
        input[i].b = 2;
        refInput[i].b = 2;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
   
    negateUDD ddd;
    bolt::cl::plus<UDD> add;
    UDD boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    UDD stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(),initial,add);//out-of-place scan
    //printf("%d %d %d %d %d\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 

TEST(TransformReduce, SerialUDD)
{
    int length = 1<<24;
    UDD initial;
    initial.a = 2;
    initial.b = 2;
    std::vector< UDD > input( length, initial );
    std::vector< UDD > refInput( length, initial );
    std::vector< UDD > refIntermediate( length);
     for(int i=0; i<length; i++) {
        input[i].a = 2;
        refInput[i].a = 2;
        input[i].b = 2;
        refInput[i].b = 2;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    negateUDD ddd;
    bolt::cl::plus<UDD> add;
    UDD boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    UDD stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(),initial,add);//out-of-place scan
    //printf("%d %d %d %d %d\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 

TEST(TransformReduce, MultiCoreUDD)
{
    int length = 1<<24;
    UDD initial;
    initial.a = 2;
    initial.b = 2;
    std::vector< UDD > input( length, initial );
    std::vector< UDD > refInput( length, initial );
    std::vector< UDD > refIntermediate( length);
     for(int i=0; i<length; i++) {
        input[i].a = 2;
        refInput[i].a = 2;
        input[i].b = 2;
        refInput[i].b = 2;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    negateUDD ddd;
    bolt::cl::plus<UDD> add;
    UDD boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    UDD stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(),initial,add);//out-of-place scan
    //printf("%d %d %d %d %d\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
*/

#if (TEST_DOUBLE == 1)
TEST(TransformReduce, MultiCoreDoubleUDD)
{
#ifdef LARGE_SIZE
     int length = 1<<24;
#else
     int length = 1<<16;
#endif
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    std::vector< tbbUDD > input( length, initial );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
     for(int i=0; i<length; i++) {
        input[i].a = 1.f;
        refInput[i].a = 1.f;
        input[i].b = 5.0;
        refInput[i].b = 5.0;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    negatetbbUDD ddd;
    bolt::cl::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    //printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
#endif

TEST(TransformReduce, DeviceVectorInt)
{
     int length = 1<<16;
     std::vector<  int > refInput( length);
     std::vector< int > refIntermediate( length );
     
     for(int i=0; i<length; i++) {
        refInput[i] = i;
     //   printf("%d \n", input[i]);
     }
     
     bolt::cl::device_vector< int > input(refInput.begin(), refInput.end());
     
     bolt::cl::control ctl = bolt::cl::control::getDefault( );

     // call transform_reduce
     //  DivUDD ddd;
     bolt::cl::negate<int> ddd;
     bolt::cl::plus<int> add;

     int boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0, add );
     ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
     int stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0); // out-of-place scan
     printf("%d %d %d\n", length, boldReduce, stdReduce);  
     // compare results
     EXPECT_EQ( stdReduce, boldReduce );
  
  
} 

TEST(TransformReduce, SerialDeviceVectorInt)
{
     int length = 1<<16;
     std::vector<  int > refInput( length);
     std::vector< int > refIntermediate( length );
     for(int i=0; i<length; i++) {
        refInput[i] = i;
     //   printf("%d \n", input[i]);
     }
     bolt::cl::device_vector< int > input(refInput.begin(), refInput.end());
     
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);
     // call transform_reduce
     //  DivUDD ddd;
     bolt::cl::negate<int> ddd;
     bolt::cl::plus<int> add;

     int boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0, add );
     ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
     int stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0); // out-of-place scan
     printf("%d %d %d\n", length, boldReduce, stdReduce);  
     // compare results
     EXPECT_EQ( stdReduce, boldReduce );
  
  
} 

TEST(TransformReduce, MultiCoreDeviceVectorInt)
{
     int length = 1<<16;
     std::vector<  int > refInput( length);
     std::vector< int > refIntermediate( length );
     for(int i=0; i<length; i++) {
        refInput[i] = i;
     //   printf("%d \n", input[i]);
     }
     bolt::cl::device_vector< int > input(refInput.begin(), refInput.end());
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
     // call transform_reduce
     //  DivUDD ddd;
     bolt::cl::negate<int> ddd;
     bolt::cl::plus<int> add;

     int boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0, add );
     ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
     int stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0); // out-of-place scan
     printf("%d %d %d\n", length, boldReduce, stdReduce);  
     // compare results
     EXPECT_EQ( stdReduce, boldReduce );
  
  
} 


TEST(TransformReduce, DeviceVectorFloat)
{
   
     int length = 1<<16;
     
     std::vector<  float > refInput( length);
     std::vector< float > refIntermediate( length );

    for(int i=0; i<length; i++) {
        refInput[i] = 2.f;
     //   printf("%d \n", input[i]);
    }
    bolt::cl::device_vector< float> input(refInput.begin(), refInput.end());
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );

    // call transform_reduce
    //  DivUDD ddd;
    bolt::cl::negate<float> ddd;
    bolt::cl::plus<float> add;

    float boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0.f); // out-of-place scan

    printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
  
} 

TEST(TransformReduce, SerialDeviceVectorFloat)
{
   
     int length = 1<<16;
     
     std::vector<  float > refInput( length);
     std::vector< float > refIntermediate( length );

    for(int i=0; i<length; i++) {
        refInput[i] = 2.f;
     //   printf("%d \n", input[i]);
    }
    bolt::cl::device_vector< float> input(refInput.begin(), refInput.end());
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call transform_reduce
    //  DivUDD ddd;
    bolt::cl::negate<float> ddd;
    bolt::cl::plus<float> add;

    float boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0.f); // out-of-place scan

    printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
  
} 

TEST(TransformReduce, MultiCoreDeviceVectorFloat)
{
   
     int length = 1<<16;
     
     std::vector<  float > refInput( length);
     std::vector< float > refIntermediate( length );

    for(int i=0; i<length; i++) {
        refInput[i] = 2.f;
     //   printf("%d \n", input[i]);
    }
    bolt::cl::device_vector< float> input(refInput.begin(), refInput.end());
        
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call transform_reduce
    //  DivUDD ddd;
    bolt::cl::negate<float> ddd;
    bolt::cl::plus<float> add;

    float boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0.f); // out-of-place scan

    printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
  
} 

/* TEST(TransformReduce, DeviceVectorUDD)
{
    int length = 1<<20;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    bolt::cl::device_vector< tbbUDD > input(  length, initial,  true );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
    /*
     for(int i=0; i<length; i++) {
        input[i].a = 1.f;
        refInput[i].a = 1.f;
        input[i].b = 5.0;
        refInput[i].b = 5.0;
    }
    */
   /* bolt::cl::control ctl = bolt::cl::control::getDefault( );

    negatetbbUDD ddd;
    bolt::cl::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 


TEST(TransformReduce, SerialDeviceVectorUDD)
{
    int length = 1<<20;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    bolt::cl::device_vector< tbbUDD > input(  length, initial,  true );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    negatetbbUDD ddd;
    bolt::cl::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 


TEST(TransformReduce, MultiCoreDeviceVectorUDD)
{
    int length = 1<<20;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    bolt::cl::device_vector< tbbUDD > input(  length, initial,  true );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    negatetbbUDD ddd;
    bolt::cl::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::cl::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
*/

//  Temporarily disabling this test because we have a known issue running on the CPU device with our 
//  Bolt iterators
/*
TEST( TransformReduceInt , KcacheTest )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > refInput( length );
    std::vector< int > temp( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      refInput[i] = i;
      refInput[i] = i+1;
      temp[i] = 0;
    }

    //Call reduce with GPU device because the default is a GPU device
    bolt::cl::control ctrl = bolt::cl::control::getDefault();
    bolt::cl::device_vector< int >gpuInput( refInput.begin(), refInput.end() );

    int initzero = 0;
    int resultGPU_OCL;
    int resultCPU_OCL;
    int resultCPU_STL;
    resultGPU_OCL = bolt::cl::transform_reduce( ctrl, gpuInput.begin(), gpuInput.end(), bolt::cl::negate<int>(),
                                                                                initzero, bolt::cl::plus<int>() );

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

    resultCPU_OCL = bolt::cl::transform_reduce( cpu_ctrl, cpuInput.begin(), cpuInput.end(), bolt::cl::negate<int>(),
                                                                                  initzero, bolt::cl::plus<int>());

    //Call reference code
    std::transform( refInput.begin(), refInput.end(), temp.begin(), std::negate<int>() );
    resultCPU_STL = std::accumulate( temp.begin(), temp.end(), initzero );

    EXPECT_EQ( resultCPU_STL, resultGPU_OCL );
    EXPECT_EQ( resultCPU_STL, resultCPU_OCL );
}
*/
TEST (cl_outputType_transform_reduce_sq_max, epr__all_raised){

  int data[6] = {-1, 0, -2, -2, 1, -3};
  bolt::cl::control my_ctl;
  my_ctl.setForceRunMode(bolt::cl::control::SerialCpu);
  //int result = bolt::cl::transform_reduce(data, data + 6, bolt::cl::modulus<int>(), 0, bolt::cl::maximum<int>());
  int result = bolt::cl::transform_reduce(my_ctl, data, data + 6, bolt::cl::square<int>(),0,bolt::cl::maximum<int>());
    
  EXPECT_EQ(9, result);
}

TEST( TransformReduceStdVectWithInit, OffsetTestDeviceVectorSerialCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::cl::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );


    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::cl::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::cl::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::cl::transform_reduce(  ctl, dVectorA.begin( ) + offset, dVectorA.end( ),
                                                          bolt::cl::negate<int>(), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( TransformReduceStdVectWithInit, OffsetTestDeviceVectorMultiCoreCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::cl::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );


    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::cl::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::cl::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::cl::transform_reduce(  ctl, dVectorA.begin( ) + offset, dVectorA.end( ),
                                                          bolt::cl::negate<int>(), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( TransformReduceStdVectWithInit, OffsetTestMultiCoreCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );



    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::cl::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::cl::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::cl::transform_reduce(  ctl, stdInput.begin( ) + offset, stdInput.end( ),
                                                          bolt::cl::negate<int>(), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( TransformReduceStdVectWithInit, OffsetTestSerialCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );



    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::cl::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::cl::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::cl::transform_reduce(  ctl, stdInput.begin( ) + offset, stdInput.end( ),
                                                          bolt::cl::negate<int>(), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

int main(int argc, char* argv[])
{
 
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Register our minidump generating logic
    //bolt::miniDumpSingleton::enableMiniDumps( );

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
#include <bolt/cl/transform_reduce.h>
#include <bolt/cl/functional.h>

#include <iostream>
#include <algorithm>  // for testing against STL functions.
#include <numeric>


template<typename T>
void printCheckMessage(bool err, std::string msg, T  stlResult, T boltResult)
{
    if (err) {
        std::cout << "*ERROR ";
    } else {
        std::cout << "PASSED ";
    }

    std::cout << msg << "  STL=" << stlResult << " BOLT=" << boltResult << std::endl;
};

template<typename T>
bool checkResult(std::string msg, T  stlResult, T boltResult)
{
    bool err =  (stlResult != boltResult);
    printCheckMessage(err, msg, stlResult, boltResult);

    return err;
};

#if (TEST_DOUBLE == 1)
// For comparing floating point values:
template<typename T>
bool checkResult(std::string msg, T  stlResult, T boltResult, double errorThresh)
{
    bool err;
    if ((errorThresh != 0.0) && stlResult) {
        double ratio = (double)(boltResult) / (double)(stlResult) - 1.0;
        err = abs(ratio) > errorThresh;
    } else {
        // Avoid div-by-zero, check for exact match.
        err = (stlResult != boltResult);
    }

    printCheckMessage(err, msg, stlResult, boltResult);
    return err;
};

#endif

// Simple test case for bolt::cl::transform_reduce:
// Perform a sum-of-squares
// Demonstrates:
//  * Use of transform_reduce function - takes two separate functors, one for transform and one for reduce.
//  * Performs a useful operation - squares each element, then adds them together
//  * Use of transform_reduce bolt is more efficient than two separate function calls due to fusion - 
//        After transform is applied, the result is immediately reduced without being written to memory.
//        Note the STL version uses two function calls - transform followed by accumulate.
//  * Operates directly on host buffers.
std::string squareMeCode = BOLT_CODE_STRING(
    template<typename T>
struct SquareMe {
    T operator() (const T &x ) const { return x*x; } ;
};
);
BOLT_CREATE_TYPENAME(SquareMe<int>);
BOLT_CREATE_TYPENAME(SquareMe<float>);
#if (TEST_DOUBLE == 1)
BOLT_CREATE_TYPENAME(SquareMe<double>);
#endif


template<typename T>
void sumOfSquares(int aSize)
{
    std::vector<T> A(aSize), Z(aSize);
    T init(0);
    for (int i=0; i < aSize; i++) {
        A[i] = (T)(i+1);
    };

    // For STL, perform the operation in two steps - transform then reduction:
    std::transform(A.begin(), A.end(), Z.begin(), SquareMe<T>());
    T stlReduce = std::accumulate(Z.begin(), Z.end(), init);

    T boltReduce = bolt::cl::transform_reduce(A.begin(), A.end(), SquareMe<T>(), init, 
                                              bolt::cl::plus<T>(), squareMeCode);
    checkResult(__FUNCTION__, stlReduce, boltReduce);
};

template<typename T>
void sumOfSquaresDeviceVector( int aSize )
{
    std::vector<T> A( aSize ), Z( aSize );
    bolt::cl::device_vector< T > dV( aSize );
    
    T init( 0 );
    for( int i=0; i < aSize; i++ )
    {
        A[i] = (T)(i+1);
        dV[i] = (T)(i+1);
    };

    // For STL, perform the operation in two steps - transform then reduction:
    std::transform(A.begin(), A.end(), Z.begin(), SquareMe<T>());
    T stlReduce = std::accumulate(Z.begin(), Z.end(), init);

    T boltReduce = bolt::cl::transform_reduce( dV.begin( ), dV.end( ), SquareMe< T >( ), init, bolt::cl::plus< T >( ), 
                                                                                                       squareMeCode ); 
    checkResult( __FUNCTION__, stlReduce, boltReduce );
};



int _tmain(int argc, _TCHAR* argv[])
{
    sumOfSquares<int>(256);
    sumOfSquaresDeviceVector<int>( 256 );
    sumOfSquares<float>(256);
    #if (TEST_DOUBLE == 1)
    sumOfSquares<double>(256);
    sumOfSquares<double>(4);
    sumOfSquares<double>(2048);
    #endif

    sumOfSquares<int>(4);
    sumOfSquares<float>(4);
    sumOfSquares<int>(2048);
    sumOfSquares<float>(2048);

    getchar();
    return 0;
}

#endif
