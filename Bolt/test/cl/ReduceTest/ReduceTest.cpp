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

// Defines the entry point for the console application.

#define OCL_CONTEXT_BUG_WORKAROUND 1
#define TEST_DOUBLE 1
#define TEST_CPU_DEVICE 0
#define BOLT_DEBUG_LOG

#include "bolt/BoltLog.h"
#include "stdafx.h"
#include <bolt/cl/iterator/counting_iterator.h>
#include <bolt/cl/reduce.h>
#include <bolt/cl/functional.h>
#include <bolt/cl/control.h>

#include <iostream>
#include <algorithm>  // for testing against STL functions.
#include <numeric>
#include <gtest/gtest.h>
#include <type_traits>

#include "common/test_common.h"
#include "common/myocl.h"
#include "bolt/miniDump.h"


void testDeviceVector()
{
    const int aSize = 1000;
    std::vector<int> hA(aSize);

    for(int i=0; i<aSize; i++) {
        hA[i] = i;
    };
    
    bolt::cl::device_vector<int> dA(hA.begin(), hA.end());
    int hSum = std::accumulate(hA.begin(), hA.end(), 0);
    int sum = bolt::cl::reduce(dA.begin(), dA.end(), 0);
    
};

BOLT_FUNCTOR(DUDD,
struct DUDD { 

     int a; 
    int b;
    bool operator() (const DUDD& lhs, const DUDD& rhs) { 
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    } 
    bool operator < (const DUDD& other) const { 
        return ((a+b) < (other.a+other.b));
    }
    bool operator > (const DUDD& other) const { 
        return ((a+b) > (other.a+other.b));
    }
    bool operator == (const DUDD& other) const { 
        return ((a+b) == (other.a+other.b));
    }

    DUDD operator + (const DUDD &rhs) const {
                DUDD tmp = *this;
                tmp.a = tmp.a + rhs.a;
                tmp.b = tmp.b + rhs.b;
                return tmp;
    }
    
    DUDD() 
        : a(0),b(0) { } 
    DUDD(int _in) 
        : a(_in), b(_in +1)  { } 
}; 
);
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< DUDD >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< DUDD >::iterator, bolt::cl::deviceVectorIteratorTemplate );
//BOLT_CREATE_TYPENAME(bolt::cl::plus<DUDD>);

BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::plus, int, DUDD );


void testTBB()
{
#ifdef LARGE_SIZE
     const int aSize = 1<<24;
#else
     const int aSize = 1<<16;
#endif
    std::vector<int> stdInput(aSize);
    std::vector<int> tbbInput(aSize);


    for(int i=0; i<aSize; i++) {
        stdInput[i] = 2;
        tbbInput[i] = 2;
    };

    int hSum = std::accumulate(stdInput.begin(), stdInput.end(), 2);
    bolt::cl::control ctl = bolt::cl::control::getDefault();

    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    int sum = bolt::cl::reduce(ctl, tbbInput.begin(), tbbInput.end(), 2);

    if(hSum == sum)
        printf ("\nTBB Test case PASSED %d %d\n", hSum, sum);
    else
        printf ("\nTBB Test case FAILED\n");

};
void testdoubleTBB()
{
#ifdef LARGE_SIZE
     const int aSize = 1<<24;
#else
     const int aSize = 1<<16;
#endif
    std::vector<double> stdInput(aSize);
    std::vector<double> tbbInput(aSize);


    for(int i=0; i<aSize; i++) {
        stdInput[i] = 3.0;
        tbbInput[i] = 3.0;
    };

    double hSum = std::accumulate(stdInput.begin(), stdInput.end(), 1.0);
    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    double sum = bolt::cl::reduce(ctl, tbbInput.begin(), tbbInput.end(), 1.0);
    if(hSum == sum)
        printf ("\nTBB Test case PASSED %lf %lf\n", hSum, sum);
    else
        printf ("\nTBB Test case FAILED\n");
}

void testDUDDTBB()
{

    const int aSize = 1<<19;
    std::vector<DUDD> stdInput(aSize);
    std::vector<DUDD> tbbInput(aSize);

    DUDD initial;
    initial.a = 1;
    initial.b = 2;

    for(int i=0; i<aSize; i++) {
        stdInput[i].a = 2;
        stdInput[i].b = 3;
        tbbInput[i].a = 2;
        tbbInput[i].b = 3;

    };

    BOLTLOG::CaptureLog *xyz =  BOLTLOG::CaptureLog::getInstance();
    xyz->Initialize();
    std::vector< BOLTLOG::FunPaths> paths;

    bolt::cl::plus<DUDD> add;
    DUDD hSum = std::accumulate(stdInput.begin(), stdInput.end(), initial,add);
    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    DUDD sum = bolt::cl::reduce(ctl, tbbInput.begin(), tbbInput.end(), initial, add);
    DUDD sum1 = bolt::cl::reduce(tbbInput.begin(), tbbInput.end(), initial, add);

    if(hSum == sum)
        printf ("\nDUDDTBB Test case PASSED %d %d %d %d\n", hSum.a, sum.a, hSum.b, sum.b);
    else
        printf ("\nDUDDTBB Test case FAILED\n");
        

    xyz->WhatPathTaken(paths);
    for(std::vector< BOLTLOG::FunPaths>::iterator parse=paths.begin(); parse!=paths.end(); parse++)
    {
        std::cout<<(*parse).fun;
        std::cout<<(*parse).path;
        std::cout<<(*parse).msg;
    }

}

void testTBBDevicevector()
{
    const int aSize = 1024;
    std::vector<int> stdInput(aSize);

    for(int i=0; i<aSize; i++) {
        stdInput[i] = i;
    };
    
    bolt::cl::device_vector<int> tbbInput(stdInput.begin(), stdInput.end());
    int hSum = std::accumulate(stdInput.begin(), stdInput.end(), 0);
    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    int sum = bolt::cl::reduce(ctl, tbbInput.begin(), tbbInput.end(), 0);
    if(hSum == sum)
        printf ("\nTBBDevicevector Test case PASSED %d %d\n", hSum, sum);
    else
        printf ("\nTBBDevicevector Test case FAILED*********\n");


};
#if defined(_WIN32)
// Super-easy windows profiling interface.
// Move to timing infrastructure when that becomes available.
__int64 StartProfile() {
    __int64 begin;
    QueryPerformanceCounter((LARGE_INTEGER*)(&begin));
    return begin;
};

void EndProfile(__int64 start, int numTests, std::string msg) {
    __int64 end, freq;
    QueryPerformanceCounter((LARGE_INTEGER*)(&end));
    QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
    double duration = (end - start)/(double)(freq);
    printf("%s %6.2fs, numTests=%d %6.2fms/test\n", msg.c_str(), duration, numTests, duration*1000.0/numTests);
};
#endif

//////////////////////////////////////////////////////////////////////////////////////
// GTEST CASES
//////////////////////////////////////////////////////////////////////////////////////
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
class ReduceArrayTest: public ::testing::Test
{
public:
    ReduceArrayTest( ): m_Errors( 0 )
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

    virtual ~ReduceArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize =  std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType, ArraySize > stdInput, boltInput, stdOutput, boltOutput;
    int m_Errors;
};

TEST( ReduceStdVectWithInit, OffsetTest)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> stdOutput( length );
    std::vector<int> boltInput( length );


    for (int i = 0; i < length; ++i)
    {
        stdInput[i] = 1;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(stdInput.begin( ) + offset, stdInput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce( boltInput.begin( ) + offset, boltInput.end( ), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}


TEST( ReduceStdVectWithInit, OffsetTestDeviceVectorSerialCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::cl::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );


    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(  stdInput.begin( ) + offset, stdInput.end( ),
                                               init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce(  ctl, dVectorA.begin( ) + offset, dVectorA.end( ),
                                                init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( ReduceStdVectWithInit, OffsetTestDeviceVectorMultiCoreCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::cl::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );

    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(stdInput.begin( ) + offset, stdInput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce( dVectorA.begin( ) + offset, dVectorA.end( ), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( ReduceStdVectWithInit, OffsetTestMultiCoreCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(stdInput.begin( ) + offset, stdInput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce( stdInput.begin( ) + offset, stdInput.end( ), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( ReduceStdVectWithInit, OffsetTestSerialCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(stdInput.begin( ) + offset, stdInput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce( stdInput.begin( ) + offset, stdInput.end( ), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}


TYPED_TEST_CASE_P( ReduceArrayTest );

TYPED_TEST_P( ReduceArrayTest, Normal )
{

    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(),ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::cl::reduce( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::cl::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( ReduceArrayTest, GPU_DeviceNormal )
{
  

   typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;  
  
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::cl::reduce(ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init, bolt::cl::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );

}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( ReduceArrayTest, CPU_DeviceNormal )
{
   typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;    
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize  > ArrayCont;

    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    bolt::cl::control ctl;
    concurrency::accelerator cpuAccelerator = concurrency::accelerator(concurrency::accelerator::cpu_accelerator);
    ctl.setAccelerator(cpuAccelerator);


    ArrayType boltReduce = bolt::cl::reduce( ctl, ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::cl::square<ArrayType>(), init,
                                                       bolt::cl::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

TYPED_TEST_P( ReduceArrayTest, MultipliesFunction )
{
   typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;    
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize  > ArrayCont;

    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::cl::reduce( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::cl::plus<ArrayType>( ));

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}

TYPED_TEST_P( ReduceArrayTest, GPU_DeviceMultipliesFunction )
{
   typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;    
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize  > ArrayCont;
#if OCL_CONTEXT_BUG_WORKAROUND
  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control c_gpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));  
#else
    ::Concurrency::accelerator accel(::Concurrency::accelerator::default_accelerator);
    bolt::cl::control c_gpu(accel);
#endif


    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::cl::reduce( c_gpu,ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::cl::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceMultipliesFunction )
{
   typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;    
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize  > ArrayCont;

    ArrayType init(0);
    //  Calling the actual functions under test
    bolt::cl::control c_cpu;
    concurrency::accelerator cpuAccelerator = concurrency::accelerator(concurrency::accelerator::cpu_accelerator);
    ctl.setAccelerator(cpuAccelerator);

    ArrayType stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    ArrayType boltReduce = bolt::cl::reduce( c_cpu,boltInput.begin( ), boltInput.end( ),init,
                                                       bolt::cl::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( stdInput, boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

#if (TEST_CPU_DEVICE == 1)
REGISTER_TYPED_TEST_CASE_P( TransformArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction,
                                           CPU_DeviceNormal, CPU_DeviceMultipliesFunction);
#else
REGISTER_TYPED_TEST_CASE_P( ReduceArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction );
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class ReduceIntegerVector: public ::testing::TestWithParam< int >
{
public:

    ReduceIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
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
class ReduceFloatVector: public ::testing::TestWithParam< int >
{
public:
    ReduceFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
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

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceDoubleVector: public ::testing::TestWithParam< int >
{
public:
    ReduceDoubleVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
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
class ReduceIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceIntegerDeviceVector( ): stdInput( GetParam( ) ),
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
class ReduceFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceFloatDeviceVector( ): stdInput( GetParam( ) )
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

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceDoubleDeviceVector( ): stdInput( GetParam( ) )
                                                         
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
class ReduceIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceIntegerNakedPointer( ): stdInput( new int[ GetParam( ) ] ), boltInput( new int[ GetParam( ) ] ), 
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
class ReduceFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceFloatNakedPointer( ): stdInput( new float[ GetParam( ) ] ), boltInput( new float[ GetParam( ) ] ), 
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

class ReduceStdVectWithInit :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    ReduceStdVectWithInit():mySize(GetParam()){
    }
};

class StdVectCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    StdVectCountingIterator():mySize(GetParam()){
    }
};

TEST_P( StdVectCountingIterator, withCountingIterator)
{
    std::vector<int> stdInput( mySize );
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
    }
    
    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce( first, last, init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST_P( StdVectCountingIterator, SerialwithCountingIterator)
{
    std::vector<int> stdInput( mySize );
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;
    
    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce( ctl, first, last, init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST_P( StdVectCountingIterator, MultiCorewithCountingIterator)
{
    std::vector<int> stdInput( mySize );
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
    }
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce( ctl, first, last, init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST_P( ReduceStdVectWithInit, withIntWdInit)
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
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::cl::plus<int>( ) );
    int boltTransformReduce= bolt::cl::reduce( boltInput.begin( ), boltInput.end( ), init, bolt::cl::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST_P( ReduceStdVectWithInit, withIntWdInitWithStdPlus)
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
    int stlTransformReduce = std::accumulate(stdInput.begin(), stdInput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::cl::reduce( boltInput.begin( ), boltInput.end( ), init, bolt::cl::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

TEST_P( ReduceStdVectWithInit, withIntWdInitWdAnyFunctor)
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
    int stlTransformReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    int boltTransformReduce= bolt::cl::reduce( boltInput.begin( ), boltInput.end( ), init, bolt::cl::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

INSTANTIATE_TEST_CASE_P( withIntWithInitValue, ReduceStdVectWithInit, ::testing::Range(1, 100, 1) );
INSTANTIATE_TEST_CASE_P( withCountingIterator, StdVectCountingIterator, ::testing::Range(1, 100, 1) );

class ReduceTestMultFloat: public ::testing::TestWithParam<int>{
protected:
    int arraySize;
public:
    ReduceTestMultFloat( ):arraySize( GetParam( ) )
    {}
};

TEST_P (ReduceTestMultFloat, multiplyWithFloats)
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

    float stlTransformReduce = std::accumulate(myArray, myArray + arraySize, 1.0f, std::multiplies<float>());
    float boltTransformReduce = bolt::cl::reduce(myBoltArray, myBoltArray + arraySize,
                                                  1.0f, bolt::cl::multiplies<float>());

    EXPECT_FLOAT_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}

TEST_P( ReduceTestMultFloat, serialFloatValuesWdControl )
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

    float stdTransformReduceValue = std::accumulate(A.begin(), A.end(), 0.0f, std::plus<float>());
    float boltClTransformReduce = bolt::cl::reduce(boltVect.begin(), boltVect.end(), 0.0f, bolt::cl::plus<float>());

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdTransformReduceValue, boltClTransformReduce );
}

INSTANTIATE_TEST_CASE_P(serialValues, ReduceTestMultFloat, ::testing::Range(1, 100, 10));
INSTANTIATE_TEST_CASE_P(multiplyWithFloatPredicate, ReduceTestMultFloat, ::testing::Range(1, 20, 1));
//end of new 2

class ReduceTestMultDouble: public ::testing::TestWithParam<int>{
protected:
        int arraySize;
public:
    ReduceTestMultDouble():arraySize(GetParam()){
    }
};

TEST_P (ReduceTestMultDouble, multiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myBoltArray[i] = myArray[i];
    }

    double stlTransformReduce = std::accumulate(myArray, myArray + arraySize, 1.0, std::multiplies<double>());

    double boltTransformReduce = bolt::cl::reduce(myBoltArray, myBoltArray + arraySize, 1.0,
                                                           bolt::cl::multiplies<double>());
    
    EXPECT_DOUBLE_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}

#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( multiplyWithDoublePredicate, ReduceTestMultDouble, ::testing::Range(1, 20, 1) );
#endif

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

        std::generate(stdInput, stdInput + size, generateRandom<double>);

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

TEST_P( ReduceIntegerVector, Normal )
{

    int init(0);
    //  Calling the actual functions under test
    int stlReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    int boltReduce = bolt::cl::reduce( boltInput.begin( ), boltInput.end( ), init,
                                                       bolt::cl::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( ReduceFloatVector, Normal )
{
    float init(0);
    //  Calling the actual functions under test
    float stlReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    float boltReduce = bolt::cl::reduce( boltInput.begin( ), boltInput.end( ),init,
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
TEST_P( ReduceDoubleVector, Inplace )
{
    double init(0);
    //  Calling the actual functions under test
    double stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    double boltReduce = bolt::cl::reduce( boltInput.begin( ), boltInput.end( ),init,
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


TEST_P( ReduceIntegerNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined(_WIN32)
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    int init(0);
    //  Calling the actual functions under test
    int stlReduce = std::accumulate(wrapStdInput,wrapStdInput + endIndex, init);

    int boltReduce = bolt::cl::reduce( wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::cl::plus<int>());
#else

    int init(0);
    //  Calling the actual functions under test
    int stlReduce = std::accumulate(stdInput,stdInput + endIndex, init);

    int boltReduce = bolt::cl::reduce( boltInput, boltInput + endIndex, init,
                                                       bolt::cl::plus<int>());
    
#endif    

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( ReduceFloatNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );
#if defined(_WIN32)
    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );

    float init(0);
    //  Calling the actual functions under test
    float stlReduce = std::accumulate(wrapStdInput,wrapStdInput + endIndex, init);

    float boltReduce = bolt::cl::reduce( wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::cl::plus<float>());
    
#else
    float init(0);
    //  Calling the actual functions under test
    float stlReduce = std::accumulate(stdInput,stdInput + endIndex, init);

    float boltReduce = bolt::cl::reduce( boltInput, boltInput + endIndex, init,
                                                       bolt::cl::plus<float>());    
#endif    

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}


#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );
#if defined(_WIN32)
    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< double* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );

    double init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::cl::negate<double>());
    double stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    double boltReduce = bolt::cl::reduce( wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::cl::plus<double>());
#else
    
    double init(0);
    //  Calling the actual functions under test
    double stlReduce = std::accumulate(stdInput,stdInput + endIndex, init);

    double boltReduce = bolt::cl::reduce( boltInput, boltInput + endIndex, init,
                                                       bolt::cl::plus<double>());        
#endif    

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif

std::array<int, 15> TestValues = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^22
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceIntegerVector, ::testing::ValuesIn( TestValues.begin(),TestValues.end()));
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceFloatVector, ::testing::Range(4096, 65536, 555 ) ); //2^12 to 2^16
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceFloatVector, ::testing::ValuesIn( TestValues.begin(),TestValues.end()));
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceDoubleVector, ::testing::Range(  65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceDoubleVector, ::testing::ValuesIn( TestValues.begin(), TestValues.end()));
#endif
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceIntegerDeviceVector, ::testing::Range(  1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                      TestValues.end()));
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceFloatDeviceVector, ::testing::Range(  1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                   TestValues.end()));
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceDoubleDeviceVector, ::testing::Range(  1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceDoubleDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                   TestValues.end() ) );
#endif
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceIntegerNakedPointer, ::testing::Range(  1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                     TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceFloatNakedPointer, ::testing::Range(  1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                   TestValues.end() ) );

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

BOLT_FUNCTOR( UDD,
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

BOLT_FUNCTOR( UDDplus,
struct UDDplus
{
   UDD operator() (const UDD &lhs, const UDD &rhs) const
   {
     UDD _result;
     _result.a = lhs.a + rhs.a;
     _result.b = lhs.b + rhs.b;
     return _result;
   }

};
);


TEST( ReduceUDD , UDDPlusOperatorInts )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< UDD > refInput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }
    bolt::cl::device_vector< UDD >input( refInput.begin(), refInput.end() );

    UDD UDDzero;
    UDDzero.a = 0;
    UDDzero.b = 0;

    // call reduce
    UDDplus plusOp;
    UDD boltReduce = bolt::cl::reduce( input.begin(), input.end(), UDDzero, plusOp );
    UDD stdReduce =  std::accumulate( refInput.begin(), refInput.end(), UDDzero, plusOp );

    EXPECT_EQ(stdReduce,boltReduce);

}


TEST( ReduceDevice , DeviceVectoroffset )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > stdinput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      stdinput[i] = i;
    }
    bolt::cl::device_vector< int > input( stdinput.begin(), stdinput.end() );
    // call reduce

    int boltReduce = bolt::cl::reduce( input.begin() + 10 , input.end(), 0, bolt::cl::plus<int>() );
    int stdReduce =  523731;

    EXPECT_EQ(boltReduce,stdReduce);

}


/* TEST( Reduceint , KcacheTest )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > refInput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      refInput[i] = i;
    }

    bolt::cl::device_vector< int > gpuInput( refInput.begin(), refInput.end() );

    //Call reduce with GPU device because the default is a GPU device
    bolt::cl::control ctrl = bolt::cl::control::getDefault();

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
    bolt::cl::device_vector< int > cpuInput( refInput.begin( ), refInput.end( ), CL_MEM_READ_WRITE, cpu_ctrl );

    int initzero = 0;
    int boltReduceGpu = bolt::cl::reduce( ctrl, gpuInput.begin(), gpuInput.end(), initzero );

    int boltReduceCpu = bolt::cl::reduce( cpu_ctrl, cpuInput.begin(), cpuInput.end(), initzero );

    //Call reference code
    int stdReduce =  std::accumulate( refInput.begin(), refInput.end(), initzero );

    EXPECT_EQ(boltReduceGpu,stdReduce);
    EXPECT_EQ(boltReduceCpu,stdReduce);
} */

INSTANTIATE_TYPED_TEST_CASE_P( Integer, ReduceArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, ReduceArrayTest, FloatTests );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TYPED_TEST_CASE_P( Double, ReduceArrayTest, DoubleTests );
#endif 




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



// Simple test case for bolt::reduce:
// Sum together specified numbers, compare against STL::accumulate function.
// Demonstrates:
//    * use of bolt with STL::vector iterators
//    * use of bolt with default plus 
//    * use of bolt with explicit plus argument
void simpleReduce1(int aSize)
{
    std::vector<int> A(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = i;
    };

    int stlReduce = std::accumulate(A.begin(), A.end(), 0);

    int boltReduce = bolt::cl::reduce(A.begin(), A.end(), 0, bolt::cl::plus<int>());
    //int boltReduce2 = bolt::cl::reduce(A.begin(), A.end());  // same as above...

    checkResult("simpleReduce1", stlReduce, boltReduce);
    //printf ("Sum: stl=%d,  bolt=%d %d\n", stlReduce, boltReduce, boltReduce2);
};


// Demonstrates use of bolt::control structure to control execution of routine.
void simpleReduce_TestControl(int aSize, int numIters, int deviceIndex)
{
    std::vector<int> A(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = i;
    };

    //  Tests need to be a bit more sophisticated before hardcoding which device to
    // use in a system (my default is device 1); 
    //  this should be configurable on the command line
    // Create an OCL context, device, queue.


  // FIXME - temporarily disable use of new control queue here:
#if OCL_CONTEXT_BUG_WORKAROUND
  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control c( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 )); 
#else
  MyOclContext ocl = initOcl(CL_DEVICE_TYPE_GPU, deviceIndex);
    bolt::cl::control c(ocl._queue);  // construct control structure from the queue.
#endif


    //printContext(c.context());

    //c.debug(bolt::cl::control::debug::Compile + bolt::cl::control::debug::SaveCompilerTemps);

    int stlReduce = std::accumulate(A.begin(), A.end(), 0);
    int boltReduce = 0;

    char testTag[2000];
#if defined(_WIN32)
    sprintf_s(testTag, 2000, "simpleReduce_TestControl sz=%d iters=%d, device=%s", aSize, numIters, 
        c.getDevice( ).getInfo<CL_DEVICE_NAME>( ).c_str( ) );
#else

    sprintf(testTag, "simpleReduce_TestControl sz=%d iters=%d, device=%s", aSize, numIters, 
        c.getDevice( ).getInfo<CL_DEVICE_NAME>( ).c_str( ) );
#endif


#if defined(_WIN32)
    __int64 start = StartProfile();
#endif
    for (int i=0; i<numIters; i++) {
        boltReduce = bolt::cl::reduce( c, A.begin(), A.end(), 0);
    }
#if defined(_WIN32)
    EndProfile(start, numIters, testTag);
#endif

    checkResult(testTag, stlReduce, boltReduce);
};


// Demonstrates use of bolt::control structure to control execution of routine.
void simpleReduce_TestSerial(int aSize)
{
    std::vector<int> A(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = i;
    };


   cl::CommandQueue myCommandQueue = bolt::cl::control::getDefaultCommandQueue();
   bolt::cl::control c(myCommandQueue);

  //  bolt::cl::control c;  // construct control structure from the queue.
   // c.forceRunMode(bolt::cl::control::SerialCpu);

    int stlReduce = std::accumulate(A.begin(), A.end(), 0);
    int boltReduce = 0;

    boltReduce = bolt::cl::reduce(c, A.begin(), A.end());


    checkResult("TestSerial", stlReduce, boltReduce);
};


void simpleReduce_countingiterator(float start,int size)
{

    bolt::cl::counting_iterator<float> first(start);
    bolt::cl::counting_iterator<float> last = first +  size;

    std::vector<float> a(size);

    for (int i=0; i < size; i++) {
        a[i] = i+start;
    };
    
    float stlReduce = std::accumulate(a.begin(), a.end(), 0.0F);
    float boltReduce = bolt::cl::reduce(first, last, 0.0F);

    checkResult("TestSerial", stlReduce, boltReduce);
};



TEST(sanity_reduce__reducedoubleStdVect, serialNegdoubleValuesWithPlusOp){
int sizeOfStdVect = 5;
std::vector<double> vect1(sizeOfStdVect);

vect1[0] = 5.6;
vect1[1] = 3.8;
vect1[2] = 4.3;
vect1[3] = 0.0;
vect1[4] = 0.0;
/*
for (int i = 0; i < sizeOfStdVect; i++){
vect1[i] = (-1) * (i + 1.0f);
}*/

bolt::cl::control my_ctl = bolt::cl::control::getDefault();

double stlAccumulate = std::accumulate(vect1.begin(), vect1.end(), 3.0);
double boltClReduce = bolt::cl::reduce(my_ctl, vect1.begin(), vect1.end(), 3.0, bolt::cl::plus<double>());
// double boltClReduce = bolt::cl::reduce(my_ctl, vect1.begin(), vect1.end(), 0.0f, bolt::cl::plus());

EXPECT_DOUBLE_EQ(stlAccumulate, boltClReduce);
}

TEST(ReduceAuto, Reduce386717)
{
  std::vector<int> vect1(100);
  std::fill(vect1.begin(), vect1.end(), 1);

  bolt::cl::control my_ctl;
  my_ctl.setForceRunMode( bolt::cl::control::Automatic );

  int stlAccumulate = 0;
  int boltClReduce = 0;

  stlAccumulate = std::accumulate(vect1.begin(), vect1.end(), 0);
  boltClReduce = bolt::cl::reduce(my_ctl, vect1.begin(), vect1.end());

  EXPECT_EQ(stlAccumulate, boltClReduce);
}



#if 0
// Disable test since the buffer interface is moving to device_vector.
void reduce_TestBuffer() {
    
    int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cl::Buffer A(CL_MEM_USE_HOST_PTR, sizeof(int) * 10, a); // create a buffer from a.

    // note type of date in the buffer ("int") explicitly specified.
    int sum = bolt::cl::reduce2<int>(A, 0, bolt::cl::plus<int>()); 
};
#endif


int _tmain(int argc, _TCHAR* argv[])
{
#if defined( ENABLE_TBB )
   // testTBB( );
  //  testdoubleTBB();
    testDUDDTBB();
   // testTBBDevicevector();
#endif
  /*  testDeviceVector();
    int numIters = 100;
    simpleReduce_TestControl(1024000, numIters, 0);
    simpleReduce_TestControl(100, 1, 0);
    simpleReduce1(256);
    simpleReduce1(1024);    
    simpleReduce_TestControl(100, 1, 0);
    simpleReduce_TestSerial(1000);
    simpleReduce_countingiterator(20.05F,10);    */
    //simpleReduce_TestControl(1024000, numIters, 1); // may fail on systems with only one GPU installed.

    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Register our minidump generating logic
//    bolt::miniDumpSingleton::enableMiniDumps( );

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


    return 0;
}

