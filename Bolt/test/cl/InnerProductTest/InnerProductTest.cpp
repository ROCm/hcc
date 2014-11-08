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

// InnerProductTest.cpp : Defines the entry point for the console application.
//
#define OCL_CONTEXT_BUG_WORKAROUND 1
#define TEST_DOUBLE 1
#pragma warning(disable: 4996)
#include <iostream>
#include <algorithm>  // for testing against STL functions.
#include <numeric>
#include <type_traits>
#include <gtest/gtest.h>
#include <bolt/unicode.h>

#include "stdafx.h"
#include "bolt/cl/inner_product.h"
#include "bolt/cl/functional.h"
#include "bolt/cl/device_vector.h"
#include "common/myocl.h"
#include "bolt/cl/control.h"
#include "bolt/cl/iterator/counting_iterator.h"
#include "bolt/miniDump.h"
#include "common/test_common.h"


void testDeviceVector()
{
    const int aSize = 64;
    std::vector<int> hA(aSize), hB(aSize);

    for(int i=0; i<aSize; i++) {
        hA[i] = i;
         hB[i] = i;
    };

    bolt::cl::device_vector<int> dA(hA.begin(),hA.end()); 
    bolt::cl::device_vector<int> dB(hB.begin(),hB.end()); 
    
    int hSum = std::inner_product(hA.begin(), hA.end(), hB.begin(), 1);

    int sum = bolt::cl::inner_product(  dA.begin(), dA.end(),
                                        dB.begin(), 1, bolt::cl::plus<int>(), bolt::cl::multiplies<int>()  );
};

#if defined(_WIN32)
// Super-easy windows profiling interface.
// Move to timing infrastructure when that becomes available.
long long  StartProfile() {
    long long begin;
    QueryPerformanceCounter((LARGE_INTEGER*)(&begin));
    return begin;
};

void EndProfile(long long  start, int numTests, std::string msg) {
    long long  end, freq;
    QueryPerformanceCounter((LARGE_INTEGER*)(&end));
    QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
    double duration = (end - start)/(double)(freq);
    printf("%s %6.2fs, numTests=%d %6.2fms/test\n", msg.c_str(), duration, numTests, duration*1000.0/numTests);
};

#endif
/////////////////////////////////////////////////////////////////////////////////////////////
//  GTEST CASES
/////////////////////////////////////////////////////////////////////////////////////////////
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
class InnerProductArrayTest: public ::testing::Test
{
public:
    InnerProductArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<ArrayType>);
        std::generate(stdInput2.begin(), stdInput2.end(), generateRandom<ArrayType>);
        boltInput = stdInput;
        boltInput2 = stdInput2;
    };

    virtual void TearDown( )
    {};

    virtual ~InnerProductArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize =  std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType, ArraySize > stdInput, boltInput, stdInput2, boltInput2;
    int m_Errors;
};

TYPED_TEST_CASE_P( InnerProductArrayTest );

TYPED_TEST_P( InnerProductArrayTest, Normal )
{
    typedef typename InnerProductArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;  
    
    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlInnerProduct = std::inner_product( InnerProductArrayTest< gtest_TypeParam_ >::stdInput.begin(),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.end(),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput2.begin(), init,
                                                   std::plus<ArrayType>(), std::multiplies<ArrayType>());

    ArrayType boltInnerProduct = bolt::cl::inner_product(  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.begin( ), InnerProductArrayTest< gtest_TypeParam_ >::boltInput.end( ), InnerProductArrayTest< gtest_TypeParam_ >::boltInput2.begin( ),init,
                                                        bolt::cl::plus<ArrayType>(),bolt::cl::multiplies<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.begin( ),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance(  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.begin( ),  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlInnerProduct, boltInnerProduct );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput,  InnerProductArrayTest< gtest_TypeParam_ >::boltInput );
    cmpStdArray< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput2,  InnerProductArrayTest< gtest_TypeParam_ >::boltInput2 );
}

TYPED_TEST_P( InnerProductArrayTest, GPU_DeviceNormal )
{
        typedef typename InnerProductArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;  
    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlInnerProduct = std::inner_product( InnerProductArrayTest< gtest_TypeParam_ >::stdInput.begin(),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.end(),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput2.begin(), init);
    ArrayType boltInnerProduct = bolt::cl::inner_product( InnerProductArrayTest< gtest_TypeParam_ >::boltInput.begin( ), InnerProductArrayTest< gtest_TypeParam_ >::boltInput.end( ), InnerProductArrayTest< gtest_TypeParam_ >::boltInput2.begin( ),init);

    typename ArrayCont::difference_type stdNumElements = std::distance(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.begin( ),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance(  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.begin( ),  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlInnerProduct, boltInnerProduct );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput,  InnerProductArrayTest< gtest_TypeParam_ >::boltInput );
    cmpStdArray< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput2,  InnerProductArrayTest< gtest_TypeParam_ >::boltInput2 );

}

TYPED_TEST_P( InnerProductArrayTest, MultipliesFunction )
{
        typedef typename InnerProductArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;  

    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlInnerProduct = std::inner_product(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.begin(),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.end(),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput2.begin(), init);

    ArrayType boltInnerProduct =bolt::cl::inner_product(  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.begin( ), InnerProductArrayTest< gtest_TypeParam_ >::boltInput.end( ), InnerProductArrayTest< gtest_TypeParam_ >::boltInput2.begin( ),init);

    typename ArrayCont::difference_type stdNumElements = std::distance(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.begin( ),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance(  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.begin( ),  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlInnerProduct, boltInnerProduct );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput,  InnerProductArrayTest< gtest_TypeParam_ >::boltInput );
    cmpStdArray< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput2,  InnerProductArrayTest< gtest_TypeParam_ >::boltInput2 );
    // FIXME - releaseOcl(ocl);
}

TYPED_TEST_P( InnerProductArrayTest, GPU_DeviceMultipliesFunction )
{
        typedef typename InnerProductArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;  
#if OCL_CONTEXT_BUG_WORKAROUND
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control c_gpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));  
#else
    ::Concurrency::accelerator accel(::Concurrency::accelerator::default_accelerator);
    bolt::cl::control c_gpu(accel);
#endif
    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlInnerProduct = std::inner_product( InnerProductArrayTest< gtest_TypeParam_ >::stdInput.begin(),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.end(),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput2.begin(), init,
                                                    std::plus< ArrayType >(), std::multiplies< ArrayType >());

    ArrayType boltInnerProduct = bolt::cl::inner_product( c_gpu, InnerProductArrayTest< gtest_TypeParam_ >::boltInput.begin( ), InnerProductArrayTest< gtest_TypeParam_ >::boltInput.end( ), InnerProductArrayTest< gtest_TypeParam_ >::boltInput2.begin(), 
                                                init, bolt::cl::plus<ArrayType>(), bolt::cl::multiplies<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.begin( ),  InnerProductArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance(  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.begin( ),  InnerProductArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlInnerProduct, boltInnerProduct );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput,  InnerProductArrayTest< gtest_TypeParam_ >::boltInput );
    cmpStdArray< ArrayType, InnerProductArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  InnerProductArrayTest< gtest_TypeParam_ >::stdInput2,  InnerProductArrayTest< gtest_TypeParam_ >::boltInput2 );
    // FIXME - releaseOcl(ocl);
}
REGISTER_TYPED_TEST_CASE_P( InnerProductArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction );

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class InnerProductIntegerVector: public ::testing::TestWithParam< int >
{
public:

    InnerProductIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                                  stdInput2( GetParam( ) ), boltInput2( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        std::generate(stdInput2.begin(), stdInput2.end(), generateRandom<int>);
        boltInput = stdInput;
        boltInput2 = stdInput2;
    }

protected:
    std::vector< int > stdInput, boltInput, stdInput2, boltInput2;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class InnerProductFloatVector: public ::testing::TestWithParam< int >
{
public:
    InnerProductFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                                stdInput2( GetParam( ) ), boltInput2( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        std::generate(stdInput2.begin(), stdInput2.end(), generateRandom<float>);
        boltInput = stdInput;
        boltInput2 = stdInput2;
    }

protected:
    std::vector< float > stdInput, boltInput, stdInput2, boltInput2;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class InnerProductIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    InnerProductIntegerDeviceVector( ): stdInput( GetParam( ) ),
                                        stdInput2( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        std::generate(stdInput2.begin(), stdInput2.end(), generateRandom<int>);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        /*for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltInput2[i] = stdInput2[i];
        }*/
    }

protected:
    std::vector< int > stdInput, stdInput2;
    //bolt::cl::device_vector< int > boltInput, boltInput2; 
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class InnerProductFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    InnerProductFloatDeviceVector( ): stdInput( GetParam( ) ), 
                                      stdInput2( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        std::generate(stdInput2.begin(), stdInput2.end(), generateRandom<float>);
        //FIXME - The above should work but the below loop is used. 
        /*for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltInput2[i] = stdInput2[i];
        }*/
    }

protected:
    std::vector< float > stdInput, stdInput2;
    //bolt::cl::device_vector< float > boltInput, boltInput2;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class InnerProductIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    InnerProductIntegerNakedPointer( ): stdInput( new int[ GetParam( ) ] ), boltInput( new int[ GetParam( ) ] ), 
                                        stdInput2( new int[ GetParam( ) ] ), boltInput2( new int[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, generateRandom<int>);
        std::generate(stdInput2, stdInput2 + size, generateRandom<int>);
		for (size_t i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
            boltInput2[i] = stdInput2[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdInput2;
        delete [] boltInput2;
};

protected:
     int* stdInput;
     int* boltInput;
     int* stdInput2;
     int* boltInput2;

};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class InnerProductFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    InnerProductFloatNakedPointer( ): stdInput( new float[ GetParam( ) ] ), boltInput( new float[ GetParam( ) ] ), 
                                      stdInput2( new float[ GetParam( ) ] ), boltInput2( new float[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, generateRandom<float>);
        std::generate(stdInput2, stdInput2 + size, generateRandom<float>);
		for (size_t i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
            boltInput2[i] = stdInput2[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdInput2;
        delete [] boltInput2;
    };

protected:
     float* stdInput;
     float* boltInput;
     float* stdInput2;
     float* boltInput2;
};

TEST( InnerProductStdVectWithInit, withIntWdInitWithStdPlusMinus)
{
    //int mySize = 10;
    int init = 10;
    size_t mySize = 1<<16;
    std::vector<int> stdInput (mySize);
    std::vector<int> stdInput2 (mySize);

    std::vector<int> boltInput (mySize);
    std::vector<int> boltInput2 (mySize);

	for (size_t i = 0; i < mySize; ++i){
        stdInput[i] = (int)i;
        stdInput2[i] = (int)(i+1);
        boltInput[i] = stdInput[i];
        boltInput2[i] = stdInput2[i];
    }
    
    //  Calling the actual functions under test
    int stlInnerProduct = std::inner_product(stdInput.begin(), stdInput.end(), stdInput2.begin(),init,
                                             std::plus<int>(), std::minus<int>() );
    int boltInnerProduct= bolt::cl::inner_product( boltInput.begin( ), boltInput.end( ), boltInput2.begin(), 
                                                   init, bolt::cl::plus<int>(), bolt::cl::minus<int>());

    EXPECT_EQ(stlInnerProduct, boltInnerProduct);
}

TEST( CPUInnerProductStdVectWithInit, withIntWdInitWithStdPlusMinus)
{
    //int mySize = 10;
    int init = 10;
    size_t mySize = 1<<16;
    std::vector<int> stdInput (mySize);
    std::vector<int> stdInput2 (mySize);

    std::vector<int> boltInput (mySize);
    std::vector<int> boltInput2 (mySize);

	for (size_t i = 0; i < mySize; ++i){
        stdInput[i] = (int)i;
        stdInput2[i] = (int)(i+1);
        boltInput[i] = stdInput[i];
        boltInput2[i] = stdInput2[i];
    }
    
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    //  Calling the actual functions under test
    int stlInnerProduct = std::inner_product(stdInput.begin(),stdInput.end(), stdInput2.begin(),init, std::plus<int>(), 
                                             std::minus<int>() );
    int boltInnerProduct= bolt::cl::inner_product(ctl, boltInput.begin( ), boltInput.end( ), boltInput2.begin(), 
                                                  init, bolt::cl::plus<int>(), bolt::cl::minus<int>());

    EXPECT_EQ(stlInnerProduct, boltInnerProduct);
}

TEST( MultiCoreInnerProductStdVectWithInit, withIntWdInitWithStdPlusMinus)
{
    //int mySize = 10;
    int init = 10;
    size_t mySize = 1<<16;
    std::vector<int> stdInput (mySize);
    std::vector<int> stdInput2 (mySize);

    std::vector<int> boltInput (mySize);
    std::vector<int> boltInput2 (mySize);

	for (size_t i = 0; i < mySize; ++i){
        stdInput[i] = (int)i;
        stdInput2[i] = (int)(i+1);
        boltInput[i] = stdInput[i];
        boltInput2[i] = stdInput2[i];
    }
    
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    //  Calling the actual functions under test
    int stlInnerProduct = std::inner_product(stdInput.begin(), stdInput.end(), stdInput2.begin(),init,std::plus<int>(),
                                              std::minus<int>() );
    int boltInnerProduct= bolt::cl::inner_product(ctl, boltInput.begin( ), boltInput.end( ), boltInput2.begin(),
                                                  init, bolt::cl::plus<int>(), bolt::cl::minus<int>());

    EXPECT_EQ(stlInnerProduct, boltInnerProduct);
}

class InnerProductTestMultFloat: public ::testing::TestWithParam<int>{
protected:
    int arraySize;
public:
    InnerProductTestMultFloat( ):arraySize( GetParam( ) )
    {}
};

class InnerProductCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    InnerProductCountingIterator():mySize(GetParam()){
    }
};

TEST_P( InnerProductCountingIterator, withCountingIterator)
{
    bolt::cl::counting_iterator<int> first1(0);
    bolt::cl::counting_iterator<int> last1 = first1 +  mySize;
    bolt::cl::counting_iterator<int> first2(1);
    int init = 10;

    std::vector<int> input1(mySize);
    std::vector<int> input2(mySize);
   

    for (int i=0; i < mySize; i++) {
        input1[i] = i;
        input2[i] = i+1;
    };
    
    int stlInnerProduct = std::inner_product(input1.begin(), input1.end(), input2.begin(),init, std::multiplies<int>(),
        std::plus<int>());
    int boltInnerProduct = bolt::cl::inner_product(  first1,
                                                     last1,
                                                     first2,
                                                     init, bolt::cl::multiplies<int>(),
                                                     bolt::cl::plus<int>());

    EXPECT_EQ(stlInnerProduct, boltInnerProduct);
}

TEST_P( InnerProductCountingIterator, SerialwithCountingIterator)
{
    bolt::cl::counting_iterator<int> first1(0);
    bolt::cl::counting_iterator<int> last1 = first1 +  mySize;
    bolt::cl::counting_iterator<int> first2(1);
    int init = 10;

    std::vector<int> input1(mySize);
    std::vector<int> input2(mySize);
   

    for (int i=0; i < mySize; i++) {
        input1[i] = i;
        input2[i] = i+1;
    };
    
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    int stlInnerProduct = std::inner_product(input1.begin(), input1.end(),input2.begin(),init, std::multiplies<int>(),
        std::plus<int>());
    int boltInnerProduct = bolt::cl::inner_product(ctl, first1, last1, first2, init, bolt::cl::multiplies<int>(),
        bolt::cl::plus<int>());

    EXPECT_EQ(stlInnerProduct, boltInnerProduct);
}

TEST_P( InnerProductCountingIterator, MultiCorewithCountingIterator)
{
    bolt::cl::counting_iterator<int> first1(0);
    bolt::cl::counting_iterator<int> last1 = first1 +  mySize;
    bolt::cl::counting_iterator<int> first2(1);
    int init = 10;

    std::vector<int> input1(mySize);
    std::vector<int> input2(mySize);
   

    for (int i=0; i < mySize; i++) {
        input1[i] = i;
        input2[i] = i+1;
    };
    
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    int stlInnerProduct = std::inner_product(input1.begin(), input1.end(), input2.begin(),init, std::multiplies<int>(),
        std::plus<int>());
    int boltInnerProduct = bolt::cl::inner_product(ctl, first1, last1, first2, init, bolt::cl::multiplies<int>(),
        bolt::cl::plus<int>());

    EXPECT_EQ(stlInnerProduct, boltInnerProduct);
}

TEST_P (InnerProductTestMultFloat, multiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];
    float* myBoltArray2 = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    myArray2[ 0 ] = 1.0f;
    myBoltArray2[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myArray2[i] = myArray2[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
        myBoltArray2[i] = myArray2[i];
    }

    float stlInnerProduct = std::inner_product(  myArray,
                                                 myArray + arraySize,
                                                 myArray2,
                                                 1.0f, std::multiplies<float>(), std::plus<float>()  );
    float boltInnerProduct = bolt::cl::inner_product(myBoltArray, myBoltArray + arraySize, myBoltArray2,
                                                     1.0f, bolt::cl::multiplies<float>(), bolt::cl::plus<float>());

    EXPECT_FLOAT_EQ(stlInnerProduct , boltInnerProduct )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
    delete [] myBoltArray2;
}

TEST_P (InnerProductTestMultFloat, CPUmultiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];
    float* myBoltArray2 = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    myArray2[ 0 ] = 1.0f;
    myBoltArray2[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myArray2[i] = myArray2[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
        myBoltArray2[i] = myArray2[i];
    }

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    float stlInnerProduct = std::inner_product(  myArray,
                                                 myArray + arraySize,
                                                 myArray2,
                                                 1.0f, std::multiplies<float>(), std::plus<float>()  );
    float boltInnerProduct = bolt::cl::inner_product(ctl, myBoltArray, myBoltArray + arraySize, myBoltArray2,
                                                     1.0f, bolt::cl::multiplies<float>(), bolt::cl::plus<float>());

    EXPECT_FLOAT_EQ(stlInnerProduct , boltInnerProduct )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
    delete [] myBoltArray2;
}

TEST_P (InnerProductTestMultFloat, MultiCoremultiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];
    float* myBoltArray2 = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    myArray2[ 0 ] = 1.0f;
    myBoltArray2[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myArray2[i] = myArray2[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
        myBoltArray2[i] = myArray2[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    float stlInnerProduct = std::inner_product(  myArray,
                                                 myArray + arraySize,
                                                 myArray2,
                                                 1.0f, std::multiplies<float>(), std::plus<float>()  );
    float boltInnerProduct = bolt::cl::inner_product(ctl, myBoltArray, myBoltArray + arraySize, myBoltArray2,
                                                     1.0f, bolt::cl::multiplies<float>(), bolt::cl::plus<float>());

    EXPECT_FLOAT_EQ(stlInnerProduct , boltInnerProduct )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
    delete [] myBoltArray2;
}

TEST_P( InnerProductTestMultFloat, serialFloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    std::vector<float> boltVect2( arraySize );
    
    float myFloatValues = CL_FLT_EPSILON;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        B[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
        boltVect2[i] = B[i];
    }

    bolt::cl::control ctl;
    float init = 0;

    float stdInnerProductValue = std::inner_product(A.begin(), A.end(), B.begin(), init );
    float boltInnerProduct = bolt::cl::inner_product(ctl.getDefault(), boltVect.begin(), boltVect.end(),
                                                     boltVect2.begin(), init );

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdInnerProductValue, boltInnerProduct );
}

TEST_P( InnerProductTestMultFloat, CPUFloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    std::vector<float> boltVect2( arraySize );
    
    float myFloatValues = CL_FLT_EPSILON;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        B[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
        boltVect2[i] = B[i];
    }

    float init = 0;

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    float stdInnerProductValue = std::inner_product(A.begin(), A.end(), B.begin(), init );
    float boltInnerProduct = bolt::cl::inner_product(ctl, boltVect.begin(), boltVect.end(), boltVect2.begin(), init );

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdInnerProductValue, boltInnerProduct );
}

TEST_P( InnerProductTestMultFloat, MultiCoreFloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    std::vector<float> boltVect2( arraySize );
    
    float myFloatValues = CL_FLT_EPSILON;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        B[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
        boltVect2[i] = B[i];
    }

    float init = 0;

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    float stdInnerProductValue = std::inner_product(A.begin(), A.end(), B.begin(), init );
    float boltInnerProduct = bolt::cl::inner_product(ctl, boltVect.begin(), boltVect.end(), boltVect2.begin(), init );

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdInnerProductValue, boltInnerProduct );
}

INSTANTIATE_TEST_CASE_P(serialValues, InnerProductCountingIterator, ::testing::Range(1, 100, 10));
INSTANTIATE_TEST_CASE_P(serialValues, InnerProductTestMultFloat, ::testing::Range(1, 100, 10));
INSTANTIATE_TEST_CASE_P(multiplyWithFloatPredicate, InnerProductTestMultFloat, ::testing::Range(1, 20, 1));
//end of new 2

class InnerProductTestMultDouble: public ::testing::TestWithParam<int>{
protected:
        int arraySize;
public:
    InnerProductTestMultDouble():arraySize(GetParam()){
    }
};
#if(TEST_DOUBLE == 1)

TEST_P (InnerProductTestMultDouble, multiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    double* myBoltArray2 = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myArray[i] = (double)i + 2.25;
        myBoltArray[i] = myArray[i];
        myBoltArray2[i] = myArray2[i];
    }

    double stlInnerProduct = std::inner_product(myArray, myArray + arraySize, myArray2, 1.0, std::multiplies<double>(),
                                                std::plus<double>());

    double boltInnerProduct = bolt::cl::inner_product(myBoltArray, myBoltArray + arraySize, myBoltArray2, 1.0, 
                                                      bolt::cl::multiplies<double>(), bolt::cl::plus<double>());
    
    EXPECT_DOUBLE_EQ(stlInnerProduct , boltInnerProduct )<<"Values do not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}

TEST_P (InnerProductTestMultDouble, CPUmultiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    double* myBoltArray2 = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myArray[i] = (double)i + 2.25;
        myBoltArray[i] = myArray[i];
        myBoltArray2[i] = myArray2[i];
    }

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    double stlInnerProduct = std::inner_product(myArray, myArray + arraySize, myArray2, 1.0, std::multiplies<double>(), 
                                                 std::plus<double>());

    double boltInnerProduct = bolt::cl::inner_product(ctl, myBoltArray, myBoltArray + arraySize, myBoltArray2, 1.0,
                                                      bolt::cl::multiplies<double>(), bolt::cl::plus<double>());
    
    EXPECT_DOUBLE_EQ(stlInnerProduct , boltInnerProduct )<<"Values do not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}

TEST_P (InnerProductTestMultDouble, MultiCoremultiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    double* myBoltArray2 = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myArray[i] = (double)i + 2.25;
        myBoltArray[i] = myArray[i];
        myBoltArray2[i] = myArray2[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    double stlInnerProduct = std::inner_product(myArray, myArray + arraySize, myArray2, 1.0, std::multiplies<double>(),
                                                std::plus<double>());

    double boltInnerProduct = bolt::cl::inner_product(ctl, myBoltArray, myBoltArray + arraySize, myBoltArray2, 1.0,
                                                      bolt::cl::multiplies<double>(), bolt::cl::plus<double>());
    
    EXPECT_DOUBLE_EQ(stlInnerProduct , boltInnerProduct )<<"Values do not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}

#endif

#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( multiplyWithDoublePredicate, InnerProductTestMultDouble, ::testing::Range(1, 20, 1) ); 
#endif

std::array<int, 15> TestValues = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( InnerProductRange, InnerProductIntegerVector, ::testing::Range( 0, 1024, 7 ) );
INSTANTIATE_TEST_CASE_P( InnerProductValues, InnerProductIntegerVector, ::testing::ValuesIn( TestValues.begin(), 
                         TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( InnerProductRange, InnerProductFloatVector, ::testing::Range( 0, 1024, 3 ) );
INSTANTIATE_TEST_CASE_P( InnerProductValues, InnerProductFloatVector, ::testing::ValuesIn( TestValues.begin(), 
                         TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( InnerProductRange, InnerProductIntegerDeviceVector, ::testing::Range( 0, 1024, 53 ) );
INSTANTIATE_TEST_CASE_P( InnerProductValues, InnerProductIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                         TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( InnerProductRange, InnerProductFloatDeviceVector, ::testing::Range( 0, 1024, 53 ) );
INSTANTIATE_TEST_CASE_P( InnerProductValues, InnerProductFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                         TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( InnerProductRange, InnerProductIntegerNakedPointer, ::testing::Range( 0, 1024, 13) );
INSTANTIATE_TEST_CASE_P( InnerProductValues, InnerProductIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                         TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( InnerProductRange, InnerProductFloatNakedPointer, ::testing::Range( 0, 1024, 13) );
INSTANTIATE_TEST_CASE_P( InnerProductValues, InnerProductFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
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

#if 1
BOLT_FUNCTOR( UDD,
struct UDD
{
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
    UDD operator + (const UDD &rhs) const
    {
      UDD _result;
      _result.a = a + rhs.a;
      _result.b = b + rhs.b;
      return _result;
    }

    UDD operator * (const UDD &rhs) const
    {
      UDD _result;
      _result.a = a * rhs.a;
      _result.b = b * rhs.b;
      return _result;
    }

    UDD()
        : a(0),b(0) { }
    UDD(int _in)
        : a(_in), b(_in +1)  { }
};
);

//BOLT_CREATE_TYPENAME( bolt::cl::multiplies< UDD > );
//BOLT_CREATE_TYPENAME( bolt::cl::plus< UDD > );
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< UDD >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< UDD >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_FUNCTOR( UDDmul,
struct UDDmul
{
   UDD operator() (const UDD &lhs, const UDD &rhs) const
   {
     UDD _result;
     _result.a = lhs.a * rhs.a;
     _result.b = lhs.b * rhs.b;
     return _result;
   }

};
);

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

TEST( InnerProductUDD , UDDPlusOperatorInts )
{
    //setup containers
    int length = 1024;
    std::vector< UDD > refInput( length );
    std::vector< UDD > refInput2( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = refInput2[i].a = i;
      refInput[i].b = refInput2[i].b = i+1;
    }
    bolt::cl::device_vector< UDD >input( refInput.begin(), refInput.end() );
    bolt::cl::device_vector< UDD >input2( refInput2.begin(), refInput2.end() );

    UDD UDDzero;
    UDDzero.a = 0;
    UDDzero.b = 0;

    // call InnerProduct
    UDDmul mulOp;
    UDDplus plusOp;

    //bolt::cl::multiplies< UDD > mulOp;
    //bolt::cl::plus< UDD > plusOp;
    UDD stdInnerProduct =  std::inner_product( refInput.begin(), refInput.end(), refInput2.begin(), 
                                               UDDzero, plusOp, mulOp );
    UDD boltInnerProduct = bolt::cl::inner_product( input.begin(), input.end(), input2.begin(),
                                                    UDDzero, plusOp , mulOp);

    EXPECT_EQ(boltInnerProduct,stdInnerProduct);

}

TEST( InnerProductUDD , CPU_UDDPlusOperatorInts )
{
    //setup containers
    int length = 1024;
    std::vector< UDD > refInput( length );
    std::vector< UDD > refInput2( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = refInput2[i].a = i;
      refInput[i].b = refInput2[i].b = i+1;
    }
    bolt::cl::device_vector< UDD >input( refInput.begin(), refInput.end() );
    bolt::cl::device_vector< UDD >input2( refInput2.begin(), refInput2.end() );

    UDD UDDzero;
    UDDzero.a = 0;
    UDDzero.b = 0;

    // call InnerProduct
    UDDmul mulOp;
    UDDplus plusOp;

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::multiplies< UDD > mulOp;
    //bolt::cl::plus< UDD > plusOp;
    UDD stdInnerProduct =  std::inner_product( refInput.begin(), refInput.end(), refInput2.begin(), 
                                               UDDzero, plusOp, mulOp );
    UDD boltInnerProduct = bolt::cl::inner_product( ctl, input.begin(), input.end(), input2.begin(),
                                                    UDDzero, plusOp , mulOp);

    EXPECT_EQ(boltInnerProduct,stdInnerProduct);

}

TEST( InnerProductOffset , DeviceVectorOffset )
{
    //setup containers
    int length = 1024;
    std::vector< int > refInput( length );
    std::vector< int > refInput2( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i] = refInput2[i] = i;
    }
    bolt::cl::device_vector< int >input( refInput.begin(), refInput.end() );
    bolt::cl::device_vector< int >input2( refInput2.begin(), refInput2.end() );

    int stdInnerProduct =  std::inner_product( refInput.begin() + 10, refInput.end(), refInput2.begin() + 10, 
                                               0, std::plus<int>(), std::multiplies<int>() );
    int boltInnerProduct = bolt::cl::inner_product( input.begin() + 10, input.end(), input2.begin() + 10,
                                                    0, bolt::cl::plus<int>() , bolt::cl::multiplies<int>() );

    EXPECT_EQ(boltInnerProduct,stdInnerProduct);

}

TEST( InnerProductOffset , HostVectorOffset )
{
    //setup containers
    int length = 1024;
    std::vector< int > refInput( length );
    std::vector< int > refInput2( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i] = refInput2[i] = i;
    }

    int stdInnerProduct =  std::inner_product( refInput.begin() + 60, refInput.end(), refInput2.begin() + 60, 
                                               0, std::plus<int>(), std::multiplies<int>() );
    int boltInnerProduct = bolt::cl::inner_product( refInput.begin() + 60, refInput.end(), refInput2.begin() + 60,
                                                    0, bolt::cl::plus<int>() , bolt::cl::multiplies<int>() );

    EXPECT_EQ(boltInnerProduct,stdInnerProduct);

}

TEST( InnerProductOffset , DeviceVectorOffsetSerialCpu )
{
    //setup containers
    int length = 1024;
    std::vector< int > refInput( length );
    std::vector< int > refInput2( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i] = refInput2[i] = i;
    }
    bolt::cl::device_vector< int >input( refInput.begin(), refInput.end() );
    bolt::cl::device_vector< int >input2( refInput2.begin(), refInput2.end() );

    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    int stdInnerProduct =  std::inner_product( refInput.begin() + 10, refInput.end(), refInput2.begin() + 10, 
                                               0, std::plus<int>(), std::multiplies<int>() );
    int boltInnerProduct = bolt::cl::inner_product( ctl, input.begin() + 10, input.end(), input2.begin() + 10,
                                                    0, bolt::cl::plus<int>() , bolt::cl::multiplies<int>() );

    EXPECT_EQ(boltInnerProduct,stdInnerProduct);

}

TEST( InnerProductOffset , HostVectorOffsetSerialCpu )
{
    //setup containers
    int length = 1024;
    std::vector< int > refInput( length );
    std::vector< int > refInput2( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i] = refInput2[i] = i;
    }

    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    int stdInnerProduct =  std::inner_product( refInput.begin() + 60, refInput.end(), refInput2.begin() + 60, 
                                               0, std::plus<int>(), std::multiplies<int>() );
    int boltInnerProduct = bolt::cl::inner_product( ctl, refInput.begin() + 60, refInput.end(), refInput2.begin() + 60,
                                                    0, bolt::cl::plus<int>() , bolt::cl::multiplies<int>() );

    EXPECT_EQ(boltInnerProduct,stdInnerProduct);

}


TEST( InnerProductUDD , MultiCore_UDDPlusOperatorInts )
{
    //setup containers
    int length = 1024;
    std::vector< UDD > refInput( length );
    std::vector< UDD > refInput2( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = refInput2[i].a = i;
      refInput[i].b = refInput2[i].b = i+1;
    }
    bolt::cl::device_vector< UDD >input( refInput.begin(), refInput.end() );
    bolt::cl::device_vector< UDD >input2( refInput2.begin(), refInput2.end() );

    UDD UDDzero;
    UDDzero.a = 0;
    UDDzero.b = 0;

    // call InnerProduct
    UDDmul mulOp;
    UDDplus plusOp;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::multiplies< UDD > mulOp;
    //bolt::cl::plus< UDD > plusOp;
    UDD stdInnerProduct =  std::inner_product( refInput.begin(), refInput.end(), refInput2.begin(),
                                               UDDzero, plusOp, mulOp );
    UDD boltInnerProduct = bolt::cl::inner_product( ctl, input.begin(), input.end(), input2.begin(),
                                                    UDDzero, plusOp , mulOp);

    EXPECT_EQ(boltInnerProduct,stdInnerProduct);

}

#endif

INSTANTIATE_TYPED_TEST_CASE_P( Integer, InnerProductArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, InnerProductArrayTest, FloatTests );


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
extern void testDeviceVector();

// Simple test case for bolt::inner_product
// Sum together specified numbers, compare against STL::inner_product function.
// Demonstrates:
//    * use of bolt with STL::vector iterators
//    * use of bolt with default plus 
//    * use of bolt with explicit plus argument
void simpleInProd1(int aSize)
{
    std::vector<int> A(aSize), B(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = B[i] = i;
    };

    int stlInProd = std::inner_product(A.begin(), A.end(), B.begin(), 0);
    int boltInProd = bolt::cl::inner_product(A.begin(), A.end(), B.begin(),0, bolt::cl::plus<int>(),
                                             bolt::cl::multiplies<int>());

    checkResult("simpleInProd1", stlInProd, boltInProd);
 
};



void floatInProd1(int aSize)
{
    std::vector<float> A(aSize), B(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = B[i] = FLT_EPSILON;
    };

    float x = 1.0;
    float stlInProd = std::inner_product(A.begin(), A.end(), B.begin(), x);
    float boltInProd = bolt::cl::inner_product(A.begin(), A.end(), B.begin(), x);

    double error_threshold = 0.0;
    checkResult("floatInProd1", stlInProd, boltInProd,0);
 
};

void ctlInprod(){
    int aSize=64;
    int numIters = 100;
     std::vector<int> A(aSize),B(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = i;
        B[i] = i;
    };

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control c( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 )); 
    //c.setDebugMode(bolt::cl::control::debug::Compile + bolt::cl::control::debug::SaveCompilerTemps);

    int stlReduce = std::inner_product(A.begin(), A.end(), B.begin(),0);
    int boltReduce = 0;
    boltReduce = bolt::cl::inner_product(c, A.begin(), A.end(), B.begin(),0);
   
    checkResult("ctlInProduct", stlReduce, boltReduce);
}

void InProdDV()
{
    const int aSize = 32;
    std::vector<int> hA(aSize), hB(aSize);
    
    for(int i=0; i<aSize; i++) {
        hA[i] = i;
        hB[i] = i;
    };
    bolt::cl::device_vector<int> dA(hA.begin(), hA.end());
    bolt::cl::device_vector<int> dB(hB.begin(), hB.end());
    
    int hSum = std::inner_product(hA.begin(), hA.end(), hB.begin(), 1);

    int sum = bolt::cl::inner_product(dA.begin(), dA.end(), dB.begin(), 1,bolt::cl::plus<int>(),
                                      bolt::cl::multiplies<int>());
    checkResult("InProductDeviceVector", hSum, sum);
    cmpArrays(hA, dA);
    cmpArrays(hB, dB);
    
};

int _tmain(int argc, _TCHAR* argv[])
{
    //Do tests the old way
    int numIters = 100;
    testDeviceVector();
    simpleInProd1(64);
    simpleInProd1(256);
    simpleInProd1(1024);
    simpleInProd1(2048);
    simpleInProd1(4096);
    floatInProd1(64);
    floatInProd1(256);
    floatInProd1(1024);
    floatInProd1(2048);
    floatInProd1(4096);
    ctlInprod();
    InProdDV();

    //Do tests GTEST way
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

    return 0;
}
