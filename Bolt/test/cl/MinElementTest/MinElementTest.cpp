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

// TransformTest.cpp : Defines the entry point for the console application.
//
#define OCL_CONTEXT_BUG_WORKAROUND 1

#define TEST_DOUBLE 1
#define TEST_LARGE_BUFFERS 0

#include <iostream>
#include <algorithm>  // for testing against STL functions.
#include <numeric>
#include <gtest/gtest.h>
#include <type_traits>

#include "bolt/cl/iterator/counting_iterator.h"
#include "bolt/cl/min_element.h"
#include "bolt/cl/functional.h"
#include "bolt/cl/control.h"
#include "stdafx.h"
#include "common/myocl.h"
#include "common/test_common.h"
#include "bolt/miniDump.h"


void testDeviceVector()
{
    const int aSize = 1000;
    std::vector<int> hA(aSize);  

    for(int i=0; i<aSize; i++) {
        hA[i] = i;
    };
	
    bolt::cl::device_vector<int> dA(hA.begin(), hA.end());
    std::vector<int>::iterator smindex = std::min_element(hA.begin(), hA.end());
    bolt::cl::device_vector<int>::iterator bmindex = bolt::cl::min_element(dA.begin(), dA.end());
    
};


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
class MinEArrayTest: public ::testing::Test
{
public:
    MinEArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<ArrayType>);
        boltInput = stdInput;
    };

    virtual void TearDown( )
    {};

    virtual ~MinEArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType, ArraySize > stdInput, boltInput;
    int m_Errors;
};

TYPED_TEST_CASE_P( MinEArrayTest );

/*TYPED_TEST_P( MinEArrayTest, Normal )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
   typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::iterator boltMinE = bolt::cl::min_element( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
}*/

TYPED_TEST_P( MinEArrayTest, comp_Normal )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end(), std::less< ArrayType >() );
   typename  ArrayCont::iterator boltMinE = bolt::cl::min_element( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::less< ArrayType >() );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
}


/*TYPED_TEST_P( MinEArrayTest, SerialNormal )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::iterator boltMinE = bolt::cl::min_element( ctl, MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
}*/

TYPED_TEST_P( MinEArrayTest, comp_SerialNormal )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end(), std::less< ArrayType >() );
   typename  ArrayCont::iterator boltMinE = bolt::cl::min_element( ctl, MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::less< ArrayType >()  );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
}


TYPED_TEST_P( MinEArrayTest, MultiCoreNormal )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::iterator boltMinE = bolt::cl::min_element( ctl, MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( MinEArrayTest, comp_MultiCoreNormal )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end(), std::less< ArrayType >() );
   typename  ArrayCont::iterator boltMinE = bolt::cl::min_element( ctl, MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::less< ArrayType >() );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( MinEArrayTest, GPU_DeviceGreaterFunction )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control c_gpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));


    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::iterator boltMinE = bolt::cl::min_element( c_gpu,MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
    // FIXME - releaseOcl(ocl);
}

//REGISTER_TYPED_TEST_CASE_P( MinEArrayTest, Normal, comp_Normal, SerialNormal, comp_SerialNormal, 
//MultiCoreNormal, comp_MultiCoreNormal, GPU_DeviceGreaterFunction );
REGISTER_TYPED_TEST_CASE_P( MinEArrayTest, comp_Normal, comp_SerialNormal, comp_MultiCoreNormal, GPU_DeviceGreaterFunction, MultiCoreNormal);


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class MinEIntegerVector: public ::testing::TestWithParam< int >
{
public:

    MinEIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        boltInput = stdInput;
    }

protected:
    std::vector< int > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class MinEFloatVector: public ::testing::TestWithParam< int >
{
public:
    MinEFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        boltInput = stdInput;
    }

protected:
    std::vector< float > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class MinEIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    MinEIntegerDeviceVector( ): stdInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        /*for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
        }*/
    }

protected:
    std::vector< int > stdInput;
    //bolt::cl::device_vector< int > boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class MinEFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    MinEFloatDeviceVector( ): stdInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        /*for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
        }*/
    }

protected:
    std::vector< float > stdInput;
    //bolt::cl::device_vector< float > boltInput;
};


class MinEStdVectWithInit :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    MinEStdVectWithInit():mySize(GetParam()){
    }
};


class MinEStdVectandCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    MinEStdVectandCountingIterator():mySize(GetParam()){
    }
};

TEST_P( MinEStdVectWithInit, withInt)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMinE = std::min_element( stdInput.begin( ), stdInput.end() );
    std::vector<int>::iterator boltMinE= bolt::cl::min_element( boltInput.begin( ), boltInput.end() );

    EXPECT_EQ( *stlMinE, *boltMinE );
}

TEST_P( MinEStdVectWithInit, CPUwithInt)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMinE = std::min_element( stdInput.begin( ), stdInput.end() );
    std::vector<int>::iterator boltMinE= bolt::cl::min_element(ctl, boltInput.begin( ), boltInput.end() );

    EXPECT_EQ( *stlMinE, *boltMinE );
}

TEST_P( MinEStdVectWithInit, MultiCorewithInt)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMinE = std::min_element( stdInput.begin( ), stdInput.end() );
    std::vector<int>::iterator boltMinE= bolt::cl::min_element(ctl, boltInput.begin( ), boltInput.end() );

    EXPECT_EQ( *stlMinE, *boltMinE );
}


//TEST_P( MinEStdVectandCountingIterator, withCountingIterator)
//{
//    bolt::cl::counting_iterator<int> first(0);
//    bolt::cl::counting_iterator<int> last = first +  mySize;
//
//    std::vector<int> a(mySize);
//
//    for (int i=0; i < mySize; i++) {
//        a[i] = i;
//    };
//
//    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end());
//    bolt::cl::counting_iterator<int> boltReduce = bolt::cl::min_element(first, last);
//
//    EXPECT_EQ(*stlReduce, *boltReduce);
//}


TEST_P( MinEStdVectandCountingIterator, comp_withCountingIterator)
{
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end(), std::less< int >());
    bolt::cl::counting_iterator<int> boltReduce = bolt::cl::min_element(first, last, bolt::cl::less< int >( ));

    EXPECT_EQ(*stlReduce, *boltReduce);
}

TEST( MinEleDevice , DeviceVectoroffset )
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

    bolt::cl::device_vector< int >::iterator  boltReduce =  bolt::cl::min_element( input.begin()+20, input.end());
    int stdReduce =  20;

    EXPECT_EQ(*boltReduce,stdReduce);

}


TEST( MinEleDevice , DeviceVectoroffsetSerial )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > stdinput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      stdinput[i] = rand();

    }
    
	bolt::cl::device_vector< int > input( stdinput.begin(), stdinput.end() );
	
	bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call reduce

    bolt::cl::device_vector< int >::iterator  boltReduce =  bolt::cl::min_element(ctl, input.begin()+20, input.end());
    std::vector<int>::iterator stdReduce =  min_element(stdinput.begin()+20, stdinput.end());;

    EXPECT_EQ(*boltReduce, *stdReduce);

}

TEST( MinEleDevice , DeviceVectoroffsetMultiCore )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > stdinput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      stdinput[i] = rand();

    }
    
	bolt::cl::device_vector< int > input( stdinput.begin(), stdinput.end() );
	
	bolt::cl::control ctl = bolt::cl::control::getDefault( );
	ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call reduce

    bolt::cl::device_vector< int >::iterator  boltReduce =  bolt::cl::min_element(ctl, input.begin()+20, input.end());
    std::vector<int>::iterator stdReduce =  min_element(stdinput.begin()+20, stdinput.end());;

    EXPECT_EQ(*boltReduce, *stdReduce);

}

//TEST_P( MinEStdVectandCountingIterator, CPUwithCountingIterator)
//{
//    bolt::cl::counting_iterator<int> first(0);
//    bolt::cl::counting_iterator<int> last = first +  mySize;
//
//    std::vector<int> a(mySize);
//
//    for (int i=0; i < mySize; i++) {
//        a[i] = i;
//    };
//
//    bolt::cl::control ctl = bolt::cl::control::getDefault( );
//    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
//
//    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end());
//    bolt::cl::counting_iterator<int> boltReduce = bolt::cl::min_element(ctl, first, last);
//
//    EXPECT_EQ(*stlReduce, *boltReduce);
//}

TEST_P( MinEStdVectandCountingIterator, CPU_compwithCountingIterator)
{
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end(), std::less< int >());
    bolt::cl::counting_iterator<int> boltReduce = bolt::cl::min_element(ctl, first, last,  bolt::cl::less< int >( ));

    EXPECT_EQ(*stlReduce, *boltReduce);
}

//TEST_P( MinEStdVectandCountingIterator, MultiCorewithCountingIterator)
//{
//    bolt::cl::counting_iterator<int> first(0);
//    bolt::cl::counting_iterator<int> last = first +  mySize;
//
//    std::vector<int> a(mySize);
//
//    for (int i=0; i < mySize; i++) {
//        a[i] = i;
//    };
//
//    bolt::cl::control ctl = bolt::cl::control::getDefault( );
//    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
//
//    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end());
//    bolt::cl::counting_iterator<int> boltReduce = bolt::cl::min_element(ctl, first, last);
//
//    EXPECT_EQ(*stlReduce, *boltReduce);
//}


TEST_P( MinEStdVectandCountingIterator, MultiCore_comp_withCountingIterator)
{
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };


    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end(), std::less< int >());
    bolt::cl::counting_iterator<int> boltReduce = bolt::cl::min_element(ctl, first, last,  bolt::cl::less< int >( ) );

    EXPECT_EQ(*stlReduce, *boltReduce);
}


INSTANTIATE_TEST_CASE_P( withInt, MinEStdVectWithInit, ::testing::Range(1, 100, 1) );
INSTANTIATE_TEST_CASE_P( withInt, MinEStdVectandCountingIterator, ::testing::Range(1, 100, 1) );

TEST( MinEleDevice , DeviceVectorUintoffset )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< unsigned int > stdinput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      stdinput[i] = i;

    }
    bolt::cl::device_vector< unsigned int > input( stdinput.begin(), stdinput.end() );
	
    // call reduce

    bolt::cl::device_vector< unsigned int >::iterator  boltReduce =  bolt::cl::min_element( input.begin()+20, input.end());
    int stdReduce =  20;

    EXPECT_EQ(*boltReduce,stdReduce);

}

class MinETestFloat: public ::testing::TestWithParam<int>{
protected:
    int arraySize;
public:
    MinETestFloat( ):arraySize( GetParam( ) )
    {}
};

TEST_P( MinETestFloat, FloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> boltVect( arraySize );

    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }
    bolt::cl::control ctl;

    std::vector<float>::iterator stdMinEValue = std::min_element(A.begin(), A.end() );
    std::vector<float>::iterator boltClMinE = bolt::cl::min_element(ctl.getDefault(), boltVect.begin(),boltVect.end());

    //compare these results with each other
    EXPECT_EQ( *stdMinEValue, *boltClMinE );
}

TEST_P( MinETestFloat, serialFloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> boltVect( arraySize );

    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    std::vector<float>::iterator stdMinEValue = std::min_element(A.begin(), A.end() );
    std::vector<float>::iterator boltClMinE = bolt::cl::min_element(ctl, boltVect.begin(), boltVect.end() );

    //compare these results with each other
    EXPECT_EQ( *stdMinEValue, *boltClMinE );
}

TEST_P( MinETestFloat, MultiCoreFloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> boltVect( arraySize );

    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    std::vector<float>::iterator stdMinEValue = std::min_element(A.begin(), A.end() );
    std::vector<float>::iterator boltClMinE = bolt::cl::min_element(ctl, boltVect.begin(), boltVect.end() );

    //compare these results with each other
    EXPECT_EQ( *stdMinEValue, *boltClMinE );
}

INSTANTIATE_TEST_CASE_P(serialFloatValuesWdControl, MinETestFloat, ::testing::Range(1, 100, 10));

#if (TEST_DOUBLE == 1)
class MinETestDouble: public ::testing::TestWithParam<int>{
protected:
        int arraySize;
public:
    MinETestDouble():arraySize(GetParam()){
    }
};

TEST_P (MinETestDouble, WithDouble)
{
    std::vector<double> A( arraySize );
    std::vector<double> boltVect( arraySize );

    double myFloatValues = 9.0625;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + double(i);
        boltVect[i] = A[i];
    }
    bolt::cl::control ctl;

    std::vector<double>::iterator stdMinEValue = std::min_element(A.begin(), A.end() );
    std::vector<double>::iterator boltClMinE=bolt::cl::min_element(ctl.getDefault(),boltVect.begin(),boltVect.end());

    //compare these results with each other
    EXPECT_EQ( *stdMinEValue, *boltClMinE );
}

TEST_P (MinETestDouble, CPUWithDouble)
{
    std::vector<double> A( arraySize );
    std::vector<double> boltVect( arraySize );

    double myFloatValues = 9.0625;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + double(i);
        boltVect[i] = A[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    std::vector<double>::iterator stdMinEValue = std::min_element(A.begin(), A.end() );
    std::vector<double>::iterator boltClMinE = bolt::cl::min_element(ctl, boltVect.begin(), boltVect.end() );

    //compare these results with each other
    EXPECT_EQ( *stdMinEValue, *boltClMinE );
}

TEST_P (MinETestDouble, MultiCoreWithDouble)
{
    std::vector<double> A( arraySize );
    std::vector<double> boltVect( arraySize );

    double myFloatValues = 9.0625;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + double(i);
        boltVect[i] = A[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    std::vector<double>::iterator stdMinEValue = std::min_element(A.begin(), A.end() );
    std::vector<double>::iterator boltClMinE = bolt::cl::min_element(ctl, boltVect.begin(), boltVect.end() );

    //compare these results with each other
    EXPECT_EQ( *stdMinEValue, *boltClMinE );
}


INSTANTIATE_TEST_CASE_P( WithDouble, MinETestDouble, ::testing::Range(1, 20, 1) );

#endif

TEST_P( MinEIntegerVector, Normal )
{

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMinE = std::min_element( stdInput.begin(), stdInput.end() );
    std::vector<int>::iterator boltMinE = bolt::cl::min_element( boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMinE == *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( MinEIntegerVector,CPU )
{

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMinE = std::min_element( stdInput.begin(), stdInput.end() );
    std::vector<int>::iterator boltMinE = bolt::cl::min_element( ctl, boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMinE == *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( MinEIntegerVector,MultiCore )
{

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMinE = std::min_element( stdInput.begin(), stdInput.end() );
    std::vector<int>::iterator boltMinE = bolt::cl::min_element( ctl, boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMinE == *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( MinEFloatVector, Normal )
{
    //  Calling the actual functions under test
    std::vector<float>::iterator stlMinE = std::min_element( stdInput.begin(), stdInput.end() );
    std::vector<float>::iterator boltMinE = bolt::cl::min_element( boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMinE == *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( MinEFloatVector, CPU )
{

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::vector<float>::iterator stlMinE = std::min_element( stdInput.begin(), stdInput.end() );
    std::vector<float>::iterator boltMinE = bolt::cl::min_element( ctl, boltInput.begin( ), boltInput.end( ));

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMinE == *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( MinEFloatVector, MultiCore )
{

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::vector<float>::iterator stlMinE = std::min_element( stdInput.begin(), stdInput.end() );
    std::vector<float>::iterator boltMinE = bolt::cl::min_element( ctl, boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMinE == *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

std::array<int, 16> TestValues = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768, 65535};
//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( MinERange, MinEIntegerVector, ::testing::Range( 1, 65535, 177 ) );
INSTANTIATE_TEST_CASE_P( MinEValues, MinEIntegerVector, ::testing::ValuesIn( TestValues.begin(), TestValues.end()));
INSTANTIATE_TEST_CASE_P( MinERange, MinEFloatVector, ::testing::Range( 1, 65535, 133 ) );
INSTANTIATE_TEST_CASE_P( MinEValues, MinEFloatVector, ::testing::ValuesIn(TestValues.begin(),TestValues.end()));
INSTANTIATE_TEST_CASE_P( MinERange,MinEIntegerDeviceVector,::testing::Range(0,65535,153));
INSTANTIATE_TEST_CASE_P( MinEValues,MinEIntegerDeviceVector,::testing::ValuesIn(TestValues.begin(),TestValues.end()));
INSTANTIATE_TEST_CASE_P( MinERange, MinEFloatDeviceVector, ::testing::Range( 0, 65535, 153 ) );
INSTANTIATE_TEST_CASE_P( MinEValues, MinEFloatDeviceVector, ::testing::ValuesIn(TestValues.begin(),TestValues.end()));

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

    UDD()
        : a(0),b(0) { }
    UDD(int _in)
        : a(_in), b(_in +1)  { }
};
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< UDD >::iterator );
//Do this to register new type: It does two things -> BOLT_CREATE_TYPENAME and BOLT_CREATE_CLCODE
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, int, UDD );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< UDD >::iterator, bolt::cl::deviceVectorIteratorTemplate );

TEST( MinEUDD , UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    // call reduce
    UDD *boltMinE =bolt::cl::min_element( refInput, refInput + length );
    UDD *stdMinE = std::min_element( refInput, refInput + length );

    EXPECT_EQ(*boltMinE,*stdMinE);

}

TEST( MinEUDD , CPU_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    // call reduce
    UDD *boltMinE =bolt::cl::min_element( ctl, refInput, refInput + length );
    UDD *stdMinE = std::min_element( refInput, refInput + length );

    EXPECT_EQ(*boltMinE,*stdMinE);

}

TEST( MinEUDD , MultiCore_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    // call reduce
    UDD *boltMinE =bolt::cl::min_element( ctl, refInput, refInput + length );
    UDD *stdMinE = std::min_element( refInput, refInput + length );

    EXPECT_EQ(*boltMinE,*stdMinE);

}

TEST( MinEUDD , Offset_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    // call reduce
    UDD *boltMinE =bolt::cl::min_element( refInput + 10, refInput + length );
    UDD *stdMinE = std::min_element( refInput + 10, refInput + length );

    EXPECT_EQ(*boltMinE,*stdMinE);

}

TEST( MinEUDD , Offset_CPU_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    // call reduce
    UDD *boltMinE =bolt::cl::min_element( ctl, refInput + 5, refInput + length );
    UDD *stdMinE = std::min_element( refInput + 5, refInput + length );

    EXPECT_EQ(*boltMinE,*stdMinE);

}

TEST( MinEUDD , Offset_MultiCore_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    // call reduce
    UDD *boltMinE =bolt::cl::min_element( ctl, refInput+10, refInput + length );
    UDD *stdMinE = std::min_element( refInput+10, refInput + length );

    EXPECT_EQ(*boltMinE,*stdMinE);

}

INSTANTIATE_TYPED_TEST_CASE_P( Integer, MinEArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, MinEArrayTest, FloatTests );


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


template<typename T>
void printCheckMessage(bool err, std::string msg, T  stlResult, T boltResult)
{
    if (err) {
        std::cout << "*ERROR ";
    } else {
        std::cout << "PASSED ";
    }

  //  std::cout << msg << "  STL=" << stlResult << " BOLT=" << boltResult << std::endl;
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



// Simple test case for bolt::min_element:
// Sum together specified numbers, compare against STL::accumulate function.
// Demonstrates:
//    * use of bolt with STL::vector iterators
//    * use of bolt with default plus
//    * use of bolt with explicit plus argument
void Mineletest(int aSize)
{
    std::vector<int> A(aSize);
    //srand(GetTickCount());
    for (int i=0; i < aSize; i++)
    {
                A[i] = rand();
    };

    std::vector<int>::iterator stlReduce = std::max_element(A.begin(), A.end());
    std::vector<int>::iterator boltReduce = bolt::cl::min_element(A.begin(), A.end(),bolt::cl::greater<int>());



    checkResult("Minelement", *stlReduce, *boltReduce);
    //printf ("Sum: stl=%d,  bolt=%d %d\n", stlReduce, boltReduce, boltReduce2);
};


// Demonstrates use of bolt::control structure to control execution of routine.
void Minele_TestControl(int aSize, int numIters, int deviceIndex)
{
    std::vector<int> A(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = i;
    };

    //  Tests need to be a bit more sophisticated before hardcoding which device to use in a system
    //  (my default is device 1);
    //  this should be configurable on the command line
    //  Create an OCL context, device, queue.


  // FIXME - temporarily disable use of new control queue here:
#if OCL_CONTEXT_BUG_WORKAROUND
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control c( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));
#else
  MyOclContext ocl = initOcl(CL_DEVICE_TYPE_GPU, deviceIndex);
    bolt::cl::control c(ocl._queue);  // construct control structure from the queue.
#endif
    //c.debug(bolt::cl::control::debug::Compile + bolt::cl::control::debug::SaveCompilerTemps);

    std::vector<int>::iterator  stlReduce = std::min_element(A.begin(), A.end());
    std::vector<int>::iterator boltReduce(A.end());

    char testTag[2000];

#if defined(_WIN32)
    sprintf_s(testTag, 2000, "Minele_TestControl sz=%d iters=%d, device=%s", aSize, numIters,
        c.getDevice( ).getInfo<CL_DEVICE_NAME>( ).c_str( ) );
#else

    sprintf(testTag, "Minele_TestControl sz=%d iters=%d, device=%s", aSize, numIters,
        c.getDevice( ).getInfo<CL_DEVICE_NAME>( ).c_str( ) );

#endif

#if defined(_WIN32)
    __int64 start = StartProfile();
#endif
    for (int i=0; i<numIters; i++) {
        boltReduce = bolt::cl::min_element( c, A.begin(), A.end());
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


    bolt::cl::control c;  // construct control structure from the queue.
    c.setForceRunMode(bolt::cl::control::SerialCpu);

    std::vector<int>::iterator stlReduce = std::min_element(A.begin(), A.end());
    std::vector<int>::iterator boltReduce = A.end();

    boltReduce = bolt::cl::min_element(c, A.begin(), A.end());


    checkResult("TestSerial", stlReduce, boltReduce);
};



void simpleMinele_countingiterator(int start,int size)
{

    bolt::cl::counting_iterator<int> first(start);
    bolt::cl::counting_iterator<int> last = first +  size;

    std::vector<int> a(size);

    for (int i=0; i < size; i++) {
        a[i] = i+start;
    };

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end());
    bolt::cl::counting_iterator<int> boltReduce = bolt::cl::min_element(first, last);



    checkResult("TestSerial", *stlReduce, *boltReduce);
};

//TEST( Min_Element , KcacheTest )
//{
//    //setup containers
//    unsigned int length = 1024;
//    std::vector< int > refInput( length );
//    for( unsigned int i = 0; i < length ; i++ )
//    {
//      refInput[i] = rand();
//    }
//
//    //Call reduce with GPU device because the default is a GPU device
//    bolt::cl::control ctrl = bolt::cl::control::getDefault();
//    bolt::cl::device_vector< int > gpuInput( refInput.begin(), refInput.end() );
//
//    bolt::cl::device_vector< int >::iterator boltGpuMin = bolt::cl::min_element(ctrl,gpuInput.begin(),gpuInput.end());
//
//    //Call reduce with CPU device
//    ::cl::Context cpuContext(CL_DEVICE_TYPE_CPU);
//    std::vector< cl::Device > devices = cpuContext.getInfo< CL_CONTEXT_DEVICES >();
//    cl::Device selectedDevice;
//
//    for(std::vector< cl::Device >::iterator iter = devices.begin();iter < devices.end(); iter++)
//    {
//        if(iter->getInfo<CL_DEVICE_TYPE> ( ) == CL_DEVICE_TYPE_CPU)
//        {
//            selectedDevice = *iter;
//            break;
//        }
//    }
//    ::cl::CommandQueue myQueue( cpuContext, selectedDevice );
//    bolt::cl::control cpu_ctrl( myQueue );  // construct control structure from the queue.
//    bolt::cl::device_vector< int > cpuInput( refInput.begin(), refInput.end() );
//    bolt::cl::device_vector<int>::iterator boltCpuMin=bolt::cl::min_element(cpu_ctrl,cpuInput.begin(),cpuInput.end());
//
//    std::vector< int >::iterator stdCpuMin;
//    //Call reference code
//    stdCpuMin =  std::min_element( refInput.begin(), refInput.end());
//
//
//    EXPECT_EQ(*boltGpuMin,*stdCpuMin);
//    EXPECT_EQ(*boltCpuMin,*stdCpuMin);
//}

//TEST( Min_Element , DISABLED_CPU_KcacheTest )
//{
//    //setup containers
//    unsigned int length = 1024;
//    std::vector< int > refInput( length );
//    for( unsigned int i = 0; i < length ; i++ )
//    {
//      refInput[i] = rand();
//    }
//
//    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
//    bolt::cl::control ctl = bolt::cl::control::getDefault( );
//    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
//
//    bolt::cl::device_vector< int > gpuInput( refInput.begin(), refInput.end() );
//    bolt::cl::device_vector< int >::iterator boltGpuMin=bolt::cl::min_element(ctl,gpuInput.begin(),gpuInput.end());
//
//    //Call reduce with CPU device
//    ::cl::Context cpuContext(CL_DEVICE_TYPE_CPU);
//    std::vector< cl::Device > devices = cpuContext.getInfo< CL_CONTEXT_DEVICES >();
//    cl::Device selectedDevice;
//
//    for(std::vector< cl::Device >::iterator iter = devices.begin();iter < devices.end(); iter++)
//    {
//        if(iter->getInfo<CL_DEVICE_TYPE> ( ) == CL_DEVICE_TYPE_CPU)
//        {
//            selectedDevice = *iter;
//            break;
//        }
//    }
//    ::cl::CommandQueue myQueue( cpuContext, selectedDevice );
//    bolt::cl::control cpu_ctrl( myQueue );  // construct control structure from the queue.
//    bolt::cl::device_vector< int > cpuInput( refInput.begin(), refInput.end() );
//    bolt::cl::device_vector<int>::iterator boltCpuMin=bolt::cl::min_element(cpu_ctrl,cpuInput.begin(),cpuInput.end());
//
//    std::vector< int >::iterator stdCpuMin;
//    //Call reference code
//    stdCpuMin =  std::min_element( refInput.begin(), refInput.end());
//
//
//    EXPECT_EQ(*boltGpuMin,*stdCpuMin);
//    EXPECT_EQ(*boltCpuMin,*stdCpuMin);
//}
//
//TEST( Min_Element , DISABLED_MultiCore_KcacheTest )
//{
//    //setup containers
//    unsigned int length = 1024;
//    std::vector< int > refInput( length );
//    for( unsigned int i = 0; i < length ; i++ )
//    {
//      refInput[i] = rand();
//    }
//
//    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
//    bolt::cl::control ctl = bolt::cl::control::getDefault( );
//    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
//
//    bolt::cl::device_vector< int > gpuInput( refInput.begin(), refInput.end() );
//    bolt::cl::device_vector< int >::iterator   boltGpuMin = bolt::cl::min_element(ctl,gpuInput.begin(),gpuInput.end());
//
//
//    //Call reduce with CPU device
//    ::cl::Context cpuContext(CL_DEVICE_TYPE_CPU);
//    std::vector< cl::Device > devices = cpuContext.getInfo< CL_CONTEXT_DEVICES >();
//    cl::Device selectedDevice;
//
//    for(std::vector< cl::Device >::iterator iter = devices.begin();iter < devices.end(); iter++)
//    {
//        if(iter->getInfo<CL_DEVICE_TYPE> ( ) == CL_DEVICE_TYPE_CPU)
//        {
//            selectedDevice = *iter;
//            break;
//        }
//    }
//    ::cl::CommandQueue myQueue( cpuContext, selectedDevice );
//    bolt::cl::control cpu_ctrl( myQueue );  // construct control structure from the queue.
//
//    bolt::cl::device_vector< int > cpuInput( refInput.begin(), refInput.end() );
//    bolt::cl::device_vector<int>::iterator boltCpuMin=bolt::cl::min_element(cpu_ctrl,cpuInput.begin(),cpuInput.end());
//
//
//    //Call reference code
//    std::vector< int >::iterator stdCpuMin =  std::min_element( refInput.begin(), refInput.end());
//
//    EXPECT_EQ(*boltGpuMin,*stdCpuMin);
//    EXPECT_EQ(*boltCpuMin,*stdCpuMin);
//}


int _tmain(int argc, _TCHAR* argv[])
{
    int numIters = 100;
#if 0
    //NON_GTEST
    testDeviceVector();
    Minele_TestControl(1024000, numIters, 0);
    Minele_TestControl(100, 1, 0);
    Mineletest(256);
    Mineletest(1024);
    Minele_TestControl(100, 1, 0);
    simpleReduce_TestSerial(1000);
    simpleMinele_countingiterator(20,10);
#endif
    //GTEST
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

