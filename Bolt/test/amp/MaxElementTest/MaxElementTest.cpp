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

#define TEST_DOUBLE 1
#define TEST_LARGE_BUFFERS 1

#include <iostream>
#include <algorithm>  // for testing against STL functions.
#include <numeric>
#include <gtest/gtest.h>
#include <type_traits>

#include "bolt/amp/max_element.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/iterator/counting_iterator.h"

#include "stdafx.h"
#include "common/test_common.h"
#include "bolt/miniDump.h"

#include <array>
#include <algorithm>

void testDeviceVector()
{
    const int aSize = 1000;
    std::vector<int> hA(aSize);  

    for(int i=0; i<aSize; i++) {
        hA[i] = i;
    };
    
    bolt::amp::device_vector<int> dA(hA.begin(), hA.end());
    std::vector<int>::iterator smindex = std::min_element(hA.begin(), hA.end());
    bolt::amp::device_vector<int>::iterator bmindex = bolt::amp::min_element(dA.begin(), dA.end());
    
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
class MaxEArrayTest: public ::testing::Test
{
public:
    MaxEArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<ArrayType>);
        boltInput = stdInput;
    };

    virtual void TearDown( )
    {};

    virtual ~MaxEArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType, ArraySize > stdInput, boltInput;
    int m_Errors;
};

TYPED_TEST_CASE_P( MaxEArrayTest );

/*TYPED_TEST_P( MinEArrayTest, Normal )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
   typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::iterator boltMinE = bolt::amp::min_element( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
}*/

TYPED_TEST_P( MaxEArrayTest, comp_Normal )
{
       typedef typename MaxEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MaxEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMaxE = std::max_element( MaxEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MaxEArrayTest< gtest_TypeParam_ >::stdInput.end(), std::less< ArrayType >() );
   typename  ArrayCont::iterator boltMaxE = bolt::amp::max_element( MaxEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::amp::less< ArrayType >() );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MaxEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MaxEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMaxE, *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MaxEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MaxEArrayTest< gtest_TypeParam_ >::stdInput, MaxEArrayTest< gtest_TypeParam_ >::boltInput );
}


/*TYPED_TEST_P( MinEArrayTest, SerialNormal )
{
       typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::iterator boltMinE = bolt::amp::min_element( ctl, MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
}*/

TYPED_TEST_P( MaxEArrayTest, comp_SerialNormal )
{
    typedef typename MaxEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MaxEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMaxE = std::max_element( MaxEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MaxEArrayTest< gtest_TypeParam_ >::stdInput.end(), std::less< ArrayType >() );
   typename  ArrayCont::iterator boltMaxE = bolt::amp::max_element( ctl, MaxEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::amp::less< ArrayType >()  );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MaxEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MaxEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMaxE, *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MaxEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MaxEArrayTest< gtest_TypeParam_ >::stdInput, MaxEArrayTest< gtest_TypeParam_ >::boltInput );
}

#if defined( ENABLE_TBB )
TYPED_TEST_P( MaxEArrayTest, MultiCoreNormal )
{
    typedef typename MaxEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MaxEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMaxE = std::max_element( MaxEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MaxEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::iterator boltMaxE = bolt::amp::max_element( ctl, MaxEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MaxEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MaxEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMaxE, *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MaxEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MaxEArrayTest< gtest_TypeParam_ >::stdInput, MaxEArrayTest< gtest_TypeParam_ >::boltInput );
}
#endif

#if defined( ENABLE_TBB )
TYPED_TEST_P( MaxEArrayTest, comp_MultiCoreNormal )
{
    typedef typename MaxEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MaxEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMaxE = std::max_element( MaxEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MaxEArrayTest< gtest_TypeParam_ >::stdInput.end(), std::less< ArrayType >() );
   typename  ArrayCont::iterator boltMaxE = bolt::amp::max_element( ctl, MaxEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::amp::less< ArrayType >() );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MaxEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MaxEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MaxEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMaxE, *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MaxEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MaxEArrayTest< gtest_TypeParam_ >::stdInput, MaxEArrayTest< gtest_TypeParam_ >::boltInput );
}
#endif
/* TYPED_TEST_P( MinEArrayTest, GPU_DeviceGreaterFunction )
{
    typedef typename MinEArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    bolt::amp::Context myContext = bolt::amp::control::getDefault( ).getContext( );
    bolt::amp::control c_gpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));


    //  Calling the actual functions under test
   typename  ArrayCont::iterator stlMinE = std::min_element( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin(), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::iterator boltMinE = bolt::amp::min_element( c_gpu,MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

   typename  ArrayCont::difference_type stdNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::stdInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::stdInput.end() );
   typename  ArrayCont::difference_type boltNumElements = std::distance( MinEArrayTest< gtest_TypeParam_ >::boltInput.begin( ), MinEArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( *stlMinE, *boltMinE );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, MinEArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( MinEArrayTest< gtest_TypeParam_ >::stdInput, MinEArrayTest< gtest_TypeParam_ >::boltInput );
    // FIXME - releaseOcl(ocl);
} */

//REGISTER_TYPED_TEST_CASE_P( MinEArrayTest, Normal, comp_Normal, SerialNormal, comp_SerialNormal, 
//MultiCoreNormal, comp_MultiCoreNormal, GPU_DeviceGreaterFunction );
REGISTER_TYPED_TEST_CASE_P( MaxEArrayTest, comp_Normal, comp_SerialNormal );
#if defined( ENABLE_TBB )
REGISTER_TYPED_TEST_CASE_P( comp_MultiCoreNormal, MultiCoreNormal );
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class MaxEIntegerVector: public ::testing::TestWithParam< int >
{
public:

    MaxEIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        boltInput = stdInput;
    }

protected:
    std::vector< int > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class MaxEFloatVector: public ::testing::TestWithParam< int >
{
public:
    MaxEFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        boltInput = stdInput;
    }

protected:
    std::vector< float > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class MaxEIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    MaxEIntegerDeviceVector( ): stdInput( GetParam( ) )
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
class MaxEFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    MaxEFloatDeviceVector( ): stdInput( GetParam( ) )
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


class MaxEStdVectWithInit :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    MaxEStdVectWithInit():mySize(GetParam()){
    }
};


class MinEStdVectandCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    MinEStdVectandCountingIterator():mySize(GetParam()){
    }
};

TEST_P( MaxEStdVectWithInit, withInt)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMaxE = std::max_element( stdInput.begin( ), stdInput.end() );
    std::vector<int>::iterator boltMaxE= bolt::amp::max_element( boltInput.begin( ), boltInput.end() );

    EXPECT_EQ( *stlMaxE, *boltMaxE );
}

TEST_P( MaxEStdVectWithInit, CPUwithInt)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMaxE = std::max_element( stdInput.begin( ), stdInput.end() );
    std::vector<int>::iterator boltMaxE= bolt::amp::max_element(ctl, boltInput.begin( ), boltInput.end() );

    EXPECT_EQ( *stlMaxE, *boltMaxE );
}

#if defined( ENABLE_TBB )
TEST_P( MaxEStdVectWithInit, MultiCorewithInt)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMaxE = std::max_element( stdInput.begin( ), stdInput.end() );
    std::vector<int>::iterator boltMaxE= bolt::amp::max_element(ctl, boltInput.begin( ), boltInput.end() );

    EXPECT_EQ( *stlMaxE, *boltMaxE );
}
#endif

TEST_P( MinEStdVectandCountingIterator, withCountingIterator)
{
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end());
    bolt::amp::counting_iterator<int> boltReduce = bolt::amp::min_element(first, last);

    EXPECT_EQ(*stlReduce, *boltReduce);
}


 TEST_P( MinEStdVectandCountingIterator, comp_withCountingIterator)
{
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end(), std::less< int >());
    bolt::amp::counting_iterator<int> boltReduce = bolt::amp::min_element(first, last, bolt::amp::less< int >( ));

    EXPECT_EQ(*stlReduce, *boltReduce);

} 

TEST( MaxEleDevice , DeviceVectoroffset )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > stdinput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      stdinput[i] = length - i;

    }
    
    bolt::amp::device_vector< int > input( stdinput.begin(), stdinput.end() );
    
    // call reduce

    bolt::amp::device_vector< int >::iterator  boltReduce =  bolt::amp::max_element( input.begin()+20, input.end());
    int stdReduce =  1004;

    EXPECT_EQ(*boltReduce,stdReduce);

}

TEST( MaxEleDevice , SerialDeviceVectoroffset )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > stdinput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      stdinput[i] = length - i;

    }
    bolt::amp::device_vector< int > input( stdinput.begin(), stdinput.end() );
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    bolt::amp::device_vector< int >::iterator  boltReduce =  bolt::amp::max_element( ctl, input.begin()+20, input.end());
    int stdReduce =  1004;

    EXPECT_EQ(*boltReduce,stdReduce);

}

#if defined( ENABLE_TBB )
TEST( MaxEleDevice , MultiCoreDeviceVectoroffset )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< int > stdinput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      stdinput[i] = length - i;

    }
    
    bolt::amp::device_vector< int > input( stdinput.begin(), stdinput.end() );
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    bolt::amp::device_vector< int >::iterator  boltReduce =  bolt::amp::max_element( ctl, input.begin()+20, input.end());
    int stdReduce =  1004;

    EXPECT_EQ(*boltReduce,stdReduce);

}
#endif

TEST_P( MinEStdVectandCountingIterator, CPUwithCountingIterator)
{
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end());
    bolt::amp::counting_iterator<int> boltReduce = bolt::amp::min_element(ctl, first, last);

    EXPECT_EQ(*stlReduce, *boltReduce);
}

 TEST_P( MinEStdVectandCountingIterator, CPU_compwithCountingIterator)
{
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end(), std::less< int >());
    bolt::amp::counting_iterator<int> boltReduce = bolt::amp::min_element(ctl, first, last,  bolt::amp::less< int >( ));

    EXPECT_EQ(*stlReduce, *boltReduce);
} 

#if defined( ENABLE_TBB )
TEST_P( MinEStdVectandCountingIterator, MultiCorewithCountingIterator)
{
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end());
    bolt::amp::counting_iterator<int> boltReduce = bolt::amp::min_element(ctl, first, last);

    EXPECT_EQ(*stlReduce, *boltReduce);
}
#endif

#if defined( ENABLE_TBB )
 TEST_P( MinEStdVectandCountingIterator, MultiCore_comp_withCountingIterator)
{
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };


    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    std::vector<int>::iterator stlReduce = std::min_element(a.begin(), a.end(), std::less< int >());
    bolt::amp::counting_iterator<int> boltReduce = bolt::amp::min_element(ctl, first, last,  bolt::amp::less< int >( ) );

    EXPECT_EQ(*stlReduce, *boltReduce);
} 
#endif

INSTANTIATE_TEST_CASE_P( withInt, MaxEStdVectWithInit, ::testing::Range(1, 100, 1) );
INSTANTIATE_TEST_CASE_P( withInt, MinEStdVectandCountingIterator, ::testing::Range(1, 100, 1) );

TEST( MaxEleDevice , DeviceVectorUintoffset )
{
    //setup containers
    unsigned int length = 1024;
    std::vector< unsigned int > stdinput( length );
    for( unsigned int i = 0; i < length ; i++ )
    {
      stdinput[i] = length - i;

    }
    bolt::amp::device_vector< unsigned int > input( stdinput.begin(), stdinput.end() );
    
    // call reduce

    bolt::amp::device_vector< unsigned int >::iterator  boltReduce =  bolt::amp::max_element( input.begin()+20, input.end());
    int stdReduce =  1004;

    EXPECT_EQ(*boltReduce,stdReduce);

}

class MaxETestFloat: public ::testing::TestWithParam<int>{
protected:
    int arraySize;
public:
    MaxETestFloat( ):arraySize( GetParam( ) )
    {}
};

TEST_P( MaxETestFloat, FloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> boltVect( arraySize );

    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }
    bolt::amp::control ctl;

    std::vector<float>::iterator stdMaxEValue = std::max_element(A.begin(), A.end() );
    std::vector<float>::iterator boltClMaxE = bolt::amp::max_element(ctl.getDefault(), boltVect.begin(),boltVect.end());

    //compare these results with each other
    EXPECT_EQ( *stdMaxEValue, *boltClMaxE );
}

TEST_P( MaxETestFloat, serialFloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> boltVect( arraySize );

    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    std::vector<float>::iterator stdMaxEValue = std::max_element(A.begin(), A.end() );
    std::vector<float>::iterator boltClMaxE = bolt::amp::max_element(ctl, boltVect.begin(), boltVect.end() );

    //compare these results with each other
    EXPECT_EQ( *stdMaxEValue, *boltClMaxE );
}

#if defined( ENABLE_TBB )
TEST_P( MaxETestFloat, MultiCoreFloatValuesWdControl )
{
    std::vector<float> A( arraySize );
    std::vector<float> boltVect( arraySize );

    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    std::vector<float>::iterator stdMaxEValue = std::max_element(A.begin(), A.end() );
    std::vector<float>::iterator boltClMaxE = bolt::amp::max_element(ctl, boltVect.begin(), boltVect.end() );

    //compare these results with each other
    EXPECT_EQ( *stdMaxEValue, *boltClMaxE );
}
#endif

INSTANTIATE_TEST_CASE_P(serialFloatValuesWdControl, MaxETestFloat, ::testing::Range(1, 100, 10));

#if (TEST_DOUBLE == 1)
class MaxETestDouble: public ::testing::TestWithParam<int>{
protected:
        int arraySize;
public:
    MaxETestDouble():arraySize(GetParam()){
    }
};

TEST_P (MaxETestDouble, WithDouble)
{
    std::vector<double> A( arraySize );
    std::vector<double> boltVect( arraySize );

    double myFloatValues = 9.0625;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + double(i);
        boltVect[i] = A[i];
    }
    bolt::amp::control ctl;

    std::vector<double>::iterator stdMaxEValue = std::max_element(A.begin(), A.end() );
    std::vector<double>::iterator boltClMaxE=bolt::amp::max_element(ctl.getDefault(),boltVect.begin(),boltVect.end());

    //compare these results with each other
    EXPECT_EQ( *stdMaxEValue, *boltClMaxE );
}

TEST_P (MaxETestDouble, CPUWithDouble)
{
    std::vector<double> A( arraySize );
    std::vector<double> boltVect( arraySize );

    double myFloatValues = 9.0625;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + double(i);
        boltVect[i] = A[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    std::vector<double>::iterator stdMaxEValue = std::max_element(A.begin(), A.end() );
    std::vector<double>::iterator boltClMaxE = bolt::amp::max_element(ctl, boltVect.begin(), boltVect.end() );

    //compare these results with each other
    EXPECT_EQ( *stdMaxEValue, *boltClMaxE );
}

#if defined( ENABLE_TBB )
TEST_P (MaxETestDouble, MultiCoreWithDouble)
{
    std::vector<double> A( arraySize );
    std::vector<double> boltVect( arraySize );

    double myFloatValues = 9.0625;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + double(i);
        boltVect[i] = A[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    std::vector<double>::iterator stdMaxEValue = std::max_element(A.begin(), A.end() );
    std::vector<double>::iterator boltClMaxE = bolt::amp::max_element(ctl, boltVect.begin(), boltVect.end() );

    //compare these results with each other
    EXPECT_EQ( *stdMaxEValue, *boltClMaxE );
}
#endif

INSTANTIATE_TEST_CASE_P( WithDouble, MaxETestDouble, ::testing::Range(1, 20, 1) );

#endif

TEST_P( MaxEIntegerVector, Normal )
{

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMaxE = std::max_element( stdInput.begin(), stdInput.end() );
    std::vector<int>::iterator boltMaxE = bolt::amp::max_element( boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMaxE == *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( MaxEIntegerVector,CPU )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMaxE = std::max_element( stdInput.begin(), stdInput.end() );
    std::vector<int>::iterator boltMaxE = bolt::amp::max_element( ctl, boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMaxE == *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

#if defined( ENABLE_TBB )
TEST_P( MaxEIntegerVector,MultiCore )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::vector<int>::iterator stlMaxE = std::max_element( stdInput.begin(), stdInput.end() );
    std::vector<int>::iterator boltMaxE = bolt::amp::max_element( ctl, boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMaxE == *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}
#endif

TEST_P( MaxEFloatVector, Normal )
{
    //  Calling the actual functions under test
    std::vector<float>::iterator stlMaxE = std::max_element( stdInput.begin(), stdInput.end() );
    std::vector<float>::iterator boltMaxE = bolt::amp::max_element( boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMaxE == *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( MaxEFloatVector, CPU )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::vector<float>::iterator stlMaxE = std::max_element( stdInput.begin(), stdInput.end() );
    std::vector<float>::iterator boltMaxE = bolt::amp::max_element( ctl, boltInput.begin( ), boltInput.end( ));

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMaxE == *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

#if defined( ENABLE_TBB )
TEST_P( MaxEFloatVector, MultiCore )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::vector<float>::iterator stlMaxE = std::max_element( stdInput.begin(), stdInput.end() );
    std::vector<float>::iterator boltMaxE = bolt::amp::max_element( ctl, boltInput.begin( ), boltInput.end( ) );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_TRUE( *stlMaxE == *boltMaxE );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif

std::array<int, 10> TestValues = {2,4,8,16,32,64,128,256,512,1024};
std::array<int,6> TestValues1 = {2048,4096, 8192,16384,32768, 65535};
//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( MaxERange, MaxEIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^22
INSTANTIATE_TEST_CASE_P( MaxEValues, MaxEIntegerVector, ::testing::ValuesIn( TestValues.begin(), TestValues.end()));
INSTANTIATE_TEST_CASE_P( MaxERange, MaxEFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16	
INSTANTIATE_TEST_CASE_P( MaxEValues, MaxEFloatVector, ::testing::ValuesIn(TestValues.begin(),TestValues.end()));
INSTANTIATE_TEST_CASE_P( MaxERange,MaxEIntegerDeviceVector,::testing::Range(0,65535,153));
INSTANTIATE_TEST_CASE_P( MaxEValues,MaxEIntegerDeviceVector,::testing::ValuesIn(TestValues.begin(),TestValues.end()));
INSTANTIATE_TEST_CASE_P( MaxERange, MaxEFloatDeviceVector, ::testing::Range( 0, 65535, 153 ) );
INSTANTIATE_TEST_CASE_P( MaxEValues, MaxEFloatDeviceVector, ::testing::ValuesIn(TestValues.begin(),TestValues.end()));

//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( MaxEValues1, MaxEIntegerVector, ::testing::ValuesIn( TestValues1.begin(), TestValues1.end()));
INSTANTIATE_TEST_CASE_P( MaxEValues1, MaxEFloatVector, ::testing::ValuesIn(TestValues1.begin(),TestValues1.end()));
INSTANTIATE_TEST_CASE_P( MaxEValues1,MaxEIntegerDeviceVector,::testing::ValuesIn(TestValues1.begin(),TestValues1.end()));
INSTANTIATE_TEST_CASE_P( MaxEValues1, MaxEFloatDeviceVector, ::testing::ValuesIn(TestValues1.begin(),TestValues1.end()));
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

//BOLT_FUNCTOR( UDD,
struct UDD
{
    int a;
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) const restrict(cpu, amp){
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    }
    bool operator < (const UDD& other) const restrict(cpu, amp) {
        return ((a+b) < (other.a+other.b));
    }
    bool operator > (const UDD& other) const restrict(cpu, amp){
        return ((a+b) > (other.a+other.b));
    }
    bool operator == (const UDD& other) const restrict(cpu, amp){
        return ((a+b) == (other.a+other.b));
    }

    UDD()
        : a(0),b(0) { }
    UDD(int _in)
        : a(_in), b(_in +1)  { }
};
//);

//BOLT_CREATE_TYPENAME( bolt::amp::device_vector< UDD >::iterator );
//Do this to register new type: It does two things -> BOLT_CREATE_TYPENAME and BOLT_CREATE_CLCODE
//BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::amp::less, int, UDD );
//BOLT_CREATE_CLCODE( bolt::amp::device_vector< UDD >::iterator, bolt::amp::deviceVectorIteratorTemplate );

TEST( MaxEUDD , UDDPlusOperatorInts )
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
    UDD *boltMaxE =bolt::amp::max_element( refInput, refInput + length );
    UDD *stdMaxE = std::max_element( refInput, refInput + length );

    EXPECT_EQ(*boltMaxE,*stdMaxE);

}

TEST( MaxEUDD , CPU_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce
    UDD *boltMaxE =bolt::amp::max_element( ctl, refInput, refInput + length );
    UDD *stdMaxE = std::max_element( refInput, refInput + length );

    EXPECT_EQ(*boltMaxE,*stdMaxE);

}

#if defined( ENABLE_TBB )
TEST( MaxEUDD , MultiCore_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    // call reduce
    UDD *boltMaxE =bolt::amp::max_element( ctl, refInput, refInput + length );
    UDD *stdMaxE = std::max_element( refInput, refInput + length );

    EXPECT_EQ(*boltMaxE,*stdMaxE);

}
#endif

TEST( MaxEUDD , Offset_UDDPlusOperatorInts )
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
    UDD *boltMaxE =bolt::amp::max_element( refInput + 10, refInput + length );
    UDD *stdMaxE = std::max_element( refInput + 10, refInput + length );

    EXPECT_EQ(*boltMaxE,*stdMaxE);

}

TEST( MaxEUDD , Offset_CPU_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce
    UDD *boltMaxE =bolt::amp::max_element( ctl, refInput + 5, refInput + length );
    UDD *stdMaxE = std::max_element( refInput + 5, refInput + length );

    EXPECT_EQ(*boltMaxE,*stdMaxE);

}

#if defined( ENABLE_TBB )
TEST( MaxEUDD , Offset_MultiCore_UDDPlusOperatorInts )
{
    //setup containers
    const int length = 1024;
    UDD refInput[ length ];
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    // call reduce
    UDD *boltMaxE =bolt::amp::max_element( ctl, refInput+10, refInput + length );
    UDD *stdMaxE = std::max_element( refInput+10, refInput + length );

    EXPECT_EQ(*boltMaxE,*stdMaxE);

}
#endif

INSTANTIATE_TYPED_TEST_CASE_P( Integer, MaxEArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, MaxEArrayTest, FloatTests );


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
void Maxeletest(int aSize)
{
    std::vector<int> A(aSize);
    //srand(GetTickCount());
    for (int i=0; i < aSize; i++)
    {
                A[i] = rand();
    };

    std::vector<int>::iterator stlReduce = std::max_element(A.begin(), A.end());
    std::vector<int>::iterator boltReduce = bolt::amp::max_element(A.begin(), A.end(),bolt::amp::greater<int>());



    checkResult("Minelement", *stlReduce, *boltReduce);
    //printf ("Sum: stl=%d,  bolt=%d %d\n", stlReduce, boltReduce, boltReduce2);
};


// Demonstrates use of bolt::control structure to control execution of routine.
void simpleReduce_TestSerial(int aSize)
{
    std::vector<int> A(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = i;
    };


    bolt::amp::control c;  // construct control structure from the queue.
    c.setForceRunMode(bolt::amp::control::SerialCpu);

    std::vector<int>::iterator stlReduce = std::max_element(A.begin(), A.end());
    std::vector<int>::iterator boltReduce = A.end();

    boltReduce = bolt::amp::max_element(c, A.begin(), A.end());


    checkResult("TestSerial", stlReduce, boltReduce);
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

TEST(sanityEPR400293, Loops){

    int size = 10;
    std::vector<int> intStdVect (size);
    bolt::amp::device_vector<int> intBoltVect (size);

    for (int i = 0 ; i < size; i++)
    {
      intBoltVect[i] = (int)std::rand()%65535;
      intStdVect[i] = intBoltVect[i];
    }

    std::vector<int>::iterator std_min_ele;
    bolt::amp::device_vector<int>::iterator bolt_min_ele;

    for (int i = 0 ; i < size; i++)
    {
      std_min_ele = std::max_element (intStdVect.begin(), intStdVect.end());
      bolt_min_ele = bolt::amp::max_element(intBoltVect.begin(), intBoltVect.end());
    }

    EXPECT_EQ( std_min_ele[0], bolt_min_ele[0] );
}

int _tmain(int argc, _TCHAR* argv[])
{
    int numIters = 100;
#if 0
    //NON_GTEST
    testDeviceVector();
    Maxele_TestControl(1024000, numIters, 0);
    Maxele_TestControl(100, 1, 0);
    Maxeletest(256);
    Maxeletest(1024);
    Maxele_TestControl(100, 1, 0);
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

