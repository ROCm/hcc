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

#include "common/stdafx.h"
#include "bolt/amp/scan.h"
#include "bolt/unicode.h"
#include "bolt/miniDump.h"
#include <gtest/gtest.h>
#include <array>
#include "bolt/amp/functional.h"
#include "bolt/amp/iterator/constant_iterator.h"
#define TEST_DOUBLE 1
#define TEST_LARGE_BUFFERS 1

#if 1

// Simple test case for bolt::inclusive_scan:
// Sum together specified numbers, compare against STL::partial_sum function.
// Demonstrates:
//    * use of bolt with STL::array iterators
//    * use of bolt with default plus 
//    * use of bolt with explicit plus argument
//template< int arraySize >
//void simpleScanArray( )
//{
  // Binary operator
  //bolt::inclusive_scan( boltA.begin( ), boltA.end(), boltA.begin( ), bolt::plus<int>( ) );

  // Invalid calls
  //bolt::inclusive_scan( boltA.rbegin( ), boltA.rend( ) );  // reverse iterators should not be supported

  //printf ("Sum: stl=%d,  bolt=%d %d %d\n", stlScan, boltScan, boltScan2, boltScan3 );
//};

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
#include "test_common.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process type parameterized tests

//  This class creates a C++ 'TYPE' out of a int value
template< int N >
class TypeValue
{
public:
    static const int value = N;
};

//  Explicit initialization of the C++ static const
template< int N >
const int TypeValue< N >::value;

//  Test fixture class, used for the Type-parameterized tests
//  Namely, the tests that use std::array and TYPED_TEST_P macros
template< typename ArrayTuple >
class ScanArrayTest: public ::testing::Test
{
public:
    ScanArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        for( int i=0; i < ArraySize; i++ )
        {
            stdInput[ i ] = 1;
            boltInput[ i ] =  stdInput[i];
        }
    };

    virtual void TearDown( )
    {};

    virtual ~ScanArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    // typedef typename std::tuple_element< 0, ArrayTuple >::type::value ArraySize;
    static const int ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;

    typename std::array< ArrayType, ArraySize > stdInput, boltInput;
    int m_Errors;
};


class scanStdVectorWithIters:public ::testing::TestWithParam<int>
{
protected:
    int myStdVectSize;
public:
    scanStdVectorWithIters():myStdVectSize(GetParam()){
    }
};

typedef scanStdVectorWithIters ScanOffsetTest;


TEST_P (ScanOffsetTest, InclOffsetTestFloat)
{
	float n = 1.f + rand()%3;
	std::vector < float > std_input( myStdVectSize, n);
    bolt::amp::device_vector< float > input( std_input.begin(), std_input.end());

    std::vector< float > refInput( myStdVectSize, n);
   // call scan
    bolt::amp::plus<float> ai2;
    bolt::amp::inclusive_scan( input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);

    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);
} 

TEST_P (ScanOffsetTest, ExclOffsetTestFloat)
{
	float n = 1.f + rand()%3;
    std::vector < float > std_input( myStdVectSize, n);
    bolt::amp::device_vector< float > input( std_input.begin(), std_input.end());
    std::vector< float > refInput( myStdVectSize,n);
   
    refInput[myStdVectSize/4] = 3.0f;

    // call scan
    bolt::amp::plus<float> ai2;

    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);
    bolt::amp::exclusive_scan(input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , 3.0f, ai2  );

    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 


TEST_P (ScanOffsetTest, InclOffsetTestFloatSerial)
{
	float n = 1.f + rand()%3;

    std::vector < float > std_input( myStdVectSize, n);
    bolt::amp::device_vector< float > input( std_input.begin(), std_input.end());
    std::vector< float > refInput( myStdVectSize, n);
   
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
   // call scan
    bolt::amp::plus<float> ai2;
    bolt::amp::inclusive_scan( ctl, input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);

    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
TEST_P (ScanOffsetTest, ExclOffsetTestFloatSerial)
{
	float n = 1.f + rand()%3;

    std::vector < float > std_input( myStdVectSize, n);
    bolt::amp::device_vector< float > input( std_input.begin(), std_input.end());
    std::vector< float > refInput( myStdVectSize,n);
   
    //refInput[myStdVectSize/4] = 3.0f;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    // call scan
    bolt::amp::plus<float> ai2;

    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);
    bolt::amp::exclusive_scan(ctl, input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , refInput[myStdVectSize/4], ai2  );
    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 

#if defined( ENABLE_TBB )
TEST_P (ScanOffsetTest, InclOffsetTestFloatMultiCore)
{
	float n = 1.f + rand()%3;

    std::vector < float > std_input( myStdVectSize, n);
    bolt::amp::device_vector< float > input( std_input.begin(), std_input.end());
    std::vector< float > refInput( myStdVectSize, n);
   
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
   // call scan
    bolt::amp::plus<float> ai2;
    bolt::amp::inclusive_scan( ctl, input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);

    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
TEST_P (ScanOffsetTest, ExclOffsetTestFloatMultiCore)
{
	float n = 1.f + rand()%3;

    std::vector < float > std_input( myStdVectSize, n);
    bolt::amp::device_vector< float > input( std_input.begin(), std_input.end());
    std::vector< float > refInput( myStdVectSize,n);
   
    refInput[myStdVectSize/4] = 3.0f;

     bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    // call scan
    bolt::amp::plus<float> ai2;

    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);
    bolt::amp::exclusive_scan(ctl, input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , 3.f, ai2  );

    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
#endif

#if (TEST_DOUBLE == 1)

TEST_P (ScanOffsetTest, InclOffsetTestDouble)
{
	double n = 1.0 + rand()%3;

    std::vector < double > std_input( myStdVectSize, n);
    bolt::amp::device_vector< double > input( std_input.begin(), std_input.end());
    std::vector< double > refInput( myStdVectSize, n);
   // call scan
    bolt::amp::plus<double> ai2;
    bolt::amp::inclusive_scan( input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);

    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
TEST_P (ScanOffsetTest, ExclOffsetTestDouble)
{
	double n = 1.0 + rand()%3;

    std::vector < double > std_input( myStdVectSize, n);
    bolt::amp::device_vector< double > input( std_input.begin(), std_input.end());
    std::vector< double > refInput( myStdVectSize,n);
   
    refInput[myStdVectSize/4] = 3.0f;

    // call scan
    bolt::amp::plus<double> ai2;

    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);
    bolt::amp::exclusive_scan(input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , 3.0f, ai2  );

    // ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    //  bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), 3.0f, ai2 );


   cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
#endif

INSTANTIATE_TEST_CASE_P(inclusiveScanIterIntLimit, ScanOffsetTest, ::testing::Range(1025, 65535, 5111)); 

//  Explicit initialization of the C++ static const
template< typename ArrayTuple >
const int ScanArrayTest< ArrayTuple >::ArraySize;

TYPED_TEST_CASE_P( ScanArrayTest );

TYPED_TEST_P( ScanArrayTest, InPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;

    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.end(), ScanArrayTest< gtest_TypeParam_ >::boltInput.begin());

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( ScanArrayTest, SerialInPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan(ctl, ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),ScanArrayTest< gtest_TypeParam_ >::boltInput.begin());

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}
#if defined( ENABLE_TBB )
TYPED_TEST_P( ScanArrayTest, MulticoreInPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan(ctl, ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),ScanArrayTest< gtest_TypeParam_ >::boltInput.begin());

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}
#endif

TYPED_TEST_P( ScanArrayTest, InPlacePlusFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;

    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ),
                                                                        bolt::amp::plus< ArrayType >( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ),
                                                                                    bolt::amp::plus< ArrayType >( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( ScanArrayTest, SerialInPlacePlusFunction )
{
     typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ),
                                                                        bolt::amp::plus< ArrayType >( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan(ctl, ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),
                                                                                    bolt::amp::plus< ArrayType >( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}
#if defined( ENABLE_TBB )
TYPED_TEST_P( ScanArrayTest, MulticoreInPlacePlusFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ),
                                                                        bolt::amp::plus< ArrayType >( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan(ctl, ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),
                                                                                    bolt::amp::plus< ArrayType >( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}
#endif

TYPED_TEST_P( ScanArrayTest, InPlaceMaxFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;

    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ),
                                                                    bolt::amp::maximum< ArrayType >( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),
                                                                                bolt::amp::maximum< ArrayType >( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( ScanArrayTest, SerialInPlaceMaxFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ),
                                                                    bolt::amp::maximum< ArrayType >( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan(ctl, ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),
                                                                                bolt::amp::maximum< ArrayType >( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}
#if defined( ENABLE_TBB )
TYPED_TEST_P( ScanArrayTest, MulticoreInPlaceMaxFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ),
                                                                    bolt::amp::maximum< ArrayType >( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan(ctl, ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),
                                                                                bolt::amp::maximum< ArrayType >( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdEnd );
    EXPECT_EQ( ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}
#endif

TYPED_TEST_P( ScanArrayTest, OutofPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;

    //  Declare temporary arrays to store results for out of place computation
    ArrayCont stdResult, boltResult;

    //  Calling the actual functions under test, out of place semantics
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdResult.begin( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),boltResult.begin());

    //  The returned iterator should be one past the end of the result array
    EXPECT_EQ( stdResult.end( ), stdEnd );
    EXPECT_EQ( boltResult.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( stdResult.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( boltResult.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( stdResult, boltResult );
}

TYPED_TEST_P( ScanArrayTest, SerialOutofPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Declare temporary arrays to store results for out of place computation
    ArrayCont stdResult, boltResult;

    //  Calling the actual functions under test, out of place semantics
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdResult.begin( ) );
   typename  ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan(ctl, ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),boltResult.begin());

    //  The returned iterator should be one past the end of the result array
    EXPECT_EQ( stdResult.end( ), stdEnd );
    EXPECT_EQ( boltResult.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( stdResult.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( boltResult.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( stdResult, boltResult );
}
#if defined( ENABLE_TBB )
TYPED_TEST_P( ScanArrayTest, MulticoreOutofPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Declare temporary arrays to store results for out of place computation
    ArrayCont stdResult, boltResult;

    //  Calling the actual functions under test, out of place semantics
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdResult.begin( ) );
    typename ArrayCont::iterator boltEnd = bolt::amp::inclusive_scan(ctl, ScanArrayTest< gtest_TypeParam_ >::boltInput.begin(),ScanArrayTest< gtest_TypeParam_ >::boltInput.end(),boltResult.begin());

    //  The returned iterator should be one past the end of the result array
    EXPECT_EQ( stdResult.end( ), stdEnd );
    EXPECT_EQ( boltResult.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( stdResult.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( boltResult.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( stdResult, boltResult );
}
#endif


REGISTER_TYPED_TEST_CASE_P( ScanArrayTest, InPlace, SerialInPlace, InPlacePlusFunction, 
                           SerialInPlacePlusFunction, InPlaceMaxFunction, SerialInPlaceMaxFunction, OutofPlace, SerialOutofPlace );
#if defined( ENABLE_TBB )
REGISTER_TYPED_TEST_CASE_P( MulticoreInPlace, MulticoreInPlacePlusFunction, 
                           MulticoreInPlaceMaxFunction, MulticoreOutofPlace );
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ScanIntegerVector: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ScanIntegerVector( ): stdInput( GetParam( ), 1 ), boltInput( GetParam( ), 1 )
    {}

protected:
    std::vector< int > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ScanFloatVector: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ScanFloatVector( ): stdInput( GetParam( ), 1.0f ), boltInput( GetParam( ), 1.0f )
    {}

protected:
    std::vector< float > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ScanDoubleVector: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ScanDoubleVector( ): stdInput( GetParam( ), 1.0 ), boltInput( GetParam( ), 1.0 )
    {}

protected:
    std::vector< double > stdInput, boltInput;
};

TEST_P( ScanIntegerVector, InclusiveInplace )
{
    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< int >::iterator boltEnd = bolt::amp::inclusive_scan( boltInput.begin( ), boltInput.end( ), 
                                                                                    boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanIntegerVector, SerialInclusiveInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< int >::iterator boltEnd = bolt::amp::inclusive_scan(ctl, boltInput.begin( ), boltInput.end( ), 
                                                                                    boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( ScanIntegerVector, MulticoreInclusiveInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< int >::iterator boltEnd = bolt::amp::inclusive_scan(ctl, boltInput.begin( ), boltInput.end( ), 
                                                                                    boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif

TEST_P( ScanFloatVector, InclusiveInplace )
{
    //  Calling the actual functions under test
    std::vector< float >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ),stdInput.begin());
    std::vector< float >::iterator boltEnd =bolt::amp::inclusive_scan(boltInput.begin(),boltInput.end(),
                                                                                    boltInput.begin());

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< float >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanFloatVector, SerialInclusiveInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    std::vector< float >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ),stdInput.begin());
    std::vector< float >::iterator boltEnd =bolt::amp::inclusive_scan(ctl, boltInput.begin(),boltInput.end(),
                                                                                    boltInput.begin());

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< float >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( ScanFloatVector, MulticoreInclusiveInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::vector< float >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ),stdInput.begin());
    std::vector< float >::iterator boltEnd =bolt::amp::inclusive_scan(ctl, boltInput.begin(),boltInput.end(),
                                                                                    boltInput.begin());

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< float >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif


TEST_P( ScanFloatVector, OffsetInclusiveInplace )
{

    int length =  (int)std::distance(stdInput.begin( ),  stdInput.end( ));
    //  Calling the actual functions under test
    std::vector< float >::iterator stdEnd  = std::partial_sum( stdInput.begin( ) + (length/2), stdInput.end( ) - (length/4),stdInput.begin()+ (length/2));
    std::vector< float >::iterator boltEnd = bolt::amp::inclusive_scan(boltInput.begin()+ (length/2),boltInput.end()- (length/4),
                                                                                    boltInput.begin()+ (length/2));

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( )- (length/4), stdEnd );
    EXPECT_EQ( boltInput.end( )- (length/4), boltEnd );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< float >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}



#if(TEST_DOUBLE == 1)
TEST_P( ScanDoubleVector, OffsetInclusiveInplace )
{

    int length =  (int)std::distance(stdInput.begin( ),  stdInput.end( ));
    //  Calling the actual functions under test
    std::vector< double >::iterator stdEnd  = std::partial_sum( stdInput.begin( ) + (length/2), stdInput.end( )- (length/4),stdInput.begin()+ (length/2));
    std::vector< double >::iterator boltEnd =bolt::amp::inclusive_scan(boltInput.begin()+ (length/2),boltInput.end()- (length/4),
                                                                                    boltInput.begin()+ (length/2));

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( )- (length/4), stdEnd );
    EXPECT_EQ( boltInput.end( )- (length/4), boltEnd );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< double >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanDoubleVector, InclusiveInplace )
{
    //  Calling the actual functions under test
    std::vector< double >::iterator stdEnd  = std::partial_sum( stdInput.begin( ),stdInput.end(),stdInput.begin());
    std::vector< double >::iterator boltEnd = bolt::amp::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                         boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< double >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanDoubleVector, SerialInclusiveInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    std::vector< double >::iterator stdEnd  = std::partial_sum( stdInput.begin( ),stdInput.end(),stdInput.begin());
    std::vector< double >::iterator boltEnd = bolt::amp::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                         boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< double >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( ScanDoubleVector, MulticoreInclusiveInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::vector< double >::iterator stdEnd  = std::partial_sum( stdInput.begin( ),stdInput.end(),stdInput.begin());
    std::vector< double >::iterator boltEnd = bolt::amp::inclusive_scan(ctl, boltInput.begin( ), boltInput.end( ),
                                                                                         boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< double >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
#endif

//  Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( Inclusive, ScanIntegerVector, ::testing::Range( 0, 1024, 1 ) );

//#if TEST_LARGE_BUFFERS
//  Test a huge range, suitable for floating point as they are less prone to overflow 
// (but floating point loses granularity at large values)
INSTANTIATE_TEST_CASE_P( Inclusive, ScanFloatVector, ::testing::Range( 0, 1048576, 74857  ) );
#if(TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( Inclusive, ScanDoubleVector, ::testing::Range( 0, 1048576, 74857  ) );
//#endif
#endif

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
	/*#if TEST_LARGE_BUFFERS
	,*/
    std::tuple< int, TypeValue< 4096 > >,
    std::tuple< int, TypeValue< 4097 > >,
    std::tuple< int, TypeValue< 65535 > >,
    std::tuple< int, TypeValue< 65536 > >
    //#endif
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
	/*#if TEST_LARGE_BUFFERS
	,*/
    std::tuple< float, TypeValue< 4096 > >,
    std::tuple< float, TypeValue< 4097 > >,
    std::tuple< float, TypeValue< 65535 > >,
    std::tuple< float, TypeValue< 65536 > >
    //#endif
> FloatTests;

INSTANTIATE_TYPED_TEST_CASE_P( Integer, ScanArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, ScanArrayTest, FloatTests );

#endif

//BOLT_FUNCTOR(uddtI2,
struct uddtI2
{
    int a;
    int b;
    // Hui
    uddtI2() restrict(cpu, amp) {}
    uddtI2(int x, int y) restrict(cpu, amp) : a(x), b(y) {}
    // User should provide explicit copy ctor
    uddtI2(const uddtI2& other) restrict(cpu, amp) : a(other.a), b(other.b) {}
    // User should provide explicit copy assign ctor
    uddtI2& operator = (const uddtI2& other) restrict(cpu, amp) {
      a = other.a;
      b = other.b;
      return *this;
    }
    bool operator==(const uddtI2& rhs) const restrict (amp,cpu)
    {
        bool equal = true;
        equal = ( a == rhs.a ) ? equal : false;
        equal = ( b == rhs.b ) ? equal : false;
        return equal;
    }
};
//);

//BOLT_CREATE_TYPENAME( bolt::amp::device_vector< uddtI2 >::iterator );
//BOLT_CREATE_CLCODE( bolt::amp::device_vector< uddtI2 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

//BOLT_FUNCTOR(AddI2,
struct AddI2
{
    uddtI2 operator()(const uddtI2 &lhs, const uddtI2 &rhs) const restrict (amp,cpu)
    {
        uddtI2 _result;
        _result.a = lhs.a+rhs.a;
        _result.b = lhs.b+rhs.b;
        return _result;
    };
}; 
//);

uddtI2 identityAddI2 = uddtI2(  0, 0 );
uddtI2 initialAddI2  = uddtI2( -1, 2 );

/******************************************************************************
 *  Mixed float and int
 *****************************************************************************/
//BOLT_FUNCTOR(uddtM3,
struct uddtM3
{
    unsigned int a;
    float        b;
    double       c;

    uddtM3() restrict(cpu, amp) {}
    uddtM3(unsigned int x, float y, double w) restrict(cpu, amp) : a(x), b(y), c(w) {}
    // User should provide explicit copy ctor
    uddtM3(const uddtM3& other) restrict(cpu, amp) : a(other.a), b(other.b), c(other.c) {}
    // User should provide explicit copy assign ctor
    uddtM3& operator = (const uddtM3& other) restrict(cpu, amp) {
      a = other.a;
      b = other.b;
      c = other.c;
      return *this;
    }
    bool operator==(const uddtM3& rhs) const restrict (amp,cpu)
    {
        bool equal = true;
        double ths = 0.00001;
        double thd = 0.0000000001;
        equal = ( a == rhs.a ) ? equal : false;
        if (rhs.b < ths && rhs.b > -ths)
            equal = ( (1.0*b - rhs.b) < ths && (1.0*b - rhs.b) > -ths) ? equal : false;
        else
            equal = ( (1.0*b - rhs.b)/rhs.b < ths && (1.0*b - rhs.b)/rhs.b > -ths) ? equal : false;
        if (rhs.c < thd && rhs.c > -thd)
            equal = ( (1.0*c - rhs.c) < thd && (1.0*c - rhs.c) > -thd) ? equal : false;
        else
            equal = ( (1.0*c - rhs.c)/rhs.c < thd && (1.0*c - rhs.c)/rhs.c > -thd) ? equal : false;
        return equal;
    }
};
//);

//BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtM3 >::iterator );
//BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtM3 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

//BOLT_FUNCTOR(MixM3,
struct MixM3
{
    uddtM3 operator()(const uddtM3 &lhs, const uddtM3 &rhs) const restrict (amp,cpu)
    {
        uddtM3 _result;
        _result.a = lhs.a^rhs.a;
        _result.b = lhs.b+rhs.b;
        _result.c = lhs.c*rhs.c;
        return _result;
    };
}; 
//);
uddtM3 identityMixM3 = uddtM3( 0, 0.f, 1.0 );
uddtM3 initialMixM3  = uddtM3( 1, 1, 1.000001 );


TEST(OffsetTest, ExclOffsetTestUdd)
{
     //setup containers
    int length = 1<<24;
    std::vector< uddtI2 > input( length, initialAddI2  );
   
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
  
    // call scan
    AddI2 ai2;

	bolt::amp::exclusive_scan( input.begin()+(length/2),    input.end()-(length/4),    input.begin()+(length/2), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin()+(length/2), refInput.end()-(length/4), refInput.begin()+(length/2), ai2);
    // compare results
    cmpArrays(refInput, input);
} 

TEST(OffsetTest, SerialExclOffsetTestUdd)
{
     //setup containers
    int length = 1<<24;
    std::vector< uddtI2 > input( length, initialAddI2  );
   
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
  
    // call scan
    AddI2 ai2;
  
	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

	bolt::amp::exclusive_scan( ctl, input.begin()+(length/2),    input.end()-(length/4),    input.begin()+(length/2), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin()+(length/2), refInput.end()-(length/4), refInput.begin()+(length/2), ai2);
    // compare results
    cmpArrays(refInput, input);
} 
#if defined( ENABLE_TBB )
TEST(OffsetTest, MultiCoreExclOffsetTestUdd)
{
     //setup containers
    int length = 1<<24;
    std::vector< uddtI2 > input( length, initialAddI2  );
   
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
  
    // call scan
    AddI2 ai2;
  
	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

	bolt::amp::exclusive_scan( ctl, input.begin()+(length/2),    input.end()-(length/4),    input.begin()+(length/2), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin()+(length/2), refInput.end()-(length/4), refInput.begin()+(length/2), ai2);
    // compare results
    cmpArrays(refInput, input);
} 
#endif

TEST(InclusiveScan, InclUdd)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  );
    //std::vector< uddtI2 > refOutput( length);

    // call scan
    AddI2 ai2;
    //bolt::amp::inclusive_scan( input.begin(), input.end(), output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan( input.begin(), input.end(), input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 

TEST(InclusiveScan, SerialInclUdd)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  );
    //std::vector< uddtI2 > refOutput( length);
 
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    // call scan
    AddI2 ai2;
    //bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan(ctl, input.begin(), input.end(), input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 
#if defined( ENABLE_TBB )
TEST(InclusiveScan, MulticoreInclUdd)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  );
    //std::vector< uddtI2 > refOutput( length);
 
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    // call scan
    AddI2 ai2;
    //bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan(ctl, input.begin(), input.end(), input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 
#endif

TEST (sanity_exclusive_scan__simple_epr377210, withIntWiCtrl)
{
	int myStdArray[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	int myBoltArray[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	
	//TAKE_AMP_CONTROL_PATH
	bolt::amp::control& my_amp_ctl= bolt::amp::control::getDefault();
	my_amp_ctl.setForceRunMode(bolt::amp::control::Automatic);
	
	bolt::amp::exclusive_scan(my_amp_ctl, myBoltArray, myBoltArray + 10, myBoltArray);
	::std::partial_sum(myStdArray, myStdArray + 10, myStdArray);

	for (int i = 1; i < 10; i++){
	   EXPECT_EQ (myStdArray[i-1], myBoltArray[i]);
	}
}

TEST (sanity_exclusive_scan__dev_vector, intsWo_Ctrl){

bolt::amp::device_vector< int > boltInput( 1024, 1 );
std::vector< int > stdInput( 1024, 1 );

//TAKE_AMP_CONTROL_PATH

bolt::amp::exclusive_scan( boltInput.begin( ), boltInput.end( ), boltInput.begin( ));
std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );

for (int i = 1; i < 1024; i++){
EXPECT_EQ(stdInput[i-1], boltInput[i]);
}

}

TEST (sanity_exclusive_scan__diff_cont2, unsigInt){
//TAKE_AMP_CONTROL_PATH
bolt::amp::control& my_amp_ctl= bolt::amp::control::getDefault();
my_amp_ctl.setForceRunMode(bolt::amp::control::Automatic);

int size = 1024;
bolt::amp::device_vector< unsigned int > boltInput( size, 5 );
bolt::amp::device_vector< unsigned int >::iterator boltEnd = bolt::amp::exclusive_scan( my_amp_ctl, boltInput.begin( ), boltInput.end( ), boltInput.begin( ), 2);

std::vector< unsigned int > stdInput( size, 5);
std::vector< unsigned int >::iterator stdEnd = bolt::amp::exclusive_scan( stdInput.begin( ), stdInput.end( ), stdInput.begin( ), 2);

EXPECT_EQ((*(boltEnd-1)), (*(stdEnd-1)));	
}

TEST (sanity_exclusive_scan__diff_cont, floatSameValuesSerialRange){
//TAKE_AMP_CONTROL_PATH
bolt::amp::control& my_amp_ctl= bolt::amp::control::getDefault();
my_amp_ctl.setForceRunMode(bolt::amp::control::Automatic);

int size = 1024;
bolt::amp::device_vector< float > boltInput( size, 1.125f );
bolt::amp::device_vector< float >::iterator boltEnd = bolt::amp::exclusive_scan( my_amp_ctl, boltInput.begin( ), boltInput.end( ), boltInput.begin( ), 2.0f);

std::vector< float > stdInput( size, 1.125f);
std::vector< float >::iterator stdEnd = bolt::amp::exclusive_scan( stdInput.begin( ), stdInput.end( ), stdInput.begin( ), 2.0f );

EXPECT_FLOAT_EQ((*(boltEnd-1)), (*(stdEnd-1)));	
}

#if (TEST_DOUBLE==1)
TEST (sanity_exclusive_scan___stdVectVsDeviceVectWithIters_d, floatSameValuesSerialRange){
//TAKE_AMP_CONTROL_PATH
bolt::amp::control& my_amp_ctl= bolt::amp::control::getDefault();
my_amp_ctl.setForceRunMode(bolt::amp::control::Automatic);

int size = 1024;
bolt::amp::device_vector< double > boltInput( size, 6.0625 );
bolt::amp::device_vector< double >::iterator boltEnd = bolt::amp::exclusive_scan( my_amp_ctl, boltInput.begin( ), boltInput.end( ), boltInput.begin( ), 1.125 );

std::vector< double > stdInput( size, 6.0625);
std::vector< double >::iterator stdEnd = bolt::amp::exclusive_scan( stdInput.begin( ), stdInput.end( ), stdInput.begin( ), 1.125);

EXPECT_DOUBLE_EQ((*(boltEnd-1)), (*(stdEnd-1)));	
} 
#endif

TEST(InclusiveScan, InclFloat)
{
    //setup containers

    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f+rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::amp::plus<float> ai2;
    //bolt::amp::inclusive_scan( input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 

TEST(InclusiveScan, SerialInclFloat)
{
    //setup containers

    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f+rand()%3;
        refInput[i] = input[i];
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    // call scan
    bolt::amp::plus<float> ai2;
    //bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 
#if defined( ENABLE_TBB )
TEST(InclusiveScan, MulticoreInclFloat)
{
    //setup containers

    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f+rand()%3;
        refInput[i] = input[i];
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    bolt::amp::plus<float> ai2;
    //bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 
#endif

#if(TEST_DOUBLE ==1)
TEST(InclusiveScan, IncluddtM3)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);
    // call scan
    MixM3 M3;
   /* bolt::amp::inclusive_scan( input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::amp::inclusive_scan( input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    cmpArrays(refInput, input);  
} 

TEST(InclusiveScan, SerialIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    // call scan
    MixM3 M3;
   /* bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::amp::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    cmpArrays(refInput, input);  
} 
#if defined( ENABLE_TBB )
TEST(InclusiveScan, MulticoreIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested with serial also
    // call scan
    MixM3 M3;
   /* bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::amp::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    cmpArrays(refInput, input);  
} 
#endif
#endif

TEST(ExclusiveScan, ExclUdd)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
    //std::vector< uddtI2 > refOutput( length);
    // call scan
    AddI2 ai2;
    //bolt::amp::exclusive_scan( input.begin(),    input.end(),    output.begin(), initialAddI2, ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan( input.begin(),    input.end(),    input.begin(), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
}

TEST(ExclusiveScan, SerialExclUdd)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
    //std::vector< uddtI2 > refOutput( length);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    // call scan
    AddI2 ai2;
    //bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialAddI2, ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
}
#if defined( ENABLE_TBB )
TEST(ExclusiveScan, MulticoreExclUdd)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
    //std::vector< uddtI2 > refOutput( length);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    AddI2 ai2;
    //bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialAddI2, ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
}
#endif

TEST(ExclusiveScan, ExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f+rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;
    // call scan
    bolt::amp::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::amp::exclusive_scan( input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );
    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::amp::exclusive_scan( input.begin(),    input.end(),    input.begin(), 3.0f, ai2 );
    // compare results
    cmpArrays(refInput, input);
} 

TEST(ExclusiveScan, SerialExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f+rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    // call scan
    bolt::amp::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );
    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::amp::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), 3.0f, ai2 );
    // compare results
    cmpArrays(refInput, input);
} 
#if defined( ENABLE_TBB )
TEST(ExclusiveScan, MulticoreExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f+rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    bolt::amp::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );
    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), 3.0f, ai2 );
    // compare results
    cmpArrays(refInput, input);
} 
#endif

#if(TEST_DOUBLE ==1)
TEST(ExclusiveScan, ExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
    // call scan
    MixM3 M3;
   /* bolt::amp::exclusive_scan(  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::amp::exclusive_scan(  input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);  
    cmpArrays(refInput, input);  
} 

TEST(ExclusiveScan, SerialExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    // call scan
    MixM3 M3;
    /*bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    cmpArrays(refOutput, output);  */

	bolt::amp::exclusive_scan(  ctl, input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);  
    cmpArrays(refInput, input);  
} 
#if defined( ENABLE_TBB )
TEST(ExclusiveScan, MulticoreExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    MixM3 M3;
    /*bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);  
    cmpArrays(refOutput, output);  */

	bolt::amp::exclusive_scan(  ctl, input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);  
    cmpArrays(refInput, input);  
} 
#endif
#endif
///////////////////////////////////////////////Device vectorTBB and serial path test///////////////////


TEST(InclusiveScan, DeviceVectorInclFloat)
{
    int length = 1<<10;
   
    //bolt::amp::device_vector< float > input(length);
    //bolt::amp::device_vector< float > output(length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
   
   for(int i=0; i<length; i++) {
        refInput[i] = 1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(refInput.begin(), refInput.end());
   //  bolt::amp::device_vector< float > input(refInput.begin(), length);
   //  bolt::amp::device_vector< float > output( refOutput.begin(), length);

    // call scan
    bolt::amp::plus<float> ai2;
    //bolt::amp::inclusive_scan( input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 

TEST(InclusiveScan, SerialDeviceVectorInclFloat)
{
    int length = 1<<10;
   
    //bolt::amp::device_vector< float > input(length);
    //bolt::amp::device_vector< float > output(length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
   
    for(int i=0; i<length; i++) {
        refInput[i] = 1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(refInput.begin(), refInput.end());

  //  bolt::amp::device_vector< float > input(refInput.begin(), length);
  //  bolt::amp::device_vector< float > output( refOutput.begin(), length);


    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    // call scan
    bolt::amp::plus<float> ai2;
    //bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 
#if defined( ENABLE_TBB )
TEST(InclusiveScan, MulticoreDeviceVectorInclFloat)
{
    int length = 1<<10;
   
    //bolt::amp::device_vector< float > output(length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
   
    for(int i=0; i<length; i++) {
        refInput[i] = 1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(refInput.begin(), refInput.end());

  // bolt::amp::device_vector< float > input(refInput.begin(), length);
  //  bolt::amp::device_vector< float > output( refOutput.begin(), length);


    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    bolt::amp::plus<float> ai2;
    //bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 
#endif


TEST(InclusiveScan, DeviceVectorIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    bolt::amp::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::amp::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);
    // call scan
    MixM3 M3;
    /*bolt::amp::inclusive_scan( input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::amp::inclusive_scan( input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    cmpArrays(refInput, input);  
} 

TEST(InclusiveScan, SerialDeviceVectorIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    bolt::amp::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::amp::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    // call scan
    MixM3 M3;
/*    bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3); 
    cmpArrays(refOutput, output); */ 

	bolt::amp::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    cmpArrays(refInput, input);
} 
#if defined( ENABLE_TBB )
TEST(InclusiveScan, MulticoreDeviceVectorIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    bolt::amp::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::amp::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    MixM3 M3;
   /* bolt::amp::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3); 
    cmpArrays(refOutput, output);  */

	bolt::amp::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    cmpArrays(refInput, input);
} 
#endif


TEST(ExclusiveScan, DeviceVectorExclFloat)
{
    //setup containers
    int length = 1<<10;
    bolt::amp::device_vector< float > input( length);
    //bolt::amp::device_vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;

    // call scan
    bolt::amp::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::amp::exclusive_scan( input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::amp::exclusive_scan( input.begin(),    input.end(),    input.begin(), 3.0f, ai2 );

    // compare results
    cmpArrays(refInput, input);
}

TEST(ExclusiveScan, SerialDeviceVectorExclFloat)
{
    //setup containers
    int length = 1<<10;
    bolt::amp::device_vector< float > input( length);
    //bolt::amp::device_vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    // call scan
    bolt::amp::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::amp::exclusive_scan(ctl, input.begin(),    input.end(),    input.begin(), 3.0f, ai2 );

    // compare results
    cmpArrays(refInput, input);
}
#if defined( ENABLE_TBB )
TEST(ExclusiveScan, MulticoreDeviceVectorExclFloat)
{
    //setup containers
    int length = 1<<10;
    bolt::amp::device_vector< float > input( length);
    //bolt::amp::device_vector< float > output( length);
    std::vector< float > refInput( length);
   // std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    // call scan
    bolt::amp::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	
	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::amp::exclusive_scan(ctl, input.begin(),    input.end(),    input.begin(), 3.0f, ai2 );

    // compare results
    cmpArrays(refInput, input);
}
#endif


TEST(ExclusiveScan, DeviceVectorExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    bolt::amp::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::amp::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
    // call scan
    MixM3 M3;
   /* bolt::amp::exclusive_scan( input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::amp::exclusive_scan( input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 

TEST(ExclusiveScan, SerialDeviceVectorExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    bolt::amp::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::amp::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    // call scan
    MixM3 M3;
    /*bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::amp::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 
#if defined( ENABLE_TBB )
TEST(ExclusiveScan, MulticoreDeviceVectorExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    bolt::amp::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::amp::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    MixM3 M3;
    /*bolt::amp::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output); */ 

	bolt::amp::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 
#endif

TEST(AMPTileLimitTest, AMPTileLimitTest)
{
    //setup containers
    for(int i=24 ; i<28; i++)
	{
		int size = 1<<i;
		bolt::amp::device_vector< int > input( size, 2 );
		std::vector< int > refInput( size, 2 );  refInput[0] = 2;
		bolt::amp::exclusive_scan( input.begin(),    input.end(),    input.begin(), 2, bolt::amp::plus<int>() );
		::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), bolt::amp::plus<int>());
		cmpArrays(refInput, input);  
	}
} 

TEST(ScanByKeyEPR392210, constant_iterator)
{
    const int length = 1<<10;

    int value = 100;
    std::vector< int > svInVec1( length );
    std::vector< int > svInVec2( length );
    
    std::vector< int > svOutVec( length );
    std::vector< int > stlOut( length );

    bolt::amp::device_vector< int > dvInVec1( length );
    bolt::amp::device_vector< int > dvOutVec( length );

    bolt::amp::constant_iterator<int> constIter1 (value);
    bolt::amp::constant_iterator<int> constIter2 (10);

    bolt::amp::plus<int> pls;
    int n = (int) 1 + rand()%10;

    bolt::amp::control ctl;
    ctl.setForceRunMode( bolt::amp::control::SerialCpu );
    bolt::amp::inclusive_scan( ctl, constIter1, constIter1 + length, dvOutVec.begin(), pls );
    
    std::vector<int> const_vector(length,value);
    std::partial_sum(const_vector.begin(), const_vector.end(), stlOut.begin(), pls);
    
    for(int i =0; i< length; i++)
    {
      EXPECT_EQ( dvOutVec[i], stlOut[i]);
    }
}


int _tmain(int argc, _TCHAR* argv[])
{
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;

    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Register our minidump generating logic
    #if defined(_WIN32)
    bolt::miniDumpSingleton::enableMiniDumps( );
    #endif
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
