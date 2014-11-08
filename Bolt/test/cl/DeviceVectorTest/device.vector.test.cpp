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

#include "stdafx.h"
#include <vector>
#include <array>
#include "bolt/unicode.h"
#include "bolt/miniDump.h"
#include "gtest/gtest.h"
///////////////////////////////////////////////////////////////////////////////////////
//CL and AMP device_vector tests are integrated.To use AMP tests change AMP_TESTS to 1
///////////////////////////////////////////////////////////////////////////////////////
//#define AMP_TESTS 0

#if AMP_TESTS
    #include "bolt/amp/functional.h"
    #include "bolt/amp/device_vector.h"
    #define BCKND amp

#else

    #include "bolt/cl/functional.h"
    #include "bolt/cl/device_vector.h"
    #include "bolt/cl/fill.h"
    #include "common/test_common.h"
    #define BCKND cl

#endif

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
            boltInput[ i ] = 1;
        }
    };

    virtual void TearDown( )
    {};

    virtual ~ScanArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    // typedef typename std::tuple_element< 0, ArrayTuple >::type::value ArraySize;
    static const size_t ArraySize =  std::tuple_element< 1, ArrayTuple >::type::value;

    typename std::array< ArrayType, ArraySize > stdInput, boltInput;
    int m_Errors;
};

TYPED_TEST_CASE_P( ScanArrayTest );

TYPED_TEST_P( ScanArrayTest, InPlace )
{
   
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    

}

TYPED_TEST_P( ScanArrayTest, InPlacePlusFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;        

}

TYPED_TEST_P( ScanArrayTest, InPlaceMaxFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;       

}

TYPED_TEST_P( ScanArrayTest, OutofPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;       

}

REGISTER_TYPED_TEST_CASE_P( ScanArrayTest, InPlace, InPlacePlusFunction, InPlaceMaxFunction, OutofPlace );

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
}

TEST_P( ScanFloatVector, InclusiveInplace )
{
}

TEST_P( ScanDoubleVector, InclusiveInplace )
{
}

////  Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
//INSTANTIATE_TEST_CASE_P( Inclusive, ScanIntegerVector, ::testing::Range( 0, 1024, 1 ) );
//
////  Test a huge range, suitable for floating point as they are less prone to overflow (but floating point loses granularity at large values)
//INSTANTIATE_TEST_CASE_P( Inclusive, ScanFloatVector, ::testing::Range( 0, 1048576, 4096 ) );
//INSTANTIATE_TEST_CASE_P( Inclusive, ScanDoubleVector, ::testing::Range( 0, 1048576, 4096 ) );

typedef ::testing::Types< 
    std::tuple< int, TypeValue< 1 > >,
    //std::tuple< int, TypeValue< bolt::scanMultiCpuThreshold - 1 > >,
    //std::tuple< int, TypeValue< bolt::scanGpuThreshold - 1 > >,
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
    //std::tuple< int, TypeValue< 131032 > >,       // uncomment these to generate failures; stack overflow
    //std::tuple< int, TypeValue< 262154 > >,
    std::tuple< int, TypeValue< 65536 > >
> IntegerTests;

typedef ::testing::Types< 
    std::tuple< float, TypeValue< 1 > >,
    //std::tuple< float, TypeValue< bolt::scanMultiCpuThreshold - 1 > >,
    //std::tuple< float, TypeValue< bolt::scanGpuThreshold - 1 > >,
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

//INSTANTIATE_TYPED_TEST_CASE_P( Integer, ScanArrayTest, IntegerTests );
//INSTANTIATE_TYPED_TEST_CASE_P( Float, ScanArrayTest, FloatTests );

TEST( Constructor, ContainerIteratorEmpty )
{
    bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::iterator itBegin = dV.begin( );
    bolt::BCKND::device_vector< int >::iterator itEnd = dV.end( );

    EXPECT_TRUE( itBegin == itEnd );
    EXPECT_TRUE( dV.empty() );
}

TEST( Constructor, ConstContainerConstIteratorEmpty )
{
    const bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::const_iterator itBegin = dV.begin( );
    bolt::BCKND::device_vector< int >::const_iterator itEnd = dV.end( );

    EXPECT_TRUE( itBegin == itEnd );
}

TEST( Constructor, ConstContainerConstIteratorCEmpty )
{
    const bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::const_iterator itBegin = dV.cbegin( );
    bolt::BCKND::device_vector< int >::const_iterator itEnd = dV.cend( );

    EXPECT_TRUE( itBegin == itEnd );
}

TEST( Constructor, ContainerConstIteratorCEmpty )
{
    bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::const_iterator itBegin = dV.cbegin( );
    bolt::BCKND::device_vector< int >::const_iterator itEnd = dV.cend( );

    EXPECT_TRUE( itBegin == itEnd );
}

TEST( Constructor, ContainerConstIteratorEmpty )
{
    bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::const_iterator itBegin = dV.begin( );
    bolt::BCKND::device_vector< int >::const_iterator itEnd = dV.end( );

    EXPECT_TRUE( itBegin == itEnd );
}

TEST( Constructor, Size5AndValue3OperatorValueType )
{
    bolt::BCKND::device_vector< int > dV( 5, 3 );
    EXPECT_EQ( 5, dV.size( ) );

    EXPECT_EQ( 3, dV[ 0 ] );
    EXPECT_EQ( 3, dV[ 1 ] );
    EXPECT_EQ( 3, dV[ 2 ] );
    EXPECT_EQ( 3, dV[ 3 ] );
    EXPECT_EQ( 3, dV[ 4 ] );
}

TEST( Iterator, Compatibility )
{
    bolt::BCKND::device_vector< int > dV;
    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::iterator Iter0( dV, 0 );
    bolt::BCKND::device_vector< int >::const_iterator cIter0( dV, 0 );
    EXPECT_TRUE( Iter0 == cIter0 );

    bolt::BCKND::device_vector< int >::iterator Iter1( dV, 0 );
    bolt::BCKND::device_vector< int >::const_iterator cIter1( dV, 1 );
    EXPECT_TRUE( Iter1 != cIter1 );
}

TEST( Iterator, OperatorEqual )
{
    bolt::BCKND::device_vector< int > dV;
    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::iterator Iter0( dV, 0 );
    bolt::BCKND::device_vector< int >::iterator cIter0( dV, 0 );
    EXPECT_TRUE( Iter0 == cIter0 );

    bolt::BCKND::device_vector< int >::const_iterator Iter1( dV, 0 );
    bolt::BCKND::device_vector< int >::const_iterator cIter1( dV, 1 );
    EXPECT_TRUE( Iter1 != cIter1 );

    bolt::BCKND::device_vector< int > dV2;

    bolt::BCKND::device_vector< int >::const_iterator Iter2( dV, 0 );
    bolt::BCKND::device_vector< int >::const_iterator cIter2( dV2, 0 );
    EXPECT_TRUE( Iter2 != cIter2 );
}

//TODO Add all test cases for Reverse and Const Reverse Iterator
// insert/erase using base(), self, and constructing a base iterator

TEST( Constructor, ContainerReverseIteratorEmpty )
{
    bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::reverse_iterator itBegin = dV.rbegin( );
    bolt::BCKND::device_vector< int >::reverse_iterator itEnd = dV.rend( );

    EXPECT_TRUE( itBegin == itEnd );    
}

TEST( Constructor, ConstContainerConstReverseIteratorEmpty )
{
    const bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::const_reverse_iterator itBegin = dV.rbegin( );
    bolt::BCKND::device_vector< int >::const_reverse_iterator itEnd = dV.rend( );

    EXPECT_TRUE( itBegin == itEnd );
    
}

TEST( Constructor, ConstContainerConstReverseIteratorCEmpty )
{
    const bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::const_reverse_iterator itBegin = dV.crbegin( );
    bolt::BCKND::device_vector< int >::const_reverse_iterator itEnd = dV.crend( );

    EXPECT_TRUE( itBegin == itEnd );
}

TEST( Constructor, ContainerConstReverseIteratorCEmpty )
{
    bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::const_reverse_iterator itBegin = dV.crbegin( );
    bolt::BCKND::device_vector< int >::const_reverse_iterator itEnd = dV.crend( );

    EXPECT_TRUE( itBegin == itEnd );
}

TEST( Constructor, ContainerConstReverseIteratorEmpty )
{
    bolt::BCKND::device_vector< int > dV;

    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::const_reverse_iterator itBegin = dV.rbegin( );
    bolt::BCKND::device_vector< int >::const_reverse_iterator itEnd = dV.rend( );

    EXPECT_TRUE( itBegin == itEnd );
}


TEST( ReverseIterator, CompatibilityReverse )
{
    bolt::BCKND::device_vector< int > dV;
    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::reverse_iterator Iter0( dV, 0 );
    bolt::BCKND::device_vector< int >::const_reverse_iterator cIter0( dV, 0 );
    EXPECT_TRUE( Iter0 == cIter0 );

    bolt::BCKND::device_vector< int >::reverse_iterator Iter1( dV, 0 );
    bolt::BCKND::device_vector< int >::const_reverse_iterator cIter1( dV, 1 );
    EXPECT_TRUE( Iter1 != cIter1 );
}

TEST( ReverseIterator, OperatorEqualReverse )
{
    bolt::BCKND::device_vector< int > dV;
    EXPECT_EQ( 0, dV.size( ) );

    bolt::BCKND::device_vector< int >::reverse_iterator Iter0( dV, 0 );
    bolt::BCKND::device_vector< int >::reverse_iterator cIter0( dV, 0 );
    EXPECT_TRUE( Iter0 == cIter0 );

    bolt::BCKND::device_vector< int >::const_reverse_iterator Iter1( dV, 0 );
    bolt::BCKND::device_vector< int >::const_reverse_iterator cIter1( dV, 1 );
    EXPECT_TRUE( Iter1 != cIter1 );

    bolt::BCKND::device_vector< int > dV2;

    bolt::BCKND::device_vector< int >::const_reverse_iterator Iter2( dV, 0 );
    bolt::BCKND::device_vector< int >::const_reverse_iterator cIter2( dV2, 0 );
    EXPECT_TRUE( Iter2 != cIter2 );
}

TEST( VectorReverseIterator, Size6AndValue7Dereference )
{
    bolt::BCKND::device_vector< int > dV( 6ul, 7 );
    EXPECT_EQ( 6, dV.size( ) );

    bolt::BCKND::device_vector< int >::reverse_iterator myIter = dV.rbegin( );
    EXPECT_EQ( 7, *(myIter + 0) );
    EXPECT_EQ( 7, *(myIter + 1) );
    EXPECT_EQ( 7, *(myIter + 2) );
    EXPECT_EQ( 7, *(myIter + 3) );
    EXPECT_EQ( 7, *(myIter + 4) );
    EXPECT_EQ( 7, *(myIter + 5) );
}

//
//TEST( VectorReverseIterator, Size6AndValue7OperatorValueType )
//{
//    bolt::BCKND::device_vector< int > dV( 6, 7 );
//    EXPECT_EQ( 6, dV.size( ) );
//
//    bolt::BCKND::device_vector< int >::reverse_iterator myIter = dV.rbegin( );
//
//    EXPECT_EQ(  myIter[ 0 ],7 );
//    EXPECT_EQ(  myIter[ 1 ],7 );
//    EXPECT_EQ(  myIter[ 2 ],7 );
//    EXPECT_EQ(  myIter[ 3 ],7 );
//    EXPECT_EQ(  myIter[ 4 ],7 );
//    EXPECT_EQ(  myIter[ 5 ],7 );
//}

TEST( VectorReverseIterator, ArithmeticAndEqual )
{
    bolt::BCKND::device_vector< int > dV( 5 );
    EXPECT_EQ( 5, dV.size( ) );

    bolt::BCKND::device_vector< int >::reverse_iterator myIter = dV.rbegin( );
    *myIter = 1;
    ++myIter;
    *myIter = 2;
    myIter++;
    *myIter = 3;
    myIter += 1;
    *(myIter + 0) = 4;
    *(myIter + 1) = 5;
    myIter += 1;

    EXPECT_EQ( 1, dV[ 4 ] );
    EXPECT_EQ( 2, dV[ 3 ] );
    EXPECT_EQ( 3, dV[ 2 ] );
    EXPECT_EQ( 4, dV[ 1 ] );
    EXPECT_EQ( 5, dV[ 0 ] );
}



//Reverse Iterator test cases end

TEST( VectorReference, OperatorEqual )
{
    bolt::BCKND::device_vector< int > dV( 5 );

    dV[ 0 ] = 1;
    dV[ 1 ] = 2;
    dV[ 2 ] = 3;
    dV[ 3 ] = 4;
    dV[ 4 ] = 5;

    EXPECT_EQ( 1, dV[ 0 ] );
    EXPECT_EQ( 2, dV[ 1 ] );
    EXPECT_EQ( 3, dV[ 2 ] );
    EXPECT_EQ( 4, dV[ 3 ] );
    EXPECT_EQ( 5, dV[ 4 ] );
}

TEST( VectorReference, OperatorValueType )
{
    bolt::BCKND::device_vector< int > dV( 5 );

    dV[ 0 ] = 1;
    dV[ 1 ] = 2;
    dV[ 2 ] = 3;
    dV[ 3 ] = 4;
    dV[ 4 ] = 5;

    std::vector< int > readBack( 5 );
    readBack[ 0 ] = dV[ 0 ];
    readBack[ 1 ] = dV[ 1 ];
    readBack[ 2 ] = dV[ 2 ];
    readBack[ 3 ] = dV[ 3 ];
    readBack[ 4 ] = dV[ 4 ];

    EXPECT_EQ( readBack[ 0 ], dV[ 0 ] );
    EXPECT_EQ( readBack[ 1 ], dV[ 1 ] );
    EXPECT_EQ( readBack[ 2 ], dV[ 2 ] );
    EXPECT_EQ( readBack[ 3 ], dV[ 3 ] );
    EXPECT_EQ( readBack[ 4 ], dV[ 4 ] );
}

//TEST( VectorIterator, Size6AndValue7OperatorValueType )
//{
//    bolt::BCKND::device_vector< int > dV( 6, 7 );
//    EXPECT_EQ( 6, dV.size( ) );
//
//    bolt::BCKND::device_vector< int >::iterator myIter = dV.begin( );
//
//    EXPECT_EQ(  myIter[ 0 ],7 );
//    EXPECT_EQ(  myIter[ 1 ],7 );
//    EXPECT_EQ(  myIter[ 2 ],7 );
//    EXPECT_EQ(  myIter[ 3 ],7 );
//    EXPECT_EQ(  myIter[ 4 ],7 );
//    EXPECT_EQ(  myIter[ 5 ],7 );
//}

TEST( VectorIterator, Size6AndValue7Dereference )
{
    bolt::BCKND::device_vector< int > dV( 6ul, 7 );
    EXPECT_EQ( 6, dV.size( ) );

    bolt::BCKND::device_vector< int >::iterator myIter = dV.begin( );

    EXPECT_EQ( 7, *(myIter + 0) );
    EXPECT_EQ( 7, *(myIter + 1) );
    EXPECT_EQ( 7, *(myIter + 2) );
    EXPECT_EQ( 7, *(myIter + 3) );
    EXPECT_EQ( 7, *(myIter + 4) );
    EXPECT_EQ( 7, *(myIter + 5) );
}

TEST( VectorIterator, ArithmeticAndEqual )
{
    bolt::BCKND::device_vector< int > dV( 5 );
    EXPECT_EQ( 5, dV.size( ) );

    bolt::BCKND::device_vector< int >::iterator myIter = dV.begin( );

    *myIter = 1;
    ++myIter;
    *myIter = 2;
    myIter++;
    *myIter = 3;
    myIter += 1;
    *(myIter + 0) = 4;
    *(myIter + 1) = 5;
    myIter += 1;

    EXPECT_EQ( 1, dV[ 0 ] );
    EXPECT_EQ( 2, dV[ 1 ] );
    EXPECT_EQ( 3, dV[ 2 ] );
    EXPECT_EQ( 4, dV[ 3 ] );
    EXPECT_EQ( 5, dV[ 4 ] );
}

TEST( Vector, Erase )
{
    bolt::BCKND::device_vector< int > dV( 5 );
    EXPECT_EQ( 5, dV.size( ) );

    dV[ 0 ] = 1;
    dV[ 1 ] = 2;
    dV[ 2 ] = 3;
    dV[ 3 ] = 4;
    dV[ 4 ] = 5;

    bolt::BCKND::device_vector< int >::iterator myIter = dV.begin( );
    myIter += 2;

    bolt::BCKND::device_vector< int >::iterator myResult = dV.erase( myIter );
    EXPECT_EQ( 4, dV.size( ) );
    EXPECT_EQ( 4, *myResult );

    EXPECT_EQ( 1, dV[ 0 ] );
    EXPECT_EQ( 2, dV[ 1 ] );
    EXPECT_EQ( 4, dV[ 2 ] );
    EXPECT_EQ( 5, dV[ 3 ] );
}

TEST( Vector, Clear )
{
    bolt::BCKND::device_vector< int > dV( 5ul, 3 );
    EXPECT_EQ( 5, dV.size( ) );

    dV.clear( );
    EXPECT_EQ( 0, dV.size( ) );
}

TEST( Vector, EraseEntireRange )
{
    bolt::BCKND::device_vector< int > dV( 5 );
    EXPECT_EQ( 5, dV.size( ) );

    dV[ 0 ] = 1;
    dV[ 1 ] = 2;
    dV[ 2 ] = 3;
    dV[ 3 ] = 4;
    dV[ 4 ] = 5;

    bolt::BCKND::device_vector< int >::iterator myBegin = dV.begin( );
    bolt::BCKND::device_vector< int >::iterator myEnd = dV.end( );

    bolt::BCKND::device_vector< int >::iterator myResult = dV.erase( myBegin, myEnd );
    EXPECT_EQ( 0, dV.size( ) );
}

TEST( Vector, EraseSubRange )
{
    bolt::BCKND::device_vector< int > dV( 5 );
    EXPECT_EQ( 5, dV.size( ) );

    dV[ 0 ] = 1;
    dV[ 1 ] = 2;
    dV[ 2 ] = 3;
    dV[ 3 ] = 4;
    dV[ 4 ] = 5;

    bolt::BCKND::device_vector< int >::iterator myBegin = dV.begin( );
    bolt::BCKND::device_vector< int >::iterator myEnd = dV.end( );
    myEnd -= 2;

    bolt::BCKND::device_vector< int >::iterator myResult = dV.erase( myBegin, myEnd );
    EXPECT_EQ( 2, dV.size( ) );
}

TEST( Vector, InsertBegin )
{
    bolt::BCKND::device_vector< int > dV( 5 );
    EXPECT_EQ( 5, dV.size( ) );
    dV[ 0 ] = 1;
    dV[ 1 ] = 2;
    dV[ 2 ] = 3;
    dV[ 3 ] = 4;
    dV[ 4 ] = 5;

    bolt::BCKND::device_vector< int >::iterator myResult = dV.insert( dV.cbegin( ), 7 );
    EXPECT_EQ( 7, *myResult );
    EXPECT_EQ( 6, dV.size( ) );
}

TEST( Vector, InsertEnd )
{
    bolt::BCKND::device_vector< int > dV( 5ul, 3 );
    EXPECT_EQ( 5, dV.size( ) );

    bolt::BCKND::device_vector< int >::iterator myResult = dV.insert( dV.cend( ), 1 );
    EXPECT_EQ( 1, *myResult );
    EXPECT_EQ( 6, dV.size( ) );
}

TEST( Vector, DataRead )
{
    bolt::BCKND::device_vector< int > dV( 5ul, 3 );
    EXPECT_EQ( 5, dV.size( ) );
    dV[ 0 ] = 1;
    dV[ 1 ] = 2;
    dV[ 2 ] = 3;
    dV[ 3 ] = 4;
    dV[ 4 ] = 5;

    bolt::BCKND::device_vector< int >::pointer mySP = dV.data( );

    EXPECT_EQ( 1, mySP[ 0 ] );
    EXPECT_EQ( 2, mySP[ 1 ] );
    EXPECT_EQ( 3, mySP[ 2 ] );
    EXPECT_EQ( 4, mySP[ 3 ] );
    EXPECT_EQ( 5, mySP[ 4 ] );
}

TEST( Vector, DataWrite )
{
    bolt::BCKND::device_vector< int > dV( 5 );
    EXPECT_EQ( 5, dV.size( ) );

    bolt::BCKND::device_vector< int >::pointer mySP = dV.data( );
    mySP[ 0 ] = 1;
    mySP[ 1 ] = 2;
    mySP[ 2 ] = 3;
    mySP[ 3 ] = 4;
    mySP[ 4 ] = 5;

    EXPECT_EQ( 1, mySP[ 0 ] );
    EXPECT_EQ( 2, mySP[ 1 ] );
    EXPECT_EQ( 3, mySP[ 2 ] );
    EXPECT_EQ( 4, mySP[ 3 ] );
    EXPECT_EQ( 5, mySP[ 4 ] );
}

TEST( Vector, wdSpecifyingSize )
{
    size_t mySize = 10;
    bolt::BCKND::device_vector<int> myIntDevVect;
    int myIntArray[10] = {2, 3, 5, 6, 76, 5, 8, -10, 30, 34};

	for (size_t i = 0; i < mySize; ++i){
        myIntDevVect.push_back(myIntArray[i]);
    }

    size_t DevSize = myIntDevVect.size();
    
    EXPECT_EQ (mySize, DevSize);
}

TEST( Vector, InsertFloatRangeEmpty )
{
    bolt::BCKND::device_vector< float > dV;
    EXPECT_EQ( 0, dV.size( ) );

    dV.insert( dV.cbegin( ), 5, 7.0f );
    EXPECT_EQ( 5, dV.size( ) );
    EXPECT_FLOAT_EQ( 7.0f, dV[ 0 ] );
    EXPECT_FLOAT_EQ( 7.0f, dV[ 1 ] );
    EXPECT_FLOAT_EQ( 7.0f, dV[ 2 ] );
    EXPECT_FLOAT_EQ( 7.0f, dV[ 3 ] );
    EXPECT_FLOAT_EQ( 7.0f, dV[ 4 ] );
}

//TEST( Vector, InsertIntegerRangeEmpty )
//{
//    bolt::BCKND::device_vector< int > dV;
//    EXPECT_EQ( 0, dV.size( ) );
//
//    dV.insert( dV.cbegin( ), 5, 7 );
//    EXPECT_EQ( 5, dV.size( ) );
//    EXPECT_EQ( 7, dV[ 0 ] );
//    EXPECT_EQ( 7, dV[ 1 ] );
//    EXPECT_EQ( 7, dV[ 2 ] );
//    EXPECT_EQ( 7, dV[ 3 ] );
//    EXPECT_EQ( 7, dV[ 4 ] );
//}

TEST( Vector, InsertFloatRangeIterator )
{
    bolt::BCKND::device_vector< float > dV;
    EXPECT_EQ( 0, dV.size( ) );

    std::vector< float > sV( 5 );
    sV[ 0 ] = 1.0f;
    sV[ 1 ] = 2.0f;
    sV[ 2 ] = 3.0f;
    sV[ 3 ] = 4.0f;
    sV[ 4 ] = 5.0f;

    dV.insert( dV.cbegin( ), sV.begin( ), sV.end( ) );
    EXPECT_EQ( 5, dV.size( ) );
    EXPECT_FLOAT_EQ( 1.0f, dV[ 0 ] );
    EXPECT_FLOAT_EQ( 2.0f, dV[ 1 ] );
    EXPECT_FLOAT_EQ( 3.0f, dV[ 2 ] );
    EXPECT_FLOAT_EQ( 4.0f, dV[ 3 ] );
    EXPECT_FLOAT_EQ( 5.0f, dV[ 4 ] );
}

TEST( Vector, Resize )
{
    bolt::BCKND::device_vector< float > dV;
    EXPECT_EQ( 0, dV.size( ) );

    std::vector< float > sV( 10 );
    for(int i=0; i<10; i++)
    {
        sV[i] = (float)i;
    }

    dV.insert( dV.cbegin( ), sV.begin( ), sV.end( ) );
    EXPECT_EQ( 10, dV.size( ) );
    dV.resize(7);
    EXPECT_EQ( 7, dV.size( ) );
    for(int i=0; i<7; i++)
    {
        EXPECT_FLOAT_EQ( (float)i, dV[ i ] );
    }
    dV.resize(15, 7);
    EXPECT_EQ( 15, dV.size( ) );
    for(int i=0; i<15; i++)
    {
        if(i<7)
            EXPECT_FLOAT_EQ( (float)i, dV[ i ] );
        else
            EXPECT_FLOAT_EQ( 7.0f, dV[ i ] );
    }
}

TEST( Vector, ShrinkToFit)
{
	bolt::BCKND::device_vector< int > dV(100);
	EXPECT_EQ(dV.size(),dV.capacity());
	dV.reserve(200);
	EXPECT_EQ(200,dV.capacity());
	dV.shrink_to_fit();
	EXPECT_EQ(dV.size(),dV.capacity());
#if 0
	//Just like that.
	for(int i=0; i<(2<<21);i+=(2<<3)){
		dV.reserve(i);
		dV.shrink_to_fit();
		EXPECT_EQ(dV.size(),dV.capacity());
	}

#endif
}

TEST( Vector, DataRoutine )
{
    std::vector<int>a( 100 );

    //  Initialize data (could use constructor?)
    std::fill( a.begin( ),a.end( ), 100 );

    //  deep copy, all data is duplicated
    bolt::BCKND::device_vector< int > da( a.begin( ),a.end( ) );

    //  Change value in device memory
    da[50] = 0;

    //  Get host readable pointer to device memory
    bolt::BCKND::device_vector< int >::pointer pDa = da.data();

    EXPECT_EQ( 0, pDa[50] );
}

TEST( DeviceVector, Swap )
{
    bolt::BCKND::device_vector< int > dV( 5ul, 3 ), dV2(5ul, 10);
    EXPECT_EQ( 5, dV.size( ) );
    dV.swap(dV2);
    EXPECT_EQ(3, dV2[0]);
    EXPECT_EQ(10, dV[0]);

}
#if 1
TEST( DeviceVector, Assign )
{
    bolt::BCKND::device_vector< int > dV( 3, 98 );
    std::vector< int > stdV( 3, 98 );
    EXPECT_EQ(dV[0], stdV[0]);
    dV.assign(10, 89 );
    stdV.assign(10, 89 );
    for(int i=0; i<10; i++)
    {
        EXPECT_EQ(stdV[i], dV[i]);
    }
    dV.assign(5, 98 );
    stdV.assign(5, 98 );
    EXPECT_EQ(stdV.capacity(), dV.capacity());
    EXPECT_EQ(stdV.size(), dV.size());

    for(int i=0; i<5; i++)
    {
        EXPECT_EQ(stdV[i], dV[i]);
    }
    std::vector<float> stdFloatVect(1024);
    for(int i=0; i<1024; i++)
    {
        stdFloatVect[i] = (float)i;
    }
    bolt::BCKND::device_vector<float> dvFloatVect(stdFloatVect.begin(), 1024);
    for(int i=0; i<1024; i++)
    {
        EXPECT_FLOAT_EQ((float)i, dvFloatVect[i]);
    }
    dvFloatVect.assign(1022, 89 );
    stdFloatVect.assign(1022, 89 );
    for(int i=0; i<1022; i++)
    {
        /*This loop is iterating to 1022 elements only because stdFloatVect[i] throws an exception 
        if accessing beyond 1022 elements*/
        EXPECT_FLOAT_EQ(stdFloatVect[i], dvFloatVect[i]);
    }
}

TEST( DeviceVector, AssignIterator )
{
    bolt::BCKND::device_vector< int > dV( 1024 );
    std::vector< int > stdV( 1024 );
    for(int i=0; i<1024; i++)
    {
        stdV[i] = rand();
    }
    dV.assign(stdV.begin(), stdV.end() );
    for(int i=0; i<1024; i++)
    {
        EXPECT_EQ(stdV[i], dV[i]);
    }
}
#endif
TEST( VectorIterator, BackFront )
{
    std::vector< int > stdV( 1024 );
    for(int i=0; i<1024; i++)
    {
        stdV[i] = rand();
    }
    bolt::BCKND::device_vector< int > dV( stdV.begin(), stdV.end() );
    EXPECT_EQ( stdV.front(), dV.front( ) );
    EXPECT_EQ( stdV.back(),  dV.back( ) );
        
    std::cout << "max_size = " << stdV.max_size();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test fill() calls used in device_vector i.e. device_vector( ... ), resize( ), assign( )
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOLT_FUNCTOR(DUMMY,
struct DUMMY
{
    float a;
    float b;
    float c;

  bool operator == (const DUMMY &lhs) const
  {
    if(lhs.a == a && lhs.b == b && lhs.c == c) return true;
    else return false;
  }

  void operator = (const DUMMY &rhs)
  {
    a = rhs.a;
    b = rhs.b;
    c = rhs.c;
  }


};);

BOLT_FUNCTOR(Int_3,
struct Int_3
{
    int a;
    int b;
    int c;

  bool operator == (const Int_3 &lhs) const
  {
    if(lhs.a == a && lhs.b == b && lhs.c == c) return true;
    else return false;
  }

  void operator = (const Int_3 &rhs)
  {
    a = rhs.a;
    b = rhs.b;
    c = rhs.c;
  }


};);

BOLT_TEMPLATE_REGISTER_NEW_ITERATOR(bolt::cl::device_vector, int, DUMMY);
BOLT_TEMPLATE_REGISTER_NEW_ITERATOR(bolt::cl::device_vector, int, Int_3);


TEST(DeviceVectorFill, AssignFltUDD)
{

  DUMMY init;
  init.a = init.b = init.c = CL_M_PI_4_F;

  DUMMY glut;
  glut.a = CL_M_PI_F;
  glut.b = CL_M_LOG10E_F;
  glut.c = CL_M_LOG2E_F;

  bolt::cl::device_vector<DUMMY> dv(72,init);
  std::vector<DUMMY> hv(72, init);

  dv.assign(1024, glut);
  hv.assign(1024, glut);

  cmpArrays( hv, dv );

}

TEST(DeviceVectorFill, AssignFltUDDEmpty)
{

  DUMMY init;
  init.a = init.b = init.c = CL_M_PI_4_F;

  DUMMY glut;
  glut.a = CL_M_PI_F;
  glut.b = CL_M_LOG10E_F;
  glut.c = CL_M_LOG2E_F;

  bolt::cl::device_vector<DUMMY> dv;
  std::vector<DUMMY> hv;

  dv.assign(1023, glut);
  hv.assign(1023, glut);

  cmpArrays( hv, dv );

}

TEST(DeviceVectorFill, AssignFlt)
{

  float init;
  init = CL_M_PI_4_F;

  float glut;
  glut = CL_M_PI_F;

  bolt::cl::device_vector<float> dv(72,init);
  std::vector<float> hv(72, init);

  dv.assign(1024, glut);
  hv.assign(1024, glut);

  cmpArrays( hv, dv );

}

TEST(DeviceVectorFill, AssignFltEmpty)
{

  float init;
  init = CL_M_PI_4_F;

  float glut;
  glut = CL_M_PI_F;

  bolt::cl::device_vector<float> dv;
  std::vector<float> hv;

  dv.assign(1023, glut);
  hv.assign(1023, glut);

  cmpArrays( hv, dv );

}

TEST(DeviceVectorFill, device_vector_constructor)
{

  DUMMY init;
  init.a = init.b = init.c = FLT_EPSILON;

  bolt::cl::device_vector<DUMMY> dvec( 1024, init );
  std::vector<DUMMY> hvec( 1024, init );

  cmpArrays( hvec, dvec );


}

TEST(DeviceVectorFill, ResizeEmptyVector)
{

  bolt::cl::device_vector<DUMMY> dv;
  std::vector<DUMMY> hv;
  dv.resize(1000);
  hv.resize(1000);

  EXPECT_EQ( hv.size(), dv.size() );


}


TEST(DeviceVectorFill, ResizeEmptyVectorUDDWithValues)
{
  DUMMY init;
  init.a = init.b = init.c = FLT_EPSILON;

  bolt::cl::device_vector<DUMMY> dv;
  std::vector<DUMMY> hv;

  dv.resize(1000,init);
  hv.resize(1000,init);

  cmpArrays( hv, dv );


}

TEST(DeviceVectorFill, ResizeEmptyVectorFltWithValues)
{
  float init;
  init = FLT_EPSILON;

  bolt::cl::device_vector<float> dv;
  std::vector<float> hv;

  dv.resize(1000,init);
  hv.resize(1000,init);

  cmpArrays( hv, dv );


}

TEST(DeviceVectorFill, Resize)
{

  bolt::cl::device_vector<DUMMY> dv(10);
  std::vector<DUMMY> hv(10);

  dv.resize(1000); 
  hv.resize(1000);

  EXPECT_EQ( hv.size(), dv.size() );


}

TEST(DeviceVectorFill, ResizeFloatUDDWithValues)
{
  DUMMY init;
  init.a = init.b = init.c = CL_M_PI_4_F;

  DUMMY glut;
  glut.a = CL_M_PI_F;
  glut.b = CL_M_LOG10E_F;
  glut.c = CL_M_LOG2E_F;

  bolt::cl::device_vector<DUMMY> dv(72,init);
  std::vector<DUMMY> hv(72, init);

  dv.resize(1024, glut);
  hv.resize(1024, glut);

  cmpArrays( hv, dv );


}

TEST(DeviceVectorFill, ResizeFloatWithValues)
{
  float init;
  init =  CL_M_PI_4_F;

  float glut;
  glut =  CL_M_PI_F;

  bolt::cl::device_vector<float> dv(72,init);
  std::vector<float> hv(72, init);

  dv.resize(1024, glut);
  hv.resize(1024, glut);

  cmpArrays( hv, dv );


}


TEST(DeviceVectorFill, ResizeWithIntUDDValues)
{
  Int_3 init;
  init.a = init.b = init.c = 10;

  Int_3 glut;
  glut.a = 23;
  glut.b = 23;
  glut.c = 23;

  bolt::cl::device_vector<Int_3> dv(72,init);
  std::vector<Int_3> hv(72, init);

  dv.resize(1024, glut);
  hv.resize(1024, glut);

  {
    bolt::cl::device_vector< Int_3 >::pointer tst =  dv.data( );
    std::vector<Int_3> hvt(1024);

    for(unsigned int i = 0; i < 1024 ; i++ ) 
      hvt[i] = tst[i];

  }
  cmpArrays( hv, dv );


}

TEST(DeviceVectorFill, ResizeWithIntValues)
{
  int init;
  init = 10;

  int glut;
  glut = 23;

  bolt::cl::device_vector<int> dv(72,init);
  std::vector<int> hv(72, init);

  dv.resize(1024, glut);
  hv.resize(1024, glut);

  cmpArrays( hv, dv );


}


//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class FillUDDFltVector: public ::testing::TestWithParam< int >
{
    

public:

    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    FillUDDFltVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
      DUMMY init_pi;
      init_pi.a = init_pi.b = init_pi.c = CL_M_PI_F; 
      std::fill(stdInput.begin(), stdInput.end(), init_pi);
      bolt::cl::fill(boltInput.begin(), boltInput.end(), init_pi);
    }

protected:
    std::vector< DUMMY > stdInput;
    bolt::cl::device_vector< DUMMY > boltInput;
};



TEST_P(FillUDDFltVector, VariableSizeResizeWithValues)
{

  DUMMY glut;
  glut.a = CL_M_PI_F;
  glut.b = CL_M_LOG10E_F;
  glut.c = CL_M_LOG2E_F;

  boltInput.resize(1024, glut);
  stdInput.resize(1024, glut);

  cmpArrays( stdInput, boltInput );


}

TEST_P(FillUDDFltVector, VariableAssignFlt)
{

  DUMMY glut;
  glut.a = CL_M_PI_F;
  glut.b = CL_M_LOG10E_F;
  glut.c = CL_M_LOG2E_F;

  boltInput.assign(1024, glut);
  stdInput.assign(1024, glut);

  cmpArrays( stdInput, boltInput );

}


INSTANTIATE_TEST_CASE_P( VariableSizeResizeWithValues, FillUDDFltVector, ::testing::Range( 0, 1048576, 4096 ) );
INSTANTIATE_TEST_CASE_P( VariableAssignFlt, FillUDDFltVector, ::testing::Range( 0, 1048576, 4096 ) );

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class FillUDDIntVector: public ::testing::TestWithParam< int >
{
    

public:

    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    FillUDDIntVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
      Int_3 init_;
      init_.a = init_.b = init_.c = 33; 
      std::fill(stdInput.begin(), stdInput.end(), init_);
      bolt::cl::fill(boltInput.begin(), boltInput.end(), init_);
    }

protected:
    std::vector< Int_3 > stdInput;
    bolt::cl::device_vector< Int_3  > boltInput;
};

TEST_P(FillUDDIntVector, VariableSizeResizeWithValues)
{

  Int_3 glut;
  glut.a =9115123;
  glut.b =9117123;
  glut.c =112123;

  boltInput.resize(1024, glut);
  stdInput.resize(1024, glut);

  cmpArrays( stdInput, boltInput );


}

TEST_P(FillUDDIntVector, VariableAssignInt)
{

  Int_3 glut;
  glut.a =9115123;
  glut.b =9117123;
  glut.c =112123;

  boltInput.assign(1024, glut);
  stdInput.assign(1024, glut);

  cmpArrays( stdInput, boltInput );


}


INSTANTIATE_TEST_CASE_P( VariableSizeResizeWithValues, FillUDDIntVector, ::testing::Range( 0, 1048576, 4096 ) );

TEST(BUG, BUG398791)
{
    int length = 100;
    bolt::cl::device_vector<int> dv1(length);
    bolt::cl::device_vector<int> dv2(length);

    for(int i=0; i<length; i++)
    {
        dv1[i] = i;
        dv2[i] = dv1[i];
        EXPECT_EQ( dv2[i], i );
    }
}

int _tmain(int argc, _TCHAR* argv[])
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
