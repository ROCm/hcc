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
#define TEST_MULTICORE_TBB_SORT 1
#define TEST_LARGE_BUFFERS 1
#define TEST_OFFSET_BUFFERS 1 
#define GOOGLE_TEST 1
#define BKND amp 
#define SORT_FUNC stable_sort



#include "common/stdafx.h"
#include "bolt/amp/stablesort.h"
#include "bolt/unicode.h"
#include "bolt/amp/functional.h"

#include <gtest/gtest.h>
#include <type_traits>

#include "common/stdafx.h"
#include "common/test_common.h"
#include "bolt/miniDump.h"

#include <array>
#include <algorithm>



#if (GOOGLE_TEST == 1)

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
//This is a compare routine for naked pointers.


#if ( TEST_DOUBLE == 1)
// UDD which contains four doubles
struct uddtD4
{
    double a;
    double b;
    double c;
    double d;

    uddtD4() restrict(cpu, amp) {}
    uddtD4(double x, double y, double z, double w) restrict(cpu, amp)
     : a(x), b(y), c(z), d(w) {}
    bool operator==(const uddtD4& rhs) const restrict(amp,cpu)
    {
        bool equal = true;
        double th = 0.0000000001;
        if (rhs.a < th && rhs.a > -th)
            equal = ( (1.0*a - rhs.a) < th && (1.0*a - rhs.a) > -th) ? equal : false;
        else
            equal = ( (1.0*a - rhs.a)/rhs.a < th && (1.0*a - rhs.a)/rhs.a > -th) ? equal : false;
        if (rhs.b < th && rhs.b > -th)
            equal = ( (1.0*b - rhs.b) < th && (1.0*b - rhs.b) > -th) ? equal : false;
        else
            equal = ( (1.0*b - rhs.b)/rhs.b < th && (1.0*b - rhs.b)/rhs.b > -th) ? equal : false;
        if (rhs.c < th && rhs.c > -th)
            equal = ( (1.0*c - rhs.c) < th && (1.0*c - rhs.c) > -th) ? equal : false;
        else
            equal = ( (1.0*c - rhs.c)/rhs.c < th && (1.0*c - rhs.c)/rhs.c > -th) ? equal : false;
        if (rhs.d < th && rhs.d > -th)
            equal = ( (1.0*d - rhs.d) < th && (1.0*d - rhs.d) > -th) ? equal : false;
        else
            equal = ( (1.0*d - rhs.d)/rhs.d < th && (1.0*d - rhs.d)/rhs.d > -th) ? equal : false;
        return equal;
    }
};

// Adds all four double elements and returns true if lhs_sum > rhs_sum

struct AddD4
{
    bool operator()(const uddtD4 &lhs, const uddtD4 &rhs) const restrict(amp,cpu)
    {

        if( ( lhs.a + lhs.b + lhs.c + lhs.d ) > ( rhs.a + rhs.b + rhs.c + rhs.d) )
            return true;
        return false;
    };
}; 


uddtD4 identityAddD4 = uddtD4( 1.0, 1.0, 1.0, 1.0 );
uddtD4 initialAddD4  = uddtD4( 1.00001, 1.000003, 1.0000005, 1.00000007 );

//This test cse fails
TEST(StableSortUDD_largepower_test, AddDouble4)
{
    //setup containers
    int length = 33554432; // 2^25
    bolt::amp::device_vector< uddtD4 > input(  length, initialAddD4, true  );
    std::vector< uddtD4 > refInput( length, initialAddD4 );

    // call sort
    AddD4 ad4gt;
    bolt::BKND::SORT_FUNC(input.begin(),    input.end(), ad4gt);
    std::SORT_FUNC( refInput.begin(), refInput.end(), ad4gt );

    // compare results
    cmpArrays(refInput, input);
}

TEST(StableSortUDD, AddDouble4)
{
    //setup containers
    int length = (1<<8);
    bolt::amp::device_vector< uddtD4 > input(  length, initialAddD4, true  );
    std::vector< uddtD4 > refInput( length, initialAddD4 );

    // call sort
    AddD4 ad4gt;
    bolt::BKND::SORT_FUNC(input.begin(),    input.end(), ad4gt);
    std::SORT_FUNC( refInput.begin(), refInput.end(), ad4gt );

    // compare results
    cmpArrays(refInput, input);
}

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
class StableSortArrayTest: public ::testing::Test
{
public:
    StableSortArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        boltInput = stdInput;
        stdOffsetIn = stdInput;
        boltOffsetIn = stdInput;
    };

    virtual void TearDown( )
    {};

    virtual ~StableSortArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    std::array< ArrayType, ArraySize > stdInput, boltInput, stdOffsetIn, boltOffsetIn;
    int m_Errors;
};

TYPED_TEST_CASE_P( StableSortArrayTest );


#if (TEST_MULTICORE_TBB_SORT == 1)

#if ( TEST_DOUBLE == 1)
#if defined( ENABLE_TBB )
TEST(MultiCoreCPU, MultiCoreAddDouble4)
{
    //setup containers
    int length = (1<<8);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    bolt::amp::device_vector< uddtD4 > input(  length, initialAddD4, true  );
    std::vector< uddtD4 > refInput( length, initialAddD4 );
    
    // call sort
    AddD4 ad4gt;
    bolt::BKND::SORT_FUNC(input.begin(), input.end(), ad4gt);
    std::SORT_FUNC( refInput.begin(), refInput.end(), ad4gt );

    // compare results
    cmpArrays(refInput, input);
}
#endif
#endif

float func()
{
    return (float)(rand() * rand() * rand() );
}

#if(TEST_LARGE_BUFFERS == 1)
TEST( DefaultGPU, Normal )
{
    int length = 1<<21;
    std::vector< float > stdInput( length, 0.0 );

    std::generate(stdInput.begin(), stdInput.end(), func );
    bolt::amp::device_vector< float > boltInput(  stdInput.begin(), stdInput.end()  );

    bolt::amp::control ctl = bolt::amp::control::getDefault( );

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ));
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< float >::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    bolt::amp::device_vector< float >::difference_type boltNumElements = std::distance( boltInput.begin( ),
                                                                                        boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    
    //cmpArrays( stdInput, boltInput );
    for(int i=1;i<length;i= i<<1)
        EXPECT_FLOAT_EQ( stdInput[i], boltInput[i] );
}
#endif

/* TEST( MultiCoreCPU, MultiCoreNormal )
{
    int length = 1025;
    bolt::cl::device_vector< float > boltInput(  length, 0.0, CL_MEM_READ_WRITE, true  );
    std::vector< float > stdInput( length, 0.0 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ));
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< float >::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    bolt::cl::device_vector< float >::difference_type boltNumElements = std::distance( boltInput.begin( ), 
                                                                                        boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
} */
#endif

/*
TEST( SerialCPU, SerialNormal )
{
    int length = 1025;
    bolt::cl::device_vector< float > boltInput(  length, 0.0, CL_MEM_READ_WRITE, true  );
    std::vector< float > stdInput( length, 0.0 );

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ));
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< float >::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    bolt::cl::device_vector< float >::difference_type boltNumElements = std::distance( boltInput.begin( ), 
                                                                                        boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
} */

TYPED_TEST_P( StableSortArrayTest, Normal )
{
    typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    
    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
    bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex );
        bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}

TYPED_TEST_P( StableSortArrayTest, GreaterFunction )
{
    typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ), std::greater< ArrayType >() );
    
    bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::amp::greater< ArrayType >( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    
    //OFFSET Test cases
    //  Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, std::greater< ArrayType >() );
        bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, bolt::amp::greater< ArrayType >( )  );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}


TYPED_TEST_P( StableSortArrayTest, LessFunction )
{
        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ), std::less<ArrayType>());
    bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::amp::less<ArrayType>() );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, std::less< ArrayType >() );
        bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, bolt::amp::less< ArrayType >( )  );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}

REGISTER_TYPED_TEST_CASE_P( StableSortArrayTest, Normal, 
                                           GreaterFunction,
                                           LessFunction);

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class StableSortIntegerVector: public ::testing::TestWithParam< int >
{
public:

    StableSortIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        boltInput = stdInput;
    }

protected:
    std::vector< int > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortFloatVector: public ::testing::TestWithParam< int >
{
public:
    StableSortFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        boltInput = stdInput;    
    }

protected:
    std::vector< float > stdInput, boltInput;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortDoubleVector: public ::testing::TestWithParam< int >
{
public:
    StableSortDoubleVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        boltInput = stdInput;    
    }

protected:
    std::vector< double > stdInput, boltInput;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
     //Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortIntegerDeviceVector( ): stdInput( GetParam( ) ), 
                                stdOffsetIn( GetParam( ) ), ArraySize ( GetParam( ) )
								//boltInput( static_cast<size_t>(GetParam( )) ), boltOffsetIn( static_cast<size_t>(GetParam( ) ))
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            //boltInput[i] = stdInput[i];
            //boltOffsetIn[i] = stdInput[i];
            stdOffsetIn[i] = stdInput[i];
        }
    }

protected:
    std::vector< int > stdInput, stdOffsetIn;
    //bolt::amp::device_vector< int > boltInput, boltOffsetIn;
    const int ArraySize;
};


//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortFloatDeviceVector( ): stdInput( GetParam( ) ), 
                              stdOffsetIn( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            //boltInput[i] = stdInput[i];
            //boltOffsetIn[i] = stdInput[i];
            stdOffsetIn[i] = stdInput[i];
        }
    }

protected:
    std::vector< float > stdInput, stdOffsetIn;
    //bolt::cl::device_vector< float > boltInput, boltOffsetIn;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortDoubleDeviceVector( ): stdInput( GetParam( ) ) ,
                               stdOffsetIn( GetParam( ) ) 
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            //boltInput[i] = stdInput[i];
            //boltOffsetIn[i] = stdInput[i];
            stdOffsetIn[i] = stdInput[i];
        }
    }

protected:
    std::vector< double > stdInput, stdOffsetIn;
    //bolt::cl::device_vector< double > boltInput, boltOffsetIn;
};
#endif


/********* Test case to reproduce SuiCHi bugs ******************/
struct UDD { 
    int a; 
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) const restrict(amp,cpu){ 
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    } 
    bool operator < (const UDD& other) const restrict(amp,cpu){ 
        return ((a+b) < (other.a+other.b));
    }
    bool operator > (const UDD& other) const restrict(amp,cpu){ 
        return ((a+b) > (other.a+other.b));
    }
    bool operator == (const UDD& other) const restrict(amp,cpu){ 
        return ((a+b) == (other.a+other.b));
    }
    UDD() 
        : a(0),b(0) { } 
    UDD(int _in) 
        : a(_in), b(_in +1)  { } 
}; 


    struct sortBy_UDD_a {
        bool operator() (const UDD& a, const UDD& b) const restrict(amp,cpu)
        { 
            return (a.a>b.a); 
        };
    };

    struct sortBy_UDD_b {
        bool operator() (UDD& a, UDD& b) const restrict(amp,cpu)
        { 
            return (a.b>b.b); 
        };
    };

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortUDDDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortUDDDeviceVector( ): stdInput( GetParam( ) ),
                            stdOffsetIn( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            //boltInput[i] = stdInput[i];
            //boltOffsetIn[i] = stdInput[i];
            stdOffsetIn[i] = stdInput[i];
        }
    }

protected:
    std::vector< UDD > stdInput,stdOffsetIn;
    //bolt::cl::device_vector< UDD > boltInput,boltOffsetIn;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortIntegerNakedPointer( ): stdInput( new int[ GetParam( ) ] ), boltInput( new int[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, rand);
        for (size_t i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
    };

protected:
     int* stdInput;
     int* boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortFloatNakedPointer( ): stdInput( new float[ GetParam( ) ] ), boltInput( new float[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, rand);
        for (size_t i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
    };

protected:
     float* stdInput;
     float* boltInput;
};

#if (TEST_DOUBLE ==1 )
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortDoubleNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortDoubleNakedPointer( ): stdInput( new double[ GetParam( ) ] ), boltInput( new double[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, rand);

        for( size_t i=0; i < size; i++ )
        {
            boltInput[ i ] = stdInput[ i ];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
    };

protected:
     double* stdInput;
     double* boltInput;
};
#endif

class StableSortCountingIterator :public ::testing::TestWithParam<int>{
protected:
     int mySize;
public:
    StableSortCountingIterator(): mySize(GetParam()){
    }
};

//StableSort with Fancy Iterator would result in compilation error!

/* TEST_P(StableSortCountingIterator, withCountingIterator)
{
    std::vector<int> a(mySize);

    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first + mySize;
    
    for(int i=0; i< a.size() ; i++)
    {
        a[i] = i;
    }

    std::sort( a.begin( ), a.end( ));
    bolt::BKND::SORT_FUNC( first, last); // This is logically wrong!

    cmpArrays( a, first);

} */


TEST_P( StableSortIntegerVector, Normal )
{
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin(),stdInput.end());
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( StableSortIntegerVector, SerialCPU )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC(ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end( ) );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( StableSortIntegerVector, MultiCoreCPU )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC(ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end( ) );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
// Come Back here
TEST_P( StableSortFloatVector, Normal )
{
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( StableSortFloatVector, SerialCPU)
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( StableSortFloatVector, MultiCoreCPU)
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< float >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
#if (TEST_DOUBLE == 1)
TEST_P( StableSortDoubleVector, Inplace )
{
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< double >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( StableSortDoubleVector, SerialInplace )
{
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< double >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( StableSortDoubleVector, MulticoreInplace )
{
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< double >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
#endif
#if (TEST_DEVICE_VECTOR == 1)
TEST_P( StableSortIntegerDeviceVector, Inplace )
{
    bolt::amp::device_vector< int > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< int > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
     
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );



    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end());
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

	#if (TEST_OFFSET_BUFFERS == 1)  
    //OFFSET Test cases
    //  Calling the actual functions under test
    typedef std::vector< int >::value_type valtype;
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
    #endif
}

TEST_P( StableSortIntegerDeviceVector, SerialInplace )
{
    bolt::amp::device_vector< int > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< int > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC(ctl,  boltInput.begin( ), boltInput.end( ) );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    typedef std::vector< int >::value_type valtype;
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}
#if defined( ENABLE_TBB )
TEST_P( StableSortIntegerDeviceVector, MultiCoreInplace )
{
    bolt::amp::device_vector< int > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< int > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC(ctl,  boltInput.begin( ), boltInput.end( ) );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    typedef std::vector< int >::value_type valtype;
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}
#endif

TEST_P( StableSortUDDDeviceVector, Inplace )
{
    bolt::amp::device_vector< UDD > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< UDD > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< UDD >::value_type valtype;
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end());
    std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test

    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortUDDDeviceVector, SerialInplace )
{
    bolt::amp::device_vector< UDD > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< UDD > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< UDD >::value_type valtype;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC(ctl,  boltInput.begin( ), boltInput.end( ) );

    std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test

    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}
#if defined( ENABLE_TBB )
TEST_P( StableSortUDDDeviceVector, MultiCoreInplace )
{
    bolt::amp::device_vector< UDD > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< UDD > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< UDD >::value_type valtype;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC(ctl,  boltInput.begin( ), boltInput.end( ) );

    std::vector< UDD >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< UDD >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}
#endif

TEST_P( StableSortFloatDeviceVector, Inplace )
{
    bolt::amp::device_vector< float > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< float > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< float >::value_type valtype;
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ),stdInput.end());
    std::vector< valtype >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test

    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortFloatDeviceVector, SerialInplace )
{
    bolt::amp::device_vector< float > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< float > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< float >::value_type valtype;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin(),stdInput.end());
    std::vector< valtype >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test

    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}
#if defined( ENABLE_TBB )
TEST_P( StableSortFloatDeviceVector, MultiCoreInplace )
{
    bolt::amp::device_vector< float > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< float > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< float >::value_type valtype;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin(),stdInput.end());
    std::vector< valtype >::iterator::difference_type boltNumElements = std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}
#endif

#if (TEST_DOUBLE == 1)
TEST_P( StableSortDoubleDeviceVector, Inplace )
{
    bolt::amp::device_vector< double > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< double > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< double >::value_type valtype;
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin(),stdInput.end());
    std::vector< valtype >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortDoubleDeviceVector, SerialInplace )
{
    bolt::amp::device_vector< double > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< double > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
     
    typedef std::vector< double >::value_type valtype;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin(),stdInput.end());
    std::vector< valtype >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

#if defined( ENABLE_TBB )
TEST_P( StableSortDoubleDeviceVector, MulticoreInplace )
{
    bolt::amp::device_vector< double > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::amp::device_vector< double > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
     
    typedef std::vector< double >::value_type valtype;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin(),stdInput.end());
    std::vector< valtype >::iterator::difference_type boltNumElements =std::distance(boltInput.begin(),boltInput.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some aribitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, std::less< valtype >() );
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::amp::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}
#endif
#endif
#endif
#if defined(_WIN32)
TEST_P( StableSortIntegerNakedPointer, Inplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( StableSortIntegerNakedPointer, SerialInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( ctl, wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#if defined( ENABLE_TBB )
TEST_P( StableSortIntegerNakedPointer, MultiCoreInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( ctl, wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif

TEST_P( StableSortFloatNakedPointer, Inplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );

    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( StableSortFloatNakedPointer, SerialInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );

    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( ctl, wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#if defined( ENABLE_TBB )
TEST_P( StableSortFloatNakedPointer, MultiCoreInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );

    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( ctl, wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif

#if (TEST_DOUBLE == 1)
TEST_P( StableSortDoubleNakedPointer, Inplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );

    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( StableSortDoubleNakedPointer, SerialInplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#if defined( ENABLE_TBB )
TEST_P( StableSortDoubleNakedPointer, MulticoreInplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif
#endif

#endif
//std::array<int, 12> TestValues = {2,4,8,16,32,64,128,256,512,1024,2048,4096}; // 2^1 to 2^12
//std::array<int, 5> TestValues2 = {4096, 8192,16384,32768, 65536};  // 2^12 to 2^16
//std::array<int, 6> TestValues3 = {65536, 131072, 262144, 524288, 1048576, 2097152}; // 2^16 to 2^21
//std::array<int, 8> TestValues4 = {2, 8, 64, 512, 4096,32768, 262144, 2097152}; // 2^1 to 2^21 in steps of 2^3

std::array<int, 15> TestValues = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};

//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^22
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                                                                            TestValues.end() ) );

                                                                            
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16															
//INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortFloatVector, ::testing::ValuesIn( TestValues2.begin(), 
//                                                                        TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortFloatVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                        TestValues.end() ) );																	
                                                                        
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
//INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortDoubleVector, ::testing::ValuesIn( TestValues3.begin(), 
//                                                                            TestValues3.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortDoubleVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                            TestValues.end() ) );																	
#endif
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );
                                                                                
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortUDDDeviceVector, ::testing::Range( 1, 32768, 3276  ) ); // 1 to 2^15
//INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortUDDDeviceVector, ::testing::ValuesIn( TestValues4.begin(), 
//                                                                                TestValues4.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortUDDDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );
                                                                                
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortFloatDeviceVector, ::testing::Range( 1, 32768, 3276  ) ); // 1 to 2^15
//INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortFloatDeviceVector, ::testing::ValuesIn( TestValues2.begin(),
//                                                                                TestValues2.end()));
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                TestValues.end()));                                                                               
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortDoubleDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
//INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortDoubleDeviceVector, ::testing::ValuesIn(TestValues3.begin(),
//                                                                                    TestValues3.end()));
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortDoubleDeviceVector, ::testing::ValuesIn(TestValues.begin(),
                                                                                    TestValues.end()));
#endif
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortIntegerNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                                    TestValues.end()));
                                                                       
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortFloatNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
//INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortFloatNakedPointer, ::testing::ValuesIn( TestValues2.begin(), 
//                                                                                TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );                                                                               
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortDoubleNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
//INSTANTIATE_TEST_CASE_P( StableSort, StableSortDoubleNakedPointer, ::testing::ValuesIn( TestValues3.begin(),
//                                                                            TestValues3.end() ) );
INSTANTIATE_TEST_CASE_P( StableSort, StableSortDoubleNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                            TestValues.end() ) );																		
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
    std::tuple< int, TypeValue< 4096 > >,
    std::tuple< int, TypeValue< 4097 > >
    #if (TEST_LARGE_BUFFERS == 1)
    , /*This coma is needed*/
    std::tuple< int, TypeValue< 8192 > >,
    std::tuple< int, TypeValue< 16384 > >,//13
    std::tuple< int, TypeValue< 32768 > >,//14
    std::tuple< int, TypeValue< 65535 > >,//15
    std::tuple< int, TypeValue< 65536 > >,//16
    std::tuple< int, TypeValue< 131072 > >,//17    
    std::tuple< int, TypeValue< 262144 > >,//18    
    std::tuple< int, TypeValue< 524288 > >,//19    
    std::tuple< int, TypeValue< 1048576 > >,//20    
    std::tuple< int, TypeValue< 2097152 > >,//21    
    std::tuple< int, TypeValue< 4194304 > >,//22    
    std::tuple< int, TypeValue< 8388608 > >,//23
    std::tuple< int, TypeValue< 16777216 > >,//24
    std::tuple< int, TypeValue< 33554432 > >,//25
    std::tuple< int, TypeValue< 67108864 > >//26
#endif
> IntegerTests;

typedef ::testing::Types< 
    std::tuple< unsigned int, TypeValue< 1 > >,
    std::tuple< unsigned int, TypeValue< 31 > >,
    std::tuple< unsigned int, TypeValue< 32 > >,
    std::tuple< unsigned int, TypeValue< 63 > >,
    std::tuple< unsigned int, TypeValue< 64 > >,
    std::tuple< unsigned int, TypeValue< 127 > >,
    std::tuple< unsigned int, TypeValue< 128 > >,
    std::tuple< unsigned int, TypeValue< 129 > >,
    std::tuple< unsigned int, TypeValue< 1000 > >,
    std::tuple< unsigned int, TypeValue< 1053 > >,
    std::tuple< unsigned int, TypeValue< 4096 > >,
    std::tuple< unsigned int, TypeValue< 4097 > >
    #if (TEST_LARGE_BUFFERS == 1)
    , /*This coma is needed*/
    std::tuple< unsigned int, TypeValue< 8192 > >,
    std::tuple< unsigned int, TypeValue< 16384 > >,//13
    std::tuple< unsigned int, TypeValue< 32768 > >,//14
    std::tuple< unsigned int, TypeValue< 65535 > >,//15
    std::tuple< unsigned int, TypeValue< 65536 > >,//16
    std::tuple< unsigned int, TypeValue< 131072 > >,//17    
    std::tuple< unsigned int, TypeValue< 262144 > >,//18    
    std::tuple< unsigned int, TypeValue< 524288 > >,//19    
    std::tuple< unsigned int, TypeValue< 1048576 > >,//20    
    std::tuple< unsigned int, TypeValue< 2097152 > >,//21    
    std::tuple< unsigned int, TypeValue< 4194304 > >,//22    
    std::tuple< unsigned int, TypeValue< 8388608 > >,//23
    std::tuple< unsigned int, TypeValue< 16777216 > >,//24
    std::tuple< unsigned int, TypeValue< 33554432 > >,//25
    std::tuple< unsigned int, TypeValue< 67108864 > >//26
#endif

> UnsignedIntegerTests;

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
    std::tuple< float, TypeValue< 4097 > >
    #if (TEST_LARGE_BUFFERS == 1)
    , /*This coma is needed*/
    std::tuple< float, TypeValue< 65535 > >,
    std::tuple< float, TypeValue< 65536 > >
    #endif
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
    std::tuple< double, TypeValue< 4097 > >
    #if (TEST_LARGE_BUFFERS == 1)
    , /*This coma is needed*/
    std::tuple< double, TypeValue< 65535 > >,
    std::tuple< double, TypeValue< 65536 > >
    #endif
> DoubleTests;
#endif 



template< typename ArrayTuple >
class StableSortUDDArrayTest: public ::testing::Test
{
public:
    StableSortUDDArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        boltInput = stdInput;
    };

    virtual void TearDown( )
    {};

    virtual ~StableSortUDDArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType, ArraySize > stdInput, boltInput;
    int m_Errors;
};

TYPED_TEST_CASE_P( StableSortUDDArrayTest );

TYPED_TEST_P( StableSortUDDArrayTest, Normal )
{
    typedef typename StableSortUDDArrayTest<gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortUDDArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    
    //  Calling the actual functions under test

    std::SORT_FUNC( StableSortUDDArrayTest<gtest_TypeParam_ >::stdInput.begin( ), StableSortUDDArrayTest<gtest_TypeParam_ >::stdInput.end( ), UDD() );
    bolt::BKND::SORT_FUNC( StableSortUDDArrayTest<gtest_TypeParam_ >::boltInput.begin( ), StableSortUDDArrayTest<gtest_TypeParam_ >::boltInput.end( ), UDD() );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortUDDArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortUDDArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortUDDArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortUDDArrayTest< gtest_TypeParam_ >::boltInput.end() );
    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortUDDArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortUDDArrayTest< gtest_TypeParam_ >::stdInput, StableSortUDDArrayTest< gtest_TypeParam_ >::boltInput );

    std::SORT_FUNC( StableSortUDDArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortUDDArrayTest< gtest_TypeParam_ >::stdInput.end( ), sortBy_UDD_a() );
    bolt::BKND::SORT_FUNC( StableSortUDDArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortUDDArrayTest< gtest_TypeParam_ >::boltInput.end( ), sortBy_UDD_a() );

    stdNumElements = std::distance( StableSortUDDArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortUDDArrayTest< gtest_TypeParam_ >::stdInput.end() );
    boltNumElements = std::distance( StableSortUDDArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortUDDArrayTest< gtest_TypeParam_ >::boltInput.end() );
    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortUDDArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortUDDArrayTest< gtest_TypeParam_ >::stdInput, StableSortUDDArrayTest< gtest_TypeParam_ >::boltInput );

}

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
    std::tuple< UDD, TypeValue< 4097 > >
    #if (TEST_LARGE_BUFFERS == 1)
    , /*This coma is needed*/
    std::tuple< UDD, TypeValue< 65535 > >,
    std::tuple< UDD, TypeValue< 65536 > >
    #endif
> UDDTests;

//INSTANTIATE_TYPED_TEST_CASE_P( clLong, StableSortArrayTest, clLongTests );
INSTANTIATE_TYPED_TEST_CASE_P( Integer, StableSortArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( UnsignedInteger, StableSortArrayTest, UnsignedIntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, StableSortArrayTest, FloatTests );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TYPED_TEST_CASE_P( Double, StableSortArrayTest, DoubleTests );
#endif 
REGISTER_TYPED_TEST_CASE_P( StableSortUDDArrayTest,  Normal);
INSTANTIATE_TYPED_TEST_CASE_P( UDDTest, StableSortUDDArrayTest, UDDTests );

class withStdVect: public ::testing::TestWithParam<int>{
protected:
    int sizeOfInputBuffer;
public:
    withStdVect():sizeOfInputBuffer(GetParam()){
    }
};



INSTANTIATE_TEST_CASE_P(sortDescending, withStdVect, ::testing::Range(50, 100, 1));

TEST (sanity_sort__withBoltClDevVectDouble_epr, floatSerial){
    int sizeOfInputBufer = 64; //test case is failing for all values greater than 32
    std::vector<double>  stdVect(0);
    bolt::amp::device_vector<double>  boltVect(0);

    for (int i = 0 ; i < sizeOfInputBufer; i++){
        double dValue = rand();
        dValue = dValue/rand();
        dValue = dValue*rand();
        stdVect.push_back(dValue);
        boltVect.push_back(dValue);
    }
    std::SORT_FUNC(stdVect.begin(), stdVect.end(), std::greater<double>( ) );
    bolt::BKND::SORT_FUNC(boltVect.begin(), boltVect.end(), bolt::amp::greater<double>( ) );

    for (int i = 0 ; i < sizeOfInputBufer; i++){
        EXPECT_DOUBLE_EQ(stdVect[i], boltVect[i]);
    }
}

TEST (rawArrayTest, floatarray){
    const int sizeOfInputBufer = 8192; //test case is failing for all values greater than 32
    float  stdArray[sizeOfInputBufer];
    float  boltArray[sizeOfInputBufer];
    float  backupArray[sizeOfInputBufer];

    for (int i = 0 ; i < sizeOfInputBufer; i++){
        float fValue = (float)rand();
        fValue = fValue/rand();
        fValue = fValue*rand()*rand();
        stdArray[i] = boltArray[i] = fValue;
    }
    std::SORT_FUNC( stdArray, stdArray+sizeOfInputBufer, std::greater<float>( ) );
    bolt::BKND::SORT_FUNC( boltArray, boltArray+sizeOfInputBufer, bolt::amp::greater<float>( ) );

    for (int i = 0 ; i < sizeOfInputBufer; i++)
    {
        EXPECT_FLOAT_EQ(stdArray[i], boltArray[i]);
    }

    //Offset tests 
    for (int i = 0 ; i < sizeOfInputBufer; i++)
    {
       stdArray[i] = boltArray[i] = backupArray[i];
    }

    std::SORT_FUNC( stdArray+17, stdArray+sizeOfInputBufer-129, std::greater<float>( ) );
    bolt::BKND::SORT_FUNC( boltArray, boltArray+sizeOfInputBufer, bolt::amp::greater<float>( ) );

    for (int i = 0 ; i < sizeOfInputBufer; i++)
    {
        EXPECT_FLOAT_EQ(stdArray[i], boltArray[i]);
    }

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
    std::cout << "Test Completed. Press Enter to exit.\n .... ";
    //getchar();
    return retVal;
}



#else

#include "bolt/amp/iterator/counting_iterator.h"
#include <bolt/amp/sort.h>
#include <bolt/amp/functional.h>
#include <array>
#include <algorithm>
int main ()
{
    const int ArraySize = 8192;
    typedef std::array< int, ArraySize > ArrayCont;
    ArrayCont stdOffsetIn,stdInput;
    ArrayCont boltOffsetIn,boltInput;
    std::generate(stdInput.begin(), stdInput.end(), rand);
    boltInput = stdInput;
    boltOffsetIn = stdInput;
    stdOffsetIn = stdInput;

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    //  Loop through the array and compare all the values with each other
    for(int i=0;i< ArraySize;i++)
    {
        if(stdInput[i] = boltInput[i])
            continue;
        else 
            std::cout << "Failed at i " << i << " -- stdInput[i] " << stdInput[i] << " boltInput[i] = " << boltInput[i] << "\n";
    }
    //  OFFSET Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex );
        bolt::BKND::SORT_FUNC( boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex );

        //  Loop through the array and compare all the values with each other
        for(int i=0;i< ArraySize;i++)
        {
            if(stdOffsetIn[i] = boltOffsetIn[i])
                continue;
            else 
                std::cout << "Failed at i " << i << " -- stdInput[i] " << stdInput[i] << " boltInput[i] = " << boltInput[i] << "\n";
        }
    }


}
#endif
