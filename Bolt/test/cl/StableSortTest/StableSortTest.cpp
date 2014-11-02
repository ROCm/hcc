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
#define TEST_LARGE_BUFFERS 0
#define GOOGLE_TEST 1
#define BKND cl 
#define SORT_FUNC stable_sort

#if (GOOGLE_TEST == 1)
#include <gtest/gtest.h>
#include "common/stdafx.h"
#include "common/myocl.h"
#include "common/test_common.h"
#include "bolt/cl/iterator/counting_iterator.h"

#include <bolt/cl/stablesort.h>
#include <bolt/miniDump.h>
//#include <bolt/unicode.h>
#include <bolt/cl/functional.h>

#include <boost/shared_array.hpp>
#include <array>
#include <algorithm>
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
//This is a compare routine for naked pointers.

#if ( TEST_DOUBLE == 1)
// UDD which contains four doubles
BOLT_FUNCTOR(uddtD4,
struct uddtD4
{
    double a;
    double b;
    double c;
    double d;

    bool operator==(const uddtD4& rhs) const
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
);
// Functor for UDD. Adds all four double elements and returns true if lhs_sum > rhs_sum
BOLT_FUNCTOR(AddD4,
struct AddD4
{
    bool operator()(const uddtD4 &lhs, const uddtD4 &rhs) const
    {

        if( ( lhs.a + lhs.b + lhs.c + lhs.d ) > ( rhs.a + rhs.b + rhs.c + rhs.d) )
            return true;
        return false;
    };
}; 
);
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< AddD4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< AddD4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

uddtD4 identityAddD4 = { 1.0, 1.0, 1.0, 1.0 };
uddtD4 initialAddD4  = { 1.00001, 1.000003, 1.0000005, 1.00000007 };
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtD4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtD4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

TEST(StableSortUDD, AddDouble4)
{
    //setup containers
    int length = (1<<8);
    bolt::cl::device_vector< uddtD4 > input(  length, initialAddD4,  CL_MEM_READ_WRITE, true  );
    std::vector< uddtD4 > refInput( length, initialAddD4 );

    // call sort
    AddD4 ad4gt;
    bolt::BKND::SORT_FUNC(input.begin(),    input.end(), ad4gt);
    std::SORT_FUNC( refInput.begin(), refInput.end(), ad4gt );

    // compare results
    cmpArrays(refInput, input);
}

TEST(StableSortUDD, GPUAddDouble4)
{

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.
    //setup containers
    int length = (1<<8);
    bolt::cl::device_vector< uddtD4 > input(  length, initialAddD4,  CL_MEM_READ_WRITE, true  );
    std::vector< uddtD4 > refInput( length, initialAddD4 );

    // call sort
    AddD4 ad4gt;
    bolt::BKND::SORT_FUNC(c_gpu, input.begin(), input.end(), ad4gt );
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
TEST(MultiCoreCPU, MultiCoreAddDouble4)
{
    //setup containers
    int length = (1<<8);
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::device_vector< uddtD4 > input(  length, initialAddD4, CL_MEM_READ_WRITE, true  );
    std::vector< uddtD4 > refInput( length, initialAddD4 );
    
    // call sort
    AddD4 ad4gt;
    bolt::BKND::SORT_FUNC(input.begin(), input.end(), ad4gt);
    std::SORT_FUNC( refInput.begin(), refInput.end(), ad4gt );

    // compare results
    cmpArrays(refInput, input);
}
#endif

float func()
{
    return (float)(rand() * rand() * rand() );
}

//#if(TEST_LARGE_BUFFERS == 1)
TEST( DefaultGPU, Normal )
{
    int length = 2097152; //2^21
    std::vector< float > stdInput( length, 0.0 );

    std::generate(stdInput.begin(), stdInput.end(), func );
    bolt::cl::device_vector< float > boltInput(  stdInput.begin(), stdInput.end()  );

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );

    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ));
    bolt::BKND::SORT_FUNC( ctl, boltInput.begin( ), boltInput.end( ) );

    std::vector< float >::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    bolt::cl::device_vector< float >::difference_type boltNumElements = std::distance( boltInput.begin( ),
                                                                                        boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    
    //cmpArrays( stdInput, boltInput );
    for(int i=1;i<length;i= i<<1)
        EXPECT_FLOAT_EQ( stdInput[i], boltInput[i] );
}
//#endif

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
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
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

TYPED_TEST_P( StableSortArrayTest, GPU_DeviceNormal )
{
    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ));
    bolt::BKND::SORT_FUNC( c_gpu, StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    //OFFSET TEst cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex );
        bolt::BKND::SORT_FUNC( c_gpu, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }

}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( StableSortArrayTest, CPU_DeviceNormal )
{
        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    

    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ));
    bolt::BKND::SORT_FUNC( c_cpu, StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    
    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex );
        bolt::BKND::SORT_FUNC( c_cpu, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}
#endif

TYPED_TEST_P( StableSortArrayTest, GreaterFunction )
{
        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ), std::greater< ArrayType >() );
    
    bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::greater< ArrayType >( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    
    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, std::greater< ArrayType >() );
        bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, bolt::cl::greater< ArrayType >( )  );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}

TYPED_TEST_P( StableSortArrayTest, GPU_DeviceGreaterFunction )
{
        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    
    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();

    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ), std::greater< ArrayType >( ));
    bolt::BKND::SORT_FUNC( c_gpu, StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::greater< ArrayType >( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, std::greater< ArrayType >() );
        bolt::BKND::SORT_FUNC( c_gpu, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, bolt::cl::greater< ArrayType >( )  );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( StableSortArrayTest, CPU_DeviceGreaterFunction )
{
        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ), std::greater< ArrayType >( ));
    bolt::BKND::SORT_FUNC( c_cpu, StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::greater< ArrayType >( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, std::greater< ArrayType >() );
        bolt::BKND::SORT_FUNC( c_cpu, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, bolt::cl::greater< ArrayType >( )  );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}
#endif
TYPED_TEST_P( StableSortArrayTest, LessFunction )
{
        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ), std::less<ArrayType>());
    bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::less<ArrayType>() );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, std::less< ArrayType >() );
        bolt::BKND::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, bolt::cl::less< ArrayType >( )  );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}

TYPED_TEST_P( StableSortArrayTest, GPU_DeviceLessFunction )
{
        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    
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
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ), std::less< ArrayType >( ));
    bolt::BKND::SORT_FUNC( c_gpu, StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::less< ArrayType >( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, std::less< ArrayType >() );
        bolt::BKND::SORT_FUNC( c_gpu, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, bolt::cl::less< ArrayType >( )  );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( StableSortArrayTest, CPU_DeviceLessFunction )
{
        typedef typename StableSortArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ), std::less< ArrayType >( ));
    bolt::BKND::SORT_FUNC( c_cpu, StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ), bolt::cl::less< ArrayType >( ) );

    typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = StableSortArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > StableSortArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< StableSortArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        std::SORT_FUNC( StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, std::less< ArrayType >() );
        bolt::BKND::SORT_FUNC( c_cpu, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, StableSortArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, bolt::cl::less< ArrayType >( )  );

        typename ArrayCont::difference_type stdNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::stdInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::stdInput.end( ) );
        typename ArrayCont::difference_type boltNumElements = std::distance( StableSortArrayTest< gtest_TypeParam_ >::boltInput.begin( ), StableSortArrayTest< gtest_TypeParam_ >::boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, StableSortArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( StableSortArrayTest< gtest_TypeParam_ >::stdInput, StableSortArrayTest< gtest_TypeParam_ >::boltInput );
    }
}
#endif

#if (TEST_CPU_DEVICE == 1)
REGISTER_TYPED_TEST_CASE_P( StableSortArrayTest, Normal, GPU_DeviceNormal, 
                                           GreaterFunction, GPU_DeviceGreaterFunction,
                                           LessFunction, GPU_DeviceLessFunction, CPU_DeviceNormal, 
                                           CPU_DeviceGreaterFunction, CPU_DeviceLessFunction);
#else
REGISTER_TYPED_TEST_CASE_P( StableSortArrayTest, Normal, GPU_DeviceNormal, 
                                           GreaterFunction, GPU_DeviceGreaterFunction,
                                           LessFunction, GPU_DeviceLessFunction );
#endif


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
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortIntegerDeviceVector( ): stdInput( GetParam( ) ), 
                                stdOffsetIn( GetParam( ) ), ArraySize ( GetParam( ) )
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
    //bolt::cl::device_vector< int > boltInput, boltOffsetIn;
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
BOLT_FUNCTOR(UDD,
struct UDD { 
    int a; 
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) const{ 
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

BOLT_FUNCTOR(sortBy_UDD_a,
    struct sortBy_UDD_a {
        bool operator() (const UDD& a, const UDD& b) const
        { 
            return (a.a>b.a); 
        };
    };
);

BOLT_FUNCTOR(sortBy_UDD_b,
    struct sortBy_UDD_b {
        bool operator() (UDD& a, UDD& b) 
        { 
            return (a.b>b.b); 
        };
    };
);
BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::greater, int, UDD);
BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::less, int, UDD);
BOLT_TEMPLATE_REGISTER_NEW_ITERATOR(bolt::cl::device_vector, int, UDD);

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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

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

TEST_P( StableSortIntegerVector, MultiCoreCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

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

TEST_P( StableSortFloatVector, MultiCoreCPU)
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

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
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
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

TEST_P( StableSortDoubleVector, MulticoreInplace )
{
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
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
#if (TEST_DEVICE_VECTOR == 1)
TEST_P( StableSortIntegerDeviceVector, Inplace )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< int > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
     
    //  Calling the actual functions under test
    std::SORT_FUNC( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::SORT_FUNC( boltInput.begin( ), boltInput.end( ) );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end());
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin(),boltInput.end());

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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortIntegerDeviceVector, SerialInplace )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< int > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortIntegerDeviceVector, MultiCoreInplace )
{
    bolt::cl::device_vector< int > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< int > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortUDDDeviceVector, Inplace )
{
    bolt::cl::device_vector< UDD > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< UDD > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

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
    bolt::cl::device_vector< UDD > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< UDD > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< UDD >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortUDDDeviceVector, MultiCoreInplace )
{
    bolt::cl::device_vector< UDD > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< UDD > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< UDD >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortFloatDeviceVector, Inplace )
{
    bolt::cl::device_vector< float > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< float > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

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
    bolt::cl::device_vector< float > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< float > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< float >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortFloatDeviceVector, MultiCoreInplace )
{
    bolt::cl::device_vector< float > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< float > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
    typedef std::vector< float >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

#if (TEST_DOUBLE == 1)
TEST_P( StableSortDoubleDeviceVector, Inplace )
{
    bolt::cl::device_vector< double > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< double > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
    
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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

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
    bolt::cl::device_vector< double > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< double > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
     
    typedef std::vector< double >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

        std::vector< valtype >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end( ) );
        std::vector< valtype >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        //cmpStdArray< valtype, ArraySize >::cmpArrays( stdInput, boltInput );
        cmpArrays( stdInput, boltInput );
    }
}

TEST_P( StableSortDoubleDeviceVector, MulticoreInplace )
{
    bolt::cl::device_vector< double > boltInput(stdInput.begin( ), stdInput.end( ) );
    bolt::cl::device_vector< double > boltOffsetIn (stdOffsetIn.begin( ), stdOffsetIn.end( ) );
     
    typedef std::vector< double >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
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
        bolt::BKND::SORT_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, bolt::cl::less< valtype >( )  );

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
#if defined(_WIN32)
TEST_P( StableSortIntegerNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    size_t endIndex = GetParam( );

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

TEST_P( StableSortIntegerNakedPointer, MultiCoreInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    size_t endIndex = GetParam( );

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


TEST_P( StableSortFloatNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    size_t endIndex = GetParam( );

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

TEST_P( StableSortFloatNakedPointer, MultiCoreInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    size_t endIndex = GetParam( );

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

#if (TEST_DOUBLE == 1)
TEST_P( StableSortDoubleNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

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
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
TEST_P( StableSortDoubleNakedPointer, MulticoreInplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    std::SORT_FUNC( wrapStdInput, wrapStdInput + endIndex );
    //std::SORT_FUNC( stdInput, stdInput + endIndex );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::SORT_FUNC( wrapBoltInput, wrapBoltInput + endIndex );
    //bolt::BKND::SORT_FUNC( boltInput, boltInput + endIndex );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif

#endif
std::array<int, 15> TestValues = {2,4,8,16,32,64,128,256,512,1024,2048};
std::array<int, 15> TestValues2 = {4096,8192,16384,32768};

//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                                                                            TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																			
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortIntegerVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                            TestValues2.end() ) );
//#endif
                                                                            
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortFloatVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                        TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																		
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortFloatVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                        TestValues2.end() ) );
//#endif																		
                                                                        
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortDoubleVector, ::testing::Range(65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortDoubleVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                            TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																		
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortDoubleVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                            TestValues2.end() ) );
//#endif																			
#endif
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																				
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortIntegerDeviceVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                TestValues2.end() ) );
//#endif
                                                                                
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortUDDDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortUDDDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																				
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortUDDDeviceVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                TestValues2.end() ) );
//#endif
                                                                                
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																				
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortFloatDeviceVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                TestValues2.end()));
//#endif
                                                                                
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortDoubleDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortDoubleDeviceVector, ::testing::ValuesIn(TestValues.begin(),
                                                                                    TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																						
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortDoubleDeviceVector, ::testing::ValuesIn(TestValues2.begin(),
                                                                                    TestValues2.end()));
//#endif																					
#endif
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortIntegerNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                                    TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																					
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortIntegerNakedPointer, ::testing::ValuesIn( TestValues2.begin(),
                                                                                    TestValues2.end()));
//#endif
                                                                                    
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortFloatNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortValues, StableSortFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																				
INSTANTIATE_TEST_CASE_P( StableSortValues2, StableSortFloatNakedPointer, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                TestValues2.end() ) );
//#endif
                                                                                
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortRange, StableSortDoubleNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSort, StableSortDoubleNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                            TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																			
INSTANTIATE_TEST_CASE_P( StableSort2, StableSortDoubleNakedPointer, ::testing::ValuesIn( TestValues2.begin(),
                                                                            TestValues2.end() ) );		
//#endif																			
#endif

typedef ::testing::Types< 
    std::tuple< cl_long, TypeValue< 1 > >,
    std::tuple< cl_long, TypeValue< 31 > >,
    std::tuple< cl_long, TypeValue< 32 > >,
    std::tuple< cl_long, TypeValue< 63 > >,
    std::tuple< cl_long, TypeValue< 64 > >,
    std::tuple< cl_long, TypeValue< 127 > >,
    std::tuple< cl_long, TypeValue< 128 > >,
    std::tuple< cl_long, TypeValue< 129 > >,
    std::tuple< cl_long, TypeValue< 1000 > >,
    std::tuple< cl_long, TypeValue< 1053 > >,
    std::tuple< cl_long, TypeValue< 4096 > >,
    std::tuple< cl_long, TypeValue< 4097 > >,
    std::tuple< cl_long, TypeValue< 8192 > >,
    std::tuple< cl_long, TypeValue< 16384 > >,//13
    std::tuple< cl_long, TypeValue< 32768 > >,//14
    std::tuple< cl_long, TypeValue< 65535 > >,//15
    std::tuple< cl_long, TypeValue< 65536 > >,//16
    std::tuple< cl_long, TypeValue< 131072 > >,//17    
    std::tuple< cl_long, TypeValue< 262144 > >,//18    
    std::tuple< cl_long, TypeValue< 524288 > >,//19    
    std::tuple< cl_long, TypeValue< 1048576 > >,//20    
    std::tuple< cl_long, TypeValue< 2097152 > >//21 
	#if (TEST_LARGE_BUFFERS == 1)
    , /*This coma is needed*/
    std::tuple< cl_long, TypeValue< 4194304 > >,//22    
    std::tuple< cl_long, TypeValue< 8388608 > >,//23
    std::tuple< cl_long, TypeValue< 16777216 > >,//24
    std::tuple< cl_long, TypeValue< 33554432 > >,//25
    std::tuple< cl_long, TypeValue< 67108864 > >//26
#endif
> clLongTests;

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
    std::tuple< int, TypeValue< 8192 > >,
    std::tuple< int, TypeValue< 16384 > >,//13
    std::tuple< int, TypeValue< 32768 > >,//14
    std::tuple< int, TypeValue< 65535 > >,//15
    std::tuple< int, TypeValue< 65536 > >,//16
    std::tuple< int, TypeValue< 131072 > >,//17    
    std::tuple< int, TypeValue< 262144 > >,//18    
    std::tuple< int, TypeValue< 524288 > >,//19    
    std::tuple< int, TypeValue< 1048576 > >,//20    
    std::tuple< int, TypeValue< 2097152 > >//21 
	#if (TEST_LARGE_BUFFERS == 1)
    , /*This coma is needed*/
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
    std::tuple< unsigned int, TypeValue< 4097 > >,
    std::tuple< unsigned int, TypeValue< 8192 > >,
    std::tuple< unsigned int, TypeValue< 16384 > >,//13
    std::tuple< unsigned int, TypeValue< 32768 > >,//14
    std::tuple< unsigned int, TypeValue< 65535 > >,//15
    std::tuple< unsigned int, TypeValue< 65536 > >,//16
    std::tuple< unsigned int, TypeValue< 131072 > >,//17    
    std::tuple< unsigned int, TypeValue< 262144 > >,//18    
    std::tuple< unsigned int, TypeValue< 524288 > >,//19    
    std::tuple< unsigned int, TypeValue< 1048576 > >,//20    
    std::tuple< unsigned int, TypeValue< 2097152 > >//21
	#if (TEST_LARGE_BUFFERS == 1)
    , /*This coma is needed*/
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
    std::tuple< UDD, TypeValue< 4097 > >,
    std::tuple< UDD, TypeValue< 65535 > >,
    std::tuple< UDD, TypeValue< 65536 > >
> UDDTests;

INSTANTIATE_TYPED_TEST_CASE_P( clLong, StableSortArrayTest, clLongTests );
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


TEST_P (withStdVect, intSerialValuesWithDefaulFunctorWithClControl){
    std::vector <int> my_vect(sizeOfInputBuffer);
    std::vector <int> std_vect(sizeOfInputBuffer);

    for (int i = 0 ; i < sizeOfInputBuffer; ++i){
        my_vect[i] = rand() % 65535;
        std_vect[i] = my_vect[i];
        //srand(1);
    }

    // Create an OCL context, device, queue.
    MyOclContext ocl = initOcl(CL_DEVICE_TYPE_GPU, 0); //zero stand for one GPU

    bolt::cl::control c(ocl._queue);  // construct control structure from the queue.
    
    std::SORT_FUNC(std_vect.begin(), std_vect.end());
    bolt::BKND::SORT_FUNC(c, my_vect.begin(), my_vect.end());
    
    for (int i = 0 ; i < sizeOfInputBuffer; ++i){
        //std::cout<<"buffer size: "<<sizeOfInputBuffer<<std::endl;
        EXPECT_EQ(my_vect[i], std_vect[i])<<"Failed at i = "<<i<<std::endl;
    }
} 
TEST_P (withStdVect, intSerialValuesWithDefaulFunctorWithClControlGreater){
    std::vector <int> my_vect(sizeOfInputBuffer);
    std::vector <int> std_vect(sizeOfInputBuffer);

    for (int i = 0 ; i < sizeOfInputBuffer; ++i){
        my_vect[i] = rand() % 65535;
        std_vect[i] = my_vect[i];
        //srand(1);
    }

    // Create an OCL context, device, queue.
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    std::vector< cl::Platform > platforms;
    bolt::cl::V_OPENCL( cl::Platform::get( &platforms ), "Platform::get() failed" );

    // Device info
    std::vector< cl::Device > devices;
    //Select the first Platform
    platforms.at( 0 ).getDevices( deviceType, &devices );
    //Create an OpenCL context with the first device
    cl::Device device = devices.at( 0 );
    cl::Context myContext( device );
    cl::CommandQueue myQueue( myContext, device );

    //  Now that the device we want is selected and we have created our own cl::CommandQueue, set it as the
    //  default cl::CommandQueue for the Bolt API
    bolt::cl::control boltControl = bolt::cl::control::getDefault( );
    boltControl.setCommandQueue( myQueue );

    // Control setup:
    boltControl.setWaitMode(bolt::cl::control::BusyWait);

    std::SORT_FUNC(std_vect.begin(), std_vect.end(), std::greater<int>());
    bolt::BKND::SORT_FUNC(boltControl, my_vect.begin(), my_vect.end(), bolt::cl::greater<int>());
    
    for (int i = 0 ; i < sizeOfInputBuffer; ++i){
        //std::cout<<"buffer size: "<<sizeOfInputBuffer<<std::endl;
        EXPECT_EQ(my_vect[i], std_vect[i])<<"Failed at i = "<<i<<std::endl;
    }
}
INSTANTIATE_TEST_CASE_P(sortDescending, withStdVect, ::testing::Range(50, 100, 1));

TEST (sanity_sort__withBoltClDevVectDouble_epr, floatSerial){
    bolt::cl::device_vector<double>::size_type sizeOfInputBufer = 64; //test case is failing for all values greater than 32
    std::vector<double>  stdVect(0);
    bolt::cl::device_vector<double>  boltVect(0);

    for (bolt::cl::device_vector<double>::size_type i = 0 ; i < sizeOfInputBufer; i++){
        double dValue = rand();
        dValue = dValue/rand();
        dValue = dValue*rand();
        stdVect.push_back(dValue);
        boltVect.push_back(dValue);
    }
    std::SORT_FUNC(stdVect.begin(), stdVect.end(), std::greater<double>( ) );
    bolt::BKND::SORT_FUNC(boltVect.begin(), boltVect.end(), bolt::cl::greater<double>( ) );

    for (bolt::cl::device_vector<double>::size_type i = 0 ; i < sizeOfInputBufer; i++){
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
        stdArray[i] = boltArray[i] = backupArray[i] = fValue;
    }
    std::SORT_FUNC( stdArray, stdArray+sizeOfInputBufer, std::greater<float>( ) );
    bolt::BKND::SORT_FUNC( boltArray, boltArray+sizeOfInputBufer, bolt::cl::greater<float>( ) );

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
    bolt::BKND::SORT_FUNC( boltArray+17, boltArray+sizeOfInputBufer-129, bolt::cl::greater<float>( ) );

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

#include "bolt/cl/iterator/counting_iterator.h"
#include <bolt/cl/sort.h>
#include <bolt/cl/functional.h>
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
