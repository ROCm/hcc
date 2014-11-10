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
#define TEST_CPU_DEVICE 1
#define TEST_MULTICORE_TBB_SEARCH 1
#define TEST_LARGE_BUFFERS 0
#define GOOGLE_TEST 1
#define BKND cl 
#define SEARCH_FUNC binary_search

#if (GOOGLE_TEST == 1)
#include <gtest/gtest.h>
#include "common/stdafx.h"
#include "common/myocl.h"
#include "bolt/cl/binary_search.h"
#include "common/test_common.h"
#include "bolt/cl/iterator/counting_iterator.h"

#include "bolt/cl/sort.h"

#include <bolt/cl/sort.h>
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

    bool operator<(const uddtD4 &rhs) const
    {

        if( ( a + b + c + d ) < ( rhs.a + rhs.b + rhs.c + rhs.d) )
            return true;
        return false;
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
    }
}; 
);
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< AddD4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< AddD4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );
BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::less, int, uddtD4);

uddtD4 identityAddD4 = { 1.0, 1.0, 1.0, 1.0 };
uddtD4 initialAddD4  = { 1.00001, 1.000003, 1.0000005, 1.00000007 };

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtD4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtD4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

TEST(BSearch, StdclLong)  
{
        // test length
        int length = (1<<8);

        std::vector<cl_long> bolt_source(length);
        std::vector<cl_long> std_source(length);

        // populate source vector with random ints
        for (int j = 0; j < length; j++)
        {
            bolt_source[j] = (cl_long)rand();
            std_source[j] = bolt_source[j];
        }
    
        int search_index = rand()%length;
        cl_long val = bolt_source[search_index];
        cl_long std_val = std_source[search_index];
        bool stdresult, boltresult;

        //Sorting the Input
        std::sort(std_source.begin(), std_source.end());
        bolt::cl::sort(bolt_source.begin(), bolt_source.end());

        // perform search
        stdresult = std::binary_search(std_source.begin(), std_source.end(), std_val);
        boltresult = bolt::cl::binary_search(bolt_source.begin(), bolt_source.end(), val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

} 

TEST(BSearch, Serial_StdclLong)  
{
        // test length
        int length = (1<<8);

        std::vector<cl_long> bolt_source(length);
        std::vector<cl_long> std_source(length);

        // populate source vector with random ints
        for (int j = 0; j < length; j++)
        {
            bolt_source[j] = (cl_long)rand();
            std_source[j] = bolt_source[j];
        }
    
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

        //Sorting the Input
        std::sort(std_source.begin(), std_source.end());
        bolt::cl::sort(bolt_source.begin(), bolt_source.end());

        int search_index = rand()%length;
        cl_long val = bolt_source[search_index];
        cl_long std_val = std_source[search_index];
        bool stdresult, boltresult;

        //perform search
        stdresult = std::binary_search(std_source.begin(), std_source.end(), std_val);
        boltresult = bolt::cl::binary_search(bolt_source.begin(), bolt_source.end(), val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

} 


TEST(BSearch, MultiCore_StdclLong)  
{
        // test length
        int length = (1<<8);

        std::vector<cl_long> bolt_source(length);
        std::vector<cl_long> std_source(length);

        // populate source vector with random ints
        for (int j = 0; j < length; j++)
        {
            bolt_source[j] = (cl_long)rand();
            std_source[j] = bolt_source[j];
        }
    
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

         //Sorting the Input
        std::sort(std_source.begin(), std_source.end());
        bolt::cl::sort(bolt_source.begin(), bolt_source.end());

        int search_index = rand()%length;
        cl_long val = bolt_source[search_index];
        cl_long std_val = std_source[search_index];
        bool stdresult, boltresult;

        // perform search
        stdresult = std::binary_search(std_source.begin(), std_source.end(), std_val);
        boltresult = bolt::cl::binary_search(bolt_source.begin(), bolt_source.end(), val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

} 

TEST(BSearch, DevclLong)  
{
        // test length
        int length = (1<<8);
        std::vector<cl_long> std_source(length);
        // populate source vector with random ints
        for (int j = 0; j < length; j++)
        {
            std_source[j] = (cl_long)rand();
        }
        bolt::cl::device_vector<cl_long> bolt_source(std_source.begin(),std_source.end());

    
         //Sorting the Input
        std::sort(std_source.begin(), std_source.end());
        bolt::cl::sort(bolt_source.begin(), bolt_source.end());

        int search_index = rand()%length;
        cl_long val = bolt_source[search_index];
        cl_long std_val = std_source[search_index];
        bool stdresult, boltresult;

        // perform search
        stdresult = std::binary_search(std_source.begin(), std_source.end(), std_val);
        boltresult = bolt::cl::binary_search(bolt_source.begin(), bolt_source.end(), val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

} 

TEST(BSearch, Serial_DevclLong)  
{
        // test length
        int length = (1<<8);

        std::vector<cl_long> std_source(length);

        for (int j = 0; j < length; j++)
        {
            std_source[j] = (cl_long)rand();
        }
        bolt::cl::device_vector<cl_long> bolt_source(std_source.begin(),std_source.end());

    
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

         //Sorting the Input
        std::sort(std_source.begin(), std_source.end());
        bolt::cl::sort(bolt_source.begin(), bolt_source.end());

        int search_index = rand()%length;
        cl_long val = bolt_source[search_index];
        cl_long std_val = std_source[search_index];
        bool stdresult, boltresult;

        // perform search
        stdresult = std::binary_search(std_source.begin(), std_source.end(), std_val);
        boltresult = bolt::cl::binary_search(ctl, bolt_source.begin(), bolt_source.end(), val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

}

TEST(BSearch, MultiCore_DevclLong)  
{
        // test length
        int length = (1<<8);

        std::vector<cl_long> std_source(length);

        for (int j = 0; j < length; j++)
        {
            std_source[j] = (cl_long)rand();
        }
        bolt::cl::device_vector<cl_long> bolt_source(std_source.begin(),std_source.end());
    
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

         //Sorting the Input
        std::sort(std_source.begin(), std_source.end());
        bolt::cl::sort(bolt_source.begin(), bolt_source.end());

        int search_index = rand()%length;
        cl_long val = bolt_source[search_index];
        cl_long std_val = std_source[search_index];
        bool stdresult, boltresult;

        // perform search
        stdresult = std::binary_search(std_source.begin(), std_source.end(), std_val);
        boltresult = bolt::cl::binary_search(ctl, bolt_source.begin(), bolt_source.end(), val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

}

TEST(BSearchUDD, AddDouble4)
{
    //setup containers
    int length = (1<<8);
    bolt::cl::device_vector< uddtD4 > input(  length, initialAddD4,  CL_MEM_READ_WRITE, true  );
    std::vector< uddtD4 > refInput( length, initialAddD4 );

    // call sort
    AddD4 ad4gt;
    bolt::BKND::sort(input.begin(), input.end(), ad4gt);
    std::sort( refInput.begin(), refInput.end(), ad4gt );

    bool stdresult, boltresult;

    int index = rand() % length;
    uddtD4 std_val = refInput[index];
    uddtD4 val = input[index];

    // perform search
    stdresult = std::binary_search(refInput.begin(), refInput.end(), std_val);
    boltresult = bolt::cl::binary_search(input.begin(), input.end(), val);
   

    // compare results
    EXPECT_EQ( stdresult ,  boltresult);
}

TEST(BSearchUDD, GPUAddDouble4)
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
    bolt::BKND::sort(c_gpu, input.begin(), input.end(), ad4gt );
    std::sort( refInput.begin(), refInput.end(), ad4gt );

    bool stdresult, boltresult;

    int index = rand() % length;
    uddtD4 std_val = refInput[index];
    uddtD4 val = input[index];

    // perform search
    stdresult = std::binary_search(refInput.begin(), refInput.end(), std_val);
    boltresult = bolt::cl::binary_search(c_gpu, input.begin(), input.end(), val);
    
    // compare results
    EXPECT_EQ( stdresult ,  boltresult);
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
class BSearchArrayTest: public ::testing::Test
{
public:
    BSearchArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(std_source.begin(), std_source.end(), rand);
        std::sort(std_source.begin(), std_source.end());
        bolt_source = std_source;
        stdOffsetIn = std_source;
        boltOffsetIn = bolt_source;
        std_val = 0, val = 0;
    };

    virtual void TearDown( )
    {};

    virtual ~BSearchArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    ArrayType std_val, val;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    std::array< ArrayType, ArraySize > std_source, bolt_source, stdOffsetIn, boltOffsetIn;
    int m_Errors;
};

TYPED_TEST_CASE_P( BSearchArrayTest );

#if ( TEST_DOUBLE == 1)

#if (TEST_MULTICORE_TBB_SEARCH == 1)
TEST(MultiCoreCPU, MultiCoreAddDouble4)
{
    //setup containers
    int length = (1<<8);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::device_vector< uddtD4 > input(  length, initialAddD4, CL_MEM_READ_WRITE, true  );
    std::vector< uddtD4 > refInput( length, initialAddD4 );
    
    // call search
    AddD4 ad4gt;
    bolt::BKND::sort(ctl, input.begin(), input.end(), ad4gt);
    std::sort( refInput.begin(), refInput.end(), ad4gt );

    bool stdresult, boltresult;

    int index = rand() % length;
    uddtD4 std_val = refInput[index];
    uddtD4 val = input[index];

    // perform search
    stdresult = std::binary_search(refInput.begin(), refInput.end(), std_val);
    boltresult = bolt::cl::binary_search(ctl, input.begin(), input.end(), val);

    // compare results
    EXPECT_EQ( stdresult ,  boltresult);
}
#endif 

TEST( DefaultGPU, Normal )
{
    int length = 1025;
    bolt::cl::device_vector< float > boltInput(  length, 0.0, CL_MEM_READ_WRITE, true  );
    std::vector< float > stdInput( length, 0.0 );

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );

    // populate source vector with random ints
    for (int j = 0; j < length; j++)
    {
         boltInput[j] = (float)rand();
         stdInput[j] = boltInput[j];
    }
    
    // Sorting the Input
    std::sort(stdInput.begin(), stdInput.end());
    bolt::cl::sort(ctl, boltInput.begin(), boltInput.end());

    int index = rand()%length;
    float std_val = stdInput[index];
    float val = boltInput[index];
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), std_val);
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    // GoogleTest Comparison
    EXPECT_EQ( stdresult ,  boltresult);

}

TEST( SerialCPU, SerialNormal )
{
    int length = 1025;
    bolt::cl::device_vector< float > boltInput(  length, 0.0, CL_MEM_READ_WRITE, true  );
    std::vector< float > stdInput( length, 0.0 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
  
    // populate source vector with random ints
    for (int j = 0; j < length; j++)
    {
         boltInput[j] = (float)rand();
         stdInput[j] = boltInput[j];
    }
    
    // Sorting the Input
    std::sort(stdInput.begin(), stdInput.end());
    bolt::cl::sort(ctl, boltInput.begin(), boltInput.end());

    int index = rand()%length;
    float std_val = stdInput[index];
    float val = boltInput[index];
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), std_val);
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    // GoogleTest Comparison
    EXPECT_EQ( stdresult ,  boltresult);
}

#if (TEST_MULTICORE_TBB_SEARCH == 1)
TEST( MultiCoreCPU, MultiCoreNormal )
{
    int length = 1025;
    bolt::cl::device_vector< float > boltInput(  length, 0.0, CL_MEM_READ_WRITE, true  );
    std::vector< float > stdInput( length, 0.0 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
  
    // populate source vector with random ints
    for (int j = 0; j < length; j++)
    {
         boltInput[j] = (float)rand();
         stdInput[j] = boltInput[j];
    }
    
    // Sorting the Input
    std::sort(stdInput.begin(), stdInput.end());
    bolt::cl::sort(ctl, boltInput.begin(), boltInput.end());

    int index = rand()%length;
    float std_val = stdInput[index];
    float val = boltInput[index];
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), std_val);
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    // GoogleTest Comparison
    EXPECT_EQ( stdresult ,  boltresult);
}
#endif
#endif

TYPED_TEST_P( BSearchArrayTest, Normal )
{
    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC(BSearchArrayTest< gtest_TypeParam_ >::std_source.begin(), 
      BSearchArrayTest< gtest_TypeParam_ >::std_source.end(), BSearchArrayTest< gtest_TypeParam_ >::std_val);
    boltresult = bolt::BKND::SEARCH_FUNC(BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin(), 
      BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end(), BSearchArrayTest< gtest_TypeParam_ >::val);
 
    //  Both collections should have the same number of elements
    EXPECT_EQ( stdresult ,  boltresult );

    //  OFFSET Calling the actual functions under test
    size_t startIndex = 17; //Some arbitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some arbitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {   
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

        // perform search
        stdresult = std::SEARCH_FUNC(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, 
          BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex , BSearchArrayTest< gtest_TypeParam_ >::std_val);
        boltresult = bolt::BKND::SEARCH_FUNC(BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex,
          BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, BSearchArrayTest< gtest_TypeParam_ >::val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

    }
}

TYPED_TEST_P( BSearchArrayTest, GPU_DeviceNormal )
{
    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::std_source.begin( ), 
      BSearchArrayTest< gtest_TypeParam_ >::std_source.end( ), BSearchArrayTest< gtest_TypeParam_ >::std_val);
    boltresult = bolt::BKND::SEARCH_FUNC( c_gpu, BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin( ), 
      BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end( ), BSearchArrayTest< gtest_TypeParam_ >::val );

    EXPECT_EQ( stdresult, boltresult );

    //OFFSET TEst cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

         // perform search
        stdresult = std::SEARCH_FUNC(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, 
          BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex , BSearchArrayTest< gtest_TypeParam_ >::std_val);
        boltresult = bolt::BKND::SEARCH_FUNC(c_gpu, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, 
          BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, BSearchArrayTest< gtest_TypeParam_ >::val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

    }

}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( BSearchArrayTest, CPU_DeviceNormal )
{
    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdInput.begin( ), 
      BSearchArrayTest< gtest_TypeParam_ >::stdInput.end( ), BSearchArrayTest< gtest_TypeParam_ >::std_val);
    boltresult = bolt::BKND::SEARCH_FUNC( c_cpu, BSearchArrayTest< gtest_TypeParam_ >::boltInput.begin( ), 
      BSearchArrayTest< gtest_TypeParam_ >::boltInput.end( ), BSearchArrayTest< gtest_TypeParam_ >::val );

    EXPECT_EQ( stdresult, boltresult );
    
    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

         // perform search
        stdresult = std::SEARCH_FUNC(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, 
         BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex , BSearchArrayTest< gtest_TypeParam_ >::std_val);
        boltresult = bolt::BKND::SEARCH_FUNC(c_cpu, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex,
         BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, BSearchArrayTest< gtest_TypeParam_ >::val);

        // GoogleTest Comparison
        EXPECT_EQ( stdresult ,  boltresult);

    }
}
#endif

TYPED_TEST_P( BSearchArrayTest, GreaterFunction )
{
    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    std::sort(BSearchArrayTest< gtest_TypeParam_ >::std_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::std_source.end(), 
      std::greater< ArrayType >()); 
    bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end(),
      bolt::cl::greater< ArrayType >());
    
    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::std_source.begin( ), BSearchArrayTest< gtest_TypeParam_ >::std_source.end( ),
      BSearchArrayTest< gtest_TypeParam_ >::std_val, std::greater< ArrayType >() ); 
    boltresult = bolt::BKND::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin( ), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end( ),
      BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::greater< ArrayType >( ) );

    EXPECT_EQ(  stdresult, boltresult );
    
    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

        std::sort(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.end(), std::greater< ArrayType >()); 
        bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.end(), bolt::cl::greater< ArrayType >());

        stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex,
          BSearchArrayTest< gtest_TypeParam_ >::std_val, std::greater< ArrayType >() );
        boltresult = bolt::BKND::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, 
          BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::greater< ArrayType >( )  );

        EXPECT_EQ( stdresult, boltresult );
    }
}

TYPED_TEST_P( BSearchArrayTest, GPU_DeviceGreaterFunction )
{
    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    //  The first time our routines get called, we compile the library kernels with a certain context
    //  OpenCL does not allow the context to change without a recompile of the kernel

    //  Create a new command queue for a different device, but use the same context as was provided
    //  by the default control device
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();

    ::cl::CommandQueue myQueue( myContext, devices[ 0 ] );
    bolt::cl::control c_gpu( myQueue );  // construct control structure from the queue.

    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    std::sort(BSearchArrayTest< gtest_TypeParam_ >::std_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::std_source.end(), std::greater< ArrayType >()); 
    bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end(), bolt::cl::greater< ArrayType >());

    //  Calling the actual functions under test
    BSearchArrayTest< gtest_TypeParam_ >::stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::std_source.begin( ), BSearchArrayTest< gtest_TypeParam_ >::std_source.end( ),
      BSearchArrayTest< gtest_TypeParam_ >::std_val, std::greater< ArrayType >() ); 
    BSearchArrayTest< gtest_TypeParam_ >::boltresult = bolt::BKND::SEARCH_FUNC( c_gpu, BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin( ), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end( ),     
      BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::greater< ArrayType >( ) );

    EXPECT_EQ(  stdresult, boltresult );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

        std::sort(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.end(), std::greater< ArrayType >()); 
        bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.end(), bolt::cl::greater< ArrayType >());

        stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, 
           BSearchArrayTest< gtest_TypeParam_ >::std_val, std::greater< ArrayType >() );
        boltresult = bolt::BKND::SEARCH_FUNC( c_gpu, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, 
           BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::greater< ArrayType >( )  );

        EXPECT_EQ( stdresult, boltresult );

    }
}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( BSearchArrayTest, CPU_DeviceGreaterFunction )
{
    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    std::sort(BSearchArrayTest< gtest_TypeParam_ >::std_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::std_source.end(), std::greater< ArrayType >()); 
    bolt::cl::sort(c_cpu, BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end(), bolt::cl::greater< ArrayType >());

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdInput.begin( ), BSearchArrayTest< gtest_TypeParam_ >::stdInput.end( ),
      BSearchArrayTest< gtest_TypeParam_ >::std_val, std::greater< ArrayType >() ); 
    boltresult = bolt::BKND::SEARCH_FUNC( c_cpu, BSearchArrayTest< gtest_TypeParam_ >::boltInput.begin( ), BSearchArrayTest< gtest_TypeParam_ >::boltInput.end( ), 
      BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::greater< ArrayType >( ) );

    EXPECT_EQ(  stdresult, boltresult );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

        std::sort(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.end(), std::greater< ArrayType >()); 
        bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.end(), bolt::cl::greater< ArrayType >());

        stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex,
           BSearchArrayTest< gtest_TypeParam_ >::std_val, std::greater< ArrayType >() );
        boltresult = bolt::BKND::SEARCH_FUNC( c_cpu, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex,
           BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::greater< ArrayType >( )  );

        EXPECT_EQ( stdresult, boltresult );
    }
}
#endif

TYPED_TEST_P( BSearchArrayTest, LessFunction )
{
    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    std::sort(BSearchArrayTest< gtest_TypeParam_ >::std_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::std_source.end(), std::less< ArrayType >()); 
    bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end(), bolt::cl::less< ArrayType >());

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::std_source.begin( ), BSearchArrayTest< gtest_TypeParam_ >::std_source.end( ), 
      BSearchArrayTest< gtest_TypeParam_ >::std_val, std::less< ArrayType >() ); 
    boltresult = bolt::BKND::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin( ), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end( ), 
      BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::less< ArrayType >( ) );

    EXPECT_EQ(  stdresult, boltresult );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

        std::sort(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.end(), std::less< ArrayType >()); 
        bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.end(), bolt::cl::less< ArrayType >());

        stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, 
           BSearchArrayTest< gtest_TypeParam_ >::std_val,  std::less< ArrayType >() );
        boltresult = bolt::BKND::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, 
           BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::less< ArrayType >( )  );

        EXPECT_EQ( stdresult, boltresult );
    }
}

TYPED_TEST_P( BSearchArrayTest, GPU_DeviceLessFunction )
{
    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

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
   
    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    std::sort(BSearchArrayTest< gtest_TypeParam_ >::std_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::std_source.end(), std::less< ArrayType >()); 
    bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin(), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end(), bolt::cl::less< ArrayType >());

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::std_source.begin( ), BSearchArrayTest< gtest_TypeParam_ >::std_source.end( ), 
       BSearchArrayTest< gtest_TypeParam_ >::std_val, std::less< ArrayType >() ); 
    boltresult = bolt::BKND::SEARCH_FUNC( c_gpu, BSearchArrayTest< gtest_TypeParam_ >::bolt_source.begin( ), BSearchArrayTest< gtest_TypeParam_ >::bolt_source.end( ), 
       BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::less< ArrayType >( ) );

    EXPECT_EQ(  stdresult, boltresult );


    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

        std::sort(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.end(), std::less< ArrayType >()); 
        bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.end(), bolt::cl::less< ArrayType >());

        stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, 
           BSearchArrayTest< gtest_TypeParam_ >::std_val, std::less< ArrayType >() );
        boltresult = bolt::BKND::SEARCH_FUNC(c_gpu, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex,
           BSearchArrayTest< gtest_TypeParam_ >::val, bolt::cl::less< ArrayType >( )  );

        EXPECT_EQ( stdresult, boltresult );
    }
}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P(BSearchArrayTest, CPU_DeviceLessFunction )
{
    typedef typename BSearchArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, BSearchArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    size_t index = rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
    BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
    BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];
    bool stdresult, boltresult;

    std::sort(BSearchArrayTest< gtest_TypeParam_ >::stdInput.begin(), BSearchArrayTest< gtest_TypeParam_ >::stdInput.end(), std::less< ArrayType >()); 
    bolt::cl::sort(c_cpu, BSearchArrayTest< gtest_TypeParam_ >::sboltInput.begin(), BSearchArrayTest< gtest_TypeParam_ >::boltInput.end(), bolt::cl::less< ArrayType >());

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdInput.begin( ), BSearchArrayTest< gtest_TypeParam_ >::stdInput.end( ), 
      BSearchArrayTest< gtest_TypeParam_ >::std_val); //, std::less< ArrayType >() ); 
    boltresult = bolt::BKND::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::boltInput.begin( ), BSearchArrayTest< gtest_TypeParam_ >::boltInput.end( ),
      BSearchArrayTest< gtest_TypeParam_ >::val); //, bolt::cl::less< ArrayType >( ) );

    EXPECT_EQ( stdresult, boltresult );

    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = BSearchArrayTest< gtest_TypeParam_ >::ArraySize -17; //Some aribitrary offset position
    if( (( startIndex > BSearchArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< BSearchArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {
        index = startIndex + rand()%BSearchArrayTest< gtest_TypeParam_ >::ArraySize;
        BSearchArrayTest< gtest_TypeParam_ >::std_val = BSearchArrayTest< gtest_TypeParam_ >::std_source[index];
        BSearchArrayTest< gtest_TypeParam_ >::val = BSearchArrayTest< gtest_TypeParam_ >::bolt_source[index];

        std::sort(BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.end(), std::less< ArrayType >()); 
        bolt::cl::sort(BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin(), BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.end(), bolt::cl::less< ArrayType >());

        stdresult = std::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, 
           BSearchArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, BSearchArrayTest< gtest_TypeParam_ >::std_val); //, std::less< ArrayType >() );
        boltresult = bolt::BKND::SEARCH_FUNC( BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, 
           BSearchArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, BSearchArrayTest< gtest_TypeParam_ >::val); //, bolt::cl::less< ArrayType >( )  );

        EXPECT_EQ( stdresult, boltresult );
    }
}
#endif

#if (TEST_CPU_DEVICE == 1)
REGISTER_TYPED_TEST_CASE_P( BSearchArrayTest, Normal, GPU_DeviceNormal, 
                                           GreaterFunction, GPU_DeviceGreaterFunction,
                                           LessFunction, GPU_DeviceLessFunction, CPU_DeviceNormal, 
                                           CPU_DeviceGreaterFunction, CPU_DeviceLessFunction);
#else
REGISTER_TYPED_TEST_CASE_P( BSearchArrayTest, Normal, GPU_DeviceNormal, 
                                           GreaterFunction, GPU_DeviceGreaterFunction,
                                           LessFunction, GPU_DeviceLessFunction );

#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

//class BSearchStdVector_MulValues: public ::testing::TestWithParam< int >
//{
//public:
//
//    static int gen_int_random(void)
//    {
//        return rand()%100;
//    }
//    BSearchStdVector_MulValues( ): stdInput( GetParam( ) ), boltInput( GetParam( )), values(GetParam( )), std_result(GetParam( )), bolt_result(GetParam( )) 
//    {
//        std::generate(stdInput.begin(), stdInput.end(), gen_int_random);
//        std::sort(stdInput.begin(), stdInput.end());
//        boltInput = stdInput;
//        std::generate(values.begin(), values.end(), gen_int_random);
//    }
//
//protected:
//    std::vector< int > stdInput, boltInput, values, std_result, bolt_result;
//};


class BSearchIntegerVector: public ::testing::TestWithParam< int >
{
public:

    static int gen_int_random(void)
    {
        return rand()%100;
    }
    BSearchIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), gen_int_random);
        std::sort(stdInput.begin(), stdInput.end());
        boltInput = stdInput;
    }

protected:
    std::vector< int > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class BSearchFloatVector: public ::testing::TestWithParam< int >
{
public:
    static float gen_float_random(void)
    {
        int toggle = 0;
        if(toggle == 0)
        {
            toggle = 1;
            return (float)-rand();
        }
        else
        {
            toggle = 0;
            return (float)rand();
        }
    }
    BSearchFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), gen_float_random);
        std::sort(stdInput.begin(), stdInput.end());
        boltInput = stdInput;    
    }

protected:
    std::vector< float > stdInput, boltInput;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class BSearchDoubleVector: public ::testing::TestWithParam< int >
{
public:
    static double gen_double_random(void)
    {
        int toggle = 0;
        if(toggle == 0)
        {
            toggle = 1;
            return (double)-rand();
        }
        else
        {
            toggle = 0;
            return (double)rand();
        }
    }
    BSearchDoubleVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), gen_double_random);
        std::sort(stdInput.begin(), stdInput.end());
        boltInput = stdInput;    
    }

protected:
    std::vector< double > stdInput, boltInput;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class BSearchIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    static int gen_int_random(void)
    {
        return rand()%100;
    }
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    BSearchIntegerDeviceVector( ): stdInput( GetParam( ) ), boltInput( static_cast<size_t>( GetParam( ) ) ), 
                                stdOffsetIn( GetParam( ) ), boltOffsetIn( static_cast<size_t>( GetParam( ) ) ), ArraySize ( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), gen_int_random);
        std::sort(stdInput.begin(), stdInput.end());
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOffsetIn[i] = stdInput[i];
            stdOffsetIn[i] = stdInput[i];
        }
    }

protected:
    std::vector< int > stdInput, stdOffsetIn;
    bolt::cl::device_vector< int > boltInput, boltOffsetIn;
    const int ArraySize;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class BSearchFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    static float gen_float_random(void)
    {
        int toggle = 0;
        if(toggle == 0)
        {
            toggle = 1;
            return (float)-rand();
        }
        else
        {
            toggle = 0;
            return (float)rand();
        }
    }
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    BSearchFloatDeviceVector( ): stdInput( GetParam( ) ), boltInput( static_cast<size_t>( GetParam( ) ) ), 
                              stdOffsetIn( GetParam( ) ), boltOffsetIn( static_cast<size_t>( GetParam( ) ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), gen_float_random);
        std::sort(stdInput.begin(), stdInput.end());
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOffsetIn[i] = stdInput[i];
            stdOffsetIn[i] = stdInput[i];
        }
    }

protected:
    std::vector< float > stdInput, stdOffsetIn;
    bolt::cl::device_vector< float > boltInput, boltOffsetIn;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class BSearchDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    static double gen_double_random(void)
    {
        int toggle = 0;
        if(toggle == 0)
        {
            toggle = 1;
            return (double)-rand();
        }
        else
        {
            toggle = 0;
            return (double)rand();
        }
    }
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    BSearchDoubleDeviceVector( ): stdInput( GetParam( ) ), boltInput( static_cast<size_t>( GetParam( ) ) ),
                               stdOffsetIn( GetParam( ) ), boltOffsetIn( static_cast<size_t>( GetParam( ) ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), gen_double_random);
        std::sort(stdInput.begin(), stdInput.end());
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOffsetIn[i] = stdInput[i];
            stdOffsetIn[i] = stdInput[i];
        }
    }

protected:
    std::vector< double > stdInput, stdOffsetIn;
    bolt::cl::device_vector< double > boltInput, boltOffsetIn;
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
    void operator = (const UDD& other) { 
        a = other.a;
        b = other.b;
    }

    UDD() 
        : a(0),b(0) { } 
    UDD(int _in) 
        : a(_in), b(_in +1)  { } 
}; 
);

BOLT_FUNCTOR(BSearchBy_UDD_a,
    struct BSearchBy_UDD_a {
        bool operator() (const UDD& a, const UDD& b) const
        { 
            return (a.a>b.a); 
        };
    };
);

BOLT_FUNCTOR(BSearchBy_UDD_b,
    struct BSearchBy_UDD_b {
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
class BSearchUDDDeviceVector: public ::testing::TestWithParam< int >
{
public:
    static UDD gen_UDD_random(void)
    {
        UDD temp;
        int toggle = 0;
        if(toggle == 0)
        {
            toggle = 1;
            temp.a = -rand();
            temp.b = rand();
            return temp;
        }
        else
        {
            toggle = 0;
            temp.a = rand();
            temp.b = -rand();
            return temp;
        }
    }
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    BSearchUDDDeviceVector( ): stdInput( GetParam( ) ), boltInput( static_cast<size_t>( GetParam( ) ) ),
                            stdOffsetIn( GetParam( ) ), boltOffsetIn( static_cast<size_t>( GetParam( ) ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), gen_UDD_random);
        std::sort(stdInput.begin(), stdInput.end());
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOffsetIn[i] = stdInput[i];
            stdOffsetIn[i] = stdInput[i];
        }
    }

protected:
    std::vector< UDD > stdInput,stdOffsetIn;
    bolt::cl::device_vector< UDD > boltInput,boltOffsetIn;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class BSearchIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    static int gen_int_random(void)
    {
        int toggle = 0;
        if(toggle == 0)
        {
            toggle = 1;
            return -rand();
        }
        else
        {
            toggle = 0;
            return rand();
        }
    }
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    BSearchIntegerNakedPointer( ): stdInput( new int[ GetParam( ) ] ), boltInput( new int[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, gen_int_random);
        std::sort(stdInput, stdInput + size);
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
class BSearchFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    static float gen_float_random(void)
    {
        int toggle = 0;
        if(toggle == 0)
        {
            toggle = 1;
            return (float)-rand();
        }
        else
        {
            toggle = 0;
            return (float)rand();
        }
    }
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    BSearchFloatNakedPointer( ): stdInput( new float[ GetParam( ) ] ), boltInput( new float[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, gen_float_random);
        std::sort(stdInput, stdInput + size);
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
class BSearchDoubleNakedPointer: public ::testing::TestWithParam< int >
{
public:
    static double gen_double_random(void)
    {
        int toggle = 0;
        if(toggle == 0)
        {
            toggle = 1;
            return (double)-rand();
        }
        else
        {
            toggle = 0;
            return (double)rand();
        }
    }
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    BSearchDoubleNakedPointer( ): stdInput( new double[ GetParam( ) ] ), boltInput( new double[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, gen_double_random);
        std::sort(stdInput, stdInput + size);

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

class BSearchCountingIterator :public ::testing::TestWithParam<int>{
protected:
     int mySize;
public:
    BSearchCountingIterator(): mySize(GetParam()){
    }
};


TEST_P(BSearchCountingIterator, withCountingIterator)
{
    std::vector<int> a(mySize);

    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first + mySize;
    
    for(int i=0; i< mySize ; i++)
    {
        a[i] = i;
    }

    int val = rand() % mySize;

    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( a.begin( ), a.end( ), val); 
    boltresult = bolt::BKND::SEARCH_FUNC( first, last, val);
    
    EXPECT_EQ(  stdresult, boltresult );

} 


//TEST_P( BSearchStdVector_MulValues, Greater )
//{
//
//    std::sort( stdInput.begin( ), stdInput.end( ), std::greater< int >() );
//    bolt::cl::sort( boltInput.begin( ), boltInput.end( ), bolt::cl::greater< int >());
//
//    //  Calling the actual functions under test
//    int n = (int)values.size();
//    for(int i=0;i<n;i++)
//       std_result[i] = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), values[i], std::greater< int >() );
//
//    bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), values.begin(), values.end(), bolt_result.begin(), bolt::cl::greater< int >());
//
//    cmpArrays(std_result, bolt_result);
//}

TEST_P( BSearchIntegerVector, Greater )
{
    bool stdresult, boltresult;
  
    int val = rand()%10;

    std::sort( stdInput.begin( ), stdInput.end( ), std::greater< int >() );
    bolt::cl::sort( boltInput.begin( ), boltInput.end( ), bolt::cl::greater< int >());

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val, std::greater< int >() );
    boltresult = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), val, bolt::cl::greater< int >());

    EXPECT_EQ(stdresult, boltresult);
}

TEST_P( BSearchIntegerVector, SerialGreater )
{
    bool stdresult, boltresult;
  
    int val = rand()%10;
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    std::sort( stdInput.begin( ), stdInput.end( ), std::greater< int >() );
    bolt::cl::sort( ctl, boltInput.begin( ), boltInput.end( ), bolt::cl::greater< int >());

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val, std::greater< int >() );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val, bolt::cl::greater< int >());

    EXPECT_EQ(stdresult, boltresult);
}

TEST_P( BSearchIntegerVector, MultiCoreGreater )
{
    bool stdresult, boltresult;
  
    int val = rand()%10;
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    std::sort( stdInput.begin( ), stdInput.end( ), std::greater< int >() );
    bolt::cl::sort( ctl, boltInput.begin( ), boltInput.end( ), bolt::cl::greater< int >());

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val, std::greater< int >() );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val, bolt::cl::greater< int >());

    EXPECT_EQ(stdresult, boltresult);
}


TEST_P( BSearchIntegerVector, Normal )
{
    bool stdresult, boltresult;
  
    int val = rand()%10;

    //  Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}

TEST_P( BSearchIntegerVector, SerialCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    int val = rand()%10;
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}

TEST_P( BSearchIntegerVector, MultiCoreCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    int val = rand()%10;

    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}


TEST_P( BSearchFloatVector, Normal )
{
    
    float val = (float) rand();
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}

TEST_P( BSearchFloatVector, SerialCPU)
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    float val = (float) rand();
  
    bool stdresult, boltresult;

    //Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}

TEST_P( BSearchFloatVector, MultiCoreCPU)
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    float val = (float) rand();
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}

#if (TEST_DOUBLE == 1)
TEST_P( BSearchDoubleVector, Normal )
{
    double val = (double) rand();
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}

TEST_P( BSearchDoubleVector, SerialNormal)
{
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    double val = (double) rand();
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}

TEST_P( BSearchDoubleVector, MulticoreNormal )
{   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
   
    double val = (double) rand();
    bool stdresult, boltresult;

    // Calling the actual functions under test
    stdresult = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val);

    EXPECT_EQ(stdresult, boltresult);
}

#endif
#if (TEST_DEVICE_VECTOR == 1)
TEST_P( BSearchIntegerDeviceVector, Normal )
{

    int val = rand()%10;
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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
      
        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); 
        boltresult = bolt::BKND::SEARCH_FUNC( boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); 

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchIntegerDeviceVector, SerialNormal )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    int val = rand()%10;
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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
        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); //, std::less< valtype >() );
        boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); //, bolt::cl::less< valtype >( ) );

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchIntegerDeviceVector, MultiCoreNormal )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

   
    int val = rand()%10;
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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

        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); //, std::less< valtype >() );
        boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); //, bolt::cl::less< valtype >( ) );

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchUDDDeviceVector, Normal )
{
    typedef std::vector< UDD >::value_type valtype;

   
    UDD val = gen_UDD_random();
 
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

    //OFFSET Test cases
    //  Calling the actual functions under test

    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = GetParam( ) -17; //Some arbitrary offset position
    int ArraySize = GetParam( );
    if( (( startIndex > GetParam( ) ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< GetParam( ) << "\n";
    }    
    else
    {
        
        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); 
        boltresult = bolt::BKND::SEARCH_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); 

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchUDDDeviceVector, SerialNormal)
{
    typedef std::vector< UDD >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    UDD val = gen_UDD_random();
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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
      
        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); 
        boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val);

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchUDDDeviceVector, MultiCoreNormal )
{
    typedef std::vector< UDD >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    UDD val = gen_UDD_random();
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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
        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); 
        boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); 

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchFloatDeviceVector, Normal )
{
    typedef std::vector< float >::value_type valtype;

    float val = (float) rand();
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC(boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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

        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); //, std::less< valtype >() );
        boltresult = bolt::BKND::SEARCH_FUNC( boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); //, bolt::cl::less< valtype >( ) );

        EXPECT_EQ( stdresult, boltresult );

    }
}

TEST_P( BSearchFloatDeviceVector, SerialNormal )
{
    typedef std::vector< float >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    float val = (float) rand();
   
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val, std::less< valtype >() );
    boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltInput.begin( ), boltInput.end( ), val, bolt::cl::less< valtype >( ) );

    EXPECT_EQ(stdresult, boltresult);

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
        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val, std::less< valtype >() );
        boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val, bolt::cl::less< valtype >( ) );

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchFloatDeviceVector, MultiCoreNormal )
{
    typedef std::vector< float >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    float val = (float) rand();
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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
       
        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); //, std::less< valtype >() );
        boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); //, bolt::cl::less< valtype >( ) );

        EXPECT_EQ( stdresult, boltresult );
    }
}

#if (TEST_DOUBLE == 1)
TEST_P(BSearchDoubleDeviceVector, Normal )
{
    typedef std::vector< double >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    double val = (double) rand();
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC(boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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
        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); 
        boltresult = bolt::BKND::SEARCH_FUNC( boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); 

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchDoubleDeviceVector, SerialNormal )
{
    typedef std::vector< double >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    double val = (double) rand();
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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

        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); //, std::less< valtype >() );
        boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); //, bolt::cl::less< valtype >( ) );

        EXPECT_EQ( stdresult, boltresult );
    }
}

TEST_P( BSearchDoubleDeviceVector, MulticoreNormal )
{
    typedef std::vector< double >::value_type valtype;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    double val = (double) rand();
    bool stdresult, boltresult;

    //  Calling the actual functions under test
    stdresult  = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
    boltresult = bolt::BKND::SEARCH_FUNC(ctl, boltInput.begin( ), boltInput.end( ), val );

    EXPECT_EQ(stdresult, boltresult);

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

        stdresult = std::SEARCH_FUNC( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex, val); 
        boltresult = bolt::BKND::SEARCH_FUNC( ctl, boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex, val); 

        EXPECT_EQ( stdresult, boltresult );
    }
}

#endif 
#endif

/*TEST_P( BSearchIntegerNakedPointer, Normal )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    bolt::cl::sort( wrapBoltInput, wrapBoltInput + endIndex );

    int val = rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC( wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}

TEST_P( BSearchIntegerNakedPointer, SerialNormal )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::sort( ctl, wrapBoltInput, wrapBoltInput + endIndex );
   
    int val = rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}

TEST_P( BSearchIntegerNakedPointer, MultiCoreNormal )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::sort( ctl, wrapBoltInput, wrapBoltInput + endIndex );

    int val = rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}


TEST_P( BSearchFloatNakedPointer, Normal )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );

    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::sort( wrapBoltInput, wrapBoltInput + endIndex );

    float val = (float) rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC( wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}

TEST_P( BSearchFloatNakedPointer, SerialNormal )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );

    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::sort( ctl, wrapBoltInput, wrapBoltInput + endIndex );

    float val = (float) rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}

TEST_P( BSearchFloatNakedPointer, MultiCoreNormal )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );

    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::sort( ctl, wrapBoltInput, wrapBoltInput + endIndex );

    float val = (float) rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC( ctl, wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}

#if (TEST_DOUBLE == 1)
TEST_P( BSearchDoubleNakedPointer, Normal )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );

    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::sort( wrapBoltInput, wrapBoltInput + endIndex );

    double val = (double) rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC( wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}

TEST_P( BSearchDoubleNakedPointer, SerialNormal )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::sort(ctl, wrapBoltInput, wrapBoltInput + endIndex );

    double val = (double) rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC(ctl, wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}

TEST_P( BSearchDoubleNakedPointer, MulticoreNormal )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    std::sort( wrapStdInput, wrapStdInput + endIndex );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
    bolt::BKND::sort( wrapBoltInput, wrapBoltInput + endIndex );
    
    double val = (double) rand();
    bool stdresult, boltresult;

    stdresult = std::SEARCH_FUNC( wrapStdInput, wrapStdInput + endIndex, val );
    boltresult = bolt::BKND::SEARCH_FUNC(ctl, wrapBoltInput, wrapBoltInput + endIndex, val );

    EXPECT_EQ( stdresult, boltresult );
}


#endif
*/

std::array<int, 10> TestValues = {2,4,8,16,32,64,128,256,512,1024};
std::array<int, 5> TestValues2 = {2048,4096,8192,16384,32768};
//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier

INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P( BSearchValues, BSearchIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                                                                            TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16	
INSTANTIATE_TEST_CASE_P( BSearchValues, BSearchFloatVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                        TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( BSearchValues, BSearchDoubleVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                            TestValues.end() ) );
//#if (TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( BSearchValues2, BSearchDoubleVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                            TestValues2.end() ) );
//#endif
#endif

INSTANTIATE_TEST_CASE_P(BSearchRange, BSearchCountingIterator, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P( BSearchValues,BSearchCountingIterator, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );

INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( BSearchValues, BSearchIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchUDDDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( BSearchValues, BSearchUDDDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( BSearchValues, BSearchFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                TestValues.end()));
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchDoubleDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( BSearchValues, BSearchDoubleDeviceVector, ::testing::ValuesIn(TestValues.begin(),
                                                                                    TestValues.end()));
//#if (TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( BSearchValues2, BSearchDoubleDeviceVector, ::testing::ValuesIn(TestValues2.begin(),
                                                                                    TestValues2.end()));
//#endif
#endif
INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchIntegerNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( BSearchValues, BSearchIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                                    TestValues.end()));
INSTANTIATE_TEST_CASE_P( BSearchtRange, BSearchFloatNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P(BSearchValues, BSearchFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( BSearchRange, BSearchDoubleNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( BSearch, BSearchDoubleNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                     TestValues.end() ) );
//#if (TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( BSearch2, BSearchDoubleNakedPointer, ::testing::ValuesIn( TestValues2.begin(),
                                                                     TestValues2.end() ) );
//#endif
#endif


//#if (TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( BSearchValues2, BSearchIntegerVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                            TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( BSearchValues2, BSearchFloatVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                        TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( BSearchValues2,BSearchCountingIterator, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( BSearchValues2, BSearchIntegerDeviceVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( BSearchValues2, BSearchUDDDeviceVector, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( BSearchValues2, BSearchFloatDeviceVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                TestValues2.end()));
INSTANTIATE_TEST_CASE_P( BSearchValues2, BSearchIntegerNakedPointer, ::testing::ValuesIn( TestValues2.begin(),
                                                                                    TestValues2.end()));
INSTANTIATE_TEST_CASE_P(BSearchValues2, BSearchFloatNakedPointer, ::testing::ValuesIn( TestValues2.begin(), 
                                                                                TestValues2.end() ) );
//#endif

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



//template< typename ArrayTuple >
//class BSearchUDDArrayTest: public ::testing::Test
//{
//public:
//    BSearchUDDArrayTest( ): m_Errors( 0 )
//    {}
//
//    
//    virtual void SetUp( )
//    {
//        std::generate(stdInput.begin(), stdInput.end(), rand);
//        boltInput = stdInput;
//    };
//
//    virtual void TearDown( )
//    {};
//
//    virtual ~BSearchUDDArrayTest( )
//    {}
//
//protected:
//    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
//    static const size_t ArraySize = typename std::tuple_element< 1, ArrayTuple >::type::value;
//    typename std::array< ArrayType, ArraySize > stdInput, boltInput;
//    int m_Errors;
//};
//
//TYPED_TEST_CASE_P( BSearchUDDArrayTest );
//
//TYPED_TEST_P(BSearchUDDArrayTest, Normal )
//{
//    typedef std::array< ArrayType, ArraySize > ArrayCont;
//    //  Calling the actual functions under test
//    bool std_result, bolt_result;
//    UDD val = gen_UDD_random();
//
//    std::sort( stdInput.begin( ), stdInput.end( ), UDD() );
//    bolt::BKND::sort( boltInput.begin( ), boltInput.end( ), UDD() );
//
//    std_result = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
//    bolt_result = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), val );
//
//    EXPECT_EQ( std_result, bolt_result);
//
//    std::sort( stdInput.begin( ), stdInput.end( ), BSearchBy_UDD_a() );
//    bolt::BKND::sort( boltInput.begin( ), boltInput.end( ), BSearchBy_UDD_a() );
//
//    std_result = std::SEARCH_FUNC( stdInput.begin( ), stdInput.end( ), val );
//    bolt_result = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), val );
//   
//    EXPECT_EQ( std_result, bolt_result);
//
//}
//
//typedef ::testing::Types< 
//    std::tuple< UDD, TypeValue< 1 > >,
//    std::tuple< UDD, TypeValue< 31 > >,
//    std::tuple< UDD, TypeValue< 32 > >,
//    std::tuple< UDD, TypeValue< 63 > >,
//    std::tuple< UDD, TypeValue< 64 > >,
//    std::tuple< UDD, TypeValue< 127 > >,
//    std::tuple< UDD, TypeValue< 128 > >,
//    std::tuple< UDD, TypeValue< 129 > >,
//    std::tuple< UDD, TypeValue< 1000 > >,
//    std::tuple< UDD, TypeValue< 1053 > >,
//    std::tuple< UDD, TypeValue< 4096 > >,
//    std::tuple< UDD, TypeValue< 4097 > >,
//    std::tuple< UDD, TypeValue< 65535 > >,
//    std::tuple< UDD, TypeValue< 65536 > >
//> UDDTests;
//
//typedef ::testing::Types< 
//    std::tuple< cl_ushort, TypeValue< 1 > >,
//    std::tuple< cl_ushort, TypeValue< 31 > >,
//    std::tuple< cl_ushort, TypeValue< 32 > >,
//    std::tuple< cl_ushort, TypeValue< 63 > >,
//    std::tuple< cl_ushort, TypeValue< 64 > >,
//    std::tuple< cl_ushort, TypeValue< 127 > >,
//    std::tuple< cl_ushort, TypeValue< 128 > >,
//    std::tuple< cl_ushort, TypeValue< 129 > >,
//    std::tuple< cl_ushort, TypeValue< 1000 > >,
//    std::tuple< cl_ushort, TypeValue< 1053 > >,
//    std::tuple< cl_ushort, TypeValue< 4096 > >,
//    std::tuple< cl_ushort, TypeValue< 4097 > >,
//    std::tuple< cl_ushort, TypeValue< 65535 > >,
//    std::tuple< cl_ushort, TypeValue< 65536 > >
//> cl_ushortTests;
//
//typedef ::testing::Types< 
//    std::tuple< cl_short, TypeValue< 1 > >,
//    std::tuple< cl_short, TypeValue< 31 > >,
//    std::tuple< cl_short, TypeValue< 32 > >,
//    std::tuple< cl_short, TypeValue< 63 > >,
//    std::tuple< cl_short, TypeValue< 64 > >,
//    std::tuple< cl_short, TypeValue< 127 > >,
//    std::tuple< cl_short, TypeValue< 128 > >,
//    std::tuple< cl_short, TypeValue< 129 > >,
//    std::tuple< cl_short, TypeValue< 1000 > >,
//    std::tuple< cl_short, TypeValue< 1053 > >,
//    std::tuple< cl_short, TypeValue< 4096 > >,
//    std::tuple< cl_short, TypeValue< 4097 > >,
//    std::tuple< cl_short, TypeValue< 65535 > >,
//    std::tuple< cl_short, TypeValue< 65536 > >
//> cl_shortTests;
//
//
//INSTANTIATE_TYPED_TEST_CASE_P( cl_ushort, BSearchArrayTest, cl_ushortTests );
//INSTANTIATE_TYPED_TEST_CASE_P( cl_short, BSearchArrayTest, cl_shortTests );
//INSTANTIATE_TYPED_TEST_CASE_P( Integer, BSearchArrayTest, IntegerTests );
//INSTANTIATE_TYPED_TEST_CASE_P( UnsignedInteger, BSearchArrayTest, UnsignedIntegerTests );
//INSTANTIATE_TYPED_TEST_CASE_P( Float, BSearchArrayTest, FloatTests );
//INSTANTIATE_TYPED_TEST_CASE_P( clLong, BSearchArrayTest, clLongTests );
//#if (TEST_DOUBLE == 1)
//INSTANTIATE_TYPED_TEST_CASE_P( Double, BSearchArrayTest, DoubleTests );
//#endif 
//REGISTER_TYPED_TEST_CASE_P( BSearchUDDArrayTest,  Normal);
//INSTANTIATE_TYPED_TEST_CASE_P( UDDTest, BSearchUDDArrayTest, UDDTests );

#if (TEST_LARGE_BUFFERS == 1)
TEST (withStdVect, intSerialValuesWithDefaulFunctorWithClControl){

    int sizeOfInputBuffer = 2048;

    std::vector <int> my_vect(sizeOfInputBuffer);
    std::vector <int> std_vect(sizeOfInputBuffer);

    for (int i = 0 ; i < sizeOfInputBuffer; ++i){
        my_vect[i] = rand() % 65535;
        std_vect[i] = my_vect[i];
    }

    // Create an OCL context, device, queue.
    MyOclContext ocl = initOcl(CL_DEVICE_TYPE_GPU, 0); //zero stand for one GPU

    bolt::cl::control c(ocl._queue);  // construct control structure from the queue.
    
    std::sort(std_vect.begin(), std_vect.end());
    bolt::BKND::sort(c, my_vect.begin(), my_vect.end());
    
    bool stdresult, myresult;
    int my_val, std_val;

    for (int i = 0 ; i < sizeOfInputBuffer; ++i)
    {
        my_val = my_vect[i];
        std_val = std_vect[i];
       
        stdresult = std::binary_search(std_vect.begin(), std_vect.end(), std_val);
        myresult = bolt::BKND::SEARCH_FUNC(c, my_vect.begin(), my_vect.end(), my_val);

        EXPECT_EQ(stdresult, myresult)<<"Failed at i = "<<i<<std::endl;
    }
} 

TEST (withStdVect_Greater, intSerialValuesWithDefaulFunctorWithClControlGreater){

    int sizeOfInputBuffer = 2048;

    std::vector <int> my_vect(sizeOfInputBuffer);
    std::vector <int> std_vect(sizeOfInputBuffer);

    for (int i = 0 ; i < sizeOfInputBuffer; ++i){
        my_vect[i] = rand() % 65535;
        std_vect[i] = my_vect[i];
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

    std::sort(std_vect.begin(), std_vect.end(), std::greater<int>());
    bolt::BKND::sort(boltControl, my_vect.begin(), my_vect.end(), bolt::cl::greater<int>());
    
    bool stdresult, myresult;
    int my_val, std_val;

    for (int i = 0 ; i < sizeOfInputBuffer; ++i)
    {
        my_val = my_vect[i];
        std_val = std_vect[i];
       
        stdresult = std::binary_search(std_vect.begin(), std_vect.end(), std_val,std::greater<int>() );
        myresult = bolt::BKND::SEARCH_FUNC(boltControl, my_vect.begin(), my_vect.end(), my_val, bolt::cl::greater<int>());

        EXPECT_EQ(stdresult, myresult)<<"Failed at i = "<<i<<std::endl;
    }
}
#endif

#if (TEST_LARGE_BUFFERS == 1)
TEST (rawArrayTest, floatarray){
	const int sizeOfInputBufer = 8192; 
	float  stdArray[sizeOfInputBufer];
    float  boltArray[sizeOfInputBufer];
    float  backupArray[sizeOfInputBufer];

	for (int i = 0 ; i < sizeOfInputBufer; i++){
	    float fValue = (float)rand();
        fValue = fValue/rand();
        fValue = fValue*rand()*rand();
        stdArray[i] = boltArray[i] = fValue;
	}
	std::sort( stdArray, stdArray+sizeOfInputBufer, std::greater<float>( ) );
	bolt::BKND::sort( boltArray, boltArray+sizeOfInputBufer, bolt::cl::greater<float>( ) );

    bool stdresult, myresult;
    float my_val, std_val;

	for (int i = 0 ; i < sizeOfInputBufer; i++)
    {
        my_val = boltArray[i];
        std_val = stdArray[i];
       
        stdresult = std::binary_search(stdArray, stdArray+sizeOfInputBufer, std_val, std::greater<float>( ) );
        myresult = bolt::BKND::SEARCH_FUNC( boltArray, boltArray+sizeOfInputBufer, my_val, bolt::cl::greater<float>( ) );

        EXPECT_FLOAT_EQ(stdresult, myresult)<<"Failed at i = "<<i<<std::endl;

	}

    //Offset tests 
	for (int i = 0 ; i < sizeOfInputBufer; i++)
    {
	   stdArray[i] = boltArray[i] = backupArray[i];
	}

	std::sort( stdArray+17, stdArray+sizeOfInputBufer-129, std::greater<float>( ) );
	bolt::BKND::sort( boltArray+17, boltArray+sizeOfInputBufer-129, bolt::cl::greater<float>( ) );

	for (int i = 0 ; i < sizeOfInputBufer; i++)
    {
        my_val = boltArray[i];
        std_val = stdArray[i];
       
        stdresult = std::binary_search(stdArray+17, stdArray+sizeOfInputBufer-129, std_val, std::greater<float>( ) );
        myresult = bolt::BKND::SEARCH_FUNC( boltArray+17, boltArray+sizeOfInputBufer-129, my_val, bolt::cl::greater<float>( ) );

        EXPECT_FLOAT_EQ(stdresult, myresult)<<"Offset Test Failed at i = "<<i<<std::endl;

	}

}
#endif

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

#include "bolt/cl/iterator/counting_iterator.h"
#include <bolt/cl/sort.h>
#include "bolt/cl/binary_search.h"
#include <bolt/cl/functional.h>
#undef NOMINMAX
#include <array>
#include <algorithm>
#include <random>

int random_gen()
{
    return -rand();
}

int main ()
{
    const int ArraySize = 8192;
    typedef std::array< int, ArraySize > ArrayCont;
    ArrayCont stdOffsetIn,stdInput;
    ArrayCont boltOffsetIn,boltInput;
    int minimum = -31000;
    int maximum = 31000;
    std::default_random_engine rnd;
    std::generate(stdInput.begin(), stdInput.end(),  random_gen);
    boltInput = stdInput;
    boltOffsetIn = stdInput;
    stdOffsetIn = stdInput;

	bool stdresult, myresult;

    //  Calling the actual functions under test
    std::sort( stdInput.begin( ), stdInput.end( ) );
    bolt::BKND::sort( boltInput.begin( ), boltInput.end( ) );
    /*for(int i=0;i< ArraySize;i++)
    {
            std::cout << stdInput[i]  << "\n";
    }*/
 
    int my_val, std_val;

    //  Loop through the array and compare all the values with each other
    for(int i=0;i< ArraySize;i++)
    {
 
        my_val =  boltInput[i];
        std_val = stdInput[i];
       
        stdresult = std::binary_search(stdInput.begin( ),  stdInput.end( ),  std_val );
        myresult = bolt::BKND::SEARCH_FUNC( boltInput.begin( ), boltInput.end( ), my_val);

        if(stdresult == myresult)
            continue; 
        else
            std::cout<<"Test Failed at i = "<<i<<std::endl;
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
        std::sort( stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex );
        bolt::BKND::sort( boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex );

        //  Loop through the array and compare all the values with each other
        for(int i=0;i< ArraySize;i++)
        {
            my_val =  boltOffsetIn[i];
            std_val = stdOffsetIn[i];
       
            stdresult = std::binary_search(stdOffsetIn.begin( ) + startIndex, stdOffsetIn.begin( ) + endIndex ,  std_val );
            myresult = bolt::BKND::SEARCH_FUNC(boltOffsetIn.begin( ) + startIndex, boltOffsetIn.begin( ) + endIndex , my_val);

            if(stdresult == myresult)
                continue;
            else
                std::cout<<"Offset Test Failed at i = "<<i<<std::endl;
        }
    }


}
#endif
