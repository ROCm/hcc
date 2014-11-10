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
#define TEST_LARGE_BUFFERS 0

#pragma warning(disable: 4244) // Disabling possible loss of data warning
#if defined (_WIN32)
#include <xutility>
#endif

#include "bolt/cl/iterator/counting_iterator.h"

#include <bolt/cl/copy.h>
#include <bolt/cl/functional.h>
#include <bolt/miniDump.h>
#include <bolt/cl/device_vector.h>

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>

#include "common/stdafx.h"
#include "common/myocl.h"

#define BOLT_TEST_MAX_FAILURES 1
#include "test_common.h"

//  Author: kfk
//  Originally, the char b was not an array.  However, testing on my machine showed driver hangs and unit test
//  failures.  Reading the documentation of clEnqueueFillBuffer(), OpenCL seems to like structs of certain 
//  sizes {1, 2, 4, 8, 16, 32, 64, 128}.  Making this struct one of those magic sizes seems to make the driver
//  resets go away.
//  \todo:  Need to research and find a way to handle structs of arbitrary size.
BOLT_FUNCTOR(UserStruct,
struct UserStruct
{
    bool a;
    char b[ 4 ];
    int c;
    float d;
    //double e;

    UserStruct():
        a(true),
        c(3),
        d(4.f)//,
        //e(5.0)
    {
        for( unsigned i = 0; i < 4; ++i )
            b[ i ] = 0;
    }

    bool operator==(const UserStruct& rhs) const
    {
        return
            (a == rhs.a) &&
            //(b == rhs.b) &&
            (c == rhs.c) &&
            (d == rhs.d) //&&
            //(e == rhs.e)
            ;
    }

};
);  // end BOLT_FUNCTOR

BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, int, UserStruct);

/******************************************************************************
 * Tests
 *****************************************************************************/
// device_vector vs std_vector
// copy vs copy_n
// primitive vs struct
// mult64 vs not mult
// zero vs positive

static const int numLengths = 24;
static const int lengths[24] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 63, 64,
    65, 1023, 1024, 1025 };
//static const int numLengths = 1;
//static const int lengths[1] = {13};



template< size_t N >
class TypeValue
{
public:
    static const size_t value = N;
};


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
class CopyArrayTest: public ::testing::Test
{
public:
    CopyArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), rand);
        std::generate(stdOffsetOut.begin(), stdOffsetOut.end(), rand);
        boltInput = stdInput;
        stdOffsetIn = stdInput;
        boltOffsetIn = stdInput;
        boltOffsetOut =  stdOffsetOut;
    };

    virtual void TearDown( )
    {};

    virtual ~CopyArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    std::array< ArrayType, ArraySize > stdInput, boltInput, stdOffsetIn, boltOffsetIn;
    std::array< ArrayType, ArraySize > stdOutput, boltOutput, stdOffsetOut, boltOffsetOut;
    int m_Errors;
};



TYPED_TEST_CASE_P( CopyArrayTest );

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( CopyArrayTest,CPU_DeviceNormal )
{

    typedef typename CopyArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, CopyArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c_cpu(oclcpu._queue);  // construct control structure from the queue.

    //  Calling the actual functions under test
    std::copy( CopyArrayTest< gtest_TypeParam_ >::stdInput.begin( ), CopyArrayTest< gtest_TypeParam_ >::stdInput.end( ), CopyArrayTest< gtest_TypeParam_ >::stdOutput.begin() );
    bolt::cl::copy( c_cpu, CopyArrayTest< gtest_TypeParam_ >::boltInput.begin( ), CopyArrayTest< gtest_TypeParam_ >::boltInput.end( ) , CopyArrayTest< gtest_TypeParam_ >::boltOutput.begin());

    typename ArrayCont::difference_type stdNumElements = std::distance( CopyArrayTest< gtest_TypeParam_ >::stdOutput.begin( ), CopyArrayTest< gtest_TypeParam_ >::stdOutput.end() );
     typename ArrayCont::difference_type boltNumElements = std::distance( CopyArrayTest< gtest_TypeParam_ >::boltOutput.begin( ), CopyArrayTest< gtest_TypeParam_ >::boltOutput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, CopyArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( CopyArrayTest< gtest_TypeParam_ >::stdOutput, CopyArrayTest< gtest_TypeParam_ >::boltOutput );
    
    //OFFSET Test cases
    //  Calling the actual functions under test
    size_t startIndex = 17; //Some aribitrary offset position
    size_t endIndex   = CopyArrayTest< gtest_TypeParam_ >::ArraySize - 17; //Some aribitrary offset position
    if( (( startIndex > CopyArrayTest< gtest_TypeParam_ >::ArraySize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< CopyArrayTest< gtest_TypeParam_ >::ArraySize << "\n";
    }    
    else
    {

        //std::copy( CopyArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) , CopyArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, CopyArrayTest< gtest_TypeParam_ >::stdOffsetOut.begin() + endIndex);
      //  std::copy( CopyArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ), CopyArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( )+ startIndex, CopyArrayTest< gtest_TypeParam_ >::boltOffsetOut.begin( )+ endIndex);


        std::copy( CopyArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + startIndex, CopyArrayTest< gtest_TypeParam_ >::stdOffsetIn.begin( ) + endIndex, CopyArrayTest< gtest_TypeParam_ >::stdOffsetOut.begin() );
        bolt::cl::copy( c_cpu, CopyArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + startIndex, CopyArrayTest< gtest_TypeParam_ >::boltOffsetIn.begin( ) + endIndex, CopyArrayTest< gtest_TypeParam_ >::boltOffsetOut.begin() );

         typename ArrayCont::difference_type stdNumElements = std::distance( CopyArrayTest< gtest_TypeParam_ >::stdOffsetOut.begin( ), CopyArrayTest< gtest_TypeParam_ >::stdOffsetOut.end( ) );
         typename ArrayCont::difference_type boltNumElements = std::distance(  CopyArrayTest< gtest_TypeParam_ >::boltOffsetOut.begin( ),  CopyArrayTest< gtest_TypeParam_ >::boltOffsetOut.end( ) );

        //  Both collections should have the same number of elements
        EXPECT_EQ( stdNumElements, boltNumElements );

        //  Loop through the array and compare all the values with each other
        cmpStdArray< ArrayType, CopyArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays(  CopyArrayTest< gtest_TypeParam_ >::stdOffsetOut,  CopyArrayTest< gtest_TypeParam_ >::boltOffsetOut );
    }
}







REGISTER_TYPED_TEST_CASE_P( CopyArrayTest, CPU_DeviceNormal);

INSTANTIATE_TYPED_TEST_CASE_P( clLong, CopyArrayTest, clLongTests );
INSTANTIATE_TYPED_TEST_CASE_P( Integer, CopyArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( UnsignedInteger, CopyArrayTest, UnsignedIntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, CopyArrayTest, FloatTests );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TYPED_TEST_CASE_P( Double, CopyArrayTest, DoubleTests );
#endif 

#endif

TEST( CopyStdVectWithInt, OffsetTest)
{
    int length = 1024;

    std::vector<int> stdInput( length );
    std::vector<int> stdOutput( length );
    std::vector<int> boltInput( length );
    std::vector<int> boltOutput( length );


    for (int i = 0; i < length; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int offset = 100;
    std::copy(stdInput.begin( ) + offset, stdInput.end( ), stdOutput.begin() );
    bolt::cl::copy( boltInput.begin( ) + offset, boltInput.end( ), boltOutput.begin());

    cmpArrays(stdOutput, boltOutput);
}


TEST( CopyStdVectWithIntFloat, OffsetTest)
{
    int length = 1024;

    std::vector<int> stdInput( length );
    std::vector<float> stdOutput( length );
    std::vector<int> boltInput( length );
    std::vector<float> boltOutput( length );


    for (int i = 0; i < length; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int offset = 100;
    std::copy(stdInput.begin( ) + offset, stdInput.end( ), stdOutput.begin() );
    bolt::cl::copy( boltInput.begin( ) + offset, boltInput.end( ), boltOutput.begin());

    cmpArrays(stdOutput, boltOutput);
}

TEST( CopyDevVectWithInt, OffsetTest)
{
    int length = 1024;

    std::vector<int> stdInput( length );
    std::vector<int> stdOutput( length );
    bolt::cl::device_vector<int> boltInput( length );
    bolt::cl::device_vector<int> boltOutput( length );


    for (int i = 0; i < length; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int offset = 100;
    std::copy(stdInput.begin( ) + offset, stdInput.end( ), stdOutput.begin() );
    bolt::cl::copy( boltInput.begin( ) + offset, boltInput.end( ), boltOutput.begin());

    cmpArrays(stdOutput, boltOutput);
}

TEST( CopyDevVectWithIntFloat, OffsetTest)
{
    int length = 1024;

    std::vector<int> stdInput( length );
    std::vector<float> stdOutput( length );
    bolt::cl::device_vector<int> boltInput( length );
    bolt::cl::device_vector<float> boltOutput( length );



    for (int i = 0; i < length; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int offset = 100;
    std::copy(stdInput.begin( ) + offset, stdInput.end( ), stdOutput.begin() );
    bolt::cl::copy( boltInput.begin( ) + offset, boltInput.end( ), boltOutput.begin());

    cmpArrays(stdOutput, boltOutput);
}


TEST(Copy, FancyDeviceIterator) 
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);
        
        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector

        std::vector<int> dest(length);

        //Bolt destination vector
        bolt::cl::device_vector<int> destination(length);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 


TEST(Copy, SerialFancyDeviceIterator) 
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector

        std::vector<int> dest(length);

        //Bolt destination vector
        bolt::cl::device_vector<int> destination(length);

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(ctl, first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, MultiCoreFancyDeviceIterator) 
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector

        std::vector<int> dest(length);

        //Bolt destination vector
        bolt::cl::device_vector<int> destination(length);

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(ctl, first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, FancyDeviceIntFloat) 
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector

        std::vector<float> dest(length);

        //Bolt destination vector
        bolt::cl::device_vector<float> destination(length);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, SerialFancyDeviceIntFloat) 
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu);

        //STL destination vector

        std::vector<float> dest(length);

        //Bolt destination vector
        bolt::cl::device_vector<float> destination(length);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(ctl, first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, MultiCoreFancyDeviceIntFloat) 
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

        //STL destination vector

        std::vector<float> dest(length);

        //Bolt destination vector
        bolt::cl::device_vector<float> destination(length);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(ctl, first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, FancyRandomIterator)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector
        std::vector<int> dest(length);
        //Bolt destination vector
        std::vector<int> destination(length);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, SerialFancyRandomIterator)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector
        std::vector<int> dest(length);
        //Bolt destination vector
        std::vector<int> destination(length);

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(ctl, first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, MultiCoreFancyRandomIterator)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector
        std::vector<int> dest(length);
        //Bolt destination vector
        std::vector<int> destination(length);

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(ctl, first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, FancyRandomIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector
        std::vector<float> dest(length);
        //Bolt destination vector
        std::vector<float> destination(length);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
} 

TEST(Copy, SerialFancyRandomIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector
        std::vector<float> dest(length);
        //Bolt destination vector
        std::vector<float> destination(length);

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(ctl, first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
}

TEST(Copy, MultiCoreFancyRandomIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector
        std::vector<float> dest(length);
        //Bolt destination vector
        std::vector<float> destination(length);

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(ctl, first, last, destination.begin());
       
        // GoogleTest Comparison
        cmpArrays(dest, destination);
    }
}

//Copy with Fancy Iterator as destination results in Complilation Failure!
/*TEST(Copy, RandomFancyIterator)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        //Bolt Destination fancy vector
        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
        }; 

        //STL destination vector
        std::vector<int> dest(length);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(a.begin(), a.end(), first); // This is logically wrong!
       
        // GoogleTest Comparison
        //cmpArrays(dest, first, length);
    }
}

TEST(Copy, DeviceFancyIterator)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        //Bolt Destination fancy vector
        bolt::cl::counting_iterator<int> first(0);
        bolt::cl::counting_iterator<int> last = first + length;

        std::vector<int> a(length);
        bolt::cl::device_vector<int> b(length);

        for (int i=0; i < length; i++) {
            a[i] = i;
            b[i] = a[i];
        }; 

        //STL destination vector
        std::vector<int> dest(length);

        // perform copy
        std::copy(a.begin(), a.end(), dest.begin());
        bolt::cl::copy(b.begin(), b.end(), first); // This is logically wrong!
       
        // GoogleTest Comparison
        //cmpArrays(dest, first, length);
    }
}*/

TEST(Copy, DeviceRandom)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> src(length);
        std::vector<int> dest(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        // perform copy
        bolt::cl::copy(src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, SerialDeviceRandom)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> src(length);
        std::vector<int> dest(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

        // perform copy
        bolt::cl::copy(ctl, src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, MultiCoreDeviceRandom)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> src(length);
        std::vector<int> dest(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

        // perform copy
        bolt::cl::copy(ctl, src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, DeviceRandomIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> src(length);
        std::vector<float> dest(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        // perform copy
        bolt::cl::copy(src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, SerialDeviceRandomIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> src(length);
        std::vector<float> dest(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

        // perform copy
        bolt::cl::copy(ctl, src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, MultiCoreDeviceRandomIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> src(length);
        std::vector<float> dest(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

        // perform copy
        bolt::cl::copy(ctl, src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, RandomDevice)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> dest(length);
        std::vector<int> src(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        // perform copy
        bolt::cl::copy(src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, SerialRandomDevice)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> dest(length);
        std::vector<int> src(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

        // perform copy
        bolt::cl::copy(ctl, src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, MultiCoreRandomDevice)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<int> dest(length);
        std::vector<int> src(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

        // perform copy
        bolt::cl::copy(ctl, src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, RandomDeviceIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<float> dest(length);
        std::vector<int> src(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        // perform copy
        bolt::cl::copy(src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, SerialRandomDeviceIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<float> dest(length);
        std::vector<int> src(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

        // perform copy
        bolt::cl::copy(ctl, src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, MultiCoreRandomDeviceIntFloat)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];

        bolt::cl::device_vector<float> dest(length);
        std::vector<int> src(length);

        for (int i=0; i < length; i++) {
            src[i] = i;
        }; 

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

        // perform copy
        bolt::cl::copy(ctl, src.begin(), src.end(), dest.begin());
       
        // GoogleTest Comparison
        cmpArrays(src, dest);
    }
} 

TEST(Copy, DevclLong)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<cl_long> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = (cl_long)rand();
        }

        // destination vector
        bolt::cl::device_vector<cl_long> destination(length);
        // perform copy
        bolt::cl::copy(source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 

TEST(Copy, SerialDevclLong)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<cl_long> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = (cl_long)rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

        // destination vector
        bolt::cl::device_vector<cl_long> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 

TEST(Copy, MultiCoreDevclLong)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<cl_long> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = (cl_long)rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

        // destination vector
        bolt::cl::device_vector<cl_long> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 

TEST(Copy, DevUnsignedInt)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<unsigned int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = abs( rand());
        }

       // destination vector
        bolt::cl::device_vector<unsigned int > destination(length);
        // perform copy
        bolt::cl::copy(source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 

TEST(Copy, SerialDevUnsignedInt)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<unsigned int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = abs(rand());
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

        // destination vector
        bolt::cl::device_vector<unsigned int > destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 

TEST(Copy, MultiCoreDevUnsignedInt)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<unsigned int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = abs(rand());
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

        // destination vector
        bolt::cl::device_vector<unsigned int > destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 

TEST(Copy, DevShort)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<short> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

       // destination vector
        bolt::cl::device_vector<short> destination(length);
        // perform copy
        bolt::cl::copy(source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 


TEST(Copy, SerialDevShort)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<short> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

       // destination vector
        bolt::cl::device_vector<short> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 

TEST(Copy, MultiCoreDevShort)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<short> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

       // destination vector
        bolt::cl::device_vector<short> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination);

    }
} 


TEST(Copy, DevPrim)  // Default code path
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

       // destination vector
        bolt::cl::device_vector<int> destination(length);
        // perform copy
        bolt::cl::copy(source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination);
    }
} 

TEST(Copy, AutomaticDevPrim) //Automatic Code Path
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::Automatic); 

        // destination vector
        bolt::cl::device_vector<int> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination);
    }
}

TEST(Copy, SerialDevPrim) // Serial Code Path
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }
        
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
        
        // destination vector
        bolt::cl::device_vector<int> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination);
    }
}

TEST(Copy, MultiCoreDevPrim) // MultiCore CPU - TBB Path
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }
        
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
        
        // destination vector
        bolt::cl::device_vector<int> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination);
    }
}

TEST(CopyN, DevclLong)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<cl_long> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = (cl_long)rand();
        }

        // destination vector
        bolt::cl::device_vector<cl_long> destination(length);
        // perform copy
        bolt::cl::copy_n(source.begin(), length, destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination, length);

    }
} 

TEST(CopyN, SerialDevclLong)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<cl_long> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = (cl_long)rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

        // destination vector
        bolt::cl::device_vector<cl_long> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination, length);

    }
} 

TEST(CopyN, MultiCoreDevclLong)  
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<cl_long> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = (cl_long)rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

        // destination vector
        bolt::cl::device_vector<cl_long> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison

        cmpArrays(source, destination, length);

    }
} 

TEST(CopyN, DevPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }
        
        // destination vector
        bolt::cl::device_vector<int> destination(length);
        // perform copy
        bolt::cl::copy_n(source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, AutomaticDevPrim)
{    
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }
    
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::Automatic); 
        
        // destination vector
        bolt::cl::device_vector<int> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, SerialDevPrim)
{    
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }
    
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
        
        // destination vector
        bolt::cl::device_vector<int> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, MultiCoreDevPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

        // destination vector
        bolt::cl::device_vector<int> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}



TEST(Copy, StdPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }
      
        // destination vector
        std::vector<int> destination(length);
        // perform copy
        bolt::cl::copy(source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination);
    }
}

TEST(Copy, AutomaticStdPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::Automatic); 

        // destination vector
        std::vector<int> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination);
    }
}

TEST(Copy, SerialStdPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
        
        // destination vector
        std::vector<int> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination);
    }
}

TEST(Copy, MultiCoreStdPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
         
        // destination vector
        std::vector<int> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination);
    }
}



TEST(CopyN, StdPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        // destination vector
        std::vector<int> destination(length);
        // perform copy
        bolt::cl::copy_n(source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, AutomaticStdPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::Automatic); 
        
        // destination vector
        std::vector<int> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, SerialStdPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
        
        // destination vector
        std::vector<int> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, MultiCoreStdPrim)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<int> source(length);
        for (int j = 0; j < length; j++)
        {
            source[j] = rand();
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
                
        // destination vector
        std::vector<int> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}


TEST(Copy, DevStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        // destination vector
        bolt::cl::device_vector<UserStruct> destination(length, UserStruct(), CL_MEM_READ_WRITE, false);
        // perform copy
        bolt::cl::copy(source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(Copy, AutomaticDevStruct)
{
        for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::Automatic); // tested with serial also
                
        // destination vector
        bolt::cl::device_vector<UserStruct> destination(length, UserStruct(), CL_MEM_READ_WRITE, false);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(Copy, SerialDevStruct)
{
        for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
                
        // destination vector
        bolt::cl::device_vector<UserStruct> destination(length, UserStruct(), CL_MEM_READ_WRITE, false);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(Copy, MultiCoreDevStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
                
        // destination vector
        bolt::cl::device_vector<UserStruct> destination(length, UserStruct(), CL_MEM_READ_WRITE, false);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}


TEST(CopyN, DevStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
                
        // destination vector
        bolt::cl::device_vector<UserStruct> destination(length, UserStruct(), CL_MEM_READ_WRITE, false);
        // perform copy
        bolt::cl::copy_n(source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, AutomaticDevStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::Automatic);
                
        // destination vector
        bolt::cl::device_vector<UserStruct> destination(length, UserStruct(), CL_MEM_READ_WRITE, false);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, SerialDevStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
                
        // destination vector
        bolt::cl::device_vector<UserStruct> destination(length, UserStruct(), CL_MEM_READ_WRITE, false);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, MultiCoreDevStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        bolt::cl::device_vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
                
        // destination vector
        bolt::cl::device_vector<UserStruct> destination(length, UserStruct(), CL_MEM_READ_WRITE, false);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}



TEST(Copy, StdStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                // destination vector
        std::vector<UserStruct> destination(length);
        // perform copy
        bolt::cl::copy(source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(Copy, AutomaticStdStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::Automatic); 
                
        // destination vector
        std::vector<UserStruct> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(Copy, SerialStdStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
                
        // destination vector
        std::vector<UserStruct> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(Copy, MultiCoreStdStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
                
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
                
        // destination vector
        std::vector<UserStruct> destination(length);
        // perform copy
        bolt::cl::copy(ctl, source.begin(), source.end(), destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}


TEST(CopyN, StdStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
        
        // destination vector
        std::vector<UserStruct> destination(length);
        // perform copy
        bolt::cl::copy_n(source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, AutomaticStdStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }

        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::Automatic); 
    
        // destination vector
        std::vector<UserStruct> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, SerialStdStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
    
        bolt::cl::control ctl = bolt::cl::control::getDefault( );
        ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    
        // destination vector
        std::vector<UserStruct> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}

TEST(CopyN, MultiCoreStdStruct)
{
    for (int i = 0; i < numLengths; i++)
    {
        // test length
        int length = lengths[i];
        // populate source vector with random ints
        std::vector<UserStruct> source;
        for (int j = 0; j < length; j++)
        {
            UserStruct us;
            us.a = (bool) (rand()%2 ? true : false);
            us.b[0] = (char) (rand()%128);
            us.b[1] = (char) (rand()%128);
            us.b[2] = (char) (rand()%128);
            us.b[3] = (char) (rand()%128);
            us.c = (int)  (rand());
            us.d = (float) (1.f*rand());
            //us.e = (double) (1.0*rand()/rand());
            source.push_back(us);
        }
    
         bolt::cl::control ctl = bolt::cl::control::getDefault( );
         ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    
        // destination vector
        std::vector<UserStruct> destination(length);
        // perform copy
        bolt::cl::copy_n(ctl, source.begin(), length, destination.begin());
        // GoogleTest Comparison
        cmpArrays(source, destination, length);
    }
}


TEST (copyArrWithDiffTypes, IntAndFloats)
{
    int arraySize = 100;
       
    int* sourceArr1; 
    float *sourceFloatArr1;
#if TEST_DOUBLE
    double *sourceDoubleArr1;
#endif
    int* destArr1; 
    float *destFloatArr1;
#if TEST_DOUBLE
    double *destDoubleArr1;
#endif
    sourceArr1 = (int *) malloc (arraySize* sizeof (int));
    destArr1= (int *) malloc (arraySize * sizeof (int));
   
    sourceFloatArr1 = (float*) malloc (arraySize* sizeof (float));
    destFloatArr1 = (float *) malloc (arraySize * sizeof(float));
#if TEST_DOUBLE
    sourceDoubleArr1 = (double *) malloc (arraySize * sizeof(double));
    destDoubleArr1 = (double *) malloc (arraySize * sizeof(double));
#endif
   
    for (int i = 0; i < arraySize; i++){
       sourceArr1[i] = 56535 - i;
    }
   
    for (int i = 0; i < arraySize ; i++){
       sourceFloatArr1[i] = ( float )i  + 0.125f;
    }
#if TEST_DOUBLE
    for (int i = 0; i < arraySize ; i++){
       sourceDoubleArr1[i] = ( double )i  + 0.0009765625;
    }
#endif
    //using bolt::cl::control
   
    bolt::cl::control useThisControl = bolt::cl::control::getDefault();
    
    //copying int array as a whole to all there types of arrays :)
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destArr1);                   //no prob
    //cmpArrays(sourceArr1, destArr1, arraySize);
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destFloatArr1);        //no prob
    //cmpArrays(sourceArr1, destFloatArr1, arraySize);
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destDoubleArr1);        //no prob
#endif
    //cmpArrays(sourceArr1, destDoubleArr1, arraySize);
   
    //copying float array as a whole to all there types of arrays :)
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destArr1);     //data loss
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destFloatArr1);    //no prob
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destDoubleArr1);   //no prob
#endif
   
    //copying double array as a whole to all there types of arrays :)
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destArr1);     //data loss
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destFloatArr1);   //data loss
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destDoubleArr1);  //no prob
#endif
   }


TEST (SerialcopyArrWithDiffTypes, IntAndFloats)
{
    int arraySize = 100;
       
    int* sourceArr1; 
    float *sourceFloatArr1;
#if TEST_DOUBLE
    double *sourceDoubleArr1;
#endif
    int* destArr1; 
    float *destFloatArr1;
#if TEST_DOUBLE
    double *destDoubleArr1;
#endif
    sourceArr1 = (int *) malloc (arraySize* sizeof (int));
    destArr1= (int *) malloc (arraySize * sizeof (int));
   
    sourceFloatArr1 = (float*) malloc (arraySize* sizeof (float));
    destFloatArr1 = (float *) malloc (arraySize * sizeof(float));
#if TEST_DOUBLE
    sourceDoubleArr1 = (double *) malloc (arraySize * sizeof(double));
    destDoubleArr1 = (double *) malloc (arraySize * sizeof(double));
#endif
   
    for (int i = 0; i < arraySize; i++){
       sourceArr1[i] = 56535 - i;
    }
   
    for (int i = 0; i < arraySize ; i++){
       sourceFloatArr1[i] = ( float )i  + 0.125f;
    }
#if TEST_DOUBLE
    for (int i = 0; i < arraySize ; i++){
       sourceDoubleArr1[i] = ( double )i  + 0.0009765625;
    }
#endif
    //using bolt::cl::control

    bolt::cl::control useThisControl = bolt::cl::control::getDefault( );
    useThisControl.setForceRunMode(bolt::cl::control::SerialCpu); 


    //copying int array as a whole to all there types of arrays :)
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destArr1);                   //no prob
    //cmpArrays(sourceArr1, destArr1, arraySize);
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destFloatArr1);        //no prob
    //cmpArrays(sourceArr1, destFloatArr1, arraySize);
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destDoubleArr1);        //no prob
#endif
    //cmpArrays(sourceArr1, destDoubleArr1, arraySize);
   
    //copying float array as a whole to all there types of arrays :)
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destArr1);     //data loss
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destFloatArr1);    //no prob
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destDoubleArr1);   //no prob
#endif
   
    //copying double array as a whole to all there types of arrays :)
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destArr1);     //data loss
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destFloatArr1);   //data loss
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destDoubleArr1);  //no prob
#endif
}


TEST (MultiCorecopyArrWithDiffTypes, IntAndFloats)
{
    int arraySize = 100;
       
    int* sourceArr1; 
    float *sourceFloatArr1;
#if TEST_DOUBLE
    double *sourceDoubleArr1;
#endif
    int* destArr1; 
    float *destFloatArr1;
#if TEST_DOUBLE
    double *destDoubleArr1;
#endif
    sourceArr1 = (int *) malloc (arraySize* sizeof (int));
    destArr1= (int *) malloc (arraySize * sizeof (int));
   
    sourceFloatArr1 = (float*) malloc (arraySize* sizeof (float));
    destFloatArr1 = (float *) malloc (arraySize * sizeof(float));
#if TEST_DOUBLE
    sourceDoubleArr1 = (double *) malloc (arraySize * sizeof(double));
    destDoubleArr1 = (double *) malloc (arraySize * sizeof(double));
#endif
   
    for (int i = 0; i < arraySize; i++){
       sourceArr1[i] = 56535 - i;
    }
   
    for (int i = 0; i < arraySize ; i++){
       sourceFloatArr1[i] = ( float )i  + 0.125f;
    }
#if TEST_DOUBLE
    for (int i = 0; i < arraySize ; i++){
       sourceDoubleArr1[i] = ( double )i  + 0.0009765625;
    }
#endif
    //using bolt::cl::control

    bolt::cl::control useThisControl = bolt::cl::control::getDefault( );
    useThisControl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 


    //copying int array as a whole to all there types of arrays :)
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destArr1);                   //no prob
    //cmpArrays(sourceArr1, destArr1, arraySize);
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destFloatArr1);        //no prob
    //cmpArrays(sourceArr1, destFloatArr1, arraySize);
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceArr1, sourceArr1 + arraySize, destDoubleArr1);        //no prob
#endif
    //cmpArrays(sourceArr1, destDoubleArr1, arraySize);
   
    //copying float array as a whole to all there types of arrays :)
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destArr1);     //data loss
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destFloatArr1);    //no prob
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceFloatArr1, sourceFloatArr1 + arraySize, destDoubleArr1);   //no prob
#endif
   
    //copying double array as a whole to all there types of arrays :)
#if TEST_DOUBLE
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destArr1);     //data loss
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destFloatArr1);   //data loss
    bolt::cl::copy(useThisControl, sourceDoubleArr1, sourceDoubleArr1 + arraySize, destDoubleArr1);  //no prob
#endif
}


TEST (copyIntBoltCLDevVect, withIntAsWhole){ 
    int devVectSize = 10;
    bolt::cl::control& ctrl = bolt::cl::control::getDefault();
    
    bolt::cl::device_vector<int> sourceIntDevVect(10);//, 0, CL_MEM_READ_WRITE, false, ctrl );
    bolt::cl::device_vector<int>   destIntDevVect(10);//, 0, CL_MEM_READ_WRITE, false, ctrl ); 
    
    
    for (int i = 0; i < devVectSize; i++){ 
        sourceIntDevVect[i] = 56535 - i; 
    } 
    
    //bolt::cl::control& ctrl = bolt::cl::control::getDefault(); 
    
    bolt::cl::copy (ctrl, sourceIntDevVect.begin(), sourceIntDevVect.end(), destIntDevVect.begin()); 
    
    for (int i = 0; i < devVectSize; ++i){ 
        EXPECT_EQ(sourceIntDevVect[i], destIntDevVect[i]);
    }
} 

TEST (SerialcopyIntBoltCLDevVect, withIntAsWhole){ 
    int devVectSize = 10;

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::SerialCpu); 
    
    bolt::cl::device_vector<int> sourceIntDevVect(10);//, 0, CL_MEM_READ_WRITE, false, ctrl );
    bolt::cl::device_vector<int>   destIntDevVect(10);//, 0, CL_MEM_READ_WRITE, false, ctrl ); 
    
    
    for (int i = 0; i < devVectSize; i++){ 
        sourceIntDevVect[i] = 56535 - i; 
    } 
    
    //bolt::cl::control& ctrl = bolt::cl::control::getDefault(); 
    
    bolt::cl::copy (ctrl, sourceIntDevVect.begin(), sourceIntDevVect.end(), destIntDevVect.begin()); 
    
    for (int i = 0; i < devVectSize; ++i){ 
        EXPECT_EQ(sourceIntDevVect[i], destIntDevVect[i]);
    }
} 

TEST (MultiCorecopyIntBoltCLDevVect, withIntAsWhole){ 
    int devVectSize = 10;

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    
    bolt::cl::device_vector<int> sourceIntDevVect(10);//, 0, CL_MEM_READ_WRITE, false, ctrl );
    bolt::cl::device_vector<int>   destIntDevVect(10);//, 0, CL_MEM_READ_WRITE, false, ctrl ); 
    
    
    for (int i = 0; i < devVectSize; i++){ 
        sourceIntDevVect[i] = 56535 - i; 
    } 
    
    //bolt::cl::control& ctrl = bolt::cl::control::getDefault(); 
    
    bolt::cl::copy (ctrl, sourceIntDevVect.begin(), sourceIntDevVect.end(), destIntDevVect.begin()); 
    
    for (int i = 0; i < devVectSize; ++i){ 
        EXPECT_EQ(sourceIntDevVect[i], destIntDevVect[i]);
    }
} 

TEST (stdIntToIntCopy, offsetCopy){ 
    int length = 100;
    int splitSize = 25;
    std::vector<int> v(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        v[i*splitSize + count] = count+1;
        }
    
    std::vector<int> stdv(length);
    memcpy(stdv.data(), v.data(), 25 * sizeof(int));      // Copy 25 elements to device vector

    bolt::cl::copy( stdv.begin(), stdv.begin()+25, stdv.begin()+25);//Copy 1st set of 25 eles to 2nd set of 25 eles
    bolt::cl::copy_n(stdv.begin(), 25, stdv.begin() + 50);//Copy 1st set of 25 elements to 3rd set of 25 elmnts
    bolt::cl::copy_n(stdv.begin(), 25, stdv.begin() + 75);//Copy 1st set of 25 elements to 4th set of 25 elemnts
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_EQ(stdv[i], v[i]);
    }
} 

TEST (stdIntToIntCopy, SerialoffsetCopy){ 
    int length = 100;
    int splitSize = 25;
    std::vector<int> v(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        v[i*splitSize + count] = count+1;
        }
    
    std::vector<int> stdv(length);
    memcpy(stdv.data(), v.data(), 25 * sizeof(int)); // Copy 25 elements to device vector

    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::SerialCpu); 

    bolt::cl::copy( ctrl,stdv.begin(),stdv.begin()+25,stdv.begin()+25);//Copy 1st set of 25 eles to 2nd set of 25 eles
    bolt::cl::copy_n(ctrl, stdv.begin(), 25, stdv.begin() + 50);//Copy 1st set of 25 elements to 3rd set of 25 elmnts
    bolt::cl::copy_n(ctrl, stdv.begin(), 25, stdv.begin() + 75);//Copy 1st set of 25 elements to 4th set of 25 elemnts
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_EQ(stdv[i], v[i]);
    }
} 

TEST (stdIntToIntCopy, MultiCoreoffsetCopy){ 
    int length = 100;
    int splitSize = 25;
    std::vector<int> v(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        v[i*splitSize + count] = count+1;
        }
    
    std::vector<int> stdv(length);
    memcpy(stdv.data(), v.data(), 25 * sizeof(int));      // Copy 25 elements to device vector

    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

    bolt::cl::copy( ctrl,stdv.begin(),stdv.begin()+25,stdv.begin()+25);//Copy 1st set of 25 eles to 2nd set of 25 eles
    bolt::cl::copy_n(ctrl, stdv.begin(), 25, stdv.begin() + 50);//Copy 1st set of 25 elements to 3rd set of 25 elmnts
    bolt::cl::copy_n(ctrl, stdv.begin(), 25, stdv.begin() + 75);//Copy 1st set of 25 elements to 4th set of 25 elemnts
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_EQ(stdv[i], v[i]);
    }
} 



TEST (dvIntToIntCopy, offsetCopy){ 
    int length = 100;
    int splitSize = 25;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        dvIn[i*splitSize + count] = count+1;
        }
    
    bolt::cl::device_vector<int> dvOut(length);
    {
        bolt::cl::device_vector<int>::pointer dpOut=dvOut.data();
        bolt::cl::device_vector<int>::pointer dpIn=dvIn.data();
        memcpy(dpOut.get(), dpIn.get(), 25 * sizeof(int)); // Copy 25 elements to device vector
    }
    
    bolt::cl::copy( dvIn.begin(), dvIn.begin()+25, dvOut.begin()+25);//Copy 1st set of 25 eles to 2nd set of 25 eles
    bolt::cl::copy_n(dvIn.begin(), 25, dvOut.begin() + 50);//Copy 1st set of 25 elements to 3rd set of 25 elmnts
    bolt::cl::copy_n(dvIn.begin(), 25, dvOut.begin() + 75);//Copy 1st set of 25 elements to 4th set of 25 elemnts
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_EQ(dvOut[i], dvIn[i]);
    }
} 

TEST (dvIntToIntCopy, SerialoffsetCopy){ 
    int length = 100;
    int splitSize = 25;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        dvIn[i*splitSize + count] = count+1;
        }
    
    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::SerialCpu); 

    bolt::cl::device_vector<int> dvOut(length);
    {
        bolt::cl::device_vector<int>::pointer dpOut=dvOut.data();
        bolt::cl::device_vector<int>::pointer dpIn=dvIn.data();
        memcpy(dpOut.get(), dpIn.get(), 25 * sizeof(int));      // Copy 25 elements to device vector
    }
    
    bolt::cl::copy(ctrl,dvIn.begin(),dvIn.begin()+25,dvOut.begin()+25);//Copy 1st set of 25 eles to 2nd set of 25 eles
    bolt::cl::copy_n(ctrl, dvIn.begin(), 25, dvOut.begin() + 50);//Copy 1st set of 25 elements to 3rd set of 25 elmnts
    bolt::cl::copy_n(ctrl, dvIn.begin(), 25, dvOut.begin() + 75);//Copy 1st set of 25 elements to 4th set of 25 elemnts
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_EQ(dvOut[i], dvIn[i]);
    }
} 

TEST (dvIntToIntCopy, MultiCoreoffsetCopy){ 
    int length = 100;
    int splitSize = 25;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        dvIn[i*splitSize + count] = count+1;
        }
    
    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

    bolt::cl::device_vector<int> dvOut(length);
    {
        bolt::cl::device_vector<int>::pointer dpOut=dvOut.data();
        bolt::cl::device_vector<int>::pointer dpIn=dvIn.data();
        memcpy(dpOut.get(), dpIn.get(), 25 * sizeof(int));      // Copy 25 elements to device vector
    }
    

    bolt::cl::copy(ctrl,dvIn.begin(),dvIn.begin()+25,dvOut.begin()+25);//Copy 1st set of 25 eles to 2nd set of 25 eles
    bolt::cl::copy_n(ctrl, dvIn.begin(), 25, dvOut.begin() + 50);//Copy 1st set of 25 elements to 3rd set of 25 elmnts
    bolt::cl::copy_n(ctrl, dvIn.begin(), 25, dvOut.begin() + 75);//Copy 1st set of 25 elements to 4th set of 25 elemnts
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_EQ(dvOut[i], dvIn[i]);
    }
} 


TEST (stdIntToFloatCopy, offsetCopy){ 
    int length = 100;
    int splitSize = 25;
    std::vector<int> v(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        v[i*splitSize + count] = count+1;
        }
    
    bolt::cl::device_vector<float> dv(length);
    
    //Copy 0- 24 elements
    bolt::cl::copy( v.begin(), v.begin()+25, dv.begin());
    //Copy 25- 49 elements
    bolt::cl::copy( v.begin()+25, v.begin()+50, dv.begin()+25);  // Copy 1st set of 25 elements to 2nd set of 25 elements
    //Copy 50- 74 elements 
    bolt::cl::copy_n(v.begin(), 25, dv.begin() + 50);  // Copy 1st set of 25 elements to 3rd set of 25 elements
    //Copy 75- 99 elements 
    bolt::cl::copy_n(v.begin()+25, 25, dv.begin() + 75);  // Copy 1st set of 25 elements to 4th set of 25 elements
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_FLOAT_EQ(dv[i], (float)v[i]);
    }
} 

TEST (stdIntToFloatCopy, SerialoffsetCopy){ 
    int length = 100;
    int splitSize = 25;
    std::vector<int> v(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        v[i*splitSize + count] = count+1;
        }
    
    bolt::cl::device_vector<float> dv(length);
    
    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::SerialCpu); 

    //Copy 0- 24 elements
    bolt::cl::copy( ctrl, v.begin(), v.begin()+25, dv.begin());
    //Copy 25- 49 elements
    bolt::cl::copy( ctrl, v.begin()+25, v.begin()+50, dv.begin()+25);//Copy 1st set of 25 elmnts to 2nd set of 25 elmnts
    //Copy 50- 74 elements 
    bolt::cl::copy_n(ctrl, v.begin(), 25, dv.begin() + 50);  // Copy 1st set of 25 elements to 3rd set of 25 elements
    //Copy 75- 99 elements 
    bolt::cl::copy_n(ctrl, v.begin()+25, 25, dv.begin() + 75);  // Copy 1st set of 25 elements to 4th set of 25 elements
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_FLOAT_EQ(dv[i], (float)v[i]);
    }
} 

TEST (stdIntToFloatCopy, MultiCoreoffsetCopy){ 
    int length = 100;
    int splitSize = 25;
    std::vector<int> v(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        v[i*splitSize + count] = count+1;
        }
    
    bolt::cl::device_vector<float> dv(length);
    
    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

    //Copy 0- 24 elements
    bolt::cl::copy( ctrl, v.begin(), v.begin()+25, dv.begin());
    //Copy 25- 49 elements
    bolt::cl::copy(ctrl, v.begin()+25, v.begin()+50, dv.begin()+25);//Copy 1st set of 25 elmnts to 2nd set of 25 elmnts
    //Copy 50- 74 elements 
    bolt::cl::copy_n(ctrl, v.begin(), 25, dv.begin() + 50);  // Copy 1st set of 25 elements to 3rd set of 25 elements
    //Copy 75- 99 elements 
    bolt::cl::copy_n(ctrl, v.begin()+25, 25, dv.begin() + 75);  //Copy 1st set of 25 elements to 4th set of 25 elements
    
    for (int i = 0; i < length; ++i){ 
        EXPECT_FLOAT_EQ(dv[i], (float)v[i]);
    }
} 


TEST (dvIntToFloatCopy, offsetCopy){ 
    int length = 100;
    int splitSize = 25;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        dvIn[i*splitSize + count] = count+1;
        }
    
    bolt::cl::device_vector<float> dvOut(length);
    //memcpy will copy the bit representation but std::copy will convert it.
    bolt::cl::copy(dvIn.begin(), dvIn.begin()+25, dvOut.begin());
    for (int i = 0; i < splitSize; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }
    bolt::cl::copy(dvIn.begin()+25,dvIn.begin()+50,dvOut.begin()+25);//Copy 1st set of 25 elmnts to 2nd set of 25 elmts
    for (int i = 25; i < splitSize+25; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }
    bolt::cl::copy_n(dvIn.begin(), 25, dvOut.begin()+50);  // Copy 1st set of 25 elements to 3rd set of 25 elements
    for (int i = 50; i < splitSize+50; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }

    bolt::cl::copy_n(dvIn.begin()+25, 25, dvOut.begin()+75);  // Copy 1st set of 25 elements to 4th set of 25 elements
    for (int i = 75; i < length; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }
} 


TEST (dvIntToFloatCopy, SerialoffsetCopy){ 
    int length = 100;
    int splitSize = 25;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        dvIn[i*splitSize + count] = count+1;
        }
    
            
    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::SerialCpu); 

    bolt::cl::device_vector<float> dvOut(length);
    //memcpy will copy the bit representation but std::copy will convert it.
    bolt::cl::copy(ctrl, dvIn.begin(), dvIn.begin()+25, dvOut.begin());
    for (int i = 0; i < splitSize; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }

    bolt::cl::copy(ctrl,dvIn.begin()+25,dvIn.begin()+50,dvOut.begin()+25);//Copy 1st set of 25 els to 2nd set of 25 ele

    for (int i = 25; i < splitSize+25; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }
    bolt::cl::copy_n(ctrl, dvIn.begin(), 25, dvOut.begin()+50); //Copy 1st set of 25 elements to 3rd set of 25 elements
    for (int i = 50; i < splitSize+50; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }

    bolt::cl::copy_n(ctrl, dvIn.begin()+25, 25, dvOut.begin()+75);//Copy 1st set of 25 elemnts to 4th set of 25 elemnts
    for (int i = 75; i < length; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }
} 

TEST (dvIntToFloatCopy, MultiCoreoffsetCopy){ 
    int length = 100;
    int splitSize = 25;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements to the expected result
    for(int i=0;i<length/splitSize;i++)
        for(int count=0; count < splitSize; ++count) {
                        dvIn[i*splitSize + count] = count+1;
        }
    
            
    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

    bolt::cl::device_vector<float> dvOut(length);
    //memcpy will copy the bit representation but std::copy will convert it.
    bolt::cl::copy(ctrl, dvIn.begin(), dvIn.begin()+25, dvOut.begin());
    for (int i = 0; i < splitSize; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }
    bolt::cl::copy(ctrl,dvIn.begin()+25,dvIn.begin()+50,dvOut.begin()+25);//Copy 1st set of 25 ele to 2nd set of 25 ele
    for (int i = 25; i < splitSize+25; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }
    bolt::cl::copy_n(ctrl, dvIn.begin(), 25, dvOut.begin()+50);//Copy 1st set of 25 elements to 3rd set of 25 elements
    for (int i = 50; i < splitSize+50; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }

    bolt::cl::copy_n(ctrl, dvIn.begin()+25, 25, dvOut.begin()+75);//Copy 1st set of 25 elemnts to 4th set of 25 elemnts
    for (int i = 75; i < length; ++i){ 
        EXPECT_FLOAT_EQ(dvOut[i], (float) dvIn[i]);
    }
} 

//#if (TEST_LARGE_BUFFERS == 1)
TEST (dvIntToFloatlargeBufferCopy, offsetCopy){ 
    int length = 4096;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements to the expected result
    for(int i=0;i<length;i++)
    {
         dvIn[i] = i+1;
    }
    for(int copyStride= 17; copyStride<300; copyStride+=17)
    {
        bolt::cl::device_vector<float> dvOut(length);
        //memcpy will copy the bit representation but std::copy will convert it.
        for(int i=0; i<(length-copyStride*2); i+=copyStride*2)
        {
            int offset = i;
            bolt::cl::copy(dvIn.begin()+offset, dvIn.begin()+offset+copyStride, dvOut.begin()+offset);
            for (int j = 0; j < copyStride; ++j){ 
                {
                EXPECT_FLOAT_EQ(dvOut[offset+j], (float) dvIn[offset+j]);
                }
            }

            offset = i+copyStride;
            bolt::cl::copy(dvIn.begin()+offset, dvIn.begin()+offset+copyStride, dvOut.begin()+offset);
            for (int j = 0; j < copyStride; ++j){ 
                {
                EXPECT_FLOAT_EQ(dvOut[offset+j], (float) dvIn[offset+j]);
                }
            }
        }
    }
} 

TEST (dvIntToFloatlargeBufferCopy, SerialoffsetCopy){ 
    int length = 4096;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements to the expected result
    for(int i=0;i<length;i++)
    {
         dvIn[i] = i+1;
    }

    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::SerialCpu); 

    for(int copyStride= 17; copyStride<300; copyStride+=17)
    {
        bolt::cl::device_vector<float> dvOut(length);
        //memcpy will copy the bit representation but std::copy will convert it.
        for(int i=0; i<(length-copyStride*2); i+=copyStride*2)
        {
            int offset = i;
            bolt::cl::copy(ctrl, dvIn.begin()+offset, dvIn.begin()+offset+copyStride, dvOut.begin()+offset);
            for (int j = 0; j < copyStride; ++j){ 
                {
                EXPECT_FLOAT_EQ(dvOut[offset+j], (float) dvIn[offset+j]);
                }
            }

            offset = i+copyStride;
            bolt::cl::copy(ctrl, dvIn.begin()+offset, dvIn.begin()+offset+copyStride, dvOut.begin()+offset);
            for (int j = 0; j < copyStride; ++j){ 
                {
                EXPECT_FLOAT_EQ(dvOut[offset+j], (float) dvIn[offset+j]);
                }
            }
        }
    }
} 


TEST (dvIntToFloatlargeBufferCopy, MultiCoreoffsetCopy){ 
    int length = 4096;
    bolt::cl::device_vector<int> dvIn(length);
    //Set all the elements to the expected result
    for(int i=0;i<length;i++)
    {
         dvIn[i] = i+1;
    }

    bolt::cl::control ctrl = bolt::cl::control::getDefault( );
    ctrl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 

    for(int copyStride= 17; copyStride<300; copyStride+=17)
    {
        bolt::cl::device_vector<float> dvOut(length);
        //memcpy will copy the bit representation but std::copy will convert it.
        for(int i=0; i<(length-copyStride*2); i+=copyStride*2)
        {
            int offset = i;
            bolt::cl::copy(ctrl, dvIn.begin()+offset, dvIn.begin()+offset+copyStride, dvOut.begin()+offset);
            for (int j = 0; j < copyStride; ++j){ 
                {
				
                EXPECT_FLOAT_EQ(dvOut[offset+j], (float) dvIn[offset+j]);

                }
            }

            offset = i+copyStride;
            bolt::cl::copy(ctrl, dvIn.begin()+offset, dvIn.begin()+offset+copyStride, dvOut.begin()+offset);
            for (int j = 0; j < copyStride; ++j){ 
                {

                EXPECT_FLOAT_EQ(dvOut[offset+j], (float) dvIn[offset+j]);

                }
            }
        }
    }
} 
//#endif

BOLT_FUNCTOR(ichar,
struct ichar
{
  int x;
  char c;
  bool operator==(const ichar& rhs) const
  {
      return ( x == rhs.x );
  }

  void set_values(int ux, char uc)
  {
    x = ux;
    c = uc;
  }

};
);
BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, int, ichar);

TEST (copyStructOdd, BufferTest)
{
  int length = 1024;
  //std::vector<ichar> destination(length), sauce(length);
  bolt::cl::device_vector<ichar> destination(length), sauce(length);
  for (int i = 0; i < length; i++)
  {
     ichar user;
     user.set_values(100,'b');
     sauce[i] = user;
  }

  bolt::cl::copy( sauce.begin(), sauce.end(), destination.begin());
  cmpArrays( sauce, destination, length );

}


TEST (copyStructOdd, BufferTestSameBuffer)
{
  int length = 1024;
  //std::vector<ichar> destination(length), sauce(length);
  bolt::cl::device_vector<ichar> sauce(length);
  for (int i = 0; i < 128; i++)
  {
     ichar user;
     user.set_values(100,'b');
     sauce[i] = user;
  }

  bolt::cl::copy( sauce.begin(), sauce.begin() + 128, sauce.begin() + 128);

  ichar user2;
  user2  = sauce[129];
  std::cout<<"vals are";
  std::cout<<user2.c<<std::endl;
  std::cout<<user2.x<<std::endl;

}



TEST (copyStructOdd, BufferTestHostToDevice)
{
  int length = 1024;
  std::vector<ichar> sauce(length);
  bolt::cl::device_vector<ichar> destination(length);
  for (int i = 0; i < length; i++)
  {
     ichar user;
     user.set_values(100,'b');
     sauce[i] = user;
  }

  bolt::cl::copy( sauce.begin(), sauce.end(), destination.begin());
  cmpArrays( sauce, destination );

}

int main(int argc, char* argv[])
{
    //  Register our minidump generating logic
#if defined(_WIN32)
    bolt::miniDumpSingleton::enableMiniDumps( );
#endif

    // Define MEMORYREPORT on windows platfroms to enable debug memory heap checking
#if defined( MEMORYREPORT ) && defined( _WIN32 )
    TCHAR logPath[ MAX_PATH ];
    ::GetCurrentDirectory( MAX_PATH, logPath );
    ::_tcscat_s( logPath, _T( "\\MemoryReport.txt") );

    // We leak the handle to this file, on purpose, so that the ::_CrtSetReportFile() can output it's memory 
    // statistics on app shutdown
    HANDLE hLogFile;
    hLogFile = ::CreateFile( logPath, GENERIC_WRITE, 
        FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );

    ::_CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
    ::_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
    ::_CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );

    ::_CrtSetReportFile( _CRT_ASSERT, hLogFile );
    ::_CrtSetReportFile( _CRT_ERROR, hLogFile );
    ::_CrtSetReportFile( _CRT_WARN, hLogFile );

    int tmp = ::_CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
    tmp |= _CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF;
    ::_CrtSetDbgFlag( tmp );

    // By looking at the memory leak report that is generated by this debug heap, there is a number with 
    // {} brackets that indicates the incremental allocation number of that block.  If you wish to set
    // a breakpoint on that allocation number, put it in the _CrtSetBreakAlloc() call below, and the heap
    // will issue a bp on the request, allowing you to look at the call stack
    // ::_CrtSetBreakAlloc( 1833 );

#endif /* MEMORYREPORT */

    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    ////  Set the standard OpenCL wait behavior to help debugging
    //bolt::cl::control& myControl = bolt::cl::control::getDefault( );
    //myControl.waitMode( bolt::cl::control::NiceWait );

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

