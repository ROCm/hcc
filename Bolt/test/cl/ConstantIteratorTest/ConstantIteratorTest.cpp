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
#include <vector>
#include <array>

#include "bolt/cl/iterator/constant_iterator.h"
#include "bolt/cl/iterator/counting_iterator.h"
#include "bolt/cl/iterator/transform_iterator.h"
#include "bolt/cl/transform.h"
#include "bolt/cl/reduce.h"
#include "bolt/cl/functional.h"
#include "bolt/miniDump.h"
#include "bolt/unicode.h"

#include <gtest/gtest.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track

template< typename T >
::testing::AssertionResult cmpArrays( const T ref, const T calc, size_t N )
{
    for( size_t i = 0; i < N; ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

template< typename T, size_t N >
::testing::AssertionResult cmpArrays( const T (&ref)[N], const T (&calc)[N] )
{
    for( size_t i = 0; i < N; ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

//  Primary class template for std::array types
//  The struct wrapper is necessary to partially specialize the member function
template< typename T, size_t N >
struct cmpStdArray
{
    static ::testing::AssertionResult cmpArrays( const std::array< T, N >& ref, const std::array< T, N >& calc )
    {
        for( size_t i = 0; i < N; ++i )
        {
            EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );
    }
};

//  Partial template specialization for float types
//  Partial template specializations only works for objects, not functions
template< size_t N >
struct cmpStdArray< float, N >
{
    static ::testing::AssertionResult cmpArrays( const std::array< float, N >& ref, const std::array< float, N >& calc )
    {
        for( size_t i = 0; i < N; ++i )
        {
            EXPECT_FLOAT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );
    }
};

//  Partial template specialization for float types
//  Partial template specializations only works for objects, not functions
template< size_t N >
struct cmpStdArray< double, N >
{
    static ::testing::AssertionResult cmpArrays( const std::array< double, N >& ref, const std::array< double, N >& calc )
    {
        for( size_t i = 0; i < N; ++i )
        {
            EXPECT_DOUBLE_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );
    }
};

//  The following cmpArrays verify the correctness of std::vectors's
template< typename T >
::testing::AssertionResult cmpArrays( const std::vector< T >& ref, const std::vector< T >& calc )
{
    for( size_t i = 0; i < ref.size( ); ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

::testing::AssertionResult cmpArrays( const std::vector< float >& ref, const std::vector< float >& calc )
{
    for( size_t i = 0; i < ref.size( ); ++i )
    {
        EXPECT_FLOAT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

::testing::AssertionResult cmpArrays( const std::vector< double >& ref, const std::vector< double >& calc )
{
    for( size_t i = 0; i < ref.size( ); ++i )
    {
        EXPECT_DOUBLE_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

//  A very generic template that takes two container, and compares their values assuming a vector interface
template< typename S, typename B >
::testing::AssertionResult cmpArrays( const S& ref, const B& calc )
{
    for( size_t i = 0; i < ref.size( ); ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests

//  Can't get tests to stop warning on floating point
// typedef ::testing::Types< int, float, double > StdTypes;
typedef ::testing::Types< int > StdTypes;

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
template< typename Type >
class ConstantIterator: public ::testing::Test
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ConstantIterator( )
    {}

protected:
};
TYPED_TEST_CASE_P( ConstantIterator );

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
template< typename Type >
class CountingIterator: public ::testing::Test
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    CountingIterator( )
    {}

protected:
};
TYPED_TEST_CASE_P( CountingIterator );

//BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::constant_iterator, int, char );
//
//TEST( ConstantIterator, PrintClCode )
//{
//    std::cout << TypeName< bolt::cl::constant_iterator< int > >::get( ) << std::endl;
//    std::cout << ClCode< bolt::cl::constant_iterator< int > >::get( ) << std::endl << std::endl;
//
//    std::cout << TypeName< bolt::cl::constant_iterator< float > >::get( ) << std::endl;
//    std::cout << ClCode< bolt::cl::constant_iterator< float > >::get( ) << std::endl << std::endl;
//
//    std::cout << TypeName< bolt::cl::constant_iterator< double > >::get( ) << std::endl;
//    std::cout << ClCode< bolt::cl::constant_iterator< double > >::get( ) << std::endl << std::endl;
//
//    std::cout << TypeName< bolt::cl::constant_iterator< char > >::get( ) << std::endl;
//    std::cout << ClCode< bolt::cl::constant_iterator< char > >::get( ) << std::endl << std::endl;
//}

TYPED_TEST_P( ConstantIterator, HostSideNaive )
{
    bolt::cl::constant_iterator< TypeParam > cIter( 3 );

    EXPECT_EQ( 3, *cIter );
    EXPECT_EQ( 3, cIter[ 0 ] );
    EXPECT_EQ( 3, cIter[ 1 ] );
    EXPECT_EQ( 3, cIter[ 100 ] );
}

TYPED_TEST_P( ConstantIterator, StdTransformVector )
{
    std::vector< TypeParam > stdVec( 4 );
    stdVec[ 0 ] = 1;
    stdVec[ 1 ] = 10;
    stdVec[ 2 ] = 100;
    stdVec[ 3 ] = 1000;

    // Transform the array, adding 42 to the entire vector
    bolt::cl::transform( stdVec.begin( ), stdVec.end( ), bolt::cl::make_constant_iterator( 42 ),
                      stdVec.begin( ), bolt::cl::plus< TypeParam >( ) );

    EXPECT_EQ( 43, stdVec[ 0 ] );
    EXPECT_EQ( 52, stdVec[ 1 ] );
    EXPECT_EQ( 142, stdVec[ 2 ] );
    EXPECT_EQ( 1042, stdVec[ 3 ] );
}

TYPED_TEST_P( ConstantIterator, DeviceTransformVector )
{
    bolt::cl::device_vector< TypeParam > devVec( 4 );
    devVec[ 0 ] = 1;
    devVec[ 1 ] = 10;
    devVec[ 2 ] = 100;
    devVec[ 3 ] = 1000;

    // Transform the array, adding 42 to the entire vector
    bolt::cl::transform( devVec.begin( ), devVec.end( ), bolt::cl::make_constant_iterator( 42 ),
                      devVec.begin( ), bolt::cl::plus< TypeParam >( ) );

    EXPECT_EQ( 43, devVec[ 0 ] );
    EXPECT_EQ( 52, devVec[ 1 ] );
    EXPECT_EQ( 142, devVec[ 2 ] );
    EXPECT_EQ( 1042, devVec[ 3 ] );
}

REGISTER_TYPED_TEST_CASE_P( ConstantIterator, HostSideNaive, StdTransformVector, DeviceTransformVector );

INSTANTIATE_TYPED_TEST_CASE_P( TypedTests, ConstantIterator, StdTypes );

TYPED_TEST_P( CountingIterator, HostSideNaive )
{
    bolt::cl::counting_iterator< TypeParam > iter( 42 );

    EXPECT_EQ( 42, iter[ 0 ] );
    EXPECT_EQ( 45, iter[ 3 ] );
    EXPECT_EQ( 142, iter[ 100 ] );
}

TYPED_TEST_P( CountingIterator, StdTransformVector )
{
    // initialize the data vector to be sequential numbers
    std::vector< TypeParam > stdVec( 3 );
    bolt::cl::transform( stdVec.begin( ), stdVec.end( ), bolt::cl::make_counting_iterator( 42 ), stdVec.begin( ),
        bolt::cl::plus< TypeParam >( ) );

    EXPECT_EQ( 42, stdVec[ 0 ] );
    EXPECT_EQ( 43, stdVec[ 1 ] );
    EXPECT_EQ( 44, stdVec[ 2 ] );
}

TYPED_TEST_P( CountingIterator, DeviceTransformVector )
{
    // initialize the data vector to be sequential numbers
    std::vector< TypeParam > devVec( 3 );
    bolt::cl::transform( devVec.begin( ), devVec.end( ), bolt::cl::make_counting_iterator( 42 ), devVec.begin( ),
        bolt::cl::plus< TypeParam >( ) );

    EXPECT_EQ( 42, devVec[ 0 ] );
    EXPECT_EQ( 43, devVec[ 1 ] );
    EXPECT_EQ( 44, devVec[ 2 ] );
}

BOLT_FUNCTOR(square,
    struct square
    {
        int operator() (const int& x)  const { return x + 2; }
        typedef int result_type;
    };
);


std::string temp_str = BOLT_CODE_STRING(typedef bolt::cl::transform_iterator< square, bolt::cl::device_vector< int >::iterator > trf_sq_itr;);
BOLT_CREATE_TYPENAME( trf_sq_itr );

//BOLT_CREATE_TYPENAME( (bolt::cl::transform_iterator< square< int >, bolt::cl::device_vector< int >::iterator >)  );
BOLT_CREATE_CLCODE  ( trf_sq_itr, ClCode<square>::get() + bolt::cl::deviceTransformIteratorTemplate + temp_str);

//#define BOLT_CREATE_TYPENAME( Type ) \
//    template<> struct TypeName< Type > { static std::string get( ) { return #Type; } };


REGISTER_TYPED_TEST_CASE_P( CountingIterator, HostSideNaive, StdTransformVector, DeviceTransformVector );

INSTANTIATE_TYPED_TEST_CASE_P( TypedTests, CountingIterator, StdTypes );

/* /brief List of possible tests
 * Two input transform with first input a constant iterator
 * One input transform with a constant iterator
*/
int _tmain(int argc, _TCHAR* argv[])
{
    //  Register our minidump generating logic
    //bolt::miniDumpSingleton::enableMiniDumps( );

    //  Initialize googletest; this removes googletest specific flags from command line
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    bool print_clInfo = false;
    cl_uint userPlatform = 0;
    cl_uint userDevice = 0;
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;

    try
    {
        // Declare supported options below, describe what they do
        po::options_description desc( "Scan GoogleTest command line options" );
        desc.add_options()
            ( "help,h",         "produces this help message" )
            ( "queryOpenCL,q",  "Print queryable platform and device info and return" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( 0 ),	"Specify the platform under test" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( 0 ),	"Specify the device under test" )
            //( "gpu,g",         "Force instantiation of all OpenCL GPU device" )
            //( "cpu,c",         "Force instantiation of all OpenCL CPU device" )
            //( "all,a",         "Force instantiation of all OpenCL devices" )
            ;

        ////  All positional options (un-named) should be interpreted as kernelFiles
        //po::positional_options_description p;
        //p.add("kernelFiles", -1);

        //po::variables_map vm;
        //po::store( po::command_line_parser( argc, argv ).options( desc ).positional( p ).run( ), vm );
        //po::notify( vm );

        po::variables_map vm;
        po::store( po::parse_command_line( argc, argv, desc ), vm );
        po::notify( vm );

        if( vm.count( "help" ) )
        {
            //	This needs to be 'cout' as program-options does not support wcout yet
            std::cout << desc << std::endl;
            return 0;
        }

        if( vm.count( "queryOpenCL" ) )
        {
            print_clInfo = true;
        }

        //  The following 3 options are not implemented yet; they are meant to be used with ::clCreateContextFromType()
        if( vm.count( "gpu" ) )
        {
            deviceType	= CL_DEVICE_TYPE_GPU;
        }

        if( vm.count( "cpu" ) )
        {
            deviceType	= CL_DEVICE_TYPE_CPU;
        }

        if( vm.count( "all" ) )
        {
            deviceType	= CL_DEVICE_TYPE_ALL;
        }

    }
    catch( std::exception& e )
    {
        std::cout << _T( "Scan GoogleTest error condition reported:" ) << std::endl << e.what() << std::endl;
        return 1;
    }

    //  Query OpenCL for available platforms
    cl_int err = CL_SUCCESS;

    // Platform vector contains all available platforms on system
    std::vector< cl::Platform > platforms;
    //std::cout << "HelloCL!\nGetting Platform Information\n";
    bolt::cl::V_OPENCL( cl::Platform::get( &platforms ), "Platform::get() failed" );

    if( print_clInfo )
    {
        bolt::cl::control::printPlatforms( );
        return 0;
    }

    //  Do stuff with the platforms
    std::vector<cl::Platform>::iterator i;
    if(platforms.size() > 0)
    {
        for(i = platforms.begin(); i != platforms.end(); ++i)
        {
            if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
    }
    bolt::cl::V_OPENCL( err, "Platform::getInfo() failed" );

    // Device info
    std::vector< cl::Device > devices;
    bolt::cl::V_OPENCL( platforms.front( ).getDevices( CL_DEVICE_TYPE_ALL, &devices ), "Platform::getDevices() failed" );

    cl::Context myContext( devices.at( userDevice ) );
    cl::CommandQueue myQueue( myContext, devices.at( userDevice ) );
    bolt::cl::control::getDefault( ).setCommandQueue( myQueue );

    std::string strDeviceName = bolt::cl::control::getDefault( ).getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
    bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    std::cout << "Device under test : " << strDeviceName << std::endl;

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
