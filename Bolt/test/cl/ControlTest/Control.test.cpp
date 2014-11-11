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

#include "bolt/cl/control.h"
#include "bolt/cl/functional.h"
#include "bolt/cl/device_vector.h"
#include "bolt/cl/scan.h"

#include "bolt/unicode.h"
#include "bolt/miniDump.h"
#include "bolt/countof.h"

#include <gtest/gtest.h>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track

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
    for( int i = 0; i < static_cast<int> (ref.size( ) ); ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests

class ReferenceControlTest: public testing::Test 
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReferenceControlTest( ): myControl( bolt::cl::control::getDefault( ) )
    {}

    virtual void SetUp( )
    {
    };

    virtual void TearDown( )
    {
        myControl.freeBuffers( );
    };

protected:
    bolt::cl::control& myControl;
};

class CopyControlTest: public testing::Test
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    CopyControlTest( ): myControl( bolt::cl::control::getDefault( ) )
    {}

    virtual void SetUp( )
    {
    };

    virtual void TearDown( )
    {
    };

protected:
    bolt::cl::control myControl;
};

TEST_F( ReferenceControlTest, init )
{
    size_t internalBuffSize = myControl.totalBufferSize( );

    EXPECT_EQ( 0, internalBuffSize );
}

TEST_F( ReferenceControlTest, zeroMemory )
{
    myControl.acquireBuffer( 100 * sizeof( int ) );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 400, internalBuffSize );

    myControl.freeBuffers( );
    internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 0, internalBuffSize );
}

TEST_F( ReferenceControlTest, acquire1Buffer )
{
    bolt::cl::control::buffPointer myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 400, internalBuffSize );
}

TEST_F( ReferenceControlTest, acquire1BufferReleaseAcquireSame )
{
    bolt::cl::control::buffPointer myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );
    myBuff.reset( );

    myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 400, internalBuffSize );
}

TEST_F( ReferenceControlTest, acquire1BufferReleaseAcquireSmaller )
{
    bolt::cl::control::buffPointer myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );
    myBuff.reset( );

    myBuff = myControl.acquireBuffer( 99 * sizeof( int ) );
    myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 400, internalBuffSize );
}

TEST_F( ReferenceControlTest, acquire1BufferReleaseAcquireBigger )
{
    bolt::cl::control::buffPointer myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );
    myBuff.reset( );

    myBuff = myControl.acquireBuffer( 101 * sizeof( int ) );
    myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 404, internalBuffSize );
}

TEST_F( ReferenceControlTest, acquire2BufferEqual )
{
    bolt::cl::control::buffPointer myBuff1 = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount1 = myBuff1->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount1 );

    bolt::cl::control::buffPointer myBuff2 = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount2 = myBuff2->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount2 );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 800, internalBuffSize );
}

TEST_F( ReferenceControlTest, acquire2BufferBigger )
{
    bolt::cl::control::buffPointer myBuff1 = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount1 = myBuff1->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount1 );

    bolt::cl::control::buffPointer myBuff2 = myControl.acquireBuffer( 101 * sizeof( int ) );
    cl_uint myRefCount2 = myBuff2->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount2 );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 804, internalBuffSize );
}

TEST_F( ReferenceControlTest, acquire2BufferSmaller )
{
    bolt::cl::control::buffPointer myBuff1 = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount1 = myBuff1->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount1 );

    bolt::cl::control::buffPointer myBuff2 = myControl.acquireBuffer( 99 * sizeof( int ) );
    cl_uint myRefCount2 = myBuff2->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount2 );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 796, internalBuffSize );
}

TEST_F( CopyControlTest, init )
{
    size_t internalBuffSize = myControl.totalBufferSize( );

    EXPECT_EQ( 0, internalBuffSize );
}

TEST_F( CopyControlTest, zeroMemory )
{
    myControl.acquireBuffer( 100 * sizeof( int ) );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 400, internalBuffSize );

    myControl.freeBuffers( );
    internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 0, internalBuffSize );
}

TEST_F( CopyControlTest, acquire1Buffer )
{
    bolt::cl::control::buffPointer myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 400, internalBuffSize );
}

TEST_F( CopyControlTest, acquire1BufferReleaseAcquireSame )
{
    bolt::cl::control::buffPointer myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );
    myBuff.reset( );

    myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 400, internalBuffSize );
}

TEST_F( CopyControlTest, acquire1BufferReleaseAcquireSmaller )
{
    bolt::cl::control::buffPointer myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );
    myBuff.reset( );

    myBuff = myControl.acquireBuffer( 99 * sizeof( int ) );
    myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 400, internalBuffSize );
}

TEST_F( CopyControlTest, acquire1BufferReleaseAcquireBigger )
{
    bolt::cl::control::buffPointer myBuff = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );
    myBuff.reset( );

    myBuff = myControl.acquireBuffer( 101 * sizeof( int ) );
    myRefCount = myBuff->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 404, internalBuffSize );
}

TEST_F( CopyControlTest, acquire2BufferEqual )
{
    bolt::cl::control::buffPointer myBuff1 = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount1 = myBuff1->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount1 );

    bolt::cl::control::buffPointer myBuff2 = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount2 = myBuff2->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount2 );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 800, internalBuffSize );
}

TEST_F( CopyControlTest, acquire2BufferBigger )
{
    bolt::cl::control::buffPointer myBuff1 = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount1 = myBuff1->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount1 );

    bolt::cl::control::buffPointer myBuff2 = myControl.acquireBuffer( 101 * sizeof( int ) );
    cl_uint myRefCount2 = myBuff2->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount2 );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 804, internalBuffSize );
}

TEST_F( CopyControlTest, acquire2BufferSmaller )
{
    bolt::cl::control::buffPointer myBuff1 = myControl.acquireBuffer( 100 * sizeof( int ) );
    cl_uint myRefCount1 = myBuff1->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount1 );

    bolt::cl::control::buffPointer myBuff2 = myControl.acquireBuffer( 99 * sizeof( int ) );
    cl_uint myRefCount2 = myBuff2->getInfo< CL_MEM_REFERENCE_COUNT >( );
    EXPECT_EQ( 1, myRefCount2 );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 796, internalBuffSize );
}

TEST_F( CopyControlTest, ScanIntegerVector )
{
    bolt::cl::device_vector< int > boltInput1( 1024, 1 );
    bolt::cl::device_vector< int > boltInput2( 1024, 1 );
    std::vector< int > stdInput( 1024, 1 );

    std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );

    bolt::cl::inclusive_scan( myControl, boltInput1.begin( ), boltInput1.end( ), boltInput1.begin( ) );
    cmpArrays( stdInput, boltInput1 );

    bolt::cl::inclusive_scan( myControl, boltInput2.begin( ), boltInput2.end( ), boltInput2.begin( ) );
    cmpArrays( stdInput, boltInput2 );

    size_t internalBuffSize = myControl.totalBufferSize( );
    EXPECT_EQ( 2049, internalBuffSize );
}

int _tmain(int argc, _TCHAR* argv[])
{
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Register our minidump generating logic
    //bolt::miniDumpSingleton::enableMiniDumps( );

    /******************************************************************************
     * Default Benchmark Parameters
     *****************************************************************************/
    cl_uint userPlatform = 0;
    cl_uint userDevice = 0;
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    bool defaultDevice = true;
    bool print_clInfo = false;
    bool hostMemory = false;
    bolt::cl::control& ctrl = bolt::cl::control::getDefault();

    /******************************************************************************
     * Parameter Parsing
     ******************************************************************************/
    try
    {
        // Declare the supported options.
        po::options_description desc( "OpenCL sort command line options" );
        desc.add_options()
            ( "help,h",         "produces this help message" )
            ( "version",        "Print queryable version information from the Bolt CL library" )
            ( "queryOpenCL,q",  "Print queryable platform and device info and return" )
            ( "gpu,g",          "Report only OpenCL GPU devices" )
            ( "cpu,c",          "Report only OpenCL CPU devices" )
            ( "all,a",          "Report all OpenCL devices" )
            ( "hostMemory,m",   "Allocate vectors in host memory, otherwise device memory" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( 0 ),
                "Specify the platform under test using the index reported by -q flag" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( 0 ),
                "Specify the device under test using the index reported by the -q flag.  "
                    "Index is relative with respect to -g, -c or -a flags" )
            ;

        po::variables_map vm;
        po::store( po::parse_command_line( argc, argv, desc ), vm );
        po::notify( vm );

        if( vm.count( "version" ) )
        {
            cl_uint libMajor, libMinor, libPatch;
            bolt::cl::getVersion( libMajor, libMinor, libPatch );

            const int indent = countOf( "Bolt version: " );
            bolt::tout << std::left << std::setw( indent ) << _T( "Bolt version: " )
                << libMajor << _T( "." )
                << libMinor << _T( "." )
                << libPatch << std::endl;
        }

        if( vm.count( "help" ) )
        {
            //  This needs to be 'cout' as program-options does not support wcout yet
            std::cout << desc << std::endl;
            return 0;
        }

        if( vm.count( "queryOpenCL" ) )
        {
            print_clInfo = true;
        }

        if( vm.count( "gpu" ) )
        {
            deviceType  = CL_DEVICE_TYPE_GPU;
        }
        
        if( vm.count( "cpu" ) )
        {
            deviceType  = CL_DEVICE_TYPE_CPU;
        }

        if( vm.count( "all" ) )
        {
            deviceType  = CL_DEVICE_TYPE_ALL;
        }

        if( vm.count( "hostMemory" ) )
        {
            hostMemory = true;
        }
    }
    catch( std::exception& e )
    {
        std::cout << _T( "Sort Benchmark error condition reported:" ) << std::endl << e.what() << std::endl;
        return 1;
    }

    /******************************************************************************
    * Initialize platforms and devices                                            *
    * /todo we should move this logic inside of the control class                 *
    ******************************************************************************/
    //  Query OpenCL for available platforms
    cl_int err = CL_SUCCESS;

    // Platform vector contains all available platforms on system
    std::vector< ::cl::Platform > platforms;
    bolt::cl::V_OPENCL( ::cl::Platform::get( &platforms ), "Platform::get() failed" );

    // Device info
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();

    ::cl::CommandQueue myQueue( myContext, devices.at( userDevice ) , CL_QUEUE_PROFILING_ENABLE);
    ctrl.setCommandQueue( myQueue );
    std::string strDeviceName = ctrl.getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
    bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    std::cout << "Device under test : " << strDeviceName << std::endl;

    if( print_clInfo )
    {
        bolt::cl::control::printPlatforms( true, deviceType );
        return 0;
    }

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
