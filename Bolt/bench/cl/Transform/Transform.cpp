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

#include "bolt/unicode.h"
#include "bolt/statisticalTimer.h"
#include "bolt/countof.h"
#include "bolt/cl/transform.h"
#include "bolt/cl/functional.h"

const std::streamsize colWidth = 26;

BOLT_FUNCTOR( SaxpyFunctor,
struct SaxpyFunctor
{
    float m_A;
    SaxpyFunctor( float a ) : m_A( a )
    {};

    float operator( )( const float& xx, const float& yy )
    {
        return m_A * xx + yy;
    };
};
);  // end BOLT_FUNCTOR

int main( int argc, char* argv[] )
{
    /******************************************************************************
     * Default Benchmark Parameters
     *****************************************************************************/
    cl_uint userPlatform = 0;
    cl_uint userDevice = 0;
    size_t iterations = 0;
    size_t length = 0;
    size_t algo = 1;
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    bool defaultDevice = true;
    bool print_clInfo = false;
    bool systemMemory = false;
    bool deviceMemory = false;
    bool runTBB = false;
    bool runBOLT = false;
    bool runSTL = false;

    std::string filename;
    size_t numThrowAway = 10;
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
            ( "systemMemory,S", "Allocate vectors in system memory, otherwise device memory" )
            ( "deviceMemory,D", "Allocate vectors in system memory, otherwise device memory" )
            ( "tbb,T",          "Benchmark TBB MULTICORE CPU Code" )
            ( "bolt,B",         "Benchmark Bolt OpenCL Libray" )
            ( "serial,E",       "Benchmark Serial Code STL Libray" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( 0 ),
                "Specify the platform under test using the index reported by -q flag" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( 0 ),
                "Specify the device under test using the index reported by the -q flag.  "
                    "Index is relative with respect to -g, -c or -a flags" )
            ( "length,l",       po::value< size_t >( &length )->default_value( 4096 ),
                "Specify the length of sort array" )
            ( "iterations,i",   po::value< size_t >( &iterations )->default_value( 1 ),
                "Number of samples in timing loop" )
            ( "filename,f",     po::value< std::string >( &filename )->default_value( "bench.xml" ),
                "Name of output file" )
            ( "throw-away",     po::value< size_t >( &numThrowAway )->default_value( 0 ),
                "Number of trials to skip averaging" )
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

        if( vm.count( "systemMemory" ) )
        {
            systemMemory = true;
        }
        if( vm.count( "deviceMemory" ) )
        {
            deviceMemory = true;
        }
        if( vm.count( "tbb" ) )
        {
            runTBB = true;
        }
        if( vm.count( "bolt" ) )
        {
            runBOLT = true;
        }
        if( vm.count( "serial" ) ) 
        {
            runSTL = true;
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

    if( runSTL )
    {
        ctrl.setForceRunMode( bolt::cl::control::SerialCpu );  // choose serial std::scan
    }

    if( runTBB )
    {
        ctrl.setForceRunMode( bolt::cl::control::MultiCoreCpu );  // choose tbb tbb::parallel_scan
    }

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

    bolt::statTimer& myTimer = bolt::statTimer::getInstance( );
    double sortGB = 0;
    myTimer.Reserve( 1, iterations );
    size_t timerId   = myTimer.getUniqueID( _T( "transform" ), 0 );

    SaxpyFunctor s(100.0);

    if( systemMemory )
    {
        std::vector< int > input1( length, 1 );
        std::vector< int > input2( length, 1 );
        std::vector< int > output( length );

        for( unsigned i = 0; i < iterations; ++i )
        {
            myTimer.Start( timerId );
            bolt::cl::transform( input1.begin(), input1.end(), input2.begin(), output.begin(), s );
            myTimer.Stop( timerId );
        }
    }
    else
    {
        bolt::cl::device_vector< int > input1( length, 1 );
        bolt::cl::device_vector< int > input2( length, 1 );
        bolt::cl::device_vector< int > output( length );

        for( unsigned i = 0; i < iterations; ++i )
        {
            myTimer.Start( timerId );
            bolt::cl::transform( input1.begin(), input1.end(), input2.begin(), output.begin(), s );
            myTimer.Stop( timerId );
        }
    }

    //  Remove all timings that are outside of 2 stddev (keep 65% of samples); we ignore outliers to get a more consistent result
    size_t pruned = myTimer.pruneOutliers( 1.0 );
    double testTime = myTimer.getAverageTime( timerId );
    double testMB = ( length * sizeof( int ) ) / ( 1024.0 * 1024.0);
    double testGB = testMB/ 1024.0;
    double MKeys = length / ( 1024.0 * 1024.0 );

    bolt::tout << std::left;
    bolt::tout << std::setw( colWidth ) << _T( "Transform profile: " ) << _T( "[" ) << iterations-pruned << _T( "] samples" ) << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Size (MKeys): " ) << MKeys << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Size (MB): " ) << testMB << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Time (ms): " ) << testTime*1000.0 << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (GB/s): " ) << testGB / testTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (MKeys/s): " ) << MKeys / testTime << std::endl;
    bolt::tout << std::endl;

//  bolt::tout << myTimer;

    return 0;
}