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
#include "bolt/cl/stablesort.h"
#define cNTiles 64

const std::streamsize colWidth = 26;

template< typename container >
void sortTest( bolt::statTimer& statTimer, size_t length, size_t iter )
{
        size_t sortId   = statTimer.getUniqueID( _T( "sort" ), 0 );

        container input( length );
        container backup( length );

        {
            container::pointer mySP = backup.data( );

            /// Generate buffer of random data
            std::generate( &mySP[ 0 ], &mySP[ length ], rand );

            /// Already sorted data
            //for( int i = 0; i < backup.size( ); ++i )
            //{
            //    mySP[ i ] = i;
            //}

            /// Create even/odd partition; evens got left, odds got right
            //for( int i = 0; i < backup.size( ); ++i )
            //{
            //    if( (i & 1) == 0 )
            //        mySP[ i/2 ] = i;
            //    else
            //        mySP[ (length >> 1) + i/2 ] = i;
            //}
        }

        for( size_t i = 0; i < iter; ++i )
        {
            input = backup;

            //{
            //container::pointer mySP = input.data( );
            //}

            statTimer.Start( sortId );
            bolt::cl::stable_sort( input.begin( ), input.end( ) );
            statTimer.Stop( sortId );

            //{
            //container::pointer mySP = input.data( );
            //}
        }
}

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
    bool validate = false;
    bool compareSerial = false;
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
            ( "validate,v",     "Validate Bolt sort against serial CPU sort" )
            //( "serial",         "Use the STL std::sort" )
            ( "filename,f",     po::value< std::string >( &filename )->default_value( "bench.xml" ),
                "Name of output file" )
            ( "throw-away",     po::value< size_t >( &numThrowAway )->default_value( 0 ),
                "Number of trials to skip averaging" )
            //( "test,t",         po::value< size_t >( &algo )->default_value( 1 ), 
            //    "Algorithm used [1,2,3,4]  1:SORT_BOLT UINT, 2:SORT_BOLT INT, 3:SORT_AMP_SHOC, 4:STD SORT" )
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

        if( vm.count( "validate" ) )
        {
            validate = true;
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
    size_t sortId   = myTimer.getUniqueID( _T( "sort" ), 0 );

    std::vector< unsigned int > input( length );
    std::vector< unsigned int > backup( length );
    if( algo == 1 )
    {
        std::cout<<"Running CL - UINT RADIX SORT\n";

        if( systemMemory )
        {
            sortTest< std::vector< unsigned int > >( myTimer, length, iterations );
        }
        else
        {
            sortTest< bolt::cl::device_vector< unsigned int > >( myTimer, length, iterations );
        }
        sortGB = ( input.size( ) * sizeof( unsigned int ) ) / (1024.0 * 1024.0 * 1024.0);
    }
    else if( algo == 2 )
    {
        std::cout<<"Running CL INT SORT\n";

        if( systemMemory )
        {
            sortTest< std::vector< int > >( myTimer, length, iterations );
        }
        else
        {
            sortTest< bolt::cl::device_vector< int > >( myTimer, length, iterations );
        }
        sortGB = ( input.size( ) * sizeof( int ) ) / (1024.0 * 1024.0 * 1024.0);
    }
    else if( algo == 3 )
    {
//        std::vector< unsigned int > input( length );
//        std::vector< unsigned int > backup( length );
//        std::generate(backup.begin(), backup.end(),rand);
//
//#if (_MSC_VER == 1700)
//        std::cout<<"Running UINT AMP_SORT\n";
//
//        for(int run = 0; run < iterations; ++ run)
//        {
//            // Allocate space for temporary integers and histograms.
//            input = backup;
//            array<unsigned int> dIntegers( static_cast< unsigned int >( length ), input.begin());
//            array<unsigned int> dTmpIntegers( static_cast< unsigned int >( length ) );
//            array<unsigned int> dTmpHistograms(cNTiles * 16);
//            // Execute and time the kernel.
//            myTimer.Start( sortId );
//            Sort(dIntegers, dTmpIntegers, dTmpHistograms);
//            myTimer.Stop( sortId );
//        }
//        sortGB = ( input.size( ) * sizeof( unsigned int ) ) / (1024.0 * 1024.0 * 1024.0);
//#else 
//        std::cout<<"Visual Studio 2010 does not support C++ AMP\n";
//#endif 
    }
    else if( algo == 4 )
    {
        std::vector< unsigned int > input( length );
        std::vector< unsigned int > backup( length );
        std::generate(backup.begin(), backup.end(),rand);
        std::cout<<"Running STD_SORT\n";
        for( unsigned i = 0; i < iterations; ++i )
        {
            input = backup;
            myTimer.Start( sortId );
            std::stable_sort( input.begin( ), input.end( ));
            myTimer.Stop( sortId );
        }
        sortGB = ( input.size( ) * sizeof( unsigned int ) ) / (1024.0 * 1024.0 * 1024.0);
    }
    else
    {
        std::cout <<"Invalid SORT algorithm specified\n";
    }

    //  Remove all timings that are outside of 2 stddev (keep 65% of samples); we ignore outliers to get a more consistent result
    double MKeys = input.size( ) / ( 1024.0 * 1024.0 );
    size_t pruned = myTimer.pruneOutliers( 1.0 );
    double sortTime = myTimer.getAverageTime( sortId );
    //double sortGB = ( input.size( ) * sizeof( int ) ) / (1024.0 * 1024.0 * 1024.0);

    bolt::tout << std::left;
    bolt::tout << std::setw( colWidth ) << _T( "Stable Sort profile: " ) << _T( "[" ) << iterations-pruned << _T( "] samples" ) << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Size (MKeys): " ) << MKeys << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Size (GB): " ) << sortGB << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Time (s): " ) << sortTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (GB/s): " ) << sortGB / sortTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (MKeys/s): " ) << MKeys / sortTime << std::endl;
    bolt::tout << std::endl;

    return 0;
}