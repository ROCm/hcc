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
#include "bolt/cl/scan.h"

const std::streamsize colWidth = 26;

int _tmain( int argc, _TCHAR* argv[] )
{
    cl_uint userPlatform = 0;
    cl_uint userDevice = 0;
    size_t iterations = 0;
    size_t length = 0;
    size_t algo = 1;
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    bool defaultDevice = true;
    bool print_clInfo = false;
    bool systemMemory = false;

    /******************************************************************************
    * Parameter parsing                                                           *
    ******************************************************************************/
    try
    {
        // Declare the supported options.
        po::options_description desc( "OpenCL CopyBuffer command line options" );
        desc.add_options()
            ( "help,h",			"produces this help message" )
            ( "version,v",		"Print queryable version information from the Bolt CL library" )
            ( "queryOpenCL,q",  "Print queryable platform and device info and return" )
            ( "gpu,g",          "Report only OpenCL GPU devices" )
            ( "cpu,c",          "Report only OpenCL CPU devices" )
            ( "all,a",          "Report all OpenCL devices" )
            ( "systemMemory,s", "Allocate vectors in system memory, otherwise device memory" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( 0 ), "Specify the platform under test using the index reported by -q flag" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( 0 ), "Specify the device under test using the index reported by the -q flag.  "
                    "Index is relative with respect to -g, -c or -a flags" )
            ( "length,l",       po::value< size_t >( &length )->default_value( 1048576 ), "Specify the length of scan array" )
            ( "iterations,i",   po::value< size_t >( &iterations )->default_value( 50 ), "Number of samples in timing loop" )
			//( "algo,a",		    po::value< size_t >( &algo )->default_value( 1 ), "Algorithm used [1,2]  1:SCAN_BOLT, 2:XYZ" )//Not used in this file
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
            //	This needs to be 'cout' as program-options does not support wcout yet
            std::cout << desc << std::endl;
            return 0;
        }

        if( vm.count( "queryOpenCL" ) )
        {
            print_clInfo = true;
        }

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

        if( vm.count( "systemMemory" ) )
        {
            systemMemory = true;
        }

    }
    catch( std::exception& e )
    {
        std::cout << _T( "Scan Benchmark error condition reported:" ) << std::endl << e.what() << std::endl;
        return 1;
    }

    /******************************************************************************
    * Initialize platforms and devices                                            *
    * /todo we should move this logic inside of the control class                 *
    ******************************************************************************/
    //  Query OpenCL for available platforms
    cl_int err = CL_SUCCESS;

    // Platform vector contains all available platforms on system
    std::vector< cl::Platform > platforms;
    bolt::cl::V_OPENCL( cl::Platform::get( &platforms ), "Platform::get() failed" );

    if( print_clInfo )
    {
        //  /todo: port the printing code from test/scan to control class
        //std::for_each( platforms.begin( ), platforms.end( ), printPlatformFunctor( 0 ) );
        return 0;
    }

    // Device info
    std::vector< cl::Device > devices;
    bolt::cl::V_OPENCL( platforms.at( userPlatform ).getDevices( deviceType, &devices ), "Platform::getDevices() failed" );

    cl::Context myContext( devices.at( userDevice ) );
    cl::CommandQueue myQueue( myContext, devices.at( userDevice ) );

    //  Now that the device we want is selected and we have created our own cl::CommandQueue, set it as the
    //  default cl::CommandQueue for the Bolt API
    bolt::cl::control::getDefault( ).setCommandQueue( myQueue );

    std::string strDeviceName = bolt::cl::control::getDefault( ).getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
    bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    std::cout << "Device under test : " << strDeviceName << std::endl;

    /******************************************************************************
    * Benchmark logic                                                             *
    ******************************************************************************/
    bolt::statTimer& myTimer = bolt::statTimer::getInstance( );
    myTimer.Reserve( 1, iterations );
    size_t scanId	= myTimer.getUniqueID( _T( "copybuffer" ), 0 );

    size_t pruned = 0;
    double scanTime = std::numeric_limits< double >::max( );
    double scanGB = ( length * sizeof( int ) ) / (1024.0 * 1024.0 * 1024.0);
    ::cl::CommandQueue& boltQueue = bolt::cl::control::getDefault( ).getCommandQueue( );

    //  ::cl::Buffer can not handle buffers of size 0
    if( length > 0 )
    {
        if( systemMemory )
        {
            std::vector< int > input( length, 1 );
            std::vector< int > output( length );
            ::cl::Buffer inputBuffer( bolt::cl::control::getDefault( ).getContext( ), CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, length * sizeof( int ), input.data( ) );
            ::cl::Buffer outputBuffer( bolt::cl::control::getDefault( ).getContext( ), CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY, length * sizeof( int ), output.data( ) );

            for( unsigned i = 0; i < iterations; ++i )
            {
                myTimer.Start( scanId );
                boltQueue.enqueueCopyBuffer( inputBuffer, outputBuffer, 0, 0, length * sizeof( int ) );
                void* tmpPtr = boltQueue.enqueueMapBuffer( outputBuffer, true, CL_MAP_READ, 0, length * sizeof( int ) );
                boltQueue.enqueueUnmapMemObject( outputBuffer, tmpPtr );
                boltQueue.finish( );
                myTimer.Stop( scanId );
            }
        }
        else
        {
            ::cl::Buffer inputBuffer( bolt::cl::control::getDefault( ).getContext( ), CL_MEM_READ_ONLY, length * sizeof( int ) );
            ::cl::Buffer outputBuffer( bolt::cl::control::getDefault( ).getContext( ), CL_MEM_WRITE_ONLY, length * sizeof( int ) );

            for( unsigned i = 0; i < iterations; ++i )
            {
                myTimer.Start( scanId );
                boltQueue.enqueueCopyBuffer( inputBuffer, outputBuffer, 0, 0, length * sizeof( int ) );
                boltQueue.finish( );
                myTimer.Stop( scanId );
            }
        }

        //	Remove all timings that are outside of 2 stddev (keep 65% of samples); we ignore outliers to get a more consistent result
        pruned = myTimer.pruneOutliers( 1.0 );
        scanTime = myTimer.getAverageTime( scanId );
    }
    else
    {
        iterations = 0;
    }

    bolt::tout << std::left;
    bolt::tout << std::setw( colWidth ) << _T( "CopyBuffer profile: " ) << _T( "[" ) << iterations-pruned << _T( "] samples" ) << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Size (GB): " ) << scanGB << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Time (s): " ) << scanTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (GB/s): " ) << scanGB / scanTime << std::endl;
    bolt::tout << std::endl;

//	bolt::tout << myTimer;

    return 0;
}