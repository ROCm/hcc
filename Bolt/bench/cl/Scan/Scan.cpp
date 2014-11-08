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

/******************************************************************************
 *  Scan Benchmark
 *****************************************************************************/

// extra code has been added to allow benchmarking of a
// user defined struct "vecN" and an arbitrary associative-commutative
// operator vecNplus
// use the preprocessor commands of USE_VECN or EXCLUSIVE scan below

#define BOLT_ENABLE_PROFILING

#include "bolt/AsyncProfiler.h"
AsyncProfiler aProfiler("default");

#include "stdafx.h"
#include "bolt/unicode.h"
#include "bolt/statisticalTimer.h"
#include "bolt/countof.h"
#include "bolt/cl/scan.h"

#include "bolt/cl/functional.h"
#include <fstream>
const std::streamsize colWidth = 26;

#define USE_VECN 0

#if USE_VECN
#define VECN(X) X
#define TYPE vecN
#define INIT_VAL init_vecN
#define EMPTY_VAL empty_vecN
#define BIN_OP fNp
#else
#define VECN(X)
#define TYPE int
#define INIT_VAL 1
#define EMPTY_VAL 0
#define BIN_OP bolt::cl::plus<int>()
#endif


BOLT_FUNCTOR(vecN,
struct vecN
{
    int a;
    int b;
    //int c;
    //int d;

    bool operator==(const vecN& rhs) const
    {
        bool equal = true;
        equal = (a==rhs.a) ? equal : false;
        equal = (b==rhs.b) ? equal : false;
        //equal = (lhs.c==rhs.c) ? equal : false;
        //equal = (lhs.d==rhs.d) ? equal : false;
        return equal;
    }
};
);
vecN init_vecN = { 1, 2 };
vecN empty_vecN = { 0, 0 };

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< vecN >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< vecN >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_FUNCTOR(vecNplus,
struct vecNplus
{
    vecN operator()(const vecN &lhs, const vecN &rhs) const
    {
        vecN _result;
        _result.a = lhs.a+rhs.a;
        _result.b = lhs.b+rhs.b;
        //_result.c = lhs.c+rhs.c;
        //_result.d = lhs.d+rhs.d;
        return _result;
    };
}; 
);
vecNplus fNp;



int _tmain( int argc, _TCHAR* argv[] )
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
    bool hostMemory = false;
    bool serial = false;
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
        po::options_description desc( "OpenCL Scan command line options" );
        desc.add_options()
            ( "help,h",			"produces this help message" )
            ( "version",		"Print queryable version information from the Bolt CL library" )
            ( "queryOpenCL,q",  "Print queryable platform and device info and return" )
            ( "gpu,g",          "Report only OpenCL GPU devices" )
            ( "cpu,c",          "Report only OpenCL CPU devices" )
            ( "all,a",          "Report all OpenCL devices" )
            ( "reference-serial,r", "Run reference serial algorithm (std::scan)." )
            ( "hostMemory,m",   "Allocate vectors in host memory, otherwise device memory" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( 0 ),
                "Specify the platform under test using the index reported by -q flag" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( 0 ),
                "Specify the device under test using the index reported by the -q flag.  "
                    "Index is relative with respect to -g, -c or -a flags" )
            ( "length,l",       po::value< size_t >( &length )->default_value( 4096 ),
                "Specify the length of scan array" )
            ( "iterations,i",   po::value< size_t >( &iterations )->default_value( 1 ),
                "Number of samples in timing loop" )
            ( "validate,v",     "Validate Bolt scan against serial CPU scan" )
            ( "compare-serial",     "Compare speedup Bolt scan against serial CPU scan" )
            ( "filename,f",     po::value< std::string >( &filename )->default_value( "bench.xml" ),
                "Name of output file" )
            ( "throw-away",   po::value< size_t >( &numThrowAway )->default_value( 0 ),
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

        if( vm.count( "hostMemory" ) )
        {
            hostMemory = true;
        }

        if( vm.count( "reference-serial" ) )
        {
            serial = true;
            hostMemory = true;
        }

        if( vm.count( "validate" ) )
        {
            validate = true;
        }

        if( vm.count( "compare-serial" ) )
        {
            compareSerial = true;
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

    if ( serial )
    {
        ctrl.setForceRunMode( bolt::cl::control::SerialCpu );  // choose serial std::scan
    }

    // Platform vector contains all available platforms on system
    std::vector< ::cl::Platform > platforms;
    bolt::cl::V_OPENCL( ::cl::Platform::get( &platforms ), "Platform::get() failed" );

    if( print_clInfo ) { return 0; }

    // Device info
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
    ::cl::CommandQueue myQueue( myContext, devices.at( userDevice ) , CL_QUEUE_PROFILING_ENABLE);
    ctrl.setCommandQueue( myQueue );
    std::string strDeviceName = ctrl.getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
    bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );
    ctrl.setWGPerComputeUnit( 40 ); // for Tahiti
    std::cout << "Device under test : " << strDeviceName << std::endl;

    /******************************************************************************
     * Setup Vectors
     *****************************************************************************/
    bolt::cl::device_vector< TYPE > input( length, INIT_VAL, CL_MEM_READ_WRITE, true, ctrl ); // device
    bolt::cl::device_vector< TYPE > output( length, EMPTY_VAL, CL_MEM_READ_WRITE, false, ctrl );
    std::vector< TYPE > hInput(  length, INIT_VAL ); // host 
    std::vector< TYPE > hOutput( length, EMPTY_VAL );
    std::vector< TYPE > refInput( length, INIT_VAL); // reference
    std::vector< TYPE > refOutput(length, EMPTY_VAL);

    /******************************************************************************
     * Benchmark logic
     *****************************************************************************/
    bolt::statTimer& myTimer = bolt::statTimer::getInstance( );
    bolt::statTimer& stdTimer = bolt::statTimer::getInstance( );
    myTimer.Reserve( 2, iterations );
    size_t scanId	= myTimer.getUniqueID( _T( "boltScan" ), 0 );
    size_t stdScanId	= myTimer.getUniqueID( _T( "stdScan" ), 1 );

    /******************************************************************************
     * Perform Benchmarking
     *****************************************************************************/
    aProfiler.throwAway( numThrowAway );
    if( hostMemory )
    {
        for( unsigned int i = 0; i < numThrowAway; ++i )
        {
            bolt::cl::inclusive_scan( ctrl, hInput.begin( ), hInput.end( ), hOutput.begin( ), BIN_OP );
        }
        for( unsigned int i = 0; i < iterations; ++i )
        {
            myTimer.Start( scanId );
            bolt::cl::inclusive_scan( ctrl, hInput.begin( ), hInput.end( ), hOutput.begin( ), BIN_OP );
            myTimer.Stop( scanId );
        }
    }
    else
    {
        for( unsigned int i = 0; i < numThrowAway; ++i )
        {
            bolt::cl::inclusive_scan( ctrl, input.begin( ), input.end( ), output.begin( ), BIN_OP );
        }
        for( unsigned int i = 0; i < iterations; ++i )
        {
            myTimer.Start( scanId );
            bolt::cl::inclusive_scan( ctrl, input.begin( ), input.end( ), output.begin( ), BIN_OP );
            myTimer.Stop( scanId );
        }
    } // if systemMemory

    /******************************************************************************
     * Serial Scan (if needed)
     *****************************************************************************/
    if (validate || compareSerial)
    {
        myTimer.Start( stdScanId );
        ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), BIN_OP );
        myTimer.Stop( stdScanId );
    }

        /******************************************************************************
         * Validate
         *****************************************************************************/
        if (validate)
        {
            printf("Validating Results...\n");
            int maxPrint = 16;
            int errorCount = 0;
            TYPE val;
            TYPE refVal;
            bool equal = true;
            for (size_t j = 0; j < length; j++)
            {
                val = hostMemory ? hOutput[j] : output[j];
                refVal = refOutput[j];

                if (!(val == refVal))
                {
                    equal = false;
                    errorCount++;
                    if (errorCount < maxPrint)
                    {
                        int s = (int)(j)+1;
#if USE_VECN
                        int a = val.a;
                        int b = val.b;
                        //int c = val.c;
                        //int d = val.d;
                        int ra = refVal.a;
                        int rb = refVal.b;
                        //int rc = refVal.c;
                        //int rd = refVal.d;
                        printf("Bolt[%i]: %i, %i\n Std[%i]: %i, %i\n", j,
                            a, b, j, ra, rb );
                        //printf("Bolt[%i]: %i, %i, %i, %i\n Std[%i]: %i, %i, %i, %i\n", j,
                        //    a, b, c, d, j, ra, rb, rc, rd );
#else
                        printf("Bolt[%i]: %i\n Std: %i\n",
                            (int)j,
                            (int)output[j],
                            (int)refOutput[j]);
#endif

                        
                    }// end error count
                    else
                    {
                        break;
                    }
                }// if error
            } // for all elements
            if (equal)
            {
                printf("Validation PASSED\n\n");
            }
            else
            {
                printf("Validation FAILED!\n");
            }
        } // if validate

    //	Remove all timings that are outside of 2 stddev (keep 65% of samples);
    //  we ignore outliers to get a more consistent result
    size_t pruned = myTimer.pruneOutliers( 1.0 );
    double scanTime = myTimer.getAverageTime( scanId );
    double scanTimeMs = scanTime * 1000.0;
    double scanMB = ( length * sizeof( TYPE ) ) / (1024.0 * 1024.0);
    double scanGB = scanMB / 1024.0;

    bolt::tout << std::left;
    bolt::tout << std::setw( colWidth ) << _T( "Scan profile: " ) << _T( "[" )
        << iterations-pruned << _T( "] samples" ) << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "   Size (MB): " ) << scanMB << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "   Time (ms): " ) << scanTimeMs << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "Speed (GB/s): " ) << scanGB / scanTime << std::endl;

    if (compareSerial || validate)
    {
        double stdScanTime = myTimer.getAverageTime( stdScanId );
        double stdScanTimeMs = stdScanTime * 1000.0;
        double speedup = stdScanTime / scanTime;
        bolt::tout << std::setw( colWidth ) << _T( "CPU Time (ms): " ) << stdScanTime << std::endl;
        bolt::tout << std::setw( colWidth ) << _T( "CPU Speed (GB/s): " ) << scanGB / stdScanTime << std::endl;
        bolt::tout << std::setw( colWidth ) << _T( "GPU Speedup: " ) << speedup << std::endl;
    }


    bolt::tout << std::endl;

//	bolt::tout << myTimer;

    aProfiler.end();
    std::ofstream outFile( filename.c_str() );
    aProfiler.writeSum( outFile );
    outFile.close();

    return 0;
}