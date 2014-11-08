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
 *  Transform Scan Benchmark
 *****************************************************************************/

// extra code has been added to allow benchmarking of a
// user defined struct "vecN" and an arbitrary associative-commutative
// operator vecNplus
// use the preprocessor commands of USE_VECN or EXCLUSIVE scan below

#define BOLT_ENABLE_PROFILING
#include "stdafx.h"

#include "bolt/unicode.h"
#include "bolt/statisticalTimer.h"
#include "bolt/countof.h"
#include "bolt/cl/transform_scan.h"
#include "bolt/AsyncProfiler.h"

AsyncProfiler aProfiler;

const std::streamsize colWidth = 26;

#define USE_VECN 1
#define EXCLUSIVE 0
#define CALC_SPEEDUP 0

BOLT_FUNCTOR(vecN,
struct vecN
{
    //double a;
    //double b;
    //double c;
    //double d;
    int a;
    int b;
    //int c;
    //int d;
    //int e, f, g, h;
    //float e, f, g, h;
    //float i, j, k, l;
    //float m, n, o, p;

    bool operator==(const vecN& rhs) const
    {
        bool equal = true;
        double th = 0.00001;
        equal = ( (1.0*a - rhs.a)/rhs.a < th && (1.0*a - rhs.a)/rhs.a > -th) ? equal : false;
        equal = ( (1.0*b - rhs.b)/rhs.b < th && (1.0*b - rhs.b)/rhs.b > -th) ? equal : false;
        //equal = ( (1.0*c - rhs.c)/rhs.c < th && (1.0*c - rhs.c)/rhs.c > -th) ? equal : false;
        //equal = ( (1.0*d - rhs.d)/rhs.d < th && (1.0*d - rhs.d)/rhs.d > -th) ? equal : false;
/*
        equal = ( (1.0*e - rhs.e)/rhs.e < th && (1.0*e - rhs.e)/rhs.e > -th) ? equal : false;
        equal = ( (1.0*f - rhs.f)/rhs.f < th && (1.0*f - rhs.f)/rhs.f > -th) ? equal : false;
        equal = ( (1.0*g - rhs.g)/rhs.g < th && (1.0*g - rhs.g)/rhs.g > -th) ? equal : false;
        equal = ( (1.0*h - rhs.h)/rhs.h < th && (1.0*h - rhs.h)/rhs.h > -th) ? equal : false;
*/
        return equal;
    }
};
);

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
/*
        _result.e = lhs.e+rhs.e;
        _result.f = lhs.f+rhs.f;
        _result.g = lhs.g+rhs.g;
        _result.h = lhs.h+rhs.h;
*/
        return _result;
    };
}; 
);
vecNplus vNp;

BOLT_FUNCTOR(vecNsquare,
struct vecNsquare
{
    vecN operator()(const vecN &rhs) const
    {
        vecN _result;
        _result.a = rhs.a*rhs.a;
        _result.b = rhs.b*rhs.b;
        //_result.c = lhs.c+rhs.c;
        //_result.d = lhs.d+rhs.d;
/*
        _result.e = lhs.e+rhs.e;
        _result.f = lhs.f+rhs.f;
        _result.g = lhs.g+rhs.g;
        _result.h = lhs.h+rhs.h;
*/
        return _result;
    };
}; 
);
vecNsquare vNs;


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
    bool serial = false;
    bolt::cl::control& ctrl = bolt::cl::control::getDefault();

    /******************************************************************************
    * Parameter parsing                                                           *
    ******************************************************************************/
    try
    {
        // Declare the supported options.
        po::options_description desc( "OpenCL Scan command line options" );
        desc.add_options()
            ( "help,h",			"produces this help message" )
            ( "version,v",		"Print queryable version information from the Bolt CL library" )
            ( "queryOpenCL,q",  "Print queryable platform and device info and return" )
            ( "gpu,g",          "Report only OpenCL GPU devices" )
            ( "cpu,c",          "Report only OpenCL CPU devices" )
            ( "all,a",          "Report all OpenCL devices" )
            ( "reference-serial,r", "Run reference serial algorithm (std::scan)." )
            ( "systemMemory,s", "Allocate vectors in system memory, otherwise device memory" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( 0 ), "Specify the platform under test using the index reported by -q flag" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( 0 ), "Specify the device under test using the index reported by the -q flag.  "
                    "Index is relative with respect to -g, -c or -a flags" )
            ( "length,l",       po::value< size_t >( &length )->default_value( 1<<26 ), "Specify the length of scan array" )
            ( "iterations,i",   po::value< size_t >( &iterations )->default_value( 10 ), "Number of samples in timing loop" )
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

        if( vm.count( "reference-serial" ) )
        {
            serial = true;
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

    if ( serial )
    {
        ctrl.setForceRunMode( bolt::cl::control::SerialCpu );  // choose serial std::scan
    }

    // Platform vector contains all available platforms on system
    std::vector< ::cl::Platform > platforms;
    bolt::cl::V_OPENCL( ::cl::Platform::get( &platforms ), "Platform::get() failed" );

    if( print_clInfo )
    {
        return 0;    }

    // Device info
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();

    ::cl::CommandQueue myQueue( myContext, devices.at( userDevice ) , CL_QUEUE_PROFILING_ENABLE);

    //  Now that the device we want is selected and we have created our own cl::CommandQueue, set it as the
    //  default cl::CommandQueue for the Bolt API
    ctrl.setCommandQueue( myQueue );


    std::string strDeviceName = ctrl.getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
    bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    std::cout << "Device under test : " << strDeviceName << std::endl;

    /******************************************************************************
    * Benchmark logic                                                             *
    ******************************************************************************/
    bolt::statTimer& myTimer = bolt::statTimer::getInstance( );
    bolt::statTimer& stdTimer = bolt::statTimer::getInstance( );
    myTimer.Reserve( 2, iterations );
    size_t scanId	= myTimer.getUniqueID( _T( "boltScan" ), 0 );
    size_t stdScanId	= myTimer.getUniqueID( _T( "stdScan" ), 1 );
    int reportLength = 30;
    float reportFrequency = 1.f*iterations/reportLength;
    float nextReport = 0.f;
    bolt::cl::square<int> squareInt;
    bolt::cl::plus<int> plusInt;

    if( systemMemory )
    {

        std::vector< int > input( length, 1 );
        std::vector< int > output( length );

        for( unsigned i = 0; i < iterations; ++i )
        {
            myTimer.Start( scanId );
            bolt::cl::transform_inclusive_scan( ctrl, input.begin( ), input.end( ), output.begin( ), squareInt, plusInt );
            myTimer.Stop( scanId );
        }
    }
    else
    {


#if USE_VECN

        vecN init_vecN;
        init_vecN.a = 1;//1.00001;
        init_vecN.b = 2;//1.000001;
        //init_vecN.c = 3;//1.0000001;
        //init_vecN.d = 4;//1.00000001;
/*
        init_vecN.e = 1.001f;
        init_vecN.f = 1.001f;
        init_vecN.g = 1.001f;
        init_vecN.h = 1.001f;
*/
        //printf("init_vecN: %f,%f,%f,%f (Bolt)\n",init_vecN.a,init_vecN.b,init_vecN.c,init_vecN.d);


        vecN empty_vecN;
        empty_vecN.a = 0;
        empty_vecN.b = 0;
        //empty_vecN.c = 0;
        //empty_vecN.d = 0;
/*
        empty_vecN.e = 0;//1.f;
        empty_vecN.f = 0;//1.f;
        empty_vecN.g = 0;//1.f;
        empty_vecN.h = 0;//1.f;
*/
        //printf("empty_vecN: %f,%f,%f,%f (Bolt)\n",empty_vecN.a,empty_vecN.b,empty_vecN.c,empty_vecN.d);
        bolt::cl::device_vector< vecN > input( length, init_vecN, CL_MEM_READ_WRITE, true, ctrl );
        bolt::cl::device_vector< vecN > output( length, empty_vecN, CL_MEM_READ_WRITE, false, ctrl );

#else
        bolt::cl::device_vector< int > input( length, 1, CL_MEM_READ_WRITE, true, ctrl);
        bolt::cl::device_vector< int > output( length, 0, CL_MEM_READ_WRITE, false, ctrl);
#endif

        for( unsigned i = 0; i < iterations; ++i )
        {
            if (i>nextReport)
            {
                //printf("#");fflush(stdout);
                //nextReport += reportFrequency;
            }
            myTimer.Start( scanId );
#if EXCLUSIVE
#if USE_VECN
            bolt::cl::transform_exclusive_scan( ctrl, input.begin( ), input.end( ), output.begin( ), vNs, empty_vecN, vNp );
#else
            bolt::cl::transform_exclusive_scan( ctrl, input.begin( ), input.end( ), output.begin( ), squareInt, 0, plusInt); // here
#endif
#else
#if USE_VECN
            bolt::cl::transform_inclusive_scan( ctrl, input.begin( ), input.end( ), output.begin( ), vNs, vNp );
#else
            bolt::cl::transform_inclusive_scan( ctrl, input.begin( ), input.end( ), output.begin( ), squareInt, plusInt );
#endif
#endif
            myTimer.Stop( scanId );
        }

#if CALC_SPEEDUP

#if USE_VECN
        ::std::vector< vecN > refInput(  length, init_vecN);
        ::std::vector< vecN > refOutput( length, empty_vecN);
#if EXCLUSIVE
        refInput[0] = empty_vecN;
#endif
#else
        ::std::vector< int > refInput(  length, 1);
        ::std::vector< int > refOutput( length, 0);
#if EXCLUSIVE
        refInput[0] = 0;
#endif
#endif

        myTimer.Start( stdScanId );
#if USE_VECN
        ::std::transform(refInput.begin(), refInput.end(), refInput.begin(), vNs);
        ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), vNp);
#else
        ::std::transform(refInput.begin(), refInput.end(), refInput.begin(), vNs);
        ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin());
#endif
        myTimer.Stop( stdScanId );
#if USE_VECN
        /*
        vecN last = output[length-1];
        double a = last.a;
        double b = last.b;
        double c = last.c;
        double d = last.d;
        vecN refLast = refOutput[length-1];
        double ra = refLast.a;
        double rb = refLast.b;
        double rc = refLast.c;
        double rd = refLast.d;
        
        printf("Final Element:\n\t%.7e,%.7e,%.7e,%.7e (Bolt)\n\t%.7e,%.7e,%.7e,%.7e (Std)\n",
            a,
            b,
            c,
            d,
            ra,
            rb,
            rc,
            rd
            );
            */
#else
        //printf("Final Element: %i (Bolt), %i (Std)\n", (int)output[length-1], (int)refOutput[length-1]);
#endif
        //bolt::cl::device_vector< int >::pointer pData = output.data( );
        //output.data();
#if 0
        printf("Comparing Results...\n");
        int maxPrint = 16;
        int errorCount = 0;
        for (size_t j = 0; j < length; j++)
        {
            vecN val = output[j];
            vecN refVal = refOutput[j];
            if (!(val == refVal))
            {
                errorCount++;
                if (errorCount < maxPrint)
                {
                    int s = (int)(j)+1;

                    vecN last = output[j];
                    float a = last.a;
                    float b = last.b;
                    float c = last.c;
                    float d = last.d;
                    vecN refLast = refOutput[j];
                    float ra = refLast.a;
                    float rb = refLast.b;
                    float rc = refLast.c;
                    float rd = refLast.d;
                    printf("[%i]: %.7e,%.7e,%.7e,%.7e (Bolt)\n[%i]: %.7e,%.7e,%.7e,%.7e (Std)\n", j,
                        a,
                        b,
                        c,
                        d,
                        j,
                        ra,
                        rb,
                        rc,
                        rd );

                    //printf("Input[%i] = %i; Bolt = %i; StdRef = %i\n",
                    //    (int)j,
                    //    (int)input[j],
                    //    (int)output[j],
                    //    (int)refOutput[j]);
                }// end error count
            }// if error
        } // for all elements
#endif // compare results

#endif // calc speedup
    }

    //	Remove all timings that are outside of 2 stddev (keep 65% of samples); we ignore outliers to get a more consistent result
    size_t pruned = myTimer.pruneOutliers( 1.0 );
    double scanTime = myTimer.getAverageTime( scanId );
#if USE_VECN
    double scanMB = ( length * sizeof( vecN ) ) / (1024.0 * 1024.0);
#else
    double scanMB = ( length * sizeof( int ) ) / (1024.0 * 1024.0);
#endif
    double scanGB = scanMB / 1024.0;

    bolt::tout << std::left;
#if CALC_SPEEDUP
    double stdScanTime = myTimer.getAverageTime( stdScanId );
    double speedup = stdScanTime / scanTime;
    bolt::tout << std::setw( colWidth ) << _T( "Transform_Scan profile: " ) << _T( "[" ) << iterations-pruned << _T( "] samples" ) << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "       Size (MB): " ) << scanMB << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "        Time (s): " ) << scanTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (GB/s): " ) << scanGB / scanTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Ref Time (s): " ) << stdScanTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "         Speedup: " ) << speedup << std::endl;
#else
    bolt::tout << std::setw( colWidth ) << _T( "Transfom_Scan profile: " ) << _T( "[" ) << iterations-pruned << _T( "] samples" ) << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Size (GB): " ) << scanGB << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Time (s): " ) << scanTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (GB/s): " ) << scanGB / scanTime << std::endl;
#endif
    bolt::tout << std::endl;

//	bolt::tout << myTimer;


#ifdef BOLT_ENABLE_PROFILING
    aProfiler.end();
    aProfiler.writeSum(std::cout);
#endif

    return 0;
}