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


#include "bolt/cl/iterator/permutation_iterator.h"
#include "bolt/cl/iterator/transform_iterator.h"
#include "bolt/cl/transform.h"
#include "bolt/cl/generate.h"
#include "bolt/cl/functional.h"
#include "bolt/cl/distance.h"
#include "bolt/miniDump.h"
#include "bolt/unicode.h"

#include <gtest/gtest.h>
#include "common/test_common.h"
#include <boost/program_options.hpp>
#define BCKND cl

namespace po = boost::program_options;

BOLT_FUNCTOR(add_4,
    struct add_4
    {
        int operator() (const int x)  const { return x + 4; }
        typedef int result_type;
    };
);

BOLT_FUNCTOR(add_3,
    struct add_3
    {
        int operator() (const int x)  const { return x + 3; }
        typedef int result_type;
    };
);


int global_id = 0;

int get_global_id(int i)
{
    return global_id++;
}

BOLT_FUNCTOR(gen_input,
    struct gen_input
    {
        int operator() ()  const { return get_global_id(0); }
        typedef int result_type;
    };
);

BOLT_FUNCTOR(gen_input_plus_1,
    struct gen_input_plus_1
    {
        int operator() ()  const { return get_global_id(0) + 1; }
        typedef int result_type;
    };
);

BOLT_FUNCTOR(UDD, 
struct UDD
{
    int i;
    float f;
  
    bool operator == (const UDD& other) const {
        return ((i == other.i) && (f == other.f));
    }
    
    UDD()
        : i(0), f(0) { }
    UDD(int _in)
        : i(_in), f((float)(_in+2) ){ }

};
);

/*Create Device Vector Iterators*/
BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, int, UDD);

/*Create Transform iterators*/
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, add_3, int);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, add_4, int);

BOLT_TEMPLATE_REGISTER_NEW_PERMUTATION_ITERATOR( bolt::cl::device_vector<int>::iterator, bolt::cl::device_vector<int>::iterator);
BOLT_TEMPLATE_REGISTER_NEW_PERMUTATION_ITERATOR( bolt::cl::device_vector<int>::iterator, bolt::cl::constant_iterator<int>);

BOLT_FUNCTOR(gen_input_udd,
    struct gen_input_udd
    {
        UDD operator() ()  const 
       { 
            int i=get_global_id(0);
            UDD temp;
            temp.i = i;
            temp.f = (float)i;
            return temp; 
        }
        typedef int result_type;
    };
);


TEST( PermutationIterator, FirstTest)
{
    {
        const int length = 1<<10;
        std::vector< int > svIndexVec( length );
        std::vector< int > svElementVec( length );
        std::vector< int > svOutVec( length );
        bolt::BCKND::device_vector< int > dvIndexVec( length );
        bolt::BCKND::device_vector< int > dvElementVec( length );        
        bolt::BCKND::device_vector< int > dvOutVec( length );

        gen_input gen;
        typedef std::vector< int >::const_iterator                                                          sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                                 dv_itr;
        typedef bolt::BCKND::permutation_iterator< std::vector< int >::const_iterator, 
                                                   std::vector< int >::const_iterator>                      sv_perm_itr;
        typedef bolt::BCKND::permutation_iterator< bolt::BCKND::device_vector< int >::iterator, 
                                                   bolt::BCKND::device_vector< int >::iterator>             dv_perm_itr;

        /*Create Iterators*/
        sv_perm_itr sv_perm_begin (svElementVec.begin(), svIndexVec.begin()), sv_perm_end (svElementVec.end(), svIndexVec.end());
        dv_perm_itr dv_perm_begin (dvElementVec.begin(), dvIndexVec.begin()), dv_perm_end (dvElementVec.end(), dvIndexVec.end());

        /*Generate inputs*/
        global_id = 0;
        std::generate(svIndexVec.begin(),   svIndexVec.end(),   gen); 
        global_id = 0;
        std::generate(svElementVec.begin(), svElementVec.end(), gen); 
        global_id = 0;
        bolt::BCKND::generate(dvIndexVec.begin(), dvIndexVec.end(), gen);
        global_id = 0;
        bolt::BCKND::generate(dvElementVec.begin(), dvElementVec.end(), gen);
        global_id = 0;

        sv_perm_itr::difference_type dist1 = bolt::cl::distance(sv_perm_begin, sv_perm_end);
        sv_perm_itr::difference_type dist2 = bolt::cl::distance(dv_perm_begin, dv_perm_end );

        EXPECT_EQ( dist1, dist2 );
        //std::cout << "distance = " << dist1 << "\n" ;

        for(int i =0; i< length; i++)
        {
            int temp1, temp2;
            temp1 = *sv_perm_begin++;
            temp2 = *dv_perm_begin++;
            EXPECT_EQ( temp1, temp2 );
        }
        global_id = 0; // Reset the global id counter
    }
}


TEST( PermutationIterator, UDDTest)
{
    {
        const int length = 1<<10;
        std::vector< int > svIndexVec( length );
        std::vector< UDD > svElementVec( length );
        std::vector< UDD > svOutVec( length );
        bolt::BCKND::device_vector< int > dvIndexVec( length );
        bolt::BCKND::device_vector< UDD > dvElementVec( length );        
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

        gen_input gen;
        gen_input_udd genUDD;
        typedef std::vector< UDD >::const_iterator                                                          sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                                 dv_itr;
        /*Create Permutation iterator with int as index and UDD as the elements*/
        typedef bolt::BCKND::permutation_iterator< std::vector< UDD >::const_iterator, 
                                                   std::vector< int >::const_iterator>                      sv_perm_itr;
        typedef bolt::BCKND::permutation_iterator< bolt::BCKND::device_vector< UDD >::iterator, 
                                                   bolt::BCKND::device_vector< int >::iterator>             dv_perm_itr;

        /*Create Iterators*/
        sv_perm_itr sv_perm_begin (svElementVec.begin(), svIndexVec.begin()), sv_perm_end (svElementVec.end(), svIndexVec.end());
        dv_perm_itr dv_perm_begin (dvElementVec.begin(), dvIndexVec.begin()), dv_perm_end (dvElementVec.end(), dvIndexVec.end());

        /*Generate inputs*/
        global_id = 0;
        std::generate(svIndexVec.begin(),   svIndexVec.end(),   gen); 
        global_id = 0;
        std::generate(svElementVec.begin(), svElementVec.end(), genUDD); 
        global_id = 0;
        bolt::BCKND::generate(dvIndexVec.begin(), dvIndexVec.end(), gen);
        global_id = 0;
        bolt::BCKND::generate(dvElementVec.begin(), dvElementVec.end(), genUDD);
        global_id = 0;

        sv_perm_itr::difference_type dist1 = bolt::cl::distance(sv_perm_begin, sv_perm_end);
        sv_perm_itr::difference_type dist2 = bolt::cl::distance(dv_perm_begin, dv_perm_end );

        EXPECT_EQ( dist1, dist2 );
        //std::cout << "distance = " << dist1 << "\n" ;

        for(int i =0; i< length; i++)
        {
            UDD temp1, temp2;
            temp1 = *sv_perm_begin++;
            temp2 = *dv_perm_begin++;
            EXPECT_EQ( temp1.i, temp2.i );
            EXPECT_EQ( temp1.f, temp2.f );
        }
        global_id = 0; // Reset the global id counter
    }
}



TEST( PermutationIterator, UnaryTransformRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIndexVec( length );
        std::vector< int > svElementVec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );
        bolt::BCKND::device_vector< int > dvIndexVec( length );
        bolt::BCKND::device_vector< int > dvElementVec( length );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;
        gen_input gen;
        gen_input_plus_1 gen_plus_1;
        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;    
        typedef bolt::BCKND::permutation_iterator< bolt::BCKND::device_vector< int >::iterator, 
                                                   bolt::BCKND::device_vector< int >::iterator>        dv_perm_itr;
        typedef bolt::BCKND::permutation_iterator< std::vector< int >::iterator, 
                                                   std::vector< int >::iterator>                       sv_perm_itr;
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIndexVec.begin(), add3), sv_trf_end1 (svIndexVec.end(), add3);
        dv_trf_itr_add3 dv_trf_begin1 (dvIndexVec.begin(), add3), dv_trf_end1 (dvIndexVec.end(), add3);

        sv_perm_itr sv_perm_begin (svElementVec.begin(), svIndexVec.begin()), sv_perm_end (svElementVec.end(), svIndexVec.end());
        dv_perm_itr dv_perm_begin (dvElementVec.begin(), dvIndexVec.begin()), dv_perm_end (dvElementVec.end(), dvIndexVec.end());

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        /*Generate inputs*/
        global_id = 0;
        std::generate(svIndexVec.begin(),   svIndexVec.end(),   gen); 
        global_id = 0;
        std::generate(svElementVec.begin(), svElementVec.end(), gen_plus_1); 
        global_id = 0;
        bolt::BCKND::generate(dvIndexVec.begin(), dvIndexVec.end(), gen);
        global_id = 0;
        bolt::BCKND::generate(dvElementVec.begin(), dvElementVec.end(), gen_plus_1);
        global_id = 0;

        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(dv_perm_begin, dv_perm_end, dvOutVec.begin(), add3);
            /*Compute expected results*/
            std::transform(sv_perm_begin, sv_perm_end, stlOut.begin(), add3);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
            //cmpArrays(svOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}


TEST( PermutationIterator, BinaryTransformRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIndexVec1( length );
        std::vector< int > svElementVec1( length );
        std::vector< int > svIndexVec2( length );
        std::vector< int > svElementVec2( length );

        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );
        bolt::BCKND::device_vector< int > dvIndexVec1( length );
        bolt::BCKND::device_vector< int > dvElementVec1( length );
        bolt::BCKND::device_vector< int > dvIndexVec2( length );
        bolt::BCKND::device_vector< int > dvElementVec2( length );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;
        gen_input gen;
        bolt::cl::plus<int> plus;
        gen_input_plus_1 gen_plus_1;
        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;    
        typedef bolt::BCKND::permutation_iterator< bolt::BCKND::device_vector< int >::iterator, 
                                                   bolt::BCKND::device_vector< int >::iterator>        dv_perm_itr;
        typedef bolt::BCKND::permutation_iterator< std::vector< int >::iterator, 
                                                   std::vector< int >::iterator>                       sv_perm_itr;
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIndexVec1.begin(), add3), sv_trf_end1 (svIndexVec1.end(), add3);
        dv_trf_itr_add3 dv_trf_begin1 (dvIndexVec1.begin(), add3), dv_trf_end1 (dvIndexVec1.end(), add3);

        sv_perm_itr sv_perm_begin1 (svElementVec1.begin(), svIndexVec1.begin()), sv_perm_end1 (svElementVec1.end(), svIndexVec1.end());
        sv_perm_itr sv_perm_begin2 (svElementVec2.begin(), svIndexVec2.begin());
        dv_perm_itr dv_perm_begin1 (dvElementVec1.begin(), dvIndexVec1.begin()), dv_perm_end1 (dvElementVec1.end(), dvIndexVec1.end());
        dv_perm_itr dv_perm_begin2 (dvElementVec2.begin(), dvIndexVec2.begin());

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        /*Generate inputs*/
        global_id = 0;
        std::generate(svIndexVec1.begin(),   svIndexVec1.end(),   gen); 
        global_id = 0;
        std::generate(svElementVec1.begin(), svElementVec1.end(), gen_plus_1); 
        global_id = 0;
        bolt::BCKND::generate(dvIndexVec1.begin(), dvIndexVec1.end(), gen);
        global_id = 0;
        bolt::BCKND::generate(dvElementVec1.begin(), dvElementVec1.end(), gen_plus_1);
        global_id = 0;

        std::generate(svIndexVec2.begin(),   svIndexVec2.end(),   gen); 
        global_id = 0;
        std::generate(svElementVec2.begin(), svElementVec2.end(), gen_plus_1); 
        global_id = 0;
        bolt::BCKND::generate(dvIndexVec2.begin(), dvIndexVec2.end(), gen);
        global_id = 0;
        bolt::BCKND::generate(dvElementVec2.begin(), dvElementVec2.end(), gen_plus_1);
        global_id = 0;

        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(dv_perm_begin1, dv_perm_end1, dv_perm_begin2, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_perm_begin1, sv_perm_end1, sv_perm_begin2, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(dv_perm_begin1, dv_perm_end1, dv_trf_begin1, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_perm_begin1, sv_perm_end1, sv_trf_begin1, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(dv_perm_begin1, dv_perm_end1, const_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_perm_begin1, sv_perm_end1, const_itr_begin, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(dv_perm_begin1, dv_perm_end1, count_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_perm_begin1, sv_perm_end1, count_itr_begin, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(dv_perm_begin1, dv_perm_end1, dvElementVec2.begin(), dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_perm_begin1, sv_perm_end1, svElementVec2.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(dvElementVec2.begin(), dvElementVec2.end(), dv_perm_begin1, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(svElementVec2.begin(), svElementVec2.end(), sv_perm_begin1, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(const_itr_begin, const_itr_end, dv_perm_begin1, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(const_itr_begin, const_itr_end, sv_perm_begin1, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, dv_perm_begin1, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, sv_perm_begin1, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a permutation Iterators*/
            bolt::cl::transform(count_itr_begin, count_itr_end, dv_perm_begin1, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(count_itr_begin, count_itr_end, sv_perm_begin1, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

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
