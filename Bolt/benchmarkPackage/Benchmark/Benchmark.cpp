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
 *  Benchmark Bolt Functions
 *****************************************************************************/
#define AMP_BENCH 100
#define CL_BENCH  101

#if !defined(BENCHMARK_CL_AMP)
#define BENCHMARK_CL_AMP CL_BENCH
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include "bolt/statisticalTimer.h"
#include "bolt/unicode.h"	

#define BOLT_BENCHMARK 1
#if (BOLT_BENCHMARK == 1)
    #include <random>   
    #if (BENCHMARK_CL_AMP ==  CL_BENCH)
        #define BENCH_BEND cl
        #include "bolt/cl/functional.h"
        #include "bolt/cl/device_vector.h"
        #include "bolt/cl/generate.h"
        #include "bolt/cl/inner_product.h"
        #include "bolt/cl/binary_search.h"
        #include "bolt/cl/copy.h"
        #include "bolt/cl/count.h"
        #include "bolt/cl/fill.h"
        #include "bolt/cl/max_element.h"
        #include "bolt/cl/min_element.h"
        #include "bolt/cl/merge.h"
        #include "bolt/cl/transform.h"
        #include "bolt/cl/transform_reduce.h"
        #include "bolt/cl/scan.h"
        #include "bolt/cl/sort.h"
        #include "bolt/cl/reduce.h"
        #include "bolt/cl/sort_by_key.h"
        #include "bolt/cl/stablesort.h"
        #include "bolt/cl/reduce_by_key.h"
        #include "bolt/cl/stablesort_by_key.h"
        #include "bolt/cl/transform_scan.h"
        #include "bolt/cl/scan_by_key.h"
        #include "bolt/cl/gather.h"
        #include "bolt/cl/scatter.h"

        //#define BOLT_PROFILER_ENABLED
        #define BOLT_BENCH_DEVICE_VECTOR_FLAGS CL_MEM_READ_WRITE,
		#define CONTAINER
        #include "bolt/AsyncProfiler.h"
        #include "bolt/countof.h"

    #elif (BENCHMARK_CL_AMP ==  AMP_BENCH)
        #define BENCH_BEND amp
        #define BOLT_BENCH_DEVICE_VECTOR_FLAGS true,
		#define CONTAINER ,concurrency::array
        #include "bolt/countof.h"
        #include "bolt/amp/functional.h"
        #include "bolt/amp/device_vector.h"
        #include "bolt/amp/generate.h"
        #include "bolt/amp/inner_product.h"
        #include "bolt/amp/binary_search.h"
        #include "bolt/amp/copy.h"
        #include "bolt/amp/count.h"
        #include "bolt/amp/fill.h"
        #include "bolt/amp/max_element.h"
        #include "bolt/amp/min_element.h"
        #include "bolt/amp/merge.h"
        #include "bolt/amp/transform.h"
        #include "bolt/amp/transform_reduce.h"
        #include "bolt/amp/scan.h"
        #include "bolt/amp/sort.h"
        #include "bolt/amp/reduce.h"
        #include "bolt/amp/sort_by_key.h"
        #include "bolt/amp/stablesort.h"
        #include "bolt/amp/reduce_by_key.h"
        #include "bolt/amp/stablesort_by_key.h"
        #include "bolt/amp/transform_scan.h"
        #include "bolt/amp/scan_by_key.h"
        #include "bolt/amp/gather.h"
        #include "bolt/amp/scatter.h"
    #endif
#else
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <iomanip>
#endif


#if defined(_WIN32)
//AsyncProfiler aProfiler("default");
#endif
const std::streamsize colWidth = 26;

#ifndef DATA_TYPE   
    #define DATA_TYPE float
#endif

#include "data_type.h"

#if (BOLT_BENCHMARK == 1)
    #if BENCHMARK_CL_AMP == CL_BENCH
        BOLT_CREATE_DEFINE(Bolt_DATA_TYPE,DATA_TYPE,unsigned int);
    #endif
#endif


//user defined data types and functions and predicates are dedined
// function generator:
#if (BOLT_BENCHMARK == 1)
    std::default_random_engine gen;
    std::uniform_real_distribution<DATA_TYPE> distr(10,1<<30);
    DATA_TYPE RandomNumber() 
    {    
        DATA_TYPE dice_roll = (DATA_TYPE)distr(gen); // generates number in the range 10..1<<31
        return (dice_roll); 
    }
#else
    thrust::default_random_engine gen;
    thrust::uniform_int_distribution<DATA_TYPE> distr(10,1<<30);
    DATA_TYPE RandomNumber() 
    {    
        DATA_TYPE dice_roll = distr(gen); // generates number in the range 10..1<<31
        //std::cout<<dice_roll<<"\n";
        return (dice_roll); 
    }
#endif
/******************************************************************************
 *  Functions Enumerated
 *****************************************************************************/
//static const size_t FList = 23;
enum functionType {
    f_binarytransform = 0,
    f_binarysearch,
    f_copy,
    f_count,
    f_fill,
    f_generate,
    f_innerproduct,
    f_maxelement,
    f_minelement,
    f_merge,
    f_reduce,
    f_reducebykey,
    f_scan,
    f_scanbykey,
    f_sort,
    f_sortbykey,
    f_stablesort,
    f_stablesortbykey,
    f_transformreduce,
    f_transformscan,
    f_unarytransform,
    f_gather,
    f_scatter,
    /*Insert Any function name before this line*/
    FList
};
static char *functionNames[] = {
    "binarytransform",
    "binarysearch",
    "copy",
    "count",
    "fill",
    "generate",
    "innerproduct",
    "maxelement",
    "minelement",
    "merge",
    "reduce",
    "reducebykey",
    "scan",
    "scanbykey",
    "sort",
    "sortbykey",
    "stablesort",
    "stablesortbykey",
    "transformreduce",
    "transformscan",
    "unarytransform",
    "gather",
    "scatter"
};

enum benchmarkType {
    bm_all,
    bm_cpu,
    bm_device,
    bm_deviceMemory,
    bm_filename,
    bm_function,
    bm_gpu,
    bm_help,
    bm_iterations,
    bm_length,
    bm_platform,
    bm_queryOpenCL,
    bm_runMode,
    bm_throwaway,
    bm_vecType,
    bm_version,
    bm_DEVICEMEMORY,
    bm_ALL,
    bm_CPU,
    bm_DEVICE,
    bm_FUNCTION,
    bm_GPU,
    bm_HELP,
    bm_ITERATIONS,
    bm_LENGTH,
    bm_RUNMODE,
    bm_PLATFORM,
    bm_QUERYOPENCL,
    bm_VECTYPE,
    bm_VERSION,
    BENCHMARK_OPTIONS_SIZE
     /*Insert Any Command Line Argument options before this line*/
};

std::string benchmark_options[]={
    "--all",
    "--cpu",
    "--device",
    "--deviceMemory",
    "--filename",
    "--function",
    "--gpu",
    "--help",
    "--iterations",
    "--length",
    "--platform",
    "--queryOpenCL",
    "--runMode",
    "--throw-away",
    "--vecType",
    "--version",
    "-D",
    "-a",
    "-c",
    "-d",
    "-f",
    "-g",
    "-h",
    "-i",
    "-l",
    "-m",
    "-p",
    "-q",
    "-t",
    "-v"
};

/******************************************************************************
 *  Data Types Enumerated
 *****************************************************************************/
enum dataType {
    t_int,
    t_vec2,
    t_vec4,
    t_vec8
};
static char *dataTypeNames[] = {
    "int1",
    "vec2",
    "vec4",
    "vec8"
};
using namespace std;

/******************************************************************************
 *  Initializers
 *****************************************************************************/
DATA_TYPE v1init = {1};
DATA_TYPE v1iden = {0};
vec2 v2init = { 1, 1 };
vec2 v2iden = { 0, 0 };
vec4 v4init = { 1, 1, 1, 1 };
vec4 v4iden = { 0, 0, 0, 0 };
vec8 v8init = { 1, 1, 1, 1, 1, 1, 1, 1 };
vec8 v8iden = { 0, 0, 0, 0, 0, 0, 0, 0 };

/******************************************************************************
 *
 *  Execute Function
 *
 *****************************************************************************/
functionType get_functionindex(std::string &fun)
{
    for(int i =0 ; i < FList; i++)
    {
        if(fun.compare(functionNames[i]) == 0)
            return (functionType)i;
    }
    std::cout<< "Specified Function not listed for Benchmark. exiting";
    exit(0);
}

/******************************************************************************
 *
 *  Execute Function Type
 *
 *****************************************************************************/
template<
    typename VectorType,
    typename Generator,
    typename UnaryFunction,
    typename BinaryFunction,
    typename BinaryFunctionMult,
    typename BinaryPredEq,
    typename BinaryPredLt,
    typename datatype,
    typename maptype>
#if (BOLT_BENCHMARK == 1)
void executeFunctionType(
    bolt::BENCH_BEND::control& ctrl,
    VectorType &input1,
    VectorType &input2,
    VectorType &input3,
    VectorType &output,
    VectorType &output_merge,
    Generator generator,
    UnaryFunction unaryFunct,
    BinaryFunction binaryFunct,
    BinaryFunctionMult binaryFuntMult,
    BinaryPredEq binaryPredEq,
    BinaryPredLt binaryPredLt,
    size_t function,
    size_t iterations,
    size_t siz,
    datatype keys,
    maptype &Map
    )
#else
void executeFunctionType(
    VectorType &input1,
    VectorType &input2,
    VectorType &input3,
    VectorType &output,
    VectorType &output_merge,
    Generator generator,
    UnaryFunction unaryFunct,
    BinaryFunction binaryFunct,
    BinaryFunctionMult binaryFuntMult,
    BinaryPredEq binaryPredEq,
    BinaryPredLt binaryPredLt,
    size_t function,
    size_t iterations,
    size_t siz,
    datatype keys,
    maptype &Map
    )
#endif
{

    bolt::statTimer& myTimer = bolt::statTimer::getInstance( );
    myTimer.Reserve( 1, iterations );
    size_t testId	= myTimer.getUniqueID( _T( "test" ), 0 );
    switch(function)
    {
    case f_merge:
        { 
           /* for(int i =0;i<input1.size();i++)
                std::cout<<input1[i]<<" ";
            std::cout<<"\n";*/
            std::cout <<  functionNames[f_merge] << std::endl;
#if (BOLT_BENCHMARK == 1)
            bolt::BENCH_BEND::sort( ctrl, input1.begin( ), input1.end( ), binaryPredLt);
            bolt::BENCH_BEND::sort( ctrl, input2.begin( ), input2.end( ), binaryPredLt);
#else
            thrust::sort( input1.begin( ), input1.end( ),  binaryPredLt);
            cudaThreadSynchronize();
            thrust::sort( input2.begin( ), input2.end( ), binaryPredLt);
            cudaThreadSynchronize();
#endif
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::merge( ctrl,input1.begin( ),input1.end( ),input2.begin( ),input2.end( ),output_merge.begin( ),binaryPredLt);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::merge( input1.begin( ),input1.end( ),input2.begin( ),input2.end( ),output_merge.begin( ),binaryPredLt);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
                
            }
        }
        break;
	
	case f_binarysearch:
        {
            bool tmp;
            typename VectorType::value_type val;
            std::cout <<  functionNames[f_binarysearch] << std::endl;
#if (BOLT_BENCHMARK == 1)
            bolt::BENCH_BEND::sort( ctrl, input1.begin( ), input1.end( ), binaryPredLt);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
            thrust::sort( input1.begin( ), input1.end( ), binaryPredLt);
            cudaThreadSynchronize();
#endif
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                int index = 0;
                if(iter!=0)
                    index = rand()%iter;
                val = input1[index];
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                tmp = bolt::BENCH_BEND::binary_search( ctrl,input1.begin( ),input1.end( ),val,binaryPredLt);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                tmp = thrust::binary_search( input1.begin( ),input1.end( ),val,binaryPredLt);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_transformreduce:
        {
            typename VectorType::value_type tmp;
            tmp=0;
            std::cout <<  functionNames[f_transformreduce] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                tmp = bolt::BENCH_BEND::transform_reduce( ctrl,input1.begin( ), input1.end( ),unaryFunct, tmp,
                                                                                                 binaryFunct);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                tmp = thrust::transform_reduce( input1.begin( ), input1.end( ),unaryFunct, tmp,
                                                                                  binaryFunct);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_stablesort:
        {
            std::cout <<  functionNames[f_stablesort] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                VectorType inputBackup = input1;
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::stable_sort(ctrl, inputBackup.begin(), inputBackup.end(),binaryPredLt);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::stable_sort( inputBackup.begin(), inputBackup.end(), binaryPredLt);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_stablesortbykey:
        {
            std::cout <<  functionNames[f_stablesortbykey] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                VectorType inputBackup = input1;
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::stable_sort_by_key(ctrl, inputBackup.begin(), inputBackup.end(),input2.begin(),binaryPredLt);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif 
#else
                thrust::stable_sort_by_key( inputBackup.begin(), inputBackup.end(),input2.begin(), binaryPredLt); 
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_sort:
        {
            std::cout <<  functionNames[f_sort] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                VectorType inputBackup = input1;
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::sort(ctrl, inputBackup.begin(), inputBackup.end(),binaryPredLt);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif 
#else
                thrust::sort( inputBackup.begin(), inputBackup.end(), binaryPredLt);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_sortbykey:
        {
            std::cout <<  functionNames[f_sortbykey] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                VectorType inputBackup = input1;
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::sort_by_key(ctrl, inputBackup.begin(), inputBackup.end(), input2.begin( ),binaryPredLt );
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif 
#else
                thrust::sort_by_key( inputBackup.begin(), inputBackup.end(), input2.begin( ), binaryPredLt);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_reduce:
        {
            typename VectorType::value_type tmp;
            tmp=0;
            std::cout <<  functionNames[f_reduce] << std::endl;

            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                tmp = bolt::BENCH_BEND::reduce(ctrl, input1.begin(), input1.end(),tmp,binaryFunct);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                tmp = thrust::reduce( input1.begin(), input1.end(),tmp,binaryFunct);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_reducebykey:
        {
                VectorType keys1(input1.size());

            std::cout <<  functionNames[f_reducebykey] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::reduce_by_key(ctrl, keys.begin(), keys.end(),input2.begin(),keys1.begin(),
                    output.begin(),binaryPredEq, binaryFunct);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif 
#else
                thrust::reduce_by_key( keys.begin(), keys.end(),input2.begin(),keys1.begin(),
                    output.begin(),binaryPredEq, binaryFunct);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_maxelement:
        {
            std::cout <<  functionNames[f_maxelement] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                typename VectorType::iterator itr = bolt::BENCH_BEND::max_element(ctrl, input1.begin(), input1.end(),
                                                                                               binaryPredLt);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                typename VectorType::iterator itr = thrust::max_element( input1.begin(), input1.end(),
                                                                                        binaryPredLt);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_minelement:
        {
            std::cout <<  functionNames[f_minelement] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                #if BENCHMARK_CL_AMP == AMP_BENCH
                typename VectorType::iterator itr = bolt::BENCH_BEND::min_element(ctrl, input1.begin(), input1.end(),
                                                                                               binaryPredLt);
                
                Amp_GPU_wait(ctrl);
                #elif BENCHMARK_CL_AMP == CL_BENCH
                typename VectorType::iterator itr = bolt::BENCH_BEND::min_element(ctrl, input1.begin(), input1.end(),
                                                                                               binaryPredLt,"");
                #endif
#else
                typename VectorType::iterator itr = thrust::min_element( input1.begin(), input1.end(),
                                                                                        binaryPredLt);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_fill:
        {
            typename VectorType::value_type tmp;
            std::cout <<  functionNames[f_fill] << std::endl;

            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::fill(ctrl, input1.begin(), input1.end(),tmp);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::fill( input1.begin(), input1.end(),tmp);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_count:
        {
            typename VectorType::value_type tmp;
            std::cout <<  functionNames[f_count] << std::endl;

            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::count(ctrl, input1.begin(), input1.end(),tmp);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::count( input1.begin(), input1.end(),tmp);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_generate:
        {
            std::cout <<  functionNames[f_generate] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::generate(ctrl, input1.begin(), input1.end(), generator );
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::generate( input1.begin(), input1.end(), generator );
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_innerproduct:
        { 
            typename VectorType::value_type tmp;
            tmp=10;
            std::cout <<  functionNames[f_innerproduct] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                tmp = bolt::BENCH_BEND::inner_product( ctrl, input1.begin( ), input1.end( ), input2.begin(), tmp, binaryFunct, binaryFuntMult);
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                tmp = thrust::inner_product( input1.begin( ), input1.end( ),input2.begin(), tmp, binaryFunct, binaryFuntMult);
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_copy:
        {
            std::cout <<  functionNames[f_copy] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::copy(ctrl, input1.begin(), input1.end(), output.begin() );
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::copy( input1.begin(), input1.end(), output.begin() );
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_unarytransform:
        {
            std::cout <<  functionNames[f_unarytransform] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::transform(ctrl, input1.begin(), input1.end(), output.begin(), unaryFunct );
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::transform( input1.begin(), input1.end(), output.begin(), unaryFunct );
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_binarytransform:
        {
            std::cout <<  functionNames[f_binarytransform] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::transform(ctrl, input1.begin(), input1.end(), input2.begin(), output.begin(), binaryFunct );
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::transform( input1.begin(), input1.end(), input2.begin(), output.begin(), binaryFunct );
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_scan:
        {
            std::cout <<  functionNames[f_scan] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::inclusive_scan(ctrl, input1.begin(), input1.end(), output.begin(), binaryFunct );
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::inclusive_scan( input1.begin(), input1.end(), output.begin(), binaryFunct );
                //thrust::inclusive_scan( input1.begin(), input1.end(), output.begin() );
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_scanbykey:
        {
            std::cout <<  functionNames[f_scanbykey] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::inclusive_scan_by_key(ctrl, keys.begin(), keys.end(), input2.begin(),
                                                   output.begin(), binaryPredEq, binaryFunct );

#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::inclusive_scan_by_key( keys.begin(), keys.end(), input2.begin(),
                                            output.begin(), binaryPredEq, binaryFunct );
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_transformscan:
        {
            std::cout <<  functionNames[f_transformscan] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::transform_inclusive_scan(ctrl, input1.begin(), input1.end(), output.begin(),
                                                                        unaryFunct, binaryFunct );
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::transform_inclusive_scan( input1.begin(), input1.end(), output.begin(),
                                                                 unaryFunct, binaryFunct );
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

    case f_gather:
        {
            std::cout <<  functionNames[f_gather] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::gather( ctrl, Map.begin( ), Map.end( ),input1.begin( ),output.begin());
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::gather( Map.begin( ), Map.end( ),input1.begin( ),output.begin());
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
             }
        }
        break;

    case f_scatter:
        {
            std::cout <<  functionNames[f_scatter] << std::endl;
            for (size_t iter = 0; iter < iterations+1; iter++)
            {
                myTimer.Start( testId );
#if (BOLT_BENCHMARK == 1)
                bolt::BENCH_BEND::scatter( ctrl, input1.begin( ),input1.end( ), Map.begin(), output.begin());
#if BENCHMARK_CL_AMP == AMP_BENCH
                Amp_GPU_wait(ctrl);
#endif
#else
                thrust::scatter(  input1.begin( ),input1.end( ), Map.begin(), output.begin());
                cudaThreadSynchronize();
#endif
                myTimer.Stop( testId );
            }
        }
        break;

        default:
            std::cout << "\nUnsupported function = " << function <<"\n"<< std::endl;
            break;
    } // switch

    size_t length = input1.size();
    double MKeys = length / ( 1024.0 * 1024.0 );
    size_t pruned = myTimer.pruneOutliers( 1.0 );
    double sortTime = myTimer.getAverageTime( testId );
    double testMB = MKeys*siz;
    double testGB = testMB/ 1024.0;
    bolt::tout << std::left;
    bolt::tout << std::setw( colWidth ) << _T( "Test profile: " ) << _T( "[" ) << iterations-pruned << _T( "] samples" ) << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Size (MKeys): " ) << MKeys << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Size (GB): " ) << testGB << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Time (s): " ) << sortTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (GB/s): " ) << testGB / sortTime << std::endl;
    bolt::tout << std::setw( colWidth ) << _T( "    Speed (MKeys/s): " ) << MKeys / sortTime << std::endl;
    bolt::tout << std::endl;

}

/******************************************************************************
 *
 *  Determine types
 *
 *****************************************************************************/
#if (BOLT_BENCHMARK == 1)
void executeFunction(
    bolt::BENCH_BEND::control& ctrl,
    size_t vecType,
    bool hostMemory,
    size_t length,
    size_t routine,
    size_t iterations )
#else
void executeFunction(
    size_t vecType,
    bool hostMemory,
    size_t length,
    size_t routine,
    size_t iterations )
#endif
{
    size_t siz;
    if (vecType == t_int)
    {
#if (BOLT_BENCHMARK == 1)

#if ((BOLT_BENCHMARK == 1)&&(BENCHMARK_CL_AMP ==  CL_BENCH))
		BOLT_ADD_DEPENDENCY(intgen, Bolt_DATA_TYPE);
#endif
		intgen                    generator;		
        bolt::BENCH_BEND::square<DATA_TYPE>   unaryFunct; 
        bolt::BENCH_BEND::plus<DATA_TYPE>     binaryFunct;
        bolt::BENCH_BEND::equal_to<DATA_TYPE> binaryPredEq;
        bolt::BENCH_BEND::less<DATA_TYPE>     binaryPredLt;
        bolt::BENCH_BEND::multiplies<DATA_TYPE>     binaryFunctMult;
#else
        intgen     generator;
        vec1square unaryFunct;
        thrust::plus<DATA_TYPE>   binaryFunct;
        thrust::equal_to<DATA_TYPE>  binaryPredEq;;
        thrust::less<DATA_TYPE>   binaryPredLt;
        thrust::multiplies<DATA_TYPE>   binaryFunctMult;
#endif

        siz = sizeof(DATA_TYPE);
        std::vector<int> Map(length);
        std::vector<DATA_TYPE> input1(length);
        std::vector<DATA_TYPE> input2(length);
        std::vector<DATA_TYPE> input3(length);
        std::vector<DATA_TYPE> output(length);
        std::vector<DATA_TYPE> output_merge(length*2) ;
        std::generate(input1.begin(), input1.end(), RandomNumber);
        std::generate(input2.begin(), input2.end(), RandomNumber);
        std::generate(input3.begin(), input3.end(), RandomNumber);
        std::generate(output.begin(), output.end(), RandomNumber);
        std::generate(output_merge.begin(), output_merge.end(), RandomNumber);
        for (size_t i = 0; i < input1.size(); i++)
        {
             Map[i] = (int)i;
        }
        std::vector<DATA_TYPE> keys(length,v1iden );  /* Keys: 1 2 2 3 3 3 4 4 4 4 5 5  5 5 5 6 6 ..... */
        int len = (int)input1.size();
        keysGeneration(keys,len);

        if (hostMemory) {
#if (BOLT_BENCHMARK == 1)
            
            executeFunctionType( ctrl, input1, input2, input3, output, output_merge, 
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,keys,Map);
#else
            
            executeFunctionType( input1, input2, input3, output, output_merge, 
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,keys,Map);
#endif
        }
        else
        {
#if (BOLT_BENCHMARK == 1)
            bolt::BENCH_BEND::device_vector<DATA_TYPE CONTAINER> binput1(input1.begin(), input1.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);
            bolt::BENCH_BEND::device_vector<DATA_TYPE CONTAINER> binput2(input2.begin(), input2.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);
            bolt::BENCH_BEND::device_vector<DATA_TYPE CONTAINER> binput3(input3.begin(), input3.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);
            bolt::BENCH_BEND::device_vector<DATA_TYPE CONTAINER> boutput(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl); 
            bolt::BENCH_BEND::device_vector<DATA_TYPE CONTAINER> boutput_merge(output_merge.begin(), output_merge.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl); 
            bolt::BENCH_BEND::device_vector<DATA_TYPE CONTAINER> bkeys(keys.begin(),keys.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);
            bolt::BENCH_BEND::device_vector<int> bMap(Map.begin(),Map.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);

            executeFunctionType( ctrl, binput1, binput2, binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,bkeys,bMap);
#else
            thrust::device_vector<DATA_TYPE> binput1(input1.begin(), input1.end());
            thrust::device_vector<DATA_TYPE> binput2(input2.begin(), input2.end());
            thrust::device_vector<DATA_TYPE> binput3(input3.begin(), input3.end());
            thrust::device_vector<DATA_TYPE> boutput(output.begin(), output.end());
            thrust::device_vector<DATA_TYPE> boutput_merge(output_merge.begin(), output_merge.end());
            thrust::device_vector<DATA_TYPE> bkeys(keys.begin(),keys.end());
            thrust::device_vector<int> bMap(Map.begin(),Map.end());

            executeFunctionType( binput1, binput2, binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,bkeys,bMap);
#endif
        }
    }

    else if (vecType == t_vec2)
    {
        vec2gen     generator;
        vec2square  unaryFunct;
        vec2plus    binaryFunct;
        vec2equal   binaryPredEq;
        vec2less    binaryPredLt;
        vec2mult    binaryFunctMult;
        siz = sizeof(vec2);

#if ((BOLT_BENCHMARK == 1)&&(BENCHMARK_CL_AMP ==  CL_BENCH))
        BOLT_ADD_DEPENDENCY(vec2, Bolt_DATA_TYPE);
#endif
        std::vector<int> Map(length);
        std::vector<vec2> input1(length);
        std::vector<vec2> input2(length);
        std::vector<vec2> input3(length);
        std::vector<vec2> output(length);
        std::vector<vec2> output_merge(length*2) ;

        std::generate(input1.begin(), input1.end(),RandomNumber);
        std::generate(input2.begin(), input2.end(),RandomNumber);
        std::generate(input3.begin(), input3.end(),RandomNumber);
        std::generate(output.begin(), output.end(),RandomNumber);
        std::generate(output_merge.begin(), output_merge.end(), RandomNumber);
        for (size_t i = 0; i < input1.size(); i++)
        {
             Map[i] = (int)i;
        }
        std::vector<vec2> keys(length,v2iden ); /* Keys: 1 2 2 3 3 3 4 4 4 4 5 5  5 5 5 6 6 ..... */
        int len = (int)input1.size();
        keysGeneration(keys,len);

        if (hostMemory) {
#if (BOLT_BENCHMARK == 1)
            executeFunctionType( ctrl, input1, input2, input3, output, output_merge, 
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,keys,Map);
#else
            executeFunctionType( input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,keys,Map);
#endif
        }
        else
        {
#if (BOLT_BENCHMARK == 1)
            bolt::BENCH_BEND::device_vector<vec2 CONTAINER> binput1(input1.begin(), input1.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec2 CONTAINER> binput2(input2.begin(), input2.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec2 CONTAINER> binput3(input3.begin(), input3.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec2 CONTAINER> boutput(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec2 CONTAINER> boutput_merge(output_merge.begin(), output_merge.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl); 
            bolt::BENCH_BEND::device_vector<vec2 CONTAINER> bkeys(keys.begin(), keys.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);
            bolt::BENCH_BEND::device_vector<int> bMap(Map.begin(),Map.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);

            executeFunctionType( ctrl, binput1, binput2,binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,bkeys,bMap);
#else
            thrust::device_vector<vec2> binput1(input1.begin(), input1.end() );
            thrust::device_vector<vec2> binput2(input2.begin(), input2.end() );
            thrust::device_vector<vec2> binput3(input3.begin(), input3.end() );
            thrust::device_vector<vec2> boutput(output.begin(), output.end() );
            thrust::device_vector<vec2> boutput_merge(output_merge.begin(), output_merge.end() );
            thrust::device_vector<vec2> bkeys(keys.begin(), keys.end());
            thrust::device_vector<int> bMap(Map.begin(),Map.end());

            executeFunctionType( binput1, binput2,binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,bkeys,bMap);
#endif
        }
    }

    else if (vecType == t_vec4)
    {
        siz = sizeof(vec4);
        vec4gen     generator;
        vec4square  unaryFunct;
        vec4plus    binaryFunct;
        vec4equal   binaryPredEq;
        vec4less    binaryPredLt;
        vec4mult    binaryFunctMult;
#if ((BOLT_BENCHMARK == 1)&&(BENCHMARK_CL_AMP ==  CL_BENCH))
        BOLT_ADD_DEPENDENCY(vec4, Bolt_DATA_TYPE);
#endif
        std::vector<int> Map(length);        
        std::vector<vec4> input1(length, v4init);
        std::vector<vec4> input2(length, v4init);
        std::vector<vec4> input3(length, v4init);
        std::vector<vec4> output(length, v4iden);
        std::vector<vec4> output_merge(length * 2, v4iden);
        std::generate(input1.begin(), input1.end(),RandomNumber);
        std::generate(input2.begin(), input2.end(),RandomNumber);
        std::generate(input3.begin(), input3.end(),RandomNumber);
        std::generate(output.begin(), output.end(),RandomNumber);
        std::generate(output_merge.begin(), output_merge.end(), RandomNumber);
        for (size_t i = 0; i < input1.size(); i++)
        {
             Map[i] = (int)i;
        }
        
        std::vector<vec4> keys(length,v4iden ); /* Keys: 1 2 2 3 3 3 4 4 4 4 5 5  5 5 5 6 6 ..... */
        int len = (int)input1.size();
        keysGeneration(keys,len);

        if (hostMemory) {
#if (BOLT_BENCHMARK == 1)
            executeFunctionType( ctrl, input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,keys,Map);
#else
            executeFunctionType( input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,keys,Map);
#endif
        }
        else
        {
#if (BOLT_BENCHMARK == 1)
            bolt::BENCH_BEND::device_vector<vec4 CONTAINER> binput1(input1.begin(), input1.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec4 CONTAINER> binput2(input2.begin(), input2.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec4 CONTAINER> binput3(input3.begin(), input3.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec4 CONTAINER> boutput(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec4 CONTAINER> boutput_merge(output_merge.begin(), output_merge.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);					
            bolt::BENCH_BEND::device_vector<vec4 CONTAINER> bkeys(keys.begin(), keys.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);
            bolt::BENCH_BEND::device_vector<int> bMap(Map.begin(),Map.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);

            executeFunctionType( ctrl, binput1, binput2, binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,bkeys,bMap);
#else
            thrust::device_vector<vec4> binput1(input1.begin(), input1.end() );
            thrust::device_vector<vec4> binput2(input2.begin(), input2.end() );
            thrust::device_vector<vec4> binput3(input3.begin(), input3.end() );
            thrust::device_vector<vec4> boutput(output.begin(), output.end() );
            thrust::device_vector<vec4> boutput_merge(output_merge.begin(), output_merge.end() );
            thrust::device_vector<vec4> bkeys(keys.begin(), keys.end());
            thrust::device_vector<int> bMap(Map.begin(),Map.end());

            executeFunctionType(  binput1, binput2, binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,bkeys,bMap);
#endif
        }
    }
    
    else if (vecType == t_vec8)
    {
        vec8gen     generator;
        vec8square  unaryFunct;
        vec8plus    binaryFunct;
        vec8equal   binaryPredEq;
        vec8less    binaryPredLt;
        vec8mult    binaryFunctMult;
        siz = sizeof(vec8);
        std::vector<int> Map(length);  
        std::vector<vec8> input1(length, v8init);
        std::vector<vec8> input2(length, v8init);
        std::vector<vec8> input3(length, v8init);
        std::vector<vec8> output(length, v8iden);
        std::vector<vec8> output_merge(length*2, v8iden);
#if ((BOLT_BENCHMARK == 1)&&(BENCHMARK_CL_AMP ==  CL_BENCH))
        BOLT_ADD_DEPENDENCY(vec8, Bolt_DATA_TYPE);
#endif
        std::generate(input1.begin(), input1.end(),RandomNumber);
        std::generate(input2.begin(), input2.end(),RandomNumber);
        std::generate(input3.begin(), input3.end(),RandomNumber);
        std::generate(output.begin(), output.end(),RandomNumber);
        std::generate(output_merge.begin(), output_merge.end(),RandomNumber);
        for (size_t i = 0; i < input1.size(); i++)
        {
             Map[i] = (int)i;
        }
        std::vector<vec8> keys(length,v8iden ); /* Keys: 1 2 2 3 3 3 4 4 4 4 5 5  5 5 5 6 6 ..... */
        int len = (int)input1.size();
        keysGeneration(keys,len);

        if (hostMemory) {
#if (BOLT_BENCHMARK == 1)
            executeFunctionType( ctrl, input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,keys,Map);
#else
            executeFunctionType( input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,keys,Map);
#endif
        }
        else
        {
#if (BOLT_BENCHMARK == 1)
            bolt::BENCH_BEND::device_vector<vec8 CONTAINER> binput1(input1.begin(), input1.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec8 CONTAINER> binput2(input2.begin(), input2.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec8 CONTAINER> binput3(input3.begin(), input3.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec8 CONTAINER> boutput(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);
            bolt::BENCH_BEND::device_vector<vec8 CONTAINER> boutput_merge(output_merge.begin(), output_merge.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS   ctrl);					
            bolt::BENCH_BEND::device_vector<vec8 CONTAINER> bkeys(keys.begin(), keys.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);
            bolt::BENCH_BEND::device_vector<int> bMap(Map.begin(),Map.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS  ctrl);

            executeFunctionType( ctrl, binput1, binput2, binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,bkeys,bMap);
#else
            thrust::device_vector<vec8> binput1(input1.begin(), input1.end() );
            thrust::device_vector<vec8> binput2(input2.begin(), input2.end() );
            thrust::device_vector<vec8> binput3(input3.begin(), input3.end() );
            thrust::device_vector<vec8> boutput(output.begin(), output.end() );
            thrust::device_vector<vec8> boutput_merge(output_merge.begin(), output_merge.end() );
            thrust::device_vector<vec8> bkeys(keys.begin(), keys.end());
            thrust::device_vector<int> bMap(Map.begin(),Map.end());

            executeFunctionType( binput1, binput2, binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct,binaryFunctMult, binaryPredEq, binaryPredLt, routine, iterations,siz,bkeys,bMap);
#endif
        }
    }

        else
        {
            std::cerr << "Unsupported vecType=" << vecType << std::endl;
        }

}

void PrintHelp()
{
    cout<<"OpenCL command line options: \n";
    cout<<"-h [ --help ]"<<"\n\t\t\t\tproduces this help message\n";
#if (BOLT_BENCHMARK == 1)
cout<<"-v [ --version ]"<<"\n\t\t\t\tPrint queryable version information from the Bolt CL library\n";
    cout<<"-q [ --queryOpenCL ]"<<"\n\t\t\t\tPrint queryable platform and device info and return\n";
    cout<<"-g [ --gpu ]"<<"\n\t\t\t\tReport only OpenCL GPU devices\n";
    cout<<"-c [ --cpu ]"<<"\n\t\t\t\tReport only OpenCL CPU devices\n";
    cout<<"-a [ --all ]"<<"\n\t\t\t\tReport all OpenCL devices\n";
    cout<<"-m [ --runMode] arg"<<"\n\t\t\t\tRun Mode: 0-Auto, 1-SerialCPU, 2-MultiCoreCPU, 3-GPU\n";
#endif
    cout<<"-q [ --queryCuda ]"<<"\n\t\t\t\tPrint queryable platform and device info and return\n";
    cout<<"-D [ --deviceMemory ]"<<"\n\t\t\t\tAllocate vectors in device memory; default is host memory\n";
    cout<<"-p [ --platform ]  arg"<<"\n\t\t\t\tSpecify the platform under test using the index reported by -q flag\n";
    cout<<"-d [ --device ] arg"<<"\n\t\t\t\tSpecify the device under test using the index reported by the -q flag. Index is relative with respect to -g, -c or -a flags\n";
    cout<<"-l [ --length ] arg"<<"\n\t\t\t\tLength of scan array\n";
    cout<<"-i [ --iterations ] arg"<<"\n\t\t\t\tNumber of samples in timing loop\n";
    cout<<"-t [ --vecType ] arg"<<"\n\t\t\t\tData Type to use: 0-(1 value), 1-(2 values),2-(4 values), 3-(8 values)\n";
    cout<<"-f [ --function] arg"<<"\n\t\t\t\tFunction or routine called for Benchmark\n";
    cout<<"--filename arg"<<"\n\t\t\t\tName of output file\n";
    cout<<"--throw-away arg"<<"\n\t\t\t\tNumber of trials to skip averaging\n";
}

/******************************************************************************
 *
 *  Main
 *
 *****************************************************************************/
int main( int argc, char* argv[] )
{
#if (BOLT_BENCHMARK == 1)
#if BENCHMARK_CL_AMP == CL_BENCH
    cl_int err = CL_SUCCESS;
    cl_uint userPlatform    = 0;
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
#endif
    unsigned int userDevice      = 0;
#endif
    size_t iterations       = 100;
    size_t length           = 1024;
    size_t vecType          = 0;
    size_t runMode          = 0;
    size_t routine          = f_binarytransform;
    size_t numThrowAway     = 0;
    std::string function_called=functionNames[routine] ;
    std::string filename    = "bench.xml";
    bool defaultDevice      = true;
    bool print_clInfo       = false;
    bool hostMemory         = true;
    /******************************************************************************
     * Parse Command-line Parameters
     ******************************************************************************/
    try
    {
        if(argc <= 1)
        {
            PrintHelp();
            return 0;
        }
        else
        {
            for(int loop = 1; loop<argc; loop++)
            {
                //std::cout<<argv[loop]<<"\n";
                switch(ValidateBenchmarkKey(argv[loop],benchmark_options,BENCHMARK_OPTIONS_SIZE))
                {
#if (BOLT_BENCHMARK == 1)
                   case bm_version:
                   case bm_VERSION:
                       {
                        unsigned int libMajor, libMinor, libPatch;
                        bolt::BENCH_BEND::getVersion( libMajor, libMinor, libPatch );
                        const int indent = countOf( "Bolt version: " );
                        bolt::tout << std::left << std::setw( indent ) << _T( "Bolt version: " )
                            << libMajor << _T( "." )
                            << libMinor << _T( "." )
                            << libPatch << std::endl;
                    }
                       break;
                   case bm_queryOpenCL:
                   case bm_QUERYOPENCL:
                       {
                           print_clInfo = true;
                       }
                       break;
#if BENCHMARK_CL_AMP == CL_BENCH
                   case bm_gpu:
                   case bm_GPU:
                       {
                           deviceType	= CL_DEVICE_TYPE_GPU;
                       }
                       break;
                   case bm_cpu:
                   case bm_CPU:
                       {
                           deviceType	= CL_DEVICE_TYPE_CPU;
                       }
                       break;
                   case bm_all:
                   case bm_ALL:
                       {
                           deviceType	= CL_DEVICE_TYPE_ALL;
                       }
                       break;
                   case bm_platform:
                   case bm_PLATFORM:
                       {
                           if((loop+1)<argc)
                           {
                               userPlatform = atoi(argv[loop+1]);
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "-p [ --platform ]   option requires one integer argument." << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
#endif
                   case bm_device:
                   case bm_DEVICE:
                       {
                           if((loop+1)<argc)
                           {
                               userDevice = atoi(argv[loop+1]);
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "-d [ --device ]   option requires one integer argument\
                                            (index reported by the -q flag.)" << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
                   case bm_runMode:
                   case bm_RUNMODE:
                       {
                           if((loop+1)<argc)
                           {
                               runMode = atoi(argv[loop+1]);
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "-m [ --runMode ]   option requires one integer argument(Run Mode.)" << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
#endif
                   case bm_help:
                   case bm_HELP:
                       {
                           PrintHelp();
                           return 0;
                       }
                       break;
                   case bm_deviceMemory:
                   case bm_DEVICEMEMORY:
                       {
                           hostMemory = false;
                       }
                       break;
                       /*Caveat:
                       1) If you provide float or double value after -l[--lenght] command line option,
                           value typecast to integer.
                       2) If you provide some random string after -l[--lenght] command line option,
                           value typecast to 0*/
                   case bm_length:
                   case bm_LENGTH:
                       {
                           if((loop+1)<argc)
                           {
                               length = atoi(argv[loop+1]);
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "-l [ --length ]   option requires one integer argument\
                                            (length of array.)" << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
                       /*Caveat:
                       1) If you provide float or double value after -i[--iterations] command line option,
                           value typecast to integer.
                       2) If you provide some random string after -i[--iterations] command line option,
                           value typecast to 0*/
                   case bm_iterations:
                   case bm_ITERATIONS:
                       {

                           if((loop+1)<argc)
                           {
                               iterations = atoi(argv[loop+1]);
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "-i [ --iterations ]   option requires one integer argument(number of iterations.)" << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
                   case bm_vecType:
                   case bm_VECTYPE:
                       {
                           if((loop+1)<argc)
                           {
                               vecType = atoi(argv[loop+1]);
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "-t [ --vecType ]   option requires one integer argument(Data Type to use.)" << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
                   case bm_function:
                   case bm_FUNCTION:
                       {
                           if((loop+1)<argc)
                           {
                               function_called = argv[loop+1];
                               routine = get_functionindex(function_called);
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "-f [ --function ]   option requires one string argument\
                                            (Function or routine called for Benchmark.)" << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
                   case bm_filename:
                       {
                           if((loop+1)<argc)
                           {
                               filename = argv[loop+1];
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "[ --filename ]   option requires one string argument\
                                            (Name of output file.)" << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
                   case bm_throwaway:
                       {
                           /*This throwaway is not used in Python script*/
                           if((loop+1)<argc)
                           {
                               numThrowAway = atoi(argv[loop+1]);
                               loop = loop + 1;
                           }
                           else
                           {
                               std::cerr << "[ --throw-away ]   option requires one integer argument\
                                            ( Number of trials to skip averaging.)" << std::endl;
                               PrintHelp();
                               return 1;
                           }
                       }
                       break;
                   default:
                       {
                       }
                 }
             }
        }
    }
    catch( std::exception& e )
    {
        std::cout << _T( "Scan Benchmark error condition reported:" ) << std::endl << e.what() << std::endl;
        return 1;
    }
    /******************************************************************************
    * Initialize platforms and devices
    ******************************************************************************/
#if defined(_WIN32)
    //aProfiler.throwAway( numThrowAway );
#endif
#if (BOLT_BENCHMARK == 1)
    bolt::BENCH_BEND::control ctrl = bolt::BENCH_BEND::control::getDefault();

    std::string strDeviceName;
    if (runMode == 1) // serial cpu
    {
        ctrl.setForceRunMode( bolt::BENCH_BEND::control::SerialCpu );
        strDeviceName = "Serial CPU";
    }
    else if (runMode == 2) // multicore cpu
    {
        ctrl.setForceRunMode( bolt::BENCH_BEND::control::MultiCoreCpu );
        strDeviceName = "MultiCore CPU";
    }
    else // gpu || automatic (RunMode == 0)
    {
#if BENCHMARK_CL_AMP == CL_BENCH

        if (runMode == 3) // GPU
        {
            ctrl.setForceRunMode( bolt::BENCH_BEND::control::OpenCL );
            strDeviceName = "GPU";
        }

        // Platform vector contains all available platforms on system
        std::vector< ::BENCH_BEND::Platform > platforms;
        bolt::BENCH_BEND::V_OPENCL( ::BENCH_BEND::Platform::get( &platforms ), "Platform::get() failed" );
        if( print_clInfo )
        {
            bolt::BENCH_BEND::control::printPlatforms(true,deviceType);
           //std::for_each( platforms.begin( ), platforms.end( ), printPlatformFunctor( 0 ) );
            return 0;
        }
        // Device info
        ::BENCH_BEND::Context myContext = bolt::BENCH_BEND::control::getDefault( ).getContext( );
        std::vector< BENCH_BEND::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
        //::BENCH_BEND::CommandQueue myQueue( myContext, devices.at( userDevice ) , CL_QUEUE_PROFILING_ENABLE);
        ::BENCH_BEND::CommandQueue myQueue( myContext, devices.at( userDevice ));
        //  Now that the device we want is selected and we have created our own BENCH_BEND::CommandQueue, set it as the
        //  default BENCH_BEND::CommandQueue for the Bolt API
        ctrl.setCommandQueue( myQueue );
        strDeviceName = ctrl.getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
        bolt::BENCH_BEND::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );
#elif BENCHMARK_CL_AMP == AMP_BENCH
        if (runMode == 3) // GPU
        {
            ctrl.setForceRunMode( bolt::BENCH_BEND::control::Gpu );
            strDeviceName = "GPU";
        }
#endif
    }
#endif
    //std::cout << "Device: " << strDeviceName << std::endl;
    /******************************************************************************
     * Select then Execute Function
     ******************************************************************************/
#if (BOLT_BENCHMARK == 1)
        executeFunction(
        ctrl,
        vecType,
        hostMemory,
        length,
        routine,
        iterations + numThrowAway
        );
#else
    executeFunction(
        vecType,
        hostMemory,
        length,
        routine,
        iterations + numThrowAway
        );
#endif
    /******************************************************************************
     * Print Results
     ******************************************************************************/
#if defined(_WIN32)
    //aProfiler.end();
#endif
    std::ofstream outFile( filename.c_str() );
#if defined(_WIN32)
    //aProfiler.writeSum( outFile );
#endif
    outFile.close();
    return 0;
}
