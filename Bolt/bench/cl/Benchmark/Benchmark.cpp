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
#include <iostream>
#include "bolt/cl/functional.h"
#include "bolt/cl/device_vector.h"
#include "bolt/cl/generate.h"
#include "bolt/cl/binary_search.h"
#include "bolt/cl/copy.h"
#include "bolt/cl/count.h"
#include "bolt/cl/fill.h"
#include "bolt/cl/max_element.h"
#include "bolt/cl/min_element.h"
#include "bolt/cl/merge.h"
#include "bolt/cl/transform.h"
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

#include <fstream>
#include <vector>
#include <bolt/unicode.h>
#include <algorithm>
#include <iomanip> 
#include "bolt/unicode.h"
#include "bolt/countof.h"
#include <boost/program_options.hpp>
#include<random>


//#define BOLT_PROFILER_ENABLED
#define BOLT_BENCH_DEVICE_VECTOR_FLAGS CL_MEM_READ_WRITE

#include "bolt/AsyncProfiler.h"
#include "bolt/statisticalTimer.h"

#if defined(_WIN32)
AsyncProfiler aProfiler("default");
#endif
const std::streamsize colWidth = 26;


//#define DATA_TYPE float
//BOLT_CREATE_DEFINE(Bolt_DATA_TYPE,DATA_TYPE,float);

#ifndef DATA_TYPE  
#define DATA_TYPE unsigned int
BOLT_CREATE_DEFINE(Bolt_DATA_TYPE,DATA_TYPE,unsigned int);
#endif // !DATA_TYPE  

// function generator:
unsigned int RandomNumber() 
{
    std::default_random_engine gen;
    std::uniform_int_distribution<unsigned int> distr(10,1<<31);
    unsigned int dice_roll = distr(gen);  // generates number in the range 10..1<<31 
    return (dice_roll); 
}

/******************************************************************************
 *  Functions Enumerated
 *****************************************************************************/

static const size_t FList = 23;

enum functionType {
    f_binarytransform,
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
    f_scatter

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

namespace po = boost::program_options;
using namespace std;
/******************************************************************************
 *  User Defined Data Types - vec2,4,8
 *****************************************************************************/

BOLT_FUNCTOR(vec2,
struct vec2
{
    DATA_TYPE a, b;
    vec2  operator =(const DATA_TYPE inp)
    {
        vec2 tmp;
        a = b = tmp.a = tmp.b = inp;
        return tmp;
    }
    bool operator==(const vec2& rhs) const
    {
        bool l_equal = true;
        l_equal = ( a == rhs.a ) ? l_equal : false;
        l_equal = ( b == rhs.b ) ? l_equal : false;
        return l_equal;
    }
  //friend ostream& operator<<(ostream& os, const vec2& dt);
};
);

  ostream& operator<<(ostream& os, const vec2& dt)
    {
        os<<dt.a<<" "<<dt.b;
        return os;
    }
  
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< vec2 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< vec2 >::iterator, bolt::cl::deviceVectorIteratorTemplate );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, DATA_TYPE, vec2 );

BOLT_FUNCTOR(vec4,
struct vec4
{
    DATA_TYPE a, b, c, d;
    vec4  operator =(const DATA_TYPE inp)
    {
        vec4 tmp;
        tmp.a = tmp.b = tmp.c = tmp.d = a = b = c=d=inp;
        return tmp;
    }
    bool operator==(const vec4& rhs) const
    {
        bool l_equal = true;
        l_equal = ( a == rhs.a ) ? l_equal : false;
        l_equal = ( b == rhs.b ) ? l_equal : false;
        l_equal = ( c == rhs.c ) ? l_equal : false;
        l_equal = ( d == rhs.d ) ? l_equal : false;
        return l_equal;
    }
   // friend ostream& operator<<(ostream& os, const vec4& dt);
};
);


    ostream& operator<<(ostream& os, const vec4& dt)
    {
        os<<dt.a<<" "<<dt.b<<" "<<dt.c<<" "<<dt.d;
        return os;
    }

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< vec4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< vec4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, DATA_TYPE, vec4 );

BOLT_FUNCTOR(vec8,
struct vec8
{
    DATA_TYPE a, b, c, d, e, f, g, h;

    vec8  operator =(const DATA_TYPE inp)
    {
        a = b = c=d=e=f=g=h=inp;

        vec8 tmp;
        tmp.a = tmp.b = tmp.c = tmp.d = a = b = c=d=e=f=g=h=inp;
        tmp.e = tmp.f = tmp.g = tmp.h = inp;
        return tmp;

    }

    bool operator==(const vec8& rhs) const
    {
        bool l_equal = true;
        l_equal = ( a == rhs.a ) ? l_equal : false;
        l_equal = ( b == rhs.b ) ? l_equal : false;
        l_equal = ( c == rhs.c ) ? l_equal : false;
        l_equal = ( d == rhs.d ) ? l_equal : false;
        l_equal = ( e == rhs.e ) ? l_equal : false;
        l_equal = ( f == rhs.f ) ? l_equal : false;
        l_equal = ( g == rhs.g ) ? l_equal : false;
        l_equal = ( h == rhs.h ) ? l_equal : false;
        return l_equal;
    }
   // friend ostream& operator<<(ostream& os, const vec8& dt);


};
);
    ostream& operator<<(ostream& os, const vec8& dt)
    {
        os<<dt.a<<" "<<dt.b<<" "<<dt.c<<" "<<dt.d<<" "<<dt.e<<" "<<dt.f<<" "<<dt.g<<" "<<dt.h;
        return os;
    }


BOLT_CREATE_TYPENAME( bolt::cl::device_vector< vec8 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< vec8 >::iterator, bolt::cl::deviceVectorIteratorTemplate );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, DATA_TYPE, vec8 );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, DATA_TYPE, vec8 );
/******************************************************************************
 *  User Defined Binary Functions - vec2,4,8plus
 *****************************************************************************/

BOLT_FUNCTOR(vec2plus,
struct vec2plus
{
    vec2 operator()(const vec2 &lhs, const vec2 &rhs) const
    {
        vec2 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        return l_result;
    };
}; 
);

BOLT_FUNCTOR(vec4plus,
struct vec4plus
{
    vec4 operator()(const vec4 &lhs, const vec4 &rhs) const
    {
        vec4 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        l_result.c = lhs.c+rhs.c;
        l_result.d = lhs.d+rhs.d;
        return l_result;
    };
}; 
);

BOLT_FUNCTOR(vec8plus,
struct vec8plus
{
    vec8 operator()(const vec8 &lhs, const vec8 &rhs) const
    {
        vec8 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        l_result.c = lhs.c+rhs.c;
        l_result.d = lhs.d+rhs.d;
        l_result.e = lhs.e+rhs.e;
        l_result.f = lhs.f+rhs.f;
        l_result.g = lhs.g+rhs.g;
        l_result.h = lhs.h+rhs.h;
        return l_result;
    };
}; 
);


/******************************************************************************
 *  User Defined Unary Functions vec2,4,8square
 *****************************************************************************/

BOLT_FUNCTOR(vec2square,
struct vec2square
{
    vec2 operator()(const vec2 &rhs) const
    {
        vec2 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        return l_result;
    };
}; 
);

BOLT_FUNCTOR(vec4square,
struct vec4square
{
    vec4 operator()(const vec4 &rhs) const
    {
        vec4 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        l_result.c = rhs.c*rhs.c;
        l_result.d = rhs.d*rhs.d;
        return l_result;
    };
}; 
);

BOLT_FUNCTOR(vec8square,
struct vec8square
{
    vec8 operator()(const vec8 &rhs) const
    {
        vec8 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        l_result.c = rhs.c*rhs.c;
        l_result.d = rhs.d*rhs.d;
        l_result.e = rhs.e*rhs.e;
        l_result.f = rhs.f*rhs.f;
        l_result.g = rhs.g*rhs.g;
        l_result.h = rhs.h*rhs.h;
        return l_result;
    };
}; 
);

/******************************************************************************
 *  User Defined Binary Predicates equal
 *****************************************************************************/

BOLT_FUNCTOR(vec2equal,
struct vec2equal
{
    bool operator()(const vec2 &lhs, const vec2 &rhs) const
    {
        return lhs == rhs;
    };
}; 
);

BOLT_FUNCTOR(vec4equal,
struct vec4equal
{
    bool operator()(const vec4 &lhs, const vec4 &rhs) const
    {
        return lhs == rhs;
    };
}; 
);

BOLT_FUNCTOR(vec8equal,
struct vec8equal
{
    bool operator()(const vec8 &lhs, const vec8 &rhs) const
    {
        return lhs == rhs;
    };
}; 
);

/******************************************************************************
 *  User Defined Binary Predicates less than
 *****************************************************************************/

BOLT_FUNCTOR(vec2less,
struct vec2less
{
    bool operator()(const vec2 &lhs, const vec2 &rhs) const
    {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        return false;
    };
}; 
);

BOLT_FUNCTOR(vec4less,
struct vec4less
{
    bool operator()(const vec4 &lhs, const vec4 &rhs) const
    {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        if (lhs.c < rhs.c) return true;
        if (lhs.d < rhs.d) return true;
        return false;
    };
}; 
);

BOLT_FUNCTOR(vec8less,
struct vec8less
{
    bool operator()(const vec8 &lhs, const vec8 &rhs) const
    {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        if (lhs.c < rhs.c) return true;
        if (lhs.d < rhs.d) return true;
        if (lhs.e < rhs.e) return true;
        if (lhs.f < rhs.f) return true;
        if (lhs.g < rhs.g) return true;
        if (lhs.h < rhs.h) return true;
        return false;
    };
}; 
);

/******************************************************************************
 *  User Defined Binary Predicates vec2,4,8square
 *****************************************************************************/

BOLT_FUNCTOR(intgen,
struct intgen
{
    DATA_TYPE operator()() const
    {
        DATA_TYPE v = 1;
        return v;
    };
}; 
);

BOLT_FUNCTOR(vec2gen,
struct vec2gen
{
    vec2 operator()() const
    {
        vec2 v = { 2, 3 };
        return v;
    };
}; 
);

BOLT_FUNCTOR(vec4gen,
struct vec4gen
{
    vec4 operator()() const
    {
        vec4 v = { 4, 5, 6, 7 };
        return v;
    };
}; 
);

BOLT_FUNCTOR(vec8gen,
struct vec8gen
{
    vec8 operator()() const
    {
        vec8 v = { 8, 9, 10, 11, 12, 13, 14, 15 };
        return v;
    };
}; 
);

/******************************************************************************
 *  Initializers
 *****************************************************************************/
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
    std::cout<< "Specified Function not listed for Benchmar. exiting";
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
    typename BinaryPredEq,
    typename BinaryPredLt >
void executeFunctionType(
    bolt::cl::control& ctrl,
    VectorType &input1,
    VectorType &input2,
    VectorType &input3,
    VectorType &output,
    VectorType &output_merge,
    Generator generator,
    UnaryFunction unaryFunct,
    BinaryFunction binaryFunct,
    BinaryPredEq binaryPredEq,
    BinaryPredLt binaryPredLt,
    size_t function,
    size_t iterations,
    size_t siz
    )
{

    bolt::statTimer& myTimer = bolt::statTimer::getInstance( );
    myTimer.Reserve( 1, iterations );
    size_t testId	= myTimer.getUniqueID( _T( "test" ), 0 );
    
switch(function)
{

            case f_merge: 
            {

            std::cout <<  functionNames[f_merge] << std::endl;

            bolt::cl::sort( ctrl, input1.begin( ), input1.end( ), binaryPredLt);
            bolt::cl::sort( ctrl, input2.begin( ), input2.end( ), binaryPredLt);

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
              
                    myTimer.Start( testId );
                    bolt::cl::merge( ctrl,input1.begin( ),input1.end( ),input2.begin( ),input2.end( ),output_merge.begin( ),binaryPredLt); 

                    myTimer.Stop( testId );
                }
            } 
            break; 

            case f_binarysearch: 
            {

            bool tmp;
  
            typename VectorType::value_type val;

            std::cout <<  functionNames[f_binarysearch] << std::endl;

            bolt::cl::sort( ctrl, input1.begin( ), input1.end( ), binaryPredLt);
            
                for (size_t iter = 0; iter < iterations+1; iter++)
                {
               
                    int index = 0;
                    if(iter!=0)
                        index = rand()%iter;

                    val = input1[index];

                    myTimer.Start( testId );
                    tmp = bolt::cl::binary_search( ctrl,input1.begin( ),input1.end( ),val,binaryPredLt); 

                    myTimer.Stop( testId );
                }
            } 
            break;

             case f_gather: 
            {            

                std::cout <<  functionNames[f_gather] << std::endl;
                bolt::cl::device_vector<DATA_TYPE> Map(input1.size());
                for( int i=0; i < input1.size() ; i++ )
                   {
                        Map[i] = i;
                   }
                for (size_t iter = 0; iter < iterations+1; iter++)
                    {
                        myTimer.Start( testId );
                        bolt::cl::gather( ctrl,Map.begin( ), Map.end( ),input1.begin( ),output.begin()); 
                        myTimer.Stop( testId );
                    }
            } 
            break;

             case f_scatter: 
            {            

                std::cout <<  functionNames[f_scatter] << std::endl;
                bolt::cl::device_vector<DATA_TYPE> Map(input1.size());
                for( int i=0; i < input1.size() ; i++ )
                   {
                        Map[i] = i;
                   }
                for (size_t iter = 0; iter < iterations+1; iter++)
                    {
                        myTimer.Start( testId );
                        bolt::cl::scatter( ctrl, input1.begin( ),input1.end( ), Map.begin(), output.begin()); 
                        myTimer.Stop( testId );
                    }
            } 
            break;


            case f_transformreduce: // fill
            {

            typename VectorType::value_type tmp;
            tmp=0;
            std::cout <<  functionNames[f_transformreduce] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    tmp = bolt::cl::transform_reduce( ctrl,input1.begin( ), input1.end( ),
                                                       unaryFunct, tmp,
                                                       binaryFunct);
                    myTimer.Stop( testId );
                }
            }
            break;

            case f_stablesort: // fill
            {
            std::cout <<  functionNames[f_stablesort] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::stable_sort(ctrl, input1.begin(), input1.end(),binaryPredLt); 
                    myTimer.Stop( testId );
                }
            }

             break;
            case f_stablesortbykey: // fill
            {
            std::cout <<  functionNames[f_stablesortbykey] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::stable_sort_by_key(ctrl, input1.begin(), input1.end(),input2.begin(),binaryPredLt); 
                    myTimer.Stop( testId );
                }
            }

             break;


            case f_reducebykey: // fill
            {
            std::cout <<  functionNames[f_reducebykey] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                   bolt::cl::reduce_by_key(ctrl, input1.begin(), input1.end(),input2.begin(),input3.begin(),
                       output.begin(),binaryPredEq, binaryFunct);
                    myTimer.Stop( testId );
                }
            }
             break;



            case f_sort: // fill
            {
            std::cout <<  functionNames[f_sort] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                     bolt::cl::sort(ctrl, input1.begin(), input1.end(),binaryPredLt);                    
                    myTimer.Stop( testId );
                }
            }

             break;

            


            case f_sortbykey: // fill
            {
            std::cout <<  functionNames[f_sortbykey] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::sort_by_key( input1.begin(), input1.end(), input2.begin( ),binaryPredLt );
                    myTimer.Stop( testId );
                }
            }

             break;

            case f_reduce: // fill
            {
            typename VectorType::value_type tmp;
            tmp=0;
            std::cout <<  functionNames[f_reduce] << std::endl;

          /*  for(VectorType::iterator itr = input1.begin();   itr  !=input1.end(); itr++)
            {
                std::cout<<*itr<<std::endl;
            }*/
                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    tmp = bolt::cl::reduce(ctrl, input1.begin(), input1.end(),tmp,binaryFunct);                    
                    myTimer.Stop( testId );
                }
            }

             break;

        case f_maxelement: // fill
            {

            std::cout <<  functionNames[f_maxelement] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    typename VectorType::iterator itr = bolt::cl::max_element(ctrl, input1.begin(), input1.end(),binaryPredLt);                    
                    myTimer.Stop( testId );
                }
            }

             break;

        case f_minelement: // fill
            {
            
            std::cout <<  functionNames[f_minelement] << std::endl;

/*                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    typename VectorType::iterator itr = bolt::cl::min_element(ctrl, input1.begin(), input1.end(),binaryPredLt);                    
                    myTimer.Stop( testId );
                }
*/
            }

             break;


        case f_fill: // fill
            {

            typename VectorType::value_type tmp;

            std::cout <<  functionNames[f_count] << std::endl;


                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::fill(ctrl, input1.begin(), input1.end(),tmp);
                    myTimer.Stop( testId );
                }
            }
            break;


        case f_count: // Count
            {
             typename VectorType::value_type tmp;
            std::cout <<  functionNames[f_count] << std::endl;


                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::count(ctrl, input1.begin(), input1.end(),tmp);
                    myTimer.Stop( testId );
                }
            }
            break;


        case f_generate: // generate
            std::cout <<  functionNames[f_generate] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::generate(ctrl, input1.begin(), input1.end(), generator );
                    myTimer.Stop( testId );
                }
            break;

        case f_copy: // copy
            std::cout <<  functionNames[f_copy] << std::endl;
                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::copy(ctrl, input1.begin(), input1.end(), output.begin() );
                    myTimer.Stop( testId );
                }
            break;

        case f_unarytransform: // unary transform
            std::cout <<  functionNames[f_unarytransform] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::transform(ctrl, input1.begin(), input1.end(), output.begin(), unaryFunct );
                    myTimer.Stop( testId );
                }
            break;

        case f_binarytransform: // binary transform
            std::cout <<  functionNames[f_binarytransform] << std::endl;
                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::transform(ctrl, input1.begin(), input1.end(), input2.begin(), output.begin(), binaryFunct );
                    myTimer.Stop( testId );
                }
            break;

        case f_scan: // scan
            std::cout <<  functionNames[f_scan] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                     myTimer.Start( testId );
            bolt::cl::inclusive_scan(
                ctrl, input1.begin(), input1.end(), output.begin(), binaryFunct );
                        myTimer.Stop( testId );
                }
            break;

        case f_transformscan: // transform_scan
            std::cout <<  functionNames[f_transformscan] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::transform_inclusive_scan(
                    ctrl, input1.begin(), input1.end(), output.begin(), unaryFunct, binaryFunct );
                    myTimer.Stop( testId );
                }
            break;

        case f_scanbykey: // scan_by_key
            std::cout <<  functionNames[f_scanbykey] << std::endl;

                for (size_t iter = 0; iter < iterations+1; iter++)
                {
                    myTimer.Start( testId );
                    bolt::cl::inclusive_scan_by_key(
                    ctrl, input1.begin(), input1.end(), input2.begin(), output.begin(), binaryPredEq, binaryFunct );
                    myTimer.Stop( testId );
                }
            break;
        
        default:
            //std::cout << "Unsupported function=" << function << std::endl;
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
void executeFunction(
    bolt::cl::control& ctrl,
    size_t vecType,
    bool hostMemory,
    size_t length,
    size_t routine,
    size_t iterations )
{
    size_t siz;
    if (vecType == t_int)
    {
        intgen                  generator;
        bolt::cl::square<DATA_TYPE>   unaryFunct;
        bolt::cl::plus<DATA_TYPE>     binaryFunct;
        bolt::cl::equal_to<DATA_TYPE> binaryPredEq;
        bolt::cl::less<DATA_TYPE>     binaryPredLt;
        siz = sizeof(DATA_TYPE);

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

        if (hostMemory) {

            executeFunctionType( ctrl, input1, input2, input3, output, output_merge, 
                generator, unaryFunct, binaryFunct, binaryPredEq, binaryPredLt, routine, iterations,siz);
        }
        else
        {
            bolt::cl::device_vector<DATA_TYPE> binput1(input1.begin(), input1.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<DATA_TYPE> binput2(input2.begin(), input2.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<DATA_TYPE> binput3(input3.begin(), input3.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<DATA_TYPE> boutput(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl); 
            bolt::cl::device_vector<DATA_TYPE> boutput_merge(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);    
            executeFunctionType( ctrl, binput1, binput2, binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct, binaryPredEq, binaryPredLt, routine, iterations,siz);
        }
    }
    else if (vecType == t_vec2)
    {
        vec2gen     generator;
        vec2square  unaryFunct;
        vec2plus    binaryFunct;
        vec2equal   binaryPredEq;
        vec2less    binaryPredLt;
        siz = sizeof(vec2);
        
        BOLT_ADD_DEPENDENCY(vec2, Bolt_DATA_TYPE);

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


        if (hostMemory) {

            executeFunctionType( ctrl, input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct, binaryPredEq, binaryPredLt, routine, iterations,siz);
        }
        else
        {
            bolt::cl::device_vector<vec2> binput1(input1.begin(), input1.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec2> binput2(input2.begin(), input2.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec2> binput3(input3.begin(), input3.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec2> boutput(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec2> boutput_merge(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl); 
            executeFunctionType( ctrl, binput1, binput2,binput3, boutput, boutput_merge,
                generator, unaryFunct, binaryFunct, binaryPredEq, binaryPredLt, routine, iterations,siz);
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

        std::vector<vec4> input1(length, v4init);
        std::vector<vec4> input2(length, v4init);
        std::vector<vec4> input3(length, v4init);
        std::vector<vec4> output(length, v4iden);
        std::vector<vec4> output_merge(length * 2, v4iden);
        BOLT_ADD_DEPENDENCY(vec4, Bolt_DATA_TYPE);
        std::generate(input1.begin(), input1.end(),RandomNumber);
        std::generate(input2.begin(), input2.end(),RandomNumber);
        std::generate(input3.begin(), input3.end(),RandomNumber);
        std::generate(output.begin(), output.end(),RandomNumber);
        std::generate(output_merge.begin(), output_merge.end(), RandomNumber);

        if (hostMemory) {

            executeFunctionType( ctrl, input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct, binaryPredEq, binaryPredLt, routine, iterations,siz);
        }
        else
        {
            bolt::cl::device_vector<vec4> binput1(input1.begin(), input1.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec4> binput2(input2.begin(), input2.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec4> binput3(input3.begin(), input3.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec4> boutput(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec4> boutput_merge(output_merge.begin(), output_merge.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            executeFunctionType( ctrl, input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct, binaryPredEq, binaryPredLt, routine, iterations,siz);
        }
    }
    else if (vecType == t_vec8)
    {
        vec8gen     generator;
        vec8square  unaryFunct;
        vec8plus    binaryFunct;
        vec8equal   binaryPredEq;
        vec8less    binaryPredLt;
       siz = sizeof(vec8);

        std::vector<vec8> input1(length, v8init);
        std::vector<vec8> input2(length, v8init);
        std::vector<vec8> input3(length, v8init);
        std::vector<vec8> output(length, v8iden);
        std::vector<vec8> output_merge(length*2, v8iden);
        BOLT_ADD_DEPENDENCY(vec8, Bolt_DATA_TYPE);
        std::generate(input1.begin(), input1.end(),RandomNumber);
        std::generate(input2.begin(), input2.end(),RandomNumber);
        std::generate(input3.begin(), input3.end(),RandomNumber);
        std::generate(output.begin(), output.end(),RandomNumber);
        std::generate(output_merge.begin(), output_merge.end(),RandomNumber);

        if (hostMemory) {

            executeFunctionType( ctrl, input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct, binaryPredEq, binaryPredLt, routine, iterations,siz);
        }
        else
        {
            bolt::cl::device_vector<vec8> binput1(input1.begin(), input1.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec8> binput2(input2.begin(), input2.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec8> binput3(input3.begin(), input3.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec8> boutput(output.begin(), output.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            bolt::cl::device_vector<vec8> boutput_merge(output_merge.begin(), output_merge.end(), BOLT_BENCH_DEVICE_VECTOR_FLAGS,  ctrl);
            executeFunctionType( ctrl, input1, input2, input3, output, output_merge,
                generator, unaryFunct, binaryFunct, binaryPredEq, binaryPredLt, routine, iterations,siz);
        }
    }
    else
    {
        std::cerr << "Unsupported vecType=" << vecType << std::endl;
    }

}


/******************************************************************************
 *
 *  Main
 *
 *****************************************************************************/
int _tmain( int argc, _TCHAR* argv[] )
{
    cl_int err = CL_SUCCESS;
    cl_uint userPlatform    = 0;
    cl_uint userDevice      = 0;
    size_t iterations       = 10;
    size_t length           = 1<<4;
    size_t vecType          = 0;
    size_t runMode          = 0;
    size_t routine          = f_scan;
    size_t numThrowAway     = 0;
     std::string function_called=functionNames[routine] ;
    std::string filename    = "bench.xml";
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    bool defaultDevice      = true;
    bool print_clInfo       = false;
    bool hostMemory         = true;

    /******************************************************************************
     * Parse Command-line Parameters
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
            ( "deviceMemory,D",   "Allocate vectors in device memory; default is host memory" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( userPlatform ),
                "Specify the platform under test using the index reported by -q flag" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( userDevice ),
                "Specify the device under test using the index reported by the -q flag.  "
                "Index is relative with respect to -g, -c or -a flags" )
            ( "length,l",       po::value< size_t >( &length )->default_value( length ),
                "Length of scan array" )
            ( "iterations,i",   po::value< size_t >( &iterations )->default_value( iterations ),
                "Number of samples in timing loop" )
            ( "vecType,t",      po::value< size_t >( &vecType )->default_value( vecType ),
                "Data Type to use: 0-(1 value), 1-(2 values), 2-(4 values), 3-(8 values)" )
            ( "runMode,m",      po::value< size_t >( &runMode )->default_value( runMode ),
                "Run Mode: 0-Auto, 1-SerialCPU, 2-MultiCoreCPU, 3-GPU" )
            ( "function,f",      po::value< std::string >( &function_called )->default_value( function_called ),
                "Number of samples in timing loop" )
            ( "filename",     po::value< std::string >( &filename )->default_value( filename ),
                "Name of output file" )
            ( "throw-away",   po::value< size_t >( &numThrowAway )->default_value( numThrowAway ),
                "Number of trials to skip averaging" )
            ;



        po::variables_map vm;

        if(argc <= 1)
        {
            std::cout << desc << std::endl;
            return 0;
        }
        po::store( po::parse_command_line( argc, argv, desc ), vm );
        po::notify( vm );

        routine = get_functionindex(function_called);


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

        if( vm.count( "deviceMemory" ) )
        {
            hostMemory = false;
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
    aProfiler.throwAway( numThrowAway );
#endif
    bolt::cl::control ctrl = bolt::cl::control::getDefault();
    
    std::string strDeviceName;
    if (runMode == 1) // serial cpu
    {
        ctrl.setForceRunMode( bolt::cl::control::SerialCpu );
        strDeviceName = "Serial CPU";
    }
    else if (runMode == 2) // multicore cpu
    {
        ctrl.setForceRunMode( bolt::cl::control::MultiCoreCpu );
        strDeviceName = "MultiCore CPU";
    }
    else // gpu || automatic
    {
        // Platform vector contains all available platforms on system
        std::vector< ::cl::Platform > platforms;
        bolt::cl::V_OPENCL( ::cl::Platform::get( &platforms ), "Platform::get() failed" );
        if( print_clInfo )
        {
            bolt::cl::control::printPlatforms(true,deviceType);
           //std::for_each( platforms.begin( ), platforms.end( ), printPlatformFunctor( 0 ) );
            return 0;
        }

        // Device info
        ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
        std::vector< cl::Device > devices = myContext.getInfo< CL_CONTEXT_DEVICES >();
        //::cl::CommandQueue myQueue( myContext, devices.at( userDevice ) , CL_QUEUE_PROFILING_ENABLE);
        ::cl::CommandQueue myQueue( myContext, devices.at( userDevice ));

        //  Now that the device we want is selected and we have created our own cl::CommandQueue, set it as the
        //  default cl::CommandQueue for the Bolt API
        ctrl.setCommandQueue( myQueue );

        strDeviceName = ctrl.getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
        bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );
    }
    std::cout << "Device: " << strDeviceName << std::endl;

    /******************************************************************************
     * Select then Execute Function
     ******************************************************************************/
    executeFunction(
        ctrl,
        vecType,
        hostMemory,
        length,
        routine,
        iterations + numThrowAway
        );

    /******************************************************************************
     * Print Results
     ******************************************************************************/
#if defined(_WIN32)
    aProfiler.end();
#endif
    std::ofstream outFile( filename.c_str() );

#if defined(_WIN32)
    aProfiler.writeSum( outFile );
#endif
    outFile.close();
    return 0;
}
