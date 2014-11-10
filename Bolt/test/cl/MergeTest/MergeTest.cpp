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

#define TEST_DOUBLE 1
#define TEST_DEVICE_VECTOR 1
#define TEST_CPU_DEVICE 0
#define GOOGLE_TEST 1
#define LARGE_SIZE 0
#if (GOOGLE_TEST == 1)

#include "common/stdafx.h"
#include "common/myocl.h"

#include <bolt/cl/merge.h>
#include <bolt/cl/functional.h>
#include <bolt/cl/iterator/constant_iterator.h>
#include <bolt/cl/iterator/counting_iterator.h>
#include <bolt/miniDump.h>

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>
#include <bolt/cl/functional.h>


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
//  This is a compare routine for naked pointers.
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
//  Fixture classes are now defined to enable googletest to process type parameterized tests

//  This class creates a C++ 'TYPE' out of a size_t value
template< size_t N >
class TypeValue
{
public:
    static const size_t value = N;
};




template <typename T>
T generateRandom()
{
    T value = rand();
    return value;

}

template< class _Type >
inline int my_binary_search( _Type value, const _Type* a, int left, int right )
{
    long low  = left;
    long high = bolt::cl::maximum<long>( left, right + 1 );
    while( low < high )
    {
        long mid = ( low + high ) / 2;
        if ( value <= a[ mid ] )	high = mid;
        else						low  = mid + 1;	
                                                    
                                                    
    }
    return high;
}

template < class Item >
inline void exchange( Item& A, Item& B )
{
    Item t = A;
    A = B;
    B = t;
}



template< class _Type >
inline void merge_dac( const _Type* t, int p1, int r1, int p2, int r2, _Type* a, int p3 )
{
    int length1 = r1 - p1 + 1;
    int length2 = r2 - p2 + 1;
    if ( length1 < length2 )
    {
        exchange(      p1,      p2 );
        exchange(      r1,      r2 );
        exchange( length1, length2 );
    }
    if ( length1 == 0 )	return;
    int q1 = ( p1 + r1 ) / 2;
    int q2 = my_binary_search( t[q1],t,p2,r2);

    int q3 = p3 + ( q1 - p1 ) + ( q2 - p2 );
    a[ q3 ] = t[ q1 ];
    merge_dac( t, p1,     q1 - 1, p2, q2 - 1, a, p3     );
    merge_dac( t, q1 + 1, r1,     q2, r2,     a, q3 + 1 );
}




TEST( Merge, IntTest)
{



#if LARGE_SIZE
  int length = 33554432; //2^25
#else
  int length = 1048576; //2^20
#endif

  std::vector<int> hVectorA( length ),
                   hVectorB( length + 10 ),
                   hVectorO( length * 2 + 10);


  for(int i=0; i < length ; i++)
  {
      hVectorA[i] = i * 2;
      hVectorB[i] = (i *2)-1;


  }

    for(int i=length; i < length+10 ; i++)
      {
             hVectorB[i] = (i *2)-1;
      }     


  std::fill( hVectorO.begin(), hVectorO.end(), 0 );

  bolt::cl::less<int> pl;

 std::vector<int> dVectorA( hVectorA.begin(), hVectorA.end() ),
                               dVectorB( hVectorB.begin(), hVectorB.end() ),
                               dVectorO( hVectorO.begin(), hVectorO.end() );

 /*
 bolt::cl::device_vector<int> dVectorA( hVectorA.begin(), hVectorA.end() ),
                               dVectorB( hVectorB.begin(), hVectorB.end() ),
                               dVectorO( hVectorO.begin(), hVectorO.end() );  */

 std::merge( hVectorA.begin(),
                  hVectorA.end(),
                  hVectorB.begin(),
                  hVectorB.end(),
                  hVectorO.begin(),
                  std::less<int>() );


    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::OpenCL);

    bolt::cl::merge( ctl,dVectorA.begin(),
    dVectorA.end(),
    dVectorB.begin(),
    dVectorB.end(),
    dVectorO.begin(), pl );

    cmpArrays(hVectorO, dVectorO);


    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::merge( ctl,dVectorA.begin(),
    dVectorA.end(),
    dVectorB.begin(),
    dVectorB.end(),
    dVectorO.begin(), pl );

    cmpArrays(hVectorO, dVectorO);

    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::merge( ctl,dVectorA.begin(),
    dVectorA.end(),
    dVectorB.begin(),
    dVectorB.end(),
    dVectorO.begin(), pl );

    cmpArrays(hVectorO, dVectorO);



}



TEST(Merge, Cl_LongTest)  
{
        // test length
        int length = 1<<8;

        std::vector<cl_long> std1_source(length);
        std::vector<cl_long> std2_source(length);
        std::vector<cl_long> std_res(length*2);
        std::vector<cl_long> bolt_res(length*2);

        // populate source vector with random ints
        for (int j = 0; j < length; j++)
        {
            std1_source[j] = (cl_long)rand();
            std2_source[j] = (cl_long)rand();
        }
    
        // perform sort
        std::sort(std1_source.begin(), std1_source.end());
        std::sort(std2_source.begin(), std2_source.end());

        std::merge(std1_source.begin(), std1_source.end(),
            std2_source.begin(), std2_source.end(),
            std_res.begin());


        bolt::cl::merge(std1_source.begin(), std1_source.end(),
            std2_source.begin(), std2_source.end(),
            bolt_res.begin());

        // GoogleTest Comparison
        cmpArrays(std_res, bolt_res);

} 




BOLT_FUNCTOR( UDD,
struct UDD
{
    int a; 
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) const {
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    } 
    bool operator < (const UDD& other) const {
        return ((a+b) < (other.a+other.b));
    }
    bool operator > (const UDD& other) const {
        return ((a+b) > (other.a+other.b));
    }
    bool operator == (const UDD& other) const {
        return ((a+b) == (other.a+other.b));
    }

    UDD operator + (const UDD &rhs) const
    {
      UDD _result;
      _result.a = a + rhs.a;
      _result.b = b + rhs.b;
      return _result;
    }

    UDD()
        : a(0),b(0) { }
    UDD(int _in)
        : a(_in), b(_in +1)  { }
};
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< UDD >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< UDD >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_FUNCTOR( UDDless,
struct UDDless
{
   bool operator() (const UDD &lhs, const UDD &rhs) const
   {

      if((lhs.a + lhs.b) < (rhs.a + rhs.b))                
     return true;
      else
          return false;
   }

};
);


TEST( MergeUDD , UDDPlusOperatorInts )
{
        int length = 1<<8;

        std::vector<UDD> std1_source(length);
        std::vector<UDD> std2_source(length);
        std::vector<UDD> std_res(length*2);
        std::vector<UDD> bolt_res(length*2);

        // populate source vector with random ints
        for (int j = 0; j < length; j++)
        {
            std1_source[j].a = rand();
            std1_source[j].b = rand();
            std2_source[j].a = rand();
            std2_source[j].b = rand();
        }
    
        // perform sort
        std::sort(std1_source.begin(), std1_source.end());
        std::sort(std2_source.begin(), std2_source.end());
      
        UDDless lessop;

        std::merge(std1_source.begin(), std1_source.end(),
            std2_source.begin(), std2_source.end(),
            std_res.begin(),lessop);


        bolt::cl::merge(std1_source.begin(), std1_source.end(),
            std2_source.begin(), std2_source.end(),
            bolt_res.begin(),lessop);

        // GoogleTest Comparison
        cmpArrays(std_res, bolt_res);

}

TEST(MergeEPR, MergeAuto388613){
    int stdVectSize1 = 10;
    int stdVectSize2 = 20;

    std::vector<int> A(stdVectSize1);
    std::vector<int> B(stdVectSize1);
    std::vector<int> stdmerge(stdVectSize2);
    std::vector<int> boltmerge(stdVectSize2);

    for (int i = 0; i < stdVectSize1; i++){
        A[i] = 10 ;
        B[i] = 20 ;
    }

    std::merge(A.begin(), A.end(), B.begin(), B.end(), stdmerge.begin());
    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::Automatic);
    bolt::cl::merge(ctl, A.begin(), A.end(), B.begin(), B.end(), boltmerge.begin());

    for(int i = 0; i < stdVectSize2; i++) {
      EXPECT_EQ(boltmerge[i],stdmerge[i]);
    }
}

TEST(sanity_merge__dev_vect_2, wi_ctrl_floats){
    int stdVectSize1 = 10;
    int stdVectSize2 = 20;
    //bolt::cl::device_vector<float> stdVect(stdVectSize);
    //bolt::cl::device_vector<float> boltVect(stdVectSize);
    //bolt::cl::device_vector<float> stdmerge(stdVectSize);
    //bolt::cl::device_vector<float> boltmerge(stdVectSize);

    std::vector<int> A(stdVectSize1);
    std::vector<int> B(stdVectSize1);
    std::vector<int> stdmerge(stdVectSize2);
    std::vector<int> boltmerge(stdVectSize2);
    
    //float myFloatValue = 1.125f;
    //int Value = 10 ;

    for (int i = 0; i < stdVectSize1; i++){
        //boltVect[i] = stdVect[i] = myFloatValue;
        A[i] = 10 ;
        B[i] = 20 ;
    }

    std::merge(A.begin(), A.end(), B.begin(), B.end(), stdmerge.begin());
    bolt::cl::merge( A.begin(), A.end(), B.begin(), B.end(), boltmerge.begin());


    for(int i = 0; i < stdVectSize2; i++) {
    EXPECT_EQ(boltmerge[i],stdmerge[i]);
    }
        
}




int main(int argc, char* argv[])
{
    //  Register our minidump generating logic
//    bolt::miniDumpSingleton::enableMiniDumps( );

    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Set the standard OpenCL wait behavior to help debugging
    //bolt::cl::control& myControl = bolt::cl::control::getDefault( );
    // myControl.waitMode( bolt::cl::control::NiceWait );
    //myControl.forceRunMode( bolt::cl::control::MultiCoreCpu );  // choose tbb

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
#else
// TransformTest.cpp : Defines the entry point for the console application.
//

#include "common/stdafx.h"
#include <bolt/cl/transform.h>
#include <bolt/cl/functional.h>

#include <iostream>
#include <algorithm>  // for testing against STL functions.

#include <boost/thread.hpp>

#include "utils.h"

extern void readFromFileTest();





// Test of a body operator which is constructed with a template argument
// Do this using the low-level macros that require manually creating the typename
// We can't use the ClCode trait because that requires a fully instantiated type, and 
// we want to pass the code for a templated myplus<T>.
std::string myplusStr = BOLT_CODE_STRING(
    template<typename T>
struct myplus  
{
    T operator()(const T &lhs, const T &rhs) const {return lhs + rhs;}
};
);
BOLT_CREATE_TYPENAME(myplus<float>);
BOLT_CREATE_TYPENAME(myplus<int>);
BOLT_CREATE_TYPENAME(myplus<double>);



void simpleTransform1(int aSize)
{
    std::string fName = __FUNCTION__ ;
    fName += ":";

    std::vector<float> A(aSize), B(aSize);
    for (int i=0; i<aSize; i++) {
        A[i] = float(i);
        B[i] = 10000.0f + (float)i;
    }


    {
        // Test1: Test case where a user creates a templatized functor "myplus<float>" and passes that
        // to customize the transformation:

        std::vector<float> Z0(aSize), Z1(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), myplus<float>());

        bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), myplus<float>(), myplusStr);
        checkResults(fName + "myplus<float>", Z0.begin(), Z0.end(), Z1.begin());
    }

    {
        //Test2:  Use a  templatized function from the provided bolt::cl functional header.  "bolt::cl::plus<float>"   
        std::vector<float> Z0(aSize), Z1(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), bolt::cl::plus<float>());

        bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), bolt::cl::plus<float>());  
        checkResults(fName + "bolt::cl::plus<float>", Z0.begin(), Z0.end(), Z1.begin());
    }


#if 0
    {
        // Test4 : try use of a simple binary function "op_sum" created by user.
        // This doesn't work - OpenCL generates an error that "op_sum isn't a type name.  Maybe need to create an
        // function signature rather than "op_sum".
        std::vector<float> Z0(aSize), Z1(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), op_sum);

        bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), op_sum, boltCode1, "op_sum");
        checkResults(fName + "Inline Binary Function", Z0.begin(), Z0.end(), Z1.begin());
    }
#endif


};




//---
// SimpleTransform2 Demonstrates:
// How to create a C++ functor object and use this to customize the transform library call.

BOLT_FUNCTOR(SaxpyFunctor,
struct SaxpyFunctor
{
    float _a;
    SaxpyFunctor(float a) : _a(a) {};

    float operator() (const float &xx, const float &yy) 
    {
        return _a * xx + yy;
    };
};
);  // end BOLT_FUNCTOR



void transformSaxpy(int aSize)
{
    std::string fName = __FUNCTION__ ;
    fName += ":";
    std::cout << fName << "(" << aSize << ")" << std::endl;

    std::vector<float> A(aSize), B(aSize), Z1(aSize), Z0(aSize);

    for (int i=0; i<aSize; i++) {
        A[i] = float(i);
        B[i] = 10000.0f + (float)i;
    }

    SaxpyFunctor sb(10.0);


    std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), sb);
    bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), sb);  

    checkResults(fName, Z0.begin(), Z0.end(), Z1.begin());

};

void transformSaxpyDeviceVector(int aSize)
{
    std::string fName = __FUNCTION__ ;
    fName += ":";
    std::cout << fName << "(" << aSize << ")" << std::endl;

    std::vector<float> A(aSize), B(aSize), Z0(aSize);
    bolt::cl::device_vector< float > dvA(aSize), dvB(aSize), dvZ(aSize);

    for (int i=0; i<aSize; i++) {
        A[i] = float(i);
        B[i] = 10000.0f + (float)i;
        dvA[i] = float(i);
        dvB[i] = 10000.0f + (float)i;
    }

    SaxpyFunctor sb(10.0);

    std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), sb);
    bolt::cl::transform(dvA.begin(), dvA.end(), dvB.begin(), dvZ.begin(), sb);

    checkResults(fName, Z0.begin(), Z0.end(), dvZ.begin());

};


void singleThreadReduction(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> *Zbolt, 
                           int aSize) 
{
    bolt::cl::transform(A.begin(), A.end(), B.begin(), Zbolt->begin(), bolt::cl::multiplies<float>());
};


void multiThreadReductions(int aSize, int iters)
{
    std::string fName = __FUNCTION__  ;
    fName += ":";

    std::vector<float> A(aSize), B(aSize);
    for (int i=0; i<aSize; i++) {
        A[i] = float(i);
        B[i] = 10000.0f + (float)i;
    }


    {
        std::vector<float> Z0(aSize), Z1(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), bolt::cl::minus<float>()); // golden answer:

        // show we only compile it once:
        for (int i=0; i<iters; i++) {
            bolt::cl::transform(A.begin(), A.end(), B.begin(), Z1.begin(), bolt::cl::minus<float>());
            checkResults(fName + "MultiIteration - bolt::cl::minus<float>", Z0.begin(), Z0.end(), Z1.begin());
        };
    }

    // Now try multiple threads:
    // FIXME - multi-thread doesn't work until we fix up the kernel to be per-thread.
    if (0) {
        static const int threadCount = 4;
        std::vector<float> Z0(aSize);
        std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), bolt::cl::multiplies<float>()); // golden answer:

        std::vector<float> ZBolt [threadCount];
        for (int i=0; i< threadCount; i++){
            ZBolt[i].resize(aSize);
        };

        boost::thread t0(singleThreadReduction, A, B, &ZBolt[0], aSize);
        boost::thread t1(singleThreadReduction,  A, B, &ZBolt[1], aSize);
        boost::thread t2(singleThreadReduction,  A, B, &ZBolt[2], aSize);
        boost::thread t3(singleThreadReduction, A, B,  &ZBolt[3], aSize);


        t0.join();
        t1.join();
        t2.join();
        t3.join();

        for (int i=0; i< threadCount; i++){
            checkResults(fName + "MultiThread", Z0.begin(), Z0.end(), ZBolt[i].begin());
        };
    }
};


//void oclTransform(int aSize)
//{
// std::vector<float> A(aSize), B(aSize);
// for (int i=0; i<aSize; i++) {
//   A[i] = float(i);
//   B[i] = 1000.0f + (float)i;
// }
// std::vector<float> Z0(aSize);
// std::transform(A.begin(), A.end(), B.begin(), Z0.begin(), bolt::cl::plus<float>()); // golden answer:              
//
// size_t bufSize = aSize * sizeof(float);
// cl::Buffer bufferA(CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, bufSize, A.data());
// //cl::Buffer bufferA(begin(A), end(A), true);
// cl::Buffer bufferB(CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, bufSize, B.data());
// cl::Buffer bufferZ(CL_MEM_WRITE_ONLY, sizeof(float) * aSize);
//
// bolt::cl::transform<float>(bufferA, bufferB, bufferZ, bolt::cl::plus<float>());
//
// float * zMapped = static_cast<float*> (cl::CommandQueue::getDefault().enqueueMapBuffer(bufferZ, true,              
//                                         CL_MAP_READ | CL_MAP_WRITE, 0/*offset*/, bufSize)); 
//
// std::string fName = __FUNCTION__ ;
// fName += ":";
//
// checkResults(fName + "oclBuffers", Z0.begin(), Z0.end(), zMapped);
//};



int _tmain(int argc, _TCHAR* argv[])
{
    simpleTransform1(256); 
    simpleTransform1(254); 
    simpleTransform1(23); 
    transformSaxpy(256);
    transformSaxpyDeviceVector(256);
    transformSaxpy(1024);
    transformSaxpyDeviceVector( 1024 );

    //multiThreadReductions(1024, 10);

    //oclTransform(1024);
    getchar();
    return 0;
}

#endif