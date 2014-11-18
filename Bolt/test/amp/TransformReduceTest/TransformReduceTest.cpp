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



#define ENABLE_GTESTS 1
#if !ENABLE_GTESTS
    #include "common/stdafx.h"
    #include <stdio.h>

    #include <numeric>
    #include <limits>
    #include <bolt/AMP/functional.h>
    #include <bolt/AMP/transform_reduce.h>
       template<typename T>
       void printCheckMessage(bool err, std::string msg, T  stlResult, T boltResult)
       {
        if (err) {
          std::cout << "*ERROR ";
        } else {
          std::cout << "PASSED ";
        }
       
        std::cout << msg << "  STL=" << stlResult << " BOLT=" << boltResult << std::endl;
       };
       
       template<typename T>
       bool checkResult(std::string msg, T  stlResult, T boltResult)
       {
        bool err =  (stlResult != boltResult);
        printCheckMessage(err, msg, stlResult, boltResult);
       
        return err;
       };
       
       
       // For comparing floating point values:
       template<typename T>
       bool checkResult(std::string msg, T  stlResult, T boltResult, double errorThresh)
       {
        bool err;
        if ((errorThresh != 0.0) && stlResult) {
          double ratio = (double)(boltResult) / (double)(stlResult) - 1.0;
          err = abs(ratio) > errorThresh;
        } else {
          // Avoid div-by-zero, check for exact match.
          err = (stlResult != boltResult);
        }
       
        printCheckMessage(err, msg, stlResult, boltResult);
        return err;
       };
       
       
       
       // Simple test case for clbolt::transform_reduce:
       // Perform a sum-of-squares
       // Demonstrates:
       //  * Use of transform_reduce function - takes two separate functors, one for transform and one for reduce.
       //  * Performs a useful operation - squares each element, then adds them together
       //  * Use of transform_reduce bolt is more efficient than two separate function calls due to fusion - 
       //        After transform is applied, the result is immediately reduced without being written to memory.
       //        Note the STL version uses two function calls - transform followed by accumulate.
       //  * Operates directly on host buffers.
       
       void sumOfSquares(int aSize)
       {
        std::vector<int> A(aSize), Z(aSize);
       
        for (int i=0; i < aSize; i++) {
          A[i] = i+1;
        };
       
       
        // For STL, perform the operation in two steps - transform then reduction:
        std::transform(A.begin(), A.end(), Z.begin(), std::negate<int>());
        int stlReduce = std::accumulate(Z.begin(), Z.end(), 0);
       
        int boltReduce = bolt::amp::transform_reduce(A.begin() ,A.end(), bolt::amp::negate<int>(),
                                                                        0, bolt::amp::plus<int>());
       
        checkResult(__FUNCTION__, stlReduce, boltReduce);
       };
       
       
       
       
       int _tmain(int argc, _TCHAR* argv[])
       {
        sumOfSquares(2000);
       
        return 0;
       }
#else
#include "common/stdafx.h"

#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/functional.h"
#include "bolt/miniDump.h"
#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>
#include "common/test_common.h"

#define TEST_CPU_DEVICE 0
#define TEST_DOUBLE 1
#define TEST_DEVICE_VECTOR 1

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process type parameterized tests

//  This class creates a C++ 'TYPE' out of a int value
template< int N >
class TypeValue
{
public:
    static const int value = N;
};
template <typename T>
T generateRandom()
{
    double value = rand();
    static bool negate = true;
    if (negate)
    {
        negate = false;
        return -(T)fmod(value, 10.0);
    }
    else
    {
        negate = true;
        return (T)fmod(value, 10.0);
    }
}
//  Test fixture class, used for the Type-parameterized tests
//  Namely, the tests that use std::array and TYPED_TEST_P macros
template< typename ArrayTuple >
class TransformArrayTest: public ::testing::Test
{
public:
    TransformArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<ArrayType>);
        stdOutput = stdInput;
        boltInput = stdInput;
        boltOutput = stdInput;
    };

    virtual void TearDown( )
    {};

    virtual ~TransformArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const int ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType, ArraySize > stdInput, boltInput, stdOutput, boltOutput;
    int m_Errors;
};

TYPED_TEST_CASE_P( TransformArrayTest );

TYPED_TEST_P( TransformArrayTest, Normal )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), [](ArrayType x){return x*x;});
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::square<ArrayType>(), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );
}
TYPED_TEST_P( TransformArrayTest, SerialNormal )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), [](ArrayType x){return x*x;});
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce(ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::square<ArrayType>(), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );
}
#if defined( ENABLE_TBB )
TYPED_TEST_P( TransformArrayTest, MulticoreNormal )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), [](ArrayType x){return x*x;});
    ArrayType stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce(ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::square<ArrayType>(), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );
}
#endif
 
TYPED_TEST_P( TransformArrayTest, GPU_DeviceNormal )
{
    Concurrency::accelerator accel(Concurrency::accelerator::default_accelerator);
    bolt::amp::control c_gpu( accel );  // construct control structure from the queue.
 
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), [](ArrayType x){return x*x;});
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce(TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::square<ArrayType>(), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );

}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceNormal )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

#if OCL_CONTEXT_BUG_WORKAROUND
    ::amp::Context myContext = bolt::amp::control::getDefault( ).context( );
    bolt::amp::control c_cpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_CPU, 0 ));  
#else
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::amp::control c_cpu(oclcpu._queue);  // construct control structure from the queue.
#endif

    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::amp::square<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce( c_cpu,TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                        bolt::amp::square<ArrayType>(), init,
                                                        bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

TYPED_TEST_P( TransformArrayTest, MultipliesFunction )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), std::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::negate<ArrayType>( ), init,
                                                       bolt::amp::plus<ArrayType>( ));

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
TYPED_TEST_P( TransformArrayTest, SerialMultipliesFunction )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    ArrayType init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), std::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce(ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::negate<ArrayType>( ), init,
                                                       bolt::amp::plus<ArrayType>( ));

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#if defined( ENABLE_TBB )
TYPED_TEST_P( TransformArrayTest, MulticoreMultipliesFunction )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 

    ArrayType init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), std::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce(ctl, TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::negate<ArrayType>( ), init,
                                                       bolt::amp::plus<ArrayType>( ));

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

TYPED_TEST_P( TransformArrayTest, GPU_DeviceMultipliesFunction )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
#if OCL_CONTEXT_BUG_WORKAROUND
    ::amp::Context myContext = bolt::amp::control::getDefault( ).context( );
    bolt::amp::control c_gpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));  
#else
    ::Concurrency::accelerator accel(::Concurrency::accelerator::default_accelerator);
    bolt::amp::control c_gpu(accel);
#endif


    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), std::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce( c_gpu,TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::negate<ArrayType>(), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceMultipliesFunction )
{
    typedef typename TransformArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont; 
#if OCL_CONTEXT_BUG_WORKAROUND
    ::amp::Context myContext = bolt::amp::control::getDefault( ).context( ); 
    bolt::amp::control c_cpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_CPU, 0 ));  
#else
    MyOclContext oclcpu = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::amp::control c_cpu(oclcpu._queue);  // construct control structure from the queue.
#endif

    ArrayType init(0);
    //  Calling the actual functions under test
    std::transform(TransformArrayTest< gtest_TypeParam_ >::stdInput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdInput.end(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), bolt::amp::negate<ArrayType>());
    ArrayType stlReduce = std::accumulate(TransformArrayTest< gtest_TypeParam_ >::stdOutput.begin(), TransformArrayTest< gtest_TypeParam_ >::stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::transform_reduce( c_cpu,TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::negate<ArrayType>(), init,
                                                       bolt::amp::plus<ArrayType>());

    typename  ArrayCont::difference_type  stdNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::stdInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename  ArrayCont::difference_type  boltNumElements = std::distance( TransformArrayTest< gtest_TypeParam_ >::boltInput.begin( ), TransformArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, TransformArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( TransformArrayTest< gtest_TypeParam_ >::stdInput, TransformArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

#if (TEST_CPU_DEVICE == 1)
REGISTER_TYPED_TEST_CASE_P( TransformArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction,
                                           CPU_DeviceNormal, CPU_DeviceMultipliesFunction);
#else
REGISTER_TYPED_TEST_CASE_P( TransformArrayTest, Normal, SerialNormal, GPU_DeviceNormal, 
                                           MultipliesFunction, SerialMultipliesFunction, GPU_DeviceMultipliesFunction );
#if defined( ENABLE_TBB )
REGISTER_TYPED_TEST_CASE_P( MulticoreNormal, MulticoreMultipliesFunction );
#endif

#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class TransformIntegerVector: public ::testing::TestWithParam< int >
{
public:

    TransformIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                               stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< int > stdInput, boltInput, stdOutput, boltOutput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformFloatVector: public ::testing::TestWithParam< int >
{
public:
    TransformFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                             stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< float > stdInput, boltInput, stdOutput, boltOutput;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleVector: public ::testing::TestWithParam< int >
{
public:
    TransformDoubleVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                              stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<double>);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< double > stdInput, boltInput, stdOutput, boltOutput;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformIntegerDeviceVector( ): stdInput( GetParam( ) ), boltInput( static_cast<int>( GetParam( ) ) ),
                                     stdOutput( GetParam( ) ), boltOutput( static_cast<int>( GetParam( ) ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    }

protected:
    std::vector< int > stdInput, stdOutput;
    bolt::amp::device_vector< int > boltInput, boltOutput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
	TransformFloatDeviceVector( ): stdInput( GetParam( ) ), boltInput( stdInput.begin(), stdInput.end() ), boltOutput( stdInput.begin(), stdInput.end() )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        stdOutput = stdInput;

        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
        }
    }

protected:
    std::vector< float > stdInput, stdOutput;
    bolt::amp::device_vector< float, concurrency::array_view > boltInput, boltOutput;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformDoubleDeviceVector( ): stdInput( GetParam( ) ), boltInput( static_cast<int>( GetParam( ) ) ),
                                                            boltOutput( static_cast<int>( GetParam( ) ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<double>);
        stdOutput = stdInput;

        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
        }
    }

protected:
    std::vector< double > stdInput, stdOutput;
    bolt::amp::device_vector< double > boltInput, boltOutput;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformIntegerNakedPointer( ): stdInput( new int[ GetParam( ) ] ), boltInput( new int[ GetParam( ) ] ), 
                                     stdOutput( new int[ GetParam( ) ] ), boltOutput( new int[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        int size = GetParam( );

        std::generate(stdInput, stdInput + size, generateRandom<int>);
		for (int i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
};

protected:
     int* stdInput;
     int* boltInput;
     int* stdOutput;
     int* boltOutput;

};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformFloatNakedPointer( ): stdInput( new float[ GetParam( ) ] ), boltInput( new float[ GetParam( ) ] ), 
                                   stdOutput( new float[ GetParam( ) ] ), boltOutput( new float[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        int size = GetParam( );

        std::generate(stdInput, stdInput + size, generateRandom<float>);
		for (int i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
    };

protected:
     float* stdInput;
     float* boltInput;
     float* stdOutput;
     float* boltOutput;
};

class transformReduceStdVectWithInit :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:	
    transformReduceStdVectWithInit():mySize(GetParam()){
    }
};

TEST_P( transformReduceStdVectWithInit, withIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int init = 10;
     //there is no std::square available
    std::transform(stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), [](int n){return n*n;} ); 
    int stlTransformReduce = std::accumulate(stdOutput.begin( ), stdOutput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::transform_reduce( boltInput.begin( ), boltInput.end(),bolt::amp::square<int>(),
                                                                                        init, bolt::amp::plus<int>());

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
TEST_P( transformReduceStdVectWithInit, SerialwithIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    //  Calling the actual functions under test
    int init = 10;
      //there is no std::square available
    std::transform(stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), [](int n){return n*n;} );
    int stlTransformReduce = std::accumulate(stdOutput.begin( ), stdOutput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                        bolt::amp::square<int>( ), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#if defined( ENABLE_TBB )
TEST_P( transformReduceStdVectWithInit, MulticorewithIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    //  Calling the actual functions under test
    int init = 10;
     //there is no std::square available
    std::transform(stdInput.begin( ), stdInput.end( ), stdOutput.begin( ), [](int n){return n*n;} ); 
    int stlTransformReduce = std::accumulate(stdOutput.begin( ), stdOutput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                        bolt::amp::square<int>( ), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#endif
TEST_P( transformReduceStdVectWithInit, withIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), [](int n){return n*n;});
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::amp::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                        bolt::amp::square<int>(), init, bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
TEST_P( transformReduceStdVectWithInit, SerialwithIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), [](int n){return n*n;});
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                            bolt::amp::square<int>(), init, bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
#if defined( ENABLE_TBB )
TEST_P( transformReduceStdVectWithInit, MulticorewithIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), [](int n){return n*n;});
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ), 
                                            bolt::amp::square<int>(), init, bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
#endif
TEST_P( transformReduceStdVectWithInit, withIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), [](int x){return x*x;});
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltTransformReduce= bolt::amp::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                        bolt::amp::square<int>(), init, bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
TEST_P( transformReduceStdVectWithInit, SerialwithIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), [](int x){return x*x;});
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltTransformReduce= bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ), 
                                            bolt::amp::square<int>(), init, bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
#if defined( ENABLE_TBB )
TEST_P( transformReduceStdVectWithInit, MulticorewithIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), [](int x){return x*x;});
    int stlTransformReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltTransformReduce= bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                            bolt::amp::square<int>(), init, bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
#endif

INSTANTIATE_TEST_CASE_P( withIntWithInitValue, transformReduceStdVectWithInit, ::testing::Range(1, 100, 10) );

class transformReduceTestMultFloat: public ::testing::TestWithParam<int>{
protected:
    int arraySize;
public:
    transformReduceTestMultFloat( ):arraySize( GetParam( ) )
    {}
};

TEST_P (transformReduceTestMultFloat, multiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
    }

#if defined (_WIN32 )
    std::transform( myArray, (float *)(myArray + arraySize), stdext::make_checked_array_iterator( myArray2,arraySize ),
        std::negate<float>( ) );
#else
    std::transform( myArray, (float *)(myArray + arraySize), myArray2, std::negate<float>( ) );
#endif

    float stlTransformReduce = std::accumulate(myArray2, myArray2 + arraySize, 1.0f, std::multiplies<float>());
    float boltTransformReduce = bolt::amp::transform_reduce(myBoltArray, myBoltArray + arraySize, 
                                                                        bolt::amp::negate<float>(),
        1.0f, bolt::amp::multiplies<float>());

    EXPECT_FLOAT_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}
TEST_P (transformReduceTestMultFloat, SerialmultiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

#if defined (_WIN32 )
    std::transform( myArray, (float *)(myArray + arraySize), stdext::make_checked_array_iterator(myArray2, arraySize), 
        std::negate<float>( ) );
#else
    std::transform( myArray, (float *)(myArray + arraySize), myArray2, std::negate<float>( ) );
#endif

    float stlTransformReduce = std::accumulate(myArray2, myArray2 + arraySize, 1.0f, std::multiplies<float>());
    float boltTransformReduce = bolt::amp::transform_reduce(ctl, myBoltArray, myBoltArray + arraySize, 
                                                                            bolt::amp::negate<float>(),
        1.0f, bolt::amp::multiplies<float>());

    EXPECT_FLOAT_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}
#if defined( ENABLE_TBB )
TEST_P (transformReduceTestMultFloat, MulticoremultiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

#if defined (_WIN32 )
    std::transform( myArray, (float *)(myArray + arraySize), stdext::make_checked_array_iterator(myArray2,arraySize), 
        std::negate<float>( ) );
#else
    std::transform( myArray, (float *)(myArray + arraySize), myArray2, std::negate<float>( ) );
#endif

    float stlTransformReduce = std::accumulate(myArray2, myArray2 + arraySize, 1.0f, std::multiplies<float>());
    float boltTransformReduce = bolt::amp::transform_reduce(ctl, myBoltArray, myBoltArray + arraySize, 
                                                                            bolt::amp::negate<float>(),
        1.0f, bolt::amp::multiplies<float>());

    EXPECT_FLOAT_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}
#endif

TEST_P( transformReduceTestMultFloat, addFloatValues )
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    
    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    std::transform(A.begin(), A.end(), B.begin(), std::negate<float>());
    float stdTransformReduceValue = std::accumulate(B.begin(), B.end(), 0.0f, std::plus<float>());
    float boltClTransformReduce = bolt::amp::transform_reduce(boltVect.begin(), boltVect.end(), 
                                    bolt::amp::negate<float>(), 0.0f, bolt::amp::plus<float>());

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdTransformReduceValue, boltClTransformReduce );
}
TEST_P( transformReduceTestMultFloat, SerialaddFloatValues)
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    
    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    std::transform(A.begin(), A.end(), B.begin(), std::negate<float>());
    float stdTransformReduceValue = std::accumulate(B.begin(), B.end(), 0.0f, std::plus<float>());
    float boltClTransformReduce = bolt::amp::transform_reduce(ctl, boltVect.begin(), boltVect.end(), 
                                        bolt::amp::negate<float>(), 0.0f, bolt::amp::plus<float>());

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdTransformReduceValue, boltClTransformReduce );
}
#if defined( ENABLE_TBB )
TEST_P( transformReduceTestMultFloat, MulticoreaddFloatValues)
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    
    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    std::transform(A.begin(), A.end(), B.begin(), std::negate<float>());
    float stdTransformReduceValue = std::accumulate(B.begin(), B.end(), 0.0f, std::plus<float>());
    float boltClTransformReduce = bolt::amp::transform_reduce(ctl, boltVect.begin(), boltVect.end(),
                                        bolt::amp::negate<float>(), 0.0f, bolt::amp::plus<float>());

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdTransformReduceValue, boltClTransformReduce );
}
#endif

INSTANTIATE_TEST_CASE_P(serialValues, transformReduceTestMultFloat, ::testing::Range(1, 100, 10));
INSTANTIATE_TEST_CASE_P(multiplyWithFloatPredicate, transformReduceTestMultFloat, ::testing::Range(1, 20, 1));
//end of new 2

class transformReduceTestMultDouble: public ::testing::TestWithParam<int>{
protected:
        int arraySize;
public:
    transformReduceTestMultDouble():arraySize(GetParam()){
    }
};
#if(TEST_DOUBLE == 1)
TEST_P (transformReduceTestMultDouble, multiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myBoltArray[i] = myArray[i];
    }

#if defined (_WIN32 )
    std::transform( myArray, myArray + arraySize, stdext::make_checked_array_iterator( myArray2, arraySize ), 
        std::negate<double>( ) );
#else
    std::transform(myArray, myArray + arraySize, myArray2, std::negate<double>());
#endif

    double stlTransformReduce = std::accumulate(myArray2, myArray2 + arraySize, 1.0, std::multiplies<double>());

    double boltTransformReduce = bolt::amp::transform_reduce(myBoltArray, myBoltArray + arraySize, 
                                bolt::amp::negate<double>(), 1.0, bolt::amp::multiplies<double>());
    
    EXPECT_DOUBLE_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}
TEST_P (transformReduceTestMultDouble, SerialmultiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myBoltArray[i] = myArray[i];
    }

#if defined (_WIN32 )
    std::transform( myArray, myArray + arraySize, stdext::make_checked_array_iterator( myArray2, arraySize ), 
        std::negate<double>( ) );
#else
    std::transform(myArray, myArray + arraySize, myArray2, std::negate<double>());
#endif
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    double stlTransformReduce = std::accumulate(myArray2, myArray2 + arraySize, 1.0, std::multiplies<double>());

    double boltTransformReduce = bolt::amp::transform_reduce(ctl, myBoltArray, myBoltArray + arraySize,
                                    bolt::amp::negate<double>(), 1.0, bolt::amp::multiplies<double>());
    
    EXPECT_DOUBLE_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}
#if defined( ENABLE_TBB )
TEST_P (transformReduceTestMultDouble, MulticoremultiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myBoltArray[i] = myArray[i];
    }

#if defined (_WIN32 )
    std::transform( myArray, myArray + arraySize, stdext::make_checked_array_iterator( myArray2, arraySize ), 
        std::negate<double>( ) );
#else
    std::transform(myArray, myArray + arraySize, myArray2, std::negate<double>());
#endif
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    double stlTransformReduce = std::accumulate(myArray2, myArray2 + arraySize, 1.0, std::multiplies<double>());

    double boltTransformReduce = bolt::amp::transform_reduce(ctl, myBoltArray, myBoltArray + arraySize,
                                    bolt::amp::negate<double>(), 1.0, bolt::amp::multiplies<double>());
    
    EXPECT_DOUBLE_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}
#endif
#endif

#if(TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( multiplyWithDoublePredicate, transformReduceTestMultDouble, ::testing::Range(1, 20, 1) );
#endif

#if (TEST_DOUBLE ==1 )
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformDoubleNakedPointer( ): stdInput( new double[ GetParam( ) ] ), boltInput( new double[ GetParam( ) ] ),
                                    stdOutput( new double[ GetParam( ) ] ), boltOutput( new double[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        int size = GetParam( );

        std::generate(stdInput, stdInput + size, generateRandom<double>);

        for( int i=0; i < size; i++ )
        {
            boltInput[ i ] = stdInput[ i ];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
    };

protected:
     double* stdInput;
     double* boltInput;
     double* stdOutput;
     double* boltOutput;
};
#endif

TEST_P( TransformIntegerVector, Normal )
{

    int init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), std::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::amp::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<int>(), init,
                                                       bolt::amp::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}
TEST_P( TransformIntegerVector, SerialNormal )
{

    int init(0);
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), std::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<int>(), init,
                                                       bolt::amp::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}
#if defined( ENABLE_TBB )
TEST_P( TransformIntegerVector, MulticoreNormal )
{

    int init(0);
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), std::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<int>(), init,
                                                       bolt::amp::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}
#endif

TEST_P( TransformFloatVector, Normal )
{
    float init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), std::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::amp::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<float>(), init,
                                                       bolt::amp::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
TEST_P( TransformFloatVector, SerialNormal )
{
    float init(0);
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), std::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<float>(), init,
                                                       bolt::amp::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( TransformFloatVector, MulticoreNormal )
{
    float init(0);
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), std::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce = bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<float>(), init,
                                                       bolt::amp::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleVector, Inplace )
{
    double init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::amp::negate<double>());
    double stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    double boltReduce = bolt::amp::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<double>(), init,
                                                       bolt::amp::plus<double>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( TransformDoubleVector, SerialInplace )
{
    double init(0);
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::amp::negate<double>());
    double stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    double boltReduce = bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<double>(), init,
                                                       bolt::amp::plus<double>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( TransformDoubleVector, MulticoreInplace )
{
    double init(0);
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::amp::negate<double>());
    double stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    double boltReduce = bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<double>(), init,
                                                       bolt::amp::plus<double>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
#endif

#if (TEST_DEVICE_VECTOR == 1)
TEST_P( TransformIntegerDeviceVector, Inplace )
{
    int init(0);
    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::amp::negate<int>());
    int stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    int boltReduce = bolt::amp::transform_reduce( boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<int>(), init,
                                                       bolt::amp::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
TEST_P( TransformFloatDeviceVector, SerialInplace )
{
    float init(0);
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::amp::negate<float>());
    float stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    float boltReduce =bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<float>(), init,
                                                       bolt::amp::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if (TEST_DOUBLE == 1)
#if defined( ENABLE_TBB )
TEST_P( TransformDoubleDeviceVector, MulticoreInplace )
{
    double init(0);
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    //  Calling the actual functions under test
    std::transform(stdInput.begin(), stdInput.end(), stdOutput.begin(), bolt::amp::negate<double>());
    double stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    double boltReduce = bolt::amp::transform_reduce(ctl, boltInput.begin( ), boltInput.end( ),
                                                       bolt::amp::negate<double>(), init,
                                                       bolt::amp::plus<double>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif
#endif
#endif

TEST_P( TransformIntegerNakedPointer, Inplace )
{
    int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
#else
    int* wrapStdInput = stdInput;
    int* wrapStdOutput = stdOutput;
    int* wrapBoltInput = boltInput;
#endif

    int init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, std::negate<int>());
    int stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    int boltReduce = bolt::amp::transform_reduce( wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<int>(), init,
                                                       bolt::amp::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
TEST_P( TransformIntegerNakedPointer, SerialInplace )
{
    int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
#else
    int* wrapStdInput = stdInput;
    int* wrapStdOutput = stdOutput;
    int* wrapBoltInput = boltInput;
#endif

    int init(0);
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, std::negate<int>());
    int stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    int boltReduce = bolt::amp::transform_reduce(ctl, wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<int>(), init,
                                                       bolt::amp::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#if defined( ENABLE_TBB )
TEST_P( TransformIntegerNakedPointer, MulticoreInplace )
{
    int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
#else
    int* wrapStdInput = stdInput;
    int* wrapStdOutput = stdOutput;
    int* wrapBoltInput = boltInput;
#endif

    int init(0);
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, std::negate<int>());
    int stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    int boltReduce = bolt::amp::transform_reduce(ctl, wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<int>(), init,
                                                       bolt::amp::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif
TEST_P( TransformFloatNakedPointer, Inplace )
{
    int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
#else
    float* wrapStdInput = stdInput;
    float* wrapStdOutput = stdOutput;
    float* wrapBoltInput = boltInput;
#endif

    float init(0);
      
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, std::negate<float>());
    float stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    float boltReduce = bolt::amp::transform_reduce(wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<float>(), init,
                                                       bolt::amp::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
TEST_P( TransformFloatNakedPointer, SerialInplace )
{
    int endIndex = GetParam( );
 
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
#else
    float* wrapStdInput = stdInput;
    float* wrapStdOutput = stdOutput;
    float* wrapBoltInput = boltInput;
#endif

    float init(0);
      
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, std::negate<float>());
    float stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    float boltReduce = bolt::amp::transform_reduce(ctl, wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<float>(), init,
                                                       bolt::amp::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#if defined( ENABLE_TBB )
TEST_P( TransformFloatNakedPointer, MulticoreInplace )
{
    int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
#else
    float* wrapStdInput = stdInput;
    float* wrapStdOutput = stdOutput;
    float* wrapBoltInput = boltInput;
#endif
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    float init(0);
      
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, std::negate<float>());
    float stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    float boltReduce = bolt::amp::transform_reduce(wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<float>(), init,
                                                       bolt::amp::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif


#if (TEST_DOUBLE == 1)
TEST_P( TransformDoubleNakedPointer, Inplace )
{
    int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< double* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
#else
    double* wrapStdInput = stdInput;
    double* wrapStdOutput = stdOutput;
    double* wrapBoltInput = boltInput;
#endif

    double init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::amp::negate<double>());
    double stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    double boltReduce = bolt::amp::transform_reduce( wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<double>(), init,
                                                       bolt::amp::plus<double>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
TEST_P( TransformDoubleNakedPointer, SerialInplace )
{
    int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< double* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
#else
    double* wrapStdInput = stdInput;
    double* wrapStdOutput = stdOutput;
    double* wrapBoltInput = boltInput;
#endif
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    double init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::amp::negate<double>());
    double stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    double boltReduce = bolt::amp::transform_reduce(ctl, wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<double>(), init,
                                                       bolt::amp::plus<double>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#if defined( ENABLE_TBB )
TEST_P( TransformDoubleNakedPointer, MulticoreInplace )
{
    int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< double* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< double* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< double* > wrapBoltInput( boltInput, endIndex );
#else
    double* wrapStdInput = stdInput;
    double* wrapStdOutput = stdOutput;
    double* wrapBoltInput = boltInput;
#endif
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 

    double init(0);
    //  Calling the actual functions under test
    std::transform(wrapStdInput, wrapStdInput + endIndex, wrapStdOutput, bolt::amp::negate<double>());
    double stlReduce = std::accumulate(wrapStdOutput,wrapStdOutput + endIndex, init);

    double boltReduce = bolt::amp::transform_reduce(ctl, wrapBoltInput, wrapBoltInput + endIndex,
                                                       bolt::amp::negate<double>(), init,
                                                       bolt::amp::plus<double>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif
#endif
std::array<int, 15> TestValues = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerVector, ::testing::Range( 0, 4096, 54 ) ); //   1 to 2^22
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                    TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16	
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( TransformValues, TransformDoubleVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                    TestValues.end() ) );
#endif
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                            TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                            TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformDoubleDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                            TestValues.end() ) );
#endif
INSTANTIATE_TEST_CASE_P( TransformRange, TransformIntegerNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(),
                                                                                            TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( TransformRange, TransformFloatNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( TransformValues, TransformFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                            TestValues.end() ) );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( TransformRange, TransformDoubleNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( Transform, TransformDoubleNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
#endif

typedef ::testing::Types< 
    std::tuple< int, TypeValue< 1 > >,
    std::tuple< int, TypeValue< 31 > >,
    std::tuple< int, TypeValue< 32 > >,
    std::tuple< int, TypeValue< 63 > >,
    std::tuple< int, TypeValue< 64 > >,
    std::tuple< int, TypeValue< 127 > >,
    std::tuple< int, TypeValue< 128 > >,
    std::tuple< int, TypeValue< 129 > >,
    std::tuple< int, TypeValue< 1000 > >,
    std::tuple< int, TypeValue< 1053 > >,
    std::tuple< int, TypeValue< 4096 > >,
    std::tuple< int, TypeValue< 4097 > >,
    std::tuple< int, TypeValue< 65535 > >,
    std::tuple< int, TypeValue< 65536 > >
> IntegerTests;

typedef ::testing::Types< 
    std::tuple< float, TypeValue< 1 > >,
    std::tuple< float, TypeValue< 31 > >,
    std::tuple< float, TypeValue< 32 > >,
    std::tuple< float, TypeValue< 63 > >,
    std::tuple< float, TypeValue< 64 > >,
    std::tuple< float, TypeValue< 127 > >,
    std::tuple< float, TypeValue< 128 > >,
    std::tuple< float, TypeValue< 129 > >,
    std::tuple< float, TypeValue< 1000 > >,
    std::tuple< float, TypeValue< 1053 > >,
    std::tuple< float, TypeValue< 4096 > >,
    std::tuple< float, TypeValue< 4097 > >,
    std::tuple< float, TypeValue< 65535 > >,
    std::tuple< float, TypeValue< 65536 > >
> FloatTests;

#if (TEST_DOUBLE == 1)
typedef ::testing::Types< 
    std::tuple< double, TypeValue< 1 > >,
    std::tuple< double, TypeValue< 31 > >,
    std::tuple< double, TypeValue< 32 > >,
    std::tuple< double, TypeValue< 63 > >,
    std::tuple< double, TypeValue< 64 > >,
    std::tuple< double, TypeValue< 127 > >,
    std::tuple< double, TypeValue< 128 > >,
    std::tuple< double, TypeValue< 129 > >,
    std::tuple< double, TypeValue< 1000 > >,
    std::tuple< double, TypeValue< 1053 > >,
    std::tuple< double, TypeValue< 4096 > >,
    std::tuple< double, TypeValue< 4097 > >,
    std::tuple< double, TypeValue< 65535 > >,
    std::tuple< double, TypeValue< 65536 > >
> DoubleTests;
#endif 

struct UDD { 
    int a; 
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) const restrict (amp,cpu){ 
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    } 
    bool operator < (const UDD& other) const restrict (amp,cpu){ 
        return ((a+b) < (other.a+other.b));
    }
    bool operator > (const UDD& other) const restrict (amp,cpu) { 
        return ((a+b) > (other.a+other.b));
    }
    bool operator == (const UDD& other) const restrict (amp,cpu) { 
        return ((a+b) == (other.a+other.b));
    }

    UDD operator + (const UDD &rhs) const restrict (amp,cpu){
                UDD tmp = *this;
                tmp.a = tmp.a + rhs.a;
                tmp.b = tmp.b + rhs.b;
                return tmp;
    }
    
    UDD() restrict (amp,cpu)
        : a(0),b(0) { } 
    UDD(int _in) restrict (amp,cpu)
        : a(_in), b(_in +1)  { } 
        
}; 


struct tbbUDD { 
    float a; 
    double b;

    tbbUDD operator + (const tbbUDD &rhs) const restrict (amp,cpu) {
                tbbUDD tmp = *this;
                tmp.a = tmp.a + rhs.a;
                tmp.b = tmp.b + rhs.b;
                return tmp;
    }
    
     bool operator() (const tbbUDD& lhs, const tbbUDD& rhs)const restrict (amp,cpu) { 
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    } 
    tbbUDD() restrict (amp,cpu)
        : a(0.f),b(0.0) { } 
    tbbUDD(int _in) restrict (amp,cpu)
        : a((float)_in), b(_in +1.0)  { } 
    bool operator == (const tbbUDD& other) const restrict (amp,cpu) { 
        return ((double)(a+b) == (double)(other.a+other.b));
    }
}; 

struct DivUDD
{
    float operator()(const UDD &rhs) const restrict (amp,cpu)
    {
        float _result = (1.f*rhs.a) / (1.f*rhs.a+rhs.b); //  a/(a+b)
        return _result;
    };
}; 

struct negateUDD
{
    UDD operator()(const UDD &rhs) const restrict (amp,cpu)
    {
       UDD temp;
       temp.a = -rhs.a;
       temp.b = -rhs.b;
       return temp;
    };
}; 
struct negatetbbUDD
{
    tbbUDD operator()(const tbbUDD &rhs) const restrict (amp,cpu)
    {
       tbbUDD temp;
       temp.a = -rhs.a;
       temp.b = -rhs.b;
       return temp;
    };
}; 
///**********************************************************
// * mixed unary operator - dtanner
// *********************************************************/
//TEST( MixedTransform, OutOfPlace )
//{
//    //int length = GetParam( );
//
//    //setup containers
//    int length = (1<<16)+23;
////    bolt::amp::negate< uddtI2 > nI2;
//    UDD initial(2);
//    //UDD identity();
//    bolt::amp::device_vector< UDD >    input( length, initial, true  );
//    bolt::amp::device_vector< float > output( length, 0.f, false );
//    std::vector< UDD > refInput( length, initial );
//    std::vector< float > refIntermediate( length, 0.f );
//    std::vector< float > refOutput(       length, 0.f );
//
//    //    T stlReduce = std::accumulate(Z.begin(), Z.end(), init);
//
//    //T boltReduce = bolt::amp::transform_reduce(A.begin(), A.end(), SquareMe<T>(), init, 
//    //                                          bolt::amp::plus<T>(), squareMeCode);
//
//    // call transform_reduce
//    DivUDD ddd;
//    bolt::plus<float> add;
//    float boldReduce = bolt::amp::transform_reduce( input.begin(), input.end(),  ddd, 0.f, add );
//    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
//    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0.f); // out-of-place scan
//
//    // compare results
//    cmpArrays(refOutput, output);
//}
//
//
//typedef ::testing::Types< 
//    std::tuple< UDD, TypeValue< 1 > >,
//    std::tuple< UDD, TypeValue< 31 > >,
//    std::tuple< UDD, TypeValue< 32 > >,
//    std::tuple< UDD, TypeValue< 63 > >,
//    std::tuple< UDD, TypeValue< 64 > >,
//    std::tuple< UDD, TypeValue< 127 > >,
//    std::tuple< UDD, TypeValue< 128 > >,
//    std::tuple< UDD, TypeValue< 129 > >,
//    std::tuple< UDD, TypeValue< 1000 > >,
//    std::tuple< UDD, TypeValue< 1053 > >,
//    std::tuple< UDD, TypeValue< 4096 > >,
//    std::tuple< UDD, TypeValue< 4097 > >,
//    std::tuple< UDD, TypeValue< 65535 > >,
//    std::tuple< UDD, TypeValue< 65536 > >
//> UDDTests;


INSTANTIATE_TYPED_TEST_CASE_P( Integer, TransformArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, TransformArrayTest, FloatTests );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TYPED_TEST_CASE_P( Double, TransformArrayTest, DoubleTests );
#endif 
//INSTANTIATE_TYPED_TEST_CASE_P( UDDTest, SortArrayTest, UDDTests );

TEST( TransformReduceStdVectWithInit, OffsetTestDeviceVector)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::amp::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );

    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::amp::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::amp::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::amp::transform_reduce(  dVectorA.begin( ) + offset, dVectorA.end( ),
                                                          bolt::amp::negate<int>(), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( TransformReduceStdVectWithInit, OffsetTestDeviceVectorSerialCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::amp::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );


    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::amp::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::amp::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::amp::transform_reduce(  ctl, dVectorA.begin( ) + offset, dVectorA.end( ),
                                                          bolt::amp::negate<int>(), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#if defined( ENABLE_TBB )
TEST( TransformReduceStdVectWithInit, OffsetTestDeviceVectorMultiCoreCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::amp::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );


    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::amp::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::amp::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::amp::transform_reduce(  ctl, dVectorA.begin( ) + offset, dVectorA.end( ),
                                                          bolt::amp::negate<int>(), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#endif

TEST( TransformReduceStdVectWithInit, OffsetTest)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::amp::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::amp::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::amp::transform_reduce(  stdInput.begin( ) + offset, stdInput.end( ),
                                                          bolt::amp::negate<int>(), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#if defined( ENABLE_TBB )
TEST( TransformReduceStdVectWithInit, OffsetTestMultiCoreCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );



    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::amp::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::amp::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::amp::transform_reduce(  ctl, stdInput.begin( ) + offset, stdInput.end( ),
                                                          bolt::amp::negate<int>(), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#endif

TEST( TransformReduceStdVectWithInit, OffsetTestSerialCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::vector<int> temp( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );



    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    // STD
    std::transform(  stdInput.begin( ) + offset, stdInput.end( ),
                     temp.begin() + offset, bolt::amp::negate<int>());
    int stlTransformReduce = std::accumulate(  temp.begin() + offset, temp.end(), init, bolt::amp::plus<int>()  );

    // BOLT
    int boltTransformReduce= bolt::amp::transform_reduce(  ctl, stdInput.begin( ) + offset, stdInput.end( ),
                                                          bolt::amp::negate<int>(), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST(TransformReduce, Float)
{
     int length = 1<<20;
     std::vector< float > input( length );
     std::vector< float > refInput( length);
     std::vector< float > refIntermediate( length );
	 for (int i = 0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
    }
   // call transform_reduce
    //  DivUDD ddd;
    bolt::amp::negate<float> ddd;
    bolt::amp::plus<float> add;
    float boldReduce = bolt::amp::transform_reduce( input.begin(), input.end(),  ddd, 4.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.f, add); //out-of-place scan

  //  printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
} 
TEST(TransformReduce, SerialFloat)
{
     int length = 1<<20;
     std::vector< float > input( length );
     std::vector< float > refInput( length);
     std::vector< float > refIntermediate( length );
	 for (int i = 0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

    // call transform_reduce
    //  DivUDD ddd;
    bolt::amp::negate<float> ddd;
    bolt::amp::plus<float> add;
    float boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, 4.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.f, add); //out-of-place scan

  //  printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
} 
#if defined( ENABLE_TBB )
TEST(TransformReduce, MulticoreFloat)
{
     int length = 1<<20;
     std::vector< float > input( length );
     std::vector< float > refInput( length);
     std::vector< float > refIntermediate( length );
	 for (int i = 0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    // call transform_reduce
    //  DivUDD ddd;
    bolt::amp::negate<float> ddd;
    bolt::amp::plus<float> add;
    float boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, 4.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.f, add);// out-of-place scan

  //  printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
} 
#endif

#if(TEST_DOUBLE == 1)
TEST(TransformReduce, Double)
{
     int length = 1<<20;
     std::vector< double > input( length );
     std::vector< double > refInput( length);
     std::vector< double > refIntermediate( length );
     for(int i=0; i<length; i++) {
        input[i] = 2.0;
        refInput[i] = 2.0;
    }
    bolt::amp::negate<double> ddd;
    bolt::amp::plus<double> add;
    double boldReduce = bolt::amp::transform_reduce( input.begin(), input.end(),  ddd, 4.0, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    double stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.0, add);//out-of-place scan

    //printf("%d %lf %lf\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_DOUBLE_EQ( stdReduce, boldReduce );
  
} 
TEST(TransformReduce, SerialDouble)
{
     int length = 1<<20;
     std::vector< double > input( length );
     std::vector< double > refInput( length);
     std::vector< double > refIntermediate( length );
     for(int i=0; i<length; i++) {
        input[i] = 2.0;
        refInput[i] = 2.0;
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    // call transform_reduce
    //  DivUDD ddd;
    bolt::amp::negate<double> ddd;
    bolt::amp::plus<double> add;
    double boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, 4.0, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    double stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.0, add);//out-of-place scan

    //printf("%d %lf %lf\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_DOUBLE_EQ( stdReduce, boldReduce );
  
} 
#if defined( ENABLE_TBB )
TEST(TransformReduce, MulticoreDouble)
{
     int length = 1<<20;
     std::vector< double > input( length );
     std::vector< double > refInput( length);
     std::vector< double > refIntermediate( length );
     for(int i=0; i<length; i++) {
        input[i] = 2.0;
        refInput[i] = 2.0;
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    // call transform_reduce
    //  DivUDD ddd;
    bolt::amp::negate<double> ddd;
    bolt::amp::plus<double> add;
    double boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, 4.0, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    double stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 4.0, add);//out-of-place scan

    //printf("%d %lf %lf\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_DOUBLE_EQ( stdReduce, boldReduce );
  
} 
#endif
#endif

TEST(TransformReduce, DefaultUDD)
{
    int length = 1<<20;
    UDD initial;
    initial.a = 2;
    initial.b = 2;
    std::vector< UDD > input( length, initial );
    std::vector< UDD > refInput( length, initial );
    std::vector< UDD > refIntermediate( length);
	for (int i = 0; i<length; i++) {
        input[i].a = 2;
        refInput[i].a = 2;
        input[i].b = 2;
        refInput[i].b = 2;
    }
    negateUDD ddd;
    bolt::amp::plus<UDD> add;
    UDD boldReduce = bolt::amp::transform_reduce( input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    UDD stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), initial,add);//out-of-place scan
    //printf("%d %d %d %d %d\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
TEST(TransformReduce, SerialUDD)
{
    int length = 1<<20;
    UDD initial;
    initial.a = 2;
    initial.b = 2;
    std::vector< UDD > input( length, initial );
    std::vector< UDD > refInput( length, initial );
    std::vector< UDD > refIntermediate( length);
	for (int i = 0; i<length; i++) {
        input[i].a = 2;
        refInput[i].a = 2;
        input[i].b = 2;
        refInput[i].b = 2;
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    negateUDD ddd;
    bolt::amp::plus<UDD> add;
    UDD boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    UDD stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), initial,add);//out-of-place scan
    //printf("%d %d %d %d %d\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
#if defined( ENABLE_TBB )
TEST(TransformReduce, MulticoreUDD)
{
    int length = 1<<20;
    UDD initial;
    initial.a = 2;
    initial.b = 2;
    std::vector< UDD > input( length, initial );
    std::vector< UDD > refInput( length, initial );
    std::vector< UDD > refIntermediate( length);
	for (int i = 0; i<length; i++) {
        input[i].a = 2;
        refInput[i].a = 2;
        input[i].b = 2;
        refInput[i].b = 2;
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    negateUDD ddd;
    bolt::amp::plus<UDD> add;
    UDD boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    UDD stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), initial,add);//out-of-place scan
    //printf("%d %d %d %d %d\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
#endif

#if(TEST_DOUBLE == 1)
TEST(TransformReduce, DoubleUDD)
{
    int length = 1<<20;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    std::vector< tbbUDD > input( length, initial );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
     for(int i=0; i<length; i++) {
        input[i].a = 1.f;
        refInput[i].a = 1.f;
        input[i].b = 5.0;
        refInput[i].b = 5.0;
    }
    negatetbbUDD ddd;
    bolt::amp::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::amp::transform_reduce(input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    //printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
TEST(TransformReduce, SerialDoubleUDD)
{
    int length = 1<<20;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    std::vector< tbbUDD > input( length, initial );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
     for(int i=0; i<length; i++) {
        input[i].a = 1.f;
        refInput[i].a = 1.f;
        input[i].b = 5.0;
        refInput[i].b = 5.0;
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    negatetbbUDD ddd;
    bolt::amp::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    //printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
#if defined( ENABLE_TBB )
TEST(TransformReduce, MulticoreDoubleUDD)
{
    int length = 1<<20;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    std::vector< tbbUDD > input( length, initial );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
     for(int i=0; i<length; i++) {
        input[i].a = 1.f;
        refInput[i].a = 1.f;
        input[i].b = 5.0;
        refInput[i].b = 5.0;
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    negatetbbUDD ddd;
    bolt::amp::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    //printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
#endif
#endif

TEST(TransformReduce, DeviceVectorInt)
{
     int length = 1<<16;
     std::vector<  int > refInput( length);
     std::vector< int > refIntermediate( length );
     bolt::amp::device_vector< int > input(length,0);
	 for (int i = 0; i<length; i++) {
        input[i] = i;
        refInput[i] = i;
     //   printf("%d \n", input[i]);
     }
     bolt::amp::negate<int> ddd;
     bolt::amp::plus<int> add;

     int boldReduce = bolt::amp::transform_reduce( input.begin(), input.end(),  ddd, 0, add );
     ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
     int stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0); // out-of-place scan
     printf("%d %d %d\n", length, boldReduce, stdReduce);  
     // compare results
     EXPECT_EQ( stdReduce, boldReduce );
  
  
} 
TEST(TransformReduce, SerialDeviceVectorInt)
{
     int length = 1<<16;
     std::vector<  int > refInput( length);
     std::vector< int > refIntermediate( length );
     bolt::amp::device_vector< int > input(length,0);
	 for (int i = 0; i<length; i++) {
        input[i] = i;
        refInput[i] = i;
     //   printf("%d \n", input[i]);
     }
     bolt::amp::control ctl = bolt::amp::control::getDefault( );
     ctl.setForceRunMode(bolt::amp::control::SerialCpu);
     // call transform_reduce
     //  DivUDD ddd;
     bolt::amp::negate<int> ddd;
     bolt::amp::plus<int> add;

     int boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0, add );
     ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
     int stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0); // out-of-place scan
     printf("%d %d %d\n", length, boldReduce, stdReduce);  
     // compare results
     EXPECT_EQ( stdReduce, boldReduce );
  
  
} 
#if defined( ENABLE_TBB )
TEST(TransformReduce, MulticoreDeviceVectorInt)
{
     int length = 1<<16;
     std::vector<  int > refInput( length);
     std::vector< int > refIntermediate( length );
     bolt::amp::device_vector< int > input(length,0);
	 for (int i = 0; i<length; i++) {
        input[i] = i;
        refInput[i] = i;
     //   printf("%d \n", input[i]);
     }
     bolt::amp::control ctl = bolt::amp::control::getDefault( );
     ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
     // call transform_reduce
     //  DivUDD ddd;
     bolt::amp::negate<int> ddd;
     bolt::amp::plus<int> add;

     int boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0, add );
     ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
     int stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0); // out-of-place scan
     printf("%d %d %d\n", length, boldReduce, stdReduce);  
     // compare results
     EXPECT_EQ( stdReduce, boldReduce );
  
  
} 
#endif

TEST(TransformReduce, DeviceVectorFloat)
{
   
     int length = 1<<16;
     
     std::vector<  float > refInput( length);
     std::vector< float > refIntermediate( length );
     bolt::amp::device_vector< float > input(length,0);

	 for (int i = 0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
     //   printf("%d \n", input[i]);
    }
    bolt::amp::negate<float> ddd;
    bolt::amp::plus<float> add;

    float boldReduce = bolt::amp::transform_reduce( input.begin(), input.end(),  ddd, 0.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0.f); // out-of-place scan

    printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
  
} 
TEST(TransformReduce, SerialDeviceVectorFloat)
{
   
     int length = 1<<16;
     
     std::vector<  float > refInput( length);
     std::vector< float > refIntermediate( length );
     bolt::amp::device_vector< float > input(length,0);

	 for (int i = 0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
     //   printf("%d \n", input[i]);
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    bolt::amp::negate<float> ddd;
    bolt::amp::plus<float> add;

    float boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0.f); // out-of-place scan

    printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
  
} 
#if defined( ENABLE_TBB )
TEST(TransformReduce, MulticoreDeviceVectorFloat)
{
   
     int length = 1<<16;
     
     std::vector<  float > refInput( length);
     std::vector< float > refIntermediate( length );
     bolt::amp::device_vector< float > input(length,0);

	 for (int i = 0; i<length; i++) {
        input[i] = 2.f;
        refInput[i] = 2.f;
     //   printf("%d \n", input[i]);
    }
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    bolt::amp::negate<float> ddd;
    bolt::amp::plus<float> add;

    float boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, 0.f, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    float stdReduce = ::std::accumulate( refIntermediate.begin(), refIntermediate.end(), 0.f); // out-of-place scan

    printf("%d %f %f\n", length, boldReduce, stdReduce);  
    // compare results
    EXPECT_FLOAT_EQ( stdReduce, boldReduce );
  
  
} 
#endif

#if(TEST_DOUBLE == 1	)
TEST(TransformReduce, DeviceVectorUDD)
{
    int length = 1<<16;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    bolt::amp::device_vector< tbbUDD > input(  length, initial,  true );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
    /*
     for(int i=0; i<length; i++) {
        input[i].a = 1.f;
        refInput[i].a = 1.f;
        input[i].b = 5.0;
        refInput[i].b = 5.0;
    }
    */
    negatetbbUDD ddd;
    bolt::amp::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::amp::transform_reduce(input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
TEST(TransformReduce, SerialDeviceVectorUDD)
{
    int length = 1<<16;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    bolt::amp::device_vector< tbbUDD > input(  length, initial,  true );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
    /*
     for(int i=0; i<length; i++) {
        input[i].a = 1.f;
        refInput[i].a = 1.f;
        input[i].b = 5.0;
        refInput[i].b = 5.0;
    }
    */
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    negatetbbUDD ddd;
    bolt::amp::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
#if defined( ENABLE_TBB )
TEST(TransformReduce, MulticoreDeviceVectorUDD)
{
    int length = 1<<16;
    tbbUDD initial;
    initial.a = 2.f;
    initial.b = 5.0;
    bolt::amp::device_vector< tbbUDD > input(  length, initial,  true );
    std::vector< tbbUDD > refInput( length, initial );
    std::vector< tbbUDD > refIntermediate( length);
    /*
     for(int i=0; i<length; i++) {
        input[i].a = 1.f;
        refInput[i].a = 1.f;
        input[i].b = 5.0;
        refInput[i].b = 5.0;
    }
    */
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    negatetbbUDD ddd;
    bolt::amp::plus<tbbUDD> add;
    tbbUDD boldReduce = bolt::amp::transform_reduce(ctl, input.begin(), input.end(),  ddd, initial, add );
    ::std::transform(   refInput.begin(), refInput.end(),  refIntermediate.begin(), ddd); // transform in-place
    tbbUDD stdReduce = ::std::accumulate(refIntermediate.begin(),refIntermediate.end(),initial,add);//out-of-place scan
    printf("%d %f %f %lf %lf\n", length, boldReduce.a, stdReduce.a, boldReduce.b, stdReduce.b);  
    // compare results
    EXPECT_EQ( stdReduce, boldReduce );
    
} 
#endif
#endif

const char * bolt_code_path_adjustment = "Automatic";

#define TAKE_AMP_CONTROL_PATH bolt::amp::control& my_amp_ctl= bolt::amp::control::getDefault(); \
if (strcmp(bolt_code_path_adjustment, "Automatic") == 0 ) \
{\
my_amp_ctl.setWaitMode( bolt::amp::control::NiceWait );\
my_amp_ctl.setForceRunMode(bolt::amp::control::Automatic);\
}\
if (strcmp(bolt_code_path_adjustment, "Gpu") == 0 )\
{\
my_amp_ctl.setWaitMode( bolt::amp::control::NiceWait );\
my_amp_ctl.setForceRunMode(bolt::amp::control::Gpu);\
}\
if (strcmp(bolt_code_path_adjustment, "MultiCoreCpu") == 0 )\
{\
my_amp_ctl.setWaitMode( bolt::amp::control::NiceWait );\
my_amp_ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);\
}\
if (strcmp(bolt_code_path_adjustment, "SerialCpu") == 0 )\
{\
my_amp_ctl.setWaitMode( bolt::amp::control::NiceWait );\
my_amp_ctl.setForceRunMode(bolt::amp::control::SerialCpu);\
}


class point{
  public:
  int xPoint;
  int yPoint;

  point()  restrict(cpu,amp)
  {
    xPoint =0;
    yPoint =0;
  }

  point(int x, int y)  restrict(cpu,amp)
  {
    xPoint = x;
    yPoint = y;
  }

  point operator + (const point &rhs) const restrict(cpu,amp)
  {
    point tmp = *this;
    tmp.xPoint = tmp.xPoint + rhs.xPoint;
    tmp.yPoint = tmp.yPoint + rhs.yPoint;
    return tmp;
  }
  point operator - (const point &rhs) const restrict(cpu,amp)
  {
    point tmp = *this;
    tmp.xPoint = tmp.xPoint - rhs.xPoint;
    tmp.yPoint = tmp.yPoint - rhs.yPoint;
    return tmp;
  }

  point operator - () const restrict(cpu,amp)
  {
    point tmp = *this;
    tmp.xPoint = -1 * (tmp.xPoint);
    tmp.yPoint = -1 * (tmp.yPoint);
    return tmp;
  }
  point operator * (const point &rhs) const restrict(cpu,amp)
  {
    point tmp = *this;
    tmp.xPoint = tmp.xPoint * rhs.xPoint;
    tmp.yPoint = tmp.yPoint * rhs.yPoint;
    return tmp;
  }

};


//failed compilation

TEST(Bug377748, userDefinedDataType)
{

  point pt1(12, 3);
  point pt2(2, 5);
  point pt(0, 0);
  TAKE_AMP_CONTROL_PATH
  std::vector<point> my_input_bolt_dev_vect(2);

  my_input_bolt_dev_vect[0].xPoint = 12 ;
  my_input_bolt_dev_vect[0].yPoint = 3 ; 


  my_input_bolt_dev_vect[1].xPoint = 2 ; 
  my_input_bolt_dev_vect[1].yPoint = 5 ; 

  bolt::amp::square<point> sq;
  bolt::amp::plus<point> pl;

  point newPt = bolt::amp::transform_reduce(my_amp_ctl, my_input_bolt_dev_vect.begin(), my_input_bolt_dev_vect.end(), 
  sq, pt, pl);

  //Expected result is newPt (148, 38)
  EXPECT_EQ (148, newPt.xPoint);
  EXPECT_EQ (34, newPt.yPoint);
}


int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );
        //  Register our minidump generating logic
    #if defined(_WIN32)
    bolt::miniDumpSingleton::enableMiniDumps( );
    #endif
    bolt::amp::control& myControl = bolt::amp::control::getDefault( );
    myControl.setWaitMode( bolt::amp::control::NiceWait );
    myControl.setForceRunMode( bolt::amp::control::Automatic );  // choose tbb


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
#endif
