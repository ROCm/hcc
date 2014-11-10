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
#include "bolt/amp/merge.h"
#include "bolt/amp/sort.h"
#include "common/test_common.h"
#include "bolt/unicode.h"
#include <gtest/gtest.h>
#include <array>



struct UDD
{
	int a;
	int b;

	bool operator() (const UDD& lhs, const UDD& rhs) const {
		return ((lhs.a + lhs.b) > (rhs.a + rhs.b));
	}
	bool operator < (const UDD& other) const {
		return ((a + b) < (other.a + other.b));
	}
	bool operator > (const UDD& other) const {
		return ((a + b) > (other.a + other.b));
	}
	bool operator == (const UDD& other) const restrict (cpu,amp)  {
		return ((a + b) == (other.a + other.b));
	}

	UDD operator + (const UDD &rhs) const
	{
		UDD _result;
		_result.a = a + rhs.a;
		_result.b = b + rhs.b;
		return _result;
	}

	UDD()
		: a(0), b(0) { }
	UDD(int _in)
		: a(_in), b(_in + 1)  { }
};


struct UDDless
{
	bool operator() (const UDD &lhs, const UDD &rhs) const restrict(cpu, amp)
	{

		if ((lhs.a + lhs.b) < (rhs.a + rhs.b))
			return true;
		else
			return false;
	}

};


#if defined ( _WIN32 )
// .........sanity test cases without controls.....//

TEST(sanity_merge_amp_doc_arrays_, wo_ctrl_ints){
	int A[10] = { 0, 10, 42, 55, -60, 60, 0, -77, 0, 37 };
	int B[10] = { 2, 6, 23, -34, 40, 43, 0, 55, -42, 0 };
	int stdmerge[20];
	int boltmerge[20];

	//for(int i = 0; i < 10; i++) {
	//std::cout << "Before val is " << A[i] <<" " << B[i] << " " << "\n";
	//	}

	bolt::amp::sort(A, A + 10);
	bolt::amp::sort(B, B + 10);
	std::merge(A, A + 10, B, B + 10, stdmerge);
	bolt::amp::merge(A, A + 10, B, B + 10, stdext::make_checked_array_iterator(boltmerge,20));


	for(int i = 0; i < 20; i++) {
		std::cout << "val is " << stdmerge[i] << " " << boltmerge[i] << " " << "\n";
	}

	cmpArrays(stdmerge,boltmerge,20);
}
#endif



TEST(MergeUDD, UDDPlusOperatorInts)
{
	int length = 1 << 8;

	std::vector<UDD> std1_source(length);
	std::vector<UDD> std2_source(length);
	std::vector<UDD> std_res(length * 2);
	std::vector<UDD> bolt_res(length * 2);

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
		std_res.begin(), lessop);


	bolt::amp::merge(std1_source.begin(), std1_source.end(),
		std2_source.begin(), std2_source.end(),
		bolt_res.begin(), lessop);

	// GoogleTest Comparison
	cmpArrays(std_res, bolt_res);

}


TEST(MergeTest, MergeAuto)
{
	int stdVectSize1 = 10;
	int stdVectSize2 = 20;

	std::vector<int> A(stdVectSize1);
	std::vector<int> B(stdVectSize1);
	std::vector<int> stdmerge(stdVectSize2);
	std::vector<int> boltmerge(stdVectSize2);

	for (int i = 0; i < stdVectSize1; i++){
		A[i] = 10;
		B[i] = 20;
	}

	std::merge(A.begin(), A.end(), B.begin(), B.end(), stdmerge.begin());
	bolt::amp::control ctl;
	ctl.setForceRunMode(bolt::amp::control::Automatic);
	bolt::amp::merge(ctl, A.begin(), A.end(), B.begin(), B.end(), boltmerge.begin());

	for (int i = 0; i < stdVectSize2; i++) {
		EXPECT_EQ(boltmerge[i], stdmerge[i]);
	}
}

#if defined ( _WIN32 )
TEST(sanity_merge_amp_doc_std_arrays_, wo_ctrl_ints){
	int A[10] = { 8, 10, 42, 55, 60, 60, 75, 77, 99, 37 };
	int B[10] = { 2, 6, 23, 34, 40, 43, 55, 55, 42, 80 };
	int stdmerge[20];
	int boltmerge[20];

	//for (int i = 0; i < 10; i++) {
	//std::cout << "Before val is " << A[i] <<" " << B[i] << " " << "\n";
	//	}

	std::sort(A, A + 10);
	std::sort(B, B + 10);
	std::merge(A, A + 10, B, B + 10, stdmerge);


	bolt::amp::sort(A, A + 10);
	bolt::amp::sort(B, B + 10);
	bolt::amp::merge(A, A + 10, B, B + 10, stdext::make_checked_array_iterator(boltmerge,20));

	

	for (int i = 0; i < 20; i++) {
		EXPECT_EQ(boltmerge[i], stdmerge[i]);
	}

}

TEST(sanity_merge_amp_ArrWithDiffTypes, WithInt){
	int arraySize = 100;
	int arraySize1 = 200;

	int* InArr1;
	int* InArr2;

	int* outArr1;
	int* outArr2;

	InArr1 = (int *)malloc(arraySize* sizeof (int));
	InArr2 = (int *)malloc(arraySize* sizeof (int));
	outArr1 = (int *)malloc(arraySize1 * sizeof (int));
	outArr2 = (int *)malloc(arraySize1 * sizeof (int));


	for (int i = 0; i < arraySize; i++){
		InArr1[i] = 56535 - i;
	}

	for (int i = 0; i < arraySize; i++){
		InArr2[i] = i;
	}

	std::sort(InArr1, InArr1 + arraySize);
	std::sort(InArr2, InArr2 + arraySize);
	std::merge(InArr1, InArr1 + arraySize, InArr2, InArr2 + arraySize, stdext::make_checked_array_iterator(outArr1,arraySize1));

	bolt::amp::sort(InArr1, InArr1 + arraySize);
	bolt::amp::sort(InArr2, InArr2 + arraySize);
	bolt::amp::merge(InArr1, InArr1 + arraySize, InArr2, InArr2 + arraySize,  stdext::make_checked_array_iterator(outArr2,arraySize1));



	for (int i = 0; i < arraySize1; i++) {
		EXPECT_EQ(outArr2[i], outArr1[i]);
	}

	free(InArr1);
	free(InArr2);
	free(outArr1);
	free(outArr2);

}


TEST(sanity_merge_amp_ArrWithDiffTypes1, WithFloats){
	int arraySize = 100;
	int arraySize1 = 200;

	float *InFloatArr1;
	float *InFloatArr2;

	float* outArr1;
	float* outArr2;

	InFloatArr1 = (float *)malloc(arraySize* sizeof (float));
	InFloatArr2 = (float *)malloc(arraySize* sizeof (float));
	outArr1 = (float *)malloc(arraySize1 * sizeof (float));
	outArr2 = (float *)malloc(arraySize1 * sizeof (float));

	for (int i = 0; i < arraySize; i++){
		InFloatArr1[i] = (float)i + 0.125f;
		InFloatArr2[i] = (float)i + 0.15f;
	}


	//copying float array as a whole to all there types of arrays :) 
	std::sort(InFloatArr1, InFloatArr1 + arraySize);
	std::sort(InFloatArr2, InFloatArr2 + arraySize);
	std::merge(InFloatArr1, InFloatArr1 + arraySize, InFloatArr2, InFloatArr2 + arraySize, stdext::make_checked_array_iterator(outArr1,arraySize1));

	bolt::amp::sort(InFloatArr1, InFloatArr1 + arraySize);
	bolt::amp::sort(InFloatArr2, InFloatArr2 + arraySize);
	bolt::amp::merge(InFloatArr1, InFloatArr1 + arraySize, InFloatArr2, InFloatArr2 + arraySize,  stdext::make_checked_array_iterator(outArr2,arraySize1) );


	for (int i = 0; i < arraySize1; i++) {
		EXPECT_EQ(outArr2[i], outArr1[i]);
	}

	free(InFloatArr1);
	free(InFloatArr2);
	free(outArr1);
	free(outArr2);

}


TEST(sanity_merge_amp_ArrWithDiffTypes2, WithDouble){
	int arraySize = 100;
	int arraySize1 = 200;

	double *InDoubleArr1;
	double *InDoubleArr2;

	double* outArr1;
	double* outArr2;

	InDoubleArr1 = (double *)malloc(arraySize* sizeof (double));
	InDoubleArr2 = (double *)malloc(arraySize* sizeof (double));
	outArr1 = (double *)malloc(arraySize1 * sizeof (double));
	outArr2 = (double *)malloc(arraySize1 * sizeof (double));

	for (int i = 0; i < arraySize; i++){
		InDoubleArr1[i] = InDoubleArr2[i] = (double)i + 0.0009765625;
	}

	//copying double array as a whole to all there types of arrays :) 
	std::sort(InDoubleArr1, InDoubleArr1 + arraySize);
	std::sort(InDoubleArr2, InDoubleArr2 + arraySize);
	std::merge(InDoubleArr1, InDoubleArr1 + arraySize, InDoubleArr2, InDoubleArr2 + arraySize, stdext::make_checked_array_iterator(outArr1,arraySize1));

	std::sort(InDoubleArr1, InDoubleArr1 + arraySize);
	std::sort(InDoubleArr2, InDoubleArr2 + arraySize);
	bolt::amp::merge(InDoubleArr1, InDoubleArr1 + arraySize, InDoubleArr2, InDoubleArr2 + arraySize, stdext::make_checked_array_iterator(outArr2,arraySize1));

	//for(int i = 0; i < arraySize1; i++ ) {
	//	std::cout << "Val After merge is " << outArr1[i] << " " << outArr2[i] << "\n" ;
	//}	

	for (int i = 0; i < arraySize1; i++) {
		EXPECT_EQ(outArr2[i], outArr1[i]);
	}

	free(InDoubleArr1);
	free(InDoubleArr2);
	free(outArr1);
	free(outArr2);
}

#endif

TEST(sanity_merge_bolt_vect, wo_control_ints)
{
	int aSize = 10;
	int Size = 20;

	std::vector<int> A(aSize);
	std::vector<int> B(aSize);
	std::vector<int>   stdmerge(Size);

	for (int i = 0; i < aSize; i++) {
		B[i] = A[i] = (int)i;
	}

	std::sort(A.begin(), A.end());
	std::sort(B.begin(), B.end());


	bolt::amp::device_vector<int> A1(A.begin(),A.end());
	bolt::amp::device_vector<int> B1(B.begin(),B.end());
	bolt::amp::device_vector<int> boltmerge(Size);

	std::merge(A.begin(), A.end(), B.begin(), B.end(), stdmerge.begin());
	bolt::amp::merge(A1.begin(), A1.end(), B1.begin(), B1.end(), boltmerge.begin());

	cmpArrays(stdmerge, boltmerge);
}



int main(int argc, char* argv[])
{

    
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );


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

