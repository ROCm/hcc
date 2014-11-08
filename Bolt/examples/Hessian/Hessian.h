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


namespace HessianMode {
	static const unsigned e_None  					= 0;
	static const unsigned e_Tbb						=	0x01;
	static const unsigned e_TbbLamda				=	0x02;
	static const unsigned e_OclCpu					=   0x04;
	static const unsigned e_OclGpuScalarMatch		=   0x08;
	static const unsigned e_OclCppGpuVector			=   0x10;
	static const unsigned e_OclCppGpuScalar			=   0x20;
	static const unsigned e_OclCppGpuScalarMatch	=   0x40;
	static const unsigned e_OclCppGpuScalarMatch2	=   0x80;
	static const unsigned e_CppAmp					=  0x100;
	static const unsigned e_CppAmpUseWorker			=  0x200;
	static const unsigned e_BoltForAmp				=  0x400;
	static const unsigned e_BoltForAmpLambda		=  0x800;
	static const unsigned e_BoltForAmpCpu			= 0x1000;

	static const unsigned e_CpuSerial				= 0x2000;
	static const unsigned e_OclCppGpuScalarMatch3	= 0x4000;

	static const unsigned e_BoltForAmp_Range		= 0x8000;
	static const unsigned e_BoltForAmp_Range_Cpu	=0x10000;
};

// commandline parms
extern int p_computeUnits;
extern int p_wgPerComputeUnit;
extern int p_localW;
extern int p_localH;
extern int p_zeroCopy;

#ifdef CPPAMP
#define _RESTRICT_AMP restrict(gpu)
#define _RESTRICT_CPUAMP restrict(cpu,amp)
#else
#define _RESTRICT_AMP 
#define _RESTRICT_CPUAMP
#endif


extern void runHessian(int argc, char* argv[]);
extern void printAllLOC();
extern void printLOC(const char *tag, const char *fileName, const char *regionName, int regionStart, int regionEnd);

// cholesky solver for symetric definite systems Ax = b
template<typename T, int size>
bool solve_ChD(T x[], T A[][size], const T b[])
{
	T sum;
	T p[size];

	for (int i = 0; i < size; i++)
	{
		for (int j = i; j< size; j++)
		{
			sum = A[i][j];
			for (int k = i-1; k >= 0; k--) sum -= A[i][k]*A[j][k];
			if (i == j) 
			{
				if (sum <= 0.0) return false;
				p[i] = sqrt(sum);
			} 
			else
				A[j][i]=sum/p[i];
		}
	}

	for ( int i = 0; i < size; i++) 
	{
		sum = b[i];
		for(int k = i-1; k >= 0; k--) sum -= A[i][k]*x[k];
		x[i] = sum/p[i];
	}

	for (int i = size - 1; i>=0; i--) 
	{ 
		sum = x[i];
		for(int k = i+1; k < size; k++) sum -= A[k][i]*x[k];
		x[i] = sum/p[i];
	}
	return true;
}
