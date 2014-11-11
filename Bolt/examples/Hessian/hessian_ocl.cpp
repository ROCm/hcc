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


#include "myocl.h"
#include "myocl_bufferpool.h"
#include "Hessian.h"
#include "hessian_ocl.h"

#include "matrix_utils.h"
#include "myocl.h"



OclBufferPool oclBufferPool;

int g_oclc_launch_end;
int g_oclc_launch_start = __LINE__+1;
bool update_trz_ocl(const MyOclContextC &ocl, cl_kernel k, H3& dH, const cl_mem I1, int ipitch, const cl_mem wI2, int wpitch, float sigma, float gradThresh, const  utils::Rect& roi)
{
	// Compute number of WG required to fill the machine:
	const int wgPerComputeUnit=p_wgPerComputeUnit; // pick a number high enough to hide latency
	int computeUnits;
	clGetDeviceInfo(ocl._device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int),&computeUnits,NULL);
	if (p_computeUnits) computeUnits = p_computeUnits; //override
	int elements     = wgPerComputeUnit * computeUnits * 16; 
	int sizeBytes = elements * sizeof(float);

	int localW = p_localW;
	int localH = p_localH;

	int globalW = computeUnits*localW;
	int globalH = wgPerComputeUnit*localH;
	globalW = (roi.width()  < globalW) ? roi.width() : globalW;
	globalH = (roi.height() < globalH) ? roi.height() : globalH;

	int err=0;
	cl_mem dest = oclBufferPool.allocCl(ocl._context, p_zeroCopy? (CL_MEM_ALLOC_HOST_PTR|CL_MEM_WRITE_ONLY) : (CL_MEM_WRITE_ONLY), sizeBytes);

	int arg=0;
	utils::Rect bRoi(roi.top+1, roi.left+1, roi.bottom-1, roi.right-1);
	clSetKernelArg(k, arg++, sizeof(dest), &dest);
	clSetKernelArg(k, arg++, sizeof(I1), &I1); 
	clSetKernelArg(k, arg++, sizeof(ipitch), &ipitch); 
	clSetKernelArg(k, arg++, sizeof(wI2), &wI2);
	clSetKernelArg(k, arg++, sizeof(wpitch), &wpitch); 

	clSetKernelArg(k, arg++, sizeof(roi.top), &bRoi.top);
	clSetKernelArg(k, arg++, sizeof(roi.left), &bRoi.left); 
	clSetKernelArg(k, arg++, sizeof(roi.bottom), &bRoi.bottom);
	clSetKernelArg(k, arg++, sizeof(roi.right), &bRoi.right); 

	float Xc = (bRoi.left + bRoi.right) / 2.0f;
	float Yc = (bRoi.top + bRoi.bottom) / 2.0f;

	clSetKernelArg(k, arg++, sizeof(Xc), &Xc);
	clSetKernelArg(k, arg++, sizeof(Yc), &Yc); 

	float sigma_sq = sigma * sigma;
	clSetKernelArg(k, arg++, sizeof(gradThresh), &gradThresh);  //thres
	clSetKernelArg(k, arg++, sizeof(sigma_sq), &sigma_sq);  //siqma_sq

	size_t globalThreads[2] = {globalW, globalH};
	size_t localThreads[2]  = {localW, localH};
	clEnqueueNDRangeKernel(ocl._queue, k, 2, NULL, globalThreads, localThreads,0,NULL,NULL);
	g_oclc_launch_end = __LINE__-1;

	// Map the data back and run the computation on the CPU:
	float * destH = (float*)clEnqueueMapBuffer(ocl._queue, dest, true/*blocking*/, CL_MAP_READ, 0/*offset*/, sizeBytes, 0, NULL, NULL, NULL);
	oclBufferPool.free(dest);

	{
		//sumBlocks(hessian, (float*)mappedDst, elements);
		const int NH = 16;
		float hessian[NH];
		for (int i=0; i<NH; i++) hessian[i] = 0.0f;
		for (int i=0; i<elements; i++)
			hessian[i%NH] += destH[i];

		double H[4][4];
		double b[4];

		for (int i=0,k=0; i<5; i++)
		{
			if (i == 4)
			{
				for (int j=0; j<4; j++) 
					b[j] = (double)hessian[k++];
			}
			else
			{
				for (int j=0; j<=i; j++) 
					H[i][j] = H[j][i] = (double)hessian[k++];
			}
		}


		double dm[4];
		dH.zeros();

		if (solve_ChD<double,4> (dm, H, b))
		{
			dH[0] = dH[4] = dm[0];
			dH[1] = dm[1];
			dH[2] = dm[2];
			dH[5] = dm[3];
			dH[3] = -dH[1];
			return true;
		}

		return false;
	}



	return true;
}
