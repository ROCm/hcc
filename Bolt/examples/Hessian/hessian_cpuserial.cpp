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

#include <string.h>

#include "hessian_cpuserial.h"
#include "hessian.h"

namespace fpu_orig {
	using namespace utils;

	inline bool isfNaN(float f) restrict(amp,cpu) {
		return *(long*)&f == 0x7fffffff;
	};


	inline float  fabs(float x)  restrict(cpu,amp) {
		return (x>=0.0f) ? x:-x; // FIXME - need bit AND hack here.
	};


	 
		void hessianTRZ( double outH[][4],
		double outb[],
		const Matrix<float>& I1, 
		const Matrix<float>& wI2,
		const utils::Rect& roi,
		float gradThresh,
		float Xc, float Yc,
		float sigma )  throw()
	{
		float H[4][4];
		float b[4];
		memset (H, 0, 16*sizeof(float));
		memset (b, 0,  4*sizeof(float));

		float e[4];
		float  rcp_sigma_sq = 1.0f/(sigma*sigma);

		for( long y = roi.top; y <= roi.bottom; y++ )
		{
			for( long x = roi.left; x <= roi.right; x++ )
			{
				float It = wI2(y, x) - I1(y, x);

				if( isfNaN(wI2(y, x+1)) | isfNaN(wI2(y, x-1)) |
					isfNaN(wI2(y+1, x)) | isfNaN(wI2(y-1, x)) | isfNaN(It))
					continue;

				float Ix = 0.5f*( wI2(y, x+1) - wI2(y, x-1) );
				float Iy = 0.5f*( wI2(y+1, x) - wI2(y-1, x) );

				if( (fabs(Ix) >= gradThresh) | (fabs(Iy) >= gradThresh) ) 
				{
					float X = (float)(x) - Xc;
					float Y = (float)(y) - Yc;

					if(sigma != 0.0f)
					{
						float weight = 1.0f / (1.0f + rcp_sigma_sq*It*It);
						It = weight*It;
						Ix = weight*Ix;
						Iy = weight*Iy; 
					}

					e[0] = Ix*X + Iy*Y;
					e[1] = Ix*Y - Iy*X;
					e[2] = Ix;
					e[3] = Iy;	

					for (long j=0; j<4; j++)
					{
						for (long i=0; i<=j; i++)
							H[i][j] += e[j]*e[i];
						b[j] -= It*e[j];
					}
				}
			}
		}

		for (long j = 0; j < 4; j++)
		{
			for(long i = 0; i<=j; i++)
				outH[i][j] = (double)H[i][j];
			outb[j] = (double)b[j];
		}

		for(long j = 0; j<(4-1); j++ )
			for(long i = j+1; i<4; i++ )
				outH[i][j] = (double)H[j][i];
	}
}


bool update_trz_cpuserial( H3& dH, const utils::Matrix<float>& I1,  const utils::Matrix<float>& wI2, float sigma, float gradThresh, const  utils::Rect& roi )
{
	static const int PARAM_CNT = 4;

	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 );
	float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
	float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);

	// state
	double H[PARAM_CNT][PARAM_CNT];
	double b[PARAM_CNT];

	fpu_orig::hessianTRZ(H, b, I1, wI2, validRgn, gradThresh, Xc, Yc, sigma);

	double dm[PARAM_CNT];
	dH.zeros();

	if (solve_ChD<double,PARAM_CNT> (dm, H, b))
	{
		dH[0] = dH[4] = dm[0];
		dH[1] = dm[1];
		dH[2] = dm[2];
		dH[5] = dm[3];
		dH[3] = -dH[1];
		return true;
	}

	return false;
};

