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

//#include "stdafx.h"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <ppl.h> // for ConCRT implementation

#include <math.h>


#include <mmintrin.h>
#include <emmintrin.h>

#include "matrix_utils.h"

#include "hessian.h"
#include "hessian_tbb.h"
#include "hessian_cpuserial.h"


#define MT_SUPPORT 1

namespace sse {

#define SIMD_WIDTH 4
#define UNROLL_WIDTH 1

#define UPDTATEB(idx) \
	{\
	__m128 bt = _mm_load_ps(&b[idx*SIMD_WIDTH]);\
	__m128 temp = _mm_mul_ps(e##idx##[0], It[0]);\
	bt = _mm_sub_ps(bt, temp);\
	_mm_store_ps(&b[idx*SIMD_WIDTH], bt);\
}

#define UPDATEH(idx,i,j) \
	{\
	__m128 ht =  _mm_load_ps(&H[idx*SIMD_WIDTH]);\
	__m128 temp = _mm_mul_ps(e##i##[0], e##j##[0]);\
	ht = _mm_add_ps(ht, temp);\
	_mm_store_ps(&H[idx*SIMD_WIDTH], ht);\
}

	static 
		void  hessianTRZ( double outH[][4],
		double outb[],
		const utils::Matrix<float>& I1, 
		const utils::Matrix<float>& wI2,
		const utils::Rect& roi,
		float gradThresh,
		float Xc, float Yc,
		float sigma)  throw()
	{
		// We adopt a straightforward outer-loop vectorization strategy
		// This version has been adjusted for 4 SIMD elements (which matches nicely with 128-bit)
		// To scale to AVX, increase N to 8.
		// Inside the loop we accumulate Hessian statistics independently for each SIMD lane
		// Then, after the image is done, we sum the Hessian elements across each of the lanes

		const __m128i const_10 = _mm_set_epi32(3,2,1,0);
		const __m128 const_1f = _mm_set1_ps(1.0f);
		const __m128i isNanVal = _mm_set1_epi32(0x7fffffff);
		const __m128 half = _mm_set1_ps(0.5f);
		const __m128 szero = _mm_set1_ps(-0.0f);

		__declspec(align(16)) float H[10*SIMD_WIDTH];
		__declspec(align(16)) float b[4*SIMD_WIDTH];
		memset( H, 0, 10*SIMD_WIDTH*sizeof(float) );
		memset( b, 0,  4*SIMD_WIDTH*sizeof(float) );

		__m128 Xcs = _mm_set1_ps(Xc);   

		__m128 rcp_sigma_sq = _mm_set1_ps( 1.0f/(sigma*sigma) );  
		__m128 gradThreshs = _mm_set1_ps( gradThresh );

		for( long y = roi.top; y <= roi.bottom; y++ ) 
		{
			float Y = (float)(y) - Yc;        
			__m128 Ys = _mm_set1_ps(Y);

			__m128i temp2 = _mm_set1_epi32(roi.left);
			temp2 = _mm_add_epi32(temp2, const_10);
			__m128 X;
			X = _mm_cvtepi32_ps(temp2);
			X = _mm_sub_ps(X, Xcs);

			__m128 Xinc = _mm_set1_ps(SIMD_WIDTH * 1.0f);           

			long x = roi.left;

			// temporary storage
			__m128 e0[UNROLL_WIDTH];
			__m128 e1[UNROLL_WIDTH];
			__m128 e2[UNROLL_WIDTH];
			__m128 e3[UNROLL_WIDTH];
			__m128 It[UNROLL_WIDTH];

			__m128i upper_limit = _mm_set1_epi32(roi.right);
			__m128i Xpos = _mm_set1_epi32(x);
			Xpos = _mm_add_epi32(Xpos, const_10);
			__m128i Xposinc = _mm_set1_epi32(SIMD_WIDTH);


			for(; x <= roi.right; x+=SIMD_WIDTH*UNROLL_WIDTH )
			{
				// ~100 instructions per iteration(MSVC8)
				// we spend around half our time in this loop.
				for (int xx = 0; xx < UNROLL_WIDTH; xx++)
				{
					// if (x(effective) > roi.right) - ignore result.
					// this avoids having to code up a messy remainder loop
					// note the source image MUST be padded to at least 15 samples for this to be safe!
					__m128i zero_mask = _mm_cmpgt_epi32(Xpos, upper_limit);                    
					Xpos = _mm_add_epi32(Xpos, Xposinc);

					// with loop peeling and stride-aligned images we could align 4 of these loads 
					// but it hardly seems worth the effort.
					__m128 wI2r = _mm_loadu_ps(&wI2(y,   x+SIMD_WIDTH*xx+1));
					__m128 wI2l = _mm_loadu_ps(&wI2(y,   x+SIMD_WIDTH*xx-1));
					__m128 wI2u = _mm_loadu_ps(&wI2(y-1, x+SIMD_WIDTH*xx  ));
					__m128 wI2d = _mm_loadu_ps(&wI2(y+1, x+SIMD_WIDTH*xx  ));
					__m128 wI2c = _mm_loadu_ps(&wI2(y,   x+SIMD_WIDTH*xx  ));
					__m128 I1c  = _mm_loadu_ps(&I1 (y,   x+SIMD_WIDTH*xx  ));

					/*
					__m128 IT = _mm_sub_ps( _mm_loadu_ps(&wI2(y,   x+SIMD_WIDTH*xx)),
					_mm_loadu_ps(&I1 (y,   x+SIMD_WIDTH*xx)));
					*/

					// if NaN, ignore result
					zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi32(_mm_castps_si128(wI2r), isNanVal));
					zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi32(_mm_castps_si128(wI2l), isNanVal));
					zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi32(_mm_castps_si128(wI2u), isNanVal));
					zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi32(_mm_castps_si128(wI2d), isNanVal));
					zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi32(_mm_castps_si128(wI2c), isNanVal));
					zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi32(_mm_castps_si128( I1c), isNanVal));

					__m128 Ix = _mm_sub_ps(wI2r, wI2l);
					__m128 Iy = _mm_sub_ps(wI2d, wI2u);
					Ix = _mm_mul_ps(Ix, half);
					Iy = _mm_mul_ps(Iy, half);

					// if either the vertical or horizontal gradient is lower than the threshold, ignore result
					zero_mask = _mm_or_si128(zero_mask, 
						_mm_castps_si128(_mm_and_ps(_mm_cmplt_ps(_mm_andnot_ps(szero, e0[xx]), gradThreshs),
						_mm_cmplt_ps(_mm_andnot_ps(szero, e1[xx]), gradThreshs))));

					__m128 IT = _mm_sub_ps(wI2c, I1c);

					// zero It, Ix, Iy depending on the Nan and thresholding conditions.
					Ix = _mm_andnot_ps(_mm_castsi128_ps(zero_mask), Ix);
					Iy = _mm_andnot_ps(_mm_castsi128_ps(zero_mask), Iy);
					IT = _mm_andnot_ps(_mm_castsi128_ps(zero_mask), IT);

					// Finally, start computing the e's     

					// special casing this loop to avoid the divison (or the test/branch) hardly seems worth the effort
					if (sigma != 0.0f)
					{
						__m128 weight;// = const_1f;
						weight = _mm_rcp_ps(_mm_add_ps( const_1f, _mm_mul_ps( rcp_sigma_sq, _mm_mul_ps(IT, IT))));
						Ix      = _mm_mul_ps(Ix, weight);
						Iy      = _mm_mul_ps(Iy, weight);
						It[xx]  = _mm_mul_ps(IT, weight);               
					}
					else
					{
						It[xx] = IT;
					}


					e0[xx] = _mm_add_ps(_mm_mul_ps(X,Ix),_mm_mul_ps(Ys,Iy));
					e1[xx] = _mm_sub_ps(_mm_mul_ps(Ys,Ix),_mm_mul_ps(X,Iy));
					e2[xx] = Ix;
					e3[xx] = Iy;

					// update position counters                                                            
					X = _mm_add_ps(X, Xinc);
				}

				UPDTATEB(0)
					UPDTATEB(1)
					UPDTATEB(2)
					UPDTATEB(3)

					UPDATEH(0,0,0);
				UPDATEH(1,1,0);
				UPDATEH(2,1,1);
				UPDATEH(3,2,0);
				UPDATEH(4,2,1);
				UPDATEH(5,2,2);
				UPDATEH(6,3,0);
				UPDATEH(7,3,1);
				UPDATEH(8,3,2);
				UPDATEH(9,3,3);            

			} // x    

		} // y

		// The final accumulation loop has to account for results in each SIMD lane 
		float *hptr = &H[0];
		float *bptr = &b[0];
		for(long j = 0; j < 4; j++)
		{
			for(long i = 0; i<=j; i++)
			{
				outH[i][j] = hptr[0];
				for (int xx = 1; xx < SIMD_WIDTH; xx++)
					outH[i][j] += hptr[xx];

				hptr += SIMD_WIDTH;
			}

			outb[j] = bptr[0];
			for (int xx = 1; xx < SIMD_WIDTH; xx++)
				outb[j] += bptr[xx];
			bptr += SIMD_WIDTH;
		}

		// as duplicte the mirror-image coefficients
		for(long j = 0; j<(4-1); j++ )
			for(long i = j+1; i<4; i++ )
				outH[i][j] = outH[j][i];
	}
}; // namespace sse


#if TBB_VECTOR
#define HESS_FUNC sse::hessianTRZ
#else
#define HESS_FUNC fpu_orig::hessianTRZ
#endif


bool update_trz_tbb( H3& dH, const utils::Matrix<float>& I1,  const utils::Matrix<float>& wI2, float sigma, float gradThresh, const  utils::Rect& roi )
{
	static const int PARAM_CNT = 4;

	class Worker
	{
	public:
		Worker( const utils::Matrix<float>& I1,  const utils::Matrix<float>& wI2, float sigma, float gradThresh, const  utils::Rect& roi )
			: _I1(I1), _wI2(wI2), _sigma(sigma), _gradThresh(gradThresh), _roi(roi) 
		{
			clearState();
		}

		// >>> tbb
		Worker( Worker& w, tbb::split ) : _I1(w._I1), _wI2(w._wI2), _sigma(w._sigma), _gradThresh(w._gradThresh), _roi(w._roi)
		{
			clearState();
		}

		void join( const Worker& w )
		{
			for( int i = 0; i < PARAM_CNT; i++ )
			{
				for( int j = 0; j < PARAM_CNT; j++ )
					H[i][j] += w.H[i][j];
			}

			for(int i = 0; i < PARAM_CNT; i++)
				b[i] += w.b[i];
		}

		void operator()( const tbb::blocked_range<int>& r )
		{
			utils::Rect stripe( r.begin(), _roi.left, r.end()-1, _roi.right );

			float Xc = 0.5f * (float)(_roi.right + _roi.left);
			float Yc = 0.5f * (float)(_roi.bottom + _roi.top);

			double Hl[PARAM_CNT][PARAM_CNT];
			double bl[PARAM_CNT];

			HESS_FUNC( Hl, bl, _I1, _wI2, stripe, _gradThresh, Xc, Yc, _sigma );

			for( int i = 0; i < PARAM_CNT; i++ )
			{
				for( int j = 0; j < PARAM_CNT; j++ )
					H[i][j] += Hl[i][j];
			}

			for(int i = 0; i < PARAM_CNT; i++)
				b[i] += bl[i];
		}
		// <<< end tbb

	private: // params
		const utils::Matrix<float>&	_I1;
		const utils::Matrix<float>&	_wI2;
		float					_sigma;
		float					_gradThresh;
		const  utils::Rect&     _roi;

		void clearState() 
		{
			memset( H, 0, PARAM_CNT*PARAM_CNT*sizeof(double) );
			memset( b, 0,           PARAM_CNT*sizeof(double) );
		}

	public: // state
		double H[PARAM_CNT][PARAM_CNT];
		double b[PARAM_CNT];
	};

	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 ); // avoid problems with gradients
	Worker w( I1, wI2, sigma, gradThresh, validRgn );

	tbb::parallel_reduce< tbb::blocked_range<int>, Worker >(
		tbb::blocked_range<int>(validRgn.top, validRgn.bottom+1), w, tbb::auto_partitioner() );

#if 1
	double dx[PARAM_CNT];


	// solve and generate update
	dH.zeros();
	if( solve_ChD<double, PARAM_CNT>(dx, w.H, w.b) )
	{
		dH[0] =  dx[0];
		dH[1] =  dx[1];

		dH[3] = -dx[1];
		dH[4] =  dx[0];

		dH[2] =  dx[2];
		dH[5] =  dx[3];

		return true;
	}
#endif

	return false; // ill conditioned hessian matrix (can not solve)
}



//----
bool update_trz_tbb_lamda( H3& dH, const utils::Matrix<float>& I1,  const utils::Matrix<float>& wI2, float sigma, float gradThresh, const  utils::Rect& roi )
{
	static const int PARAM_CNT = 4;

	class Worker
	{
	public:
		Worker() {
			memset( H, 0, PARAM_CNT*PARAM_CNT*sizeof(double) );
			memset( b, 0,           PARAM_CNT*sizeof(double) );
		};

	public: // state
		double H[PARAM_CNT][PARAM_CNT];
		double b[PARAM_CNT];
	};

	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 ); // avoid problems with gradients

	Worker w =
		tbb::parallel_reduce( 
		tbb::blocked_range<int>(validRgn.top, validRgn.bottom+1), 
		Worker(),
		[=,&I1,&wI2](const tbb::blocked_range<int> &r,  Worker init)->Worker {

			utils::Rect stripe( r.begin(), validRgn.left, r.end()-1, validRgn.right );

			float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
			float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);

			double Hl[PARAM_CNT][PARAM_CNT];
			double bl[PARAM_CNT];

			HESS_FUNC( Hl, bl, I1, wI2, stripe, gradThresh, Xc, Yc, sigma );

			for( int i = 0; i < PARAM_CNT; i++ )
			{
				for( int j = 0; j < PARAM_CNT; j++ )
					init.H[i][j] += Hl[i][j];
			}

			for(int i = 0; i < PARAM_CNT; i++)
				init.b[i] += bl[i];

			return init;

	},
		[](const Worker &x, const Worker &y)->Worker {
			Worker z;
			for( int i = 0; i < PARAM_CNT; i++ )
			{
				for( int j = 0; j < PARAM_CNT; j++ )
					z.H[i][j] = x.H[i][j] + y.H[i][j];
			}

			for(int i = 0; i < PARAM_CNT; i++)
				z.b[i] = x.b[i] + y.b[i];

			return z;
	},
		tbb::auto_partitioner());


#if 1
	double dx[PARAM_CNT];


	// solve and generate update
	dH.zeros();
	if( solve_ChD<double, PARAM_CNT>(dx, w.H, w.b) )
	{
		dH[0] =  dx[0];
		dH[1] =  dx[1];

		dH[3] = -dx[1];
		dH[4] =  dx[0];

		dH[2] =  dx[2];
		dH[5] =  dx[3];

		return true;
	}
#endif

	return false; // ill conditioned hessian matrix (can not solve)
}


#if 0
//----
bool update_trz_ppl_lambda( H3& dH, const utils::Matrix<float>& I1,  const utils::Matrix<float>& wI2, float sigma, float gradThresh, const  utils::Rect& roi )
{
	static const int PARAM_CNT = 4;

	class Worker
	{
	public:
		Worker() {
			memset( H, 0, PARAM_CNT*PARAM_CNT*sizeof(double) );
			memset( b, 0,           PARAM_CNT*sizeof(double) );
		};

	public: // state
		double H[PARAM_CNT][PARAM_CNT];
		double b[PARAM_CNT];
	};

	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 ); // avoid problems with gradients
	//Worker w( I1, wI2, sigma, gradThresh, validRgn );
	//

	Concurrency::combineable<Worker> workers;
	Concurrency::parallel_for(validRgn.top, validRgn.bottom+1, [&](int r) {
		utils::Rect stripe( r.begin(), validRgn.left, r.end()-1, validRgn.right );

		float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
		float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);

		double Hl[PARAM_CNT][PARAM_CNT];
		double bl[PARAM_CNT];

		sse::hessianTRZ( Hl, bl, I1, wI2, stripe, gradThresh, Xc, Yc, sigma );

		for( int i = 0; i < PARAM_CNT; i++ )
		{
			for( int j = 0; j < PARAM_CNT; j++ )
				init.H[i][j] += Hl[i][j];
		}

		for(int i = 0; i < PARAM_CNT; i++)
			init.b[i] += bl[i];

	});

	[=](const tbb::blocked_range<int> &r,  Worker init)->Worker {

		utils::Rect stripe( r.begin(), validRgn.left, r.end()-1, validRgn.right );

		float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
		float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);

		double Hl[PARAM_CNT][PARAM_CNT];
		double bl[PARAM_CNT];

		sse::hessianTRZ( Hl, bl, I1, wI2, stripe, gradThresh, Xc, Yc, sigma );

		for( int i = 0; i < PARAM_CNT; i++ )
		{
			for( int j = 0; j < PARAM_CNT; j++ )
				init.H[i][j] += Hl[i][j];
		}

		for(int i = 0; i < PARAM_CNT; i++)
			init.b[i] += bl[i];

		return init;

	},
		[](const Worker &x, const Worker &y)->Worker {
			Worker z;
			for( int i = 0; i < PARAM_CNT; i++ )
			{
				for( int j = 0; j < PARAM_CNT; j++ )
					z.H[i][j] = x.H[i][j] + y.H[i][j];
			}

			for(int i = 0; i < PARAM_CNT; i++)
				z.b[i] = x.b[i] + y.b[i];

			return z;
	},
		tbb::auto_partitioner());


#if 1
	double dx[PARAM_CNT];


	// solve and generate update
	dH.zeros();
	if( solve_ChD<double, PARAM_CNT>(dx, w.H, w.b) )
	{
		dH[0] =  dx[0];
		dH[1] =  dx[1];

		dH[3] = -dx[1];
		dH[4] =  dx[0];

		dH[2] =  dx[2];
		dH[5] =  dx[3];

		return true;
	}
#endif

	return false; // ill conditioned hessian matrix (can not solve)
}
#endif

