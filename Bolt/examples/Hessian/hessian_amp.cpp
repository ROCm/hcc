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



#include "hessian_amp.h"
#include "hessian.h"

#include <vector>
#include <algorithm>
#include <numeric> // for accumulate.

#define BOLT_POOL_ALLOC
#include <bolt/transform.h>
#include <bolt/transform_reduce.h>
#include <bolt/pool_alloc.h>

bolt::ArrayPool<HessianState> arrayPoolAmp;



//IF undefined, use CPU-side serial or STL routines instead of BOLT.
#define USE_BOLT

#if 0
template<typename InputIterator>
bool
	checkResult(InputIterator goldBegin, InputIterator goldEnd, InputIterator test) 
{
	int errCnt = 0;
	for (auto i=goldBegin; i!=goldEnd; i++) {
		if (*i != *test++) {
			printf ("mismatch\n");
			errCnt++;
		};
	};

	return errCnt != 0;
};


// Don't count in code-counts - used for comparison.
bool HessianState::operator != (const HessianState &w) const {
	for (int i=0; i<PARAM_CNT; i++) {
		if (H[i][0] != w.H[i][0] || 
			H[i][1] != w.H[i][1] ||
			H[i][2] != w.H[i][2] ||
			H[i][3] != w.H[i][3]) return true;
	};		
	if (b[0] != w.b[0] || 
		b[1] != w.b[1] ||
		b[2] != w.b[2] ||
		b[3] != w.b[3]) return true;

	return false;
};
#endif


int g_boltamp_kernel_start = __LINE__+1;
inline void hessianTRZ_Point(	
	HessianState &w,
	matrix_type &I1, 
	matrix_type &wI2, 
	utils::Point p,
	float gradThresh,
	float Xc, float Yc,
	float sigma_sq )  restrict(amp,cpu) /*throw()*/ 
{
	float e0, e1, e2, e3;
	float  rcp_sigma_sq = 1.0f/(sigma_sq);

	float Ix = 0.5f*( wI2(p.y,   p.x+1) - wI2(p.y,   p.x-1) );
	float Iy = 0.5f*( wI2(p.y+1, p.x)   - wI2(p.y-1, p.x) );
	float It = wI2(p.y, p.x) - I1(p.y, p.x);

	int hasNan = mymath::isfNaN(It + Ix + Iy);
	Ix = hasNan ? 0.0f : Ix;
	It = hasNan ? 0.0f : It;
	Iy = hasNan ? 0.0f : Iy;

	bool cond3 =  (mymath::fabs(Ix) >= gradThresh) | (mymath::fabs(Iy) >= gradThresh) ;
	float weight = cond3 ? 1.0f : 0.0f;

	bool cond2 = cond3 && (sigma_sq != 0.0f );
	weight = cond2 ? 1.0f / (1.0f + rcp_sigma_sq*It*It) : weight;

	float X = (float)(p.x) - Xc;
	float Y = (float)(p.y) - Yc;

	e0 = Ix*X + Iy*Y;
	e1 = Ix*Y - Iy*X;
	e2 = Ix;
	e3 = Iy;	

	w.Hx[0] =  weight * (e0 * e0); 
	w.Hx[1] =  weight * (e1 * e0);
	w.Hx[2] =  weight * (e1 * e1);
	w.Hx[3] =  weight * (e2 * e0);
	w.Hx[4] =  weight * (e2 * e1);
	w.Hx[5] =  weight * (e2 * e2);
	w.Hx[6] =  weight * (e3 * e0);
	w.Hx[7] =  weight * (e3 * e1);
	w.Hx[8] =  weight * (e3 * e2);
	w.Hx[9] =  weight * (e3 * e3);
	w.Hx[10] = weight * (It * e0); 
	w.Hx[11] = weight * (It * e1); 
	w.Hx[12] = weight * (It * e2); 
	w.Hx[13] = weight * (It * e3); 
}
int g_boltamp_kernel_end = __LINE__-1;


// Return HessianState rather than taking it as a reference.
// Should enable RVO but doesn't in Visual Studio.
static
__forceinline inline HessianState hessianTRZ_Point2(	
	matrix_type &I1, 
	matrix_type &wI2, 
	utils::Point p,
	float gradThresh,
	float Xc, float Yc,
	float sigma_sq )  restrict(amp,cpu) /*throw()*/ 
{
	HessianState w;

	float e0, e1, e2, e3;
	float  rcp_sigma_sq = 1.0f/(sigma_sq);

	float Ix = 0.5f*( wI2(p.y,   p.x+1) - wI2(p.y,   p.x-1) );
	float Iy = 0.5f*( wI2(p.y+1, p.x)   - wI2(p.y-1, p.x) );
	float It = wI2(p.y, p.x) - I1(p.y, p.x);

	int hasNan = mymath::isfNaN(It + Ix + Iy);
	Ix = hasNan ? 0.0f : Ix;
	It = hasNan ? 0.0f : It;
	Iy = hasNan ? 0.0f : Iy;

	bool cond3 =  (mymath::fabs(Ix) >= gradThresh) | (mymath::fabs(Iy) >= gradThresh) ;
	float weight = cond3 ? 1.0f : 0.0f;

	bool cond2 = cond3 && (sigma_sq != 0.0f );
	weight = cond2 ? 1.0f / (1.0f + rcp_sigma_sq*It*It) : weight;

	float X = (float)(p.x) - Xc;
	float Y = (float)(p.y) - Yc;

	e0 = Ix*X + Iy*Y;
	e1 = Ix*Y - Iy*X;
	e2 = Ix;
	e3 = Iy;	

	w.Hx[0] =  weight * (e0 * e0); 
	w.Hx[1] =  weight * (e1 * e0);
	w.Hx[2] =  weight * (e1 * e1);
	w.Hx[3] =  weight * (e2 * e0);
	w.Hx[4] =  weight * (e2 * e1);
	w.Hx[5] =  weight * (e2 * e2);
	w.Hx[6] =  weight * (e3 * e0);
	w.Hx[7] =  weight * (e3 * e1);
	w.Hx[8] =  weight * (e3 * e2);
	w.Hx[9] =  weight * (e3 * e3);
	w.Hx[10] = weight * (It * e0); 
	w.Hx[11] = weight * (It * e1); 
	w.Hx[12] = weight * (It * e2); 
	w.Hx[13] = weight * (It * e3); 

	return w;
}

bool checkChol(const HessianState &ws, H3 &dH) ;

static int g_boltamp_functor_begin=__LINE__+1;
class HessianTransform {
public:
	HessianTransform( matrix_type &I1,  matrix_type& wI2, float sigma, float gradThresh, const utils::Rect& roi )
		: 
		_I1(I1), _wI2(wI2), 
		_sigma_sq(sigma*sigma), _gradThresh(gradThresh), _roi(roi) 
	{
		_Xc = 0.5f * (float)(roi.right + roi.left);
		_Yc = 0.5f * (float)(roi.bottom + roi.top);
	};

	HessianState operator() (concurrency::index<2> i) restrict(amp,cpu)
	{
		utils::Point p(i[1],i[0]);
		return hessianTRZ_Point2( _I1, _wI2, p, _gradThresh, _Xc, _Yc, _sigma_sq);
	};

private: // params
	matrix_type	_I1;
	matrix_type	_wI2;
	float					_sigma_sq;
	float					_gradThresh;
	const  utils::Rect      _roi;
	float					_Xc;
	float					_Yc;
};
static int g_boltamp_functor_end = __LINE__-1;

//----
int g_boltamp_launch_start = 0;
int g_boltamp_launch_end = -1;
bool update_trz_boltforamp( H3& dH, matrix_type& I1, matrix_type& wI2, float sigma, float gradThresh, const  utils::Rect& roi )
{
	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 );

#ifndef USE_BOLT
	enum TestMode { e_SerialLoop, e_StlAccumulateAndReduce };
	static const TestMode testMode = e_SerialLoop;

	if (testMode == e_SerialLoop) {
		printf ("serial loop\n");
		HessianTransform w( I1, wI2, sigma, gradThresh, validRgn );
		// Explicit loop, test strip routine:
		HessianState s1;
		s1.clearState();
		float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
		float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);
		for (int y=validRgn.top; y<=validRgn.bottom; y++) {
			for (int x=validRgn.left; x<=validRgn.right; x++) {
				HessianState s2;
				//s2.clearState();
				hessianTRZ_Point(s2, w._I1, w._wI2, utils::Point(x, y), gradThresh, Xc, Yc, sigma*sigma);
				s1 = s1 + s2;
			}
		}
		return checkChol(s1, dH);  // returns -5.13
	}

	if (testMode == e_StlAccumulateAndReduce) {
		printf ("StlAccumulateAndReduce\n");
		HessianTransform w( I1, wI2, sigma, gradThresh, validRgn );
		// FIXME - need a range or sequence type here to avoid storing indices to memory.  Could make transform accept 2D reduce.
		int vpitch = (validRgn.bottom - validRgn.top + 1);
		int hpitch = (validRgn.right - validRgn.left + 1);
		std::vector<utils::Point> r(vpitch*hpitch);
		for (int i= 0; i<vpitch; i++) {
			for (int j= 0; j<hpitch; j++) {
				r[i*hpitch + j] = utils::Point(j+validRgn.left, i+validRgn.top);
			};
		}


		std::vector<HessianState> transformOut(r.size());
		std::transform(r.begin(), r.end(), transformOut.begin(), w);

		HessianState init;
		init.clearState();

		HessianState result = std::accumulate(transformOut.begin(), transformOut.end(), init, std::plus<HessianState>());
		return checkChol(result, dH); // return -4.84, new is -5.13
	}
#endif

#ifdef USE_BOLT
	g_boltamp_launch_start = __LINE__;
	g_boltamp_launch_end = g_boltamp_launch_start + 7;  //FIXME, approximation...
	HessianTransform w( I1, wI2, sigma, gradThresh, validRgn );
	if (1) {

		HessianState init;
		init.clearState();

		//HessianState result = std::accumulate(transformOut.begin(), transformOut.end(), init, std::plus<HessianState>());
		//HessianState result = std::accumulate(transformOutSTL.begin(), transformOutSTL.end(), init, std::plus<HessianState>());
		//return checkChol(result, dH); // return -4.84

		using namespace concurrency;
		HessianState result = bolt::transform_reduce(index<2>(validRgn.top,validRgn.left), extent<2>(validRgn.height(), validRgn.width()), w, init, bolt::plus<HessianState>());
		return checkChol(result, dH); // return -4.84
	};

#endif

	return false;
};



/// THis version uses a lamda to reduce BOLT function overhead.
bool update_trz_boltforamp_lambda( H3& dH, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi )
{
	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 );

	HessianState init;
	init.clearState();

	float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
	float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);
	float sigma_sq = sigma*sigma;

	using namespace concurrency;
	HessianState result = bolt::transform_reduce(index<2>(validRgn.top,validRgn.left), extent<2>(validRgn.height(), validRgn.width()),
		[=](index<2> i) restrict(cpu,amp) ->HessianState {
			utils::Point p(i[1], i[0]);
			return hessianTRZ_Point2(I1View, wI2View, p, gradThresh, Xc, Yc, sigma_sq);
	}, init, bolt::plus<HessianState>());

	return checkChol(result, dH); // return -4.84

};


/// AMP CPU experiments:



namespace fpu {
	inline static 
		void hessianTRZ_Range(HessianState &w,
		matrix_type_bolt_cpu& I1, 
		matrix_type_bolt_cpu& wI2,
		const utils::Rect& roi,
		float gradThresh,
		float Xc, float Yc,
		float sigma_sq )  throw()
	{
		float H[4][4];
		float b[4];
		memset (H, 0, 16*sizeof(float));
		memset (b, 0,  4*sizeof(float));

		float e[4];
		float  rcp_sigma_sq = 1.0f/(sigma_sq);

		for( long y = roi.top; y <= roi.bottom; y++ )
		{
			for( long x = roi.left; x <= roi.right; x++ )
			{
				float It = wI2(y, x) - I1(y, x);

				float Ix = 0.5f*( wI2(y, x+1) - wI2(y, x-1) );
				float Iy = 0.5f*( wI2(y+1, x) - wI2(y-1, x) );

				bool hasNAN1 = ( mymath::isfNaN(wI2(y, x+1)) | mymath::isfNaN(wI2(y, x-1)) |
					mymath::isfNaN(wI2(y+1, x)) | mymath::isfNaN(wI2(y-1, x)) | mymath::isfNaN(It));

				if(!hasNAN1 && (fabs(Ix) >= gradThresh) | (fabs(Iy) >= gradThresh) ) 
				{
					float X = (float)(x) - Xc;
					float Y = (float)(y) - Yc;

					if(sigma_sq != 0.0f)
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

		w.Hx[0]  +=  H[0][0]; // (e0 * e0); 
		w.Hx[1]  +=  H[0][1]; // (e1 * e0);
		w.Hx[2]  +=  H[1][1]; // (e1 * e1);
		w.Hx[3]  +=  H[0][2]; // (e2 * e0);
		w.Hx[4]  +=  H[1][2]; // (e2 * e1);
		w.Hx[5]  +=  H[2][2]; // (e2 * e2);
		w.Hx[6]  +=  H[0][3]; // (e3 * e0);
		w.Hx[7]  +=  H[1][3]; // (e3 * e1);
		w.Hx[8]  +=  H[2][3]; // (e3 * e2);
		w.Hx[9]  +=  H[3][3]; // (e3 * e3);
		w.Hx[10] +=  b[0]; // (It * e0); 
		w.Hx[11] +=  b[1]; // (It * e1); 
		w.Hx[12] +=  b[2]; // (It * e2); 
		w.Hx[13] +=  b[3]; // (It * e3); 
	}

	inline static 
		void hessianTRZ_Range_Orig(HessianState &w,
		matrix_type_bolt_cpu& I1, 
		matrix_type_bolt_cpu& wI2,
		const utils::Rect& roi,
		float gradThresh,
		float Xc, float Yc,
		float sigma_sq )  throw()
	{
		float H[4][4];
		float b[4];
		memset (H, 0, 16*sizeof(float));
		memset (b, 0,  4*sizeof(float));

		float e[4];
		float  rcp_sigma_sq = 1.0f/(sigma_sq);

		for( long y = roi.top; y <= roi.bottom; y++ )
		{
			for( long x = roi.left; x <= roi.right; x++ )
			{
				float It = wI2(y, x) - I1(y, x);

				if( mymath::isfNaN(wI2(y, x+1)) | mymath::isfNaN(wI2(y, x-1)) |
					mymath::isfNaN(wI2(y+1, x)) | mymath::isfNaN(wI2(y-1, x)) | mymath::isfNaN(It))
					continue;

				float Ix = 0.5f*( wI2(y, x+1) - wI2(y, x-1) );
				float Iy = 0.5f*( wI2(y+1, x) - wI2(y-1, x) );

				if( (fabs(Ix) >= gradThresh) | (fabs(Iy) >= gradThresh) ) 
				{
					float X = (float)(x) - Xc;
					float Y = (float)(y) - Yc;

					if(sigma_sq != 0.0f)
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

#if 0
		w.Hx[0]  =  H[0][0]; // (e0 * e0); 
		w.Hx[1]  =  H[0][1]; // (e1 * e0);
		w.Hx[2]  =  H[1][1]; // (e1 * e1);
		w.Hx[3]  =  H[0][2]; // (e2 * e0);
		w.Hx[4]  =  H[1][2]; // (e2 * e1);
		w.Hx[5]  =  H[2][2]; // (e2 * e2);
		w.Hx[6]  =  H[0][3]; // (e3 * e0);
		w.Hx[7]  =  H[1][3]; // (e3 * e1);
		w.Hx[8]  =  H[2][3]; // (e3 * e2);
		w.Hx[9]  =  H[3][3]; // (e3 * e3);
		w.Hx[10] =  b[0]; // (It * e0); 
		w.Hx[11] =  b[1]; // (It * e1); 
		w.Hx[12] =  b[2]; // (It * e2); 
		w.Hx[13] =  b[3]; // (It * e3); 
#else
		w.Hx[0]  +=  H[0][0]; // (e0 * e0); 
		w.Hx[1]  +=  H[0][1]; // (e1 * e0);
		w.Hx[2]  +=  H[1][1]; // (e1 * e1);
		w.Hx[3]  +=  H[0][2]; // (e2 * e0);
		w.Hx[4]  +=  H[1][2]; // (e2 * e1);
		w.Hx[5]  +=  H[2][2]; // (e2 * e2);
		w.Hx[6]  +=  H[0][3]; // (e3 * e0);
		w.Hx[7]  +=  H[1][3]; // (e3 * e1);
		w.Hx[8]  +=  H[2][3]; // (e3 * e2);
		w.Hx[9]  +=  H[3][3]; // (e3 * e3);
		w.Hx[10] +=  b[0]; // (It * e0); 
		w.Hx[11] +=  b[1]; // (It * e1); 
		w.Hx[12] +=  b[2]; // (It * e2); 
		w.Hx[13] +=  b[3]; // (It * e3); 
#endif


	}
}// namespace



#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>


template <typename T, typename F>
class TbbWorker {
public:
	TbbWorker(T init, F &functor, const utils::Matrix<float> &I1D, const utils::Matrix<float> &wI2D) : 
		_initState(init), _transformState(init), _functor(functor)
	{
	};

	TbbWorker( TbbWorker& tw, tbb::split ) : 
		_transformState(tw._initState), _functor(tw._functor)

	{

	};

	void join(const TbbWorker & tbbW ) 
	{
		_transformState = _transformState + tbbW._transformState;   // reduce_op here
	}

	// 2D Range version
	void operator()( const tbb::blocked_range2d<int>& r )
	{
		//static_assert(USE_ARAY_VIEW);
		for (int y=r.rows().begin(); y<r.rows().end(); y++) {
			for (int x=r.cols().begin(); x<r.cols().end(); x++) {
				_transformState = _transformState +  _functor(concurrency::index<2>(y,x));  //reduce op here
			}
		};
	}




	T _initState;
	T _transformState;
	F _functor;

};


bool update_trz_boltforamp_cpu( H3& dH, const utils::Matrix<float> &I1D, const utils::Matrix<float> &wI2D, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi )
{
	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 );
	HessianTransform w( I1View, wI2View, sigma, gradThresh, validRgn );

	HessianState init;
	init.clearState();

	TbbWorker<HessianState, HessianTransform> tbbW(init, w, I1D, wI2D);

	// warning: this code doesn't run very fast on Microsoft Visual Studio -
	// The  function calls inlined but stack temporaries re not removed (poor Return Value Opt)
	// which leads to poor overall performance.  To work around this, see the hessian_amp_range.cpp 
	// file, which moves the iteration loop into the function.

	if (1) {
		// 2D version:
		tbb::parallel_reduce(tbb::blocked_range2d<int>( validRgn.top, validRgn.bottom+1, validRgn.left, validRgn.right+1), tbbW);

		// 1D version:
		//tbb::parallel_reduce(tbb::blocked_range<int>( validRgn.top, validRgn.bottom+1), tbbW);

		return  checkChol(tbbW._transformState, dH); // return -4.84
	}



	// Serial version:
	if (0) {
		//printf ("running serial version\n");
		float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
		float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);

		utils::Rect r = validRgn;
		for (int y=r.top; y<r.bottom+1; y++) {
			for (int x=r.left; x<r.right+1; x++) {
				utils::Point p(x,y);
				init = init + hessianTRZ_Point2(I1View, wI2View, p, gradThresh, Xc, Yc, sigma*sigma);
			}
		};

		//
		return  checkChol(init, dH);
	}

};


#define BARRIER(W)  // FIXME - placeholder for future barrier insertions
#define REDUCE_STEP1(_IDX, _W) \
	if (_IDX < _W) results0[_IDX] = results0[_IDX] + results0[_IDX+_W]; \
	BARRIER(_W)

bool update_trz_amp( H3& dH, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi )
{
	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 );

	int wgPerComputeUnit =  p_wgPerComputeUnit; 
	int computeUnits     = p_computeUnits ? p_computeUnits : 10 ; // FIXME - determine from HSA Runtime

	int resultCnt = computeUnits * wgPerComputeUnit;
	static const int localH = 8;  //p_localH
	static const int localW = 8;  //p_localW
	static const int waveSize  = localH*localW; // FIXME, read from device attributes.

	HessianState init;
	init.clearState();

	float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
	float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);
	float sigma_sq = sigma*sigma;

	int globalH = wgPerComputeUnit * localH;
	int globalW = computeUnits * localW;

	globalH = (validRgn.height() < globalH) ? validRgn.height() : globalH;
	globalW = (validRgn.width() < globalW) ? validRgn.width() : globalW;

	concurrency::extent<2> launchExt(globalH, globalW);
	bolt::ArrayPool<HessianState>::PoolEntry &entry = arrayPoolAmp.alloc(concurrency::accelerator().default_view, resultCnt);
	concurrency::array<HessianState,1> &results1 = *(entry._dBuffer);
	concurrency::parallel_for_each(launchExt.tile<localH, localW>(), [=,&results1](concurrency::tiled_index<localH, localW> idx) mutable restrict(amp)
	{
		tile_static HessianState results0[waveSize]; 

		for (int yi=validRgn.top+idx.global[0]; yi<=validRgn.bottom; yi+=launchExt[0]) {
			for (int xi=validRgn.left+idx.global[1]; xi<=validRgn.right; xi+=launchExt[1]) {
				HessianState val;
				hessianTRZ_Point(val, I1View, wI2View, utils::Point(xi,yi), gradThresh, Xc, Yc, sigma_sq);
				init = init + val;
			}
		}

		// Reduce through LDS across wavefront:
		int lx = idx.local[0]*localW + idx.local[1];

		results0[lx] = init; 
		BARRIER(waveSize);

		REDUCE_STEP1(lx, 32);
		REDUCE_STEP1(lx, 16);
		REDUCE_STEP1(lx,  8);
		REDUCE_STEP1(lx,  4);
		REDUCE_STEP1(lx,  2);
		REDUCE_STEP1(lx,  1);

		// Save result of this tile to global mem
		if (lx == 0) {
			int gx = idx.tile[0] * computeUnits/*launchExt.get_num_groups()?*/ + idx.tile[1];
			results1[gx] = results0[0];
		};
	} );  //end parallel_for_each

	concurrency::copy(*entry._dBuffer, *entry._stagingBuffer);

	HessianState finalReduction = init;
	for (int i=0; i<results1.extent[0]; i++) {
		finalReduction = finalReduction +(*entry._stagingBuffer)[i];
	};

	arrayPoolAmp.free(entry);

	return checkChol(finalReduction, dH); // return -4.84
};




bool update_trz_amp_useworker( H3& dH, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi )
{
	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 );

	int wgPerComputeUnit =  p_wgPerComputeUnit; 
	int computeUnits     = p_computeUnits ? p_computeUnits : 10; // FIXME - determine from HSA Runtime

	int resultCnt = computeUnits * wgPerComputeUnit;
	static const int localH=8;  static const int localW=8;
	static const int waveSize  = localH*localW; // FIXME, read from device attributes.

	HessianState init;
	init.clearState();

	float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
	float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);
	float sigma_sq = sigma*sigma;

	concurrency::extent<2> launchExt(wgPerComputeUnit*localH, computeUnits*localW);
	concurrency::array<HessianState,1> results1(resultCnt);  // Output after reducing through LDS.

	concurrency::parallel_for_each(launchExt.tile<localH, localW>(), [=,&results1](concurrency::tiled_index<localH, localW> idx) mutable restrict(amp)
	{
		tile_static HessianState results0[waveSize]; 

		for (int yi=validRgn.top+idx.global[1]; yi<=validRgn.bottom; yi+=launchExt[1]) {
			for (int xi=validRgn.left+idx.global[0]; xi<=validRgn.right; xi+=launchExt[0]) {
				HessianState val;
				hessianTRZ_Point(val, I1View, wI2View, utils::Point(xi,yi), gradThresh, Xc, Yc, sigma_sq);
				init = init + val;
			}
		}

		// Reduce through LDS across wavefront:
		int lx = idx.local[1]*localW + idx.local[0];

		results0[lx] = init; 
		BARRIER(waveSize);

		REDUCE_STEP1(lx, 32);
		REDUCE_STEP1(lx, 16);
		REDUCE_STEP1(lx,  8);
		REDUCE_STEP1(lx,  4);
		REDUCE_STEP1(lx,  2);
		REDUCE_STEP1(lx,  1);

		// Save result of this tile to global mem
		if (lx == 0) {
			int gx = idx.tile[1] * wgPerComputeUnit/*launchExt.get_num_groups()?*/ + idx.tile[0];
			results1[gx] = results0[0];
		};
	} );  //end parallel_for_each

	//Copy partial array back to host - we'd like to use ZC memory for this final step...
	std::vector<HessianState> h_data(resultCnt);
	h_data = results1; 

	HessianState finalReduction = init;
	for (int i=0; i<results1.extent[0]; i++) {
		finalReduction = finalReduction + h_data[i];
	};

	return checkChol(finalReduction, dH); // return -4.84
};


bool checkChol(const HessianState &ws, H3 &dH) 
{

	double Hout[4][4];
	double bout[4];

	for (int i=0,k=0; i<4; i++)
	{
		for (int j=0; j<=i; j++) 
			Hout[i][j] = Hout[j][i] = (double)ws.Hx[k++];
	}

	for (int j = 0; j < 4; j++)
	{
		bout[j] = ws.Hx[10+j];
	}


	double dm[4];
	dH.zeros();

	if (solve_ChD<double,4> (dm, Hout, bout))
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




void printLOC_AMP() {
	const char *tag = "boltforamp";
	printf("\n");
	printLOC(tag, __FILE__, "kernel", g_boltamp_kernel_start, g_boltamp_kernel_end);
	printLOC(tag, __FILE__, "state", g_boltamp_state_begin, g_boltamp_state_end);
	printLOC(tag, __FILE__, "functor", g_boltamp_functor_begin, g_boltamp_functor_end);
	printLOC(tag, __FILE__, "launch", g_boltamp_launch_start, g_boltamp_launch_end);
};
