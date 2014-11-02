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

#include "hessian_amp.h"
#include "hessian_amp_range.h"
#include "hessian.h"

#define BOLT_POOL_ALLOC
#include <bolt/transform_reduce_range.h>

// Implementations for C++AMP, but where the functor takes a range rather than single point.
// Interface is simpliar to TBB, where user has to supply the code to iterate over the rang
//  Generally this is less appealing than interface which takes a single point and the iteration is all handled by the templates.
//  But moving the loop inside the user code appears necessary to get good performance with current versions of Visual Studio,
//  which don't optimally perform Return Value Optimization when a structuer is returned on the stack.

__forceinline inline void hessianTRZ_Range(	
	HessianState &w,
#if CPU_USE_ARRAY_VIEW
	matrix_type &I1, 
	matrix_type &wI2, 
#else
	const utils::Matrix<float> &I1,
	const utils::Matrix<float> &wI2,
#endif
	concurrency::index<2> topLeft,
	concurrency::index<2> bottomRight,  
	concurrency::extent<2> stride,
	float gradThresh,
	float Xc, float Yc,
	float sigma_sq )  restrict(amp, cpu) 
{
	float e0, e1, e2, e3;
	float  rcp_sigma_sq = 1.0f/(sigma_sq);

	for (int y=topLeft[0]; y<bottomRight[0]; y+=stride[0]) {
		for (int x=topLeft[1]; x<bottomRight[1]; x+=stride[1]) {

			float Ix = 0.5f*( wI2(y,   x+1) - wI2(y,   x-1) );
			float Iy = 0.5f*( wI2(y+1, x)   - wI2(y-1, x) );
			float It = wI2(y, x) - I1(y, x);

			int hasNan = mymath::isfNaN(It + Ix + Iy);
			Ix = hasNan ? 0.0f : Ix;
			It = hasNan ? 0.0f : It;
			Iy = hasNan ? 0.0f : Iy;

			bool cond3 =  (mymath::fabs(Ix) >= gradThresh) | (mymath::fabs(Iy) >= gradThresh) ;
			float weight = cond3 ? 1.0f : 0.0f;

			bool cond2 = cond3 && (sigma_sq != 0.0f );
			weight = cond2 ? 1.0f / (1.0f + rcp_sigma_sq*It*It) : weight;

			float X = (float)(x) - Xc;
			float Y = (float)(y) - Yc;

			e0 = Ix*X + Iy*Y;
			e1 = Ix*Y - Iy*X;
			e2 = Ix;
			e3 = Iy;	

			w.Hx[0] +=  weight * (e0 * e0); 
			w.Hx[1] +=  weight * (e1 * e0);
			w.Hx[2] +=  weight * (e1 * e1);
			w.Hx[3] +=  weight * (e2 * e0);
			w.Hx[4] +=  weight * (e2 * e1);
			w.Hx[5] +=  weight * (e2 * e2);
			w.Hx[6] +=  weight * (e3 * e0);
			w.Hx[7] +=  weight * (e3 * e1);
			w.Hx[8] +=  weight * (e3 * e2);
			w.Hx[9] +=  weight * (e3 * e3);
			w.Hx[10] += weight * (It * e0); 
			w.Hx[11] += weight * (It * e1); 
			w.Hx[12] += weight * (It * e2); 
			w.Hx[13] += weight * (It * e3); 
		};
	};
};


// Return W directly:
__forceinline inline HessianState hessianTRZ_Range2(	
	matrix_type &I1, 
	matrix_type &wI2, 
	concurrency::index<2> topLeft,
	concurrency::index<2> bottomRight,  
	concurrency::extent<2> stride,
	float gradThresh,
	float Xc, float Yc,
	float sigma_sq )  restrict(amp, cpu) 
{
	HessianState w;
	w.clearState();
	float e0, e1, e2, e3;
	float  rcp_sigma_sq = 1.0f/(sigma_sq);

	for (int y=topLeft[0]; y<bottomRight[0]; y+=stride[0]) {
		for (int x=topLeft[1]; x<bottomRight[1]; x+=stride[1]) {

			float Ix = 0.5f*( wI2(y,   x+1) - wI2(y,   x-1) );
			float Iy = 0.5f*( wI2(y+1, x)   - wI2(y-1, x) );
			float It = wI2(y, x) - I1(y, x);

			int hasNan = mymath::isfNaN(It + Ix + Iy);
			Ix = hasNan ? 0.0f : Ix;
			It = hasNan ? 0.0f : It;
			Iy = hasNan ? 0.0f : Iy;

			bool cond3 =  (mymath::fabs(Ix) >= gradThresh) | (mymath::fabs(Iy) >= gradThresh) ;
			float weight = cond3 ? 1.0f : 0.0f;

			bool cond2 = cond3 && (sigma_sq != 0.0f );
			weight = cond2 ? 1.0f / (1.0f + rcp_sigma_sq*It*It) : weight;

			float X = (float)(x) - Xc;
			float Y = (float)(y) - Yc;

			e0 = Ix*X + Iy*Y;
			e1 = Ix*Y - Iy*X;
			e2 = Ix;
			e3 = Iy;	

			w.Hx[0] +=  weight * (e0 * e0); 
			w.Hx[1] +=  weight * (e1 * e0);
			w.Hx[2] +=  weight * (e1 * e1);
			w.Hx[3] +=  weight * (e2 * e0);
			w.Hx[4] +=  weight * (e2 * e1);
			w.Hx[5] +=  weight * (e2 * e2);
			w.Hx[6] +=  weight * (e3 * e0);
			w.Hx[7] +=  weight * (e3 * e1);
			w.Hx[8] +=  weight * (e3 * e2);
			w.Hx[9] +=  weight * (e3 * e3);
			w.Hx[10] += weight * (It * e0); 
			w.Hx[11] += weight * (It * e1); 
			w.Hx[12] += weight * (It * e2); 
			w.Hx[13] += weight * (It * e3); 
		};
	};
	return w;
};


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


namespace AmpRange {
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

		HessianState operator() (concurrency::index<2> topLeft,
			concurrency::index<2> bottomRight,  
			concurrency::extent<2> stride) restrict(amp,cpu)
		{
			HessianState s;
			s.clearState();
			hessianTRZ_Range(s, _I1, _wI2, topLeft, bottomRight, stride, _gradThresh, _Xc, _Yc, _sigma_sq);
			return s;
		};

	public: // params
		matrix_type	_I1;
		matrix_type	_wI2;
		//private: // params
		float					_sigma_sq;
		float					_gradThresh;
		const  utils::Rect      _roi;
		float					_Xc;
		float					_Yc;
	};
};



//---
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>

template <typename T, typename F>
class TransformReduceTbbWrapper {
public:
	TransformReduceTbbWrapper(T init, F &transform_op) : 
		_initState(init), _transformState(init), _transform_op(transform_op)
	{
	};

	TransformReduceTbbWrapper( TransformReduceTbbWrapper& tw, tbb::split ) : 
		_initState(tw._initState), _transformState(tw._initState), _transform_op(tw._transform_op)
	{

	};

	void join(const TransformReduceTbbWrapper & tbbW ) 
	{
		_transformState = _transformState + tbbW._transformState;   // FIXME reduce_op here
	}


	// 2D Range version
	void operator()( const tbb::blocked_range2d<int>& r )
	{
		using namespace concurrency;

		//reduce op here
		_transformState = _transformState +  _transform_op(index<2>(r.rows().begin(), r.cols().begin()), // topLeft
			index<2>(r.rows().end(), r.cols().end()),  //bottomRight
			extent<2>(1,1));  // stride  

	}


	T _initState;
	T _transformState;
	F _transform_op;
};


bool update_trz_boltforamp_range_cpu( H3& dH, const utils::Matrix<float> &I1D, const utils::Matrix<float> &wI2D, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi )
{
	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 );
	AmpRange::HessianTransform w( I1View, wI2View, sigma, gradThresh, validRgn );

	HessianState init;
	init.clearState();

	TransformReduceTbbWrapper<HessianState, AmpRange::HessianTransform> ht(init, w);


	if (1) {
		// 2D version:
		tbb::parallel_reduce(tbb::blocked_range2d<int>( validRgn.top, validRgn.bottom+1, validRgn.left, validRgn.right+1), ht);

		return  checkChol(ht._transformState, dH); 
	}


	// Serial version:
	if (0) {
		using namespace concurrency;
		//printf ("running serial version\n");
		float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
		float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);

		utils::Rect r (validRgn.top, validRgn.left, validRgn.bottom+1, validRgn.right+1);

#if CPU_USE_ARRAY_VIEW
		//hessianTRZ_Range(ht._transformState, I1View, wI2View, index<2>(r.top, r.left), index<2> (r.bottom, r.right), extent<2>(1,1), gradThresh, Xc, Yc, sigma*sigma);
		// return 5.13
		ht._transformState.clearState();  // FIXME, shouldn't be necessary?
		ht(tbb::blocked_range2d<int>( r.top, r.bottom, r.left, r.right));
		return  checkChol(ht._transformState, dH); 
#else
		hessianTRZ_Range(init, I1D, wI2D, r, gradThresh, Xc, Yc, sigma*sigma);
		return  checkChol(init, dH);
#endif


		//
		//return  checkChol(init, dH);
	}

};


bool update_trz_boltforamp_range( H3& dH, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi )
{
	utils::Rect validRgn( roi.top+1, roi.left+1, roi.bottom-1, roi.right-1 );

	HessianState init;
	init.clearState();

	float Xc = 0.5f * (float)(validRgn.right + validRgn.left);
	float Yc = 0.5f * (float)(validRgn.bottom + validRgn.top);
	float sigma_sq = sigma*sigma;

	using namespace concurrency;
#if 1
	HessianState result = bolt::transform_reduce_range(index<2>(validRgn.top,validRgn.left), extent<2>(validRgn.height(), validRgn.width()),
		[=](index<2> topLeft, index<2> bottomRight, extent<2> stride) restrict(cpu,amp) ->HessianState {
			HessianState s;
			s.clearState();
			hessianTRZ_Range(s, I1View, wI2View, topLeft, bottomRight, stride, gradThresh, Xc, Yc, sigma_sq);
			return s;
	}, init, bolt::plus<HessianState>());
#endif


#if 0
	//std::cout << "boltforamp_range" << std::endl;
	HessianState result = bolt::transform_reduce_range(index<2>(validRgn.top,validRgn.left), extent<2>(validRgn.height(), validRgn.width()),
		[=](index<2> topLeft, index<2> bottomRight, extent<2> stride) restrict(cpu,amp) ->HessianState {
			HessianState s;
			for (int y=topLeft[0]; y<bottomRight[0]; y+=stride[0]) {
				for (int x=topLeft[1]; x<bottomRight[1]; x+=stride[1]) {
					utils::Point p(x,y);
					s = s + hessianTRZ_Point2( I1View, wI2View, p, gradThresh, Xc, Yc, sigma_sq);
				};
			};
			return s;
	}, init, bolt::plus<HessianState>());
#endif

#if 0
	HessianState result = bolt::transform_reduce_range(index<2>(validRgn.top,validRgn.left), extent<2>(validRgn.height(), validRgn.width()),
		[=](index<2> i) restrict(cpu,amp) ->HessianState {
			utils::Point p(i[1], i[0]);
			return hessianTRZ_Point2(I1View, wI2View, p, gradThresh, Xc, Yc, sigma_sq);
	}, init, bolt::plus<HessianState>());
#endif

	return checkChol(result, dH); // return -4.84

};