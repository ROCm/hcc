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

#include <amp.h>
#include <amp_math.h>

#include "matrix_utils.h"
#include <bolt/synchronized_view.h>

//typedef const concurrency::array_view< float, 2> matrix_type;
typedef const bolt::synchronized_view<float, 2>  matrix_type;

#define CPU_USE_ARRAY_VIEW 1       //0=use utils::Matrix, 1=array_view, 2=synchronized_view
#if CPU_USE_ARRAY_VIEW
typedef matrix_type matrix_type_bolt_cpu ;
#else
typedef const utils::Matrix<float> matrix_type_bolt_cpu;
#endif

static const int g_boltamp_state_begin = __LINE__+1;
static const int HX_CNT=14;
class HessianState {
public:
	// Use for reduction - sum together elements of two WorkerStates:
	HessianState operator+(const HessianState &w) const restrict(amp,cpu) {
		HessianState w2;
		for (int i=0;i<HX_CNT;i++) {
			w2.Hx[i] = this->Hx[i] + w.Hx[i];
		};
		return w2;
	};

	void clearState() restrict(cpu,amp) {
		for (int i=0;i<HX_CNT;i++) 
			Hx[i] = 0;
	};

	float Hx[HX_CNT];
};
static const int g_boltamp_state_end = __LINE__-1;





namespace mymath {

	//FIXME - use builtin functions?
	inline int isfNaN(float f) restrict(amp) {
		//return *(long*)&f == 0x7fffffff;
		return concurrency::fast_math::isnan(f);
	};
	inline int isfNaN(float f) restrict(cpu) {
		return *(long*)&f == 0x7fffffff;
	};



	inline float fabs(float x)  restrict(cpu,amp) {
		return (x>=0.0f) ? x:-x; // FIXME - need bit AND hack here.
	};
};


extern bool checkChol(const HessianState &ws, H3 &dH) ;
extern bool update_trz_boltforamp( H3& dH, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi );
extern bool update_trz_boltforamp_lambda( H3& dH, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi );
extern bool update_trz_boltforamp_cpu( H3& dH, const utils::Matrix<float> &I1, const utils::Matrix<float> &wI2, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi );
extern bool update_trz_amp( H3& dH, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi );
extern bool update_trz_amp_useworker( H3& dH, matrix_type& I1View,  matrix_type& wI2View, float sigma, float gradThresh, const  utils::Rect& roi );
extern void printLOC_AMP();
