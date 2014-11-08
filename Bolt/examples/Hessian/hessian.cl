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


float H_TRZ_val(int id, float Ix, float Iy, float It, float X, float Y)
{
	int i1 = (int) native_divide((1.f + native_sqrt(1.f + 8.f * ((float)id))), 2.f) - 1;

	// Mul e[row_id]
	float val = 0.0f;
	val = (i1 == 0) ? Ix * X + Iy * Y : val;
	val = (i1 == 1) ? Ix * Y - Iy * X : val;
	val = (i1 == 2) ? Ix : val;
	val = (i1 == 3) ? Iy : val;
	val = (i1 == 4) ? It : val;
	
	// Calculate column id
	i1 = id - ((i1 * (i1 + 1)) >> 1);
	
	// Mul e[column_id]
	val *= (i1 == 0) ? Ix * X + Iy * Y : 1;
	val *= (i1 == 1) ? Ix * Y - Iy * X : 1;
	val *= (i1 == 2) ? Ix : 1;
	val *= (i1 == 3) ? Iy : 1;
	
	return val;
}

__kernel void hessian_TRZ_x16(__global float* dst, 
							  __global float* I1, const int ipitch, 
							  __global float* WI2, const int wpitch, 
							  const int roiT, const int roiL, const int roiB, const int roiR,
							  const float Xc, const float Yc, const float thres, const float sigma_sq)
{
	int xi = roiL + get_global_id(0);
	int yi = roiT + get_global_id(1);
	int wind = wpitch*yi + xi;

	float Ix = 0.0f;
	float Iy = 0.0f;
	float It = 0.0f;

	bool cond0 = (xi > roiL && yi > roiT && xi < roiR && yi < roiB);

	Ix = cond0 ? native_divide((WI2[wind + 1] - WI2[wind - 1]), 2.0f) : Ix;
	Iy = cond0 ? native_divide((WI2[wind + wpitch] - WI2[wind - wpitch]), 2.0f) : Iy;
	It = cond0 ? I1[ipitch * yi + xi] - WI2[wind] : It;

	bool cond1 = (isnan(Ix + Iy + It));
	Ix = cond1 ? 0.0f : Ix;
	Iy = cond1 ? 0.0f : Iy;
	It = cond1 ? 0.0f : It;

	// Reduction for each warp
	int th_id = get_local_size(0) * get_local_id(1) + get_local_id(0); // thread id in the block

	int warp_base = (th_id >> 4) << 4;
	int th_off = th_id & 0xf;

	__local float sh_data[16*16];
	sh_data[th_id] = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);

	if ((fabs(Ix) >= thres) || (fabs(Iy) >= thres))
	{
		float weight = 1.0f; 
		bool cond2 = (sigma_sq != 0.0f);

		weight = cond2 ? 1.0f + native_divide(It*It, sigma_sq) : weight;	// (1+(It/sigma)^2)
		weight = cond2 ? native_divide(1.0f, (weight*weight)) : weight;		// 1 / (1+(It/sigma)^2)^2

		float X = (float)xi - Xc;
		float Y = (float)yi - Yc;

		for (int i=0, id; i<16; i++)
		{
			id = (i + th_off) & 0xf; // % 16
			sh_data[warp_base + id] += weight * H_TRZ_val(id, Ix, Iy, It, X, Y);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Inter warp reduction
	if (th_id < 128) sh_data[th_id] += sh_data[th_id + 128];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (th_id < 64) sh_data[th_id] += sh_data[th_id + 64];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (th_id < 32) sh_data[th_id] += sh_data[th_id + 32];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (th_id < 16) sh_data[th_id] += sh_data[th_id + 16];
	barrier(CLK_LOCAL_MEM_FENCE);

	int block_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);
	
	if (th_id < 14)
	{
		int ind_dst = block_id * 16 + th_id;
	    dst[ind_dst] = sh_data[th_id];
	}
}


#pragma OPENCL EXTENSION cl_amd_printf : enable

#define GS 64  // Group Size
#define HH 14  // height of structure to hold hess intermediates.
#define HESS2W 16

void reduceHess_4(int th_id, __local float *hess_temp2, __local float *hess_temp, int h)
// hess_temp contains 4 rows of 64 coeff 
// Perform a 4:1 reduction and move result to hess2 array
// Process 4 coefficients to keep the machine busy.
{
	barrier(CLK_LOCAL_MEM_FENCE);

    // First, reduce from 4 rows of 64 coeff to 4 rows of 32 coeff.
    // Each WI does two adds here.
    int tHi = (th_id >> 5) * 64; 
    int tLo = th_id & 0x1f;  
    hess_temp[tHi + tLo] += hess_temp[tHi + tLo + 32];
    //printf ("th_id=%d tHi=%d tLo=%d\n", th_id, tHi, tLo);
    tHi += 128;
    hess_temp[tHi + tLo] += hess_temp[tHi + tLo + 32];

	barrier(CLK_LOCAL_MEM_FENCE);

    // Now reduce again, each WI does one reduction and stores to hess_temp2 
    tHi = (th_id >> 4) * 64;
    tLo = th_id & 0x0f;  
    hess_temp2[h*HESS2W + th_id] =  hess_temp[tHi + tLo] + hess_temp[tHi + tLo + 16];
}


#define VW 2   // vector width
// Hessian:
// * kernel now contains a loop ; each loop iteration processes a single pixel and each work-item  now 
//   processes many pixels.  Typically we spend 100+ iterations in the loop. 
// * Use 14 registers to accumulate most of the hessian stats; use local mem and global mem
//   sparingly and only at the end for the final stages of reduction.
// * Inline formulas for each coefficient and remove use of H_TRZ_val function.
// * Kernel has same arguments and same alignment restrictions as original kernel.

// v2 version == vectorize by 2. Re-orders the ALU operations in the reduction which can lead to slightly different (but still correct) results.


__kernel 
__attribute__((reqd_work_group_size(LOCAL_W, LOCAL_H, 1)))
//__attribute__((coarse("l0S2")))
void hessian_TRZ_x16_optbs_v2(__global float* dst, 
                          __global const float* restrict I1, const int ipitch, 
                          __global const float* restrict WI2, const int wpitch, 
                          const int roiT, const int roiL, const int roiB, const int roiR,
                          const float Xc, const float Yc, const float thres, 
                          const float sigma_sq)
{
	int xi = roiL + get_global_id(0)*VW;
	int yi = roiT + get_global_id(1);

    // Reduction for each wave
    int th_id = get_local_size(0) * get_local_id(1) + get_local_id(0); // thread id in the block

    __local float hess_temp[4*GS];  
    __local float hess_temp2[16 * HESS2W];

    float2 hess00 =0;
    float2 hess01 =0;
    float2 hess02 =0;
    float2 hess03 =0;
    float2 hess04 =0;
    float2 hess05 =0;
    float2 hess06 =0;
    float2 hess07 =0;
    float2 hess08 =0;
    float2 hess09 =0;
    float2 hess10 =0;
    float2 hess11 =0;
    float2 hess12 =0;
    float2 hess13 =0;

    int xiBase = xi;
    while (yi < roiB) {
      while (xi < roiR) {
        {
            // Loop here is unrolled 2X to increase ALU packing and re-use of the load data.
            // On test systems this was a 1.6X performance improvement.
            const int wind = wpitch*yi + xi;

            float2 Ix = 0.0f;
            float2 Iy = 0.0f;
            float2 It = 0.0f;

            int2 cond0 = (int2)(((xi + 0)> roiL && yi > roiT && (xi + 0)< roiR && yi < roiB),
                                ((xi + 1)> roiL && yi > roiT && (xi + 1)< roiR && yi < roiB));

            float2 wi2_P1     = vload2(0, &WI2[wind + 1]);
            float2 wi2_M1     = vload2(0, &WI2[wind - 1]);
            float2 wi2        = vload2(0, &WI2[wind]);
            float2 wi2_Ppitch = vload2(0, &WI2[wind + wpitch]);
            float2 wi2_Mpitch = vload2(0, &WI2[wind - wpitch]);
            float2 i1_vals    = vload2(0, &I1[ipitch * yi + xi]);

            // Break out the case where cond0.s0 == cond0.s1 == 1, since this is most common.
            // This helps the compiler to group the loads into one clause.
            if (cond0.s0 & cond0.s1) {
                Ix = native_divide((wi2_P1- wi2_M1), 2.0f);
                Iy = native_divide((wi2_Ppitch- wi2_Mpitch), 2.0f);
                It = i1_vals- wi2;
            } else {
                Ix.s0 = cond0.s0 ? native_divide((wi2_P1.s0 - wi2_M1.s0), 2.0f) : Ix.s0;
                Iy.s0 = cond0.s0 ? native_divide((wi2_Ppitch.s0 - wi2_Mpitch.s0), 2.0f) : Iy.s0;
                It.s0 = cond0.s0 ? i1_vals.s0 - wi2.s0 : It.s0;
                Ix.s1 = cond0.s1 ? native_divide((wi2_P1.s1 - wi2_M1.s1), 2.0f) : Ix.s1;
                Iy.s1 = cond0.s1 ? native_divide((wi2_Ppitch.s1 - wi2_Mpitch.s1), 2.0f) : Iy.s1;
                It.s1 = cond0.s1 ? i1_vals.s1 - wi2.s1 : It.s1;
            }

            int2 cond1 = (isnan(Ix + Iy + It));
            Ix = cond1 ? 0.0f : Ix;
            Iy = cond1 ? 0.0f : Iy;
            It = cond1 ? 0.0f : It;


            int2 cond3 = (fabs(Ix) >= thres) | (fabs(Iy) >= thres);
            float2 weight = 1.0f; 
            int2 cond2 = cond3 && (sigma_sq != 0.0f);

            weight = cond2 ? 1.0f + native_divide(It*It, sigma_sq) : weight;	// (1+(It/sigma)^2)
            weight = cond2 ? native_divide((float2)(1.0f,1.0f), (weight*weight)) : weight;		// 1 / (1+(It/sigma)^2)^2

            float2 X; 
            X.s0 = (float)xi - Xc;
            X.s1 = (float)(xi+1) - Xc;
            const float2 Y = (float2)yi - Yc;

            const float2 e0 = Ix * X + Iy * Y;
            const float2 e1 = Ix * Y - Iy * X;
            const float2 e2 = Ix;
            const float2 e3 = Iy; 

            hess00 += weight * (e0 * e0);
            hess01 += weight * (e1 * e0);
            hess02 += weight * (e1 * e1);
            hess03 += weight * (e2 * e0);
            hess04 += weight * (e2 * e1);
            hess05 += weight * (e2 * e2);
            hess06 += weight * (e3 * e0);
            hess07 += weight * (e3 * e1);
            hess08 += weight * (e3 * e2);
            hess09 += weight * (e3 * e3);
            hess10 += weight * (It * e0);
            hess11 += weight * (It * e1);
            hess12 += weight * (It * e2);
            hess13 += weight * (It * e3);
        } // VW
        xi += VW*get_global_size(0);
      }
      xi = xiBase;

      yi += get_global_size(1);
	}

    hess_temp[ 0*GS+th_id] = hess00.s0 + hess00.s1;
    hess_temp[ 1*GS+th_id] = hess01.s0 + hess01.s1;
    hess_temp[ 2*GS+th_id] = hess02.s0 + hess02.s1;
    hess_temp[ 3*GS+th_id] = hess03.s0 + hess03.s1;
    reduceHess_4(th_id, hess_temp2, hess_temp, 0);

    hess_temp[ 0*GS+th_id] = hess04.s0 + hess04.s1;
    hess_temp[ 1*GS+th_id] = hess05.s0 + hess05.s1;
    hess_temp[ 2*GS+th_id] = hess06.s0 + hess06.s1;
    hess_temp[ 3*GS+th_id] = hess07.s0 + hess07.s1;
    reduceHess_4(th_id, hess_temp2, hess_temp, 4);

    hess_temp[ 0*GS+th_id] = hess08.s0 + hess08.s1;
    hess_temp[ 1*GS+th_id] = hess09.s0 + hess09.s1;
    hess_temp[ 2*GS+th_id] = hess10.s0 + hess10.s1;
    hess_temp[ 3*GS+th_id] = hess11.s0 + hess11.s1;
    reduceHess_4(th_id, hess_temp2, hess_temp, 8);

    hess_temp[ 0*GS+th_id] = hess12.s0 + hess12.s1;
    hess_temp[ 1*GS+th_id] = hess13.s0 + hess13.s1;
    reduceHess_4(th_id, hess_temp2, hess_temp, 12);

	int block_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);
    if (th_id < 14) {
        float acc = 0.0f;
        for (int j=0; j<HESS2W; j++) {
            int index = (th_id + j) & (HESS2W-1); // avoid bank conflicts.
            acc += hess_temp2[th_id*HESS2W+index];
        }

		int ind_dst = block_id * 16 + th_id;
	    dst[ind_dst] = acc;
    }
}


//__attribute__((coarse("l0S2")))
//-------------------------------------------------------------------
__kernel 
__attribute__((reqd_work_group_size(LOCAL_W, LOCAL_H, 1)))
void hessian_TRZ_x16_optbs_scalar(__global float* dst, 
                          __global const float* restrict I1, const int ipitch, 
                          __global const float* restrict WI2, const int wpitch, 
                          int roiT, int roiL, int roiB, int roiR,
                          float Xc, float Yc, 
						  float thres, 
                          float sigma_sq)
{
    int th_id = get_local_size(0) * get_local_id(1) + get_local_id(0); // thread id in the block

    __local float hess_temp[4*GS];  
    __local float hess_temp2[16 * HESS2W];

    float hess00 =0;
    float hess01 =0;
    float hess02 =0;
    float hess03 =0;
    float hess04 =0;
    float hess05 =0;
    float hess06 =0;
    float hess07 =0;
    float hess08 =0;
    float hess09 =0;
    float hess10 =0;
    float hess11 =0;
    float hess12 =0;
    float hess13 =0;

	for (int yi=roiT + get_global_id(1); yi <=roiB; yi+=get_global_size(1)) {
	    for (int xi=roiL + get_global_id(0); xi <= roiR; xi+=get_global_size(0)) {
            int wind = wpitch*yi + xi;

            float Ix =  native_divide((WI2[wind + 1] - WI2[wind - 1]), 2.0f) ;
            float Iy =  native_divide((WI2[wind + wpitch] - WI2[wind - wpitch]), 2.0f) ;
            float It =  I1[ipitch * yi + xi] - WI2[wind] ;

            bool cond1 = (isnan(Ix + Iy + It));
            Ix = cond1 ? 0.0f : Ix;
            Iy = cond1 ? 0.0f : Iy;
            It = cond1 ? 0.0f : It;

            //---
            bool cond3 = ((fabs(Ix) >= thres) | (fabs(Iy) >= thres));

            float weight = cond3 ? 1.0f : 0.0f; 
            bool cond2 = cond3 && (sigma_sq != 0.0f);

            weight = cond2 ? 1.0f + native_divide(It*It, sigma_sq) : weight;	// (1+(It/sigma)^2)
            weight = cond2 ? native_divide(1.0f, (weight*weight)) : weight;		// 1 / (1+(It/sigma)^2)^2

            float X = (float)xi - Xc;
            float Y = (float)yi - Yc;

            float e0 = Ix * X + Iy * Y;
            float e1 = Ix * Y - Iy * X;
            float e2 = Ix; // FIXME
            float e3 = Iy;  // FIXME

            hess00 += weight * (e0 * e0);
            hess01 += weight * (e1 * e0);
            hess02 += weight * (e1 * e1);
            hess03 += weight * (e2 * e0);
            hess04 += weight * (e2 * e1);
            hess05 += weight * (e2 * e2);
            hess06 += weight * (e3 * e0);
            hess07 += weight * (e3 * e1);
            hess08 += weight * (e3 * e2);
            hess09 += weight * (e3 * e3);
            hess10 += weight * (It * e0);
            hess11 += weight * (It * e1);
            hess12 += weight * (It * e2);
            hess13 += weight * (It * e3);
        } 
	}

    hess_temp[ 0*GS+th_id] = hess00;
    hess_temp[ 1*GS+th_id] = hess01;
    hess_temp[ 2*GS+th_id] = hess02;
    hess_temp[ 3*GS+th_id] = hess03;
    reduceHess_4(th_id, hess_temp2, hess_temp, 0);

    hess_temp[ 0*GS+th_id] = hess04;
    hess_temp[ 1*GS+th_id] = hess05;
    hess_temp[ 2*GS+th_id] = hess06;
    hess_temp[ 3*GS+th_id] = hess07;
    reduceHess_4(th_id, hess_temp2, hess_temp, 4);

    hess_temp[ 0*GS+th_id] = hess08;
    hess_temp[ 1*GS+th_id] = hess09;
    hess_temp[ 2*GS+th_id] = hess10;
    hess_temp[ 3*GS+th_id] = hess11;
    reduceHess_4(th_id, hess_temp2, hess_temp, 8);

    hess_temp[ 0*GS+th_id] = hess12;
    hess_temp[ 1*GS+th_id] = hess13;
    reduceHess_4(th_id, hess_temp2, hess_temp, 12);

	int block_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);
    if (th_id < 14) {
        float acc = 0.0f;
        for (int j=0; j<HESS2W; j++) {
            int index = (th_id + j) & (HESS2W-1); // avoid bank conflicts.
            acc += hess_temp2[th_id*HESS2W+index];
        }

		int ind_dst = block_id * 16 + th_id;
	    dst[ind_dst] = acc;
    }
}




class HessianState {
    float hess00;
    float hess01;
    float hess02;
    float hess03;
    float hess04;
    float hess05;
    float hess06;
    float hess07;
    float hess08;
    float hess09;
    float hess10;
    float hess11;
    float hess12;
    float hess13;
};



#define BARRIER(N) 
#define LDS_IDX(HESS, TH_ID) ((HESS)*GS + (TH_ID))

// perform one step of the reduction, across 14 elements:
void reduceStepMatch(int th_id, int w, __local float *lds_data)
{
    if (th_id < w) {
        for (int h=0; h<14; h++) {
            lds_data[LDS_IDX(h, th_id)] += lds_data[LDS_IDX(h, (th_id+w))];
        }
    }
    BARRIER(64);
};

// Match the C++AMP implementation style.
// Changes are in the handling of the final steps of the reduction, which is done in a single
// work-item rather than being spread across multiple WI as in the optimal implementation.
__kernel 
__attribute__((reqd_work_group_size(LOCAL_W, LOCAL_H, 1)))
void hessian_TRZ_x16_optbs_scalar_match(__global float* dst, 
                          __global const float* restrict I1, const int ipitch, 
                          __global const float* restrict WI2, const int wpitch, 
                          int roiT, int roiL, int roiB, int roiR,
                          float Xc, float Yc, 
						  float thres, 
                          float sigma_sq)
{
    int th_id = get_local_size(0) * get_local_id(1) + get_local_id(0); // thread id in the block

    __local float lds_data[14*GS];  

    float hess00 =0;
    float hess01 =0;
    float hess02 =0;
    float hess03 =0;
    float hess04 =0;
    float hess05 =0;
    float hess06 =0;
    float hess07 =0;
    float hess08 =0;
    float hess09 =0;
    float hess10 =0;
    float hess11 =0;
    float hess12 =0;
    float hess13 =0;

	for (int yi=roiT + get_global_id(1); yi <=roiB; yi+=get_global_size(1)) {
	    for (int xi=roiL + get_global_id(0); xi <= roiR; xi+=get_global_size(0)) {
            int wind = wpitch*yi + xi;

            float Ix =  native_divide((WI2[wind + 1] - WI2[wind - 1]), 2.0f) ;
            float Iy =  native_divide((WI2[wind + wpitch] - WI2[wind - wpitch]), 2.0f) ;
            float It =  I1[ipitch * yi + xi] - WI2[wind] ;

            bool cond1 = (isnan(Ix + Iy + It));
            Ix = cond1 ? 0.0f : Ix;
            Iy = cond1 ? 0.0f : Iy;
            It = cond1 ? 0.0f : It;

            bool cond3 = ((fabs(Ix) >= thres) | (fabs(Iy) >= thres));

            float weight = cond3 ? 1.0f : 0.0f; 
            bool cond2 = cond3 && (sigma_sq != 0.0f);

            weight = cond2 ? 1.0f + native_divide(It*It, sigma_sq) : weight;	// (1+(It/sigma)^2)
            weight = cond2 ? native_divide(1.0f, (weight*weight)) : weight;		// 1 / (1+(It/sigma)^2)^2

            float X = (float)xi - Xc;
            float Y = (float)yi - Yc;

            float e0 = Ix * X + Iy * Y;
            float e1 = Ix * Y - Iy * X;
            float e2 = Ix; // FIXME
            float e3 = Iy;  // FIXME

            hess00 += weight * (e0 * e0);
            hess01 += weight * (e1 * e0);
            hess02 += weight * (e1 * e1);
            hess03 += weight * (e2 * e0);
            hess04 += weight * (e2 * e1);
            hess05 += weight * (e2 * e2);
            hess06 += weight * (e3 * e0);
            hess07 += weight * (e3 * e1);
            hess08 += weight * (e3 * e2);
            hess09 += weight * (e3 * e3);
            hess10 += weight * (It * e0);
            hess11 += weight * (It * e1);
            hess12 += weight * (It * e2);
            hess13 += weight * (It * e3);
        } ; 
	}

    lds_data[LDS_IDX( 0,th_id)] = hess00;
    lds_data[LDS_IDX( 1,th_id)] = hess01;
    lds_data[LDS_IDX( 2,th_id)] = hess02;
    lds_data[LDS_IDX( 3,th_id)] = hess03;
    lds_data[LDS_IDX( 4,th_id)] = hess04;
    lds_data[LDS_IDX( 5,th_id)] = hess05;
    lds_data[LDS_IDX( 6,th_id)] = hess06;
    lds_data[LDS_IDX( 7,th_id)] = hess07;
    lds_data[LDS_IDX( 8,th_id)] = hess08;
    lds_data[LDS_IDX( 9,th_id)] = hess09;
    lds_data[LDS_IDX(10,th_id)] = hess10;
    lds_data[LDS_IDX(11,th_id)] = hess11;
    lds_data[LDS_IDX(12,th_id)] = hess12;
    lds_data[LDS_IDX(13,th_id)] = hess13;

    BARRIER(64);

    reduceStepMatch(th_id, 32, lds_data);
    reduceStepMatch(th_id, 16, lds_data);
    reduceStepMatch(th_id,  8, lds_data);
    reduceStepMatch(th_id,  4, lds_data);
    reduceStepMatch(th_id,  2, lds_data);
    reduceStepMatch(th_id,  1, lds_data);

    // reduced 14-element structure in LDS, need to store to global mem

    if (th_id == 0) {
        int block_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        int ind_dst = block_id * 16 ;
        for (int h=0;h<14;h++) {
            dst[ind_dst+h] = lds_data[LDS_IDX(h,0)];
        }
    };
}


// Another try to match C++AMP : 
// Use small fixed-size arrays rather than hardcoded registers.
__kernel 
__attribute__((reqd_work_group_size(LOCAL_W, LOCAL_H, 1)))
void hessian_TRZ_x16_optbs_scalar_match2(__global float* dst, 
                          __global const float* restrict I1, const int ipitch, 
                          __global const float* restrict WI2, const int wpitch, 
                          int roiT, int roiL, int roiB, int roiR,
                          float Xc, float Yc, 
						  float thres, 
                          float sigma_sq)
{
    int th_id = get_local_size(0) * get_local_id(1) + get_local_id(0); // thread id in the block

    __local float lds_data[14*GS];  

    __private float hess[HH];

    for (int i=0; i<HH; i++) {
        hess[i] = 0.0f;
    }

	for (int yi=roiT + get_global_id(1); yi <=roiB; yi+=get_global_size(1)) {
	    for (int xi=roiL + get_global_id(0); xi <= roiR; xi+=get_global_size(0)) {
            int wind = wpitch*yi + xi;

            float Ix =  native_divide((WI2[wind + 1] - WI2[wind - 1]), 2.0f) ;
            float Iy =  native_divide((WI2[wind + wpitch] - WI2[wind - wpitch]), 2.0f) ;
            float It =  I1[ipitch * yi + xi] - WI2[wind] ;

            bool cond1 = (isnan(Ix + Iy + It));
            Ix = cond1 ? 0.0f : Ix;
            Iy = cond1 ? 0.0f : Iy;
            It = cond1 ? 0.0f : It;

            bool cond3 = ((fabs(Ix) >= thres) | (fabs(Iy) >= thres));
            float weight = cond3 ? 1.0f : 0.0f; 

            bool cond2 = cond3 && (sigma_sq != 0.0f);
            weight = cond2 ? 1.0f + native_divide(It*It, sigma_sq) : weight;	// (1+(It/sigma)^2)
            weight = cond2 ? native_divide(1.0f, (weight*weight)) : weight;		// 1 / (1+(It/sigma)^2)^2

            float X = (float)xi - Xc;
            float Y = (float)yi - Yc;

            float e0 = Ix * X + Iy * Y;
            float e1 = Ix * Y - Iy * X;
            float e2 = Ix; 
            float e3 = Iy;  

            hess[ 0] += weight * (e0 * e0);
            hess[ 1] += weight * (e1 * e0);
            hess[ 2] += weight * (e1 * e1);
            hess[ 3] += weight * (e2 * e0);
            hess[ 4] += weight * (e2 * e1);
            hess[ 5] += weight * (e2 * e2);
            hess[ 6] += weight * (e3 * e0);
            hess[ 7] += weight * (e3 * e1);
            hess[ 8] += weight * (e3 * e2);
            hess[ 9] += weight * (e3 * e3);
            hess[10] += weight * (It * e0);
            hess[11] += weight * (It * e1);
            hess[12] += weight * (It * e2);
            hess[13] += weight * (It * e3);
        } 
	}

    for (int i=0; i<HH; i++) {
        lds_data[LDS_IDX( i,th_id)] = hess[ i];
    }

    BARRIER(64);

    reduceStepMatch(th_id, 32, lds_data);
    reduceStepMatch(th_id, 16, lds_data);
    reduceStepMatch(th_id,  8, lds_data);
    reduceStepMatch(th_id,  4, lds_data);
    reduceStepMatch(th_id,  2, lds_data);
    reduceStepMatch(th_id,  1, lds_data);

    // reduced 14-element structure in LDS, need to store to global mem
    if (th_id == 0) {
        int block_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        int ind_dst = block_id * 16 ;
        for (int h=0;h<14;h++) {
            dst[ind_dst+h] = lds_data[LDS_IDX(h,0)];
        }
    };
}


// Use array, but inline all indices.  Runs faster than match2 for some reason...
__kernel 
__attribute__((reqd_work_group_size(LOCAL_W, LOCAL_H, 1)))
void hessian_TRZ_x16_optbs_scalar_match3(__global float* dst, 
                          __global const float* restrict I1, const int ipitch, 
                          __global const float* restrict WI2, const int wpitch, 
                          int roiT, int roiL, int roiB, int roiR,
                          float Xc, float Yc, 
						  float thres, 
                          float sigma_sq)
{
	int xi = roiL + get_global_id(0);
	int yi = roiT + get_global_id(1);

    // Reduction for each wave
    int th_id = get_local_size(0) * get_local_id(1) + get_local_id(0); // thread id in the block

    __local float lds_data[14*GS];  

    __private float hess[HH];

#if 0
#pragma unroll HH
    for (int i=0; i<HH; i++) {
        hess[i] = 0.0f;
    }
#else
    hess[ 0] = 0.0f;
    hess[ 1] = 0.0f;
    hess[ 2] = 0.0f;
    hess[ 3] = 0.0f;
    hess[ 4] = 0.0f;
    hess[ 5] = 0.0f;
    hess[ 6] = 0.0f;
    hess[ 7] = 0.0f;
    hess[ 8] = 0.0f;
    hess[ 9] = 0.0f;
    hess[10] = 0.0f;
    hess[11] = 0.0f;
    hess[12] = 0.0f;
    hess[13] = 0.0f;
#endif

	for (int yi=roiT + get_global_id(1); yi <=roiB; yi+=get_global_size(1)) {
	    for (int xi=roiL + get_global_id(0); xi <= roiR; xi+=get_global_size(0)) {
            int wind = wpitch*yi + xi;

            float Ix =  native_divide((WI2[wind + 1] - WI2[wind - 1]), 2.0f) ;
            float Iy =  native_divide((WI2[wind + wpitch] - WI2[wind - wpitch]), 2.0f) ;
            float It =  I1[ipitch * yi + xi] - WI2[wind] ;

            bool cond1 = (isnan(Ix + Iy + It));
            Ix = cond1 ? 0.0f : Ix;
            Iy = cond1 ? 0.0f : Iy;
            It = cond1 ? 0.0f : It;

            //---
            bool cond3 = ((fabs(Ix) >= thres) | (fabs(Iy) >= thres));

            float weight = cond3 ? 1.0f : 0.0f; 
            bool cond2 = cond3 && (sigma_sq != 0.0f);

            weight = cond2 ? 1.0f + native_divide(It*It, sigma_sq) : weight;	// (1+(It/sigma)^2)
            weight = cond2 ? native_divide(1.0f, (weight*weight)) : weight;		// 1 / (1+(It/sigma)^2)^2

            float X = (float)xi - Xc;
            float Y = (float)yi - Yc;

            float e0 = Ix * X + Iy * Y;
            float e1 = Ix * Y - Iy * X;
            float e2 = Ix; // FIXME
            float e3 = Iy;  // FIXME

            hess[ 0] += weight * (e0 * e0);
            hess[ 1] += weight * (e1 * e0);
            hess[ 2] += weight * (e1 * e1);
            hess[ 3] += weight * (e2 * e0);
            hess[ 4] += weight * (e2 * e1);
            hess[ 5] += weight * (e2 * e2);
            hess[ 6] += weight * (e3 * e0);
            hess[ 7] += weight * (e3 * e1);
            hess[ 8] += weight * (e3 * e2);
            hess[ 9] += weight * (e3 * e3);
            hess[10] += weight * (It * e0);
            hess[11] += weight * (It * e1);
            hess[12] += weight * (It * e2);
            hess[13] += weight * (It * e3);
        } 
	}

#if 0
#pragma unroll HH
    for (int i=0; i<HH; i++) {
        lds_data[LDS_IDX( i,th_id)] = hess[ i];
    }
#else
    lds_data[LDS_IDX( 0,th_id)] = hess[ 0];
    lds_data[LDS_IDX( 1,th_id)] = hess[ 1];
    lds_data[LDS_IDX( 2,th_id)] = hess[ 2];
    lds_data[LDS_IDX( 3,th_id)] = hess[ 3];
    lds_data[LDS_IDX( 4,th_id)] = hess[ 4];
    lds_data[LDS_IDX( 5,th_id)] = hess[ 5];
    lds_data[LDS_IDX( 6,th_id)] = hess[ 6];
    lds_data[LDS_IDX( 7,th_id)] = hess[ 7];
    lds_data[LDS_IDX( 8,th_id)] = hess[ 8];
    lds_data[LDS_IDX( 9,th_id)] = hess[ 9];
    lds_data[LDS_IDX(10,th_id)] = hess[10];
    lds_data[LDS_IDX(11,th_id)] = hess[11];
    lds_data[LDS_IDX(12,th_id)] = hess[12];
    lds_data[LDS_IDX(13,th_id)] = hess[13];
#endif

    BARRIER(64);

    reduceStepMatch(th_id, 32, lds_data);
    reduceStepMatch(th_id, 16, lds_data);
    reduceStepMatch(th_id,  8, lds_data);
    reduceStepMatch(th_id,  4, lds_data);
    reduceStepMatch(th_id,  2, lds_data);
    reduceStepMatch(th_id,  1, lds_data);

    // reduced 14-element structure in LDS, need to store to global mem

    if (th_id == 0) {
        int block_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        int ind_dst = block_id * 16 ;
        for (int h=0;h<14;h++) {
            dst[ind_dst+h] = lds_data[LDS_IDX(h,0)];
        }
    };
}



// src is produced by hessian_TRZ_x16_optbs kernel.
// in memory it contains 16 sequential values padded with 2 zeros, ie: 
//   H0, H1, .., H13, 0, 0 
// We sum ComputeUnit*12 items for each coefficient - for Llano, that is only 60 sums/coefficient; for cypress 240.  
// So we don't do anything fancy here since we are probably mem bw-limited anyway - make sure we only read each line once
// Best to Launch with a single-work-group, size=16x16 so the coalescing lines up well.
__kernel 
void hessian_global_reduce(__global float* restrict src, int srcSizeElems, __global float * dst14)
{
    int groupId = get_group_id(0);
    int gid = get_global_id(0);

    if (gid < 16) {
        int th_off = get_local_id(0) & 0xf;
        int ht = gid;
        float acc=0; 
        int i = 0;
        while (i+th_off<srcSizeElems) {
            acc += src[i+th_off];
            i += 16;
        }

        dst14[ht] = acc;
    }
}
