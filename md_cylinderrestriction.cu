#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <time.h>

//#include <cudpp.h>
//#include <cudpp_plan.h>


__global__ void d_CylinderRestriction(float* __restrict__ V, float* __restrict__ R, const uint N, const float3 axis, const float3 center, const float R0, const float H0)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr, s;
	float hm, sm, coefr, coefh, coefs=1.0f;
	while (idx < N)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;

		/*d.x = __fmul_rn(dr.y, axis.z) - __fmul_rn(dr.z, axis.y);
		d.y = __fmul_rn(dr.z, axis.x) - __fmul_rn(dr.x, axis.z);
		d.x = __fmul_rn(dr.x, axis.y) - __fmul_rn(dr.y, axis.z);
		dm = __fsqrt_rn(__fmul_rn(d.x, d.x) + __fmul_rn(d.y, d.y) + __fmul_rn(d.z, d.z));/**/
		hm = __fmul_rn(dr.x, axis.x) + __fmul_rn(dr.y, axis.y) + __fmul_rn(dr.z, axis.z);
		s.x = dr.x - __fmul_rn(hm, axis.x);
		s.y = dr.y - __fmul_rn(hm, axis.y);
		s.z = dr.z - __fmul_rn(hm, axis.z);
		sm = __fsqrt_rn(__fmul_rn(s.x, s.x) + __fmul_rn(s.y, s.y) + __fmul_rn(s.z, s.z));
		coefr = 0.0f;
		if (sm > R0)
		{
			coefr = __fmul_rn(R0, __frcp_rn(sm)) - coefs;
			//V[idx] *= -1.0f;
			//V[idx + N] *= -1.0f;
		}		
		coefh = 0.0f;
		if (hm > H0)
		{
			coefh = (H0 - hm) * coefs;
			//V[idx + 2 * N] *= -1.0f;
		} 
		else if (hm < -H0)
		{
			coefh = -(H0 - hm) * coefs;
			//V[idx + 2 * N] *= -1.0f;
		}
		//printf("C %i | %f %f %f | %f %f %f | %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N]);
		//printf("C %i | %f %f %f | %f %f %f | %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N], s.x, s.y, s.z);
		//printf("C %i | %f %f %f | %f %f %f | %f %f\n", idx, dr.x, dr.y, dr.z, dm, sm, hm, coefr, coefh);
		R[idx] += __fmul_rn(s.x, coefr) + __fmul_rn(axis.x, coefh);
		R[idx + N] += __fmul_rn(s.y, coefr) + __fmul_rn(axis.y, coefh);
		R[idx + 2 * N] += __fmul_rn(s.z, coefr) + __fmul_rn(axis.z, coefh);
		idx += blockDim.x * gridDim.x;
	}	
}

__global__ void d_CylinderRestrictionZ(float* __restrict__ V, float* __restrict__ R, const uint N, const float3 center, const float R0, const float H0, const float HideZ)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr, s;
	float rr, hm, sm, coefr, coefh, coefs = 1.0f;
	while (idx < N)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;
		s.x = 0;
		s.y = 0;
		s.z = 0;
		rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y);
		if(dr.z < HideZ- center.z)
		{
				if (rr > R0 * R0)
				{
					rr = __frsqrt_rn(rr);
					s.x = dr.x * (R0 * rr - 1.0f);
					s.y = dr.y * (R0 * rr - 1.0f);
					V[idx] *= -1.0f;
					V[idx + N] *= -1.0f;
					//printf("In %u %e %e %e %e %e\n", idx, dr.x, dr.y, R0, s.x, s.y);
				}
				if (dr.z > H0)
				{
					s.z = H0 - dr.z;
					V[idx + 2 * N] *= -1.0f;
				}
				if (dr.z < -H0)
				{
					s.z = -H0 - dr.z;
					V[idx + 2 * N] *= -1.0f;
				}
				R[idx] += s.x;
				R[idx + N] += s.y;
				R[idx + 2 * N] += s.z;
		}
				
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_CylinderBorderPush(float* __restrict__ Fcoeff, const float Fcoeffborder, float* __restrict__ R, const uint N, const float3 center, const float R0, const float H0)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr, s;
	float rr, hm, sm, coefr, coefh, coefs = 1.0f;
	while (idx < N && Fcoeff[idx]>Fcoeffborder)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;
		s.x = 0;
		s.y = 0;
		s.z = 0;
		rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y);
		if (rr - R0 * R0>-1e-8)
		{			
			rr = __frsqrt_rn(rr);
			s.x = dr.x * (0.8f * R0 * rr - 1.0f);
			s.y = dr.y * (0.8f * R0 * rr - 1.0f);
			//printf("In %u %e %e %e %e %e\n", idx, dr.x, dr.y, R0, s.x, s.y);
		}
		if (dr.z - H0 > -1e-8)
		{
			s.z = 0.8f * H0 - dr.z;
			
		}
		if (dr.z + H0 < 1e-8)
		{
			s.z = -0.8f * H0 - dr.z;
		}
		R[idx] += s.x;
		R[idx + N] += s.y;
		R[idx + 2 * N] += s.z;
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_CylinderRestrictionZr(float* __restrict__ V, float* __restrict__ R, const float* __restrict__ dR, const uint N, const float3 center, const float R0, const float H0, const float HideZ)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("d_CylinderRestrictionZr\n");
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr, s;
	float rr, hm, sm, coefr, coefh, coefs = 1.0f, R1 = R0+dR[idx], H1 = H0+dR[idx];
	while (idx < N)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;
		s.x = 0;
		s.y = 0;
		s.z = 0;
		rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y);
		if(dr.z < HideZ- center.z)
		{
				if (rr > R1 * R1)
				{
					rr = __frsqrt_rn(rr);
					s.x = dr.x * (R1 * rr - 1.0f);
					s.y = dr.y * (R1 * rr - 1.0f);
					V[idx] *= -1.0f;
					V[idx + N] *= -1.0f;
					//printf("In %u %e %e %e %e %e\n", idx, dr.x, dr.y, R1, s.x, s.y);
				}
				if (dr.z > H1)
				{
					s.z = H1 - dr.z;
					V[idx + 2 * N] *= -1.0f;
				}
				if (dr.z < -H1)
				{
					s.z = -H1 - dr.z;
					V[idx + 2 * N] *= -1.0f;
				}
				R[idx] += s.x;
				R[idx + N] += s.y;
				R[idx + 2 * N] += s.z;
		}
				
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_CylinderRestrictionZr2(float* __restrict__ V, float* __restrict__ R, const float* __restrict__ dR, const uint N, const float3 center, const float R0, const float H0, const float3 Hide)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("d_CylinderRestrictionZr\n");
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr, s;
	float rr, hm, sm, coefr, coefh, coefs = 1.0f, R1 = R0+dR[idx], H1 = H0+dR[idx];
	while (idx < N)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;
		s.x = 0;
		s.y = 0;
		s.z = 0;
		rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y);
		if(dr.z < Hide.z - center.z)
		{
				if (rr > R1 * R1)
				{
					rr = __frsqrt_rn(rr);
					s.x = dr.x * (R1 * rr - 1.0f);
					s.y = dr.y * (R1 * rr - 1.0f);
					V[idx] *= -1.0f;
					V[idx + N] *= -1.0f;
					//printf("In %u %e %e %e %e %e\n", idx, dr.x, dr.y, R1, s.x, s.y);
				}
				if (dr.z > H1)
				{
					s.z = H1 - dr.z;
					V[idx + 2 * N] *= -1.0f;
				}
				if (dr.z < -H1)
				{
					s.z = -H1 - dr.z;
					V[idx + 2 * N] *= -1.0f;
				}
				R[idx] += s.x;
				R[idx + N] += s.y;
				R[idx + 2 * N] += s.z;
		} else
		{
			R[idx] = Hide.x;
			R[idx + N] = Hide.y;
			R[idx + 2 * N] = Hide.z;
		}
				
		idx += blockDim.x * gridDim.x;
	}
}