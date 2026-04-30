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


__global__ void d_ParallelepipedRestriction(float* __restrict__ V, float* __restrict__ R, const uint N, const float3 center, const float3 sized2)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr;// , sign;
	float shift = 1.0f;// hm, dm, sm, coefr, coefh;
	if (idx < N)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;
		
		if (dr.x > sized2.x)
		{
			R[idx] -= (dr.x - sized2.x * shift);
			V[idx] *= -1.0f;
		}		
		else if (dr.x < -sized2.x)
		{
			R[idx] -= (dr.x + sized2.x * shift);
			V[idx] *= -1.0f;
		}
			

		if (dr.y > sized2.y)
		{
			R[idx + N] -= (dr.y - sized2.y * shift);
			V[idx + N] *= -1.0f;
		}			
		else if (dr.y < -sized2.y)
		{
			R[idx + N] -= (dr.y + sized2.y * shift);
			V[idx + N] *= -1.0f;
		}
			

		if (dr.z > sized2.z)
		{
			R[idx + 2 * N] -= (dr.z - sized2.z * shift);
			V[idx + 2 * N] *= -1.0f;
		}
			
		else if (dr.z < -sized2.z)
		{
			R[idx + 2 * N] -= (dr.z + sized2.z * shift);
			V[idx + 2 * N] *= -1.0f;
		}
			
		//printf("In %u %e %e %e\n", idx, R[idx], R[idx + N], R[idx + 2 * N]);

	}	
}

__global__ void d_ParallelepipedCutRestriction(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, const uint N, const float3 center, const float3 sized2, const float3 hp)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr;// , sign;
	float shift = 1.2;// hm, dm, sm, coefr, coefh;
	if (idx < N)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;

		if (dr.x - sized2.x > 0)
		{
			R[idx] = hp.x;
			R[idx + N] = hp.y;
			R[idx + 2 * N] = hp.z;
			V[idx] = 0.0F;
			V[idx + N] = 0.0F;
			V[idx + 2 * N] = 0.0F;
			F[idx] = 0.0F;
			F[idx + N] = 0.0F;
			F[idx + 2 * N] = 0.0F;
		}
		if (dr.x + sized2.x < 0)
		{
			R[idx] = hp.x;
			R[idx + N] = hp.y;
			R[idx + 2 * N] = hp.z;
			V[idx] = 0.0F;
			V[idx + N] = 0.0F;
			V[idx + 2 * N] = 0.0F;
			F[idx] = 0.0F;
			F[idx + N] = 0.0F;
			F[idx + 2 * N] = 0.0F;
		}
		if (dr.y - sized2.y > 0)
		{
			R[idx] = hp.x;
			R[idx + N] = hp.y;
			R[idx + 2 * N] = hp.z;
			V[idx] = 0.0F;
			V[idx + N] = 0.0F;
			V[idx + 2 * N] = 0.0F;
			F[idx] = 0.0F;
			F[idx + N] = 0.0F;
			F[idx + 2 * N] = 0.0F;
		}
		if (dr.y + sized2.y < 0)
		{
			R[idx] = hp.x;
			R[idx + N] = hp.y;
			R[idx + 2 * N] = hp.z;
			V[idx] = 0.0F;
			V[idx + N] = 0.0F;
			V[idx + 2 * N] = 0.0F;
			F[idx] = 0.0F;
			F[idx + N] = 0.0F;
			F[idx + 2 * N] = 0.0F;
		}
		if (dr.z - sized2.z > 0)
		{
			R[idx] = hp.x;
			R[idx + N] = hp.y;
			R[idx + 2 * N] = hp.z;
			V[idx] = 0.0F;
			V[idx + N] = 0.0F;
			V[idx + 2 * N] = 0.0F;
			F[idx] = 0.0F;
			F[idx + N] = 0.0F;
			F[idx + 2 * N] = 0.0F;
		}
		if (dr.z + sized2.z < 0)
		{
			R[idx] = hp.x;
			R[idx + N] = hp.y;
			R[idx + 2 * N] = hp.z;
			V[idx] = 0.0F;
			V[idx + N] = 0.0F;
			V[idx + 2 * N] = 0.0F;
			F[idx] = 0.0F;
			F[idx + N] = 0.0F;
			F[idx + 2 * N] = 0.0F;
		}
		//if (idx == 938)printf("Error Cut! %i %f %f %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N], hp.z);

	}
}