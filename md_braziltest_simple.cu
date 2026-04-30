#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include "md_definedparams.h"
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <time.h>

//#include <cudpp.h>
//#include <cudpp_plan.h>

__global__ void d_braziltest_simple(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint N, const float3 c, const float RR, const float Yt, const float Yb, const float Ytr, const float Ybr, const float vt, const float Zcut)
{
	__shared__ float s_mem[2 * SMEMDIM];
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x;	
	//uint3 cm;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 r;
	float sfb = 0, sft = 0;
	while (idx < N)
	{
		r.x = R[idx] - c.x;
		r.y = R[idx + N];
		r.z = R[idx + 2 * N] - c.z;
		//printf("I0n %u %e | %e | %e %e %e\n", idx, A.z, B.z, hp.x, hp.y, hp.z);
		if (r.x * r.x + r.z * r.z < RR && r.z < Zcut)
		{
			if (r.y - Ytr > -1e-9)
			{				
				
				float fy = F[idx + N];
				if (fy > 0)
				{
					sft += fy;
				}
				if (r.y - Yt > -1e-9)
				{
					R[idx + N] = Yt;
					//V[idx] = 0;
					//V[idx + N] = 0;
					V[idx + N] = vt;					
					F[idx + N] = 0;
				}
				//printf("Uft %u %e %e %e\n", idx, (Hd2 - r.z) * C, fz, R[idx + 2 * N]);
			}
				
			if (r.y - Ybr < 1e-9)
			{
				float fy = F[idx + N];
				if (fy < 0)
				{
					sfb += fy;
				}			
				if (r.y - Yb < 1e-9)
				{					
					R[idx + N] = Yb;
					//V[idx] = 0;
					//V[idx + N] = 0;
					V[idx + N] = 0;					
					F[idx + N] = 0;
					//printf("Ufb %u %e %e | %e %e %e %e\n", idx, (Hd2 + r.z) * C, fz, R[idx + 2 * N], c.z, r.z, Hd2);
				}
				//printf("Ufb %u %e %e | %e %e %e %e\n", idx, (Hd2 + r.z) * C, fz, R[idx + 2 * N], c.z, r.z, Hd2);
			}				
		}
		//printf("I0n %u %e %e %e\n", idx, R[idx], R[idx + N], R[idx + 2 * N]);
		idx += blockDim.x * gridDim.x;
	}
	s_mem[tid] = sfb;
	s_mem[tid + SMEMDIM] = sft;
	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512)
	{
		s_mem[tid] += s_mem[tid + 512];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 512];		
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
	{
		//printf("Blok!\n");
		s_mem[tid] += s_mem[tid + 256];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 256];		
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
	{
		s_mem[tid] += s_mem[tid + 128];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 128];		
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
	{
		s_mem[tid] += s_mem[tid + 64];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 64];		
	}
	__syncthreads();
	
	// unrolling warp
	if (tid < 32)
	{
		volatile float* vsmem = s_mem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 32];		
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 16];		
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 8];		
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 4];		
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 2];		
		vsmem[tid] += vsmem[tid + 1];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 1];		
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		//printf("U %u %u %u\n", idx, blockIdx.x, blockIdx.x + gridDim.x);
		FL[blockIdx.x] = s_mem[0];
		FL[blockIdx.x + gridDim.x] = s_mem[SMEMDIM];		
	}
}