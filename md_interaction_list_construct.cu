#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

__device__ void addlink(const float* __restrict__ R, const uint  N, const uint * __restrict__ CIs, const uint idx, const uint nindx, const uint jindx, uint* __restrict__ IL, uint IonP, float aacut, uint &nkndx)
{
	uint i, jdx;
	float rr;
	float3 dr;
	for (i = 0; i < nindx; ++i)
	{
		jdx = CIs[jindx + i + N];
		dr.x = R[jdx] - R[idx];
		dr.y = R[jdx + N] - R[idx + N];
		dr.z = R[jdx + 2 * N] - R[idx + 2 * N];
		rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y) + __fmul_rn(dr.z, dr.z);
		//printf("I %i %i %u | %f\n", idx, jdx, jindx, sqrt(rr));
		if (rr < aacut && jdx != idx)
		{
			//printf("I %i %i %u | %f\n", idx, jdx, jindx, sqrt(rr));
			IL[idx * IonP + nkndx] = jdx;
			++nkndx;
		}
	}	
}

__global__ void d_ConstructInteractionListA(float* R, const uint IonP, const uint CN)
{
	
	// set thread ID
	//uint tid = threadIdx.x;
	// global index
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;//, iindx, nindx, jindx, jjindx, nkndx;
	if(idx==0) 
	{
		//printf("Error! d_ConstructInteractionListA %p\n", R);	
		printf("A %i %i %i %i\n", IonP, CN, CN+1, CN+2);	
	}
}

__global__ void d_ConstructInteractionList(const float* __restrict__ R, const uint N, const uint* __restrict__ CI, const uint* __restrict__ CIs, const uint* __restrict__ pnC, uint* __restrict__ IL,  
const uint IonP, const float a, const float _1d_a, const float aacut, const uint3 cN, const uint CN)
{
	
	// set thread ID
	//uint tid = threadIdx.x;
	// global index
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, iindx, nindx, jindx, jjindx, nkndx;
	//if(idx==0) 
	
		//printf("Error! d_ConstructInteractionList %p %p %p %p %p\n", R, CI, CIs, pnC, IL);	
		//printf("Error! d_ConstructInteractionList %u | %u %f  %f %f %u %u %u %u %u %u\n", N, IonP, a,  _1d_a, aacut, cN.x, cN.y, cN.z, CN);
		//printf("Error!  %u | %u %f  %f %f %u %u %u %u %u %u\n", CN);	
	
	uint3 indx;
	int3 indxx;
	//float rr;
	//float3 dr;
	///if(idx==0)printf("T %i %i | %f %f %f | %f %f %f \n", N, idx, A.x, A.y, A.z, L.x, L.y, L.z);
	// boundary check

while (idx < N)
	{
		
		indx.x = floorf(__fmul_rn(R[idx], _1d_a));
		indx.y = floorf(__fmul_rn(R[idx + N], _1d_a));
		indx.z = floorf(__fmul_rn(R[idx + 2 * N], _1d_a));
		iindx = indx.x + indx.y * cN.x + indx.z * cN.y * cN.x;
		indxx.x = copysignf(1.0f, (R[idx] - __fmul_rn(indx.x + 0.5f, a)));
		indxx.y = copysignf(1.0f, (R[idx + N] - __fmul_rn(indx.y + 0.5f, a)));
		indxx.z = copysignf(1.0f, (R[idx + 2 * N] - __fmul_rn(indx.z + 0.5f, a)));/**/
		//printf("CILf %i | %f %f %f %f | %u %u %u | %i %i %i\n", idx, R[idx], R[idx + N], R[idx + 2 * N], a, indx.x, indx.y, indx.z, indxx.x, indxx.y, indxx.z);
			
		nkndx = 0;
		jjindx = iindx;
		jindx = pnC[jjindx];
		nindx = pnC[jjindx + CN];
		//printf("pnC %i %u %u %u\n", idx, jindx, nindx, CIs[jindx + 0 + N]);
		//printf("Error! A %u %u %u %u %u\n", jjindx, iindx, jindx,nindx, pnC[jjindx]);
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);/**/
		//if(jjindx>=5625)printf("Error! B %u %u %u %u %u\n", jjindx, iindx, jindx,nindx, pnC[jjindx]);
		jjindx = iindx + indxx.y * cN.x;		
		
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		//if(jjindx>=5625)
		//printf("Error! C %u | %u %u %i\n", idx, iindx, cN.x, indxx.y * cN.x);
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.z * cN.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.z * cN.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.y * cN.x + indxx.z * cN.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.y * cN.x + indxx.z * cN.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);/**/
		//printf("T1 %u | %u %u %u %u %u %u %u %u | %u %u %u\n", idx, iindx, iindx + indxx.x, iindx + indxx.y * cN.x, iindx + indxx.z * cN.y * cN.x,
		//	iindx + indxx.x + indxx.y * cN.x, iindx + indxx.x + indxx.z * cN.y * cN.x, iindx + indxx.y * cN.x + indxx.z * cN.y * cN.x,
		//	iindx + indxx.x + indxx.y * cN.x + indxx.z * cN.y * cN.x, pnC[jjindx], pnC[iindx + CN], nkndx);
#ifdef pre_debugtest
		if (nkndx > IonP)
			printf("Error! d_ConstructInteractionList %i %u %u\n", idx, nkndx, IonP);
#endif // pre_debugtest

		
		//printf("T %i %u\n", idx, nkndx);
		idx += blockDim.x * gridDim.x;
	}
}



