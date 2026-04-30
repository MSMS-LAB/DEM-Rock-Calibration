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

__device__ void addnewlinkbpm(const float* __restrict__ R, const uint &N, const uint* __restrict__ CIs, const uint &idx, const uint &nindx, const uint &jindx, uint* __restrict__ IL, uint* __restrict__ ILtype, uint &IonP, float &aacut, uint& nkndx)
{
	uint i, j, jdx, kjdx, kdxfree;
	float rr;
	float3 dr;
	for (i = 0; i < nindx; ++i)
	{
		jdx = CIs[jindx + i + N];
		dr.x = R[jdx] - R[idx];
		dr.y = R[jdx + N] - R[idx + N];
		dr.z = R[jdx + 2 * N] - R[idx + 2 * N];
		rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y) + __fmul_rn(dr.z, dr.z);
		
		if (rr < aacut && rr > 1e-10)
		{	
			kdxfree = 0;
			for (j = 0; j < IonP; ++j)
			{
				kjdx = IL[idx * IonP + j];
				if (kdxfree == 0 && kjdx == UINT_MAX)
					kdxfree = idx * IonP + j;
				if (kjdx == jdx)
					break;				
			}
			if (j == IonP)
			{
				//printf("I %i %i %u | %f\n", idx, jdx, jindx, sqrt(rr));
				IL[kdxfree] = jdx;
				ILtype[kdxfree] = 0;
				//++nkndx;
			}				
		}
	}
}

__device__ uint deletelinksbpm(const float* __restrict__ R, const uint &N, const uint &idx, uint* __restrict__ IL, uint &IonP, float &aacut)
{
	uint nkndx = 0, jdx;
	float rr;
	float3 dr;
	jdx = IL[idx * IonP];
	//while (jdx < UINT_MAX && nkndx < IonP)
	for (nkndx = 0; nkndx < IonP; ++nkndx)
	{	
		jdx = IL[idx * IonP + nkndx];
		if (jdx != UINT_MAX)
		{
			dr.x = R[jdx] - R[idx];
			dr.y = R[jdx + N] - R[idx + N];
			dr.z = R[jdx + 2 * N] - R[idx + 2 * N];
			rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y) + __fmul_rn(dr.z, dr.z);
			/*if (rr < aacut)
			{

				jdx = IL[idx * IonP + nkndx];
				//printf("I %i %i %u | %f\n", idx, jdx, jindx, sqrt(rr));
			}/**/
			if (rr > aacut)
			{
				IL[idx * IonP + nkndx] = UINT_MAX;
				//jdx = UINT_MAX;
			}
		}		
	}
	return nkndx;	
}

__global__ void d_ReConstructInteractionListbpm(const float* __restrict__ R, uint N, const uint* __restrict__ CI, const uint* __restrict__ CIs, const uint* __restrict__ pnC, uint* __restrict__ IL, uint* __restrict__ ILtype, uint IonP, float a, float _1d_a, float aacut, uint3 cN, uint CN)
{
	//set thread ID
	//uint tid = threadIdx.x;
	// global index
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, iindx, nindx, jindx, jjindx, nkndx;
	uint3 indx;
	int3 indxx;
	//float rr;
	//float3 dr;
	//if(idx==0)printf("T %i %i | %f %f %f | %f %f %f \n", N, idx, A.x, A.y, A.z, L.x, L.y, L.z);
	// boundary check
	//if(idx==0)
	while (idx < N)
	{

		indx.x = floorf(__fmul_rn(R[idx], _1d_a));
		indx.y = floorf(__fmul_rn(R[idx + N], _1d_a));
		indx.z = floorf(__fmul_rn(R[idx + 2 * N], _1d_a));
		iindx = indx.x + indx.y * cN.x + indx.z * cN.y * cN.x;
		indxx.x = copysignf(1.0f, (R[idx] - __fmul_rn(indx.x + 0.5f, a)));
		indxx.y = copysignf(1.0f, R[idx + N] - __fmul_rn(indx.y + 0.5f, a));
		indxx.z = copysignf(1.0f, R[idx + 2 * N] - __fmul_rn(indx.z + 0.5f, a));
		//printf("T %i %i | %f %f %f %f | %u %u %u | %i %i %i\n", tid, idx, R[idx], R[idx + N], R[idx + 2 * N], a, indx.x, indx.y, indx.z, indxx.x, indxx.y, indxx.z);
		//if (iindx >= CN)printf("Error CN! %i %u %u\n", idx, iindx, CN);
		//nkndx = deletelinksbpm(R, N, idx, IL, IonP, aacut);
		//printf("T0 %u | %u %u\n", idx, N, nkndx);
		jjindx = iindx;
		jindx = pnC[jjindx];
		nindx = pnC[jjindx + CN];
		//printf("T1 %u | %u %u %u %u %u %u %u %u | %u\n", CN, iindx, iindx + indxx.x, iindx + indxx.y * cN.x, iindx + indxx.z * cN.y * cN.x, 
		//	iindx + indxx.x + indxx.y * cN.x, iindx + indxx.x + indxx.z * cN.y * cN.x, iindx + indxx.y * cN.x + indxx.z * cN.y * cN.x,
		//	iindx + indxx.x + indxx.y * cN.x + indxx.z * cN.y * cN.x, pnC[iindx + CN]);
		addnewlinkbpm(R, N, CIs, idx, nindx, jindx, IL, ILtype, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x; //if(jjindx>=CN)printf("Error CNj0! %i %u %u\n", idx, jjindx, CN);
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addnewlinkbpm(R, N, CIs, idx, nindx, jindx, IL, ILtype, IonP, aacut, nkndx);
		jjindx = iindx + indxx.y * cN.x; //if (jjindx >= CN)printf("Error CNj1! %i %u %u\n", idx, jjindx, CN);
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addnewlinkbpm(R, N, CIs, idx, nindx, jindx, IL, ILtype, IonP, aacut, nkndx);
		jjindx = iindx + indxx.z * cN.y * cN.x; //if (jjindx >= CN)printf("Error CNj2! %i %u %u %u %u %u %u | %f\n", idx, jjindx, indx.x, indx.y, indx.z, indxx.z, CN, R[idx + 2 * N]);
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addnewlinkbpm(R, N, CIs, idx, nindx, jindx, IL, ILtype, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.y * cN.x; //if (jjindx >= CN)printf("Error CNj3! %i %u %u\n", idx, jjindx, CN);
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addnewlinkbpm(R, N, CIs, idx, nindx, jindx, IL, ILtype, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.z * cN.y * cN.x; //if (jjindx >= CN)printf("Error CNj4! %i %u %u\n", idx, jjindx, CN);
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addnewlinkbpm(R, N, CIs, idx, nindx, jindx, IL, ILtype, IonP, aacut, nkndx);
		jjindx = iindx + indxx.y * cN.x + indxx.z * cN.y * cN.x; //if (jjindx >= CN)printf("Error CNj5! %i %u %u\n", idx, jjindx, CN);
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addnewlinkbpm(R, N, CIs, idx, nindx, jindx, IL, ILtype, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.y * cN.x + indxx.z * cN.y * cN.x; //if (jjindx >= CN)printf("Error CNj6! %i %u %u\n", idx, jjindx, CN);
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addnewlinkbpm(R, N, CIs, idx, nindx, jindx, IL, ILtype, IonP, aacut, nkndx);
		//if(nkndx >= IonP)printf("Error IonP! %i %u %u\n", idx, nkndx, IonP);
		//printf("T %i %u\n", idx, nkndx);
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_DeleteFarLinks(const float* __restrict__ R, uint N, uint* __restrict__ IL, uint* __restrict__ ILtype, float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd, uint IonP, const float aacut)
{
	//set thread ID
	//uint tid = threadIdx.x;
	// global index
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//float rr;
	//float3 dr;
	//if(idx==0)printf("T %i %i | %f %f %f | %f %f %f \n", N, idx, A.x, A.y, A.z, L.x, L.y, L.z);
	// boundary check
	//if(idx==0)
	while (idx < N)
	{
		//printf("T %i %i | %f %f %f %f | %u %u %u | %i %i %i\n", tid, idx, R[idx], R[idx + N], R[idx + 2 * N], a, indx.x, indx.y, indx.z, indxx.x, indxx.y, indxx.z);
		uint jdx, kdx;
		float rr;
		float3 dr;
		for (uint nkndx = 0; nkndx < IonP; ++nkndx)
		{
			kdx = idx * IonP + nkndx;
			jdx = IL[kdx];
			if (jdx != UINT_MAX)
			{
				dr.x = R[jdx] - R[idx];
				dr.y = R[jdx + N] - R[idx + N];
				dr.z = R[jdx + 2 * N] - R[idx + 2 * N];
				rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y) + __fmul_rn(dr.z, dr.z);
				if (rr > aacut || rr < 1e-10)
				{
					IL[kdx] = UINT_MAX;
					ILtype[kdx] = 0;
					_1d_iL = 0;
					Rij[kdx].x = 0; Rij[kdx].y = 0; Rij[kdx].z = 0;
					Oijt[kdx].x = 0; Oijt[kdx].y = 0; Oijt[kdx].z = 0;
					Fijn[kdx].x = 0; Fijn[kdx].y = 0; Fijn[kdx].z = 0;
					Fijt[kdx].x = 0; Fijt[kdx].y = 0; Fijt[kdx].z = 0;
					Mijn[kdx].x = 0; Mijn[kdx].y = 0; Mijn[kdx].z = 0;
					Mijt[kdx].x = 0; Mijt[kdx].y = 0; Mijt[kdx].z = 0;
					Mijadd[kdx].x = 0; Mijadd[kdx].y = 0; Mijadd[kdx].z = 0;
				}
			}
		}
		idx += blockDim.x * gridDim.x;
	}
}

