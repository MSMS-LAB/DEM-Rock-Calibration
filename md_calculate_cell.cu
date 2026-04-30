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

__global__ void d_CalculateCellIndex(const float* __restrict__ R, uint N, uint* __restrict__ CI, const float _1d_a, const uint3 cN)
{
	// set thread ID
	//uint tid = threadIdx.x;
	// global index
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, iindx;
	uint3 indx;
	//if(idx==0)printf("T %i %i | %f \n", N, idx, _1d_a);
	// boundary check
	if (idx < N)
	{
		indx.x = floorf(__fmul_rn(R[idx], _1d_a));
		indx.y = floorf(__fmul_rn(R[idx + N], _1d_a));
		indx.z = floorf(__fmul_rn(R[idx + 2 * N], _1d_a));
#ifdef pre_debugtest
		if (indx.x ==0 )printf("ERRROR! d_CalculateCellIndex X %u %u | %f %f\n", idx, indx.x, R[idx], _1d_a);
		if (indx.y ==0 )printf("ERRROR! d_CalculateCellIndex Y %u %u | %f %f\n", idx, indx.y, R[idx + N], _1d_a);
		if (indx.z ==0 )printf("ERRROR! d_CalculateCellIndex Z %u %u | %f %f\n", idx, indx.z, R[idx + 2 * N], _1d_a);
		if (indx.x > cN.x-1)printf("ERRROR! d_CalculateCellIndex X %u %u | %f %f\n", idx, indx.x, R[idx], _1d_a);
		if (indx.y > cN.y-1)printf("ERRROR! d_CalculateCellIndex Y %u %u | %f %f\n", idx, indx.y, R[idx + N], _1d_a);
		if (indx.z > cN.z-1)printf("ERRROR! d_CalculateCellIndex Z %u %u | %f %f\n", idx, indx.z, R[idx + 2 * N], _1d_a);
#endif // pre_debugtest
		iindx = indx.x + indx.y * cN.x + indx.z * cN.y * cN.x;
		CI[idx] = iindx;
		//CIs[idx] = iindx;
		//CIs[idx + N] = idx;
		//printf("T %i | %f %f %f %f | %u %u %u %u\n", idx, R[idx], R[idx + N], R[idx + 2 * N], _1d_a, indx.x, indx.y, indx.z, iindx);
	}
}

__global__ void d_DetermineCellPointer(const uint* __restrict__ CIs, uint* __restrict__ pnC, const uint N, const uint CN)
{
	// set thread ID
	//uint tid = threadIdx.x;
	// global index
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, iindx, nindx;
	//uint3 indx;
	//if(idx==0)printf("T %i %i | %f %f %f | %f %f %f \n", N, idx, A.x, A.y, A.z, L.x, L.y, L.z);
	// boundary check
	if(idx == 0)
	{
		iindx = CIs[idx];
		pnC[iindx] = idx;
		nindx = 1;
		while (CIs[idx + nindx] == iindx) ++nindx;		
		pnC[iindx + CN] = nindx;
	} else	if (idx < N)
	{
		iindx = CIs[idx];
		//printf("T %u %u %u | %u %u\n", tid, idx, iindx, pnC[iindx], pnC[iindx + CN]);
		if (CIs[idx - 1] != iindx)
		{
			pnC[iindx] = idx;
			nindx = 1;
			while (CIs[idx + nindx] == iindx)
			{
				++nindx;
			}
			pnC[iindx + CN] = nindx;
			//printf("C %u %u %u | %u %u\n", tid, idx, iindx, pnC[iindx], pnC[iindx + CN]);
		}				
	}
}



