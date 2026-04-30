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

/*__device__ void addlink(const float* __restrict__ R, const uint N, const uint* __restrict__ CIs, const uint idx, const uint nindx, const uint jindx, uint* __restrict__ IL, uint IonP, float aacut, uint& nkndx)
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
		
		if (rr < aacut && rr > 1e-10)
		{
			//printf("I %i %i %u | %f\n", idx, jdx, jindx, sqrt(rr));
			IL[idx * IonP + nkndx] = jdx;
			++nkndx;
		}
	}
}/**/

__global__ void d_ConstructBoundInteractions(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd, 
	const uint N, const uint IonP, const float aa_create, const float b_r)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, jdx, kdx, ii;
	float3 r;
	float rm2;
	//if(idx==0)printf("II  \n");
	while (idx < N)//Change to parallel by interactions
	{
		ii = 0;
		kdx = idx * IonP;
		jdx = IL[kdx];
		//type = ILtype[kdx];
		//fs.x = 0; fs.y = 0; fs.z = 0; ii = 0;
		//while (jdx < N && ii < IonP)
		//printf("II %u %u %u %u |  \n", kdx, idx, jdx, IonP);
		for(ii = 0; ii < IonP;++ii)
		{
			if (jdx != UINT_MAX)
			{
				r.x = R[jdx] - R[idx];
				r.y = R[jdx + N] - R[idx + N];
				r.z = R[jdx + 2 * N] - R[idx + 2 * N];
				rm2 = r.x * r.x + r.y * r.y + r.z * r.z;
				//printf("II %u %u %u | %e %e %e\n", kdx, idx, jdx, rm2, aa_create, r.z);
				if (rm2 < aa_create && rm2 > 1e-18)
				{
					//b_r[kdx] = b_r;
					float _1d_rm = __frsqrt_rn(rm2);
					_1d_iL[kdx] = _1d_rm;
					//AxialMoment[kdx] = MCf_pi * b_r * b_r * b_r * b_r * 0.25f;
					ILtype[kdx] = 1;
					Rij[kdx].x = r.x * _1d_rm;
					Rij[kdx].y = r.y * _1d_rm;
					Rij[kdx].z = r.z * _1d_rm;
					//printf("II %u %u %u | %e %e \n", kdx, idx, jdx, rm2, b_r);
				}
				else
				{
					ILtype[kdx] = 0;
				}
				Rij[kdx].x = 0*r.x;
				Rij[kdx].y = 0*r.y;
				Rij[kdx].z = 0*r.z;
				Oijt[kdx].x = 0;
				Oijt[kdx].y = 0;
				Oijt[kdx].z = 0;
				Fijn[kdx].x = 0;
				Fijn[kdx].y = 0;
				Fijn[kdx].z = 0;
				Fijt[kdx].x = 0;
				Fijt[kdx].y = 0;
				Fijt[kdx].z = 0;
				Mijn[kdx].x = 0;
				Mijn[kdx].y = 0;
				Mijn[kdx].z = 0;
				Mijt[kdx].x = 0;
				Mijt[kdx].y = 0;
				Mijt[kdx].z = 0;
				Mijadd[kdx].x = 0;
				Mijadd[kdx].y = 0;
				Mijadd[kdx].z = 0;
			}
			//++ii;
			++kdx;
			jdx = IL[kdx];
		}
		idx += blockDim.x * gridDim.x;
	}
}



