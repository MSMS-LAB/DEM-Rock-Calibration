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

//__constant__ char IM[2 * IMatrixSize];

__global__ void d_CalculateForcesLJ(const float* __restrict__ R, float* __restrict__ F, uint N, const uint* __restrict__ IL, const uint IonP, const float D, const float a2, const float _1d_a2)
{
	float r2, a2_d_r2, a4_d_r4, c, fm;
	float3 dr, fs;
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, jdx, ii;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	//if(idx==0)
	while (idx < N)
	{
		jdx = IL[idx * IonP];
		fs.x = 0; fs.y = 0; fs.z = 0; ii = 0;
		while (jdx < N)
		{
			dr.x = R[jdx] - R[idx];
			dr.y = R[jdx + N] - R[idx + N];
			dr.z = R[jdx + 2 * N] - R[idx + 2 * N];
			r2 = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y) + __fmul_rn(dr.z, dr.z);
			a2_d_r2 = a2*__frcp_rn(r2);
			a4_d_r4 = __fmul_rn(a2_d_r2, a2_d_r2);
			c = __fmul_rn(12.0f, __fmul_rn(D, _1d_a2));
			fm = c * (a4_d_r4 * a2_d_r2 - 1.0f) * a4_d_r4 * a2_d_r2;
			//fm = __fmul_rn(__fmul_rn(D, _1d_r), __fmul_rn(_1d_r, _1d_r));
			fs.x -= __fmul_rn(fm, dr.x);
			fs.y -= __fmul_rn(fm, dr.y);
			fs.z -= __fmul_rn(fm, dr.z);
			//printf("FI %u %u %f\n", idx, jdx, fm);
			++ii;
			jdx = IL[idx * IonP + ii];
		}
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		
		F[idx] = fs.x;
		F[idx + N] = fs.y;
		F[idx + 2 * N] = fs.z;		
		idx += blockDim.x * gridDim.x;
	}	
}
