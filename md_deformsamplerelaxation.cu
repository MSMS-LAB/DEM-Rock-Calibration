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


__global__ void d_DeformSampleRelaxationY(float* __restrict__ R, const uint N, const float3 center, const float Eps, const float Shift)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	
	while (idx < N)
	{
		float dry;
		//dr.x = R[idx] - center.x;
		//dr.y = R[idx + N] - center.y;
		dry = R[idx + N] - center.y;
		R[idx + N] = center.y + dry*(1.0f+Eps) + Shift;
		//R[idx + 2 * N] += Shift;
		idx += blockDim.x * gridDim.x;
	}	
}

__global__ void d_DeformSampleRelaxationZ(float* __restrict__ R, const uint N, const float3 center, const float Eps, const float Shift)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	
	while (idx < N)
	{
		float drz;
		//dr.x = R[idx] - center.x;
		//dr.y = R[idx + N] - center.y;
		drz = R[idx + 2 * N] - center.z;
		R[idx + 2 * N] = center.z + drz*(1.0f+Eps) + Shift;
		//R[idx + 2 * N] += Shift;
		idx += blockDim.x * gridDim.x;
	}	
}

