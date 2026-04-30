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


__global__ void d_CalculateIncrements(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const uint N, const float _1d_Mass_m_dt, const float dt)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	while (idx < N)
	{		
		V[idx] += __fmul_rn(F[idx], _1d_Mass_m_dt);
		V[idx + N] += __fmul_rn(F[idx + N], _1d_Mass_m_dt);
		V[idx + 2 * N] += __fmul_rn(F[idx + 2 * N], _1d_Mass_m_dt);

		R[idx] += __fmul_rn(V[idx], dt);
		R[idx + N] += __fmul_rn(V[idx + N], dt);
		R[idx + 2 * N] += __fmul_rn(V[idx + 2 * N], dt);
		idx += blockDim.x * gridDim.x;
	}	
}

__global__ void d_CalculateIncrementsViscos(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const uint N, const float _1d_Mass_m_dt, const float dt, const float vis)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	while (idx < N)
	{
		V[idx] += __fmul_rn(F[idx] - V[idx] * vis, _1d_Mass_m_dt);
		V[idx + N] += __fmul_rn(F[idx + N] - V[idx + N] * vis, _1d_Mass_m_dt);
		V[idx + 2 * N] += __fmul_rn(F[idx + 2 * N] - V[idx + 2 * N] * vis, _1d_Mass_m_dt);

		R[idx] += __fmul_rn(V[idx], dt);
		R[idx + N] += __fmul_rn(V[idx + N], dt);
		R[idx + 2 * N] += __fmul_rn(V[idx + 2 * N], dt);
		idx += blockDim.x * gridDim.x;
	}
}