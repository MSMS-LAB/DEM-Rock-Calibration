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

__global__ void d_ReduceParticleNumber(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, const uint N, const uint imaxI, const uint* __restrict__ IL,  const uint IonP, const float3 hp)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, jdx;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	//float3 dr;// , sign;
	//float shift = 1.2;// hm, dm, sm, coefr, coefh;
	if (idx == imaxI)
	{
		jdx = IL[imaxI];
		R[jdx] = hp.x;
		R[jdx + N] = hp.y;
		R[jdx + 2 * N] = hp.z;
		V[jdx] = 0.0F;
		V[jdx + N] = 0.0F;
		V[jdx + 2 * N] = 0.0F;
		F[jdx] = 0.0F;
		F[jdx + N] = 0.0F;
		F[jdx + 2 * N] = 0.0F;
		//if (idx == 938)printf("Error Cut! %i %f %f %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N], hp.z);
	}
}