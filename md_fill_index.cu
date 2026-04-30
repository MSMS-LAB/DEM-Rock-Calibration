#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"


__global__ void d_FillIndex(uint* __restrict__ CI, const uint N)
{
	// set thread ID
	//uint tid = threadIdx.x;
	// global index
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if(idx==0)printf("T %i %i | %f %f %f | %f %f %f \n", N, idx, A.x, A.y, A.z, L.x, L.y, L.z);
	// boundary check
	if (idx < N)
	{
		CI[idx] = idx;
		//printf("T %i \n", idx);
	}
}



