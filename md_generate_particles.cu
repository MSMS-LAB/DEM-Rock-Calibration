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
__global__ void d_SetParticlesInParallelepiped(float* __restrict__ R, float* __restrict__ V, uint N, const float3 c, const float3 L, const float vm)
{
	// set thread ID
	//uint tid = threadIdx.x;
	// global index
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if(idx==0)printf("T %i %i | %f %f %f | %f %f %f \n", N, idx, A.x, A.y, A.z, L.x, L.y, L.z);
	// boundary check
	while (idx < N)
	{
		V[idx] = vm * (R[idx] - 0.5);
		V[idx + N] = vm * (R[idx + N] - 0.5);
		V[idx + 2 * N] = vm * (R[idx + 2 * N] - 0.5);
		R[idx] = c.x + (R[idx] - 0.5) * L.x;
		R[idx + N] = c.y + (R[idx + N] - 0.5) * L.y;
		R[idx + 2 * N] = c.z + (R[idx + 2 * N] - 0.5) * L.z;

		
		//printf("T %i %i | %f %f %f\n", tid, idx, R[idx], R[idx + N], R[idx + 2 * N]);
		/*if (R[idx] < A.x || (R[idx] - A.x) > L.x)
			printf("T %u | %f %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N]);
		if (R[idx + N] < A.y || (R[idx + N] - A.y) > L.y)
			printf("T %u | %f %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N]);
		if (R[idx + 2 * N] < A.z || (R[idx + 2 * N] - A.z) > L.z)				
			printf("T %u | %f %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 *N]);/**/
		idx += blockDim.x * gridDim.x;
	}
}

void generateParticles(particle_data& P, additional_data& A, sample_data& S)
{	
	A.bloks = ceil(P.N / (SMEMDIM)) + 1;
	//std::cerr << int(A.gencreated) << "\n";
	std::cerr << "Generate partiicles " << A.bloks << " N=" << P.N << " " << 5 * 3 * P.N * sizeof(float) + P.N * sizeof(float4) << " " << (5 * 3 * P.N * sizeof(float) + P.N * sizeof(float4)) / (1024 * 1024) << "\n";
	if (A.gencreated == 0)	
	{
		curandCreateGenerator(&A.gen, CURAND_RNG_PSEUDO_MTGP32);
		A.gencreated = 1;
		HANDLE_ERROR(cudaMalloc((void**)&P.d_R, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_V, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_F, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_M, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_W, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_Q, P.N * sizeof(float4)));

		HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMemset((void*)P.d_F, 0, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMemset((void*)P.d_W, 0, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMemset((void*)P.d_M, 0, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMemset((void*)P.d_Q, 0, P.N * sizeof(float4)));
		//HANDLE_ERROR(cudaMalloc((void**)&P.d_U, 3 * P.N * sizeof(float)));
		P.h_F = (float*)malloc(3 * P.N * sizeof(float));
		P.h_V = (float*)malloc(3 * P.N * sizeof(float));
		P.h_R = (float*)malloc(3 * P.N * sizeof(float));
		P.h_M = (float*)malloc(3 * P.N * sizeof(float));
		P.h_W = (float*)malloc(3 * P.N * sizeof(float));
		P.h_Q = (float4*)malloc(P.N * sizeof(float4));
		curandSetPseudoRandomGeneratorSeed(A.gen, time(NULL));
		//curandGenerateNormal(A.gen, Padd.d_SR_V, 2 * P.N, 0.0f, sqrtf(1.0f * Po._1d_m * Pr.EkSpot));
		curandGenerateUniform(A.gen, P.d_R, 3 * P.N * sizeof(float));
		
		d_SetParticlesInParallelepiped<<<A.bloks, SMEMDIM>>>(P.d_R, P.d_V, P.N, S.center, S.L, S.Vgenmax);
	}
	//std::cin.get();	
}

void generateParticles_MA(particle_data& P, additional_data& A, sample_data& S)//Min Arrays
{
	A.bloks = ceil(P.N / (SMEMDIM)) + 1;
	//std::cerr << int(A.gencreated) << "\n";
	std::cerr << "Generate partiicles " << A.bloks << " N=" << P.N << " " << 5 * 3 * P.N * sizeof(float) + P.N * sizeof(float4) << " " << (5 * 3 * P.N * sizeof(float) + P.N * sizeof(float4)) / (1024 * 1024) << "\n";
	if (A.gencreated == 0)
	{
		curandCreateGenerator(&A.gen, CURAND_RNG_PSEUDO_MTGP32);
		A.gencreated = 1;
		HANDLE_ERROR(cudaMalloc((void**)&P.d_R, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_V, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_F, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_M, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&P.d_W, 3 * P.N * sizeof(float)));
		//HANDLE_ERROR(cudaMalloc((void**)&P.d_Q, P.N * sizeof(float4)));

		HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMemset((void*)P.d_F, 0, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMemset((void*)P.d_W, 0, 3 * P.N * sizeof(float)));
		HANDLE_ERROR(cudaMemset((void*)P.d_M, 0, 3 * P.N * sizeof(float)));
		//HANDLE_ERROR(cudaMemset((void*)P.d_Q, 0, P.N * sizeof(float4)));
		//HANDLE_ERROR(cudaMalloc((void**)&P.d_U, 3 * P.N * sizeof(float)));
		//P.h_F = (float*)malloc(3 * P.N * sizeof(float));
		//P.h_V = (float*)malloc(3 * P.N * sizeof(float));
		//P.h_R = (float*)malloc(3 * P.N * sizeof(float));
		//P.h_M = (float*)malloc(3 * P.N * sizeof(float));
		//P.h_W = (float*)malloc(3 * P.N * sizeof(float));
		//P.h_Q = (float4*)malloc(P.N * sizeof(float4));
		curandSetPseudoRandomGeneratorSeed(A.gen, time(NULL));
		//curandGenerateNormal(A.gen, Padd.d_SR_V, 2 * P.N, 0.0f, sqrtf(1.0f * Po._1d_m * Pr.EkSpot));
		curandGenerateUniform(A.gen, P.d_R, 3 * P.N * sizeof(float));

		d_SetParticlesInParallelepiped << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.N, S.center, S.L, S.Vgenmax);
	}
	//std::cin.get();	
}

__global__ void d_SetParticlesVelocities(float* __restrict__ V, uint N, const float vm)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	while (idx < N)
	{
		V[idx] = vm * (V[idx] - 0.5);
		V[idx + N] = vm * (V[idx + N] - 0.5);
		V[idx + 2 * N] = vm * (V[idx + 2 * N] - 0.5);

		idx += blockDim.x * gridDim.x;
	}
}

void generateVelocities(particle_data& P, additional_data& A, sample_data& S)
{
	curandGenerateUniform(A.gen, P.d_V, 3 * P.N * sizeof(float));
	d_SetParticlesVelocities << <A.bloks, SMEMDIM >> > (P.d_V, P.N, S.Vgenmax);
	//std::cin.get();	
}
