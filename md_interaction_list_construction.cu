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
#include <math.h>
#include <cub/cub.cuh>

void InteractionListInit(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL)
{	
	IL.N = P.N * IL.IonP;
	IL.aacut = IL.acut * IL.acut;
	uint_fast64_t stmp = IL.N * sizeof(uint) + IL.N * sizeof(uint) + IL.N * sizeof(float) + 7 * IL.N * sizeof(float3);
	std::cerr << "InteractionListInit " << IL.N << " " << stmp << " " << stmp / (1024 * 1024) << " | " << sqrt(IL.aacut) << "\n";
	//std::cin.get();
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_IL, IL.N * sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_ILtype, IL.N * sizeof(uint)));
	//HANDLE_ERROR(cudaMalloc((void**)&IL.d_b_r, IL.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_1d_iL, IL.N * sizeof(float)));
	//HANDLE_ERROR(cudaMalloc((void**)&IL.d_AxialMoment, IL.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_rij, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_Oijt, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_Fijn, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_Fijt, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_Mijn, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_Mijt, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMalloc((void**)&IL.d_Mijadd, IL.N * sizeof(float3)));

	HANDLE_ERROR(cudaMemset((void*)IL.d_ILtype, 0, IL.N * sizeof(uint)));
	HANDLE_ERROR(cudaMemset((void*)IL.d_1d_iL, 0, IL.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)IL.d_rij, 0, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMemset((void*)IL.d_Oijt, 0, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMemset((void*)IL.d_Fijn, 0, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMemset((void*)IL.d_Fijt, 0, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMemset((void*)IL.d_Mijn, 0, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMemset((void*)IL.d_Mijt, 0, IL.N * sizeof(float3)));
	HANDLE_ERROR(cudaMemset((void*)IL.d_Mijadd, 0, IL.N * sizeof(float3)));

	IL.h_IL = (uint*)malloc(IL.N * sizeof(uint));
	IL.h_ILtype = (uint*)malloc(IL.N * sizeof(uint));
	//IL.h_b_r = (float*)malloc(IL.N * sizeof(float));
	IL.h_1d_iL = (float*)malloc(IL.N * sizeof(float));
	//IL.h_AxialMoment = (float*)malloc(IL.N * sizeof(float));
	IL.h_rij = (float3*)malloc(IL.N * sizeof(float3));
	IL.h_Oijt = (float3*)malloc(IL.N * sizeof(float3));
	IL.h_Fijn = (float3*)malloc(IL.N * sizeof(float3));
	IL.h_Fijt = (float3*)malloc(IL.N * sizeof(float3));
	IL.h_Mijn = (float3*)malloc(IL.N * sizeof(float3));
	IL.h_Mijt = (float3*)malloc(IL.N * sizeof(float3));
	IL.h_Mijadd = (float3*)malloc(IL.N * sizeof(float3));

	A.ibloks = ceil(IL.N / (SMEMDIM)) + 1;
	memset(IL.h_ILtype, 0, IL.N * sizeof(uint));	
}

void InteractionListConstruct(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C)
{
	HANDLE_ERROR(cudaMemset((void*)IL.d_IL, 0xFF, IL.N * sizeof(uint)));
	//std::cerr<<"InteractionListConstruct "<<P.d_R<<" "<<C.d_CI<<" "<<C.d_CIs<<" "<<C.d_pnC<<" "<<IL.d_IL<<"\n";
	//std::cerr<<"InteractionListConstruct "<<P.N<<" "<<IL.IonP<<" "<<C.a<<" "<<C._1d_a<<" "<<IL.aacut<<" "<<C.Nr.x<<" "<<C.Nr.y<<" "<<C.Nr.z<<" "<<C.N<<" | "<<SMEMDIM<<"\n";
	d_ConstructInteractionList << <A.bloks, SMEMDIM >> > (P.d_R, P.N, C.d_CI, C.d_CIs, C.d_pnC, IL.d_IL, IL.IonP, C.a, C._1d_a, IL.aacut, C.Nr, C.N);
	//d_ConstructInteractionListA<< <A.bloks, SMEMDIM >> >(P.d_R, IL.IonP, C.N);
	//d_ConstructInteractionList    														 R,     N,         CI,          CIs,         pnC,          IL,     IonP,     a,     _1d_a,     aacut,    cN,   CN)

	//std::cerr << "InteractionListConstruct " << A.bloks << " " << P.d_R << " " << P.N << " " << C.d_CI << " " << C.d_CIs << " " << C.d_pnC << " " << IL.d_IL << " " << IL.IonP << " " << C.a << " " << C._1d_a << " " << IL.aacut << " " << C.N << " " << C.Nr.x << " " << C.Nr.y << " " << C.Nr.z << "\n"; std::cin.get();
}

void InteractionListReConstruct(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C)
{
	//HANDLE_ERROR(cudaMemset((void*)IL.d_IL, 0xFF, IL.N * sizeof(uint)));
	d_ReConstructInteractionList << <A.bloks, SMEMDIM >> > (P.d_R, P.N, C.d_CI, C.d_CIs, C.d_pnC, IL.d_IL, IL.IonP, C.a, C._1d_a, IL.aacut, C.Nr, C.N);
}

void InteractionListReConstructBPM(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C)
{
	//HANDLE_ERROR(cudaMemset((void*)IL.d_IL, 0xFF, IL.N * sizeof(uint)));
	d_ReConstructInteractionListbpm << <A.bloks, SMEMDIM >> > (P.d_R, P.N, C.d_CI, C.d_CIs, C.d_pnC, IL.d_IL, IL.d_ILtype, IL.IonP, C.a, C._1d_a, IL.aacut, C.Nr, C.N);
}