#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "pcuda_helper.h"
#include <cstring>

void SaveAllData(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po)
{
	std::ofstream file;
	char filename[200], filenamelast[200], filenamepart[10][100];
	strcpy(filenamelast, "");
#ifndef pre_nonlinearC
	//strcpy(filenamepart[1], "_Cc");
#endif
	//strcat(filenamelast, filenamepart[1]);
	//sprintf(filenamepart[2], "_D%i_a%.5f_ID%.3f", dn, Pnet.a_aver, Pnet.InitialDeformation);
	//strcat(filenamelast, filenamepart[2]);

	//Po.vis = 0;
	strcat(filenamelast, ".dat");
	strcpy(filename, "./result/DATA");
	strcat(filename, filenamelast);	
	std::cerr << "Save DATA " << filename << "\n";
	file.open(filename, std::ios::out | std::ios::binary);
	//std::cerr << "write_sample_data start\n"; //std::cin.get();
	write_sample_data(file, S);
	//std::cerr << "write_p0_data start\n"; //std::cin.get();
	write_particle_data(file, P);
	//std::cerr << "write_pNet_data start\n"; //std::cin.get();
	write_interaction_list_data(file, IL);
	//std::cerr << "write_potential_data start\n"; //std::cin.get();
	file.close();
}

void LoadAllData(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po)
{
	std::ifstream file;
	char filename[200], filenamelast[200], filenamepart[10][100];
	strcpy(filenamelast, "");
#ifndef pre_nonlinearC
	//strcpy(filenamepart[1], "_Cc");
#endif
	//strcat(filenamelast, filenamepart[1]);
	//sprintf(filenamepart[2], "_D%i_a%.5f_ID%.3f", dn, Pnet.a_aver, Pnet.InitialDeformation);
	//strcat(filenamelast, filenamepart[2]);
	//sprintf(filenamepart[9], "_c%.1f_Nn%i_RR%.2f", Pnet.Connectivity, int(Pnet.Nnodes), Pnet.CellDistance);
	//strcat(filenamelast, filenamepart[9]);
	strcat(filenamelast, ".dat");
	strcpy(filename, "./result/DATA");
			
	strcat(filename, filenamelast);
	std::cerr << "Read DATA " << filename << "\n";
	//std::cin.get();
	file.open(filename, std::ios::in | std::ios::binary);
	read_sample_data(file, S);
	read_particle_data(file, P);
	S.A_real.x = 1e20; S.A_real.y = 1e20; S.A_real.z = 1e20;
	S.B_real.x = -1e20; S.B_real.y = -1e20; S.B_real.z = -1e20;
	for (int i = 0; i < P.N; ++i)
	{
		if (P.h_R[i] < S.A_real.x)
			S.A_real.x = P.h_R[i];
		if (P.h_R[i] > S.B_real.x)
			S.B_real.x = P.h_R[i];
		
		if (P.h_R[i + P.N] < S.A_real.y)
			S.A_real.y = P.h_R[i + P.N];
		if (P.h_R[i + P.N] > S.B_real.y)
			S.B_real.y = P.h_R[i + P.N];
		
		if (P.h_R[i + 2 * P.N] < S.A_real.z)
			S.A_real.z = P.h_R[i + 2 * P.N];
		if (P.h_R[i + 2 * P.N] > S.B_real.z)
			S.B_real.z = P.h_R[i + 2 * P.N];		
	}
	S.size_real.x = S.B_real.x - S.A_real.x; S.size_real.y = S.B_real.y - S.A_real.y; S.size_real.z = S.B_real.z - S.A_real.z;
	S.center_real.x = 0.5 * (S.B_real.x + S.A_real.x); S.center_real.y = 0.5 * (S.B_real.y + S.A_real.y); S.center_real.z = 0.5 * (S.B_real.z + S.A_real.z);
	std::cerr << "AB real " << S.A_real.x << " " << S.A_real.y << " " << S.A_real.z << " " << S.B_real.x << " " << S.B_real.y << " " << S.B_real.z << "\n";
	read_interaction_list_data(file, IL);
	file.close();
	fprintf(stderr, "Finish Load Data\n", 0);
	A.bloks = ceil(P.N / (SMEMDIM)) + 1;
	A.ibloks = ceil(IL.N / (SMEMDIM)) + 1;	
}

/*void reloadLattice(p_data& P, p0_data& P0)
{
	//HANDLE_ERROR(cudaMemset((void*)P.d_U, 0, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(P.d_U, P0.d_U0, 2 * P0.N * sizeof(float), cudaMemcpyDeviceToDevice));
	//HANDLE_ERROR(cudaMemcpy((void*)P.d_U, (void*)P0.d_U0, 2 * P.N * sizeof(float), cudaMemcpyDeviceToDevice));   
	//fprintf(stderr, "Finish renewLattice\n");
}
/**/
void write_sample_data(std::ofstream& file, sample_data& S)
{	
	std::cerr << "write_sample_data\n";
	file.write((char*)(&S.L), sizeof(float3));
	file.write((char*)(&S.A), sizeof(float3));
	file.write((char*)(&S.B), sizeof(float3));
	file.write((char*)(&S.axis), sizeof(float3));
	file.write((char*)(&S.center), sizeof(float3));
	file.write((char*)(&S.size), sizeof(float3));
	file.write((char*)(&S.R0), sizeof(float3));
	file.write((char*)(&S.H0), sizeof(float3));	
	std::cerr << "Finish write_sample_data\n";
}

void read_sample_data(std::ifstream& file, sample_data& S)
{
	std::cerr << "read_sample_data\n";
	file.read((char*)(&S.L), sizeof(float3));
	file.read((char*)(&S.A), sizeof(float3));
	file.read((char*)(&S.B), sizeof(float3));
	file.read((char*)(&S.axis), sizeof(float3));
	file.read((char*)(&S.center), sizeof(float3));
	file.read((char*)(&S.size), sizeof(float3));
	file.read((char*)(&S.R0), sizeof(float3));
	file.read((char*)(&S.H0), sizeof(float3));	
	std::cerr << "Finish read_sample_data\n";
}

void write_particle_data(std::ofstream& file, particle_data& P)
{
	std::cerr << "write_particle_data " << P.N << "\n";
	file.write((char*)(&P.N), sizeof(uint));
	std::cerr << "Copy arrays\n";
	HANDLE_ERROR(cudaMemcpy(P.h_F, P.d_F, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(P.h_V, P.d_V, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(P.h_M, P.d_M, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(P.h_W, P.d_W, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(P.h_Q, P.d_Q, P.N * sizeof(float4), cudaMemcpyDeviceToHost));
	std::cerr << "Write arrays\n";
	file.write((char*)(P.h_F), 3 * P.N * sizeof(float));
	file.write((char*)(P.h_V), 3 * P.N * sizeof(float));
	file.write((char*)(P.h_R), 3 * P.N * sizeof(float));
	file.write((char*)(P.h_M), 3 * P.N * sizeof(float));
	file.write((char*)(P.h_W), 3 * P.N * sizeof(float));
	file.write((char*)(P.h_Q), P.N * sizeof(float4));
	std::cerr << "Finish write_particle_data\n";
}

void read_particle_data(std::ifstream& file, particle_data& P)
{
	std::cerr << "read_particle_data\n";
	file.read((char*)(&P.N), sizeof(uint));
	P._1d_N = 1.0 / P.N;
	std::cerr << "Allocate arrays HOST\n";
	P.h_F = (float*)malloc(3 * P.N * sizeof(float));
	P.h_V = (float*)malloc(3 * P.N * sizeof(float));
	P.h_R = (float*)malloc(3 * P.N * sizeof(float));
	P.h_M = (float*)malloc(3 * P.N * sizeof(float));
	P.h_W = (float*)malloc(3 * P.N * sizeof(float));
	P.h_Q = (float4*)malloc(P.N * sizeof(float4));
	std::cerr << "Allocate arrays GPU\n";
	HANDLE_ERROR(cudaMalloc((void**)&P.d_R, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_V, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_F, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_M, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_W, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_Q, P.N * sizeof(float4)));
	std::cerr << "Read arrays\n";
	file.read((char*)(P.h_F), 3 * P.N * sizeof(float));
	file.read((char*)(P.h_V), 3 * P.N * sizeof(float));
	file.read((char*)(P.h_R), 3 * P.N * sizeof(float));
	file.read((char*)(P.h_M), 3 * P.N * sizeof(float));
	file.read((char*)(P.h_W), 3 * P.N * sizeof(float));
	file.read((char*)(P.h_Q), P.N * sizeof(float4));
	std::cerr << "Copy arrays\n";
	HANDLE_ERROR(cudaMemcpy(P.d_F, P.h_F, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_V, P.h_V, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_R, P.h_R, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_M, P.h_M, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_W, P.h_W, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_Q, P.h_Q, P.N * sizeof(float4), cudaMemcpyHostToDevice));
	std::cerr << "Finish read_particle_data\n";
}

void write_interaction_list_data(std::ofstream& file, interaction_list_data& IL)
{
	std::cerr << "write_interaction_list_data " << IL.N << " " << IL.IonP << "\n";
	file.write((char*)(&IL.N), sizeof(uint));
	file.write((char*)(&IL.IonP), sizeof(uint));
	file.write((char*)(&IL.acut), sizeof(float));
	std::cerr << "Copy arrays\n";
	HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(IL.h_b_r, IL.d_b_r, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_1d_iL, IL.d_1d_iL, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(IL.h_AxialMoment, IL.d_AxialMoment, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_rij, IL.d_rij, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_Oijt, IL.d_Oijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_Fijn, IL.d_Fijn, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_Fijt, IL.d_Fijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_Mijn, IL.d_Mijn, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_Mijt, IL.d_Mijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_Mijadd, IL.d_Mijadd, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	std::cerr << "Write arrays\n";
	file.write((char*)(IL.h_IL), IL.N * sizeof(uint));
	file.write((char*)(IL.h_ILtype), IL.N * sizeof(uint));
	file.write((char*)(IL.h_1d_iL), IL.N * sizeof(float));
	//file.write((char*)(IL.h_AxialMoment), IL.N * sizeof(float));
	file.write((char*)(IL.h_rij), IL.N * sizeof(float3));
	file.write((char*)(IL.h_Oijt), IL.N * sizeof(float3));
	file.write((char*)(IL.h_Fijn), IL.N * sizeof(float3));
	file.write((char*)(IL.h_Fijt), IL.N * sizeof(float3));
	file.write((char*)(IL.h_Mijn), IL.N * sizeof(float3));
	file.write((char*)(IL.h_Mijt), IL.N * sizeof(float3));
	file.write((char*)(IL.h_Mijadd), IL.N * sizeof(float3));
	std::cerr << "Finish write_interaction_list_data\n";
}

void read_interaction_list_data(std::ifstream& file, interaction_list_data& IL)
{
	std::cerr << "read_interaction_list_data\n";
	file.read((char*)(&IL.N), sizeof(uint));
	file.read((char*)(&IL.IonP), sizeof(uint));
	file.read((char*)(&IL.acut), sizeof(float));
	IL.aacut = IL.acut * IL.acut;
	std::cerr << "Allocate arrays HOST\n";
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
	std::cerr << "Allocate arrays GPU\n";
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
	std::cerr << "Read arrays\n";
	file.read((char*)(IL.h_IL), IL.N * sizeof(uint));
	file.read((char*)(IL.h_ILtype), IL.N * sizeof(uint));
	file.read((char*)(IL.h_1d_iL), IL.N * sizeof(float));
	//file.read((char*)(IL.h_AxialMoment), IL.N * sizeof(float));
	file.read((char*)(IL.h_rij), IL.N * sizeof(float3));
	file.read((char*)(IL.h_Oijt), IL.N * sizeof(float3));
	file.read((char*)(IL.h_Fijn), IL.N * sizeof(float3));
	file.read((char*)(IL.h_Fijt), IL.N * sizeof(float3));
	file.read((char*)(IL.h_Mijn), IL.N * sizeof(float3));
	file.read((char*)(IL.h_Mijt), IL.N * sizeof(float3));
	file.read((char*)(IL.h_Mijadd), IL.N * sizeof(float3));
	std::cerr << "Copy arrays\n";
	HANDLE_ERROR(cudaMemcpy(IL.d_IL, IL.h_IL, IL.N * sizeof(uint), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_ILtype, IL.h_ILtype, IL.N * sizeof(uint), cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(IL.d_b_r, IL.h_b_r, IL.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_1d_iL, IL.h_1d_iL, IL.N * sizeof(float), cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(IL.d_AxialMoment, IL.h_AxialMoment, IL.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_rij, IL.h_rij, IL.N * sizeof(float3), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_Oijt, IL.h_Oijt, IL.N * sizeof(float3), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_Fijn, IL.h_Fijn, IL.N * sizeof(float3), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_Fijt, IL.h_Fijt, IL.N * sizeof(float3), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_Mijn, IL.h_Mijn, IL.N * sizeof(float3), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_Mijt, IL.h_Mijt, IL.N * sizeof(float3), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(IL.d_Mijadd, IL.h_Mijadd, IL.N * sizeof(float3), cudaMemcpyHostToDevice));
	std::cerr << "Finish read_interaction_list_data\n";
}
