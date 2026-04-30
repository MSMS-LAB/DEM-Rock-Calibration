#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include "pcuda_helper.h"
#include "md_data_types.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include "md.h"
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <time.h>
#include <chrono>
#include <fstream>

//#define defc_SaveLammps

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


void ccs_freePD(md_task_data& mdTD)
{
	if (mdTD.P.d_R != nullptr) { cudaFree(mdTD.P.d_R); mdTD.P.d_R = nullptr; }
	if (mdTD.P.d_F != nullptr) { cudaFree(mdTD.P.d_F); mdTD.P.d_F = nullptr; }
	if (mdTD.P.d_V != nullptr) { cudaFree(mdTD.P.d_V); mdTD.P.d_V = nullptr; }
#ifdef defc_SaveLammps
	if (mdTD.P.h_R != nullptr) { free(mdTD.P.h_R); mdTD.P.h_R = nullptr; }
#endif // defc_SaveLammps

}

void ccs_createPD(md_task_data& mdTD)
{
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_V, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_F, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_V, 0, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_F, 0, 3 * mdTD.P.N * sizeof(float)));
#ifdef defc_SaveLammps
	mdTD.P.h_R = (float*)malloc(3 * mdTD.P.N * sizeof(float));
#endif // defc_SaveLammps

	
}

void ccs_freeCD(md_task_data& mdTD)
{	
	if (mdTD.C.d_IP != nullptr) { cudaFree(mdTD.C.d_IP); mdTD.C.d_IP = nullptr; }
	if (mdTD.C.d_CI != nullptr) { cudaFree(mdTD.C.d_CI); mdTD.C.d_CI = nullptr; }
	if (mdTD.C.d_CIs != nullptr) { cudaFree(mdTD.C.d_CIs); mdTD.C.d_CIs = nullptr; }
	if (mdTD.C.d_pnC != nullptr) { cudaFree(mdTD.C.d_pnC); mdTD.C.d_pnC = nullptr; }
	if (mdTD.C.d_tmp_old != nullptr) { cudaFree(mdTD.C.d_tmp_old); mdTD.C.d_tmp_old = nullptr; mdTD.C.d_tmp = nullptr; }
}

void ccs_createCD(md_task_data& mdTD)
{
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_IP, mdTD.P.N * sizeof(uint)));
	std::cerr << "CA " << mdTD.A.bloks << " " << SMEMDIM<<"  "<<mdTD.C.N<< "\n";
    d_FillIndex <<< mdTD.A.bloks, SMEMDIM >> > (mdTD.C.d_IP, mdTD.P.N);	
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_CI, mdTD.P.N * sizeof(uint)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_CIs, 2 * mdTD.P.N * sizeof(uint)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_pnC, 2 * mdTD.C.N * sizeof(uint)));
    mdTD.C.dtmpN_old = 128;
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_tmp_old, mdTD.C.dtmpN_old));

	//mdTD.C.h_IP = (uint*)malloc(mdTD.P.N * sizeof(uint));
	//mdTD.C.h_CI = (uint*)malloc(mdTD.P.N * sizeof(uint));
	//mdTD.C.h_CIs = (uint*)malloc(2 * mdTD.P.N * sizeof(uint));
}

void ccs_freeID(md_task_data& mdTD)
{
	//std::cerr << "fid " << mdTD.IL.d_IL << " " << mdTD.IL.d_ILtype << " " << mdTD.IL.d_1d_iL << "\n";
	if (mdTD.IL.d_IL != nullptr) { cudaFree(mdTD.IL.d_IL); mdTD.IL.d_IL = nullptr; }
	if (mdTD.IL.d_ILtype != nullptr) { cudaFree(mdTD.IL.d_ILtype); mdTD.IL.d_ILtype = nullptr; }
	if (mdTD.IL.d_1d_iL != nullptr) { cudaFree(mdTD.IL.d_1d_iL); mdTD.IL.d_1d_iL = nullptr; }
#ifdef defc_SaveLammps
	if (mdTD.IL.h_IL != nullptr) { free(mdTD.IL.h_IL); mdTD.IL.h_IL = nullptr; }
	if (mdTD.IL.h_ILtype != nullptr) { free(mdTD.P.h_R); mdTD.IL.h_ILtype = nullptr; }
#endif // defc_SaveLammps

}

void ccs_createID(md_task_data& mdTD)
{
	//uint_fast64_t stmp = IL.N * sizeof(uint) + IL.N * sizeof(uint) + IL.N * sizeof(float) + 7 * IL.N * sizeof(float3);
	//std::cerr << "InteractionListInit " << IL.N << " " << stmp << " " << stmp / (1024 * 1024) << " | " << sqrt(IL.aacut) << "\n";
	//std::cin.get();
	HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_IL), mdTD.IL.N * sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_ILtype), mdTD.IL.N * sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_1d_iL), mdTD.IL.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_ILtype, 0, mdTD.IL.N * sizeof(uint)));
	HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_1d_iL, 0, mdTD.IL.N * sizeof(float)));
#ifdef defc_SaveLammps
	mdTD.IL.h_IL = (uint*)malloc(mdTD.IL.N * sizeof(uint));
	mdTD.IL.h_ILtype = (uint*)malloc(mdTD.IL.N * sizeof(uint));
#endif // defc_SaveLammps

	
	//mdTD.IL.h_1d_iL = (float*)malloc(mdTD.IL.N * sizeof(float));
}

void srd_free(md_task_data& mdTD)
{
	if (mdTD.SRD.d_Rr != nullptr) { cudaFree(mdTD.SRD.d_Rr); mdTD.SRD.d_Rr = nullptr; }
}

__global__ void d_SetSampleDistorsionSize(float* __restrict__ dR, const float C)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;	
	if(dR[idx]>0)
	{
		dR[idx] *= -C;
	} else{
		dR[idx] *= C;
	}
	//printf("Distorsion %i %f\n", idx, dR[idx]);
}

void srd_create(md_task_data& mdTD)
{	
	HANDLE_ERROR(cudaMalloc((void**)&(mdTD.SRD.d_Rr), mdTD.A.bloks * SMEMDIM * sizeof(uint)));		
	curandGenerateNormal(mdTD.A.gen, mdTD.SRD.d_Rr, mdTD.A.bloks * SMEMDIM, mdTD.SRD.M, mdTD.SRD.sD);
	d_SetSampleDistorsionSize<< <mdTD.A.bloks, SMEMDIM >> >(mdTD.SRD.d_Rr, mdTD.SRD.C);
}

void SaveParticleDATAtxt(float* h_R, uint n, const float3 hidepoint, char* Name)
{	
	std::ofstream file;
	file.open(Name, std::ios::out);
	double l_rm;
	uint l_ndelete = 0;
	for (uint i = 0; i < n; ++i)
	{
		l_rm = (h_R[i]-hidepoint.x)*(h_R[i]-hidepoint.x) + (h_R[i + n]-hidepoint.y)*(h_R[i + n]-hidepoint.y) + (h_R[i + 2 * n]-hidepoint.z)*(h_R[i + 2 * n]-hidepoint.z) ;
		if(l_rm<1e-10) ++l_ndelete;
	}
	file << n-l_ndelete << "\n";
	for (uint i = 0; i < n; ++i)
	{
		l_rm = (h_R[i]-hidepoint.x)*(h_R[i]-hidepoint.x) + (h_R[i + n]-hidepoint.y)*(h_R[i + n]-hidepoint.y) + (h_R[i + 2 * n]-hidepoint.z)*(h_R[i + 2 * n]-hidepoint.z) ;
		if(l_rm<1e-10) continue;
		file << h_R[i] * length_const << " " << h_R[i + n] * length_const << " " << h_R[i + 2 * n] * length_const << "\n";
	}
	file.close();
}

void LoadParticleDATAtxt(float*& h_R, uint &n, char* Name)
{
	double x, y, z, lc = 1.0/ length_const;
	std::ifstream file;
	file.open(Name, std::ios::in);
	file >> n;
	if (h_R != nullptr) { free(h_R); h_R = nullptr; }
	h_R = (float*)malloc(3 * n * sizeof(float));
	std::cerr << "hR " << h_R << "\n";
	for (uint i = 0; i < n; ++i)
	{
		file >> x >> y >> z;
		h_R[i] = x * lc;
		h_R[i + n] = y * lc; 
		h_R[i + 2 * n] = z * lc;		
	}
	file.close();	
}

void h_CreateCylinderSample(mp_mdparameters_data & mpMDP, md_task_data &mdTD, uint isample, char namepart[])
{   
	char filename[256] = "";
	mdTD.S = mpMDP.S[isample];
    mdTD.P.N = mpMDP.S[isample].PN;
	mpMDP.PN[isample] = mdTD.P.N;
    mdTD.IL.CalculateParameters(mdTD.P.N, mdTD.Po.a_farcut);
    mdTD.C.CalculateParameters(mdTD.S.B.x, mdTD.S.B.y, mdTD.S.B.z);
    mdTD.A.ibloks = ceil(mdTD.IL.N / (SMEMDIM)) + 1;
	mdTD.S.Vgenmax = 1e-3 * mdTD.Po.p_r / mdTD.Po.dt;
    mdTD.Po.vis = 12.0e-2;
    mdTD.Po.vism = mdTD.Po.vis * mdTD.Po.p_m;
	mdTD.Po.D = mdTD.Po.m_E * mdTD.Po.b_S / (2.0 * mdTD.Po.p_r);
	mdTD.Po.a = 2.0 * mdTD.Po.p_r; mdTD.Po._1d_a = 1.0 / mdTD.Po.a; mdTD.Po.aa = mdTD.Po.a * mdTD.Po.a;
	mdTD.Po.CalculateParameters();
	mdTD.Po.D *= 1e-15;
	//std::cerr << "Po " << mdTD.Po.Tsim << " " << mdTD.Po.Trayleigh<< " " << mdTD.Po.dt << "\n"; std::cin.get();
	mdTD.A.CalculateParameters(mdTD.P.N, mdTD.IL.N);
	std::cerr << "N " << mdTD.P.N << " " << mdTD.IL.N << " " << mdTD.C.N << "\n";
	//std::cerr << "S " << mdTD.S.B.x << " " << mdTD.S.B.y << " " << mdTD.S.B.z << "\n";
    ccs_freePD(mdTD); //std::cerr << "q1\n";
    ccs_createPD(mdTD); //std::cerr << "q2\n";
    ccs_freeCD(mdTD); //std::cerr << "q3\n";
    ccs_createCD(mdTD); //std::cerr << "q4\n";
	//std::cerr << "Ar " << mdTD.C.d_IP << " " << mdTD.P.d_R << " " << mdTD.P.d_R + 3 * mdTD.P.N << " " << mdTD.P.d_F + 3 * mdTD.P.N << "\n"; std::cin.get();
	//std::cerr << "Po " << mdTD.Po.Tsim << " " << mdTD.Po.Trayleigh<< " " << mdTD.Po.dt << "\n"; std::cin.get();
    ccs_freeID(mdTD); //std::cerr << "q5\n";
    ccs_createID(mdTD); //std::cerr << "q6\n";
	
	if(mdTD.SRD.Distortion){srd_free(mdTD);	srd_create(mdTD);}

	mpMDP.createarrays(isample, mdTD.P.N);
	//std::cin.get();
	//std::cerr << "Ar " << mdTD.C.d_IP << " " << mdTD.P.d_R << " " << mdTD.P.d_R + 3 * mdTD.P.N << " " << mdTD.P.d_F + 3 * mdTD.P.N << "\n"; std::cin.get();
    curandSetPseudoRandomGeneratorSeed(mdTD.A.gen, time(NULL));	
    curandGenerateUniform(mdTD.A.gen, mdTD.P.d_R, 3 * mdTD.P.N);
	//HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_R, 0, 3 * mdTD.P.N * sizeof(float)));
	//HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(mdTD.C.h_IP, mdTD.C.d_IP, mdTD.P.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < mdTD.P.N; ++i) std::cerr << "FI " << i << " " << mdTD.C.h_IP[i] << " " << mdTD.P.h_R[i] << "\n"; std::cin.get();
	d_SetParticlesInParallelepiped << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.N, mdTD.S.center, mdTD.S.L, mdTD.S.Vgenmax);
#ifdef defc_SaveLammps
	HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
	sprintf(filename, "./result/steps/LAMMPS/CPp_%li.txt",  0);
	if(mdTD.SaveLammpsCreateSample)SaveLammpsDATASimple(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL, mdTD.Po, filename, false);
#endif // defc_SaveLammps
	
    firerelax_data Fire;
    SetFIREData(mdTD.P, mdTD.Po, Fire);
    Fire.MaxStepsRelaxation =30000*int(Fire.Tsim/Fire.dt0);
	Fire.StepsPreRelaxation = 10000*int(Fire.Tsim/Fire.dt0);
	
	std::cerr << "FIRE Start\n";
	Fire.NPpositive = 0;
	Fire.NPnegative = 0;
	Fire.dt = Fire.dt0;
	Fire.h_FdotV = (float*)malloc(Fire.bloks4 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Fire.d_FdotV, Fire.bloks4 * sizeof(float)));
	double ilacutmax = mdTD.IL.acut, dmax = 1.5e+3 * mdTD.Po.m_E * mdTD.Po.b_S / (2.0 * mdTD.Po.p_r);
	mdTD.IL.acut *= 1e-3;
	mdTD.IL.aacut = mdTD.IL.acut * mdTD.IL.acut;
	float3 spacesized2 = { 0.8 * mdTD.S.sized2.x, 0.8 * mdTD.S.sized2.y, 0.8 * mdTD.S.sized2.z };
	//std::cerr<<"Size "<<sizeof(uint)<<" "<<sizeof(uint_fast32_t)<<" "<<sizeof(unsigned int)<<" "<<UINT_MAX<<"\n";std::cin.get();
	//double ILacutmax = mdTD.IL.acut;
	//Po.D = 5e-4 * Po.m_E * Po.b_S / (2.0 * Po.p_r);
	//Po.a = 1e-8;
	//double Dmax = 1e+3 * Po.m_E * Po.b_S / (2.0 * Po.p_r);/**/
	uint steps, i;
	double timereal = 0;
	//std::cerr << "Po" << mdTD.Po.Tsim << " " << mdTD.Po.Trayleigh<< " " << mdTD.Po.dt << "\n"; //std::cin.get();
	//std::cerr << "ParamFire " << Fire.dt <<" "<<Fire.alpha<<" "<<Fire.Tsim<<"\n"; std::cin.get();
	//RenewInteractionList_full(mdTD.P, mdTD.C, mdTD.S[isample], mdTD.A, mdTD.IL);
	CellDistribution(mdTD.P, mdTD.A, mdTD.S, mdTD.C);	
	InteractionListConstruct(mdTD.P, mdTD.A, mdTD.S, mdTD.IL, mdTD.C);
	thrust::device_ptr<float> dtp_1d_Rijm(mdTD.IL.d_1d_iL);
	//std::cin.get();
	//thrust::device_ptr<float> max_ptr = thrust::min_element(dtp_1d_Rijm, dtp_1d_Rijm + mdTD.IL.N);
	//std::cerr << "q11 " << mdTD.IL.d_1d_iL << " " << dtp_1d_Rijm << " " << dtp_1d_Rijm + mdTD.IL.N << " " << mdTD.IL.N << " " << max_ptr << "\n"; std::cin.get();
	uint l_StepsSaveLammps = 1000*int(Fire.Tsim/Fire.dt0), l_StepsIncreaseD = 4*int(Fire.Tsim/Fire.dt0), l_StepsIncreaseAcut = 300*int(Fire.Tsim/Fire.dt0),
	l_StepsIncreaseSample= 3000*int(Fire.Tsim/Fire.dt0), l_StepsPushSample[3]= {2000*int(Fire.Tsim/Fire.dt0),3000*int(Fire.Tsim/Fire.dt0),100*int(Fire.Tsim/Fire.dt0)},
	l_StepsReduceParticlesNumber= 500*int(Fire.Tsim/Fire.dt0), l_StepsIntervalReduceParticlesNumber= 300*int(Fire.Tsim/Fire.dt0);
	std::cerr<<"T "<<int(Fire.Tsim/Fire.dt0)<<" "<<Fire.StepsPreRelaxation<<" "<<l_StepsIncreaseSample<<" "<<l_StepsReduceParticlesNumber<<"\n";
	
	double l_maxintersection = 0, l_maxpredeformation=mpMDP.PreMaximumParticleIntersection, l_maxdeformation=mpMDP.MaximumParticleIntersection;
	uint l_deletedparticles=0;
	for (steps = 0; steps < Fire.StepsPreRelaxation || l_maxintersection<l_maxpredeformation; ++steps)
	{		
		if (steps % l_StepsSaveLammps  == 0 && mdTD.SaveLammpsCreateSample)
		{
			HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
			sprintf(filename, "./result/steps/LAMMPS/CPp_%li.txt", steps+1);
			SaveLammpsDATASimple(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL, mdTD.Po, filename, false);
			std::cerr<<"step "<<steps<<"\n"; //std::cin.get();
		}
		if (steps % l_StepsIncreaseD == 0 && mdTD.Po.D < 0.3e-4 * dmax)
			mdTD.Po.D *= 1.30;
		/*if (steps % 100 == 0 && mdTD.Po.a - 2.0 * mdTD.Po.p_r < -1e-10)
		{
			mdTD.Po.a *= 1.07;
			if (mdTD.Po.a - 2.0 * mdTD.Po.p_r > -1e-10)mdTD.Po.a = 2.0 * mdTD.Po.p_r;
			mdTD.Po._1d_a = 1.0 / mdTD.Po.a;
			mdTD.Po.aa = mdTD.Po.a * mdTD.Po.a;
		}*/
		

		if (steps % l_StepsIncreaseAcut == 0 && mdTD.IL.acut - ilacutmax < -1e-10)
		{
			mdTD.IL.acut *= 2.0;
			if (mdTD.IL.acut > ilacutmax)mdTD.IL.acut = ilacutmax;
			mdTD.IL.aacut = mdTD.IL.acut * mdTD.IL.acut;
		}
		CellDistribution(mdTD.P, mdTD.A, mdTD.S, mdTD.C);
		InteractionListReConstruct(mdTD.P, mdTD.A, mdTD.S, mdTD.IL, mdTD.C);
		//RenewInteractionList_New(mdTD.P, mdTD.C, mdTD.S[isample], mdTD.A, mdTD.IL); std::cerr << "e1 "<<steps<<"\n";
		d_CalculateForces << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_F, mdTD.P.N, mdTD.IL.d_IL, mdTD.IL.d_1d_iL, mdTD.IL.IonP, mdTD.Po.D, mdTD.Po.aa);// std::cerr << "e2 " << steps << "\n";
		d_CalculateIncrementsViscos << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, mdTD.P.d_V, mdTD.P.d_R, mdTD.P.N, mdTD.Po.dt_d_m, Fire.dt, mdTD.Po.vism);// std::cerr << "e3 " << steps << "\n";
		if(!mdTD.SRD.Distortion) d_CylinderRestrictionZ << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_R, mdTD.P.N, mdTD.S.center, spacesized2.x, spacesized2.z, mdTD.S.hidenpoint.z);// std::cerr << "e4 " << steps << "\n";
		else d_CylinderRestrictionZr2 << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_R, mdTD.SRD.d_Rr, mdTD.P.N, mdTD.S.center, spacesized2.x, spacesized2.z, mdTD.S.hidenpoint);
		
		if (steps == l_StepsIncreaseSample)
		{
			spacesized2.x *= 1.25;
			spacesized2.y *= 1.25;
			spacesized2.z *= 1.25;
			std::cerr<<"Sample Increase\n";
		}
		if (steps > l_StepsPushSample[0] && steps < l_StepsPushSample[1] && steps % l_StepsPushSample[2] == 0)
		{
			curandGenerateUniform(mdTD.A.gen, mdTD.P.d_F, mdTD.P.N);
			d_CylinderBorderPush << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, 0.85, mdTD.P.d_R, mdTD.P.N, mdTD.S.center, spacesized2.x, spacesized2.z);

		}
		
		//std::cin.get();
		
		if (steps % l_StepsIntervalReduceParticlesNumber == 0)
		{	
			
			thrust::device_ptr<float> max_ptr = thrust::min_element(dtp_1d_Rijm, dtp_1d_Rijm + mdTD.P.N);
			unsigned int position = max_ptr - dtp_1d_Rijm;
			float max_val = *max_ptr;
			std::cerr << "Fire prerelaxation " << steps << " " << position << " " << sqrt(max_val) << " " << sqrt(max_val) - 2.0 * mdTD.Po.p_r << " " << (sqrt(max_val) - 2.0 * mdTD.Po.p_r) / (2.0 * mdTD.Po.p_r)
				<< " | " << mdTD.IL.acut << " " << mdTD.IL.acut / ilacutmax << " " << mdTD.Po.a << " " << mdTD.Po.a / (2.0 * mdTD.Po.p_r)<<" "<< mdTD.Po.D/dmax << "\n";
			l_maxintersection = (sqrt(max_val) - 2.0 * mdTD.Po.p_r) / (2.0 * mdTD.Po.p_r);
			if(steps>l_StepsReduceParticlesNumber+l_StepsPushSample[1] &&  l_maxintersection<l_maxpredeformation) 
			{
				d_ReduceParticleNumber<< <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_F, mdTD.P.N, position, mdTD.IL.d_IL,  mdTD.IL.IonP, mdTD.S.hidenpoint);
				std::cerr<<"Particle Deleted! "<<l_maxintersection<<"\n";
				++l_deletedparticles;
			}					
		}
		timereal += Fire.dt;
		//std::cin.get();
	}
	//std::cin.get();
	timereal = 0;
	//std::cerr << "ParamFire " << Fire.dt << " | " << S.L.x << " " << S.L.y << " " << S.L.z << " | " << S.center.x << " " << S.center.y << " " << S.center.z << " | " << S.size.x << " " << S.size.y << " " << S.size.z << "\n";
	//std::cerr << "ParamFire " << Fire.dt <<" "<<Fire.alpha<<"\n"; std::cin.get();
	std::cerr << "HidePoint " << mdTD.S.hidenpoint.x <<" "<<mdTD.S.hidenpoint.y<<" "<<mdTD.S.hidenpoint.z<<"\n"; //std::cin.get();
	
	//thrust::device_ptr<float> dtp_1d_Rijm(mdTD.IL.d_1d_iL);
	Fire.alpha0 = 0.1; Fire.alpha = Fire.alpha0;
	mdTD.Po.D = 1e+0 * mdTD.Po.m_E * mdTD.Po.b_S / (2.0 * mdTD.Po.p_r);
	mdTD.Po.a = 2.0 * mdTD.Po.p_r; mdTD.Po._1d_a = 1.0 / mdTD.Po.a; mdTD.Po.aa = mdTD.Po.a * mdTD.Po.a;
	
	for (steps = 0; steps < Fire.MaxStepsRelaxation || l_maxintersection<l_maxdeformation; ++steps)
	{	
		if (steps % l_StepsSaveLammps == 0 && mdTD.SaveLammpsCreateSample)
		{
			HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
			sprintf(filename, "./result/steps/LAMMPS/CPp_%li.txt", steps+Fire.StepsPreRelaxation);
			SaveLammpsDATASimple(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL, mdTD.Po, filename, false);
		}

		RenewInteractionList_New(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL);

		d_CalculateForces << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_F, mdTD.P.N, mdTD.IL.d_IL, mdTD.IL.d_1d_iL, mdTD.IL.IonP, mdTD.Po.D, mdTD.Po.aa);
		d_FdotVEntire << < Fire.bloks4, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_F, Fire.d_FdotV, mdTD.P.N);
		cudaMemcpy(Fire.h_FdotV, Fire.d_FdotV, Fire.bloks4 * sizeof(float), cudaMemcpyDeviceToHost);
		Fire.FdotV = 0;
		for (i = 0; i < Fire.bloks4; ++i)
		{
			Fire.FdotV += Fire.h_FdotV[i];
		}
		if (Fire.FdotV > 0)
		{
			++Fire.NPpositive;
			Fire.NPnegative = 0;
			if (Fire.NPpositive > Fire.Ndelay)
			{
				Fire.dt = (Fire.dt * Fire.dtgrow < Fire.dtmax) ? Fire.dt * Fire.dtgrow : Fire.dtmax;
				Fire.alpha *= Fire.alphashrink;
			}			
		}
		else
		{
			Fire.NPpositive = 0;
			++Fire.NPnegative;
			if (Fire.NPnegative > Fire.NPnegativeMax && l_maxintersection>=l_maxdeformation)
				break;
			if (steps > Fire.Ndelay)
			{
				Fire.dt = (Fire.dt * Fire.dtshrink > Fire.dtmin) ? Fire.dt * Fire.dtshrink : Fire.dtmin;
				Fire.alpha = Fire.alpha0;
			}
			d_CalculateDecrementsHalfStepFIRE << < mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_R, mdTD.P.N, 0.5 * Fire.dt);
			cudaMemset(mdTD.P.d_V, 0, 3 * mdTD.P.N * sizeof(float));
		}
		d_CalculateIncrementsFIRE << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, mdTD.P.d_V, mdTD.P.d_R, mdTD.P.N, mdTD.Po.dt_d_m, Fire.dt, Fire.alpha);
		if(!mdTD.SRD.Distortion) d_CylinderRestrictionZ << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_R, mdTD.P.N, mdTD.S.center, mdTD.S.sized2.x, mdTD.S.sized2.z, mdTD.S.hidenpoint.z);// std::cerr << "e4 " << steps << "\n";
		else d_CylinderRestrictionZr2 << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_R, mdTD.SRD.d_Rr, mdTD.P.N, mdTD.S.center, mdTD.S.sized2.x, mdTD.S.sized2.z, mdTD.S.hidenpoint);

		
		if (steps % l_StepsIntervalReduceParticlesNumber == 0)
		{			
			thrust::device_ptr<float> max_ptr = thrust::min_element(dtp_1d_Rijm, dtp_1d_Rijm + mdTD.P.N);
			unsigned int position = max_ptr - dtp_1d_Rijm;
			float max_val = *max_ptr;
			std::cerr << "Fire relaxation " << steps << " " << position << " " << sqrt(max_val) << " " << sqrt(max_val) - 2.0 * mdTD.Po.p_r << " " << (sqrt(max_val) - 2.0 * mdTD.Po.p_r) / (2.0 * mdTD.Po.p_r)
				<< "\n";
			l_maxintersection = (sqrt(max_val) - 2.0 * mdTD.Po.p_r) / (2.0 * mdTD.Po.p_r);
			if(steps>l_StepsReduceParticlesNumber &&  l_maxintersection<l_maxdeformation) 
			{
				d_ReduceParticleNumber<< <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_F, mdTD.P.N, position, mdTD.IL.d_IL,  mdTD.IL.IonP, mdTD.S.hidenpoint);
				std::cerr<<"Particle Deleted! "<<l_maxintersection<<"\n";
				++l_deletedparticles;
			}					
		}
		timereal += Fire.dt;
		//std::cerr<<"Steps "<<steps<<" "<<Fire.alpha<<"\n"; std::cin.get();
	}
	std::cerr << "FIN FIRE! " << steps << " "<< l_maxintersection << "\n"; 
	std::cerr<<"Result density:porosity "<<(mdTD.P.N-l_deletedparticles)*mdTD.Po.p_V/mpMDP.S[isample].Vext<<" porosity: "<<1.0-(mdTD.P.N-l_deletedparticles)*mdTD.Po.p_V/mpMDP.S[isample].Vext<<"\n";//std::cin.get();
	mdTD.R.Porosity = 1.0-(mdTD.P.N-l_deletedparticles)*mdTD.Po.p_V/mpMDP.S[isample].Vext;
 
	free(Fire.h_FdotV);
	Fire.h_FdotV = nullptr;
	cudaFree(Fire.d_FdotV);
	Fire.d_FdotV = nullptr;

    HANDLE_ERROR(cudaMemcpy(mpMDP.h_R[isample], mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "./result/Sample_%u", isample);
	strcat(filename, namepart);
	strcat(filename, ".txt");
	SaveParticleDATAtxt(mpMDP.h_R[isample], mdTD.P.N, mdTD.S.hidenpoint, filename);
	ccs_freePD(mdTD);
	ccs_freeCD(mdTD);
	ccs_freeID(mdTD);
	//std::cin.get();
}

void h_LoadCylinderSample(mp_mdparameters_data& mpMDP, uint isample, char namepart[])
{
	char filename[256] = "";
	sprintf(filename, "./result/Sample_%u", isample); 
	strcat(filename, namepart);
	strcat(filename, ".txt");//std::cerr << "Load " << filename << "\n";
	LoadParticleDATAtxt(mpMDP.h_R[isample], mpMDP.PN[isample], filename);
	double l_max[3]={-DBL_MAX,-DBL_MAX,-DBL_MAX}, l_min[3]={DBL_MAX, DBL_MAX, DBL_MAX};
	for (uint l_i = 0; l_i < mpMDP.PN[isample]; ++l_i)
	{
		if(mpMDP.h_R[isample][l_i] < l_min[0]) l_min[0] = mpMDP.h_R[isample][l_i];
		if(mpMDP.h_R[isample][l_i] > l_max[0]) l_max[0] = mpMDP.h_R[isample][l_i];
		if(mpMDP.h_R[isample][l_i + mpMDP.PN[isample]] < l_min[1]) l_min[1] = mpMDP.h_R[isample][l_i + mpMDP.PN[isample]];
		if(mpMDP.h_R[isample][l_i + mpMDP.PN[isample]] > l_max[1]) l_max[1] = mpMDP.h_R[isample][l_i + mpMDP.PN[isample]];
		if(mpMDP.h_R[isample][l_i + 2 * mpMDP.PN[isample]] < l_min[2]) l_min[2] = mpMDP.h_R[isample][l_i + 2 * mpMDP.PN[isample]];
		if(mpMDP.h_R[isample][l_i + 2 * mpMDP.PN[isample]] > l_max[2]) l_max[2] = mpMDP.h_R[isample][l_i + 2 * mpMDP.PN[isample]];
	}
	std::cerr<<"Min Max "<<l_min[0]<<" "<<l_min[1]<<" "<<l_min[2]<<" | "<<l_max[0]<<" "<<l_max[1]<<" "<<l_max[2]<<"\n";//std::cin.get();
	mpMDP.S[isample].size_real.x = l_max[0] - l_min[0] + 2.0* mpMDP.mdTD[0].Po.p_r;
	mpMDP.S[isample].size_real.y = l_max[1] - l_min[1] + 2.0* mpMDP.mdTD[0].Po.p_r;
	mpMDP.S[isample].size_real.z = l_max[2] - l_min[2] + 2.0* mpMDP.mdTD[0].Po.p_r;
	mpMDP.S[isample].center_real.x = 0.5*(l_max[0] + l_min[0]);
	mpMDP.S[isample].center_real.y = 0.5*(l_max[1] + l_min[1]);
	mpMDP.S[isample].center_real.z = 0.5*(l_max[2] + l_min[2]);
	mpMDP.S[isample].PN = mpMDP.PN[isample];	
}

void h_LoadCylinderSampleMusen(mp_mdparameters_data& mpMDP, uint isample, char namepart[])
{
	char filename[256] = "";
	sprintf(filename, "./result/SampleMusen_%u", isample); 
	strcat(filename, namepart);
	strcat(filename, ".txt");//std::cerr << "Load " << filename << "\n";
	//LoadParticleDATAtxt(mpMDP.h_R[isample], mpMDP.PN[isample], filename);
	uint n, pi, pi1=0;
	double x, y, z, lc = 1.0/ length_const, l_max[3]={-DBL_MAX,-DBL_MAX,-DBL_MAX}, l_min[3]={DBL_MAX, DBL_MAX, DBL_MAX};
	std::ifstream file;
	file.open(filename, std::ios::in);
	file >> n;
	mpMDP.PN[isample] = n;
	std::cerr<<"Nparticles "<<n<<" | "<<mpMDP.S[isample].center.x<<" "<<mpMDP.S[isample].center.y<<" "<<mpMDP.S[isample].center.z<<" | "<<filename<<"\n";
	if (mpMDP.h_R[isample] != nullptr) { free(mpMDP.h_R[isample]); mpMDP.h_R[isample] = nullptr; }
	mpMDP.h_R[isample] = (float*)malloc(3 * n * sizeof(float));
	//std::cerr << "hR " << mpMDP.h_R[isample] << "\n";
	for (uint i = 0; i < n; ++i)
	{
		file >> pi >> x >> y >> z;
		if(pi!=pi1+1) std::cerr<<"Pi "<<pi<<"\n";
		mpMDP.h_R[isample][i] = x * lc + mpMDP.S[isample].center.x;
		mpMDP.h_R[isample][i + n] = y * lc + mpMDP.S[isample].center.y; 
		mpMDP.h_R[isample][i + 2 * n] = z * lc + mpMDP.S[isample].center.z;
		if(mpMDP.h_R[isample][i] < l_min[0]) l_min[0] = mpMDP.h_R[isample][i];
		if(mpMDP.h_R[isample][i] > l_max[0]) l_max[0] = mpMDP.h_R[isample][i];
		if(mpMDP.h_R[isample][i + n] < l_min[1]) l_min[1] = mpMDP.h_R[isample][i + n];
		if(mpMDP.h_R[isample][i + n] > l_max[1]) l_max[1] = mpMDP.h_R[isample][i + n];
		if(mpMDP.h_R[isample][i + 2 * n] < l_min[2]) l_min[2] = mpMDP.h_R[isample][i + 2 * n];
		if(mpMDP.h_R[isample][i + 2 * n] > l_max[2]) l_max[2] = mpMDP.h_R[isample][i + 2 * n];
		pi1 = pi;
		//std::cerr<<"P "<<mpMDP.h_R[isample][i]<<" "<<mpMDP.h_R[isample][i + n]<<" "<<mpMDP.h_R[isample][i + 2*n]<<" | "<<x<<" "<<y<<" "<<z<<"\n";//std::cin.get();
	}
	file.close();
	std::cerr<<"Min Max "<<l_min[0]<<" "<<l_min[1]<<" "<<l_min[2]<<" | "<<l_max[0]<<" "<<l_max[1]<<" "<<l_max[2]<<"\n";//std::cin.get();
	mpMDP.S[isample].size_real.x = l_max[0] - l_min[0] + 2.0* mpMDP.mdTD[0].Po.p_r;
	mpMDP.S[isample].size_real.y = l_max[1] - l_min[1] + 2.0* mpMDP.mdTD[0].Po.p_r;
	mpMDP.S[isample].size_real.z = l_max[2] - l_min[2] + 2.0* mpMDP.mdTD[0].Po.p_r;
	mpMDP.S[isample].center_real.x = 0.5*(l_max[0] + l_min[0]);
	mpMDP.S[isample].center_real.y = 0.5*(l_max[1] + l_min[1]);
	mpMDP.S[isample].center_real.z = 0.5*(l_max[2] + l_min[2]);
	mpMDP.S[isample].PN = mpMDP.PN[isample];
	
}

void h_LoadCylinderSample_f(mp_mdparameters_data& mpMDP, uint isample, char* filename)
{	
	LoadParticleDATAtxt(mpMDP.h_R[isample], mpMDP.PN[isample], filename);
	mpMDP.S[isample].PN = mpMDP.PN[isample];
}
