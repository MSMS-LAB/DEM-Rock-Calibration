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
#include <chrono>
#include <float.h>
//#define defc_SaveLammps

void cuc_freePD(md_task_data& mdTD)
{
    if (mdTD.P.d_R != nullptr) { cudaFree(mdTD.P.d_R); mdTD.P.d_R = nullptr; }
    if (mdTD.P.d_F != nullptr) { cudaFree(mdTD.P.d_F); mdTD.P.d_F = nullptr; }
    if (mdTD.P.d_V != nullptr) { cudaFree(mdTD.P.d_V); mdTD.P.d_V = nullptr; }
    if (mdTD.P.d_M != nullptr) { cudaFree(mdTD.P.d_M); mdTD.P.d_M = nullptr; }
    if (mdTD.P.d_W != nullptr) { cudaFree(mdTD.P.d_W); mdTD.P.d_W = nullptr; }
	if (mdTD.P.d_Oiwt != nullptr) { cudaFree(mdTD.P.d_Oiwt); mdTD.P.d_Oiwt = nullptr; }
	if (mdTD.P.h_R != nullptr) { free(mdTD.P.h_R); mdTD.P.h_R = nullptr; }
#ifdef defc_SaveLammps
	
	if (mdTD.P.h_V != nullptr) { free(mdTD.P.h_V); mdTD.P.h_V = nullptr; }
    if (mdTD.P.h_F != nullptr) { free(mdTD.P.h_F); mdTD.P.h_F = nullptr; }
#endif // defc_SaveLammps
}

void cuc_createPD(md_task_data& mdTD)
{
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_V, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_F, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_M, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_W, 3 * mdTD.P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_Oiwt, 3 * mdTD.P.N * sizeof(float)));
    //HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_Q, P.N * sizeof(float4)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_V, 0, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_F, 0, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_W, 0, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_M, 0, 3 * mdTD.P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_Oiwt, 0, 3 * mdTD.P.N * sizeof(float)));
    //HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_Q, 0, P.N * sizeof(float4)));
	
#ifdef defc_SaveLammps
    mdTD.P.h_R = (float*)malloc(3 * mdTD.P.N * sizeof(float));
	mdTD.P.h_V = (float*)malloc(3 * mdTD.P.N * sizeof(float));
	mdTD.P.h_F = (float*)malloc(3 * mdTD.P.N * sizeof(float));
#endif // defc_SaveLammps

    
}

void cuc_freeCD(md_task_data& mdTD)
{
    if (mdTD.C.d_tmp_old != nullptr) { cudaFree(mdTD.C.d_tmp_old); mdTD.C.d_tmp_old = nullptr; mdTD.C.d_tmp = nullptr; }
    if (mdTD.C.d_IP != nullptr) { cudaFree(mdTD.C.d_IP); mdTD.C.d_IP = nullptr; }
    if (mdTD.C.d_CI != nullptr) { cudaFree(mdTD.C.d_CI); mdTD.C.d_CI = nullptr; }
    if (mdTD.C.d_CIs != nullptr) { cudaFree(mdTD.C.d_CIs); mdTD.C.d_CIs = nullptr; }
    if (mdTD.C.d_pnC != nullptr) { cudaFree(mdTD.C.d_pnC); mdTD.C.d_pnC = nullptr; }
}

void cuc_createCD(md_task_data& mdTD)
{
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_IP, mdTD.P.N * sizeof(uint)));
    d_FillIndex << < mdTD.A.bloks, SMEMDIM >> > (mdTD.C.d_IP, mdTD.P.N); //std::cerr << "A A A " << mdTD.A.bloks << "\n";
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_CI, mdTD.P.N * sizeof(uint)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_CIs, 2 * mdTD.P.N * sizeof(uint)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_pnC, 2 * mdTD.C.N * sizeof(uint)));
    mdTD.C.dtmpN_old = 128;
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_tmp_old, mdTD.C.dtmpN_old));
}

void cuc_freeID(md_task_data& mdTD)
{
    //std::cerr << "fid " << mdTD.IL.d_IL << " " << mdTD.IL.d_ILtype << " " << mdTD.IL.d_1d_iL << "\n";
    if (mdTD.IL.d_IL != nullptr) { cudaFree(mdTD.IL.d_IL); mdTD.IL.d_IL = nullptr; }
    if (mdTD.IL.d_ILtype != nullptr) { cudaFree(mdTD.IL.d_ILtype); mdTD.IL.d_ILtype = nullptr; }
    if (mdTD.IL.d_1d_iL != nullptr) { cudaFree(mdTD.IL.d_1d_iL); mdTD.IL.d_1d_iL = nullptr; }
    if (mdTD.IL.d_rij != nullptr) { cudaFree(mdTD.IL.d_rij); mdTD.IL.d_rij = nullptr; }
    if (mdTD.IL.d_Oijt != nullptr) { cudaFree(mdTD.IL.d_Oijt); mdTD.IL.d_Oijt = nullptr; }
    if (mdTD.IL.d_Fijn != nullptr) { cudaFree(mdTD.IL.d_Fijn); mdTD.IL.d_Fijn = nullptr; }
    if (mdTD.IL.d_Fijt != nullptr) { cudaFree(mdTD.IL.d_Fijt); mdTD.IL.d_Fijt = nullptr; }
    if (mdTD.IL.d_Mijn != nullptr) { cudaFree(mdTD.IL.d_Mijn); mdTD.IL.d_Mijn = nullptr; }
    if (mdTD.IL.d_Mijt != nullptr) { cudaFree(mdTD.IL.d_Mijt); mdTD.IL.d_Mijt = nullptr; }
    if (mdTD.IL.d_Mijadd != nullptr) { cudaFree(mdTD.IL.d_Mijadd); mdTD.IL.d_Mijadd = nullptr; }
    
}

void cuc_createID(md_task_data& mdTD)
{
    //uint_fast64_t stmp = IL.N * sizeof(uint) + IL.N * sizeof(uint) + IL.N * sizeof(float) + 7 * IL.N * sizeof(float3);
    //std::cerr << "InteractionListInit " << IL.N << " " << stmp << " " << stmp / (1024 * 1024) << " | " << sqrt(IL.aacut) << "\n";
    //std::cin.get();
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_IL), mdTD.IL.N * sizeof(uint)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_ILtype), mdTD.IL.N * sizeof(uint)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_1d_iL), mdTD.IL.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_rij), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Oijt), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Fijn), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Fijt), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Mijn), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Mijt), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Mijadd), mdTD.IL.N * sizeof(float3)));

    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_ILtype, 0, mdTD.IL.N * sizeof(uint)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_1d_iL, 0, mdTD.IL.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_rij, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Oijt, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Fijn, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Fijt, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Mijn, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Mijt, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Mijadd, 0, mdTD.IL.N * sizeof(float3)));
#ifdef defc_SaveLammps
    mdTD.IL.h_IL = (uint*)malloc(mdTD.IL.N * sizeof(uint));
    mdTD.IL.h_ILtype = (uint*)malloc(mdTD.IL.N * sizeof(uint));
#endif // defc_SaveLammps

    
    //mdTD.IL.h_1d_iL = (float*)malloc(mdTD.IL.N * sizeof(float));
}

void h_CalculationUniaxialCompression(md_task_data& mdTD, float *h_R)
{
    char filename[256] = "";
    uint steps, l_i, l_j;
    //std::cerr << "HP " << mdTD.S.hidenpoint.x << " " << mdTD.S.hidenpoint.y << " " << mdTD.S.hidenpoint.z << "\n"; std::cin.get();

    mdTD.IL.CalculateParameters(mdTD.P.N, mdTD.Po.a_farcut);
    mdTD.C.CalculateParameters(mdTD.S.B.x, mdTD.S.B.y, mdTD.S.B.z);
    mdTD.A.ibloks = ceil(mdTD.IL.N / (SMEMDIM)) + 1;
    mdTD.A.CalculateParameters(mdTD.P.N, mdTD.IL.N);
	mdTD.Po.CalculateParameters();
	//std::cerr<<"Q0 "<<mdTD.Po.Trayleigh<<" "<<mdTD.Po.Tsim<<"\n";
	mdTD.Po.vis = 0.01*0.002*2.0*sqrt(mdTD.Po.p_m*mdTD.Po.m_E*mdTD.Po.b_S/mdTD.Po.b_d);
	std::cerr<<"Viscos "<<mdTD.Po.vis<<"\n";
    mdTD.Po.vism = mdTD.Po.vis * mdTD.Po.p_m;
	mdTD.Po.visI = mdTD.Po.vis * mdTD.Po.p_I;
    //std::cerr << "HP " << mdTD.S.hidenpoint.x << " " << mdTD.S.hidenpoint.y << " " << mdTD.S.hidenpoint.z << "\n"; std::cin.get();
    cuc_freePD(mdTD); //std::cerr << "q1\n";
    cuc_createPD(mdTD); //std::cerr << "q2\n";
    cuc_freeCD(mdTD); //std::cerr << "q3\n";
    cuc_createCD(mdTD); //std::cerr << "q4\n";
    cuc_freeID(mdTD); //std::cerr << "q5\n";
    cuc_createID(mdTD); //std::cerr << "D1\n";//std::cerr << "q6\n";
    //mdTD.R.dNsave = 10; 
	mdTD.R.Nsave = 1.5*mdTD.Compress.MaxStrain*mdTD.Compress.size_d.z/(fabs(mdTD.Compress.VLoad)*mdTD.Po.dt*mdTD.R.dNsave);  
	//std::cerr<<"Deform "<<mdTD.R.dNsave<<" "<<mdTD.R.Nsave<<" "<<mdTD.Compress.MaxStrain<<" "<<mdTD.Compress.size_d.z<<"\n";std::cin.get();
    ResultsUInit(mdTD.P, mdTD.A, mdTD.S, mdTD.C, mdTD.IL, mdTD.R, mdTD.Compress); //std::cerr << "D2\n"; std::cerr << mdTD.P.d_R<<" "<<h_R<<" "<< mdTD.P.N <<"\n";
    HANDLE_ERROR(cudaMemcpy(mdTD.P.d_R, h_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyHostToDevice)); //std::cerr << "D3\n";

    CellDistribution(mdTD.P, mdTD.A, mdTD.S, mdTD.C); //std::cerr << "D4\n";
    InteractionListConstruct(mdTD.P, mdTD.A, mdTD.S, mdTD.IL, mdTD.C); //std::cerr << mdTD.IL.N << " " << mdTD.IL.IonP << "\n"; std::cerr << "D5\n";
    d_ConstructBoundInteractions << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W,
        mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt, mdTD.IL.d_Fijn, mdTD.IL.d_Fijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.IL.d_Mijadd, mdTD.P.N, mdTD.IL.IonP, mdTD.Po.b_dd, mdTD.Po.b_r); //std::cerr << mdTD.IL.N<<" "<< mdTD.Po.b_dd << "\n"; std::cerr << "D6\n";
    //std::cerr<<"Q1.0\n";
	/*
	mdTD.P.h_R = (float*)malloc(3 * mdTD.P.N * sizeof(float));
	mdTD.IL.h_IL = (uint*)malloc(mdTD.IL.N * sizeof(uint));
    mdTD.IL.h_ILtype = (uint*)malloc(mdTD.IL.N * sizeof(uint));
	mdTD.IL.h_1d_iL = (float*)malloc(mdTD.IL.N * sizeof(float));
	HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_1d_iL, mdTD.IL.d_1d_iL, mdTD.IL.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
    //for (i = 0; i < mdTD.IL.N; ++i)
    //{
    //    if (i % 32 == 0)std::cerr << "\n"<<i/32<<" ";
    //    std::cerr << i << " " << mdTD.IL.h_IL << "(" << mdTD.IL.h_ILtype << ") ";
    //}
	std::ifstream file;
	uint l_n, l_IMi, l_IMj, l_Ij, l_It;
	double l_IMd, l_Id, l_d;
	double3 l_ri, l_rj;
	file.open("./result/InteractionMusen_0_0.txt", std::ios::in);
	file >> l_n;
	std::cerr<<"NInteractions "<<l_n<<" | "<<mdTD.P.N<<"\n";
	for (uint l_i = 0; l_i < l_n; ++l_i)
	{
		file >> l_IMi >> l_IMj >> l_IMd;
		l_IMi -= 24;
		l_IMj -= 24;
		l_IMd /= length_const;
		l_ri.x = mdTD.P.h_R[l_IMi]; l_ri.y = mdTD.P.h_R[l_IMi + mdTD.P.N]; l_ri.z = mdTD.P.h_R[l_IMi + 2*mdTD.P.N];
		l_rj.x = mdTD.P.h_R[l_IMj]; l_rj.y = mdTD.P.h_R[l_IMj + mdTD.P.N]; l_rj.z = mdTD.P.h_R[l_IMj + 2*mdTD.P.N];
		l_d = sqrt((l_rj.x-l_ri.x)*(l_rj.x-l_ri.x) + (l_rj.y-l_ri.y)*(l_rj.y-l_ri.y) + (l_rj.z-l_ri.z)*(l_rj.z-l_ri.z));
		for (l_j = 0; l_j < mdTD.IL.IonP; ++l_j)
		{
			l_Ij = mdTD.IL.h_IL[mdTD.IL.IonP*l_IMi + l_j];
			l_It = mdTD.IL.h_ILtype[mdTD.IL.IonP*l_IMi + l_j];
			l_Id = 1.0/mdTD.IL.h_1d_iL[mdTD.IL.IonP*l_IMi + l_j];
			if(l_Ij == l_IMj) break;
			//std::cerr<<"I "<<l_IMi<<" "<<l_Ij<<" "<<l_It<<" "<<l_Id<<"\n";
		}
		if(l_j<mdTD.IL.IonP && fabs(l_IMd-l_Id)<1e-3)
		{
			//std::cerr<<"Find Interaction "<<l_i<<" "<<l_IMi<<" "<<l_IMj<<" "<<l_IMd<<" "<<l_It<<" "<<l_Id<<"\n"; 
			//std::cin.get();
		} else {
			std::cerr<<"Not Find Interaction "<<l_j<<" | "<<l_i<<" "<<l_IMi<<" "<<l_IMj<<" "<<l_IMd<<" "<<l_It<<" "<<l_Id<<" | "<<l_IMd-l_Id<<"\n";
			std::cerr<<"P "<<l_ri.x<<" "<<l_ri.y<<" "<<l_ri.z<<" | "<<l_rj.x<<" "<<l_rj.y<<" "<<l_rj.z<<" | "<<l_d<<"\n";
			std::cin.get();			
		}
		//std::cin.get();
		//std::cerr<<"P "<<mpMDP.h_R[isample][i]<<" "<<mpMDP.h_R[isample][i + n]<<" "<<mpMDP.h_R[isample][i + 2*n]<<" | "<<x<<" "<<y<<" "<<z<<"\n";//std::cin.get();
	}
	file.close();
	exit(0);/**/
    d_DeleteFarLinks << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.N, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt,
        mdTD.IL.d_Fijn, mdTD.IL.d_Fijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.IL.d_Mijadd, mdTD.IL.IonP, mdTD.Po.aa_farcut); //std::cerr << "D7\n";
	//std::cerr<<"Q2.0\n";
    auto begin = std::chrono::high_resolution_clock::now();
    auto end = begin;
    std::chrono::nanoseconds dr;
    //std::cin.get();
	//std::cerr<<"Q1 "<<mdTD.Po.Trayleigh<<" "<<mdTD.Po.Tsim<<"\n";
    uint nloaddelay = 0;
    mdTD.R.stepsave = 0;
	//firerelax_data Fire;
	//SetFIREData(mdTD.P, mdTD.Po, Fire);
    //Fire.MaxStepsRelaxation =50000*int(Fire.Tsim/Fire.dt0);
	//Fire.StepsPreRelaxation = 0*10000*int(Fire.Tsim/Fire.dt0);	
	//Fire.NPpositive = 0;
	//Fire.NPnegative = 0;
	//Fire.dt0 *= 0.1;
	//Fire.dt = Fire.dt0;
	//Fire.h_FdotV = (float*)malloc(Fire.bloks4 * sizeof(float));
	//HANDLE_ERROR(cudaMalloc((void**)&Fire.d_FdotV, Fire.bloks4 * sizeof(float)));
	//Fire.h_MdotW = (float*)malloc(Fire.bloks4 * sizeof(float));
	//HANDLE_ERROR(cudaMalloc((void**)&Fire.d_MdotW, Fire.bloks4 * sizeof(float)));
	//std::cerr<<"Q "<<Fire.MaxStepsRelaxation<<" "<<Fire.Tsim<<" "<<Fire.dt0<<"\n";
    //HANDLE_ERROR(cudaMemcpy(P.d_V, P.h_V, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy(P.d_W, P.h_W, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
    //std::cerr << "Z " << Compress.Zb << " " << Compress.Zt << " " << A.ibloks * SMEMDIM << " " << IL.N << "\n";
    //for (uint i = 0; i < 10000 && !mdTD.Compress.Fracture; ++i)//22.8374ms
    //std::cerr << "HP " << mdTD.S.hidenpoint.x << " " << mdTD.S.hidenpoint.y << " " << mdTD.S.hidenpoint.z << "\n"; std::cin.get();
	double RelaxationEps, RelaxationShift, RelaxationVelocity=0, RelaxationForce, _1d_Mass = 1.0/(mdTD.Po.p_m*mdTD.P.N);
	uint step_t=0, l_stepsave;
    for (steps = 0; fabs(mdTD.Compress.Strain) < mdTD.Compress.MaxStrain && !mdTD.Compress.Fracture; ++steps)	
	//for (steps = 0; fabs(mdTD.Compress.Strain) < mdTD.Compress.MaxStrain; ++steps)
    {
#ifdef defc_SaveLammps
        if (steps % 10000 == 0)
        {
            HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(mdTD.P.h_V, mdTD.P.d_V, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(mdTD.P.h_F, mdTD.P.d_F, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
			double3 l_sumF={0,0,0}, l_sumV={0,0,0};
			for(uint l_i=0; l_i<mdTD.P.N; ++l_i){l_sumF.x += mdTD.P.h_F[l_i]; l_sumF.y += mdTD.P.h_F[l_i+mdTD.P.N]; l_sumF.z += mdTD.P.h_F[l_i+2*mdTD.P.N];
			l_sumV.x += mdTD.P.h_V[l_i]; l_sumV.y += mdTD.P.h_V[l_i+mdTD.P.N]; l_sumV.z += mdTD.P.h_V[l_i+2*mdTD.P.N];}
            sprintf(filename, "./result/steps/LAMMPS/CPu_%li.txt", steps);
            SaveLammpsDATASimple(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL, mdTD.Po, filename, false);
            std::cerr << "Stress " << steps << " " << mdTD.Compress.Stress * stress_const * 1e-6 << " " << mdTD.Compress.Stress << "\n";
			std::cerr << "Forse " << steps << " " << l_sumF.x * force_const << " " << l_sumF.y * force_const << " " << l_sumF.z * force_const << "\n";
			std::cerr << "Velocity " << steps << " " << l_sumV.x * velocity_const << " " << l_sumV.y * velocity_const << " " << l_sumV.z * velocity_const << " | " << mdTD.Compress.V << "\n";
			std::cerr<< "Relaxation " << -(mdTD.R.sFL[0]+mdTD.R.sFL[1]) <<" "<< RelaxationShift << " " << RelaxationEps << "\n";
			//std::cin.get();
		}
#endif // defc_SaveLammps        
        CellDistribution(mdTD.P, mdTD.A, mdTD.S, mdTD.C); //std::cerr << "D8\n";
        InteractionListReConstructBPM(mdTD.P, mdTD.A, mdTD.S, mdTD.IL, mdTD.C); //std::cerr << "D9\n";
        HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_F, 0, 3 * mdTD.P.N * sizeof(float)));
        HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_M, 0, 3 * mdTD.P.N * sizeof(float)));
        d_CalculateForcesDEM_22 << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL,
            mdTD.IL.d_rij, mdTD.IL.d_Oijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.P.d_F, mdTD.P.d_M, mdTD.P.N, mdTD.IL.IonP, mdTD.Po.dt, mdTD.Po.m_E, mdTD.Po.m_G, mdTD.Po.b_r, mdTD.Po.m_Ec, mdTD.Po.m_Gc);// std::cerr << "D10\n";
        d_CalculateForcesDEM_21 << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt, mdTD.P.d_F, mdTD.P.d_M,
            mdTD.P.N, mdTD.IL.IonP, mdTD.Po.dt, mdTD.Po.hm_E, mdTD.Po.hm_G, mdTD.Po.p_A, mdTD.Po.p_mu, mdTD.Po.p_mur, mdTD.Po.p_m, mdTD.Po.p_r); //std::cerr << "D11\n";

               
        //d_UniaxialCompression3_simple << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_F, mdTD.R.d_FL, mdTD.P.N, mdTD.Compress.center, mdTD.Compress.RR,
        //    mdTD.Compress.Top, mdTD.Compress.Bottom, mdTD.Compress.TopR, mdTD.Compress.BottomR, mdTD.Compress.V, mdTD.S.center.z + mdTD.S.spacesized2.z); //std::cerr << "D12\n";//H0.0011ms
		d_UniaxialCompression_hm << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W, mdTD.P.d_Oiwt, mdTD.P.d_F, mdTD.P.d_M,	
			mdTD.P.N, mdTD.Po.dt, 2.0*mdTD.Po.hm_E, 2.0*mdTD.Po.hm_G, mdTD.Po.p_A, mdTD.Po.p_mu, mdTD.Po.p_mur, mdTD.Po.p_m, mdTD.Po.p_r, 	
			mdTD.R.d_FL, mdTD.Compress.center, mdTD.Compress.RR, mdTD.Compress.Top, mdTD.Compress.Bottom, mdTD.Compress.V, mdTD.S.center.z + mdTD.S.spacesized2.z);
        
		if (steps % mdTD.R.dNsave == 0)
		{
			d_SumUpInteractions << <mdTD.IL.IonP*mdTD.A.bloks, SMEMDIM >> > (mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.IonP, mdTD.R.d_SI, mdTD.IL.IonP*mdTD.P.N);
			SumInteractions(mdTD.R, mdTD.IL, steps);
			l_stepsave = mdTD.R.stepsave;
			//std::cerr<<"Interactions: "<<mdTD.R.h_sSI[0]<<"\n"; exit(0);
		}
		SumForcesULoading(mdTD.R, mdTD.Compress, steps); //std::cerr << "D12.5 "<<steps<<"\n";//H0.4438ms
        //if(steps >10000)mdTD.Compress.V=0;
		d_CalculateIncrementsDEM << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, mdTD.P.d_V, mdTD.P.d_R, mdTD.P.d_M, mdTD.P.d_W, mdTD.P.N, mdTD.Po.dt_d_m, mdTD.Po.dt, mdTD.Po.dt_d_I); //std::cerr << "D13\n";//H0.0533ms
		/*if(fabs(mdTD.Compress.V)>1e-10 || steps < 10000)
			d_CalculateIncrementsDEM << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, mdTD.P.d_V, mdTD.P.d_R, mdTD.P.d_M, mdTD.P.d_W, mdTD.P.N, mdTD.Po.dt_d_m, mdTD.Po.dt, mdTD.Po.dt_d_I); //std::cerr << "D13\n";//H0.0533ms
        else
		{
			//cudaMemset(mdTD.P.d_W, 0, 3 * mdTD.P.N * sizeof(float));
			//CalculateRelaxDEMFIRE(mdTD, Fire);
			//d_CalculateIncrementsDEMViscos << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, mdTD.P.d_V, mdTD.P.d_R, mdTD.P.d_M, mdTD.P.d_W, mdTD.P.N, mdTD.Po.dt_d_m, mdTD.Po.dt, mdTD.Po.dt_d_I, mdTD.Po.vism, mdTD.Po.visI);
			RelaxationForce = -(mdTD.R.sFL[0]+mdTD.R.sFL[1]);
			RelaxationVelocity += RelaxationForce*_1d_Mass*mdTD.Po.dt;
			if(RelaxationVelocity * RelaxationForce < 0) RelaxationVelocity = 0;
			RelaxationShift = RelaxationVelocity*mdTD.Po.dt;
			//RelaxationShift = -10*_1d_Mass*mdTD.Po.dt;
			RelaxationEps = RelaxationShift/mdTD.S.spacesized2.z;
			d_DeformSampleRelaxationZ << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.N, mdTD.S.center, RelaxationEps, RelaxationShift);
			d_CalculateIncrementsDEM << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, mdTD.P.d_V, mdTD.P.d_R, mdTD.P.d_M, mdTD.P.d_W, mdTD.P.N, mdTD.Po.dt_d_m, mdTD.Po.dt, mdTD.Po.dt_d_I);

		}/**/
        d_ParallelepipedCutRestriction << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_F, mdTD.P.N, mdTD.S.center, mdTD.S.spacesized2, mdTD.S.hidenpoint); //std::cerr << "D14\n";//H0.055ms
        d_DeleteFarLinks << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.N, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt,
            mdTD.IL.d_Fijn, mdTD.IL.d_Fijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.IL.d_Mijadd, mdTD.IL.IonP, mdTD.Po.aa_farcut); //std::cerr << "D15\n";//H0.3451ms
        
        //mdTD.Compress.CompressU_S(mdTD.Compress.Stress, mdTD.Po);
		//mdTD.Compress.CompressU_S2(mdTD.R, mdTD.Po, steps);
		mdTD.Compress.CompressU_S3(mdTD.R, mdTD.Po, steps);
        if (steps % 10000 == 0)
            std::cerr << "U " << mdTD.Compress.center_d.z << " " << mdTD.Compress.size_d.z << " " << 1e+3 * (mdTD.Compress.size_d.z - mdTD.R.sz0) / mdTD.R.sz0
            << " | " << mdTD.R.sFL[0] * force_const << " N " << mdTD.R.sFL[1] * force_const << " N | "
            << mdTD.R.sFL[0] * mdTD.Compress._1d_Area * stress_const * 1e-6 << " MPa " << mdTD.R.sFL[1] * mdTD.Compress._1d_Area * stress_const * 1e-6 << " MPa | "
			<< "I " << mdTD.R.h_sSI[l_stepsave] << " | "
            << mdTD.S.hidenpoint.x << " " << mdTD.S.hidenpoint.y << " " << mdTD.S.hidenpoint.z << " | " << mdTD.R.sz0 << " " << step_t << " " << steps << "\n";
        //if (i % 100000 == 0) std::cerr << "UU " << Compress.V << " " << Po.dt << " " << Compress.size.z << " " << Compress.sized2.z << " " << Compress.Hd2 << "\n";
        //std::cin.get();
		++step_t;
    }
	CalculateLinearityDeviation(mdTD.R, mdTD.Compress);
	CalculateDestruction(mdTD.R);
	std::cerr<<"RU "<<abs(mdTD.Compress.Strain) << " " << mdTD.Compress.MaxStrain << " " <<  !mdTD.Compress.Fracture 
	<< " | " << mdTD.R.sFL[0] * mdTD.Compress._1d_Area * stress_const * 1e-6 << " MPa " << mdTD.R.sFL[1] * mdTD.Compress._1d_Area * stress_const * 1e-6 << " MPa | " << mdTD.R.Destruction<< "\n";
    //std::cerr<< "Relaxation " << -(mdTD.R.sFL[0]+mdTD.R.sFL[1]) <<" "<< RelaxationShift << " " << RelaxationEps << "\n";
	end = std::chrono::high_resolution_clock::now();
    dr = end - begin;
    double calctime = std::chrono::duration_cast<std::chrono::milliseconds>(dr).count();
    std::cerr << "Fin time " << calctime << "ms " << calctime * 1e-3 / 60.0
        << "min " << calctime * 1e-4 << "ms" << " " << 22.2632 - calctime * 1e-4 << "ms "
        "faster " << 22.2632 / (calctime * 1e-4) << " \n";
    //std::cin.get();
    begin = end;
    sprintf(filename, "./result/Uniaxial_%i.txt", mdTD.R.fileN);
    SaveSumForcesLoading(mdTD.R, filename);
    std::cerr << "FIN CALCULATION!\n";

	/*HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
	double l_max[3]={-DBL_MAX,-DBL_MAX,-DBL_MAX}, l_min[3]={DBL_MAX, DBL_MAX, DBL_MAX};
	for (uint l_i = 0; l_i < mdTD.P.N; ++l_i)
	{
		if(mdTD.P.h_R[l_i] < l_min[0]) l_min[0] = mdTD.P.h_R[l_i];
		if(mdTD.P.h_R[l_i] > l_max[0]) l_max[0] = mdTD.P.h_R[l_i];
		if(mdTD.P.h_R[l_i + mdTD.P.N] < l_min[1]) l_min[1] = mdTD.P.h_R[l_i + mdTD.P.N];
		if(mdTD.P.h_R[l_i + mdTD.P.N] > l_max[1]) l_max[1] = mdTD.P.h_R[l_i + mdTD.P.N];
		if(mdTD.P.h_R[l_i + 2 * mdTD.P.N] < l_min[2]) l_min[2] = mdTD.P.h_R[l_i + 2 * mdTD.P.N];
		if(mdTD.P.h_R[l_i + 2 * mdTD.P.N] > l_max[2]) l_max[2] = mdTD.P.h_R[l_i + 2 * mdTD.P.N];
		//std::cerr<<"P "<<mpMDP.h_R[isample][i]<<" "<<mpMDP.h_R[isample][i + n]<<" "<<mpMDP.h_R[isample][i + 2*n]<<" | "<<x<<" "<<y<<" "<<z<<"\n";//std::cin.get();
	}
	std::cerr<<"Actual Sample Size: "<<(l_max[0]-l_min[0])*length_const<<" "<<(l_max[1]-l_min[1])*length_const<<" "<<(l_max[2]-l_min[2])*length_const<<" | "<<((l_max[0]-l_min[0])+(l_max[1]-l_min[1]))*0.5*length_const/0.025<<" "<<(l_max[2]-l_min[2])*length_const/0.05<<"\n";
	/**/
    cuc_freePD(mdTD); //std::cerr << "q1\n";
    cuc_freeCD(mdTD); //std::cerr << "q3\n";
    cuc_freeID(mdTD); //std::cerr << "q5\n";
	//free(Fire.h_FdotV);
	//Fire.h_FdotV = nullptr;
	//cudaFree(Fire.d_FdotV);
	//Fire.d_FdotV = nullptr;
	//free(Fire.h_FdotV);
	//Fire.h_MdotW = nullptr;
	//cudaFree(Fire.d_MdotW);
	//Fire.d_MdotW = nullptr;
    ResultsUDelete(mdTD.R);
    //R.Nsave = 1000000;
    //R.dNsave = 1;
    

    

    //std::cerr << "T " << P.N << " " << IL.N << " " << IL.IonP << " " << A.bloks << " " << SMEMDIM << "\n";    
    //std::cerr << "Cut " << Compress.Zt << " " << Compress.Zb << " | " << S.B_real.z << " " << S.A_real.z << " " << Po.p_r << " | " <<Compress.Area*length_const*length_const << "\n";
    //d_CutCylinderSpecimen_simple << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.N, Compress.center, Compress.sized2.x* Compress.sized2.x, Compress.Hd2, S.hidenpoint);
    //std::cin.get();
    //CellDistributionInit(P, A, S, C);
    //std::cerr << "Bound " << Po.b_d << " " << Po.b_dd << " " << sqrt(Po.b_dd) << "\n";       
    //std::cin.get();
    //std::cerr << "AAAA\n";
    //HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(IL.h_1d_iL, IL.d_1d_iL, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
    /*uint i, j, ILn = 0, ILn0 = 0, Pn = 0;
    for (i = 0; i < IL.N; ++i)
    {
        j = i / IL.IonP;
        if (IL.h_IL[i] < P.N && IL.h_ILtype[i] == 1)
            ++ILn;
        if (IL.h_IL[i] < P.N && IL.h_ILtype[i] == 0)
            ++ILn0;
    }
    for (i = 0; i < P.N; ++i)    
        if (P.h_R[i + 2 * P.N] < S.center.z + S.spacesized2.z)
            ++Pn;
    
    std::cerr << "Interactions Number " << ILn << " " << ILn0 << " " << Pn << "\n";/**/
    //std::cin.get();
    //double Vm = 0.05 * 0.0125 * 0.0125 * MC_pi, Vmpp = Vm / 3574.0, pm = Po.p_V * 3574.0 * volume_const / Vm, Vr = (S.size_real.z + 2.0 * Po.p_r) * (S.size_real.x + 2.0 * Po.p_r) * (S.size_real.x + 2.0 * Po.p_r) * 0.25 * MC_pi,
    //    Vrpp = Vr / double(Pn), Vs = (S.L.x + 2.0 * Po.p_r) * (S.L.y + 2.0 * Po.p_r) * (S.L.z + 2.0 * Po.p_r);
    //std::cerr << "Sample Vm=" << Vm << " Vmpp=" << Vmpp << " pm=" << pm << " Vr=" << Vr * volume_const << " Vrpp=" << Vrpp * volume_const << " " << Po.p_V * volume_const << " " << Vr / Vs << "\n";
    //std::cin.get();
    //std::cerr << "AAAA1\n";
    /*for (int k = 0; k < IL.N; ++k)
    {
        uint i = k/IL.IonP, j = IL.h_IL[k];
        float3 r;
        double rm;
        if (IL.h_ILtype[k] == 1)
        {
            r.x = P.h_R[j] - P.h_R[i];
            r.y = P.h_R[j + P.N] - P.h_R[i + P.N];
            r.z = P.h_R[j + 2 * P.N] - P.h_R[i + 2 * P.N];
            rm = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
            std::cerr << "I " << k << " " << i << " " << j << " " << IL.h_1d_iL[k] << " " << 1.0 / IL.h_1d_iL[k]
                << " | " << r.x << " " << r.y << " " << r.z << " " << rm << "\n";
        }
    }
    std::cin.get();/**/
    /*std::cerr << "Interaction list " << IL.N << "\n";
    cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
    for (int i = 0; i < IL.N; ++i)
    {
        if (i % IL.IonP == 0)
            std::cerr << "\n" << i / IL.IonP;// << " (" << i % IL.IonP << ")";
        if (IL.h_IL[i] != UINT_MAX)
            std::cerr << " " << IL.h_IL[i] << "&" << int(IL.h_ILtype[i]);
        else if (IL.h_IL[i] < UINT_MAX && IL.h_IL[i] >= P.N)
            std::cerr << "!!!!";
        else if (IL.h_IL[i] == UINT_MAX)        
            std::cerr << " M";       
    }
    std::cin.get();/**/
    //float3* tempIL_Rij = (float3*)malloc(IL.N * sizeof(float3));
    //float3* tempIL_Oijt = (float3*)malloc(IL.N * sizeof(float3));
    //float3* tempIL_Mijn = (float3*)malloc(IL.N * sizeof(float3));
    //float3* tempIL_Mijt = (float3*)malloc(IL.N * sizeof(float3));
    
    //std::cin.get();
    
       //std::cin.get();
    /*RenewInteractionList_full(P, C, S, A, IL);
    std::cerr << "FIn2\n";
    HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
    sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", 2);
    SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);
    //std::cin.get();
    RenewInteractionList_full(P, C, S, A, IL);
    std::cerr << "FIn3\n";
    HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
    sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", 3);
    SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);/**/
    /*CellDistributionInit(P, A, S, C);
    CellDistribution(P, A, S, C);
    InteractionListInit(P, A, S, IL);
    InteractionListConstruct(P, A, S, IL, C);*/
    //std::cin.get();
    //cudaError_t cudaStatus;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    

    
}

