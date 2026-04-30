#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <cstring>

void ReadParticleDATASimple(mp_mdparameters_data & mpMDP, md_task_data &mdTD, uint isample, char namepart[], char namemusen[])
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




(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po,
	char* Name, bool flagv)
{
	double Et, Er;
	uint i, j, k, ILn = 0;
	for (i = 0; i < IL.N; ++i)
	{
		j = i / IL.IonP;
		//std::cerr << "I " << i << " " << j << " " << IL.h_IL[i] << "\n";
		if (IL.h_IL[i] < P.N && IL.h_ILtype[i]==1)
			++ILn;
	}
	//std::cin.get();
	std::ofstream file;
	file.open(Name, std::ios::out);
	file << "LAMMPS Description T0="<<Po.dt<<"  (1st line of file)\n\n";
	file << P.N << " atoms\n" << ILn << " bonds\n" << 0 << " angles\n" << 0 << " dihedrals\n" << 0 << " impropers\n";
	file << 1 << " atom types\n" << 2 << " bond types\n";
	file << S.A.x * length_const << " " << S.B.x * length_const << " xlo xhi\n" << S.A.y * length_const << " " << S.B.y * length_const << " ylo yhi\n" << S.A.z * length_const << " " << S.B.z * length_const << " zlo zhi\n";
	file << "Masses\n\n" << 1 << " " << 1.0 << "\n";
	//file << "Nonbond Coeffs\n" << 1 << " " << 1.0 << "\n" << 2 << " " << 1.0 << "\n" << 3 << " " << 1.0 << "\n" << 4 << " " << 1.0 << "\n" << 5 << " " << 1.0 << "\n";
	file << "Bond Coeffs\n\n" << 2 << " " << 1.0 << " " << 2.0 << "\n\n";
	file << "Atoms\n\n";
	for (i = 0; i < P.N; ++i)
	{
		file << i+1 << " " << "1 ";
		file << P.h_R[i] * length_const << " " << P.h_R[i + P.N] * length_const << " " << P.h_R[i + 2*P.N] * length_const << "\n";
			//<< " " << 0.5 * Po.m * P.h_V[i] * P.h_V[i] << " " << 0.5 * Po.m * P.h_V[i + P.N] * P.h_V[i + P.N]//<<"\n";
			//<< " " << P.h_V[i] << " " << P.h_V[i + P.N]
			//<< " " << P.h_F[i] << " " << P.h_F[i + P.N] << "\n";
	}
	file << "Velocities\n\n";
	for (i = 0; i < P.N; ++i)
	{
		if (flagv)
		{
			Et = 0.5 * Po.p_m * (P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N] + P.h_V[i + 2 * P.N] * P.h_V[i + 2 * P.N]) * energy_const;
			Er = 0.5 * Po.p_I * (P.h_W[i] * P.h_W[i] + P.h_W[i + P.N] * P.h_W[i + P.N] + P.h_W[i + 2 * P.N] * P.h_W[i + 2 * P.N]) * energy_const;
		}
		else
		{
			Et = 0;
			Er = 0;
		}
		
		//ek = P.h_IM[i] * (P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N]);
		file << i + 1 << " " << Et << " " << Er << " " << Et + Er << "\n";
		//std::cerr << "FI " << Padd.h_Fbound[i] << " " << Padd.h_LammpsSumF[i] << "\n"; std::cin.get();
		
		
	}
	file << "\nBonds\n\n";
	k = 1;
	for (i = 0; i < IL.N; ++i)
	{
		j = i / IL.IonP;
		if (IL.h_IL[i] < P.N && IL.h_ILtype[i] == 1)
		{			
			file << k << " " << IL.h_ILtype[i] + 1 << " " << j + 1 << " " << IL.h_IL[i] + 1 << "\n";
			++k;
		}
			
	}	
	file.close();
	//std::cerr << "Save PBM "<<"\n";
	//std::cerr << "LL " << P.N << " " << pnid2 << " | " << P.N + pnid2 << " " << P.NI << " " << j << "\n";
}
