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
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <time.h>
//#include <cudpp.h>
//#include <cudpp_plan.h>
__global__ void d_CalculateIncrementsFIRE(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const uint N, const float _1d_Mass_m_dt, const float dt, const float F_alpha)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	float vv, f, v_d_f;
	float3 v;

	while (idx < N)
	{		
	//Leapfrog
		v.x = __fmul_rn(F[idx], _1d_Mass_m_dt);
		v.y = __fmul_rn(F[idx + N], _1d_Mass_m_dt);
		v.z = __fmul_rn(F[idx + 2 * N], _1d_Mass_m_dt);

		vv = v.x * v.x + v.y * v.y + v.z * v.z;
		f = F[idx] * F[idx] + F[idx + N] * F[idx + N] + F[idx + 2 * N] * F[idx + 2 * N];
		if (f > 1e-12f)
		{
			v_d_f = F_alpha * __fsqrt_rn(vv * __frcp_rn(f));
		}
		else
		{
			v_d_f = F_alpha * __fsqrt_rn(vv) * 1e6;
		}
		v.x = (1.0f - F_alpha) * v.x + v_d_f * F[idx];
		v.y = (1.0f - F_alpha) * v.y + v_d_f * F[idx + N];
		v.z = (1.0f - F_alpha) * v.z + v_d_f * F[idx + 2 * N];
		//printf("IC %u %e %e %e |  %e %e %e | %e %e %e\n", idx, v.x, v.y, v.z, F_alpha, v_d_f, vv, F[idx], F[idx + N], F[idx + 2 * N]);
		V[idx] = v.x;
		V[idx + N] = v.y;
		V[idx + 2 * N] = v.z;
		//printf("IC %u %e %e %e %e %e %e\n", idx, v.x, v.y, v.z, R[idx], R[idx + N], R[idx + 2 * N]);
		R[idx] += __fmul_rn(v.x, dt);
		R[idx + N] += __fmul_rn(v.y, dt);
		R[idx + 2 * N] += __fmul_rn(v.z, dt);
		
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_CalculateDecrementsHalfStepFIRE(const float* __restrict__ V, float* __restrict__ R, const uint N, const float dt_d2)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//float dt_d2 = __fmul_rn(dt, 0.5f);
	while (idx < N)
	{
		//Leapfrog
		R[idx] -= __fmul_rn(V[idx], dt_d2);
		R[idx + N] -= __fmul_rn(V[idx + N], dt_d2);
		R[idx + 2 * N] -= __fmul_rn(V[idx + 2 * N], dt_d2);
		idx += blockDim.x * gridDim.x;
	}
}

void SetFIREData(particle_data& P, potential_data& Po, firerelax_data& Fire)
{
	Fire.bloks4 = P.N / (4 * SMEMDIM) + 1;
	Fire.dt0 = Po.dt;
	Fire.dtmax = 100.0 * Fire.dt0;
	Fire.dtmin = 0.001 * Fire.dt0;
	Fire.dtgrow = 1.1;
	Fire.dtshrink = 0.5;
	Fire.alpha0 = 0.25;//0.25
	Fire.alpha = Fire.alpha0;
	Fire.alphashrink = 0.99;
	Fire.NPnegativeMax = 30000;//30000
	Fire.Ndelay = 20;
	Fire.Tsim = (Po.Trayleigh>Po.Tsim) ?  Po.Tsim:Po.Trayleigh;
}

void CalculateGPUStepsContractRelaxFIRE(particle_data &P, cell_data &C, sample_data &S, additional_data &A, interaction_list_data &IL, potential_data &Po, firerelax_data &Fire)
{
	/*cudaEvent_t start, stop;
	float gpuTime;
	double gpuTimeAver = 0;/**/
	char filename[256] = "";
	std::cerr << "FIRE Start\n";
	Fire.NPpositive = 0;
	Fire.NPnegative = 0;
	Fire.dt = Fire.dt0;
	Fire.h_FdotV = (float*)malloc(Fire.bloks4 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Fire.d_FdotV, Fire.bloks4 * sizeof(float)));
	double l_CoeffSampleSize = 0.85;
	S.Vgenmax *= 1e3;
	S.sized2.x *= l_CoeffSampleSize;
	S.sized2.y *= l_CoeffSampleSize;
	S.sized2.z *= l_CoeffSampleSize;
	double ILacutmax = IL.acut;
	IL.acut *= 1e-8;
	IL.aacut = IL.acut * IL.acut;
	Po.D = 5e-4 * Po.m_E * Po.b_S / (2.0 * Po.p_r);
	Po.a = 1e-8;
	double Dmax = 1e+3 * Po.m_E * Po.b_S / (2.0 * Po.p_r);
	uint steps, i;
	double timereal = 0;

	RenewInteractionList_full(P, C, S, A, IL);
	//std::cerr << "STEP 00\n"; std::cin.get();
	//HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(C.h_CI, C.d_CI, P.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(C.h_CIs, C.d_CIs, 2 * P.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(C.h_pnC, C.d_pnC, 2 * C.N * sizeof(uint), cudaMemcpyDeviceToHost));
	thrust::device_ptr<float> dtp_1d_Rijm(IL.d_1d_iL);
	//std::cin.get();
	/*cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
	for (int i = 0; i < IL.N; ++i)
	{
		if (i % IL.IonP == 0)std::cerr << "\n" << i / IL.IonP;
		if (IL.h_IL[i] != UINT_MAX)
			std::cerr << " " << IL.h_IL[i];
		else if (IL.h_IL[i] < UINT_MAX && IL.h_IL[i] >= P.N)std::cerr << "!!!!";
		else if (IL.h_IL[i] == UINT_MAX)
		{
			std::cerr << " " << "M";
			continue;
		}
	}
	std::cerr << "\n";/**/
	//h_CheckInteractions(P, C, A, S, IL);
	//std::cin.get();
	for (steps = 0; steps < 300000; ++steps)
	{
		if (steps % 2000 == 0)
		{
			HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
			sprintf(filename, "./result/steps/LAMMPS/CPp_%li.txt", steps);
			SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);
		}
		//if (steps % 100 == 0 && Po.D < 1e-10*Dmax)
		//	Po.D *= 1.2;
		if (steps % 200 == 0 && Po.a < 2.0 * Po.p_r)
		{
			Po.a *= 1.5;
			Po._1d_a = 1.0 / Po.a;
			Po.aa = Po.a * Po.a;
		}
		else if(Po.a > 2.0 * Po.p_r)
		{
			Po.a = 2.0 * Po.p_r;
			Po._1d_a = 1.0 / Po.a;
			Po.aa = Po.a * Po.a;
		}

		if (steps % 200 == 0 && IL.acut < ILacutmax)
		{
			IL.acut *= 1.5;
			if (IL.acut > ILacutmax)IL.acut = ILacutmax;
			IL.aacut = IL.acut * IL.acut;
		}/**/

		RenewInteractionList_New(P, C, S, A, IL);
		

		d_CalculateForces << <A.bloks, SMEMDIM >> > (P.d_R, P.d_F, P.N, IL.d_IL, IL.d_1d_iL, IL.IonP, Po.D, Po.aa);
		//d_CalculateIncrements << <A.bloks, SMEMDIM >> > (P.d_F, P.d_V, P.d_R, P.N, Po.dt_d_m, Fire.dt);
		d_CalculateIncrementsViscos << <A.bloks, SMEMDIM >> > (P.d_F, P.d_V, P.d_R, P.N, Po.dt_d_m, Fire.dt, Po.vism);
		//d_CylinderRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.axis, S.center, S.R0, S.H0);		
		
		//d_CalculateIncrementsFIRE << <A.bloks, SMEMDIM >> > (P.d_F, P.d_V, P.d_R, P.N, Po.dt_d_m, Fire.dt, Fire.alpha);
		//d_CylinderRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.axis, S.center, S.sized2.x, S.sized2.z);
		d_CylinderRestrictionZ << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.center, S.sized2.x, S.sized2.z, S.hidenpoint.z);
		//d_ParallelepipedRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.center, S.sized2);

		if (steps == 50000)
		{
			S.sized2.x /= l_CoeffSampleSize;
			S.sized2.y /= l_CoeffSampleSize;
			S.sized2.z /= l_CoeffSampleSize;
		}
		if (steps % 2000 == 0)
		{
			thrust::device_ptr<float> max_ptr = thrust::min_element(dtp_1d_Rijm, dtp_1d_Rijm + P.N);
			unsigned int position = max_ptr - dtp_1d_Rijm;
			float max_val = *max_ptr;
			std::cerr << "Fire prerelaxation " << steps << " " << position << " " << sqrt(max_val) << " " << sqrt(max_val) - 2.0 * Po.p_r << " " << (sqrt(max_val) - 2.0 * Po.p_r) / (2.0 * Po.p_r)
				<< " | " << Po.D << " " << Po.D / Dmax << " " << IL.acut << " " << IL.acut / ILacutmax << " " << Po.a / (2.0 * Po.p_r) << "\n";
		}
		timereal += Fire.dt;
		//std::cin.get();
	}

	timereal = 0;
	std::cerr << "ParamFire " << Fire.dt << " | " << S.L.x << " " << S.L.y << " " << S.L.z << " | " << S.center.x << " " << S.center.y << " " << S.center.z << " | " << S.size.x << " " << S.size.y << " " << S.size.z << "\n";
	for (steps = 0; steps < Fire.MaxStepsRelaxation; ++steps)
	{
		//if (steps % 100 == 0)std::cerr << " S" << steps;//std::cin.get();

		if (steps % 10000 == 0)
		{
			/*std::cerr << "Interaction list " << IL.N << "\n";
			cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
			for (int i = 0; i < IL.N; ++i)
			{
				if (i % IL.IonP == 0)std::cerr << "\n" << i / IL.IonP;
				if (IL.h_IL[i] != UINT_MAX)
					std::cerr << " " << IL.h_IL[i];
			}
			std::cin.get();/**/
			HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
			sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", steps);
			SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);

			

			/*std::cerr << "Interaction list " << IL.N << "\n";
			cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
			cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
			for (int i = 0; i < IL.N; ++i)
			{
				if (i % IL.IonP == 0)std::cerr << "\n" << i / IL.IonP;
				if (IL.h_IL[i] != UINT_MAX)
					std::cerr << " " << IL.h_IL[i] << "&" << int(IL.h_ILtype[i]);									
				else if (IL.h_IL[i] < UINT_MAX && IL.h_IL[i] >= P.N)std::cerr << "!!!!";
				else if (IL.h_IL[i] == UINT_MAX)
				{
					std::cerr << " " << "M";
					continue;
				}				
			}
			std::cin.get();/**/
			//std::cin.get();
		}/**/
		if (steps % 100 == 0 && Po.D < 1e-2*Dmax)
			Po.D *= 1.1;
		if (steps % 200 == 0 && IL.acut < ILacutmax)
		{
			IL.acut *= 1.2;
			if (IL.acut > ILacutmax)IL.acut = ILacutmax;
			IL.aacut = IL.acut * IL.acut;
		}/**/
		
		RenewInteractionList_New(P, C, S, A, IL);
		/*if (steps % 10000 == 0)
		{			
			std::cerr << "STEP " << steps << "\n";
			HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(C.h_CI, C.d_CI, P.N * sizeof(uint), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(C.h_CIs, C.d_CIs, 2 * P.N * sizeof(uint), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(C.h_pnC, C.d_pnC, 2 * C.N * sizeof(uint), cudaMemcpyDeviceToHost));
			h_CheckInteractions(P, C, A, S, IL);
			
		}/**/
		
		d_CalculateForces << <A.bloks, SMEMDIM >> > (P.d_R, P.d_F, P.N, IL.d_IL, IL.d_1d_iL, IL.IonP, Po.D, Po.aa);
		//d_CalculateIncrements << <A.bloks, SMEMDIM >> > (P.d_F, P.d_V, P.d_R, P.N, Po._1d_Mass_m_dt, Po.dt);
		//d_CylinderRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.axis, S.center, S.R0, S.H0);		
		d_FdotVEntire << < Fire.bloks4, SMEMDIM >> > (P.d_V, P.d_F, Fire.d_FdotV, P.N);
		cudaMemcpy(Fire.h_FdotV, Fire.d_FdotV, Fire.bloks4 * sizeof(float), cudaMemcpyDeviceToHost);
		Fire.FdotV = 0;
		for (i = 0; i < Fire.bloks4; ++i)
		{
			Fire.FdotV += Fire.h_FdotV[i];
		}
		//std::cerr << "Fire.FdotV " << Fire.FdotV << "\n";
		if (Fire.FdotV > 0)
		{
			++Fire.NPpositive;
			Fire.NPnegative = 0;
			if (Fire.NPpositive > Fire.Ndelay)
			{
				Fire.dt = (Fire.dt * Fire.dtgrow < Fire.dtmax) ? Fire.dt * Fire.dtgrow : Fire.dtmax;
				Fire.alpha *= Fire.alphashrink;
			}
			//std::cerr << "FdV POS " << steps <<" "<<Fire.dt << " " << Fire.alpha << " " << Fire.FdotV << "\n";
			//std::cin.get();
		}
		else
		{
			Fire.NPpositive = 0;
			++Fire.NPnegative;
			if (Fire.NPnegative > Fire.NPnegativeMax)
				break;
			if (steps > Fire.Ndelay)
			{
				Fire.dt = (Fire.dt * Fire.dtshrink > Fire.dtmin) ? Fire.dt * Fire.dtshrink : Fire.dtmin;
				Fire.alpha = Fire.alpha0;
			}
			d_CalculateDecrementsHalfStepFIRE << < A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, 0.5*Fire.dt);
			cudaMemset(P.d_V, 0, 3 * P.N * sizeof(float));
			//std::cerr << "FdV NEG " << steps << " " << Fire.dt << " " << Fire.alpha << " " << Fire.FdotV << "\n";
			//std::cin.get();
		}
		d_CalculateIncrementsFIRE << <A.bloks, SMEMDIM >> > (P.d_F, P.d_V, P.d_R, P.N, Po.dt_d_m, Fire.dt, Fire.alpha);
		//d_CylinderRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.axis, S.center, S.sized2.x, S.sized2.z);
		d_CylinderRestrictionZ << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.center, S.sized2.x, S.sized2.z, S.hidenpoint.z);
		//d_ParallelepipedRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.center, S.sized2);

		/*if (steps == 50000)
		{
			S.sized2.x *= 1.25;
			S.sized2.y *= 1.25;
			S.sized2.z *= 1.25;
		}/**/
		//if (steps % 5000 == 0)generateVelocities(P, A, S);

		//d_CylinderRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.axis, S.center, S.R0, S.H0);
		
		if (steps % 1000 == 0)
		{	
			/*HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
			double3 r;
			for (int ii = 0; ii < P.N; ++ii)
			{
				r.x = P.h_R[ii] - S.center.x;
				r.y = P.h_R[ii + P.N] - S.center.y;
				r.z = P.h_R[ii + 2 * P.N] - S.center.z;

				if (fabs(r.x) > S.size.x * 0.5 || fabs(r.y) > S.size.y * 0.5 || fabs(r.z) > S.size.z * 0.5)
					std::cerr << "Error! Pactical fly away " << P.h_R[ii] << " " << P.h_R[ii + P.N] << " " << P.h_R[ii + 2 * P.N]
					<< " | " << S.size.x << " " << S.size.y << " " << S.size.z
					<< " | " << S.center.x << " " << S.center.y << " " << S.center.z << "\n";				
			}/**/
			//std::cerr << "CellDistributionInit S " << S.B.x << " " << S.B.y << " " << S.B.z	<< " | " << S.L.x << " " << S.L.y << " " << S.L.z
			//	<< " | " << S.center.x << " " << S.center.y << " " << S.center.z << " | " << S.size.x << " " << S.size.y << " " << S.size.z << "\n";
			//std::cin.get();
			//std::cerr << "Rmax " << IL.d_1d_iL << " " << steps << "\n";
			thrust::device_ptr<float> max_ptr = thrust::min_element(dtp_1d_Rijm, dtp_1d_Rijm+P.N);
			unsigned int position = max_ptr - dtp_1d_Rijm;
			float max_val = *max_ptr;
			std::cerr << "Fire relaxation " << steps << " " << position << " " << sqrt(max_val) << " " << sqrt(max_val) - 2.0 * Po.p_r << " " << (sqrt(max_val) - 2.0 * Po.p_r) / (2.0 * Po.p_r)
				<< " | " << Po.D << " " << Po.D / Dmax << " " << IL.acut << " " << IL.acut / ILacutmax << "\n";
		}
			
		timereal += Fire.dt;
		//std::cin.get();
		
	}
	std::cerr << "FIN FIRE! " << steps << "\n"; //std::cin.get();

	
	/*uint time, bloks, estep, Estep, esize = Padd.ElementSteps * ResultFRNum * (P.NBPT[0] + P.NBPT[1]), i, j, steps;// , timestart, t1, t2;
	double d = 0, povis0 = Po.vis, timereal=0;
	float v = Padd.V, fa_max, fb_min, Ep0 = 2.21821;
	bool contraction = true;
	time = Padd.time;	
	Fire.bloks4 = P.N / (4 * SMEMDIM) + 1;
	Fire.h_FdotV = (float*)malloc(Fire.bloks4 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Fire.d_FdotV, Fire.bloks4 * sizeof(float)));
	Padd.bloks = P.N / (SMEMDIM)+2;
	Padd.bloksb = P.NBP / (SMEMDIM)+2;
	std::cerr << "Bloks " << Padd.bloks << " " << Padd.bloksb << "\n";	
	Po.vis = 0;
	Po.vism = Po.vis * Po.m;
	cudaMemset(P.d_V, 0, 2 * P.N * sizeof(float));/**/

	free(Fire.h_FdotV);
	Fire.h_FdotV = nullptr;
	cudaFree(Fire.d_FdotV);
	Fire.d_FdotV = nullptr;
}


__global__ void d_FdotVEntire(const float* __restrict__ V, const float* __restrict__ F, float* FdotV, const uint N)
{
	// static shared memory
	__shared__ float s_mem[SMEMDIM];

	// set thread ID
	// global index, 4 blocks of input data processed at a time
	uint tid = threadIdx.x, idx = blockIdx.x * blockDim.x * 4 + threadIdx.x, i;	
	// unrolling 4 blocks
	float fdv = 0;

	// boundary check
	if (idx + 3 * blockDim.x < N)
	{
		float t_fdv0 = 0, t_fdv1 = 0, t_fdv2 = 0, t_fdv3 = 0;
		i = idx;
		t_fdv0 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		i = idx + blockDim.x;
		t_fdv1 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		i = idx + 2 * blockDim.x;
		t_fdv2 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		i = idx + 3 * blockDim.x;
		t_fdv3 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		fdv = t_fdv0 + t_fdv1 + t_fdv2 + t_fdv3;
	}
	else if (idx + 2 * blockDim.x < N)
	{
		float t_fdv0 = 0, t_fdv1 = 0, t_fdv2 = 0;
		i = idx;
		t_fdv0 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		i = idx + blockDim.x;
		t_fdv1 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		i = idx + 2 * blockDim.x;
		t_fdv2 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		fdv = t_fdv0 + t_fdv1 + t_fdv2;
	}
	else if (idx + blockDim.x < N)
	{
		float t_fdv0 = 0, t_fdv1 = 0;
		i = idx;
		t_fdv0 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		i = idx + blockDim.x;
		t_fdv1 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		fdv = t_fdv0 + t_fdv1;
	}
	else if (idx < N)
	{
		float t_fdv0 = 0;
		i = idx;
		t_fdv0 = F[i] * V[i] + F[i + N] * V[i + N] + F[i + 2 * N] * V[i + 2 * N];
		fdv = t_fdv0;
	}/**/

	//if(idx + 5 * blockDim.x >4619700)
	//   printf("TT %i %i %i %f %i\n", tid, idx, blockIdx.x, n);
	//if (ns>1e-3f)
	//   printf("TT %i %i %f\n", tid, idx, ns);
	s_mem[tid] = fdv;
	__syncthreads();

	//if(idx==0)
	//	printf("TT %i %f %f %i %i\n", tid, s_ek, e_ek, s_n, e_n);

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512)
	{
		s_mem[tid] += s_mem[tid + 512];
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
	{
		//printf("Blok!\n");
		s_mem[tid] += s_mem[tid + 256];
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
	{
		s_mem[tid] += s_mem[tid + 128];
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
	{
		s_mem[tid] += s_mem[tid + 64];
	}

	__syncthreads();
	/*if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		for (int i = 0; i < SMEMDIM; ++i)
			printf("GM %i %e\n", i, smem[i + 3 * SMEMDIM]);
	}/**/

	// unrolling warp
	if (tid < 32)
	{
		volatile float* vsmem = s_mem;
		vsmem[tid] += vsmem[tid + 32];		
		vsmem[tid] += vsmem[tid + 16];		
		vsmem[tid] += vsmem[tid + 8];		
		vsmem[tid] += vsmem[tid + 4];		
		vsmem[tid] += vsmem[tid + 2];		
		vsmem[tid] += vsmem[tid + 1];		
	}/**/

	// write result for this block to global mem
	if (tid == 0)
	{
		//printf("TT %i %i %i %f\n", tid, idx, blockIdx.x, 0);
		FdotV[blockIdx.x] = s_mem[0];
		//printf("TTT %i %i %i %f\n", tid, idx, blockIdx.x, FdotV[blockIdx.x]);
		//if (smem[tid + 3 * SMEMDIM] > 1e-3f)
		//	printf("TT %i %i %f\n", tid, idx, smem[tid + 3 * SMEMDIM]);
		//if (smem[3 * SMEMDIM] > 1e-3f)
		//printf("T %i %f\n", blockIdx.x, gridDim.x, smem[3 * SMEMDIM]);
	}/**/
}

