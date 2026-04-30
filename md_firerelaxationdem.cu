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

__global__ void d_CalculateDecrementsHalfStepDEMFIRE(const float* __restrict__ V, float* __restrict__ R, const uint N, const float dt_d2)
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

__global__ void d_CalculateIncrementsDEMFIRE(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const float* __restrict__ M, float* __restrict__ W, const uint N, const float _1d_Mass_m_dt, const float dt, const float _1d_I_m_dt,
	const float F_alpha)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	float vv, f, v_d_f, ww, m, w_d_m;
	float3 v, w;

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
		
		w.x += __fmul_rn(M[idx], _1d_I_m_dt);
		w.y += __fmul_rn(M[idx + N], _1d_I_m_dt);
		w.z += __fmul_rn(M[idx + 2 * N], _1d_I_m_dt);
		
		ww = w.x * w.x + w.y * w.y + w.z * w.z;
		m = M[idx] * M[idx] + M[idx + N] * M[idx + N] + M[idx + 2 * N] * M[idx + 2 * N];
		if (m > 1e-12f)
		{
			w_d_m = F_alpha * __fsqrt_rn(vv * __frcp_rn(f));
		}
		else
		{
			w_d_m = F_alpha * __fsqrt_rn(vv) * 1e6;
		}
		w.x = (1.0f - F_alpha) * w.x + w_d_m * M[idx];
		w.y = (1.0f - F_alpha) * w.y + w_d_m * M[idx + N];
		w.z = (1.0f - F_alpha) * w.z + w_d_m * M[idx + 2 * N];
		
		W[idx] = w.x;
		W[idx + N] = w.y;
		W[idx + 2 * N] = w.z;
		//W[idx] += __fmul_rn(M[idx], _1d_I_m_dt);
		//W[idx + N] += __fmul_rn(M[idx + N], _1d_I_m_dt);
		//W[idx + 2 * N] += __fmul_rn(M[idx + 2 * N], _1d_I_m_dt);
		
		idx += blockDim.x * gridDim.x;
	}
}

void CalculateRelaxDEMFIRE(md_task_data& mdTD, firerelax_data &Fire)
{
	std::cerr << "FIRE short relaxation Start " << mdTD.Compress.V << "\n";
	   
	//SetFIREData(mdTD.P, mdTD.Po, Fire);
    //Fire.MaxStepsRelaxation =30000*int(Fire.Tsim/Fire.dt0);
	//Fire.StepsPreRelaxation = 0*10000*int(Fire.Tsim/Fire.dt0);
	
	//std::cerr << "FIRE Start\n";
	//Fire.NPpositive = 0;
	//Fire.NPnegative = 0;
	//Fire.dt = Fire.dt0;
	//Fire.h_FdotV = (float*)malloc(Fire.bloks4 * sizeof(float));
	//HANDLE_ERROR(cudaMalloc((void**)&Fire.d_FdotV, Fire.bloks4 * sizeof(float)));
		
	uint steps = 0, i;
	double timereal = 0;
	
	char filename[256] = "";
    /*HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(mdTD.P.h_V, mdTD.P.d_V, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(mdTD.P.h_F, mdTD.P.d_F, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
	double3 l_sumF={0,0,0}, l_sumV={0,0,0};
	for(uint l_i=0; l_i<mdTD.P.N; ++l_i){l_sumF.x += mdTD.P.h_F[l_i]; l_sumF.y += mdTD.P.h_F[l_i+mdTD.P.N]; l_sumF.z += mdTD.P.h_F[l_i+2*mdTD.P.N];
	l_sumV.x += mdTD.P.h_V[l_i]; l_sumV.y += mdTD.P.h_V[l_i+mdTD.P.N]; l_sumV.z += mdTD.P.h_V[l_i+2*mdTD.P.N];}
    sprintf(filename, "./result/steps/LAMMPS/CPuf_%li.txt", steps);
    SaveLammpsDATASimple(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL, mdTD.Po, filename, false);
    std::cerr << "Stress " << steps << " " << mdTD.Compress.Stress * stress_const * 1e-6 << " " << mdTD.Compress.Stress << "\n";
	std::cerr << "Forse " << steps << " " << l_sumF.x * force_const << " " << l_sumF.y * force_const << " " << l_sumF.z * force_const << "\n";
	std::cerr << "Velocity " << steps << " " << l_sumV.x * velocity_const << " " << l_sumV.y * velocity_const << " " << l_sumV.z * velocity_const << "\n";*/
	//thrust::device_ptr<float> dtp_1d_Rijm(mdTD.IL.d_1d_iL);
	Fire.alpha0 = 0.1; Fire.alpha = Fire.alpha0;
	//cudaMemset(mdTD.P.d_W, 0, 3 * mdTD.P.N * sizeof(float));
	//std::cerr<<"AA "<<Fire.MaxStepsRelaxation<<"\n"; std::cin.get();
	for (steps = 0; steps < Fire.MaxStepsRelaxation; ++steps)
	{	
		//std::cerr<<"AA\n"; std::cin.get();
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
            sprintf(filename, "./result/steps/LAMMPS/CPuf_%li.txt", steps);
            SaveLammpsDATASimple(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL, mdTD.Po, filename, false);
            std::cerr << "Stress " << steps << " " << mdTD.Compress.Stress * stress_const * 1e-6 << " " << mdTD.Compress.Stress << "\n";
			std::cerr << "Forse " << steps << " " << l_sumF.x * force_const << " " << l_sumF.y * force_const << " " << l_sumF.z * force_const << "\n";
			std::cerr << "Velocity " << steps << " " << l_sumV.x * velocity_const << " " << l_sumV.y * velocity_const << " " << l_sumV.z * velocity_const << "\n";
			//std::cin.get();
		}

		d_CalculateForcesDEM_22 << <4 * mdTD.A.bloks, 256 >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL,
            mdTD.IL.d_rij, mdTD.IL.d_Oijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.P.d_F, mdTD.P.d_M, mdTD.P.N, mdTD.IL.IonP, mdTD.Po.dt, mdTD.Po.m_E, mdTD.Po.m_G, mdTD.Po.b_r, mdTD.Po.m_Ec, mdTD.Po.m_Gc);// std::cerr << "D10\n";
        d_CalculateForcesDEM_21 << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt, mdTD.P.d_F, mdTD.P.d_M,
            mdTD.P.N, mdTD.IL.IonP, mdTD.Po.dt, mdTD.Po.hm_E, mdTD.Po.hm_G, mdTD.Po.p_A, mdTD.Po.p_mu, mdTD.Po.p_mur, mdTD.Po.p_m, mdTD.Po.p_r); //std::cerr << "D11\n";

		d_FdotVEntire << < Fire.bloks4, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_F, Fire.d_FdotV, mdTD.P.N);
		cudaMemcpy(Fire.h_FdotV, Fire.d_FdotV, Fire.bloks4 * sizeof(float), cudaMemcpyDeviceToHost);
		Fire.FdotV = 0;
		if(steps%10000==0)
			std::cerr<<"Fire "<<steps<<" "<<Fire.FdotV<<"\n";
		for (i = 0; i < Fire.bloks4; ++i)
		{
			Fire.FdotV += Fire.h_FdotV[i];
		}
		d_MdotWEntire << < Fire.bloks4, SMEMDIM >> > (mdTD.P.d_W, mdTD.P.d_M, Fire.d_MdotW, mdTD.P.N);
		cudaMemcpy(Fire.h_MdotW, Fire.d_MdotW, Fire.bloks4 * sizeof(float), cudaMemcpyDeviceToHost);
		Fire.MdotW = 0;
		if(steps%10000==0)
			std::cerr<<"Fire "<<steps<<" "<<Fire.MdotW<<"\n";
		for (i = 0; i < Fire.bloks4; ++i)
		{
			Fire.MdotW += Fire.h_MdotW[i];
		}		

		if (fabs(Fire.FdotV) <1e-15 && fabs(Fire.MdotW) <1e-15 && steps > 100)
			break;
		else if (Fire.FdotV > 0 || Fire.MdotW > 0)
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
			if (Fire.NPnegative > Fire.NPnegativeMax)
				break;
			if (steps > Fire.Ndelay)
			{
				Fire.dt = (Fire.dt * Fire.dtshrink > Fire.dtmin) ? Fire.dt * Fire.dtshrink : Fire.dtmin;
				Fire.alpha = Fire.alpha0;
			}
			//d_CalculateDecrementsHalfStepFIRE << < mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_R, mdTD.P.N, 0.5 * Fire.dt);
			d_CalculateDecrementsHalfStepDEMFIRE << < mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_V, mdTD.P.d_R, mdTD.P.N, 0.5 * Fire.dt);
			
			cudaMemset(mdTD.P.d_V, 0, 3 * mdTD.P.N * sizeof(float));
			cudaMemset(mdTD.P.d_W, 0, 3 * mdTD.P.N * sizeof(float));
		}
		d_CalculateIncrementsDEMFIRE<< <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, mdTD.P.d_V, mdTD.P.d_R, mdTD.P.d_M, mdTD.P.d_W, mdTD.P.N, mdTD.Po.dt_d_m, mdTD.Po.dt, mdTD.Po.dt_d_I, Fire.alpha);
		
		d_CylinderRestrictionFireZr2 << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_F, mdTD.P.N, mdTD.S.center, 0.75*mdTD.S.spacesized2.x, mdTD.Compress.Hd2, mdTD.S.hidenpoint);
		//d_ParallelepipedCutRestriction << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_F, mdTD.P.N, mdTD.S.center, mdTD.S.spacesized2, mdTD.S.hidenpoint); 
		
		timereal += Fire.dt;
		//std::cerr<<"Steps "<<steps<<" "<<Fire.alpha<<"\n"; std::cin.get();
	}
	std::cerr << "FIN FIRE! " << steps << "\n"; 
	//std::cerr<<"Result density:porosity "<<(mdTD.P.N-l_deletedparticles)*mdTD.Po.p_V/mpMDP.S[isample].Vext<<" porosity: "<<1.0-(mdTD.P.N-l_deletedparticles)*mdTD.Po.p_V/mpMDP.S[isample].Vext<<"\n";//std::cin.get();
	
	//char filename[256] = "";
	{
		HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(mdTD.P.h_V, mdTD.P.d_V, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(mdTD.P.h_F, mdTD.P.d_F, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
		double3 l_sumF={0,0,0}, l_sumV={0,0,0};
		for(uint l_i=0; l_i<mdTD.P.N; ++l_i){l_sumF.x += mdTD.P.h_F[l_i]; l_sumF.y += mdTD.P.h_F[l_i+mdTD.P.N]; l_sumF.z += mdTD.P.h_F[l_i+2*mdTD.P.N];
		l_sumV.x += mdTD.P.h_V[l_i]; l_sumV.y += mdTD.P.h_V[l_i+mdTD.P.N]; l_sumV.z += mdTD.P.h_V[l_i+2*mdTD.P.N];}
		sprintf(filename, "./result/steps/LAMMPS/CPuf_%li.txt", steps);
		SaveLammpsDATASimple(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL, mdTD.Po, filename, false);
		std::cerr << "Stress " << steps << " " << mdTD.Compress.Stress * stress_const * 1e-6 << " " << mdTD.Compress.Stress << "\n";
		std::cerr << "Forse " << steps << " " << l_sumF.x * force_const << " " << l_sumF.y * force_const << " " << l_sumF.z * force_const << "\n";
		std::cerr << "Velocity " << steps << " " << l_sumV.x * velocity_const << " " << l_sumV.y * velocity_const << " " << l_sumV.z * velocity_const << "\n";
		//std::cin.get();
	}
	
    
	
	//free(Fire.h_FdotV);
	//Fire.h_FdotV = nullptr;
	//cudaFree(Fire.d_FdotV);
	//Fire.d_FdotV = nullptr;
}

__global__ void d_CylinderRestrictionFireZr2(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, const uint N, const float3 center, const float R0, const float H0, const float3 Hide)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("d_CylinderRestrictionZr\n");
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr, s;
	float rr, hm, sm, coefr, coefh, coefs = 1.0f, R1 = R0, H1 = H0;
	while (idx < N)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;
		s.x = 0;
		s.y = 0;
		s.z = 0;
		rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y);
		if(dr.z < Hide.z - center.z)
		{
				if (rr > R1 * R1)
				{
					rr = __frsqrt_rn(rr);
					s.x = dr.x * (R1 * rr - 1.0f);
					s.y = dr.y * (R1 * rr - 1.0f);
					//V[idx] *= -1.0f;
					//V[idx + N] *= -1.0f;
					//printf("In %u %e %e %e %e %e\n", idx, dr.x, dr.y, R1, s.x, s.y);
				}
				if (dr.z > H1)
				{
					s.z = H1 - dr.z;
					//V[idx + 2 * N] *= -1.0f;
					F[idx + 2 * N] = 0.0f;
				}
				if (dr.z < -H1)
				{
					s.z = -H1 - dr.z;
					//V[idx + 2 * N] *= -1.0f;
					F[idx + 2 * N] = 0.0f;
				}
				R[idx] += s.x;
				R[idx + N] += s.y;
				R[idx + 2 * N] += s.z;
		} else
		{
			R[idx] = Hide.x;
			R[idx + N] = Hide.y;
			R[idx + 2 * N] = Hide.z;
		}
				
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_MdotWEntire(const float* __restrict__ W, const float* __restrict__ M, float* MdotW, const uint N)
{
	// static shared memory
	__shared__ float s_mem[SMEMDIM];

	// set thread ID
	// global index, 4 blocks of input data processed at a time
	uint tid = threadIdx.x, idx = blockIdx.x * blockDim.x * 4 + threadIdx.x, i;	
	// unrolling 4 blocks
	float mdw = 0;

	// boundary check
	if (idx + 3 * blockDim.x < N)
	{
		float t_mdw0 = 0, t_mdw1 = 0, t_mdw2 = 0, t_mdw3 = 0;
		i = idx;
		t_mdw0 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		i = idx + blockDim.x;
		t_mdw1 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		i = idx + 2 * blockDim.x;
		t_mdw2 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		i = idx + 3 * blockDim.x;
		t_mdw3 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		mdw = t_mdw0 + t_mdw1 + t_mdw2 + t_mdw3;
	}
	else if (idx + 2 * blockDim.x < N)
	{
		float t_mdw0 = 0, t_mdw1 = 0, t_mdw2 = 0;
		i = idx;
		t_mdw0 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		i = idx + blockDim.x;
		t_mdw1 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		i = idx + 2 * blockDim.x;
		t_mdw2 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		mdw = t_mdw0 + t_mdw1 + t_mdw2;
	}
	else if (idx + blockDim.x < N)
	{
		float t_mdw0 = 0, t_mdw1 = 0;
		i = idx;
		t_mdw0 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		i = idx + blockDim.x;
		t_mdw1 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		mdw = t_mdw0 + t_mdw1;
	}
	else if (idx < N)
	{
		float t_mdw0 = 0;
		i = idx;
		t_mdw0 = M[i] * W[i] + M[i + N] * W[i + N] + M[i + 2 * N] * W[i + 2 * N];
		mdw = t_mdw0;
	}/**/

	//if(idx + 5 * blockDim.x >4619700)
	//   printf("TT %i %i %i %f %i\n", tid, idx, blockIdx.x, n);
	//if (ns>1e-3f)
	//   printf("TT %i %i %f\n", tid, idx, ns);
	s_mem[tid] = mdw;
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
		MdotW[blockIdx.x] = s_mem[0];
		//printf("TTT %i %i %i %f\n", tid, idx, blockIdx.x, FdotV[blockIdx.x]);
		//if (smem[tid + 3 * SMEMDIM] > 1e-3f)
		//	printf("TT %i %i %f\n", tid, idx, smem[tid + 3 * SMEMDIM]);
		//if (smem[3 * SMEMDIM] > 1e-3f)
		//printf("T %i %f\n", blockIdx.x, gridDim.x, smem[3 * SMEMDIM]);
	}/**/
}