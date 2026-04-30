#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include "md_definedparams.h"
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <time.h>

//#include <cudpp.h>
//#include <cudpp_plan.h>


__device__ void dd_Calculate_fijn_pwHM_2(const float* __restrict__ V, const float* __restrict__ W, const float3 &wall_V,  float3& Fiwn,
	const uint& idx, const uint& N, const float& m_E, const float& m_G, const float& p_A, const float& mP, const float& rP,
	const float3& niw, const float& riwm, float3& viwt, float& Kt, float& fiwnm)
{
	// normal force with damping
	//double Kn = 2 * prop.dEquivYoungModulus * dTemp2;
	//const double dDampingForce = -1.8257 * prop.dAlpha * dRelVelNormal * sqrt(Kn * dEquivMass);
	//const double dNormalForce = -dNormalOverlap * Kn * 2. / 3.;
	//double dTemp2 = sqrt(_collEquivRadii[iColl] * dNormalOverlap);
	//double Kt = 8 * prop.dEquivShearModulus * dTemp2;
	//CVector3 vDeltaTangOverlap = vRelVelTang * _timeStep;
	// rotate old tangential force
	//CVector3 vOldTangOverlap = _collTangOverlaps[iColl];
	//CVector3 vTangOverlap = vOldTangOverlap - vNormalVector * DotProduct(vNormalVector, vOldTangOverlap);
	//double dTangOverlapSqrLen = vTangOverlap.SquaredLength();
	//if (dTangOverlapSqrLen > 0)
	//	vTangOverlap = vTangOverlap * vOldTangOverlap.Length() / sqrt(dTangOverlapSqrLen);
	//vTangOverlap += vDeltaTangOverlap;
	//CVector3 vTangForce = vTangOverlap * Kt;
	//CVector3 vDampingTangForce = vRelVelTang * (-1.8257 * prop.dAlpha * sqrt(Kt * dEquivMass));
	// check slipping condition
	//double dNewTangForce = vTangForce.Length();
	//if (dNewTangForce > prop.dSlidingFriction * fabs(dNormalForce))
	//{
	//	vTangForce *= prop.dSlidingFriction * fabs(dNormalForce) / dNewTangForce;
	//	vTangOverlap = vTangForce / Kt;
	//}
	//else
	//	vTangForce += vDampingTangForce;
	//const CVector3 vRollingTorque1 = srcAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
	//	srcAnglVel * (-1 * prop.dRollingFriction * fabs(dNormalForce) * dPartSrcRadius / srcAnglVel.Length()) : CVector3{ 0 };
	//const CVector3 vTotalForce = vNormalVector * (dNormalForce + dDampingForce) + vTangForce;
	//const CVector3 vResultMoment1 = vNormalVector * vTangForce * dPartSrcRadius + vRollingTorque1;

	// normal force with damping
	float cef1 = __fsqrt_rn(rP * fabsf(rP - riwm));
	float Kn = 2.0f * m_E * cef1;
	Kt = 8.0f * m_G * cef1;
	//printf("pp %e %e %e", cef1);
	float3 viw;
	viw.x = wall_V.x - V[idx] + rP * (niw.y * W[idx + 2 * N] - niw.z * W[idx + N]);
	viw.y = wall_V.y - V[idx + N] + rP * (niw.z * W[idx] - niw.x * W[idx + 2 * N]);
	viw.z = wall_V.z - V[idx + 2 * N] + rP * (niw.x * W[idx + N] - niw.y * W[idx]);
	float viwnm = niw.x * viw.x + niw.y * viw.y + niw.z * viw.z;
	fiwnm = (riwm - rP) * Kn * MCf_2d3;
	cef1 = -1.8257f * p_A * viwnm * __fsqrt_rn(Kn * mP);
	//printf("F %u %e %e\n", idx, fiwnm, cef1);
	Fiwn.x = niw.x * (fiwnm + cef1);
	Fiwn.y = niw.y * (fiwnm + cef1);
	Fiwn.z = niw.z * (fiwnm + cef1);

	viwt.x = viw.x - niw.x * viwnm;
	viwt.y = viw.y - niw.y * viwnm;
	viwt.z = viw.z - niw.z * viwnm;
}

__device__ void dd_Calculate_fijtmijt_pwHM_2(const float* __restrict__ W, float* __restrict__ Oiwt, float3& Fiwt, float3& Miwt, float3& Miwadd,
	const uint& idx, const uint& N, const float3& niw, const float& m_mu, const float& p_A, const float& mP, const float& rP, const float& dt, const float& m_muroll, 
	const float3& viwt, const float& Kt, const float& fiwnm)
{
	// normal force with damping
	float3 oiwt, oiwt_={Oiwt[idx], Oiwt[idx + N], Oiwt[idx + 2 * N]};
	// rotate old tangential force
	float cef1 = niw.x * oiwt_.x + niw.y * oiwt_.y + niw.z * oiwt_.z;
	oiwt.x = oiwt_.x - niw.x * cef1;
	oiwt.y = oiwt_.y - niw.y * cef1;
	oiwt.z = oiwt_.z - niw.z * cef1;
	cef1 = __fmul_rn(oiwt.x, oiwt.x) + __fmul_rn(oiwt.y, oiwt.y) + __fmul_rn(oiwt.z, oiwt.z);
	if (cef1 > 1e-18)
	{
		cef1 = __frcp_rn(cef1);
		cef1 *= __fmul_rn(oiwt_.x, oiwt_.y) + __fmul_rn(oiwt_.y, oiwt_.y) + __fmul_rn(oiwt_.z, oiwt_.z);
		cef1 = __fsqrt_rn(cef1);
		oiwt.x *= cef1;
		oiwt.y *= cef1;
		oiwt.z *= cef1;
	}
	oiwt.x += viwt.x * dt;
	oiwt.y += viwt.y * dt;
	oiwt.z += viwt.z * dt;

	float3 fiwt;
	fiwt.x = oiwt.x * Kt;
	fiwt.y = oiwt.y * Kt;
	fiwt.z = oiwt.z * Kt;
	// check slipping condition
	float fiwtmm = __fmul_rn(fiwt.x, fiwt.x) + __fmul_rn(fiwt.y, fiwt.y) + __fmul_rn(fiwt.z, fiwt.z);
	if (fiwtmm > m_mu * m_mu * fiwnm * fiwnm)
	{
		cef1 = m_mu * fabsf(fiwnm) * __frsqrt_rn(fiwtmm);
		fiwt.x *= cef1;
		fiwt.y *= cef1;
		fiwt.z *= cef1;
		cef1 = __frcp_rn(Kt);
		oiwt.x = fiwt.x * cef1;
		oiwt.y = fiwt.y * cef1;
		oiwt.z = fiwt.z * cef1;
	}
	else
	{
		cef1 = -1.8257f * p_A * sqrt(Kt * mP);
		fiwt.x += viwt.x * cef1;
		fiwt.y += viwt.y * cef1;
		fiwt.z += viwt.z * cef1;
	}
	Oiwt[idx] = oiwt.x;
	Oiwt[idx + N] = oiwt.y;
	Oiwt[idx + 2 * N] = oiwt.z;

	Fiwt.x = fiwt.x;
	Fiwt.y = fiwt.y;
	Fiwt.z = fiwt.z;
	Miwt.x = (niw.y * fiwt.z - niw.z * fiwt.y) * rP;
	Miwt.y = (niw.z * fiwt.x - niw.x * fiwt.z) * rP;
	Miwt.z = (niw.x * fiwt.y - niw.y * fiwt.x) * rP;
	
	float3 wi;
	wi.x = W[idx];
	wi.y = W[idx + N];
	wi.z = W[idx + 2 * N];
	float wim = __fmul_rn(wi.x, wi.x) + __fmul_rn(wi.y, wi.y) + __fmul_rn(wi.z, wi.z);
	if (wim > 1e-18)
	{
		float cef1 = -m_muroll * fabsf(fiwnm) * __frsqrt_rn(wim);
		Miwadd.x = wi.x * cef1;
		Miwadd.y = wi.y * cef1;
		Miwadd.z = wi.z * cef1;
	}
	else
	{
		Miwadd.x = 0;
		Miwadd.y = 0;
		Miwadd.z = 0;
	}
}

__global__ void d_UniaxialCompression_hm(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, float* __restrict__ Oiwt, float* __restrict__ F, float* __restrict__ M,
	const uint N, const float dt, const float m_E, const float m_G, const float p_A, const float m_mu, const float m_muroll, const float mP, const float rP,	
	float* __restrict__ FL, const float3 c, const float wall_RR, const float wall_Zt, const float wall_Zb, const float wall_V, const float Zcut)
{
	__shared__ float s_mem[2 * SMEMDIM];
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x;
	float sfb = 0, sft = 0;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	//if(idx==0)
	//if(idx==0)printf("AAAAA\n");
	while (idx < N)
	{
		float3 r;
		r.x = R[idx] - c.x;
		r.y = R[idx + N] - c.y;
		r.z = R[idx + 2 * N];
		
		if (r.x * r.x + r.y * r.y < wall_RR && r.z < Zcut)
		{
			if (wall_Zt - r.z < rP)
			{	
				float3 niw={0.0f, 0.0f, 1.0f}, wallv={0.0f, 0.0f, wall_V}, fiwn, viwt, fiwt, miwt, miwadd;
				float riwm = fabsf(wall_Zt - r.z), Kt, fiwnm;				

				//_1d_rm = __frcp_rn(rijm);				
				
				dd_Calculate_fijn_pwHM_2(V, W, wallv, fiwn, idx, N, m_E, m_G, p_A, mP, rP,	niw, riwm, viwt, Kt, fiwnm);
				//dd_Calculate_fijtmijt_ppHM_2(W, Oiwt, fs, ms, kdx, niw, m_mu, p_A, mP, rP, dt, viwt, Kt, fiwnm);				
				dd_Calculate_fijtmijt_pwHM_2(W, Oiwt, fiwt, miwt, miwadd, idx, N, niw, m_mu, p_A, mP, rP, dt, m_muroll, viwt, Kt, fiwnm);
				
				//if(fiwn.z > 0) printf("Error Top Particle-Wall %u %e %e %e %e\n", idx, wall_Zt - r.z, fiwn.z, R[idx + 2 * N], fiwnm);
				//if(fiwn.z < 0)
				sft += fiwn.z;
				
				//printf("Top Particle-Wall %u %e %e %e %e %e\n", idx, wall_Zt - r.z, fiwn.z, R[idx + 2 * N], fiwnm, rP);
							
				F[idx] += fiwn.x + fiwt.x;
				F[idx + N] += fiwn.y + fiwt.y;
				F[idx + 2 * N] += fiwn.z + fiwt.z;
				M[idx] += miwt.x + miwadd.x;
				M[idx + N] += miwt.y + miwadd.y;
				M[idx + 2 * N] += miwt.z + miwadd.z;
				//printf("Uft %u %e %e %e\n", idx, (Hd2 - r.z) * C, fz, R[idx + 2 * N]);
			} else
			{
				Oiwt[idx] = 0;
				Oiwt[idx + N] = 0;
				Oiwt[idx + 2 * N] = 0;
			}
				
			if (r.z - wall_Zb < rP)
			{	
				float3 niw={0.0f, 0.0f, -1.0f}, wallv={0.0f, 0.0f, -wall_V}, fiwn, viwt, fiwt, miwt, miwadd;
				float riwm = fabsf(wall_Zb - r.z), Kt, fiwnm;				

				//_1d_rm = __frcp_rn(rijm);				
				
				dd_Calculate_fijn_pwHM_2(V, W, wallv, fiwn, idx, N, m_E, m_G, p_A, mP, rP,	niw, riwm, viwt, Kt, fiwnm);
				//dd_Calculate_fijtmijt_ppHM_2(W, Oiwt, fs, ms, kdx, niw, m_mu, p_A, mP, rP, dt, viwt, Kt, fiwnm);				
				dd_Calculate_fijtmijt_pwHM_2(W, Oiwt, fiwt, miwt, miwadd, idx, N, niw, m_mu, p_A, mP, rP, dt, m_muroll, viwt, Kt, fiwnm);
				
				//if(fiwn.z < 0) printf("Error Bottom Particle-Wall %u %e %e %e\n", idx, wall_Zb - r.z, fiwn.z, R[idx + 2 * N]);
				//if(fiwn.z > 0)
				sfb += fiwn.z;

				//printf("Bottom Particle-Wall %u %e %e %e %e %e\n", idx, wall_Zb - r.z, fiwn.z, R[idx + 2 * N], fiwnm, Kt);
							
				F[idx] += fiwn.x + fiwt.x;
				F[idx + N] += fiwn.y + fiwt.y;
				F[idx + 2 * N] += fiwn.z + fiwt.z;
				M[idx] += miwt.x + miwadd.x;
				M[idx + N] += miwt.y + miwadd.y;
				M[idx + 2 * N] += miwt.z + miwadd.z;
				//printf("Uft %u %e %e %e\n", idx, (Hd2 - r.z) * C, fz, R[idx + 2 * N]);
			} else
			{
				Oiwt[idx] = 0;
				Oiwt[idx + N] = 0;
				Oiwt[idx + 2 * N] = 0;
			}
			
			
		}		
		idx += blockDim.x * gridDim.x;
	}
	//if(fabsf(sfb)+fabsf(sft)>1e-18)	printf("Sum Particle-Wall %u %e %e\n", tid, sfb, sft);
	s_mem[tid] = -sfb;
	s_mem[tid + SMEMDIM] = -sft;
	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512)
	{
		s_mem[tid] += s_mem[tid + 512];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 512];		
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
	{
		//printf("Blok!\n");
		s_mem[tid] += s_mem[tid + 256];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 256];		
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
	{
		s_mem[tid] += s_mem[tid + 128];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 128];		
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
	{
		s_mem[tid] += s_mem[tid + 64];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 64];		
	}
	__syncthreads();
	
	// unrolling warp
	if (tid < 32)
	{
		volatile float* vsmem = s_mem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 32];		
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 16];		
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 8];		
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 4];		
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 2];		
		vsmem[tid] += vsmem[tid + 1];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 1];		
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		//printf("U %u %u %u\n", idx, blockIdx.x, blockIdx.x + gridDim.x);
		FL[blockIdx.x] = s_mem[0];
		FL[blockIdx.x + gridDim.x] = s_mem[SMEMDIM];		
	}
}


__global__ void d_BrazilCompression_hm(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, float* __restrict__ Oiwt, float* __restrict__ F, float* __restrict__ M,
	const uint N, const float dt, const float m_E, const float m_G, const float p_A, const float m_mu, const float m_muroll, const float mP, const float rP,	
	float* __restrict__ FL, const float3 c, const float wall_RR, const float wall_Yt, const float wall_Yb, const float wall_V, const float Zcut)
{
	__shared__ float s_mem[2 * SMEMDIM];
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x;
	float sfb = 0, sft = 0;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	//if(idx==0)
	//if(idx==0)printf("AAAAA\n");
	while (idx < N)
	{
		float3 r;
		r.x = R[idx] - c.x;
		r.y = R[idx + N];
		r.z = R[idx + 2 * N] - c.z;
		
		if (r.x * r.x + r.z * r.z < wall_RR && r.z < Zcut)
		{
			if (wall_Yt - r.y < rP)
			{	
				float3 niw={0.0f, 1.0f, 0.0f}, wallv={0.0f, wall_V, 0.0f}, fiwn, viwt, fiwt, miwt, miwadd;
				float riwm = fabsf(wall_Yt - r.y), Kt, fiwnm;				

				//_1d_rm = __frcp_rn(rijm);				
				
				dd_Calculate_fijn_pwHM_2(V, W, wallv, fiwn, idx, N, m_E, m_G, p_A, mP, rP,	niw, riwm, viwt, Kt, fiwnm);
				//dd_Calculate_fijtmijt_ppHM_2(W, Oiwt, fs, ms, kdx, niw, m_mu, p_A, mP, rP, dt, viwt, Kt, fiwnm);				
				dd_Calculate_fijtmijt_pwHM_2(W, Oiwt, fiwt, miwt, miwadd, idx, N, niw, m_mu, p_A, mP, rP, dt, m_muroll, viwt, Kt, fiwnm);
				
				//if(fiwn.z > 0) printf("Error Top Particle-Wall %u %e %e %e %e\n", idx, wall_Zt - r.z, fiwn.z, R[idx + 2 * N], fiwnm);
				//if(fiwn.z < 0)
				sft += fiwn.y;
				
				//printf("Top Particle-Wall %u %e %e %e %e %e\n", idx, wall_Zt - r.z, fiwn.z, R[idx + 2 * N], fiwnm, rP);
							
				F[idx] += fiwn.x + fiwt.x;
				F[idx + N] += fiwn.y + fiwt.y;
				F[idx + 2 * N] += fiwn.z + fiwt.z;
				M[idx] += miwt.x + miwadd.x;
				M[idx + N] += miwt.y + miwadd.y;
				M[idx + 2 * N] += miwt.z + miwadd.z;
				//printf("Uft %u %e %e %e\n", idx, (Hd2 - r.z) * C, fz, R[idx + 2 * N]);
			} else
			{
				Oiwt[idx] = 0;
				Oiwt[idx + N] = 0;
				Oiwt[idx + 2 * N] = 0;
			}
				
			if (r.y - wall_Yb < rP)
			{	
				float3 niw={0.0f, -1.0f, 0.0f}, wallv={0.0f, -wall_V, 0.0f}, fiwn, viwt, fiwt, miwt, miwadd;
				float riwm = fabsf(wall_Yb - r.y), Kt, fiwnm;				

				//_1d_rm = __frcp_rn(rijm);				
				
				dd_Calculate_fijn_pwHM_2(V, W, wallv, fiwn, idx, N, m_E, m_G, p_A, mP, rP,	niw, riwm, viwt, Kt, fiwnm);
				//dd_Calculate_fijtmijt_ppHM_2(W, Oiwt, fs, ms, kdx, niw, m_mu, p_A, mP, rP, dt, viwt, Kt, fiwnm);				
				dd_Calculate_fijtmijt_pwHM_2(W, Oiwt, fiwt, miwt, miwadd, idx, N, niw, m_mu, p_A, mP, rP, dt, m_muroll, viwt, Kt, fiwnm);
				
				//if(fiwn.z < 0) printf("Error Bottom Particle-Wall %u %e %e %e\n", idx, wall_Zb - r.z, fiwn.z, R[idx + 2 * N]);
				//if(fiwn.z > 0)
				sfb += fiwn.y;

				//printf("Bottom Particle-Wall %u %e %e %e %e %e\n", idx, wall_Zb - r.z, fiwn.z, R[idx + 2 * N], fiwnm, Kt);
							
				F[idx] += fiwn.x + fiwt.x;
				F[idx + N] += fiwn.y + fiwt.y;
				F[idx + 2 * N] += fiwn.z + fiwt.z;
				M[idx] += miwt.x + miwadd.x;
				M[idx + N] += miwt.y + miwadd.y;
				M[idx + 2 * N] += miwt.z + miwadd.z;
				//printf("Uft %u %e %e %e\n", idx, (Hd2 - r.z) * C, fz, R[idx + 2 * N]);
			} else
			{
				Oiwt[idx] = 0;
				Oiwt[idx + N] = 0;
				Oiwt[idx + 2 * N] = 0;
			}
			
			
		}		
		idx += blockDim.x * gridDim.x;
	}
	//if(fabsf(sfb)+fabsf(sft)>1e-18)	printf("Sum Particle-Wall %u %e %e\n", tid, sfb, sft);
	s_mem[tid] = -sfb;
	s_mem[tid + SMEMDIM] = -sft;
	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512)
	{
		s_mem[tid] += s_mem[tid + 512];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 512];		
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
	{
		//printf("Blok!\n");
		s_mem[tid] += s_mem[tid + 256];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 256];		
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
	{
		s_mem[tid] += s_mem[tid + 128];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 128];		
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
	{
		s_mem[tid] += s_mem[tid + 64];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 64];		
	}
	__syncthreads();
	
	// unrolling warp
	if (tid < 32)
	{
		volatile float* vsmem = s_mem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 32];		
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 16];		
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 8];		
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 4];		
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 2];		
		vsmem[tid] += vsmem[tid + 1];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 1];		
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		//printf("U %u %u %u\n", idx, blockIdx.x, blockIdx.x + gridDim.x);
		FL[blockIdx.x] = s_mem[0];
		FL[blockIdx.x + gridDim.x] = s_mem[SMEMDIM];		
	}
}