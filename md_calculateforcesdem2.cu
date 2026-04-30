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
#include <time.h>

//#include <cudpp.h>
//#include <cudpp_plan.h>

//__constant__ char IM[2 * IMatrixSize];

__device__ void dd_Calculate_rijm_nij_1(const float* __restrict__ R, const uint& idx, const uint& jdx, const uint& N, float3& nij, float& rijm)
{
	float3 rij;
	float _1d_rm;
	rij.x = R[jdx] - R[idx];
	rij.y = R[jdx + N] - R[idx + N];
	rij.z = R[jdx + 2 * N] - R[idx + 2 * N];
	//CVector3 currentBond = GetSolidBond(Particles().Coord(_iRight), Particles().Coord(_iLeft), m_PBC);
	//double dDistanceBetweenCenters = currentBond.Length();
	rijm = norm3df(rij.x, rij.y, rij.z); //__fmul_rn(rij.x, rij.x) + __fmul_rn(rij.y, rij.y) + __fmul_rn(rij.z, rij.z);
	_1d_rm = __frcp_rn(rijm);	
	nij.x = rij.x * _1d_rm;
	nij.y = rij.y * _1d_rm;
	nij.z = rij.z * _1d_rm;
}

__device__ void dd_Calculate_M123_bpm_1(const float* __restrict__ R, const float* __restrict__ W, float3* __restrict__ Rij, const uint& kdx, const uint& idx, const uint& jdx, const uint& N, const float dt, const float3& nij, float3& M1, float3& M2, float3& M3)
{
	//CVector3 tempVector = _bondPrevBonds[i] * currentBond;
	//CVector3 Phi = currentContact * (DotProduct(sumAngleVelocity, currentContact) * _timeStep * 0.5);
	//CMatrix3 M(1 + tempVector.z * Phi.z + tempVector.y * Phi.y, Phi.z - tempVector.z - tempVector.y * Phi.x, -Phi.y - tempVector.z * Phi.x + tempVector.y,
	//	tempVector.z - Phi.z - tempVector.x * Phi.y, tempVector.z * Phi.z + 1 + tempVector.x * Phi.x, -tempVector.z * Phi.y + Phi.x - tempVector.x,
	//	-tempVector.y - tempVector.x * Phi.z + Phi.y, -tempVector.y * Phi.z + tempVector.x - Phi.x, tempVector.y * Phi.y + tempVector.x * Phi.x + 1);
	float3 rij, tmp;
	//float _1d_rm;
	rij.x = R[jdx] - R[idx];
	rij.y = R[jdx + N] - R[idx + N];
	rij.z = R[jdx + 2 * N] - R[idx + 2 * N];
	tmp.x = Rij[kdx].y * nij.z - Rij[kdx].z * nij.y;
	tmp.y = Rij[kdx].z * nij.x - Rij[kdx].x * nij.z;
	tmp.z = Rij[kdx].x * nij.y - Rij[kdx].y * nij.x;
	
	float cef = ((W[jdx] + W[idx]) * nij.x + (W[jdx + N] + W[idx + N]) * nij.y + (W[jdx + 2 * N] + W[idx + 2 * N]) * nij.z) * dt * 0.5f;
	float3 phi;
	phi.x = nij.x * cef;
	phi.y = nij.y * cef;
	phi.z = nij.z * cef;
	
	Rij[kdx].x = nij.x;
	Rij[kdx].y = nij.y;
	Rij[kdx].z = nij.z;

	M1.x = 1.0f + tmp.z * phi.z + tmp.y * phi.y;
	M1.y = phi.z - tmp.z - tmp.y * phi.x;
	M1.z = -phi.y - tmp.z * phi.x + tmp.y;
	M2.x = tmp.z - phi.z - tmp.x * phi.y;
	M2.y = tmp.z * phi.z + 1.0f + tmp.x * phi.x;
	M2.z = -tmp.z * phi.y + phi.x - tmp.x;
	M3.x = -tmp.y - tmp.x * phi.z + phi.y;
	M3.y = -tmp.y * phi.z + tmp.x - phi.x;
	M3.z = tmp.y * phi.y + tmp.x * phi.x + 1.0f;
}

__device__ void dd_Calculate_fijn_bpm_1(const float* __restrict__ _1d_iL, float3 &Fijn, const uint& kdx, const float3& nij, const float& rijm, const float& m_E, const float& b_r)
{
	// calculate the force
	//double dStrainTotal = (dDistanceBetweenCenters - dBondInitLength) / dBondInitLength;
	//CVector3 vNormalForce = currentContact * (-1 * dBondCrossCut * _bondNormalStiffnesses[i] * dStrainTotal);
	float cef = b_r * b_r * MCf_pi * m_E * (rijm - __frcp_rn(_1d_iL[kdx])) * _1d_iL[kdx];
	Fijn.x += nij.x * cef;
	Fijn.y += nij.y * cef;
	Fijn.z += nij.z * cef;
}

/*__device__ void dd_Calculate_fijt_bpm(const float* __restrict__ V, const float* __restrict__ W, const float* __restrict__ _1d_iL, float3* __restrict__ Oijt, float3* __restrict__ Fijt, const uint& kdx, const uint& idx, const uint& jdx, const uint& N,
	const float3& nij, const float3& M1, const float3& M2, const float3& M3, const float& rijm, const float dt, const float& m_G, const float& b_r)
{
	//CVector3 relativeVelocity = _partVels[_bondLeftIDs[i]] - _partVels[_bondRightIDs[i]] - sumAngleVelocity * currentBond * 0.5;
	//CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
	//CVector3 tangentialVelocity = relativeVelocity - normalVelocity;
	//_bondTangentialOverlaps[i] = M * _bondTangentialOverlaps[i] - tangentialVelocity * _timeStep;
	//const CVector3 vTangentialForce = _bondTangentialOverlaps[i] * (_bondTangentialStiffnesses[i] * dBondCrossCut / dBondInitLength);
	float3 vij;
	vij.x = V[jdx] - V[idx] - 0.5f * (W[jdx] + W[idx]) * rijm;
	vij.y = V[jdx + N] - V[idx + N] - 0.5f * (W[jdx + N] - W[idx + N]) * rijm;
	vij.z = V[jdx + 2 * N] - V[idx + 2 * N] - 0.5f * (W[jdx + 2 * N] - W[idx + 2 * N]) * rijm;
	//printf("In %u %u %u | %e %e %e\n", kdx, idx, jdx, vij.x,  wijsum.y, 0);


	float cef = nij.x * vij.x + nij.y * vij.y + nij.z * vij.z;
	float3 oijt;
	oijt.x = M1.x * Oijt[kdx].x + M1.y * Oijt[kdx].y + M1.z * Oijt[kdx].z - (vij.x - nij.x * cef) * dt;
	oijt.y = M2.x * Oijt[kdx].x + M2.y * Oijt[kdx].y + M2.z * Oijt[kdx].z - (vij.y - nij.y * cef) * dt;
	oijt.z = M3.x * Oijt[kdx].x + M3.y * Oijt[kdx].y + M3.z * Oijt[kdx].z - (vij.z - nij.z * cef) * dt;


	cef = m_G * b_r * b_r * MCf_pi * _1d_iL[kdx];
	Fijt[kdx].x = oijt.x * cef;
	Fijt[kdx].y = oijt.y * cef;
	Fijt[kdx].z = oijt.z * cef;
}/**/

__device__ void dd_Calculate_mijn_bpm_1(const float* __restrict__ W, const float* __restrict__ _1d_iL, float3* __restrict__ Mijn, float3 &Mijnn, const uint& kdx, const uint& idx, const uint& jdx, const uint& N,
	const float3& nij, const float3& M1, const float3& M2, const float3& M3, const float dt, const float& m_G, const float& b_r)
{
	// normal angle velocity
	//CVector3 normalAngleVel = currentContact * DotProduct(currentContact, relAngleVel);
	//const CVector3 vBondNormalMoment = M * _bondNormalMoments[i] - normalAngleVel * (_timeStep * 2 * dBondAxialMoment * _bondTangentialStiffnesses[i] / dBondInitLength);	
	float cef1 = nij.x * (W[jdx] - W[idx]) + nij.y * (W[jdx + N] - W[idx + N]) + nij.z * (W[jdx + 2 * N] - W[idx + 2 * N]);
	float cef2 = dt * MCf_pi * b_r * b_r * b_r * b_r * 0.5f * m_G * _1d_iL[kdx];//2.0*0.25
	float3 mijn;
	mijn.x = M1.x * Mijn[kdx].x + M1.y * Mijn[kdx].y + M1.z * Mijn[kdx].z + nij.x * cef1 * cef2;
	mijn.y = M2.x * Mijn[kdx].x + M2.y * Mijn[kdx].y + M2.z * Mijn[kdx].z + nij.y * cef1 * cef2;
	mijn.z = M3.x * Mijn[kdx].x + M3.y * Mijn[kdx].y + M3.z * Mijn[kdx].z + nij.z * cef1 * cef2;
	Mijn[kdx].x = mijn.x;
	Mijn[kdx].y = mijn.y;
	Mijn[kdx].z = mijn.z;
	Mijnn.x += mijn.x;
	Mijnn.y += mijn.y;
	Mijnn.z += mijn.z;
}

__device__ void dd_Calculate_mijt_bpm_1(const float* __restrict__ W, const float* __restrict__ _1d_iL, float3* __restrict__ Mijt, float3 &Mijtt, const uint& kdx, const uint& idx, const uint& jdx, const uint& N,
	const float3& nij, const float3& M1, const float3& M2, const float3& M3, const float dt, const float& m_E, const float& b_r)
{
	// normal angle velocity	
	//CVector3 normalAngleVel = currentContact * DotProduct(currentContact, relAngleVel);	
	//CVector3 tangAngleVel = relAngleVel - normalAngleVel;
	//const CVector3 vBondTangentialMoment = M * _bondTangentialMoments[i] - tangAngleVel * (_timeStep * dBondAxialMoment * _bondNormalStiffnesses[i] / dBondInitLength);
	float cef1 = nij.x * (W[jdx] - W[idx]) + nij.y * (W[jdx + N] - W[idx + N]) + nij.z * (W[jdx + 2 * N] - W[idx + 2 * N]);
	float cef2 = dt * MCf_pi * b_r * b_r * b_r * b_r * 0.25f * m_E * _1d_iL[kdx];
	float3 mijt;
	mijt.x = M1.x * Mijt[kdx].x + M1.y * Mijt[kdx].y + M1.z * Mijt[kdx].z + (W[jdx] - W[idx] - nij.x * cef1) * cef2;
	mijt.y = M2.x * Mijt[kdx].x + M2.y * Mijt[kdx].y + M2.z * Mijt[kdx].z + (W[jdx + N] - W[idx + N] - nij.y * cef1) * cef2;
	mijt.z = M3.x * Mijt[kdx].x + M3.y * Mijt[kdx].y + M3.z * Mijt[kdx].z + (W[jdx + 2 * N] - W[idx + 2 * N] - nij.z * cef1) * cef2;
	Mijt[kdx].x = mijt.x;
	Mijt[kdx].y = mijt.y;
	Mijt[kdx].z = mijt.z;
	Mijtt.x += mijt.x;
	Mijtt.y += mijt.y;
	Mijtt.z += mijt.z;
}

__device__ void dd_Calculate_fijt_mijunsym_bpm_1(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W, const float* __restrict__ _1d_iL, float3* __restrict__ Oijt, float3 &Fijt, float3 &Mijadd, const uint& kdx, const uint& idx, const uint& jdx, const uint& N,
	const float3& nij, const float3& M1, const float3& M2, const float3& M3, const float& rijm, const float dt, const float& m_G, const float& b_r)
{
	//CVector3 relativeVelocity = _partVels[_bondLeftIDs[i]] - _partVels[_bondRightIDs[i]] - sumAngleVelocity * currentBond * 0.5;
	//CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
	//CVector3 tangentialVelocity = relativeVelocity - normalVelocity;
	//_bondTangentialOverlaps[i] = M * _bondTangentialOverlaps[i] - tangentialVelocity * _timeStep;
	//const CVector3 vTangentialForce = _bondTangentialOverlaps[i] * (_bondTangentialStiffnesses[i] * dBondCrossCut / dBondInitLength);
	//const CVector3 vUnsymMoment = currentBond * 0.5 * vTangentialForce;
	float3 rij, wijsd2;
	rij.x = R[jdx] - R[idx];
	rij.y = R[jdx + N] - R[idx + N];
	rij.z = R[jdx + 2 * N] - R[idx + 2 * N];
	wijsd2.x = 0.5f * (W[jdx] + W[idx]);
	wijsd2.y = 0.5f * (W[jdx + N] + W[idx + N]);
	wijsd2.z = 0.5f * (W[jdx + 2 * N] + W[idx + 2 * N]);
	float3 vij;
	vij.x = V[jdx] - V[idx] - wijsd2.y * rij.z + wijsd2.z * rij.y;
	vij.y = V[jdx + N] - V[idx + N] - wijsd2.z * rij.x + wijsd2.x * rij.z;
	vij.z = V[jdx + 2 * N] - V[idx + 2 * N] - wijsd2.x * rij.y + wijsd2.y * rij.x;
	//printf("In %u %u %u | %e %e %e\n", kdx, idx, jdx, vij.x,  wijsum.y, 0);

	float cef = nij.x * vij.x + nij.y * vij.y + nij.z * vij.z;
	float3 oijt;
	oijt.x = (M1.x * Oijt[kdx].x + M1.y * Oijt[kdx].y + M1.z * Oijt[kdx].z) + (vij.x - nij.x * cef) * dt;
	oijt.y = (M2.x * Oijt[kdx].x + M2.y * Oijt[kdx].y + M2.z * Oijt[kdx].z) + (vij.y - nij.y * cef) * dt;
	oijt.z = (M3.x * Oijt[kdx].x + M3.y * Oijt[kdx].y + M3.z * Oijt[kdx].z) + (vij.z - nij.z * cef) * dt;
	//printf("oijt %e %e %e\n", oijt.x, oijt.y, oijt.z);
	//printf("oijt %e %e %e | %e\n", M2.x, M2.y, M2.z, (vij.y - nij.y * cef));	
	cef = m_G * b_r * b_r * MCf_pi * _1d_iL[kdx];
	Fijt.x += oijt.x * cef;
	Fijt.y += oijt.y * cef;
	Fijt.z += oijt.z * cef;
	Oijt[kdx].x = oijt.x;
	Oijt[kdx].y = oijt.y;
	Oijt[kdx].z = oijt.z;
	
	Mijadd.x += 0.5f * (rij.y * oijt.z - rij.z * oijt.y) * cef;
	Mijadd.y += 0.5f * (rij.z * oijt.x - rij.x * oijt.z) * cef;
	Mijadd.z += 0.5f * (rij.x * oijt.y - rij.y * oijt.x) * cef;
}

__device__ void dd_Calculate_fijn_ppHM_1(const float* __restrict__ V, const float* __restrict__ W, float3 &Fijn,
	const uint& kdx, const uint& idx, const uint& jdx, const uint& N,
	const float& m_E, const float& m_G, const float& p_A, const float& mP, const float& rP,
	const float3& nij, const float& rijm, float3& vijt, float& Kt, float& fijnm)
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
	float cef1 = __fsqrt_rn(rP * fabsf(rP - 0.5f * rijm));
	float Kn = 2.0f * m_E * cef1;
	Kt = 8.0f * m_G * cef1;
	//printf("pp %e %e %e", cef1);
	float3 vij;
	vij.x = V[jdx] - V[idx] + rP * (nij.y * (W[jdx + 2 * N] + W[idx + 2 * N]) - nij.z * (W[jdx + N] + W[idx + N]));
	vij.y = V[jdx + N] - V[idx + N] + rP * (nij.z * (W[jdx] + W[idx]) - nij.x * (W[jdx + 2 * N] + W[idx + 2 * N]));
	vij.z = V[jdx + 2 * N] - V[idx + 2 * N] + rP * (nij.x * (W[jdx + N] + W[idx + N]) - nij.y * (W[jdx] + W[idx]));
	float vijnm = nij.x * vij.x + nij.y * vij.y + nij.z * vij.z;	
	fijnm = (rijm - 2.0f * rP) * Kn * MCf_2d3;
	cef1 = -1.8257f * p_A * vijnm * __fsqrt_rn(Kn * 0.5f * mP);
	Fijn.x += nij.x * (fijnm + cef1);
	Fijn.y += nij.y * (fijnm + cef1);
	Fijn.z += nij.z * (fijnm + cef1);

	vijt.x = vij.x - nij.x * vijnm;
	vijt.y = vij.y - nij.y * vijnm;
	vijt.z = vij.z - nij.z * vijnm;	
}

__device__ void dd_Calculate_fijtmijt_ppHM_1(float3* __restrict__ Oijt, float3 &Fijt, float3 &Mijt, 
	const uint& kdx, const float3& nij, const float& m_mu, const float& p_A, const float& mP, const float& rP, const float& dt, 
	const float3& vijt, const float &Kt, const float& fijnm)
{
	// normal force with damping
	float3 oijt;	
	// rotate old tangential force
	float cef1 = nij.x * Oijt[kdx].x + nij.y * Oijt[kdx].y + nij.z * Oijt[kdx].z;
	oijt.x = Oijt[kdx].x - nij.x * cef1;
	oijt.y = Oijt[kdx].y - nij.y * cef1;
	oijt.z = Oijt[kdx].z - nij.z * cef1;
	cef1 = __fmul_rn(oijt.x, oijt.x) + __fmul_rn(oijt.y, oijt.y) + __fmul_rn(oijt.z, oijt.z);
	if (cef1 > 1e-18)
	{
		cef1 = __frcp_rn(cef1);
		cef1 *= __fmul_rn(Oijt[kdx].x, Oijt[kdx].x) + __fmul_rn(Oijt[kdx].y, Oijt[kdx].y) + __fmul_rn(Oijt[kdx].z, Oijt[kdx].z);
		cef1 = __fsqrt_rn(cef1);
		oijt.x *= cef1;
		oijt.y *= cef1;
		oijt.z *= cef1;
	}
	oijt.x += vijt.x * dt;
	oijt.y += vijt.y * dt;
	oijt.z += vijt.z * dt;

	float3 fijt;
	fijt.x = oijt.x * Kt;
	fijt.y = oijt.y * Kt;
	fijt.z = oijt.z * Kt;
	// check slipping condition
	float fijtmm = __fmul_rn(fijt.x, fijt.x) + __fmul_rn(fijt.y, fijt.y) + __fmul_rn(fijt.z, fijt.z);
	if (fijtmm > m_mu * m_mu * fijnm * fijnm)
	{
		cef1 = m_mu * fabsf(fijnm) * __frsqrt_rn(fijtmm);
		fijt.x *= cef1;
		fijt.y *= cef1;
		fijt.z *= cef1;
		cef1 = __frcp_rn(Kt);
		oijt.x = fijt.x * cef1;
		oijt.y = fijt.y * cef1;
		oijt.z = fijt.z * cef1;
	}
	else
	{
		cef1 = -1.8257f * p_A * sqrt(Kt * 0.5f * mP);
		fijt.x += vijt.x * cef1;
		fijt.y += vijt.y * cef1;
		fijt.z += vijt.z * cef1;
	}
	Oijt[kdx].x = oijt.x;
	Oijt[kdx].y = oijt.y;
	Oijt[kdx].z = oijt.z;

	Fijt.x += fijt.x;
	Fijt.y += fijt.y;
	Fijt.z += fijt.z;
	Mijt.x += (nij.y * fijt.z - nij.z * fijt.y) * rP;
	Mijt.y += (nij.z * fijt.x - nij.x * fijt.z) * rP;
	Mijt.z += (nij.x * fijt.y - nij.y * fijt.x) * rP;
}

__device__ void dd_Calculate_mijroll_ppHM_1(const float* __restrict__ W, float3 &Mijadd, const uint& kdx, const uint& idx, const uint& N, const float& m_muroll, const float& fijnm)
{
	float3 wi;
	wi.x = W[idx];
	wi.y = W[idx + N];
	wi.z = W[idx + 2 * N];
	float wim = __fmul_rn(wi.x, wi.x) + __fmul_rn(wi.y, wi.y) + __fmul_rn(wi.z, wi.z);
	if (wim > 1e-18)
	{
		float cef1 = -m_muroll * fabsf(fijnm) * __frsqrt_rn(wim);
		Mijadd.x += wi.x * cef1;
		Mijadd.y += wi.y * cef1;
		Mijadd.z += wi.z * cef1;
	}
	else
	{
		Mijadd.x = 0;
		Mijadd.y = 0;
		Mijadd.z = 0;
	}
}

__global__ void d_CalculateForcesDEM_11(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float* __restrict__ F, float* __restrict__ M,
	const	uint N, const uint IonP, const float dt,	
	const float m_E, const float m_G, const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP)
{
	//float rr, fm, _1d_r;
	//float3 dr, fs;
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, jdx, kdx;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	//if(idx==0)
	//if(idx==0)printf("AAAAA\n");
	while (idx < N)
	{	
		float3 fs = { 0,0,0 }, ms = { 0,0,0 };
		for (kdx = idx * IonP; kdx < (idx + 1) * IonP; ++kdx)
		{
			jdx = IL[kdx];			
			if (jdx != UINT_MAX && ILtype[kdx] == 0)
			{
				float3 nij;
				float rijm;
				dd_Calculate_rijm_nij_1(R, idx, jdx, N, nij, rijm);
				if (rijm < 2.0f * rP)
				{
					//printf("In %u %u %u | %e %e %e | %e %e %e | %e %e %e\n", kdx, idx, jdx, rij.x, rij.y, rij.z, wij.x, wij.y, wij.z, wijsum.x, wijsum.y, wijsum.z);
					float3 vijt;
					float Kt, fijnm;
					//CHECK Overlap!
					dd_Calculate_fijn_ppHM_1(V, W, fs, kdx, idx, jdx, N, m_E, m_G, p_A, mP, rP, nij, rijm, vijt, Kt, fijnm);
					//printf("pp %e %e %e %e %e\n", 1.0f*mP, 1.0f*rP);
					//printf("pp %e %e %e %e %e\n", vijt.x, vijt.y, vijt.z, Kt, fijnm);
					//printf("pp %u %u %u | %e %e %e | %e %e\n", kdx, idx, jdx, fijnm, rijm);
					//Fijn[kdx].x = 0;
					//Fijn[kdx].y = 0;
					//Fijn[kdx].z = 0;
					dd_Calculate_fijtmijt_ppHM_1(Oijt, fs, ms, kdx, nij, m_mu, p_A, mP, rP, dt, vijt, Kt, fijnm);
					//printf("pp %e %e %e |\n", Kt);
					//Fijt[kdx].x = 0;
					//Fijt[kdx].y = 0;
					//Fijt[kdx].z = 0;
					//Oijt[kdx].x = 0;
					//Oijt[kdx].y = 0;
					//Oijt[kdx].z = 0;
					//Mijn[kdx].x = 0;
					//Mijn[kdx].y = 0;
					//Mijn[kdx].z = 0;
					//Mijt[kdx].x = 0;
					//Mijt[kdx].y = 0;
					//Mijt[kdx].z = 0;
					dd_Calculate_mijroll_ppHM_1(W, ms, kdx, idx, N, m_muroll, fijnm);
					//Mijadd[kdx].x = 0;
					//Mijadd[kdx].y = 0;
					//Mijadd[kdx].z = 0;	
				}
				else
				{
					//Fijn[kdx].x = 0; Fijn[kdx].y = 0; Fijn[kdx].z = 0;
					//Fijt[kdx].x = 0; Fijt[kdx].y = 0; Fijt[kdx].z = 0;
					Oijt[kdx].x = 0; Oijt[kdx].y = 0; Oijt[kdx].z = 0;
					//Mijn[kdx].x = 0; Mijn[kdx].y = 0; Mijn[kdx].z = 0;
					//Mijt[kdx].x = 0; Mijt[kdx].y = 0; Mijt[kdx].z = 0;					
					//Mijadd[kdx].x = 0; Mijadd[kdx].y = 0; Mijadd[kdx].z = 0;
				}
			}
			else if (jdx == UINT_MAX)
			{
				//Fijn[kdx].x = 0; Fijn[kdx].y = 0; Fijn[kdx].z = 0;
				//Fijt[kdx].x = 0; Fijt[kdx].y = 0; Fijt[kdx].z = 0;
				Oijt[kdx].x = 0; Oijt[kdx].y = 0; Oijt[kdx].z = 0;
				//Mijn[kdx].x = 0; Mijn[kdx].y = 0; Mijn[kdx].z = 0;
				//Mijt[kdx].x = 0; Mijt[kdx].y = 0; Mijt[kdx].z = 0;
				//Mijadd[kdx].x = 0; Mijadd[kdx].y = 0; Mijadd[kdx].z = 0;
			}
		}
		F[idx] += fs.x;
		F[idx + N] += fs.y;
		F[idx + 2*N] += fs.z;
		M[idx] += ms.x;
		M[idx + N] += ms.y;
		M[idx + 2 * N] += ms.z;
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_CalculateForcesDEM_12(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float* __restrict__ F, float* __restrict__ M,
	const	uint N, const uint IonP, const float dt, const float m_E, const float m_G, const float b_r)
{
	//float rr, fm, _1d_r;
	//float3 dr, fs;
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, jdx, kdx;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	//if(idx==0)
	//if(idx==0)printf("AAAAA\n");
	while (idx < N)
	{
		uint type;
		float3 fs = { 0,0,0 }, ms = { 0,0,0 };
		for (kdx = idx * IonP; kdx < (idx + 1) * IonP; ++kdx)
		{
			jdx = IL[kdx];
			type = ILtype[kdx];
			if (jdx != UINT_MAX && type == 1)
			{				
				float3 nij;
				float rijm;
				dd_Calculate_rijm_nij_1(R, idx, jdx, N, nij, rijm);
				//printf("In %u %u %u | %e %e %e | %e %e %e | %e %e %e\n", kdx, idx, jdx, rij.x, rij.y, rij.z, wij.x, wij.y, wij.z, wijsum.x, wijsum.y, wijsum.z);
				float3 M1, M2, M3;
				dd_Calculate_fijn_bpm_1(_1d_iL, fs, kdx, nij, rijm, m_E, b_r);
				
				//printf("In %u %u %u | %e %e %e | %e \n", kdx, idx, jdx, Fijn[kdx].x, Fijn[kdx].y, Fijn[kdx].z, rijm);

				//if(Fijn[kdx].x* Fijn[kdx].x+ Fijn[kdx].y* Fijn[kdx].y+ Fijn[kdx].z* Fijn[kdx].z>1e-4)
				//printf("In %u %u %u | %e %e %e %e | %e %e %e\n", kdx, idx, jdx, rijm, rijm - 1.0 / _1d_iL[kdx], 1.0 / _1d_iL[kdx], Fijn[kdx].x * nij.x + Fijn[kdx].y * nij.y + Fijn[kdx].z * nij.z,
				//	nij.x*rijm, nij.y * rijm, nij.z * rijm);

				dd_Calculate_M123_bpm_1(R, W, Rij, kdx, idx, jdx, N, dt, nij, M1, M2, M3);
					////dd_Calculate_fijt_bpm(V, W, _1d_iL, Oijt, Fijt, kdx, idx, jdx, N, nij, M1, M2, M3, rijm, dt, m_G, b_r);
				dd_Calculate_mijn_bpm_1(W, _1d_iL, Mijn, ms, kdx, idx, jdx, N, nij, M1, M2, M3, dt, m_G, b_r);
				//Mijn[kdx].x = 0; Mijn[kdx].y = 0; Mijn[kdx].z = 0;
				//if(kdx == 35763)printf("In %u %u %u | %e %e %e\n", kdx, idx, jdx, Rij[kdx].x, Rij[kdx].y, Rij[kdx].z);
				dd_Calculate_mijt_bpm_1(W, _1d_iL, Mijt, ms, kdx, idx, jdx, N, nij, M1, M2, M3, dt, m_E, b_r);

				dd_Calculate_fijt_mijunsym_bpm_1(R, V, W, _1d_iL, Oijt, fs, ms, kdx, idx, jdx, N, nij, M1, M2, M3, rijm, dt, m_G, b_r);
				//Fijt[kdx].x = 0; Fijt[kdx].y = 0; Fijt[kdx].z = 0;
				//printf("In %u %u %u | %e %e %e | %e %e %e\n", kdx, idx, jdx, Fijt[kdx].x, Fijt[kdx].y, Fijt[kdx].z, Mijadd[kdx].x, Mijadd[kdx].y, Mijadd[kdx].z);
			}			
		}
		F[idx] += fs.x;
		F[idx + N] += fs.y;
		F[idx + 2 * N] += fs.z;
		M[idx] += ms.x;
		M[idx + N] += ms.y;
		M[idx + 2 * N] += ms.z;
		idx += blockDim.x * gridDim.x;
	}
}