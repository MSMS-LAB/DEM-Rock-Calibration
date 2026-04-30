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

__device__ void dd_CalculateForceParticleParticle(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W, 
	float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, 
	float3* __restrict__ Fij, const uint &N, const uint &idx, const uint &jdx, const uint &kdx, 
	const float &dt, const float &m_E, const float &m_G, const float &p_A, const float& m_mu, const float& m_muroll, const float &Rp, const float &Mp)
{
	float3 rij, rijd2, wi, wij, wijsum, vij, nij, phi, M1, M2, M3, vijn, vijt, wijn, wijt, fijn, fijt, mijn, mijt, oijt, Munsym, fijdampt, oijt_old, mroll;
	float rm, _1d_rm, rm0, _1d_rm0, tmp, sij, Iij, cef1, cef2, cef3, eps, vijnn, oijn, kn, kt, fijdampnn, fijnn, fijtt, oijtt2, wimm;
	//kdx//unsigned iColl = _collActivityIndices[iActivColl];
	//idx//unsigned iSrcPart = _collSrcIDs[iColl];
	//jdx//unsigned iDstPart = _collDstIDs[iColl];
	//paramE,G//SInteractProps prop = _interactProps[_collInteractPropIDs[iColl]];
	//double dNormalOverlap = _collNormalOverlaps[iColl];
	//double dEquivMass = _collEquivMasses[iColl];
	//const CVector3 srcAnglVel = _partAnglVels[iSrcPart];
	//const CVector3 dstAnglVel = _partAnglVels[iDstPart];
	//double dPartSrcRadius = _partRadii[iSrcPart];
	//double dPartDstRadius = _partRadii[iDstPart];

	rij.x = R[jdx] - R[idx];
	rij.y = R[jdx + N] - R[idx + N];
	rij.z = R[jdx + 2 * N] - R[idx + 2 * N];
	//const CVector3 vContactVector = _collContactVectors[iColl];
	//const CVector3 vRcSrc = vContactVector * (dPartSrcRadius / (dPartSrcRadius + dPartDstRadius));
	//const CVector3 vRcDst = vContactVector * (-dPartDstRadius / (dPartSrcRadius + dPartDstRadius));
	rm = norm3df(rij.x, rij.y, rij.z); //__fmul_rn(rij.x, rij.x) + __fmul_rn(rij.y, rij.y) + __fmul_rn(rij.z, rij.z);
	oijn = 2.0f * Rp - rm;
	_1d_rm = __frcp_rn(rm);
	//rm0 = InitialLength[kdx];
	nij.x = rij.x * _1d_rm;
	nij.y = rij.y * _1d_rm;
	nij.z = rij.z * _1d_rm;
	//const CVector3 vNormalVector = vContactVector.Normalized();
	wi.x = W[idx];
	wi.y = W[idx + N];
	wi.z = W[idx + 2 * N];
	wij.x = W[jdx] - W[idx];
	wij.y = W[jdx + N] - W[idx + N];
	wij.z = W[jdx + 2 * N] - W[idx + 2 * N];
	wijsum.x = W[jdx] + W[idx];
	wijsum.y = W[jdx + N] + W[idx + N];
	wijsum.z = W[jdx + 2 * N] + W[idx + 2 * N];
	// relative velocity (normal and tangential)
	vij.x = V[jdx] - V[idx] - 0.5f * wijsum.x * Rp;
	vij.y = V[jdx + N] - V[idx + N] - 0.5f * wijsum.y * Rp;
	vij.z = V[jdx + 2 * N] - V[idx + 2 * N] - 0.5f * wijsum.z * Rp;
	//const CVector3 vRelVel = _partVels[iDstPart] + dstAnglVel * vRcDst - (_partVels[iSrcPart] + srcAnglVel * vRcSrc);
	vijnn = nij.x * vij.x + nij.y * vij.y + nij.z * vij.z;
	vijn.x = nij.x * vijnn;
	vijn.y = nij.y * vijnn;
	vijn.z = nij.z * vijnn;
	//const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
	//const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
	vijt.x = vij.x - vijn.x;
	vijt.y = vij.y - vijn.x;
	vijt.z = vij.z - vijn.x;
	//const CVector3 vRelVelTang = vRelVel - vRelVelNormal;

	//normal and tangential overlaps
	//oijt.x = vijt.x * dt;
	//oijt.y = vijt.y * dt;
	//oijt.z = vijt.z * dt;
	//CVector3 vDeltaTangOverlap = vRelVelTang * _timeStep;

	// a set of parameters for fast access
	tmp = __frsqrt_rn(0.5f * Rp * oijn);
	//double dTemp2 = sqrt(_collEquivRadii[iColl] * dNormalOverlap);

	// normal force with damping
	kn = 2.0f * m_E * tmp;
	//double Kn = 2 * prop.dEquivYoungModulus * dTemp2;
	fijdampnn = -1.8257f * p_A * vijnn * __frsqrt_rn(kn * Mp);
	//const double dDampingForce = -1.8257 * prop.dAlpha * dRelVelNormal * sqrt(Kn * dEquivMass);
	fijnn = -oijn * kn * MCf_2d3;
	//const double dNormalForce = -dNormalOverlap * Kn * 2. / 3.;

	// increment of tangential force with damping
	kt = 8.0f * m_G * tmp;
	//double Kt = 8 * prop.dEquivShearModulus * dTemp2;
	cef1 = -1.8257f * p_A * __frsqrt_rn(kt * Mp);
	fijdampt.x = vijt.x * cef1;
	fijdampt.y = vijt.y * cef1;
	fijdampt.z = vijt.z * cef1;
	//CVector3 vDampingTangForce = vRelVelTang * (-1.8257 * prop.dAlpha * sqrt(Kt * dEquivMass));

	// rotate old tangential force
	oijt_old.x = Oijt[idx].x;
	oijt_old.y = Oijt[idx].y;
	oijt_old.z = Oijt[idx].z;
	//CVector3 vOldTangOverlap = _collTangOverlaps[iColl];
	cef1 = nij.x * oijt_old.x + nij.y * oijt_old.y + nij.z * oijt_old.z;
	oijt.x = oijt_old.x - nij.x * cef1 + vijt.x * dt;
	oijt.y = oijt_old.y - nij.y * cef1 + vijt.y * dt;
	oijt.z = oijt_old.z - nij.z * cef1 + vijt.z * dt;
	//CVector3 vTangOverlap = vOldTangOverlap - vNormalVector * DotProduct(vNormalVector, vOldTangOverlap);
	//oijtt2 = __fmul_rn(oijt_old.x, oijt_old.x) + __fmul_rn(oijt_old.y, oijt_old.y) + __fmul_rn(oijt_old.z, oijt_old.z);
	//double dTangOverlapSqrLen = vTangOverlap.SquaredLength();
	//if (dTangOverlapSqrLen > 0)
	//	vTangOverlap = vTangOverlap * vOldTangOverlap.Length() / sqrt(dTangOverlapSqrLen);
	//vTangOverlap += vDeltaTangOverlap;

	fijt.x = oijt.x * kt;
	fijt.y = oijt.y * kt;
	fijt.z = oijt.z * kt;
	//CVector3 vTangForce = vTangOverlap * Kt;

	// check slipping condition
	fijtt = norm3df(fijt.x, fijt.y, fijt.z);
	cef2 = m_mu * fabsf(fijnn);
	if (fijtt > cef2)
	{
		cef2 *= __frcp_rn(fijtt);
		fijt.x *= cef2;
		fijt.y *= cef2;
		fijt.z *= cef2;
		cef3 = __frcp_rn(kt);
		oijt.x = fijt.x * cef3;
		oijt.y = fijt.y * cef3;
		oijt.z = fijt.z * cef3;

	}
	else
	{
		fijt.x += fijdampt.x;
		fijt.y += fijdampt.y;
		fijt.z += fijdampt.z;
	}
	//double dNewTangForce = vTangForce.Length();
	//if (dNewTangForce > prop.dSlidingFriction * fabs(dNormalForce))
	//{
	//	vTangForce *= prop.dSlidingFriction * fabs(dNormalForce) / dNewTangForce;
	//	vTangOverlap = vTangForce / Kt;
	//}
	//else
	//	vTangForce += vDampingTangForce;

	// calculate rolling torque
	wimm = __fmul_rn(wi.x, wi.x) + __fmul_rn(wi.y, wi.y) + __fmul_rn(wi.z, wi.z);
	if (wimm > 1e-18)
	{
		cef1 = m_muroll * fabsf(fijnn) * __frsqrt_rn(wimm);
		mroll.x = -wi.x * cef1;
		mroll.y = -wi.y * cef1;
		mroll.z = -wi.z * cef1;
	}
	else
	{
		mroll.x = 0;
		mroll.y = 0;
		mroll.z = 0;
	}

	//const CVector3 vRollingTorque1 = srcAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
	//	srcAnglVel * (-1 * prop.dRollingFriction * fabs(dNormalForce) * dPartSrcRadius / srcAnglVel.Length()) : CVector3{ 0 };
	//const CVector3 vRollingTorque2 = dstAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
	//	dstAnglVel * (-1 * prop.dRollingFriction * fabs(dNormalForce) * dPartDstRadius / dstAnglVel.Length()) : CVector3{ 0 };


	// calculate moments and forces
	//const CVector3 vTotalForce = vNormalVector * (dNormalForce + dDampingForce) + vTangForce;
	//const CVector3 vResultMoment1 = vNormalVector * vTangForce * dPartSrcRadius + vRollingTorque1;
	//const CVector3 vResultMoment2 = vNormalVector * vTangForce * dPartDstRadius + vRollingTorque2;

	Rij[kdx].x = rij.x;
	Rij[kdx].y = rij.y;
	Rij[kdx].z = rij.z;
	//_bondPrevBonds[i] = currentBond;
	Fij[kdx].x = nij.x * (fijnn + fijdampnn) + fijt.x;
	Fij[kdx].y = nij.y * (fijnn + fijdampnn) + fijt.y;
	Fij[kdx].z = nij.z * (fijnn + fijdampnn) + fijt.z;
	Oijt[idx].x = oijt.x;
	Oijt[idx].y = oijt.y;
	Oijt[idx].z = oijt.z;
	//Mij.x = nij.x * fijt * Rp + mroll.x;
	//Mij.y = nij.x * fijt * Rp + mroll.y;
	//Mij.z = nij.x * fijt * Rp + mroll.z;
	
	
	
	//store results in collision
	//_collTangOverlaps[iColl] = vTangOverlap;
	//_collTotalForces[iColl] = vTotalForce;

	// apply moments and forces
	//CUDA_VECTOR3_ATOMIC_ADD(_partForces[iSrcPart], vTotalForce);
	//CUDA_VECTOR3_ATOMIC_SUB(_partForces[iDstPart], vTotalForce);
	//CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iSrcPart], vResultMoment1);
	//CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iDstPart], vResultMoment2);
}
/**/