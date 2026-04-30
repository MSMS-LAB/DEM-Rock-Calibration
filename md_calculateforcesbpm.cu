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



__device__ void dd_CalculateForceElastic(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W, float* __restrict__ F, float* __restrict__ M,
	const float* __restrict__ ab_r, const float* __restrict__ _1d_iL, const float* __restrict__ AxialMoment,
	float3 &rij_p, float3 &oijt_p, float3 &mijn_p, float3& mijt_p, float3& fijn_p, float3& fijt_p, float3& Munsym, const uint &N, const uint& idx, const uint& jdx, const uint &kdx,
	const float &dt, const float& m_E, const float& m_G)
{

	//printf("In %u %u %u \n", kdx, idx, jdx);
	// relative angle velocity of contact partners
	float3 rij, wij, wijsum, vij, nij, tmp, phi, M1, M2, M3, vijn, vijt, wijn, wijt, oijt, fijn, fijt, mijn, mijt;
	float rm, _1d_rm, rm0, _1d_rm0, b_r, areaij, Iij, cef1, cef2, cef3, epsijn;
	wij.x = W[jdx] - W[idx];
	wij.y = W[jdx + N] - W[idx + N];
	wij.z = W[jdx + 2 * N] - W[idx + 2 * N];
	wijsum.x = W[jdx] + W[idx];
	wijsum.y = W[jdx + N] + W[idx + N];
	wijsum.z = W[jdx + 2 * N] + W[idx + 2 * N];
	//CVector3 relAngleVel = _partAnglVels[_bondLeftIDs[i]] - _partAnglVels[_bondRightIDs[i]];
	//CVector3 sumAngleVelocity = _partAnglVels[_bondLeftIDs[i]] + _partAnglVels[_bondRightIDs[i]];

	//CVector3 relAngleVel = Particles().AnglVel(_iLeft) - Particles().AnglVel(_iRight);
	
	// the bond in the global coordinate system
	rij.x = R[jdx] - R[idx];
	rij.y = R[jdx + N] - R[idx + N];
	rij.z = R[jdx + 2 * N] - R[idx + 2 * N];
	//printf("In %u %u %u | %e %e %e | %e %e %e | %e %e %e\n", kdx, idx, jdx, rij.x, rij.y, rij.z, wij.x, wij.y, wij.z, wijsum.x, wijsum.y, wijsum.z);
	
	
	
	//CVector3 vNormalForce = currentContact * (-1 * dBondCrossCut * _bondNormalStiffnesses[i] * dStrainTotal);
	//double dBondInitLength = _bondInitialLengths[i];
	b_r = ab_r[kdx];
	areaij = b_r * b_r * MCf_pi;
	//const double dBondCrossCut = _bondCrossCuts[i];
	Iij = AxialMoment[kdx];
	//const double dBondAxialMoment = _bondAxialMoments[i];
	//printf("In %u %u %u | %e %e %e %e %e\n", kdx, idx, jdx, rm, Iij);
	// optimized
	vij.x = V[jdx] - V[idx] - 0.5f * wijsum.x * rm;
	vij.y = V[jdx + N] - V[idx + N] - 0.5f * wijsum.y * rm;
	vij.z = V[jdx + 2 * N] - V[idx + 2 * N] - 0.5f * wijsum.z * rm;
	//printf("In %u %u %u | %e %e %e\n", kdx, idx, jdx, vij.x,  wijsum.y, 0);
	//CVector3 relativeVelocity = _partVels[_bondLeftIDs[i]] - _partVels[_bondRightIDs[i]] - sumAngleVelocity * currentBond * 0.5;
	
	//CVector3 currentContact = currentBond / dDistanceBetweenCenters;
	tmp.x = rij_p.y * rij.z - rij_p.z * rij.y;
	tmp.y = rij_p.z * rij.x - rij_p.x * rij.z;
	tmp.z = rij_p.x * rij.y - rij_p.y * rij.x;
	//CVector3 tempVector = _bondPrevBonds[i] * currentBond;
	cef1 = (wijsum.x * rij.x + wijsum.y * rij.y + wijsum.z * rij.z) * dt * 0.5f;
	phi.x = rij.x * cef1;
	phi.y = rij.y * cef1;
	phi.z = rij.z * cef1;
	
	//CVector3 Phi = currentContact * (DotProduct(sumAngleVelocity, currentContact) * _timeStep * 0.5);
	M1.x = 1 + tmp.z * phi.z + tmp.y * phi.y;
	M1.y = phi.z - tmp.z - tmp.y * phi.x;
	M1.z = -phi.y - tmp.z * phi.x + tmp.y;
	M2.x = tmp.z - phi.z - tmp.x * phi.y;
	M2.y = tmp.z * phi.z + 1 + tmp.x * phi.x;
	M2.z = -tmp.z * phi.y + phi.x - tmp.x;
	M3.x = -tmp.y - tmp.x * phi.z + phi.y;
	M3.y = -tmp.y * phi.z + tmp.x - phi.x;
	M3.z = tmp.y * phi.y + tmp.x * phi.x + 1;
	//CMatrix3 M(1 + tempVector.z * Phi.z + tempVector.y * Phi.y, Phi.z - tempVector.z - tempVector.y * Phi.x, -Phi.y - tempVector.z * Phi.x + tempVector.y,
	//	tempVector.z - Phi.z - tempVector.x * Phi.y, tempVector.z * Phi.z + 1 + tempVector.x * Phi.x, -tempVector.z * Phi.y + Phi.x - tempVector.x,
	//	-tempVector.y - tempVector.x * Phi.z + Phi.y, -tempVector.y * Phi.z + tempVector.x - Phi.x, tempVector.y * Phi.y + tempVector.x * Phi.x + 1);
	cef2 = nij.x * vij.x + nij.y * vij.y + nij.z * vij.z;
	vijn.x = nij.x * cef2;
	vijn.y = nij.y * cef2;
	vijn.z = nij.z * cef2;
	//CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
	vijt.x = vij.x - vijn.x;
	vijt.y = vij.y - vijn.x;
	vijt.z = vij.z - vijn.x;
	//CVector3 tangentialVelocity = relativeVelocity - normalVelocity;
	
	// normal angle velocity
	cef3 = nij.x * wij.x + nij.y * wij.y + nij.z * wij.z;
	wijn.x = nij.x * cef3;
	wijn.y = nij.y * cef3;
	wijn.z = nij.z * cef3;
	//CVector3 normalAngleVel = currentContact * DotProduct(currentContact, relAngleVel);
	wijt.x = wij.x - wijn.x;
	wijt.y = wij.y - wijn.x;
	wijt.z = wij.z - wijn.x;
	//CVector3 tangAngleVel = relAngleVel - normalAngleVel;

	
	
	oijt.x = M1.x * oijt_p.x + M1.y * oijt_p.y + M1.z * oijt_p.z - vijt.x * dt;
	oijt.y = M2.x * oijt_p.x + M2.y * oijt_p.y + M2.z * oijt_p.z - vijt.y * dt;
	oijt.z = M3.x * oijt_p.x + M3.y * oijt_p.y + M3.z * oijt_p.z - vijt.z * dt;
	//_bondTangentialOverlaps[i] = M * _bondTangentialOverlaps[i] - tangentialVelocity * _timeStep;

	cef1 =  m_G * areaij * _1d_rm0;
	fijt.x = oijt.x * cef1;
	fijt.y = oijt.y * cef1;
	fijt.z = oijt.z * cef1;
	//const CVector3 vTangentialForce = _bondTangentialOverlaps[i] * (_bondTangentialStiffnesses[i] * dBondCrossCut / dBondInitLength);
	cef1 = dt * 2.0f * Iij * m_G * _1d_rm0;
	mijn.x = M1.x * mijn_p.x + M1.y * mijn_p.y + M1.z * mijn_p.z - wijn.x * cef1;
	mijn.y = M2.x * mijn_p.x + M2.y * mijn_p.y + M2.z * mijn_p.z - wijn.y * cef1;
	mijn.z = M3.x * mijn_p.x + M3.y * mijn_p.y + M3.z * mijn_p.z - wijn.z * cef1;
	//const CVector3 vBondNormalMoment = M * _bondNormalMoments[i] - normalAngleVel * (_timeStep * 2 * dBondAxialMoment * _bondTangentialStiffnesses[i] / dBondInitLength);
	cef1 = dt * Iij * m_E * _1d_rm0;
	mijt.x = M1.x * mijt_p.x + M1.y * mijt_p.y + M1.z * mijt_p.z - wijt.x * cef1;
	mijt.y = M2.x * mijt_p.x + M2.y * mijt_p.y + M2.z * mijt_p.z - wijt.y * cef1;
	mijt.z = M3.x * mijt_p.x + M3.y * mijt_p.y + M3.z * mijt_p.z - wijt.z * cef1;
	//const CVector3 vBondTangentialMoment = M * _bondTangentialMoments[i] - tangAngleVel * (_timeStep * dBondAxialMoment * _bondNormalStiffnesses[i] / dBondInitLength);
	
	Munsym.x = 0.5f * (rij.y * fijt.z - rij.z * fijt.y);
	Munsym.y = 0.5f * (rij.z * fijt.x - rij.x * fijt.z);
	Munsym.z = 0.5f * (rij.x * fijt.y - rij.y * fijt.x);
	//const CVector3 vUnsymMoment = currentBond * 0.5 * vTangentialForce;
	/**/
	//const CVector3 partForce = vNormalForce + vTangentialForce;
	//const CVector3 partMoment1 = vBondNormalMoment + vBondTangentialMoment - vUnsymMoment;
	//const CVector3 partMoment2 = vBondNormalMoment + vBondTangentialMoment + vUnsymMoment;
	//CUDA_VECTOR3_ATOMIC_ADD(_partForces[_bondLeftIDs[i]], partForce);
	//CUDA_VECTOR3_ATOMIC_ADD(_partMoments[_bondLeftIDs[i]], partMoment1);
	//Ftoti.x = fijn.x + fijt.x;
	//Ftoti.y = fijn.x + fijt.x;
	//Ftoti.z = fijn.x + fijt.x;

	/*if (m_vConstantModelParameters[0] != 0.0) ToDo
	{
		// check the bond destruction
		double forceLength = vNormalForce.Length();
		if (dStrainTotal <= 0)	// compression
			forceLength *= -1;
		const double maxStress1 = forceLength / dBondCrossCut;
		const double maxStress2 = vBondTangentialMoment.Length() * _bondDiameters[i] / (2.0 * dBondAxialMoment);
		const double maxTorque1 = vTangentialForce.Length() / dBondCrossCut;
		const double maxTorque2 = vBondNormalMoment.Length() * _bondDiameters[i] / (4.0 * dBondAxialMoment);

		bool bondBreaks = false;
		if (m_vConstantModelParameters[0] == 1.0 &&		// standard breakage criteria
			(maxStress1 + maxStress2 >= _bondNormalStrengths[i]
				|| maxTorque1 + maxTorque2 >= _bondTangentialStrengths[i]))
			bondBreaks = true;
		if (m_vConstantModelParameters[0] == 2.0 &&		// alternative breakage criteria
			(maxStress1 >= _bondNormalStrengths[i]
				|| maxStress2 >= _bondNormalStrengths[i] && dStrainTotal > 0
				|| maxTorque1 >= _bondTangentialStrengths[i]
				|| maxTorque2 >= _bondTangentialStrengths[i]))
			bondBreaks = true;
		if (bondBreaks)
		{
			_bondActivities[i] = false;
			_bondEndActivities[i] = _time;
			continue; // if bond is broken do not apply forces and moments
		}
	}/**/

	// apply forces and moments directly to particles
	//const CVector3 partForce = vNormalForce + vTangentialForce;
	//const CVector3 partMoment1 = vBondNormalMoment + vBondTangentialMoment - vUnsymMoment;
	//const CVector3 partMoment2 = vBondNormalMoment + vBondTangentialMoment + vUnsymMoment;
	//CUDA_VECTOR3_ATOMIC_ADD(_partForces[_bondLeftIDs[i]], partForce);
	//CUDA_VECTOR3_ATOMIC_ADD(_partMoments[_bondLeftIDs[i]], partMoment1);
	//CUDA_VECTOR3_ATOMIC_SUB(_partForces[_bondRightIDs[i]], partForce);
	//CUDA_VECTOR3_ATOMIC_SUB(_partMoments[_bondRightIDs[i]], partMoment2);	
}