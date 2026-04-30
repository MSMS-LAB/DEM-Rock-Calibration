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

void h_CalculateForceElastic(const float* R, const float* V, const float* W, float* F, float* M,
	const float* ab_r, const float* _1d_iL, const float* AxialMoment,
	float3* Rij, float3* OverlapTau, float3* Mijn, float3* Mijt, 
	float3* Fij, const uint N, const uint idx, const uint jdx, const uint kdx, 
	const float dt, const float m_E, const float m_G, const float m_Gcritn, const float m_Gcritt, uint type)
{

	//printf("In %u %u %u \n", kdx, idx, jdx);
	// relative angle velocity of contact partners
	float3 rij, wij, wijsum, vij, nij, tmp, phi, M1, M2, M3, vijn, vijt, wijn, wijt, fijn, fijt, mijn, mijt, otij, Munsym;
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
	//std::cerr << "C1 " << wij.x << " " << wij.y << " " << wij.z << " | " << wijsum.x << " " << wijsum.y << " " << wijsum.z << " | " << rij.x << " " << rij.y << " " << rij.z << "\n";
	//printf("In %u %u %u | %e %e %e | %e %e %e | %e %e %e\n", kdx, idx, jdx, rij.x, rij.y, rij.z, wij.x, wij.y, wij.z, wijsum.x, wijsum.y, wijsum.z);
	//CVector3 currentBond = GetSolidBond(Particles().Coord(_iRight), Particles().Coord(_iLeft), m_PBC);
	rm = sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z); //__fmul_rn(rij.x, rij.x) + __fmul_rn(rij.y, rij.y) + __fmul_rn(rij.z, rij.z);
	_1d_rm = 1.0/rm;
	//double dDistanceBetweenCenters = currentBond.Length();
	_1d_rm0 = _1d_iL[kdx];
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
	//std::cerr << "C2 " << " | " << rij.x << " " << rij.y << " " << rij.z << " | " << vij.x << " " << vij.y << " " << vij.z << " | " << rm << " " << rm0 << " " << b_r << " " << areaij << " " << Iij << "\n";
	//printf("In %u %u %u | %e %e %e\n", kdx, idx, jdx, vij.x,  wijsum.y, 0);
	//CVector3 relativeVelocity = _partVels[_bondLeftIDs[i]] - _partVels[_bondRightIDs[i]] - sumAngleVelocity * currentBond * 0.5;
	nij.x = rij.x * _1d_rm;
	nij.y = rij.y * _1d_rm;
	nij.z = rij.z * _1d_rm;
	//CVector3 currentContact = currentBond / dDistanceBetweenCenters;
	std::cerr << "C1 " << nij.x << " " << nij.y << " " << nij.z << " | " << rm << "\n";
	tmp.x = Rij[kdx].y * rij.z - Rij[kdx].z * rij.y;
	tmp.y = Rij[kdx].z * rij.x - Rij[kdx].x * rij.z;
	tmp.z = Rij[kdx].x * rij.y - Rij[kdx].y * rij.x;
	//CVector3 tempVector = _bondPrevBonds[i] * currentBond;
	cef1 = (wijsum.x * rij.x + wijsum.y * rij.y + wijsum.z * rij.z) * dt * 0.5f;
	phi.x = rij.x * cef1;
	phi.y = rij.y * cef1;
	phi.z = rij.z * cef1;
	//std::cerr << "C3 " << " | " << nij.x << " " << nij.y << " " << nij.z << " | " << phi.x << " " << phi.y << " " << phi.z << " | " << cef1 << "\n";
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
	vijt.y = vij.y - vijn.y;
	vijt.z = vij.z - vijn.z;
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
	//std::cerr << "C4 " << " | " << vijn.x << " " << vijn.y << " " << vijn.z << " | " << vijt.x << " " << vijt.y << " " << vijt.z << " | " << wijn.x << " " << wijn.y << " " << wijn.z << " | " << wijt.x << " " << wijt.y << " " << wijt.z << "\n";
	// calculate the force
	rm0 = 1.0/ _1d_rm0;
	epsijn = (rm - rm0) * _1d_rm0;
	//double dStrainTotal = (dDistanceBetweenCenters - dBondInitLength) / dBondInitLength;
	cef1 = -areaij * m_E * epsijn;
	fijn.x = nij.x * cef1;
	fijn.y = nij.y * cef1;
	fijn.z = nij.z * cef1;
	//CVector3 vNormalForce = currentContact * (-1 * dBondCrossCut * _bondNormalStiffnesses[i] * dStrainTotal);

	otij.x = M1.x * OverlapTau[kdx].x + M1.y * OverlapTau[kdx].y + M1.z * OverlapTau[kdx].z - vijt.x * dt;
	otij.y = M2.x * OverlapTau[kdx].x + M2.y * OverlapTau[kdx].y + M2.z * OverlapTau[kdx].z - vijt.y * dt;
	otij.z = M3.x * OverlapTau[kdx].x + M3.y * OverlapTau[kdx].y + M3.z * OverlapTau[kdx].z - vijt.z * dt;
	//_bondTangentialOverlaps[i] = M * _bondTangentialOverlaps[i] - tangentialVelocity * _timeStep;
	//std::cerr << "C5 " << otij.x << " " << otij.y << " " << otij.z <<"\n";
	std::cerr << "C5 " << M2.x << " " << M2.y << " " << M2.z << " " << vijt.y << " " << M2.z << "\n";
	cef1 =  m_G * areaij * _1d_rm0;
	fijt.x = otij.x * cef1;
	fijt.y = otij.y * cef1;
	fijt.z = otij.z * cef1;
	//const CVector3 vTangentialForce = _bondTangentialOverlaps[i] * (_bondTangentialStiffnesses[i] * dBondCrossCut / dBondInitLength);
	cef1 = dt * 2.0f * Iij * m_G * _1d_rm0;
	mijn.x = M1.x * Mijn[kdx].x + M1.y * Mijn[kdx].y + M1.z * Mijn[kdx].z - wijn.x * cef1;
	mijn.y = M2.x * Mijn[kdx].x + M2.y * Mijn[kdx].y + M2.z * Mijn[kdx].z - wijn.y * cef1;
	mijn.z = M3.x * Mijn[kdx].x + M3.y * Mijn[kdx].y + M3.z * Mijn[kdx].z - wijn.z * cef1;
	//const CVector3 vBondNormalMoment = M * _bondNormalMoments[i] - normalAngleVel * (_timeStep * 2 * dBondAxialMoment * _bondTangentialStiffnesses[i] / dBondInitLength);
	cef1 = dt * Iij * m_E * _1d_rm0;
	mijt.x = M1.x * Mijt[kdx].x + M1.y * Mijt[kdx].y + M1.z * Mijt[kdx].z - wijt.x * cef1;
	mijt.y = M2.x * Mijt[kdx].x + M2.y * Mijt[kdx].y + M2.z * Mijt[kdx].z - wijt.y * cef1;
	mijt.z = M3.x * Mijt[kdx].x + M3.y * Mijt[kdx].y + M3.z * Mijt[kdx].z - wijt.z * cef1;
	//const CVector3 vBondTangentialMoment = M * _bondTangentialMoments[i] - tangAngleVel * (_timeStep * dBondAxialMoment * _bondNormalStiffnesses[i] / dBondInitLength);

	Mijn[kdx].x = mijn.x;
	Mijn[kdx].y = mijn.y;
	Mijn[kdx].z = mijn.z;
	//_bondNormalMoments[i] = vBondNormalMoment;
	Mijt[kdx].x = mijt.x;
	Mijt[kdx].y = mijt.y;
	Mijt[kdx].z = mijt.z;
	//_bondTangentialMoments[i] = vBondTangentialMoment;

	Munsym.x = 0.5f * (rij.y * fijt.z - rij.z * fijt.y);
	Munsym.y = 0.5f * (rij.z * fijt.x - rij.x * fijt.z);
	Munsym.z = 0.5f * (rij.x * fijt.y - rij.y * fijt.x);

	//std::cerr << "C5 " <<epsijn << " "<< " | " << otij.x << " " << otij.y << " " << otij.z << " | " << fijn.x << " " << fijn.y << " " << fijn.z << " | " << fijt.x << " " << fijt.y << " " << fijt.z << " | " << mijn.x << " " << mijn.y << " " << mijn.z << " | " << mijt.x << " " << mijt.y << " " << mijt.z << " | " << Munsym.x << " " << Munsym.y << " " << Munsym.z << "\n";

	//const CVector3 vUnsymMoment = currentBond * 0.5 * vTangentialForce;
	Rij[kdx].x = rij.x;
	Rij[kdx].y = rij.y;
	Rij[kdx].z = rij.z;
	//_bondPrevBonds[i] = currentBond;
	Fij[kdx].x = fijn.x + fijt.x;
	Fij[kdx].y = fijn.y + fijt.y;
	Fij[kdx].z = fijn.z + fijt.z;

	printf("C7 %u %u %u %i | %e %e %e | %e %e %e | %e %e %e | %e %e %e | %e %e %e\n", kdx, idx, jdx, int(type), fijn.x, fijn.y, fijn.z, fijt.x, fijt.y, fijt.z, mijn.x, mijn.y, mijn.z, mijt.x, mijt.y, mijt.z, Munsym.x, Munsym.y, Munsym.z);


	//_bondTotalForces[i] = vNormalForce + vTangentialForce;
	float Sn, St, MSn, MSt, _1d_areaij, _1d_Iij;
	_1d_areaij = 1.0/areaij;
	if (signbit(epsijn))
		Sn = -sqrt(fijn.x * fijn.x + fijn.y * fijn.y + fijn.z * fijn.z) * _1d_areaij;
	else
		Sn = sqrt(fijn.x * fijn.x + fijn.y * fijn.y + fijn.z * fijn.z) * _1d_areaij;

	_1d_Iij = 1.0/Iij;
	St = sqrt(mijt.x * mijt.x + mijt.y * mijt.y + mijt.z * mijt.z) * b_r * _1d_Iij;
	MSn = sqrt(mijn.x * mijn.x + mijn.y * mijn.y + mijn.z * mijn.z) * 0.5f * b_r * _1d_Iij;
	MSt = sqrt(fijt.x * fijt.x + fijt.y * fijt.y + fijt.z * fijt.z) * _1d_areaij;
	if ((Sn + St > m_Gcritn || MSn + MSt > m_Gcritt))
	{
		fijn.x = 0;
		fijn.y = 0;
		fijn.z = 0;
		fijt.x = 0;
		fijt.y = 0;
		fijt.z = 0;
		mijn.x = 0;
		mijn.y = 0;
		mijn.z = 0;
		mijt.x = 0;
		mijt.y = 0;
		mijt.z = 0;
		Munsym.x = 0;
		Munsym.y = 0;
		Munsym.z = 0;
		type = 0;
	}
	std::cerr << "C6 " << epsijn << " " << " | " << fijn.x + fijt.x << " " << fijn.y + fijt.y << " " << fijn.z + fijt.z << " | " << mijn.x + mijt.x - Munsym.x << " " << mijn.y + mijt.y - Munsym.y << " " << mijn.z + mijt.z - Munsym.z << "\n";

	//printf("In %u %u %u \n", kdx, idx, jdx);
	//printf("In %u %u %u | %e %e %e | %e %e %e | %e %e %e | %e %e %e | %e %e %e\n", kdx, idx, jdx, fijn.x, fijn.y, fijn.z, 
	//	fijt.x, fijt.y, fijt.z, mijn.x, mijn.y, mijn.z, mijt.x, mijt.y, mijt.z, Munsym.x, Munsym.y, Munsym.z);
	F[idx] += fijn.x + fijt.x;
	F[idx + N] += fijn.y + fijt.y;
	F[idx + 2 * N] += fijn.z + fijt.z;
	M[idx] += mijn.x + mijt.x - Munsym.x;
	M[idx + N] += mijn.y + mijt.y - Munsym.y;
	M[idx + 2 * N] += mijn.z + mijt.z - Munsym.z;/**/
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