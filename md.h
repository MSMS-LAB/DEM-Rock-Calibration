#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "md_data_types.h"

#define pre_debugtest


void initArrays(additional_data &A, particle_data& P, cell_data& C, interaction_list_data& IL, result_data& R);
void deleteArrays(additional_data &A, particle_data& P, cell_data& C, interaction_list_data& IL, result_data& R);
void free_particle_data(particle_data& P);
void free_interaction_list_data(interaction_list_data& IL);

void generateParticles(particle_data& P, additional_data& A, sample_data& S);
__global__ void d_SetParticlesInParallelepiped(float* __restrict__ R, float* __restrict__ V, uint N, const float3 c, const float3 L, const float vm);
void generateVelocities(particle_data& P, additional_data& A, sample_data& S);


void CellDistributionInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C);
void CellDistribution(particle_data& P, additional_data& A, sample_data& S, cell_data& C);
__global__ void d_FillIndex(uint* __restrict__ CI, const uint N);
__global__ void d_CalculateCellIndex(const float* __restrict__ R, uint N, uint* __restrict__ CI, const float _1d_a, const uint3 cN);
__global__ void d_DetermineCellPointer(const uint* __restrict__ CIs, uint* __restrict__ pnC, const uint N, const uint CN);

void InteractionListInit(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL);
void InteractionListConstruct(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C);

__global__ void d_ConstructInteractionList(const float* __restrict__ R, const uint N, const uint* __restrict__ CI, const uint* __restrict__ CIs, const uint* __restrict__ pnC, uint* __restrict__ IL,  
const uint IonP, const float a, const float _1d_a, const float aacut, const uint3 cN, const uint CN);
__global__ void d_ConstructInteractionListA(float*  R, const uint IonP,  const uint CN);
__device__ void addlink(const float* __restrict__ R, const uint N, const uint* __restrict__ CIs, const uint idx, const uint nindx, const uint jindx, uint* __restrict__ IL, uint IonP, float aacut, uint& nkndx);

void RenewInteractionList_full(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL);

__global__ void d_CalculateForces(const float* __restrict__ R, float* __restrict__ F, const uint N, const uint* __restrict__ IL, float* __restrict__ Rijm, const uint IonP, const float D, const float aa);
__global__ void d_CalculateForcesLJ(const float* __restrict__ R, float* __restrict__ F, uint N, const uint* __restrict__ IL, const uint IonP, const float D, const float a2, const float _1d_a2);
__global__ void d_CalculateIncrements(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R, const uint N, const float _1d_Mass_m_dt, const float dt);
__global__ void d_CylinderRestriction(float* __restrict__ V, float* __restrict__ R, const uint N, const float3 axis, const float3 center, const float R0, const float H0);
__global__ void d_ParallelepipedRestriction(float* __restrict__ V, float* __restrict__ R, const uint N, const float3 center, const float3 sized2);
__global__ void d_CylinderRestrictionZ(float* __restrict__ V, float* __restrict__ R, const uint N, const float3 center, const float R0, const float H0, const float HideZ);
__global__ void d_CylinderBorderPush(float* __restrict__ Fcoeff, const float Fcoeffborder, float* __restrict__ R, const uint N, const float3 center, const float R0, const float H0);
__global__ void d_CylinderRestrictionZr(float* __restrict__ V, float* __restrict__ R, const float* __restrict__ dR, const uint N, const float3 center, const float R0, const float H0, const float HideZ);
__global__ void d_CylinderRestrictionZr2(float* __restrict__ V, float* __restrict__ R, const float* __restrict__ dR, const uint N, const float3 center, const float R0, const float H0, const float3 Hide);
__global__ void d_ReduceParticleNumber(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, const uint N, const uint imaxI, const uint* __restrict__ IL,  const uint IonP, const float3 hp);
__global__ void d_CalculateIncrementsViscos(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const uint N, const float _1d_Mass_m_dt, const float dt, const float vis);

void CalculateGPUStepsContractRelaxFIRE(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po, firerelax_data& Fire);
void SetFIREData(particle_data& P, potential_data& Po, firerelax_data& Fire);
__global__ void d_CalculateIncrementsFIRE(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R, const uint N, const float _1d_Mass_m_dt, const float dt, const float F_alpha);
__global__ void d_CalculateDecrementsHalfStepFIRE(const float* __restrict__ V, float* __restrict__ R, const uint N, const float dt_d2);
__global__ void d_FdotVEntire(const float* __restrict__ V, const float* __restrict__ F, float* FdotV, const uint N);

void RenewInteractionList_New(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL);
void InteractionListReConstruct(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C);
__global__ void d_ReConstructInteractionList(const float* __restrict__ R, uint N, const uint* __restrict__ CI, const uint* __restrict__ CIs, const uint* __restrict__ pnC, uint* __restrict__ IL, uint IonP, float a, float _1d_a, float aacut, uint3 cN, uint CN);
__device__ void addnewlink(const float* __restrict__ R, const uint& N, const uint* __restrict__ CIs, const uint& idx, const uint& nindx, const uint& jindx, uint* __restrict__ IL, uint& IonP, const float& aacut, uint& nkndx);
__device__ uint deletelinks(const float* __restrict__ R, const uint& N, const uint& idx, uint* __restrict__ IL, uint& IonP, float& aacut, uint& nkndx);

__global__ void d_braziltest_simple(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint N, const float3 c, const float RR, const float Yt, const float Yb, const float Ytr, const float Ybr, const float vt, const float Zcut);
__global__ void d_braziltest_simple2(float* __restrict__ R, float* V, float* F, float* __restrict__ FL, const uint N, const float3 c, const float RR, const float Yt, const float Yb, const float Ytr, const float Ybr, const float vt, const float Zcut);

__global__ void d_UniaxialCompression_simple(const float* __restrict__ R, const float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint N, const float3 c, const float RR, const float Hd2, const float C, const float mu, const float Zcut);
__global__ void d_UniaxialCompression2_simple(float* __restrict__ R, const float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint N, const float3 c, const float RR, const float Hd2, const float C, const float mu, const float Zcut);
__global__ void d_UniaxialCompression3_simple(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint N, const float3 c, const float RR, const float Zt, const float Zb, const float Ztr, const float Zbr, const float vt, const float Zcut);
__global__ void d_UniaxialCompression_hm(const float* __restrict__ R, const float* __restrict__ V,	const float* __restrict__ W, float* __restrict__ Oiwt, float* __restrict__ F, float* __restrict__ M,	
	const uint N, const float dt, const float m_E, const float m_G, const float p_A, const float m_mu, const float m_muroll, const float mP, const float rP,	
	float* __restrict__ FL, const float3 c, const float wall_RR, const float wall_Zt, const float wall_Zb, const float wall_V, const float Zcut);
__global__ void d_BrazilCompression_hm(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, float* __restrict__ Oiwt, float* __restrict__ F, float* __restrict__ M,
	const uint N, const float dt, const float m_E, const float m_G, const float p_A, const float m_mu, const float m_muroll, const float mP, const float rP,	
	float* __restrict__ FL, const float3 c, const float wall_RR, const float wall_Yt, const float wall_Yb, const float wall_V, const float Zcut);

__global__ void d_ParallelepipedCutRestriction(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, const uint N, const float3 center, const float3 sized2, const float3 hp);

__global__ void d_ConstructBoundInteractions(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	const	uint N, const uint IonP, const float aa_create, const float b_r);
__global__ void d_CalculateForcesDEM(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	const	uint N, const uint IonP, const float dt,
	const float m_E, const float m_G, const float b_r,
	const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP);
//__device__ void dd_CalculateForceElastic(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W, float* __restrict__ F, float* __restrict__ M,
//	const float* __restrict__ b_r, const float* __restrict__ InitialLength, const float* __restrict__ AxialMoment,
//	float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt,
//	float3* __restrict__ Fij, const uint& N, const uint& idx, const uint& jdx, const uint& kdx,
//	const float& dt, const float& m_E, const float& m_G, const float& m_Gcritn, const float& m_Gcritt, uint& type);
__device__ void dd_CalculateForceElastic(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W, float* __restrict__ F, float* __restrict__ M,
	const float* __restrict__ b_r, const float* __restrict__ _1d_iL, const float* __restrict__ AxialMoment,
	float3& rij_p, float3& oijt, float3& mijn, float3& mijt, float3& fijn, float3& fijt, float3& Munsym, 
	const uint& N, const uint& idx, const uint& jdx, const uint& kdx, const float& dt, const float& m_E, const float& m_G);

__device__ void dd_CalculateForceParticleParticle(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W,
	float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt,
	float3* __restrict__ Fij, const uint& N, const uint& idx, const uint& jdx, const uint& kdx,
	const float& dt, const float& m_E, const float& m_G, const float& p_A, const float& m_mu, const float& m_muroll, const float& Rp, const float& Mp);
__global__ void d_CalculateIncrementsDEM(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const float* __restrict__ M, float* __restrict__ W,
	const uint N, const float _1d_Mass_m_dt, const float dt, const float _1d_I_m_dt);
__global__ void d_CalculateIncrementsDEMViscos(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const float* __restrict__ M, float* __restrict__ W, const uint N, const float _1d_Mass_m_dt, const float dt, const float _1d_I_m_dt, const float vis, const float visw);
__global__ void d_DeformSampleRelaxationY(float* __restrict__ R, const uint N, const float3 center, const float Eps, const float Shift);
__global__ void d_DeformSampleRelaxationZ(float* __restrict__ R, const uint N, const float3 center, const float Eps, const float Shift);

	
void CalculateRelaxDEMFIRE(md_task_data& mdTD, firerelax_data &Fire);
__global__ void d_CalculateDecrementsHalfStepDEMFIRE(const float* __restrict__ V, float* __restrict__ R, const uint N, const float dt_d2);
__global__ void d_CalculateIncrementsDEMFIRE(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const float* __restrict__ M, float* __restrict__ W, const uint N, const float _1d_Mass_m_dt, const float dt, const float _1d_I_m_dt,
	const float F_alpha);
__global__ void d_CylinderRestrictionFireZr2(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, const uint N, const float3 center, const float R0, const float H0, const float3 Hide);
__global__ void d_MdotWEntire(const float* __restrict__ W, const float* __restrict__ M, float* MdotW, const uint N);


void SaveAllData(particle_data &P, cell_data &C, sample_data &S, additional_data &A, interaction_list_data &IL, potential_data &Po);
void LoadAllData(particle_data &P, cell_data &C, sample_data &S, additional_data &A, interaction_list_data &IL, potential_data &Po);

void write_sample_data(std::ofstream& file, sample_data& S);
void read_sample_data(std::ifstream& file, sample_data& S);
void write_particle_data(std::ofstream& file, particle_data& P);
void read_particle_data(std::ifstream& file, particle_data& P);
void write_interaction_list_data(std::ofstream& file, interaction_list_data& IL);
void read_interaction_list_data(std::ifstream& file, interaction_list_data& P);

void SaveLammpsDATASimple(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po,
	char* Name, bool flagv = true);


void h_CalculateForceElastic(const float* R, const float* V, const float* W, float* F, float* M,
	const float* b_r, const float* _1d_iL, const float* AxialMoment, float3* Rij, float3* Oijt, float3* Mijn, float3* Mijt, float3* Fij,
	const uint N, const interaction_list_data& IL, const particle_data& P, const float dt, const float m_E, const float m_G, const float m_Gcritn, const float m_Gcritt);

__device__ void dd_Calculate_rijm_nij(const float* __restrict__ R, const uint& idx, const uint& jdx, const uint& N, float3& nij, float& rijm);

__global__ void d_SumUpForcesDEM(const uint* __restrict__ IL, const float3* __restrict__ Fijn, const float3* __restrict__ Fijt, const float3* __restrict__ Mijn, const float3* __restrict__ Mijt, const float3* __restrict__ Mijadd,
	float* __restrict__ F, float* __restrict__ M, const	uint N, const uint IonP);
__global__ void d_SumUpForcesDEMViscos(const uint* __restrict__ IL, const float3* __restrict__ Fijn, const float3* __restrict__ Fijt, const float3* __restrict__ Mijn, const float3* __restrict__ Mijt, const float3* __restrict__ Mijadd,
	float* __restrict__ F, float* __restrict__ M, const float* __restrict__ V, const float* __restrict__ W, const	uint N, const uint IonP, const float nuV, const float nuW);


__global__ void d_CheckBreakConditionDEM(const uint* __restrict__ IL, const float* __restrict__ _1d_iL,
	const float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	uint* __restrict__ ILtype, const uint IN, const float m_Gcritn, const float m_Gcritt, const float b_r);

__global__ void d_CutCylinderSpecimen_simple(float* __restrict__ R, float* __restrict__ V, const uint N, const float3 center, const float RR0, const float H0, const float3 hidenpoint);
__global__ void d_DeleteFarLinks(const float* __restrict__ R, uint N, uint* __restrict__ IL, uint* __restrict__ ILtype, float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd, uint IonP, const float aacut);


__global__ void d_CalculateForcesDEM_1(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	const	uint N, const uint IonP, const float dt,
	const float m_E, const float m_G, const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP);
__global__ void d_CalculateForcesDEM_2(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	const	uint N, const uint IonP, const float dt, const float m_E, const float m_G, const float b_r);

void RenewInteractionList_BPM(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL);
void InteractionListReConstructBPM(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C);
__global__ void d_ReConstructInteractionListbpm(const float* __restrict__ R, uint N, const uint* __restrict__ CI, const uint* __restrict__ CIs, const uint* __restrict__ pnC, uint* __restrict__ IL, uint* __restrict__ ILtype, uint IonP, float a, float _1d_a, float aacut, uint3 cN, uint CN);
__device__ void addnewlinkbpm(const float* __restrict__ R, const uint& N, const uint* __restrict__ CIs, const uint& idx, const uint& nindx, const uint& jindx, uint* __restrict__ IL, uint* __restrict__ ILtype, uint& IonP, float& aacut, uint& nkndx);

void ResultsUInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C, interaction_list_data& IL, result_data& R, compression_data& Compress);
void ResultsBInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C, interaction_list_data& IL, result_data& R, compression_data& Compress);
void ResultsUDelete(result_data& R);
void SumForcesULoading(result_data& R, compression_data& Compress, uint n);
void SumForcesBLoading(result_data& R, compression_data& Compress, uint n);
void CalculateLinearityDeviation(result_data& R, compression_data & Compress);
void SaveSumForcesLoading(result_data& R, char* Name);
void CalculateDestruction(result_data& R);
void SumInteractions(result_data& R, interaction_list_data& IL, uint n);
__global__ void d_SumUpInteractions(uint* __restrict__ IL, uint* __restrict__ ILtype, uint IonP, uint* __restrict__ SI, uint N);


void h_CheckInteractions(particle_data& P, cell_data& C, additional_data& A, sample_data& S, interaction_list_data& IL);
void CheckDATA(particle_data& P, potential_data& Po, cell_data& C, additional_data& A, sample_data& S, interaction_list_data& IL);

void ReadParticlesCSV(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po, char* Name, uint N);

__global__ void d_CalculateForcesDEM_11(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float* __restrict__ F, float* __restrict__ M,
	const	uint N, const uint IonP, const float dt,
	const float m_E, const float m_G, const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP);
__global__ void d_CalculateForcesDEM_12(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float* __restrict__ F, float* __restrict__ M,
	const	uint N, const uint IonP, const float dt, const float m_E, const float m_G, const float b_r);

__global__ void d_CalculateForcesDEM_21(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float* __restrict__ F, float* __restrict__ M,
	const	uint N, const uint IonP, const float dt,
	const float m_E, const float m_G, const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP);

__global__ void d_CalculateForcesDEM_22(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float* __restrict__ F, float* __restrict__ M,
	const	uint N, const uint IonP, const float dt, const float m_E, const float m_G, const float b_r, const float m_Gcritn, const float m_Gcritt);
	
__global__ void d_CalculateForcesDEM_22_ncb(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint* __restrict__ IL, uint* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float* __restrict__ F, float* __restrict__ M,
	const	uint N, const uint IonP, const float dt, const float m_E, const float m_G, const float b_r, const float m_Gcritn, const float m_Gcrittd);


