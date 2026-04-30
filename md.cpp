#include "md_data_types.h"
#include "md.h"
//
void initArrays(additional_data &A, particle_data& P, cell_data &C, interaction_list_data& IL, result_data& R)
{
	//A.gencreated = 0;
	//A.gen = nullptr;

	
	P.d_F = nullptr;
	P.d_V = nullptr;
	P.d_U = nullptr;
	P.d_R = nullptr;
	P.d_W = nullptr;
	P.d_M = nullptr;	
	P.d_Q = nullptr;
	P.h_F = nullptr;
	P.h_V = nullptr;
	P.h_U = nullptr;
	P.h_R = nullptr;	
	P.h_W = nullptr;
	P.h_M = nullptr;
	P.h_Q = nullptr;

	C.d_tmp = nullptr;
	C.d_tmp_old = nullptr;
	C.d_IP = nullptr;
	C.d_CI = nullptr;
	C.d_CIs = nullptr;
	C.d_pnC = nullptr;
	C.h_IP = nullptr;
	C.h_CI = nullptr;
	C.h_CIs = nullptr;
	C.h_pnC = nullptr;
	
	IL.d_IL = nullptr;
	IL.d_ILtype = nullptr;
	//IL.d_b_r = nullptr;
	IL.d_1d_iL = nullptr;
	//IL.d_AxialMoment = nullptr;
	IL.d_rij = nullptr;
	IL.d_Oijt = nullptr;
	IL.d_Fijn = nullptr;
	IL.d_Fijt = nullptr;
	IL.d_Mijn = nullptr;
	IL.d_Mijt = nullptr;
	IL.d_Mijadd = nullptr;
	IL.h_IL = nullptr;
	IL.h_ILtype = nullptr;
	//IL.h_b_r = nullptr;
	IL.h_1d_iL = nullptr;
	//IL.h_AxialMoment = nullptr;
	IL.h_rij = nullptr;
	IL.h_Oijt = nullptr;
	IL.h_Fijn = nullptr;
	IL.h_Fijt = nullptr;
	IL.h_Mijn = nullptr;
	IL.h_Mijt = nullptr;
	IL.h_Mijadd = nullptr;
	
	R.d_FL = nullptr;
	R.h_FL = nullptr;
	R.h_sFL = nullptr;
	/*P.h_F = nullptr;
	P.h_V = nullptr;
	P.h_U = nullptr;
	P.h_Ir0 = nullptr;
	P.h_BPR = nullptr;
	P.h_IM = nullptr;
	P.h_1d_IM = nullptr;

	P.d_F = nullptr;
	P.d_V = nullptr;
	P.d_U = nullptr;
	P.d_Ir0 = nullptr;
	P.d_BPR = nullptr;
	P.d_1d_IM = nullptr;

	P.h_In = nullptr;
	P.h_BP = nullptr;
	P.h_ShIn = nullptr;

	P.d_In = nullptr;
	P.d_BP = nullptr;
	P.d_ShIn = nullptr;
	P.h_BPDfi = nullptr;

	P0.h_RU0 = nullptr;
	P0.d_RU0 = nullptr;
	P0.d_U0 = nullptr;

	Padd.d_Fmm = nullptr;
	Padd.d_FResult = nullptr;
	Padd.h_V = nullptr;
	Padd.h_Fmm = nullptr;
	Padd.h_FResult = nullptr;
	Padd.d_Fbound = nullptr;
	Padd.h_Fbound = nullptr;
	Padd.h_LammpsAddParticles = nullptr;
	Padd.h_LammpsSumF = nullptr;
	Padd.h_Fbound0 = nullptr;
	Padd.d_Ebound = nullptr;
	//Padd.h_Ebound0 = nullptr;
	Padd.h_Ebound = nullptr;
	Padd.h_Ebound0 = nullptr;
	Padd.h_Ek0 = nullptr;
	Padd.d_Esum = nullptr;
	Padd.h_Esum = nullptr;
	Padd.h_Ubound = nullptr;
	Padd.h_Ubound0 = nullptr;
	Padd.d_Ubound = nullptr;

	Pnet.h_S = nullptr;
	Pnet.h_Sc = nullptr;

	ED.h_Step = nullptr;
	ED.h_E = nullptr;
	ED.h_sE = nullptr;
	ED.h_Ebound0 = nullptr;
	ED.h_Ek0 = nullptr;

	Padd.EF.h_EFb = nullptr;
	Padd.EF.h_EFb0 = nullptr;
	Padd.EF.h_CEF = nullptr;

	Padd.EF.d_EFb = nullptr;
	Padd.EF.d_EFb0 = nullptr;
	Padd.EF.d_CEF = nullptr;
	Padd.EF.EFMinMax = nullptr;
	Padd.EF.CEFAverage = nullptr;
	Padd.EF.EFcell0 = nullptr;
	Padd.EF.EFcell = nullptr;*/
}


void deleteArrays(additional_data &A, particle_data& P, cell_data& C, interaction_list_data& IL, result_data& R)
{
	free_particle_data(P);

	cudaFree(C.d_tmp_old);
	C.d_tmp_old = nullptr;
	C.d_tmp = nullptr;
	cudaFree(C.d_IP);
	C.d_IP = nullptr;
	cudaFree(C.d_CI);
	C.d_CI = nullptr;
	cudaFree(C.d_CIs);
	C.d_CIs = nullptr;
	cudaFree(C.d_pnC);
	C.d_pnC = nullptr;
	free(C.h_IP);
	C.h_IP = nullptr;
	free(C.h_CI);
	C.h_CI = nullptr;
	free(C.h_CIs);
	C.h_CIs = nullptr;
	free(C.h_pnC);
	C.h_pnC = nullptr;
	
	free_interaction_list_data(IL);

	cudaFree(R.d_FL);
	R.d_FL = nullptr;
	free(R.h_FL);
	R.h_FL = nullptr;
	free(R.h_FL);
	R.h_sFL = nullptr;
	/*free(P.h_U);
	P.h_U = nullptr;
	free(P.h_V);
	P.h_V = nullptr;

	free(P.h_VU);
	P.h_VU = nullptr;
	free(P.h_VV);
	P.h_VV = nullptr;

	free(P.h_FU);
	P.h_FU = nullptr;
	free(P.h_FV);
	P.h_FV = nullptr;

	//free(P.h_EkU);
	//P.h_EkU = nullptr;
	//free(P.h_EkV);
	//P.h_EkV = nullptr;	

	cudaFree(P.d_U);
	P.d_U = nullptr;
	cudaFree(P.d_V);
	P.d_V = nullptr;

	//cudaFree(P.d_U1);
	//P.d_U1 = nullptr;
	//cudaFree(P.d_V1);
	//P.d_V1 = nullptr;

	cudaFree(P.d_VU);
	P.d_VU = nullptr;
	cudaFree(P.d_VV);
	P.d_VV = nullptr;

	//cudaFree(P.d_VU1);
	//P.d_VU1 = nullptr;
	//cudaFree(P.d_VV1);
	//P.d_VV1 = nullptr;

	cudaFree(P.d_FU);
	P.d_FU = nullptr;
	cudaFree(P.d_FV);
	P.d_FV = nullptr;

	P.N = 0;
	P._1d_N = 0;

	free(P0.h_RU0);
	P0.h_RU0 = nullptr;
	free(P0.h_RV0);
	P0.h_RV0 = nullptr;
	free(P0.h_Ri);
	P0.h_Ri = nullptr;

	cudaFree(P0.d_RU0);
	P0.d_RU0 = nullptr;
	cudaFree(P0.d_RV0);
	P0.d_RV0 = nullptr;
	cudaFree(P0.d_Ri);
	P0.d_Ri = nullptr;

	P0.N = 0;
	P0._1d_N = 0;

	free(Padd.h_bD4);
	Padd.h_bD4 = nullptr;
	free(Padd.h_IM);
	Padd.h_IM = nullptr;
	free(Padd.h_Ek);
	Padd.h_Ek = nullptr;

	cudaFree(Padd.d_SR_V);
	Padd.d_SR_V = nullptr;
	cudaFree(Padd.d_ER_V);
	Padd.d_ER_V = nullptr;
	cudaFree(Padd.d_bD4);
	Padd.d_bD4 = nullptr;
	cudaFree(Padd.d_Ek);
	Padd.d_Ek = nullptr;*/

	//curandDestroyGenerator(A.gen);
	//A.gencreated = 0;
}

void free_particle_data(particle_data& P)
{
	cudaFree(P.d_R);
	P.d_R = nullptr;
	cudaFree(P.d_F);
	P.d_F = nullptr;
	cudaFree(P.d_V);
	P.d_V = nullptr;
	cudaFree(P.d_U);
	P.d_U = nullptr;
	cudaFree(P.d_M);
	P.d_M = nullptr;
	cudaFree(P.d_W);
	P.d_W = nullptr;
	cudaFree(P.d_Q);
	P.d_Q = nullptr;
	free(P.h_F);
	P.h_F = nullptr;
	free(P.h_V);
	P.h_V = nullptr;
	free(P.h_R);
	P.h_R = nullptr;
	free(P.h_M);
	P.h_M = nullptr;
	free(P.h_W);
	P.h_W = nullptr;
	free(P.h_Q);
	P.h_Q = nullptr;
}

void free_interaction_list_data(interaction_list_data& IL)
{
	cudaFree(IL.d_IL);
	IL.d_IL = nullptr;
	cudaFree(IL.d_ILtype);
	IL.d_ILtype = nullptr;
	//cudaFree(IL.d_b_r);
	//IL.d_b_r = nullptr;
	cudaFree(IL.d_1d_iL);
	IL.d_1d_iL = nullptr;
	//cudaFree(IL.d_AxialMoment);
	//IL.d_AxialMoment = nullptr;
	cudaFree(IL.d_rij);
	IL.d_rij = nullptr;
	cudaFree(IL.d_Oijt);
	IL.d_Oijt = nullptr;
	cudaFree(IL.d_Fijn);
	IL.d_Fijn = nullptr;
	cudaFree(IL.d_Fijt);
	IL.d_Fijt = nullptr;
	cudaFree(IL.d_Mijn);
	IL.d_Mijn = nullptr;
	cudaFree(IL.d_Mijt);
	IL.d_Mijt = nullptr;
	cudaFree(IL.d_Mijadd);
	IL.d_Mijadd = nullptr;

	free(IL.h_IL);
	IL.h_IL = nullptr;
	free(IL.h_ILtype);
	IL.h_ILtype = nullptr;
	//free(IL.h_b_r);
	//IL.h_b_r = nullptr;
	free(IL.h_1d_iL);
	IL.h_1d_iL = nullptr;
	//free(IL.h_AxialMoment);
	//IL.h_AxialMoment = nullptr;
	free(IL.h_rij);
	IL.h_rij = nullptr;
	free(IL.h_Oijt);
	IL.h_Oijt = nullptr;
	free(IL.h_Fijn);
	IL.h_Fijn = nullptr;
	free(IL.h_Fijt);
	IL.h_Fijt = nullptr;
	free(IL.h_Mijn);
	IL.h_Mijn = nullptr;
	free(IL.h_Mijt);
	IL.h_Mijt = nullptr;
	free(IL.h_Mijadd);
	IL.h_Mijadd = nullptr;
	
	
}