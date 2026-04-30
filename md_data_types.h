#pragma once
#include <cuda.h>
#include <vector_functions.h>
#include <curand.h>
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "md_phys_constants.h"
const uint SMEMDIM = 64;

struct particle_data
{
	float* h_F, * h_V, * h_U, *h_R, * h_W, * h_M;
	float4* h_Q;
	float* d_F, * d_V, * d_U, * d_R, *d_W, *d_M, *d_Oiwt;
	float4* d_Q;
	//float* d_VU1, *d_U1, *d_VV1, *d_V1;
	//float* h_EkU, *h_EkV, *d_EkU, *d_EkV;
	uint N;
	float _1d_N;
};

struct cell_data
{
	//float* h_F, * h_V, * h_U;
	//float* d_F, * d_V, * d_U;
	//int* h_i;
	//int* d_i;
	void* d_tmp, *d_tmp_old;
	
	uint* d_IP, * d_CI, * d_CIs, *d_pnC;
	uint* h_IP, *h_CI, * h_CIs, *h_pnC;
	//uint2*;
	//float* d_VU1, *d_U1, *d_VV1, *d_V1;
	//float* h_EkU, *h_EkV, *d_EkU, *d_EkV;
	size_t dtmpN, dtmpN_old;
	uint N;
	uint3 Nr;
	float _1d_N, a, _1d_a;
	float3 _1d_Nr;
	void CalculateParameters(double s_b_x, double s_b_y, double s_b_z)
	{
		_1d_a = 1.0 / double(a);
		Nr.x = floorf(s_b_x * _1d_a) + 1;
		Nr.y = floorf(s_b_y * _1d_a) + 1;
		Nr.z = floorf(s_b_z * _1d_a) + 4;		
		N = Nr.x * Nr.y * Nr.z;		
		dtmpN_old = 128;		
		_1d_Nr.x = 1.0 / double(Nr.x);
		_1d_Nr.y = 1.0 / double(Nr.y);
		_1d_Nr.z = 1.0 / double(Nr.z);
		std::cerr << "Cparam " << _1d_a << " " << Nr.x << " " << Nr.y << " " << Nr.z << " " << N << " " << _1d_Nr.x << " " << _1d_Nr.y << " " << _1d_Nr.z << "\n";
		//std::cerr << "Cparam " << s_b_x << " " << s_b_y << " " << s_b_z << " " << s_b_x * _1d_a << " " << s_b_y * _1d_a << " " << s_b_z * _1d_a << "\n";
	}
};

struct additional_data
{
	curandGenerator_t gen;
	//uint Nold;
	int_fast8_t gencreated;
	uint bloks, ibloks;
	additional_data():gencreated(1) { curandCreateGenerator(&(gen), CURAND_RNG_PSEUDO_MTGP32); }
	~additional_data() {gencreated=0; curandDestroyGenerator(gen);}
	void CalculateParameters(uint p_n, uint i_n)
	{
		bloks = ceil(p_n / (SMEMDIM)) + 1;
		ibloks = ceil(i_n / (SMEMDIM)) + 1;
		if (gencreated == 0)
		{
			curandCreateGenerator(&(gen), CURAND_RNG_PSEUDO_MTGP32);
			gencreated = 1;
			std::cerr << "GEnCreate\n";
		}
	}
};

struct sample_data
{
	double3 A_real, B_real, size_real, center_real;
	double Vext;
	float3 L, A, B;
	float3 axis, center, size, sized2, hidenpoint, spacesize, spacesized2;
	float R0, H0, Vgenmax;
	uint PN;
	void SetSpaceSizeSi(double x, double y, double z)
	{
		spacesize.x = x / length_const;
		spacesize.y = y / length_const;
		spacesize.z = z / length_const;
		spacesized2.x = 0.5 * spacesize.x; spacesized2.y = 0.5 * spacesize.y; spacesized2.z = 0.5 * spacesize.z;
		std::cerr << "L1param " << spacesize.x << " " << spacesize.y << " " << spacesize.z << " " << spacesized2.x << " " << spacesized2.y << " " << spacesized2.z << "\n";
	}
	void SetLSi(double x, double y, double z, double p_r)
	{
		L.x = x / length_const - 2.0 * p_r;
		L.y = y / length_const - 2.0 * p_r;
		L.z = z / length_const - 2.0 * p_r;		
		std::cerr << "L2param " << L.x << " " << L.y << " " << L.z << " " << p_r << "\n";
	}
	void SetAxis(double x, double y, double z)
	{
		axis.x = x; axis.y = y; axis.z = z;
	}
	void CalculateParameters(double C_a, double p_r)
	{
		A.x = 1.01 * C_a; A.y = 1.01 * C_a; A.z = 1.01 * C_a;
		B.x = A.x + spacesize.x; B.y = A.y + spacesize.y; B.z = A.z + spacesize.z;		
		center.x = 0.5 * (A.x + B.x); center.y = 0.5 * (A.y + B.y); center.z = 0.5 * (A.z + B.z);
		size = L;
		sized2.x = 0.5 * L.x; sized2.y = 0.5 * L.y; sized2.z = 0.5 * L.z;
		R0 = 0.5 * L.x; H0 = 0.5 * L.z;
		Vext = MC_pi * (0.5 * L.x + p_r) * (0.5 * L.x + p_r) * (L.z + 2.0 * p_r);
		hidenpoint.x = A.x + 0.5 * spacesize.x;
		hidenpoint.y = A.y + 0.5 * spacesize.y;
		hidenpoint.z = (floorf(B.z / C_a) + 3 - 0.5) * C_a;
		A_real.x = center.x - sized2.x; A_real.y = center.y - sized2.y; A_real.z = center.z - sized2.z;
		B_real.x = center.x + sized2.x; B_real.y = center.y + sized2.y; B_real.z = center.z + sized2.z;
		size_real.x = B_real.x - A_real.x; size_real.y = B_real.y - A_real.y; size_real.z = B_real.z - A_real.z;
		center_real.x = 0.5 * (B_real.x + A_real.x); center_real.y = 0.5 * (B_real.y + A_real.y); center_real.z = 0.5 * (B_real.z + A_real.z);
		std::cerr << "Sparam " << hidenpoint.x << " " << hidenpoint.y << " " << hidenpoint.z << "\n";
	}
};

struct interaction_list_data
{
	uint* d_IL;
	uint* d_ILtype;
	float* d_1d_iL;// , * d_b_r, * d_AxialMoment;
	float3* d_rij, * d_Oijt, * d_Fijn, * d_Fijt, * d_Mijn, * d_Mijt, * d_Mijadd;
	uint* h_IL;
	uint* h_ILtype;
	float* h_1d_iL;// , * h_b_r, * h_AxialMoment;
	float3* h_rij, * h_Oijt, * h_Fijn, * h_Fijt, * h_Mijn, * h_Mijt, * h_Mijadd;
		
	uint IonP, N;
	float acut, aacut;
	void CalculateParameters(uint p_n, double a_farcut)
	{
		acut = a_farcut;
		aacut = acut * acut;
		N = p_n * IonP;
		std::cerr << "Iparam " << acut << " " << aacut << " " << N << "\n";
	}
};

struct potential_data
{
	double a, _1d_a, aa, c, p_m, p_1d_m, p_r, p_e, p_A, p_V, p_mu, p_mur, hm_E, hm_G, dt, k, vis, vism, visI, D, dt_d_m, p_I, dt_d_I,
		m_E, m_G, m_nu,	m_Ec, m_Gc, m_ro, b_r, b_S, b_d, b_dd,
		//m_E, m_G, particlematerial_Poisson,
		a_farcut, aa_farcut, Trayleigh, Tsim, nuV, nuW;
	//e - the coefficient of restitution
	void CalculateParameters()
	{
		m_G = 0.5 * m_E / (1.0 + m_nu);
		b_S = MC_pi * b_r * b_r;
		p_V = 4.0 * MC_1d3 * MC_pi * p_r * p_r * p_r;
		p_m = m_ro * p_V;
		b_dd = b_d * b_d;
		p_1d_m = 1.0 / p_m;
		p_I = 0.4 * p_m * p_r * p_r;
		hm_E = 0.5 * m_E / (1.0 - m_nu * m_nu);
		hm_G = 0.5 * m_G / (2.0 - m_nu);
		p_A = log(p_e) / sqrt(MC_pi * MC_pi + log(p_e) * log(p_e));
		aa_farcut = a_farcut * a_farcut;
		Trayleigh = MC_pi * p_r * sqrt(2.0 * m_ro * (1.0 + m_nu)) / (sqrt(m_E) * (0.163 * m_nu + 0.8766));
		double tmp[2] = { m_E * b_S / (2.0 * p_r), m_G * b_S / (2.0 * p_r) };
		if (tmp[0] > tmp[1])
			Tsim = 2.0 * sqrt(p_m / tmp[0]);
		else
			Tsim = 2.0 * sqrt(p_m / tmp[1]);

		if (Trayleigh < Tsim)
			dt = 0.05 * Trayleigh;
		else
			dt = 0.05 * Tsim;
		//Po.dt *= 0.1 * 10;
		D = 1e-15 * m_E * b_S / (2.0 * p_r);
		//std::cerr << "Po " << Po.D << "\n";
		a = (2.0 + 0 * 1e-6) * p_r;
		_1d_a = 1.0 / a;
		aa = a * a;
		dt_d_m = p_1d_m * dt;
		dt_d_I = dt / p_I;
	}
};

struct result_data
{
	float* d_FL;
	float* h_FL;
	double* h_sFL;
	uint* d_SI;
	uint* h_SI;
	uint* h_sSI;
	size_t N, bloks;
	uint Nsave, dNsave, stepsave, fileN;
	double sFL[2], sz0, Destruction, Porosity;
};

struct compression_data
{
	bool Loading, Fracture;
	float3 center, size, sized2;
	double3 size_d, center_d;	
	float R, RR, Hd2, C, mu, Area, _1d_Area;
	double V, VLoad, Top, Bottom, TopR, BottomR, FractureStress, FractureStrain, MinStress, ZeroStress, Stress, Strain, MaxStrain, Size0, ConditionFracture, LinearityDeviation;
	uint StepsStop;
	//uint StepsMaxStrain;
	void CalculateUParameters(double3& size_real, double3& center_real, potential_data& Po, double minstress, double maxstrain, double p_conditionfracture)
	{		
		size_d.x = size_real.x; size_d.y = size_real.y; size_d.z = size_real.z;
		size.x = size_d.x; size.y = size_d.y; size.z = size_d.z;
		Size0 = size_d.z;
		sized2.x = 0.5 * size_d.x; sized2.y = 0.5 * size_d.y; sized2.z = 0.5 * size_d.z;
		center_d.x = center_real.x; center_d.y = center_real.y; center_d.z = center_real.z;
		Bottom = center_d.z - 0.5 * size_d.z;
		BottomR = Bottom + Po.p_r;
		Top = center_d.z + 0.5 * size_d.z;
		TopR = Top - Po.p_r;
		center.x = center_d.x; center.y = center_d.y; center.z = center_d.z;
		Hd2 = sized2.z;
		R = 1.5 * sized2.x;
		RR = R * R;
		Area = MC_pi * (sized2.x + Po.p_r) * (sized2.x + Po.p_r);
		_1d_Area = 1.0 / Area;
		//std::cerr<<"Area "<<Area*(length_const*length_const)<<"\n";
		C = Po.m_E * Po.b_S / (4.0 * Po.p_r);
		mu = Po.p_mu;
		MinStress = minstress;
		ZeroStress = 0.33 * MinStress;
		FractureStress = 0;
		FractureStrain = 0;
		Strain = 0;
		Fracture = false;
		Loading = false;
		MaxStrain = maxstrain;
		ConditionFracture = p_conditionfracture;
		std::cerr<<"Top Bottom "<<Top<<" "<<Bottom<<"\n";
	}
	void CompressU(double pressure, potential_data& Po)
	{
		if (pressure < 1e-9)
		{
			//++nloaddelay;
			V = 10.0*VLoad;
		}
		else
		{
			//nloaddelay = 0;
			V = VLoad;
		}		
		size_d.z += V * Po.dt;
		Top += V * Po.dt;
		TopR = Top - 2.0 * Po.p_r;
		size.z = size_d.z;
		sized2.z = 0.5 * size_d.z;
		center_d.z += V * Po.dt;
		center.z = center_d.z;
		Hd2 = sized2.z;
		Strain = (size_d.z - Size0) / Size0;
	}
	void CompressU_2(result_data& R, potential_data& Po, uint &step)
	{
		double pS = 0.5*(-R.sFL[0]+R.sFL[1]), pD = fabs(R.sFL[0]+R.sFL[1]);
		if (pS*_1d_Area < 1e-9)
		{
			//++nloaddelay;
			V = 10.0*VLoad;
		} else if(pD>0.01*pS && pS*_1d_Area>2.0e+6/stress_const)
		{
			V = 0;
			//--step;
			//std::cerr<<"Wait "<<R.sFL[0]<<" "<<R.sFL[1]<<" "<<pS<<" "<<pD<<"\n";std::cin.get();
		}
		else
		{
			//nloaddelay = 0;
			V = VLoad;
		}		
		size_d.z += V * Po.dt;
		Top += V * Po.dt;
		TopR = Top - 2.0 * Po.p_r;
		size.z = size_d.z;
		sized2.z = 0.5 * size_d.z;
		center_d.z += V * Po.dt;
		center.z = center_d.z;
		Hd2 = sized2.z;
		Strain = (size_d.z - Size0) / Size0;
	}
	void CompressU_S(double pressure, potential_data& Po)
	{
		if (pressure < 1e-9)
		{
			//++nloaddelay;
			V = 10.0*VLoad;
		}
		else
		{
			//nloaddelay = 0;
			V = VLoad;
		}		
		size_d.z += V * Po.dt;
		Top += 0.5*V * Po.dt;
		TopR = Top - Po.p_r;
		Bottom -= 0.5*V * Po.dt;
		BottomR = Bottom + Po.p_r;		
		size.z = size_d.z;
		sized2.z = 0.5 * size_d.z;
		center_d.z += 0*V * Po.dt;
		center.z = center_d.z;
		Hd2 = sized2.z;
		Strain = (size_d.z - Size0) / Size0;
	}
	void CompressU_S2(result_data& R, potential_data& Po, uint &step)
	{
		double pS = 0.5*fabs(-R.sFL[0]+R.sFL[1]), pD = fabs(R.sFL[0]+R.sFL[1]);
		if (pS*_1d_Area < 1e-9)
		{
			//++nloaddelay;
			V = 10.0*VLoad;
		} else if(pD>0.05*pS && pS*_1d_Area>2.0e+6/stress_const)//
		{
			V = 0;
			//--step;
			//std::cerr<<"Wait "<<R.sFL[0]<<" "<<R.sFL[1]<<" "<<pS<<" "<<pD<<"\n";std::cin.get();
		}
		else
		{
			//nloaddelay = 0;
			V = VLoad;
		}		
		size_d.z += V * Po.dt;
		Top += 0.5*V * Po.dt;
		TopR = Top - Po.p_r;
		Bottom -= 0.5*V * Po.dt;
		BottomR = Bottom + Po.p_r;		
		size.z = size_d.z;
		sized2.z = 0.5 * size_d.z;
		center_d.z += 0*V * Po.dt;
		center.z = center_d.z;
		Hd2 = sized2.z;
		Strain = (size_d.z - Size0) / Size0;
	}
	void CompressU_S3(result_data& R, potential_data& Po, uint &step)
	{
		double pS = 0.5*fabs(-R.sFL[0]+R.sFL[1]), pD = fabs(R.sFL[0]+R.sFL[1]);
		if (pS*_1d_Area < 1e-9)
		{
			//++nloaddelay;
			V = 10.0*VLoad;
		} else if(pD>0.20*pS && pS*_1d_Area>2.0e+6/stress_const)//
		{
			if( fabs(V)>1e-15 ) StepsStop = 0;
			V = 0;
			++StepsStop;
			if(StepsStop>30000) {Fracture = true; std::cerr<<"NON SYMMETRIC Uniaxial!\n";}
			//--step;
			//std::cerr<<"Wait "<<R.sFL[0]<<" "<<R.sFL[1]<<" "<<pS<<" "<<pD<<"\n";std::cin.get();
		}/**/
		else
		{
			//nloaddelay = 0;
			V = VLoad;
		}		
		size_d.z += V * Po.dt;
		Top += 0.5*V * Po.dt;
		TopR = Top - Po.p_r;
		Bottom -= 0.5*V * Po.dt;
		BottomR = Bottom + Po.p_r;		
		size.z = size_d.z;
		sized2.z = 0.5 * size_d.z;
		center_d.z += 0*V * Po.dt;
		center.z = center_d.z;
		Hd2 = sized2.z;
		Strain = (size_d.z - Size0) / Size0;
	}
	void CalculateBParameters(double3& size_real, double3& center_real, potential_data& Po, double minstress, double maxstrain, double p_conditionfracture)
	{
		size_d.x = size_real.x; size_d.y = size_real.y; size_d.z = size_real.z;
		size.x = size_d.x; size.y = size_d.y; size.z = size_d.z;
		Size0 = size_d.y;
		sized2.x = 0.5 * size_d.x; sized2.y = 0.5 * size_d.y; sized2.z = 0.5 * size_d.z;
		center_d.x = center_real.x; center_d.y = center_real.y; center_d.z = center_real.z;
		Bottom = center_d.y - 0.5 * size_d.y;
		BottomR = Bottom + Po.p_r;
		Top = center_d.y + 0.5 * size_d.y;
		TopR = Top - Po.p_r;
		center.x = center_d.x; center.y = center_d.y; center.z = center_d.z;
		Hd2 = sized2.y;
		R = (sized2.x > sized2.z) ? sized2.x : 1.05*sized2.z;
		RR = R * R;
		Area = 0.5 * 2 * MC_pi * (sized2.x) * (size.z);
		_1d_Area = 1.0 / Area;
		C = Po.m_E * Po.b_S / (4.0 * Po.p_r);
		mu = Po.p_mu;
		MinStress = minstress;
		ZeroStress = 0.5 * MinStress;
		FractureStress = 0;
		FractureStrain = 0;
		Strain = 0;
		Fracture = false;
		Loading = false;
		MaxStrain = maxstrain;
		ConditionFracture = p_conditionfracture;
		std::cerr<<"Top Bottom "<<Top<<" "<<Bottom<<"\n";
	}
	void CompressB(double pressure, potential_data& Po)
	{
		if (pressure < 1e-9)
		{
			//++nloaddelay;
			V = 10.0 * VLoad;
		}
		else
		{
			//nloaddelay = 0;
			V = VLoad;
		}
		size_d.y += V * Po.dt;
		Top += V * Po.dt;
		TopR = Top - 2.0 * Po.p_r;
		size.y = size_d.y;
		sized2.y = 0.5 * size_d.y;
		center_d.y += V * Po.dt;
		center.y = center_d.y;
		Hd2 = sized2.y;
		Strain = (size_d.y - Size0) / Size0;
	}
	void CompressB_S2(result_data& R, potential_data& Po, uint &step)
	{
		double pS = 0.5*(-R.sFL[0]+R.sFL[1]), pD = fabs(R.sFL[0]+R.sFL[1]);
		if (pS*_1d_Area < 1e-9)
		{
			//++nloaddelay;
			V = 10.0 * VLoad;
		} else if(pD>0.05*pS && pS*_1d_Area>2.0e+6/stress_const) // 
		{
			V = 0;
			//--step;
			//std::cerr<<"Wait "<<R.sFL[0]<<" "<<R.sFL[1]<<" "<<pS<<" "<<pD<<"\n";std::cin.get();
		}
		else
		{
			//nloaddelay = 0;
			V = VLoad;
		}
		size_d.y += V * Po.dt;
		Top += 0.5*V * Po.dt;
		TopR = Top - Po.p_r;
		Bottom -= 0.5*V * Po.dt;
		BottomR = Bottom + Po.p_r;
		size.y = size_d.y;
		sized2.y = 0.5 * size_d.y;
		center_d.y += 0*V * Po.dt;
		center.y = center_d.y;
		Hd2 = sized2.y;
		Strain = (size_d.y - Size0) / Size0;
	}
	void CompressB_S3(result_data& R, potential_data& Po, uint &step)
	{
		double pS = 0.5*(-R.sFL[0]+R.sFL[1]), pD = fabs(R.sFL[0]+R.sFL[1]);
		if (pS*_1d_Area < 1e-9)
		{
			//++nloaddelay;
			V = 10.0 * VLoad;
		} else if(pD>0.20*pS && pS*_1d_Area>2.0e+6/stress_const) // 
		{
			if( fabs(V)>1e-15 ) StepsStop = 0;
			V = 0;
			++StepsStop;
			if(StepsStop>30000) {Fracture = true; std::cerr<<"NON SYMMETRIC Brazil!\n";}
			//--step;
			//std::cerr<<"Wait "<<R.sFL[0]<<" "<<R.sFL[1]<<" "<<pS<<" "<<pD<<"\n";std::cin.get();
		}/**/
		else
		{
			//nloaddelay = 0;
			V = VLoad;
		}
		size_d.y += V * Po.dt;
		Top += 0.5*V * Po.dt;
		TopR = Top - Po.p_r;
		Bottom -= 0.5*V * Po.dt;
		BottomR = Bottom + Po.p_r;
		size.y = size_d.y;
		sized2.y = 0.5 * size_d.y;
		center_d.y += 0*V * Po.dt;
		center.y = center_d.y;
		Hd2 = sized2.y;
		Strain = (size_d.y - Size0) / Size0;
	}
};

struct firerelax_data
{
	uint bloks4, NPpositiveMax, NPnegativeMax, NPpositive, NPnegative, Ndelay, MaxStepsRelaxation, StepsPreRelaxation;
	float dtmax, dtmin, dt0, dt, alpha0, alpha, dtgrow, dtshrink, alphashrink, FdotV, MdotW, Tsim;
	float* h_FdotV, * d_FdotV, *h_MdotW, * d_MdotW;
};



struct sample_random_distortion
{
	double M, sD, C;
	float* d_Rr;
	bool Distortion;
};

struct md_task_data
{
	particle_data P;
	cell_data C;
	sample_data S;
	additional_data A;
	interaction_list_data IL;
	potential_data Po;
	result_data R;
	compression_data Compress;
	sample_random_distortion SRD;
	bool SaveLammpsCreateSample;
	
};

struct mp_result_data
{

};

struct mp_mdparameters_data
{
	md_task_data* mdTD;
	sample_data S[2];
	
	float* h_R[2];
	uint PN[2];
	double MaximumParticleIntersection, PreMaximumParticleIntersection, UVelocity, BVelocity, UMinStress, BMinStress, UMaxStrain, BMaxStrain, UConditionFracture, BConditionFracture;
	mp_mdparameters_data(uint mdtd_n=1)
	{
		h_R[0] = nullptr;
		h_R[1] = nullptr;
		mdTD = new md_task_data[mdtd_n];
	}
	void createarrays(uint isample, uint p_n)
	{		
		h_R[isample] = (float*)malloc(3 * p_n * sizeof(float));
	}
	void deletearrays()
	{		
		if (h_R[0] != nullptr) { free(h_R[0]); h_R[0] = nullptr; }
		if (h_R[1] != nullptr) { free(h_R[1]); h_R[1] = nullptr; }
		delete[] mdTD; mdTD = nullptr;
	}
};