#pragma once
#include <cuda.h>
#include <vector_functions.h>
#include <curand.h>
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "md_phys_constants.h"

struct mp_task_data
{
	uint NSteps, NParam, NResult, Step, NSamples, dNSaveU, dNSaveB;
	double MaterialDeltaFactor, InitialPorosityU, InitialPorosityB;
	double* ParameterVariation, * ResultsGoal, * MaterialParametersSteps, * ResultsSteps,
		* RFunction, * dRFunction, * DRFunction, * FunctionalParametersWeight, * FunctionalParametersWeightMultiplier, * Functional, * dFunctional, * DFunctional,
		* ddFunctional, *RSamples;// , * ParameterVariationValue
	uint8_t ParameterVariationType; 
	bool CalculateUTest,  CalculateBTest;
	mp_task_data()
	{
		NSteps = 0;
		NParam = 0;
		NResult = 0;
		NSamples = 0;
		ParameterVariationType = 2;
		ParameterVariation = nullptr;
		//ParameterVariationValue = nullptr;
		ResultsGoal = nullptr;
		MaterialParametersSteps = nullptr;
		ResultsSteps = nullptr;
		RFunction = nullptr;
		dRFunction = nullptr;
		DRFunction = nullptr;
		FunctionalParametersWeight = nullptr;
		FunctionalParametersWeightMultiplier = nullptr;
		Functional = nullptr;
		dFunctional = nullptr;
		DFunctional = nullptr;
		ddFunctional = nullptr;
		
		RSamples = nullptr;

	}
	void createarrays()
	{
		if (NSteps > 0)
		{
			if (NParam > 0)
			{
				MaterialParametersSteps = new double[NSteps * NParam];
				ParameterVariation = new double[(NParam + 1) * NParam];
				if (NResult > 0)
				{

					RFunction = new double[NResult * (NParam + 1)];
					dRFunction = new double[NResult * (NParam + 1)];
					DRFunction = new double[NResult * (NParam + 1)];
				}
				dFunctional = new double[NParam + 1];
				ddFunctional = new double[NParam + 1];
				memset(dFunctional, 0, (NParam + 1) * sizeof(double));
				//ParameterVariationValue = new double[NParam + 1];

			}
			else std::cerr << "Error! Nparam=" << NParam << "\n";

			if (NResult > 0)
			{
				ResultsSteps = new double[NSteps * NResult];
				FunctionalParametersWeight = new double[NResult];
				FunctionalParametersWeightMultiplier = new double[NResult];
				Functional = new double[NResult];
				DFunctional = new double[NResult];
				ResultsGoal = new double[NResult];
			}
			else std::cerr << "Error! Nresult=" << NResult << "\n";
		}
		else std::cerr << "Error! NSteps=" << NSteps << "\n";
		if(NSamples > 0)
		{
			if (NResult > 0)
			{
				RSamples = new double[(NResult+1)*NSamples];
			}
		}

	}
	void deletearrays()
	{
		std::cerr << "T0 "<< MaterialParametersSteps<<"\n";
		//delete[] MaterialParametersSteps; MaterialParametersSteps = nullptr; std::cerr << "T1\n";
		/*delete[]ParameterVariation; std::cerr << "T2\n";
		delete[]ResultsSteps; std::cerr << "T3\n";
		delete[]RFunction; std::cerr << "T4\n";
		delete[]dRFunction; std::cerr << "T5\n";
		delete[]DRFunction; std::cerr << "T6\n";
		delete[]FunctionalParametersWeight; std::cerr << "T7\n";
		delete[]Functional; std::cerr << "T8\n";
		delete[]dFunctional; std::cerr << "T9\n";
		delete[]DFunctional; std::cerr << "T10\n";
		delete[]ddFunctional; std::cerr << "T11\n";
		delete[]ResultsGoal; std::cerr << "T12\n";/**/
		delete []RSamples; RSamples = nullptr;
	}

	void SetMaterialContractions()
	{
		double d = 0.4, a;
		if (MaterialParametersSteps[NParam * (Step + 1) + 0] < 1e+5 / stress_const)
			MaterialParametersSteps[NParam * (Step + 1) + 0] = 1e+5 / stress_const;
		if (MaterialParametersSteps[NParam * (Step + 1) + 1] < 1e+5 / stress_const)
			MaterialParametersSteps[NParam * (Step + 1) + 1] = 1e+5 / stress_const;
		a = (MaterialParametersSteps[NParam * (Step + 1) + 0] - MaterialParametersSteps[NParam * (Step + 1) + 1]) / MaterialParametersSteps[NParam * (Step + 1) + 0];
		/*if (a < d)
		{
			MaterialParametersSteps[NParam * (Step + 1) + 1] = (1.0 + d) * MaterialParametersSteps[NParam * (Step + 1) + 0];
		}
		else if (a > d)
		{
			MaterialParametersSteps[NParam * (Step + 1) + 1] = (1.0 - d) * MaterialParametersSteps[NParam * (Step + 1) + 0];
		}/**/
		if (MaterialParametersSteps[NParam * (Step + 1) + 2] < 1e+8 / stress_const)
			MaterialParametersSteps[NParam * (Step + 1) + 2] = 1e+8 / stress_const;
		if (MaterialParametersSteps[NParam * (Step + 1) + 3] < 1e-6 / length_const)
			MaterialParametersSteps[NParam * (Step + 1) + 3] = 1e-6 / length_const;
		if (MaterialParametersSteps[NParam * (Step + 1) + 4] < 1e-5 / length_const)
			MaterialParametersSteps[NParam * (Step + 1) + 4] = 1e-5 / length_const;
	}
	void CalculateStartParameters()
	{
		uint i, j;
		memset(ParameterVariation, 0, sizeof(double) * (NParam + 1) * NParam);
		//ParameterVariationValue[0] = 0;
		//for (i = 1; i < NParam+1; ++i)ParameterVariationValue[i] = 1.0;
		for (j = 0; j < NResult; ++j)
			FunctionalParametersWeight[j] = FunctionalParametersWeightMultiplier[j] / (ResultsGoal[j] * ResultsGoal[j]);
		if(!CalculateUTest){FunctionalParametersWeight[0]=1; FunctionalParametersWeight[0]=1; FunctionalParametersWeight[3]=1; ResultsGoal[0]=1; ResultsGoal[1]=1; ResultsGoal[3]=1;} 
		if(!CalculateBTest){FunctionalParametersWeight[2]=100; ResultsGoal[2]=1;}
		uint varflags[8][5] = { {1,1,1,0,0},{0,0,0,1,1},{1,1,1,1,1},{0,1,1,1,1},{0,0,1,1,1},{1,0,0,1,1},{1,0,0,0,0},{0,1,1,1,0} };
		uint flagstep = ParameterVariationType;// Step % 2;
		for (i = 0; i < NParam; ++i)
		{
			ParameterVariation[NParam * (i + 1) + i] = varflags[flagstep][i] * MaterialDeltaFactor * MaterialParametersSteps[Step * NParam + i];
		}
	}
	void CalculateNewParameters()
	{
		uint i, j, k;
		for (j = 0; j < NResult; ++j)
		{
			//RFunction[i] = ResultsSteps[i + NResult * Step];
			dRFunction[j] = 0;
		}
		for (i = 0; i < NParam; ++i)
		{
			//std::cerr << "PVV " << i << " " << ParameterVariationValue[i+1] << "\n";
			//if (ParameterVariation[i+1] > 1e-12)
			//{
			for (j = 0; j < NResult; ++j)
			{
				if (ParameterVariation[NParam * (j + 1) + j] > 1e-12)
					dRFunction[NResult * (i + 1) + j] = (RFunction[NResult * (i + 1) + j] - RFunction[j]) / (ParameterVariation[NParam * (j + 1) + j]);
				else
					dRFunction[NResult * (i + 1) + j] = 0;
				std::cerr << "dRF " << i << " " << j << " " << dRFunction[NResult * (i + 1) + j] << " " << RFunction[NResult * (i + 1) + j] << " " << RFunction[j] << " " << ParameterVariation[NParam * (j + 1) + j] << "\n";
			}

			//}
		}//std::cin.get();
		for (j = 0; j < NResult; ++j)
		{
			DFunctional[j] = ResultsGoal[j] - RFunction[j];
			Functional[j] = FunctionalParametersWeight[j] * DFunctional[j] * DFunctional[j];
			//std::cerr << "F " << j << " " << Functional[j] << "\n";
		}
		double bcoeff, dF_0 = 0, dF_1 = 0;
		for (i = 0; i < NParam; ++i)
		{
			ddFunctional[i + 1] = dFunctional[i + 1];
			dF_0 += dFunctional[i + 1] * dFunctional[i + 1];
			dFunctional[i + 1] = 0;
			for (j = 0; j < NResult; ++j)
			{
				dFunctional[i + 1] += -2 * FunctionalParametersWeight[j] * DFunctional[j] * dRFunction[NResult * (i + 1) + j];
			}
			dF_1 += dFunctional[i + 1] * dFunctional[i + 1];
		}
		if (dF_0 > 1e-12)
			bcoeff = dF_1 / dF_0;
		else
			bcoeff = 0;
		for (i = 0; i < NParam; ++i)
		{
			ddFunctional[i + 1] = dFunctional[i + 1] + bcoeff * ddFunctional[i + 1];
		}
		double tcoeff, a0 = 0, a1 = 0;
		for (j = 0; j < NResult; ++j)
		{
			for (i = 0; i < NParam; ++i)
			{
				a1 += FunctionalParametersWeight[j] * DFunctional[j] * dRFunction[NResult * (i + 1) + j] * ddFunctional[i + 1];
				for (k = 0; k < NParam; ++k)
				{
					a0 += FunctionalParametersWeight[j] * dRFunction[NResult * (i + 1) + j] * ddFunctional[i + 1] * dRFunction[NResult * (k + 1) + j] * ddFunctional[k + 1];
				}
			}
		}
		tcoeff = -a1 / a0;
		for (i = 0; i < NParam; ++i)
		{
			MaterialParametersSteps[(Step + 1) * NParam + i] = MaterialParametersSteps[(Step)*NParam + i] - tcoeff * ddFunctional[i + 1];
		}
		SetMaterialContractions();
		memset(ParameterVariation, 0, sizeof(double) * (NParam + 1) * NParam);
		uint varflags[8][5] = { {1,1,1,0,0},{0,0,0,1,1},{1,1,1,1,1},{0,1,1,1,1},{0,0,1,1,1},{1,0,0,1,1},{1,0,0,0,0},{0,1,1,1,0} };
		uint flagstep = 2;// Step % 2;
		for (i = 0; i < NParam; ++i)
		{
			ParameterVariation[NParam * (i + 1) + i] = varflags[flagstep][i] * MaterialDeltaFactor * MaterialParametersSteps[(Step + 1) * NParam + i];
		}
	}
	void CalculateNewParameters2()
	{
		//double normcoefficients[5] = { 1e+11 / stress_const, 1e8 / stress_const, 1e8 / stress_const, 1e-3 / length_const, 1e-4 / length_const };
		double normcoefficients[5] = { 1.0, 1.0, 1.0, 1.0, 1.0 };
		uint i, j, k;
		for (j = 0; j < NResult; ++j)
		{
			//RFunction[i] = ResultsSteps[i + NResult * Step];
			dRFunction[j] = 0;
		}
		for (i = 0; i < NParam; ++i)
		{
			//std::cerr << "PVV " << i << " " << ParameterVariationValue[i+1] << "\n";
			//if (ParameterVariation[i+1] > 1e-12)
			//{
			for (j = 0; j < NResult; ++j)
			{
				if (ParameterVariation[NParam * (i + 1) + i] > 1e-12)
				{
					dRFunction[NResult * (i + 1) + j] = normcoefficients[i] * (RFunction[NResult * (i + 1) + j] - RFunction[j]) / (ParameterVariation[NParam * (i + 1) + i]);
					std::cerr << "dRF " << i << " " << j << " " << normcoefficients[i] << " " << RFunction[NResult * (i + 1) + j] << " " << RFunction[j] << " " << (ParameterVariation[NParam * (i + 1) + i]) << "\n";
				}
				else
					dRFunction[NResult * (i + 1) + j] = 0;
				std::cerr << "dRF! " << i << " " << j << " " << dRFunction[NResult * (i + 1) + j] << "\n";
			}

			//}
		}//std::cin.get();
		for (j = 0; j < NResult; ++j)
		{
			DFunctional[j] = ResultsGoal[j] - RFunction[j];
			Functional[j] = FunctionalParametersWeight[j] * DFunctional[j] * DFunctional[j];
			std::cerr << "F " << j << " " << Functional[j] << " " << ResultsGoal[j] << " " << RFunction[j] << " " << FunctionalParametersWeight[j] << "\n";
		}
		double bcoeff, dF_0 = 0, dF_1 = 0;
		for (i = 0; i < NParam; ++i)
		{
			ddFunctional[i + 1] = dFunctional[i + 1];
			dF_0 += dFunctional[i + 1] * dFunctional[i + 1];
			dFunctional[i + 1] = 0;
			for (j = 0; j < NResult; ++j)
			{
				dFunctional[i + 1] += -2 * FunctionalParametersWeight[j] * DFunctional[j] * dRFunction[NResult * (i + 1) + j];
				std::cerr << "dF " << i << " " << j << " " << FunctionalParametersWeight[j] << " " << DFunctional[j] << " " << dRFunction[NResult * (i + 1) + j] << "\n";
			}
			std::cerr << "dF " << i << " " << dFunctional[i + 1] << "\n";
			dF_1 += dFunctional[i + 1] * dFunctional[i + 1];
			
		}
		std::cerr << "dF " << dF_0 << " " << dF_1 << "\n";
		if (dF_0 > 1e-12)
			bcoeff = dF_1 / dF_0;
		else
			bcoeff = 0;
		for (i = 0; i < NParam; ++i)
		{
			ddFunctional[i + 1] = dFunctional[i + 1] + bcoeff * ddFunctional[i + 1];
		}
		double tcoeff, a0 = 0, a1 = 0;
		for (j = 0; j < NResult; ++j)
		{
			for (i = 0; i < NParam; ++i)
			{
				a1 += FunctionalParametersWeight[j] * DFunctional[j] * dRFunction[NResult * (i + 1) + j] * ddFunctional[i + 1];
				for (k = 0; k < NParam; ++k)
				{
					a0 += FunctionalParametersWeight[j] * dRFunction[NResult * (i + 1) + j] * ddFunctional[i + 1] * dRFunction[NResult * (k + 1) + j] * ddFunctional[k + 1];
				}
				std::cerr << "a " << FunctionalParametersWeight[j] << " " << DFunctional[j] << " " << dRFunction[NResult * (i + 1) + j] << " " << ddFunctional[i + 1] << "\n";
			}
		}
		std::cerr << "a " << a0 << " " << a1 << "\n";
		tcoeff = -a1 / a0;
		for (i = 0; i < NParam; ++i)
		{
			MaterialParametersSteps[(Step + 1) * NParam + i] = MaterialParametersSteps[(Step)*NParam + i] - tcoeff * ddFunctional[i + 1] / normcoefficients[i];
			std::cerr << "MPS " << tcoeff << " " << ddFunctional[i + 1] << " " << normcoefficients[i] << "\n";
		}
		SetMaterialContractions();
		memset(ParameterVariation, 0, sizeof(double) * (NParam + 1) * NParam);
		uint varflags[8][5] = { {1,1,1,0,0},{0,0,0,1,1},{1,1,1,1,1},{0,1,1,1,1},{0,0,1,1,1},{1,0,0,1,1},{1,0,0,0,0},{0,1,1,1,0} };
		uint flagstep = ParameterVariationType;// Step % 2;
		for (i = 0; i < NParam; ++i)
		{
			ParameterVariation[NParam * (i + 1) + i] = varflags[flagstep][i] * MaterialDeltaFactor * MaterialParametersSteps[(Step + 1) * NParam + i];
			std::cerr << "PV " << varflags[flagstep][i] << " " << MaterialDeltaFactor << " " << MaterialParametersSteps[(Step + 1) * NParam + i] << "\n";
		}
	}
	void SaveResults(char* Name)
	{
		std::ofstream file;
		file.open(Name, std::ios::out | std::ios::app);
		file << "Step: " << Step << "\n";
		file << "Parameters (E, Ecrit, Gcrit, boundR, boundDistance): " << MaterialParametersSteps[(Step) * NParam + 0] * stress_const << " " << MaterialParametersSteps[(Step) * NParam + 1] * stress_const
			<< " " << MaterialParametersSteps[(Step) * NParam + 2] * stress_const << " " << MaterialParametersSteps[(Step) * NParam + 3] * length_const << " " << MaterialParametersSteps[(Step) * NParam + 4] * length_const << " " << "\n";
		file << "Results (UniaxialFractureStrain UniaxialFractureStress BrasilFractureStress): " << RFunction[0] << " " << RFunction[1] * stress_const << " " << RFunction[2] * stress_const << "\n";
		double l_sum=0; file << "DFunctional";for (uint j = 0; j < NResult; ++j) {file<<" "<<Functional[j]; l_sum+=Functional[j];} file<<" | "<<l_sum<<"\n";
		
		file.close();
		//std::cerr << "Save PBM " << Name << "\n"; std::cin.get();
		//std::cerr << "LL " << P.N << " " << pnid2 << " | " << P.N + pnid2 << " " << P.NI << " " << j << "\n";
	}
	
	void SaveSamplesResults(char* p_Name, const uint8_t& p_firstline, const uint32_t& p_iSample)
	{
		std::ofstream file;
		file.open(p_Name, std::ios::out | std::ios::app);
		if(p_firstline == 1)file << "Sample UniaxialFractureStrain UniaxialFractureStress BrasilFractureStrain BrasilFractureStress\n";
		
		file << p_iSample << " " << RFunction[NResult * p_iSample + 0] << " " << RFunction[NResult * p_iSample + 1] * stress_const << " " << RFunction[NResult * p_iSample + 2] << " " << RFunction[NResult * p_iSample + 3] * stress_const<< "\n";
		file.close();
		//std::cerr << "Save PBM " << Name << "\n"; std::cin.get();
		//std::cerr << "LL " << P.N << " " << pnid2 << " | " << P.N + pnid2 << " " << P.NI << " " << j << "\n";
	}
};
