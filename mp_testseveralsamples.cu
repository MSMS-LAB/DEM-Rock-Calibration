#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include "pcuda_helper.h"
#include "md_data_types.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include "md.h"
#include "mp.h"
#include "mp_task.h"
#include <chrono>


void TestSeveralSamples()
{
    mp_task_data mpTask;
	mpTask.NSteps = 1;	
    mpTask.NSamples = 1;
	mpTask.NParam = 5;
	mpTask.NResult = 3;	
    mpTask.createarrays();
	//std::cerr << "q0\n";
	mpTask.MaterialParametersSteps[0] = 1.95754e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 2.74056e+11 / stress_const;
    mpTask.MaterialParametersSteps[2] = 2.23809e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000272858 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000285998 / length_const;/**/
	
	mpTask.MaterialParametersSteps[0] = 1.95754e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 2.74056e+11 / stress_const;
    mpTask.MaterialParametersSteps[2] = 2.99257e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000209362 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000307471 / length_const;/**/
	
	mpTask.MaterialParametersSteps[0] = 1.71066e+10 / stress_const;
    mpTask.MaterialParametersSteps[1] = 6.18389e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 5.03332e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000409345 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00046797 / length_const;
	
	
	mpTask.CalculateUTest = true;
	mpTask.CalculateBTest = false;
	bool l_CreateSamples = true;
	
	mpTask.InitialPorosityU = 0.375;
	mpTask.InitialPorosityB = 0.37;
	//mpTask.ParameterVariationType = 2;//0 - stress parameters, 1 - bond parameters, 2 - all parameters
	mpTask.dNSaveU = 100; 
	mpTask.dNSaveB = 1000; 
	uint l_FirstCreationSampleNumber = 0;
	char filename[256] = "", filenamepart[256] = "";
	
	       
    //std::cerr << "q1\n"; std::cin.get();
    mp_mdparameters_data mpMDP;
	//std::cerr << "q1\n";
    SetInitialParameters(mpMDP.mdTD[0], mpTask);
    //std::cerr << "q3\n"; mpTask.CalculateNewParameters();
    //std::cerr << "q4\n"; //std::cin.get();
    double sc = 1.0;
    float3 LSi[2] = { {sc * 0.025, sc * 0.025, sc * 0.05},{sc * 0.05, sc * 0.05, sc * 0.02} }, SpaceSize[2] = { {0.04, 0.04, 0.06}, {0.065, 0.065, 0.03} };

	uint i;
    for (i = l_FirstCreationSampleNumber; i < mpTask.NSamples && l_CreateSamples; ++i)
	{
		mpMDP.S[0].SetSpaceSizeSi(SpaceSize[0].x, SpaceSize[0].y, SpaceSize[0].z);
		mpMDP.S[0].SetLSi(LSi[0].x, LSi[0].y, LSi[0].z, mpMDP.mdTD[0].Po.p_r);
		mpMDP.S[0].SetAxis(0.0, 0.0, 1.0);
		mpMDP.S[0].CalculateParameters(mpMDP.mdTD[0].C.a, mpMDP.mdTD[0].Po.p_r);
		mpMDP.S[0].PN = (1.0 - mpTask.InitialPorosityU) * mpMDP.S[0].Vext / mpMDP.mdTD[0].Po.p_V;
		mpMDP.S[1].SetSpaceSizeSi(SpaceSize[1].x, SpaceSize[1].y, SpaceSize[1].z);
		mpMDP.S[1].SetLSi(LSi[1].x, LSi[1].y, LSi[1].z, mpMDP.mdTD[0].Po.p_r);
		mpMDP.S[1].SetAxis(0.0, 0.0, 1.0);
		mpMDP.S[1].CalculateParameters(mpMDP.mdTD[0].C.a, mpMDP.mdTD[0].Po.p_r);
		mpMDP.S[1].PN = (1.0 - mpTask.InitialPorosityB) * mpMDP.S[1].Vext / mpMDP.mdTD[0].Po.p_V;
		initArrays(mpMDP.mdTD[0].A, mpMDP.mdTD[0].P, mpMDP.mdTD[0].C, mpMDP.mdTD[0].IL, mpMDP.mdTD[0].R);/**/
		mpMDP.MaximumParticleIntersection = -0.0001;
		mpMDP.PreMaximumParticleIntersection = -0.02;
		
		mpMDP.mdTD[0].SRD.Distortion=false;
		mpMDP.mdTD[0].SRD.C = 0.0003/length_const; mpMDP.mdTD[0].SRD.M = 0; mpMDP.mdTD[0].SRD.sD = 1.0;
		
		mpMDP.mdTD[0].SaveLammpsCreateSample = false;
		
		if(mpTask.CalculateUTest) {sprintf(filenamepart, "_%i", i); h_CreateCylinderSample(mpMDP, mpMDP.mdTD[0], 0, filenamepart);} //std::cerr << "Fin CS1\n"; //std::cin.get();
		if(mpTask.CalculateBTest) {sprintf(filenamepart, "_%i", i); h_CreateCylinderSample(mpMDP, mpMDP.mdTD[0], 1, filenamepart);} //std::cerr << "Fin CS1 "<<i<<" \n"; std::cin.get();
	}
	
	mpMDP.UVelocity = -2.0 * 0.02 / velocity_const;
	mpMDP.BVelocity = -10.0 * 0.02 / velocity_const;
	
	mpMDP.UMinStress = 10e6 / stress_const; mpMDP.UMaxStrain = 0.03;  mpMDP.UConditionFracture = 0.1; //stress decrease to UConditionFracture from maximum
	mpMDP.BMinStress = 2.1e6 / stress_const; mpMDP.BMaxStrain = 0.04; mpMDP.BConditionFracture = 0.9;//stress decrease to UConditionFracture from maximum
	//std::cin.get();
	std::cerr<<"TestSeveralSamples(): Samples Created\n";
	//char filename[256] = "";
	mpMDP.S[0].SetSpaceSizeSi(SpaceSize[0].x, SpaceSize[0].y, SpaceSize[0].z);
	mpMDP.S[0].SetLSi(LSi[0].x, LSi[0].y, LSi[0].z, mpMDP.mdTD[0].Po.p_r);
	mpMDP.S[0].SetAxis(0.0, 0.0, 1.0);
	mpMDP.S[0].CalculateParameters(mpMDP.mdTD[0].C.a, mpMDP.mdTD[0].Po.p_r);
	mpMDP.S[0].PN = (1.0 - mpTask.InitialPorosityU) * mpMDP.S[0].Vext / mpMDP.mdTD[0].Po.p_V;
	mpMDP.S[1].SetSpaceSizeSi(SpaceSize[1].x, SpaceSize[1].y, SpaceSize[1].z);
	mpMDP.S[1].SetLSi(LSi[1].x, LSi[1].y, LSi[1].z, mpMDP.mdTD[0].Po.p_r);
	mpMDP.S[1].SetAxis(0.0, 0.0, 1.0);
	mpMDP.S[1].CalculateParameters(mpMDP.mdTD[0].C.a, mpMDP.mdTD[0].Po.p_r);
	mpMDP.S[1].PN = (1.0 - mpTask.InitialPorosityB) * mpMDP.S[1].Vext / mpMDP.mdTD[0].Po.p_V;
	initArrays(mpMDP.mdTD[0].A, mpMDP.mdTD[0].P, mpMDP.mdTD[0].C, mpMDP.mdTD[0].IL, mpMDP.mdTD[0].R);/**/
	mpMDP.MaximumParticleIntersection = -0.0001;
	mpMDP.PreMaximumParticleIntersection = -0.02;
	
	
	
	strcpy(filename, "./result/CalculationSamplesResults.dat");
    mpTask.Step = 0;
    mpTask.CalculateStartParameters();	
	for (i = 0; i < mpTask.NSamples; ++i)
	{
		mpTask.Step = i;
		
		if(mpTask.CalculateUTest) 
		{
			sprintf(filenamepart, "_%i", i);
			h_LoadCylinderSample(mpMDP, 0, filenamepart);
			mpMDP.mdTD[0].S = mpMDP.S[0];
			mpMDP.mdTD[0].P.N = mpMDP.PN[0];
			mpMDP.mdTD[0].Compress.VLoad = mpMDP.UVelocity;
			mpMDP.mdTD[0].Compress.CalculateUParameters(mpMDP.mdTD[0].S.size_real, mpMDP.mdTD[0].S.center_real, mpMDP.mdTD[0].Po, mpMDP.UMinStress, mpMDP.UMaxStrain, mpMDP.UConditionFracture);
			//std::cerr << "F1\n"; std::cin.get();
			std::cerr << "Material " << mpMDP.mdTD[0].Po.m_E << " " << mpMDP.mdTD[0].Po.m_Ec << " " << mpMDP.mdTD[0].Po.m_Gc << " " << mpMDP.mdTD[0].Po.b_r << " " << mpMDP.mdTD[0].Po.b_d << " | " << mpMDP.mdTD[0].Po.hm_E << " " << mpMDP.mdTD[0].Po.b_dd << "\n";
			mpMDP.mdTD[0].R.dNsave = mpTask.dNSaveU;
			mpMDP.mdTD[0].R.fileN = i;
			h_CalculationUniaxialCompression(mpMDP.mdTD[0], mpMDP.h_R[0]);
			mpTask.RFunction[mpTask.NResult * i + 0] = mpMDP.mdTD[0].Compress.FractureStrain;
			mpTask.RFunction[mpTask.NResult * i + 1] = mpMDP.mdTD[0].Compress.FractureStress;
			
			std::cerr << "RU " << mpMDP.mdTD[0].Compress.FractureStrain << " " << mpMDP.mdTD[0].Compress.FractureStress * stress_const << "\n"; //std::cin.get();
		}
		//std::cin.get();
		if(mpTask.CalculateBTest)
		{
			sprintf(filenamepart, "_%i", i);
			h_LoadCylinderSample(mpMDP, 1, filenamepart);
			mpMDP.mdTD[0].S = mpMDP.S[1];
			mpMDP.mdTD[0].P.N = mpMDP.PN[1];
			mpMDP.mdTD[0].Compress.VLoad = mpMDP.BVelocity;
			mpMDP.mdTD[0].Compress.CalculateBParameters(mpMDP.mdTD[0].S.size_real, mpMDP.mdTD[0].S.center_real, mpMDP.mdTD[0].Po, mpMDP.BMinStress, mpMDP.BMaxStrain, mpMDP.BConditionFracture);
			//std::cerr << "F1\n"; std::cin.get();
			std::cerr << "Material " << mpMDP.mdTD[0].Po.m_E << " " << mpMDP.mdTD[0].Po.m_Ec << " " << mpMDP.mdTD[0].Po.m_Gc << " " << mpMDP.mdTD[0].Po.b_r << " " << mpMDP.mdTD[0].Po.b_d << " | " << mpMDP.mdTD[0].Po.hm_E << " " << mpMDP.mdTD[0].Po.b_dd << "\n";
			mpMDP.mdTD[0].R.dNsave = mpTask.dNSaveB;
			mpMDP.mdTD[0].R.fileN = i;			
			h_CalculationBrazilTest(mpMDP.mdTD[0], mpMDP.h_R[1]);
			mpTask.RFunction[mpTask.NResult * i + 2] = mpMDP.mdTD[0].Compress.FractureStrain;
			mpTask.RFunction[mpTask.NResult * i + 3] = mpMDP.mdTD[0].Compress.FractureStress;
			std::cerr << "RB " << mpMDP.mdTD[0].Compress.FractureStress * stress_const << "\n";
		}
		
		mpTask.SaveSamplesResults(filename, (i==0)?1:0, i);
	}

    std::cerr << "Fin\n"; //std::cin.get();
    mpTask.deletearrays();
}

