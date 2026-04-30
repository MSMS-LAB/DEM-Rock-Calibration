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

void SetInitialParameters(md_task_data&mdTD, mp_task_data &mpTask)
{
    mdTD.IL.IonP = 32;
    mdTD.Po.m_nu = 0.247;
    mdTD.Po.m_E = mpTask.MaterialParametersSteps[0];
    mdTD.Po.m_Ec = mpTask.MaterialParametersSteps[1];
    mdTD.Po.m_Gc = mpTask.MaterialParametersSteps[2];
    mdTD.Po.m_ro = 2610 / density_const;
    mdTD.Po.b_r = mpTask.MaterialParametersSteps[3];
    mdTD.Po.p_r = 0.5 * 0.001 / length_const;    
    mdTD.Po.b_d = 2.0 * mdTD.Po.p_r + mpTask.MaterialParametersSteps[4];
    mdTD.Po.p_e = 0.6;
    mdTD.Po.p_mu = 0.45;
    mdTD.Po.p_mur = 0.05;
    mdTD.Po.nuV = 1e-5;
    mdTD.Po.nuW = 1e-3;
    mdTD.Po.a_farcut = 3.2 * mdTD.Po.p_r;
    mdTD.C.a = 2.0 * mdTD.Po.a_farcut;    
    mdTD.Po.CalculateParameters();
    std::cerr << "Po " << mdTD.Po.p_r << " " << mdTD.Po.a_farcut << " " << mdTD.C.a << "\n";
    
    //std::cerr << "Material " << "E=" << Po.m_E << " G=" << Po.m_G << " Ecrit=" << Po.m_Ec << " Gcrit=" << Po.m_Gc << " density=" << Po.m_ro << "\n"
    //    << "Particle" << " Radius=" << Po.p_r << " " << " Volume=" << Po.p_V << " Mass=" << Po.p_m << " Inertia=" << Po.p_I << " e=" << Po.p_e << " Alpha=" << Po.p_A
    //    << " HM_E=" << Po.hm_E << " HM_G=" << Po.hm_G << " Friction=" << Po.p_mu << " RollFriction=" << Po.p_mur << "\n"
    //    << "Bound" << " Radius=" << Po.b_r << " " << " Area=" << Po.b_S << " Distance=" << Po.b_d << " Inertia=" << Po.p_I << " e=" << Po.p_e << " Alpha=" << Po.p_A << "\n";

    //std::cerr << "Alpha " << Po.p_A << " " << Po.p_m << " " << Po.p_r << " " << -1.8257 * Po.p_A * sqrt(2.0f * Po.hm_E * 0.5f * Po.p_m) << " " << -2.0 * sqrt(5.0 / 6.0) * Po.p_e * sqrt(2.0f * Po.hm_E * 0.5f * Po.p_m) << "\n";

    //std::cerr << "Test " << (Po.p_r - 2.0f * Po.p_r) * 2.0f * Po.hm_E * sqrt(Po.p_r * fabsf(Po.p_r - 0.5f * Po.p_r)) * MCf_2d3 << " " << Po.b_r * Po.b_r * MCf_pi * Po.m_E * (Po.p_r - 2.0 * Po.p_r) / (2.0 * Po.p_r) << "\n";
    //std::cerr << "Test1 " << Po.m_Ec / MC_pi * Po.b_r * Po.b_r * Po.m_E << "\n";
    //std::cerr << "Phm " << Po.m_E << " " << Po.m_G << " " << Po.hm_E << " " << Po.hm_G << "\n";
    //std::cin.get();
}

bool SetVariationParameters(md_task_data& mdTD, mp_task_data& mpTask, uint iparam)
{
    mdTD.IL.IonP = 32;
    mdTD.Po.m_nu = 0.247;
    mdTD.Po.m_E = mpTask.MaterialParametersSteps[mpTask.Step * mpTask.NParam + 0] + mpTask.ParameterVariation[mpTask.NParam * iparam + 0];
    mdTD.Po.m_Ec = mpTask.MaterialParametersSteps[mpTask.Step * mpTask.NParam + 1] + mpTask.ParameterVariation[mpTask.NParam * iparam + 1];
    mdTD.Po.m_Gc = mpTask.MaterialParametersSteps[mpTask.Step * mpTask.NParam + 2] + mpTask.ParameterVariation[mpTask.NParam * iparam + 2];
    mdTD.Po.m_ro = 2610 / density_const;
    mdTD.Po.b_r = mpTask.MaterialParametersSteps[mpTask.Step * mpTask.NParam + 3] + mpTask.ParameterVariation[mpTask.NParam * iparam + 3];
    mdTD.Po.p_r = 0.5 * 0.001 / length_const;
    mdTD.Po.b_d = 2.0 * mdTD.Po.p_r + mpTask.MaterialParametersSteps[mpTask.Step * mpTask.NParam + 4] + mpTask.ParameterVariation[mpTask.NParam * iparam + 4];
    mdTD.Po.p_e = 0.6;
    mdTD.Po.p_mu = 0.45;
    mdTD.Po.p_mur = 0.05;
    mdTD.Po.nuV = 1e-5;
    mdTD.Po.nuW = 1e-3;
    mdTD.Po.a_farcut = 2.9 * mdTD.Po.p_r;
    mdTD.C.a = 2.0 * mdTD.Po.a_farcut;
    mdTD.Po.CalculateParameters();
    //std::cerr << "Po " << mdTD.Po.p_r << " " << mdTD.Po.a_farcut << " " << mdTD.C.a << "\n";
    if (iparam > 0 && mpTask.ParameterVariation[mpTask.NParam * iparam + 0] + mpTask.ParameterVariation[mpTask.NParam * iparam + 1] + mpTask.ParameterVariation[mpTask.NParam * iparam + 2] + mpTask.ParameterVariation[mpTask.NParam * iparam + 3] + mpTask.ParameterVariation[mpTask.NParam * iparam + 4] < 1e-12)return false;
    return true;
    //std::cerr << "Material " << "E=" << Po.m_E << " G=" << Po.m_G << " Ecrit=" << Po.m_Ec << " Gcrit=" << Po.m_Gc << " density=" << Po.m_ro << "\n"
    //    << "Particle" << " Radius=" << Po.p_r << " " << " Volume=" << Po.p_V << " Mass=" << Po.p_m << " Inertia=" << Po.p_I << " e=" << Po.p_e << " Alpha=" << Po.p_A
    //    << " HM_E=" << Po.hm_E << " HM_G=" << Po.hm_G << " Friction=" << Po.p_mu << " RollFriction=" << Po.p_mur << "\n"
    //    << "Bound" << " Radius=" << Po.b_r << " " << " Area=" << Po.b_S << " Distance=" << Po.b_d << " Inertia=" << Po.p_I << " e=" << Po.p_e << " Alpha=" << Po.p_A << "\n";

    //std::cerr << "Alpha " << Po.p_A << " " << Po.p_m << " " << Po.p_r << " " << -1.8257 * Po.p_A * sqrt(2.0f * Po.hm_E * 0.5f * Po.p_m) << " " << -2.0 * sqrt(5.0 / 6.0) * Po.p_e * sqrt(2.0f * Po.hm_E * 0.5f * Po.p_m) << "\n";

    //std::cerr << "Test " << (Po.p_r - 2.0f * Po.p_r) * 2.0f * Po.hm_E * sqrt(Po.p_r * fabsf(Po.p_r - 0.5f * Po.p_r)) * MCf_2d3 << " " << Po.b_r * Po.b_r * MCf_pi * Po.m_E * (Po.p_r - 2.0 * Po.p_r) / (2.0 * Po.p_r) << "\n";
    //std::cerr << "Test1 " << Po.m_Ec / MC_pi * Po.b_r * Po.b_r * Po.m_E << "\n";
    //std::cerr << "Phm " << Po.m_E << " " << Po.m_G << " " << Po.hm_E << " " << Po.hm_G << "\n";
    //std::cin.get();
}

void CalculateStepCG(mp_mdparameters_data mpMDP, mp_task_data& mpTask)
{
    uint i;
    for (i = 0; i < mpTask.NParam + 1; ++i)
    {
        if (!SetVariationParameters(mpMDP.mdTD[0], mpTask, i) || !mpTask.CalculateUTest)
        {
            mpTask.RFunction[mpTask.NResult * i + 0] = 0;
            mpTask.RFunction[mpTask.NResult * i + 1] = 0;
            std::cerr << "SkipU! " << i << "\n";
            continue;
        }
        mpMDP.mdTD[0].S = mpMDP.S[0];
        mpMDP.mdTD[0].P.N = mpMDP.PN[0];
        mpMDP.mdTD[0].Compress.VLoad = mpMDP.UVelocity;
        mpMDP.mdTD[0].Compress.CalculateUParameters(mpMDP.mdTD[0].S.size_real, mpMDP.mdTD[0].S.center_real, mpMDP.mdTD[0].Po, mpMDP.UMinStress, mpMDP.UMaxStrain, mpMDP.UConditionFracture);
        //std::cerr << "F1\n"; std::cin.get();
		std::cerr << "Material " << mpMDP.mdTD[0].Po.m_E << " " << mpMDP.mdTD[0].Po.m_Ec << " " << mpMDP.mdTD[0].Po.m_Gc << " " << mpMDP.mdTD[0].Po.b_r << " " << mpMDP.mdTD[0].Po.b_d << " | " << mpMDP.mdTD[0].Po.hm_E << " " << mpMDP.mdTD[0].Po.b_dd << "\n";
		mpMDP.mdTD[0].R.dNsave = mpTask.dNSaveU;
		mpMDP.mdTD[0].R.fileN = 0;
        h_CalculationUniaxialCompression(mpMDP.mdTD[0], mpMDP.h_R[0]);
        mpTask.RFunction[mpTask.NResult * i + 0] = mpMDP.mdTD[0].Compress.FractureStrain;
        mpTask.RFunction[mpTask.NResult * i + 1] = mpMDP.mdTD[0].Compress.FractureStress;
		mpTask.RFunction[mpTask.NResult * i + 3] = mpMDP.mdTD[0].Compress.LinearityDeviation;
        //mpTask.RFunction[mpTask.NResult * i + 2] = 0.1 * mpMDP.mdTD[0].Compress.FractureStress;//TEST! ERROR!
        std::cerr << "RU " << mpMDP.mdTD[0].Compress.FractureStrain << " " << mpMDP.mdTD[0].Compress.FractureStress * stress_const << "\n"; //exit(0); std::cin.get(); 
    }/**/
    //std::cin.get();
    for (i = 0; i < mpTask.NParam + 1; ++i)
    {
        if (!SetVariationParameters(mpMDP.mdTD[0], mpTask, i)  || !mpTask.CalculateBTest)
        {
            mpTask.RFunction[mpTask.NResult * i + 2] = 1.0;
            std::cerr << "SkipB! " << i << "\n";
            continue;
        }
        mpMDP.mdTD[0].S = mpMDP.S[1];
        mpMDP.mdTD[0].P.N = mpMDP.PN[1];
        mpMDP.mdTD[0].Compress.VLoad = mpMDP.BVelocity;
        mpMDP.mdTD[0].Compress.CalculateBParameters(mpMDP.mdTD[0].S.size_real, mpMDP.mdTD[0].S.center_real, mpMDP.mdTD[0].Po, mpMDP.BMinStress, mpMDP.BMaxStrain, mpMDP.BConditionFracture);
        //std::cerr << "F1\n"; std::cin.get();
		mpMDP.mdTD[0].R.dNsave = mpTask.dNSaveB;
		mpMDP.mdTD[0].R.fileN = 0;
        h_CalculationBrazilTest(mpMDP.mdTD[0], mpMDP.h_R[1]);
        mpTask.RFunction[mpTask.NResult * i + 2] = mpMDP.mdTD[0].Compress.FractureStress;
        std::cerr << "RB " << mpMDP.mdTD[0].Compress.FractureStress * stress_const << "\n"; //std::cin.get();
    }/**/
    //SetVariationParameters(mpMDP.mdTD[0], mpTask, 0);
    std::cerr << "Material " << mpMDP.mdTD[0].Po.m_E << " " << mpMDP.mdTD[0].Po.m_Ec << " " << mpMDP.mdTD[0].Po.m_Gc << " " << mpMDP.mdTD[0].Po.b_r << " " << mpMDP.mdTD[0].Po.b_d << " | " << mpMDP.mdTD[0].Po.hm_E << " " << mpMDP.mdTD[0].Po.b_dd << "\n";
    for (i = 0; i < mpTask.NParam + 1; ++i)
    {
        //mpTask.RFunction[mpTask.NResult * i + 0] = mpTask.ResultsGoal[0] * (1 - 0.05 * rand() * _1d_RAND_MAX_double);
        //mpTask.RFunction[mpTask.NResult * i + 1] = mpTask.ResultsGoal[1] * (1 - 0.05 * rand() * _1d_RAND_MAX_double);
        //mpTask.RFunction[mpTask.NResult * i + 2] = mpTask.ResultsGoal[2] * (1 - 0.05 * rand() * _1d_RAND_MAX_double);
        std::cerr << "R " << i << " " << mpTask.RFunction[mpTask.NResult * i + 0] << " " << mpTask.RFunction[mpTask.NResult * i + 1] << " " << mpTask.RFunction[mpTask.NResult * i + 2] << "\n";
    }
    
    //std::cin.get();
}


void DetermineConjGradient()
{
    mp_task_data mpTask;
    mpTask.NSteps = 50;
    mpTask.NParam = 5;
    mpTask.NResult = 4;
    mpTask.MaterialDeltaFactor = 0.1;
    mpTask.createarrays();
    mpTask.ResultsGoal[0] = 0.00480002;//Granite
    mpTask.ResultsGoal[1] = 104.0497482e6 / stress_const;
    mpTask.ResultsGoal[2] = 10.5e6 / stress_const;
	mpTask.ResultsGoal[3] = 1.0;
    mpTask.MaterialParametersSteps[0] = 1.95255e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 5.1447e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 3.7131e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.5 * 0.000330015 / length_const;
    mpTask.MaterialParametersSteps[4] = 5.93169e-05 / length_const;
  
    /*mpTask.MaterialParametersSteps[0] = 1.99883e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 2.79836e+11 / stress_const;
    mpTask.MaterialParametersSteps[2] = 5.30608e+08  / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.0011395 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00192661 / length_const;/**/
	
	/*mpTask.MaterialParametersSteps[0] = 2.01146e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 2.81604e+11 / stress_const;
    mpTask.MaterialParametersSteps[2] = 1e+08  / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.0016149 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00192807 / length_const;/**/
	
	/*mpTask.MaterialParametersSteps[0] = 1.97257e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 2.7616e+11 / stress_const;
    mpTask.MaterialParametersSteps[2] = 2.80966e+09  / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000257025 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00129587 / length_const;/**/  
	
	/*mpTask.MaterialParametersSteps[0] = 1.95628e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 2.73879e+11 / stress_const;
    mpTask.MaterialParametersSteps[2] = 1.22354e+09  / stress_const;
    mpTask.MaterialParametersSteps[3] = 5.21357e-05 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000552088 / length_const;/**/  
	
	/*mpTask.MaterialParametersSteps[0] = 1.95959e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 2.74343e+11 / stress_const;
    mpTask.MaterialParametersSteps[2] = 1.15813e+09 / stress_const;
    mpTask.MaterialParametersSteps[3] = 1e-06 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000551787  / length_const;/**/  
	
	/*mpTask.MaterialParametersSteps[0] = 1.95394e+11 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 2.04051e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 2.04051e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000244244 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000105119  / length_const;/**/
	
	/*mpTask.MaterialParametersSteps[0] = 9.51145e+10 / stress_const;//
    mpTask.MaterialParametersSteps[1] = 1.96948e+09 / stress_const;
    mpTask.MaterialParametersSteps[2] = 2.00935e+09 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000195753 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00027099 / length_const;/**/
	
	/*mpTask.MaterialParametersSteps[0] = 1.91066e+10 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 5.3021e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 4.55066e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000496129 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000351215 / length_const;/**/
	
	/*mpTask.MaterialParametersSteps[0] = 1.71066e+10 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 5.51498e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 5.13888e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000407446 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00046686 / length_const;/**/
	
	/*mpTask.MaterialParametersSteps[0] = 1.95255e+11 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 5.02234e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 5.27279e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000347366 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000174706 / length_const;/**/
	
	mpTask.MaterialParametersSteps[0] = 2.55255e+11 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 0.10234e+10 / stress_const;
    mpTask.MaterialParametersSteps[2] = 0.15279e+10 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000347366 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000174706 / length_const;/**/
	
	mpTask.MaterialParametersSteps[0] = 2.55255e+11 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 4*1.08958e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 4*1.54683e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000362243 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.4*0.000292946 / length_const;/**/
	
	mpTask.MaterialParametersSteps[0] = 0.7*2.55255e+11 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 2*8*1.08958e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 2*8*1.54683e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.4*0.000362243 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000292946 / length_const;/**/
	
	mpTask.MaterialParametersSteps[0] = 1.78678e+11 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 1.65684e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 1.85246e+09 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.000560159 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.000292946  / length_const;/**/
	
	mpTask.MaterialParametersSteps[0] = 0.8*1.71066e+10 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 0.15*6.18389e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 0.15*5.03332e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 2*0.000409345 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00046797  / length_const;/**/
	
	/*mpTask.MaterialParametersSteps[0] = 0.2*2.0528e+10 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 10*1.17714e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 10*1.07324e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 0.00404331 / length_const;
    mpTask.MaterialParametersSteps[4] = 10*0.00046797  / length_const;/**/
	
	mpTask.MaterialParametersSteps[0] = 0.45*1.71066e+10 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 1.0*6.18389e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 1.0*5.03332e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 4*0.000409345 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00046797  / length_const;/**/
	
	mpTask.FunctionalParametersWeightMultiplier[0] = 50; // WeightMultiplier Uniaxial strain
	mpTask.FunctionalParametersWeightMultiplier[1] = 100;// WeightMultiplier Uniaxial stress
	mpTask.FunctionalParametersWeightMultiplier[2] = 1;// WeightMultiplier Brazilian stress
	mpTask.FunctionalParametersWeightMultiplier[3] = 10000;// WeightMultiplier Uniaxial LinearityDeviation
	
	mpTask.CalculateUTest = true;
	mpTask.CalculateBTest = false;
	mpTask.ParameterVariationType = 7;//0 - stress parameters; 1 - bond parameters; 2 - all parameters; 3 - all w/o Young's modulus; 
			//4 - Shear max stress, bond parameters; 5 - Young's modulus, bond parameters; 6 - Young's modulus; 7 - Critical streses, bond radii;
	bool l_ChangeParameterVariationType = false;
	uint32_t l_NParameterVariationType = 2;
	uint32_t l_ArrayParameterVariationType[l_NParameterVariationType] = {0,1};
	mpTask.InitialPorosityU = 0.375;//0.392//0.375
	mpTask.InitialPorosityB = 0.37;
	
	       
    //std::cerr << "q1\n"; std::cin.get();
    mp_mdparameters_data mpMDP;
    SetInitialParameters(mpMDP.mdTD[0], mpTask);
    //std::cerr << "q3\n"; mpTask.CalculateNewParameters();
    //std::cerr << "q4\n"; std::cin.get();
    double sc = 1.0;
    float3 LSi[2] = { {sc * 0.025, sc * 0.025, sc * 0.05},{sc * 0.05, sc * 0.05, sc * 0.02} }, SpaceSize[2] = { {0.04, 0.04, 0.06}, {0.065, 0.065, 0.03} };

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
	mpMDP.mdTD[0].SaveLammpsCreateSample = false;
	mpMDP.MaximumParticleIntersection = -0.001;//-0.0001
	mpMDP.PreMaximumParticleIntersection = -0.1;//-0.02
	
	mpMDP.mdTD[0].SRD.Distortion=true;
	mpMDP.mdTD[0].SRD.C = 0.0005/length_const; mpMDP.mdTD[0].SRD.M = 0; mpMDP.mdTD[0].SRD.sD = 1.0;
	
	mpMDP.UVelocity = -5.0 * 0.02 / velocity_const;
	mpMDP.BVelocity = -5.0 * 0.02 / velocity_const;
	
	mpMDP.UMinStress = 10e6 / stress_const; mpMDP.UMaxStrain = 0.025;  mpMDP.UConditionFracture = 0.2; //stress decrease to UConditionFracture from maximum
	mpMDP.BMinStress = 2.1e6 / stress_const; mpMDP.BMaxStrain = 0.04; mpMDP.BConditionFracture = 0.8;//stress decrease to UConditionFracture from maximum
	
	mpTask.dNSaveU = 100;
	mpTask.dNSaveB = 1000;
    //std::cin.get();
    char filename[256] = "";
    sprintf(filename, "_small");
    if(mpTask.CalculateUTest) h_CreateCylinderSample(mpMDP, mpMDP.mdTD[0], 0); std::cerr << "Fin CS1\n"; //std::cin.get();
    if(mpTask.CalculateBTest) h_CreateCylinderSample(mpMDP, mpMDP.mdTD[0], 1); std::cerr << "Fin CS2\n"; //std::cin.get();/**/
	
    //std::cin.get();
    //std::cerr << "HP " << mpMDP.S[0].hidenpoint.x << " " << mpMDP.S[0].hidenpoint.y << " " << mpMDP.S[0].hidenpoint.z << "\n"; std::cin.get();

    h_LoadCylinderSample(mpMDP, 0);
    h_LoadCylinderSample(mpMDP, 1);
    //std::cerr << "HP " << mpMDP.S[0].hidenpoint.x << " " << mpMDP.S[0].hidenpoint.y << " " << mpMDP.S[0].hidenpoint.z << "\n"; std::cin.get();

    std::cerr << "Load\n"; std::cerr << mpMDP.h_R[0] <<" " << mpMDP.PN[0] << " " << mpMDP.h_R[1] << " " << mpMDP.PN[1] << "\n";
    //std::cin.get();
    strcpy(filename, "./result/CalculationResults.dat");
    uint step=0;
    mpTask.Step = step;
    mpTask.CalculateStartParameters();
    for (step = 0; step < mpTask.NSteps; ++step)
    {
		if(l_ChangeParameterVariationType) mpTask.ParameterVariationType = l_ArrayParameterVariationType[step%l_NParameterVariationType];
        mpTask.Step = step;
        CalculateStepCG(mpMDP, mpTask);
        //std::cerr << "save start\n";        
        mpTask.CalculateNewParameters2();
		mpTask.SaveResults(filename);
        std::cerr << "StepDG " << step << "\n"; 
		
		
		//std::cin.get();
        //if(step>3)std::cin.get();

    }
    std::cerr << "Fin\n"; //std::cin.get();
    mpTask.deletearrays();
}

