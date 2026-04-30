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
#include <string>
#include <ctime> 

void CalculateOneSample(const std::string &p_MaterialString, const bool &p_CreateSample, 
const bool &p_CalculateUniaxial, const bool &p_CalculateBrazil, const bool &p_ForceUSampleCreation, const bool &p_ForceBSampleCreation)
{
	std::cerr<<"Start CalculateOneSample\n";
	uint l_i=0, l_MS_start=0, l_MS_finish;
	double l_MaterialParameters[6];
	std::string l_MaterialParameter;
    mp_task_data mpTask;
	mpTask.NSteps = 1;	
    mpTask.NSamples = 1;
	mpTask.NParam = 6;
	mpTask.NResult = 9;	
    mpTask.createarrays();
	mpTask.ResultsGoal[0] = 0.00480002;//Granite
    mpTask.ResultsGoal[1] = 104.0497482e6 / stress_const;
    mpTask.ResultsGoal[2] = 10.5e6 / stress_const;
	mpTask.ResultsGoal[3] = 1.0;
	//std::cerr<<"RG "<<mpTask.ResultsGoal[2]<<"\n";
	l_MS_start = p_MaterialString.find("(");
	std::cerr<<"MP0 |"<<p_MaterialString<<"\n";
	for(l_i=0; l_i<6; ++l_i)
	{
		l_MS_finish = p_MaterialString.find_first_of(",)", l_MS_start+1);
		l_MaterialParameter = p_MaterialString.substr(l_MS_start+1, l_MS_finish-l_MS_start-1);
		//std::cerr<<"MP "<<l_MaterialParameter<<" | "<<l_MS_start<<" "<<l_MS_finish<<"\n";
		l_MaterialParameters[l_i] = stod(l_MaterialParameter);
		l_MS_start = l_MS_finish;		
	}
	
	
	mpTask.MaterialParametersSteps[0] = l_MaterialParameters[0] / stress_const;
    mpTask.MaterialParametersSteps[1] = l_MaterialParameters[1] / stress_const;
    mpTask.MaterialParametersSteps[2] = l_MaterialParameters[2] / stress_const;
    mpTask.MaterialParametersSteps[3] = l_MaterialParameters[3] / length_const;
    mpTask.MaterialParametersSteps[4] = l_MaterialParameters[4] / length_const;
	mpTask.MaterialParametersSteps[5] = l_MaterialParameters[5];
	//std::cerr<<"AAA3\n";
	std::cerr<<"Material parameters "<<mpTask.MaterialParametersSteps[0]<<" "<<mpTask.MaterialParametersSteps[1]<<" "
	<<mpTask.MaterialParametersSteps[2]<<" "<<mpTask.MaterialParametersSteps[3]<<" "
	<<mpTask.MaterialParametersSteps[4]<<" "<<mpTask.MaterialParametersSteps[5]<<"\n";
	
	mpTask.FunctionalParametersWeightMultiplier[0] = 50; // WeightMultiplier Uniaxial strain
	mpTask.FunctionalParametersWeightMultiplier[1] = 100;// WeightMultiplier Uniaxial stress
	mpTask.FunctionalParametersWeightMultiplier[2] = 1;// WeightMultiplier Brazilian stress
	mpTask.FunctionalParametersWeightMultiplier[3] = 10000;// WeightMultiplier Uniaxial LinearityDeviation
	
	/*mpTask.MaterialParametersSteps[0] = 0.45*1.71066e+10 / stress_const;//1.91066e+10
    mpTask.MaterialParametersSteps[1] = 1.0*6.18389e+08 / stress_const;
    mpTask.MaterialParametersSteps[2] = 1.0*5.03332e+08 / stress_const;
    mpTask.MaterialParametersSteps[3] = 4*0.000409345 / length_const;
    mpTask.MaterialParametersSteps[4] = 0.00046797  / length_const;/**/
	
	//exit(0);
	std::cerr<<"AAA2\n";
	mpTask.CalculateUTest = p_CalculateUniaxial;
	mpTask.CalculateBTest = p_CalculateBrazil;
	//bool l_CreateSamples = true;
	
	mpTask.InitialPorosityU = mpTask.MaterialParametersSteps[5];//0.375;
	mpTask.InitialPorosityB = mpTask.MaterialParametersSteps[5];//0.375;
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

	uint i = 0;
    //std::cerr<<"RG2 "<<mpTask.ResultsGoal[2]<<"\n";
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
	

	mpMDP.mdTD[0].R.Porosity = mpTask.MaterialParametersSteps[5];
	if((mpTask.CalculateUTest && p_CreateSample) || p_ForceUSampleCreation) {sprintf(filenamepart, "_%i", i); h_CreateCylinderSample(mpMDP, mpMDP.mdTD[0], 0, filenamepart);} //std::cerr << "Fin CS1\n"; //std::cin.get();
	double UPorosity = mpMDP.mdTD[0].R.Porosity;
	//exit(0);
	
	mpMDP.UVelocity = -5.0 * 0.02 / velocity_const;
	mpMDP.BVelocity = -5.0 * 0.02 / velocity_const;
	
	mpMDP.UMinStress = 10e6 / stress_const; mpMDP.UMaxStrain = 0.01;  mpMDP.UConditionFracture = 0.1; //stress decrease to UConditionFracture from maximum
	mpMDP.BMinStress = 2.1e6 / stress_const; mpMDP.BMaxStrain = 0.03; mpMDP.BConditionFracture = 0.1;//stress decrease to UConditionFracture from maximum
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
	
	auto l_starttime = std::chrono::system_clock::now();
    //temp3
	//std::cerr<<"AAA1\n";
	strcpy(filename, "./result/CalculationSamplesResults.dat");
    mpTask.Step = 0;
    mpTask.CalculateStartParameters();	
	mpTask.Step = i;
	//std::cerr<<"RG3 "<<mpTask.ResultsGoal[2]<<"\n";
	//mpTask.CalculateUTest = false;
	if(mpTask.CalculateUTest) 
	{
		sprintf(filenamepart, "_%i", i);
		h_LoadCylinderSample(mpMDP, 0, filenamepart);
		//h_LoadCylinderSampleMusen(mpMDP, 0, filenamepart);
		mpMDP.mdTD[0].S = mpMDP.S[0];
		mpMDP.mdTD[0].P.N = mpMDP.PN[0];
		mpMDP.mdTD[0].Compress.VLoad = mpMDP.UVelocity;
		mpMDP.mdTD[0].Compress.CalculateUParameters(mpMDP.mdTD[0].S.size_real, mpMDP.mdTD[0].S.center_real, mpMDP.mdTD[0].Po, mpMDP.UMinStress, mpMDP.UMaxStrain, mpMDP.UConditionFracture);
		//std::cerr << "F1\n"; std::cin.get();
		std::cerr << "Material " << mpMDP.mdTD[0].Po.m_E << " " << mpMDP.mdTD[0].Po.m_Ec << " " << mpMDP.mdTD[0].Po.m_Gc << " " << mpMDP.mdTD[0].Po.b_r << " " << mpMDP.mdTD[0].Po.b_d << " | " << mpMDP.mdTD[0].Po.hm_E << " " << mpMDP.mdTD[0].Po.b_dd << "\n";
		mpMDP.mdTD[0].R.dNsave = mpTask.dNSaveU;
		mpMDP.mdTD[0].R.fileN = i;
		h_CalculationUniaxialCompression(mpMDP.mdTD[0], mpMDP.h_R[0]);
		mpTask.RFunction[0] = mpMDP.mdTD[0].Compress.FractureStrain;
		mpTask.RFunction[1] = mpMDP.mdTD[0].Compress.FractureStress;
		mpTask.RFunction[2] = mpMDP.mdTD[0].Compress.LinearityDeviation;
		mpTask.RFunction[3] = mpMDP.mdTD[0].R.Destruction;		
		std::cerr << "RU " << mpMDP.mdTD[0].Compress.FractureStrain << " " << mpMDP.mdTD[0].Compress.FractureStress * stress_const << "\n"; //std::cin.get();
	} else
	{
		mpTask.RFunction[0] = 1e-2;
		mpTask.RFunction[1] = 1e6 / stress_const;
		mpTask.RFunction[2] = 1.0;
		mpTask.RFunction[3] = 1.0;
	}
	double l_DF; //l_FunctionalParametersWeight[4] = {50.0/(mpTask.ResultsGoal[0]*mpTask.ResultsGoal[0]), 
	//100.0/(mpTask.ResultsGoal[1]*mpTask.ResultsGoal[1]), 1.0/(mpTask.ResultsGoal[2]*mpTask.ResultsGoal[2]),
	//10000.0/(mpTask.ResultsGoal[3]*mpTask.ResultsGoal[3])}, 
	
	mpTask.RFunction[8] = 0;
	l_DF = mpTask.ResultsGoal[0] - mpTask.RFunction[0]; mpTask.RFunction[8] += mpTask.FunctionalParametersWeight[0]*l_DF*l_DF;
	l_DF = mpTask.ResultsGoal[1] - mpTask.RFunction[1]; mpTask.RFunction[8] += mpTask.FunctionalParametersWeight[1]*l_DF*l_DF;
	l_DF = mpTask.ResultsGoal[3] - mpTask.RFunction[2]; mpTask.RFunction[8] += mpTask.FunctionalParametersWeight[3]*l_DF*l_DF;
	l_DF = (1.0 - mpTask.RFunction[3]) < 0.15 ? 1000 : 0; mpTask.RFunction[8] += l_DF;
	std::cerr<<"ER "<<mpTask.RFunction[8]<<'\n';
	if(mpTask.RFunction[8]>15 || !p_CalculateBrazil) mpTask.CalculateBTest = false; //mpTask.CalculateBTest = true;
	if(p_CalculateBrazil) mpTask.CalculateBTest = true;
	
	mpTask.MaterialParametersSteps[5] = (UPorosity > mpTask.MaterialParametersSteps[5]) ? UPorosity : mpTask.MaterialParametersSteps[5];
	mpMDP.mdTD[0].R.Porosity = mpTask.MaterialParametersSteps[5];
	//mpTask.CalculateBTest = false;
	if((mpTask.CalculateBTest && p_CreateSample) || p_ForceBSampleCreation) {sprintf(filenamepart, "_%i", i); h_CreateCylinderSample(mpMDP, mpMDP.mdTD[0], 1, filenamepart); } //std::cerr << "Fin CS1 "<<i<<" \n"; std::cin.get();	
	double BPorosity = mpMDP.mdTD[0].R.Porosity;	
	mpTask.MaterialParametersSteps[5] = (UPorosity > BPorosity) ? UPorosity : BPorosity;
	//std::cerr<<"AAA0\n";
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
		mpTask.RFunction[4] = mpMDP.mdTD[0].Compress.FractureStrain;
		mpTask.RFunction[5] = mpMDP.mdTD[0].Compress.FractureStress;
		mpTask.RFunction[7] = mpMDP.mdTD[0].R.Destruction;
		std::cerr << "RB " << mpMDP.mdTD[0].Compress.FractureStress * stress_const << "\n";
	} else
	{
		mpTask.RFunction[4] = 1e-2;
		mpTask.RFunction[5] = 1e6 / stress_const;
		mpTask.RFunction[7] = 1.0;
	}	
	
	mpTask.RFunction[6] = 0;
	std::cerr<<"Delta ";
	l_DF = mpTask.ResultsGoal[0] - mpTask.RFunction[0]; mpTask.RFunction[8] += mpTask.FunctionalParametersWeight[0]*l_DF*l_DF;
	std::cerr<<"| "<<l_DF<<" "<<mpTask.FunctionalParametersWeight[0]<<" "<<mpTask.ResultsGoal[0]<<" ";
	l_DF = mpTask.ResultsGoal[1] - mpTask.RFunction[1]; mpTask.RFunction[8] += mpTask.FunctionalParametersWeight[1]*l_DF*l_DF;
	std::cerr<<"| "<<l_DF<<" "<<mpTask.FunctionalParametersWeight[1]<<" "<<mpTask.ResultsGoal[1]<<" ";
	l_DF = mpTask.ResultsGoal[3] - mpTask.RFunction[2]; mpTask.RFunction[8] += mpTask.FunctionalParametersWeight[3]*l_DF*l_DF;
	std::cerr<<"| "<<l_DF<<" "<<mpTask.FunctionalParametersWeight[3]<<" "<<mpTask.ResultsGoal[3]<<" ";
	l_DF = mpTask.ResultsGoal[2] - mpTask.RFunction[5]; mpTask.RFunction[8] += mpTask.FunctionalParametersWeight[2]*l_DF*l_DF;
	std::cerr<<"| "<<l_DF<<" "<<mpTask.FunctionalParametersWeight[2]<<" "<<mpTask.ResultsGoal[2]<<"\n";
	std::cout << "ResultCalculateOneSample["<<mpTask.MaterialParametersSteps[0]*stress_const<<","
	<<mpTask.MaterialParametersSteps[1]*stress_const<<","<<mpTask.MaterialParametersSteps[2]*stress_const<<","
	<<mpTask.MaterialParametersSteps[3]*length_const<<","<<mpTask.MaterialParametersSteps[4]*length_const<<","<<mpTask.MaterialParametersSteps[5]<<","
	<<mpTask.RFunction[0]<<","<<mpTask.RFunction[1]*stress_const<<","<<mpTask.RFunction[2]<<","<<mpTask.RFunction[3]<<","
	<<mpTask.RFunction[4]<<","<<mpTask.RFunction[5]*stress_const<<","<<mpTask.RFunction[6]<<","<<mpTask.RFunction[7]<<"]\n";
	
	//DFunctional[j] = ResultsGoal[j] - RFunction[j];
	//Functional[j] = FunctionalParametersWeight[j] * DFunctional[j] * DFunctional[j];
	
	//mpTask.SaveSamplesResults(filename, (i==0)?1:0, i);
	auto l_endtime = std::chrono::system_clock::now();
	std::time_t l_start_time = std::chrono::system_clock::to_time_t(l_starttime);
	std::time_t l_end_time = std::chrono::system_clock::to_time_t(l_endtime);
	
	std::cout<<"Time "<<std::ctime(&l_start_time)<<" | "<<std::ctime(&l_end_time)<<"\n";

    std::cerr << "Fin\n"; //std::cin.get();
    mpTask.deletearrays();
}

