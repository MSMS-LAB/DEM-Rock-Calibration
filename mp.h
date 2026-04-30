#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "md_data_types.h"
#include "mp_task.h"

void CalculateOneSample(const std::string &p_MaterialString, const bool &p_CreateSample, const bool &p_CalculateUniaxial, const bool &p_CalculateBrazil, 
const bool &p_ForceUSampleCreation, const bool &p_ForceBSampleCreation);
void DetermineConjGradient();
void TestSeveralSamples();

void SetInitialParameters(md_task_data&mdTD, mp_task_data &mpTask);
bool SetVariationParameters(md_task_data& mdTD, mp_task_data& mpTask, uint iparam);
void CalculateStepCG(mp_mdparameters_data mpMDP, mp_task_data& mpTask);

void h_CreateCylinderSample(mp_mdparameters_data& mpMDP, md_task_data& mdTD, uint isample, char namepart[] = "");
void h_LoadCylinderSample(mp_mdparameters_data& mpMDP, uint isample, char namepart[] = "");
void h_LoadCylinderSampleMusen(mp_mdparameters_data& mpMDP, uint isample, char namepart[]);
void h_LoadCylinderSample_f(mp_mdparameters_data& mpMDP, uint isample, char* filename);

void h_CalculationUniaxialCompression(md_task_data& mdTD, float* h_R);
void h_CalculationBrazilTest(md_task_data& mdTD, float* h_R);
