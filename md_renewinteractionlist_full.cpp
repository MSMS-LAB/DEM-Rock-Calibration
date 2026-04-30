
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include "md_data_types.h"
#include "md.h"

void RenewInteractionList_full(particle_data &P, cell_data &C, sample_data &S, additional_data &A, interaction_list_data &IL)
{
    CellDistribution(P, A, S, C);
    //std::cerr << "Cell list\n";
    //cudaMemcpy(C.h_CI, C.d_CI, P.N * sizeof(uint), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < P.N; ++i)  std::cerr << i << " "<< C.h_CI[i] << "\n"; std::cin.get();
    //std::cerr << "Cells list\n";
    //cudaMemcpy(C.h_CIs, C.d_CIs, 2*P.N * sizeof(uint), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < P.N; ++i) std::cerr << C.h_CIs[i] << " " << C.h_CIs[i+P.N] << "\n"; std::cin.get();
    //std::cerr << "CellpnC list "<< C.N<<"\n";
    //cudaMemcpy(C.h_pnC, C.d_pnC, 2*C.N * sizeof(uint), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < C.N; ++i) std::cerr<<i<<" " << C.h_pnC[i] << " " << C.h_pnC[i + C.N] << "\n"; std::cin.get();
    InteractionListConstruct(P, A, S, IL, C);
    /*std::cerr << "Interaction list " << IL.N << "\n";
    cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < IL.N; ++i)  std::cerr << i / IL.IonP << " "<< IL.h_IL[i] << "\n"; std::cin.get();
    for (int i = 0; i < IL.N; ++i)
    {
        if (i % IL.IonP == 0)std::cerr << "\n"<< i / IL.IonP;
        if (IL.h_IL[i] < UINT_MAX)
            std::cerr << " " << IL.h_IL[i];
        if (IL.h_IL[i] == UINT_MAX)
            std::cerr << " M";
    }
    std::cin.get();/**/
}

void RenewInteractionList_New(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL)
{
    CellDistribution(P, A, S, C);
    InteractionListReConstruct(P, A, S, IL, C);
    /*std::cerr << "Interaction list " << IL.N << "\n";
    cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < IL.N; ++i)  std::cerr << i / IL.IonP << " "<< IL.h_IL[i] << "\n"; std::cin.get();
    for (int i = 0; i < IL.N; ++i)
    {
        if (i % IL.IonP == 0)std::cerr << "\n" << i / IL.IonP;
        if (IL.h_IL[i] < UINT_MAX)
            std::cerr << " " << IL.h_IL[i];
        if (IL.h_IL[i] == UINT_MAX)
            std::cerr << " M";
    }
    std::cin.get();/**/
}

void RenewInteractionList_BPM(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL)
{
    CellDistribution(P, A, S, C);
    InteractionListReConstructBPM(P, A, S, IL, C);
    /*std::cerr << "Interaction list " << IL.N << "\n";
    cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < IL.N; ++i)  std::cerr << i / IL.IonP << " "<< IL.h_IL[i] << "\n"; std::cin.get();
    for (int i = 0; i < IL.N; ++i)
    {
        if (i % IL.IonP == 0)std::cerr << "\n" << i / IL.IonP;
        if (IL.h_IL[i] < UINT_MAX)
            std::cerr << " " << IL.h_IL[i];
        if (IL.h_IL[i] == UINT_MAX)
            std::cerr << " M";
    }
    std::cin.get();/**/
}
