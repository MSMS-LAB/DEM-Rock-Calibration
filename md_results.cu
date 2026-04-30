#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <math.h>

void ResultsUInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C, interaction_list_data& IL, result_data &R, compression_data& Compress)
{	
	R.N = 2 * A.bloks;
	R.bloks = A.bloks;	
	R.stepsave = 0;
	R.sz0 = Compress.size.z;
	HANDLE_ERROR(cudaMalloc((void**)&R.d_FL, R.N * sizeof(float)));
	R.h_FL = (float*)malloc(R.N * sizeof(float));
	R.h_sFL = (double*)malloc(3 * R.Nsave * sizeof(double));
	HANDLE_ERROR(cudaMalloc((void**)&R.d_SI, IL.IonP*A.bloks * sizeof(uint)));
	R.h_SI = (uint*)malloc(IL.IonP*A.bloks * sizeof(uint));
	R.h_sSI = (uint*)malloc(3 * R.Nsave * sizeof(uint));
	Compress.FractureStress = 0;
}

void ResultsBInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C, interaction_list_data& IL, result_data &R, compression_data& Compress)
{	
	R.N = 2 * A.bloks;
	R.bloks = A.bloks;	
	R.stepsave = 0;
	R.sz0 = Compress.size.y;
	HANDLE_ERROR(cudaMalloc((void**)&R.d_FL, R.N * sizeof(float)));
	R.h_FL = (float*)malloc(R.N * sizeof(float));
	R.h_sFL = (double*)malloc(3 * R.Nsave * sizeof(double));
	HANDLE_ERROR(cudaMalloc((void**)&R.d_SI, IL.IonP*A.bloks * sizeof(uint)));
	R.h_SI = (uint*)malloc(IL.IonP*A.bloks * sizeof(uint));
	R.h_sSI = (uint*)malloc(3 * R.Nsave * sizeof(uint));
	Compress.FractureStress = 0;
}

void ResultsUDelete(result_data& R)
{
	if (R.d_FL != nullptr) { cudaFree(R.d_FL); R.d_FL = nullptr; }
	if (R.h_FL != nullptr) { free(R.h_FL); R.h_FL = nullptr; }
	if (R.h_sFL != nullptr) { free(R.h_sFL); R.h_sFL = nullptr; }
	if (R.d_SI != nullptr) { cudaFree(R.d_SI); R.d_SI = nullptr; }
	if (R.h_SI != nullptr) { free(R.h_SI); R.h_SI = nullptr; }
	if (R.h_sSI != nullptr) { free(R.h_sSI); R.h_sSI = nullptr; }
}

void SumForcesULoading(result_data& R, compression_data & Compress, uint n)
{
	HANDLE_ERROR(cudaMemcpy(R.h_FL, R.d_FL, R.N * sizeof(float), cudaMemcpyDeviceToHost));
	uint ir;
	R.sFL[0] = 0;	
	for (ir = 0; ir < R.bloks; ++ir)
	{
		R.sFL[0] += R.h_FL[ir];
	}
	R.sFL[1] = 0;
	for (ir = R.bloks; ir < 2*R.bloks; ++ir)
	{
		R.sFL[1] += R.h_FL[ir];
	}
	if (n % R.dNsave == 0)
	{
		R.h_sFL[3 * R.stepsave] = (Compress.size_d.z - R.sz0) / R.sz0;		
		R.h_sFL[3 * R.stepsave + 1] = R.sFL[0] * Compress._1d_Area;
		R.h_sFL[3 * R.stepsave + 2] = R.sFL[1] * Compress._1d_Area;
		//std::cerr<<"SUF "<<R.stepsave<< " "<< R.h_sFL[3 * R.stepsave] << " " << R.h_sFL[3 * R.stepsave + 1] * stress_const << " " << R.h_sFL[3 * R.stepsave + 2] * stress_const << "\n"; //std::cin.get();
		if(R.stepsave>=R.Nsave-10)std::cerr<<"Error! ResultU StepSave>NSave "<< R.stepsave<<" "<<R.Nsave<<" "<<Compress.Strain<<"\n";
		if(fabs(Compress.V)>1e-9)++R.stepsave;
	}
	Compress.Stress = 0.5 * (fabs(R.sFL[0]) + fabs(R.sFL[1])) * Compress._1d_Area;
	if (!Compress.Loading && Compress.Stress > Compress.MinStress)
	{
		Compress.Loading = true;
	}
	if (Compress.FractureStress < Compress.Stress)
	{
		Compress.FractureStress = Compress.Stress;
		Compress.FractureStrain = (R.sz0 - Compress.size_d.z) / R.sz0;
	}
	if (Compress.Loading)
	{
		//std::cerr << "SumForcesULoading F " << Compress.FractureStress << " " << Compress.Stress << " " << Compress.ZeroStress << "\n";		
		if (Compress.Stress < Compress.ZeroStress || Compress.Stress  < Compress.ConditionFracture*Compress.FractureStress )
		{
			Compress.Fracture = true;			
			//std::cerr << "SumForcesULoading F " << Compress.FractureStress << " " << Compress.Stress << " " << Compress.ZeroStress << " " << Compress.ConditionFracture << "\n";
		}
	}
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}

void SumForcesBLoading(result_data& R, compression_data& Compress, uint n)
{
	HANDLE_ERROR(cudaMemcpy(R.h_FL, R.d_FL, R.N * sizeof(float), cudaMemcpyDeviceToHost));
	uint ir;
	R.sFL[0] = 0;
	for (ir = 0; ir < R.bloks; ++ir)
	{
		R.sFL[0] += R.h_FL[ir];
	}
	R.sFL[1] = 0;
	for (ir = R.bloks; ir < 2 * R.bloks; ++ir)
	{
		R.sFL[1] += R.h_FL[ir];
	}
	if (n % R.dNsave == 0)
	{
		R.h_sFL[3 * R.stepsave] = (Compress.size_d.y - R.sz0) / R.sz0;
		//std::cerr<<"BTest "<<R.stepsave<<" "<<Compress.size_d.y<<" "<<R.sz0<<"\n";
		R.h_sFL[3 * R.stepsave + 1] = R.sFL[0] * Compress._1d_Area;
		R.h_sFL[3 * R.stepsave + 2] = R.sFL[1] * Compress._1d_Area;
		//std::cerr<<"SBF "<<R.stepsave<< " "<< R.h_sFL[3 * R.stepsave] << " " << R.h_sFL[3 * R.stepsave + 1] * stress_const << " " << R.h_sFL[3 * R.stepsave + 2] * stress_const << "\n"; //std::cin.get();
		if(R.stepsave>=R.Nsave-10)std::cerr<<"Error! ResultB StepSave>NSave "<< R.stepsave<<" "<<R.Nsave<<" "<<Compress.Strain<<"\n";
		if(fabs(Compress.V)>1e-9)++R.stepsave;
	}
	Compress.Stress = 0.5 * (fabs(R.sFL[0]) + fabs(R.sFL[1])) * Compress._1d_Area;
	if (!Compress.Loading && Compress.Stress > Compress.MinStress)
	{
		Compress.Loading = true;
	}
	if (Compress.FractureStress < Compress.Stress || Compress.Stress  < Compress.ConditionFracture*Compress.FractureStress)
	{
		Compress.FractureStress = Compress.Stress;
		Compress.FractureStrain = (R.sz0 - Compress.size_d.y) / R.sz0;
	}
	if (Compress.Loading)
	{
		//std::cerr << "F " << Compress.FractureStress << " " << Compress.Stress << "\n";
		
		if (Compress.Stress < Compress.ZeroStress)
		{
			Compress.Fracture = true;
		}
	}
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}

void CalculateLinearityDeviation(result_data& R, compression_data & Compress)
{
	double l_eps = 0, l_deps = 0, l_A = Compress.FractureStress/Compress.FractureStrain, l_dStress = 0;
	Compress.LinearityDeviation = 0;
	uint l_i = 0;		
	do
	{
		Compress.LinearityDeviation += l_deps*l_dStress*l_dStress;
		++l_i;
		l_eps = -R.h_sFL[3 * l_i];
		l_deps = -0.5*(R.h_sFL[3 * (l_i+1)] - R.h_sFL[3 * (l_i-1)]);
		l_dStress = (l_A*l_eps - 0.5*(R.h_sFL[3 * l_i + 2] - R.h_sFL[3 * l_i + 1]));
		//std::cerr<< l_i <<" " << l_eps <<" "<< l_deps <<" "<< l_dStress <<" | "<< Compress.FractureStrain <<" "<< R.stepsave <<"\n";
	} while(l_eps < Compress.FractureStrain-2*l_deps && l_i<R.stepsave);
	Compress.LinearityDeviation /= (Compress.FractureStrain*Compress.FractureStress*Compress.FractureStress);
	Compress.LinearityDeviation += 1.0;
	std::cerr << "CalculateLinearityDeviation " << Compress.LinearityDeviation <<" "<< Compress.FractureStrain << " "<< Compress.LinearityDeviation<< "\n";
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}


void SaveSumForcesLoading(result_data& R, char* Name)
{
	std::ofstream file;
	file.open(Name, std::ios::out);
	file << "EpsZ StressZ_bottom StressZ_top\n";
	uint i;
	for (i = 0; i < R.stepsave; ++i)
	{
		file << R.h_sFL[3 * i] << " " << R.h_sFL[3 * i + 1] * stress_const << " " << R.h_sFL[3 * i + 2] * stress_const << "\n";
	}	
	file.close();
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}

void CalculateDestruction(result_data& R)
{
	R.Destruction = R.h_sSI[R.stepsave-1]/double(R.h_sSI[0]);	
	std::cerr << "CalculateDestruction " << R.Destruction <<" "<< R.h_sSI[R.stepsave-1] << " "<< R.h_sSI[0]<< "\n";
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}

void SumInteractions(result_data& R, interaction_list_data& IL, uint n)
{
	HANDLE_ERROR(cudaMemcpy(R.h_SI, R.d_SI, IL.IonP*R.bloks * sizeof(uint), cudaMemcpyDeviceToHost));
	uint l_i, l_si = 0;
	for (l_i = 0; l_i < IL.IonP*R.bloks; ++l_i)
	{
		l_si += R.h_SI[l_i];
		//std::cerr<<"I "<<R.h_SI[l_i]<<"\n";
	}	
	if (n % R.dNsave == 0)
	{
		R.h_sSI[R.stepsave] = l_si;
		//std::cerr<<"SI "<<R.stepsave<<" "<<R.h_sSI[R.stepsave] << "\n"; //std::cin.get();
		if(R.stepsave>=R.Nsave-10)std::cerr<<"Error! ResultU SumInteractions StepSave>NSave "<< R.stepsave<<" "<<R.Nsave<<"\n";
	}	
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}

__global__ void d_SumUpInteractions(uint* __restrict__ IL, uint* __restrict__ ILtype, uint IonP, uint* __restrict__ SI, uint N)
{
	__shared__ float s_mem[SMEMDIM];
	uint idx = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x, sumi=0;	
	//uint3 cm;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	
	while (idx < N)
	{
		if (IL[idx] != UINT_MAX && ILtype[idx]==1)
		{
			//printf("I0n %u %u %u\n", idx, IL[idx], ILtype[idx]);
			++sumi;
		}
		
		
		//printf("I0n %u %e %e %e\n", idx, R[idx], R[idx + N], R[idx + 2 * N]);
		idx += blockDim.x * gridDim.x;
	}
	s_mem[tid] = sumi;
	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512)
	{
		s_mem[tid] += s_mem[tid + 512];
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
	{
		//printf("Blok!\n");
		s_mem[tid] += s_mem[tid + 256];
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
	{
		s_mem[tid] += s_mem[tid + 128];
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
	{
		s_mem[tid] += s_mem[tid + 64];
	}
	__syncthreads();
	
	// unrolling warp
	if (tid < 32)
	{
		volatile float* vsmem = s_mem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		//printf("U %u %u %u\n", idx, blockIdx.x, blockIdx.x + gridDim.x);
		SI[blockIdx.x] = s_mem[0];		
	}
}