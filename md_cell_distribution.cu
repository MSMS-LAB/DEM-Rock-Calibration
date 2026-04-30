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
#include <iostream>
#include <math.h>
#include <cub/cub.cuh>

void CellDistributionInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C)
{		
	C._1d_a = 1.0 / double(C.a);
	C.Nr.x = floorf(S.B.x * C._1d_a) + 1;
	C.Nr.y = floorf(S.B.y * C._1d_a) + 1;
	C.Nr.z = floorf(S.B.z * C._1d_a) + 3;
	S.hidenpoint.x = S.A.x + 0.5 * S.spacesize.x;
	S.hidenpoint.y = S.A.y + 0.5 * S.spacesize.y;
	S.hidenpoint.z = (C.Nr.z-0.5) * C.a;
	std::cerr << "CellDistributionInit S " << S.B.x << " " << S.B.y << " " << S.B.z 
		<< " | " << S.L.x << " " << S.L.y << " " << S.L.z 
		<< " | " << S.center.x << " " << S.center.y << " " << S.center.z
		<< " | " << S.size.x << " " << S.size.y << " " << S.size.z << "\n";
	std::cerr << S.B.z <<" "<< floorf(S.B.z * C._1d_a)<< " "<<C.a<<" "<< C._1d_a << "\n";
	std::cerr << "CellDistributionInit S.hidenpoint " << S.hidenpoint.x << " " << S.hidenpoint.y << " " << S.hidenpoint.z << " " << S.hidenpoint.z * C._1d_a << "\n"; 
	std::cerr << "CellDistributionInit C.Nr " << C.Nr.x << " " << C.Nr.y << " " << C.Nr.z << "\n";
	//std::cin.get();
	C.N = C.Nr.x * C.Nr.y * C.Nr.z;
	std::cerr << "CellDistributionInit " << A.bloks << " CN=" << C.N << " PN=" << P.N << " " << (4 * P.N + 2 * C.N) * sizeof(uint) << " " << ((4 * P.N + 2 * C.N) * sizeof(uint)) / (1024 * 1024) << "\n";

	HANDLE_ERROR(cudaMalloc((void**)&C.d_IP, P.N * sizeof(uint)));
	d_FillIndex << <A.bloks, SMEMDIM >> > (C.d_IP, P.N);
	HANDLE_ERROR(cudaMalloc((void**)&C.d_CI, P.N * sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**)&C.d_CIs, 2 * P.N * sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**)&C.d_pnC, 2 * C.N * sizeof(uint)));
	//std::cerr << "C.d_pnC " << C.d_pnC << "\n"; std::cin.get();
	C.dtmpN_old = 128;
	HANDLE_ERROR(cudaMalloc((void**)&C.d_tmp_old, C.dtmpN_old));
	C.h_IP = (uint*)malloc(P.N * sizeof(uint));
	C.h_CI = (uint*)malloc(P.N * sizeof(uint));
	C.h_CIs = (uint*)malloc(2 * P.N * sizeof(uint));
	C.h_pnC = (uint*)malloc(2 * C.N * sizeof(uint));
	C._1d_Nr.x = 1.0 / double(C.Nr.x);
	C._1d_Nr.y = 1.0 / double(C.Nr.y);
	C._1d_Nr.z = 1.0 / double(C.Nr.z);
}

void CellDistribution(particle_data& P, additional_data& A, sample_data& S, cell_data& C)
{
	//std::cerr << "A " << A.bloks << "\n";	
	//HANDLE_ERROR(cudaMemcpy(C.h_IP, C.d_IP, P.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < P.N; ++i) std::cerr << i << " " << C.h_IP[i] << "\n";
	//std::cin.get();
	//std::cerr << "C " << C._1d_a << " " << C.Nr.x << " " << C.Nr.y << " " << C.Nr.z << " " << P.N << "\n";
	d_CalculateCellIndex << <A.bloks, SMEMDIM >> > (P.d_R, P.N, C.d_CI, C._1d_a, C.Nr);
	//HANDLE_ERROR(cudaMemcpy(C.h_CI, C.d_CI, P.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < P.N; ++i)if (C.h_CI[i] > C.N || C.h_CI[i] > C.N - C.Nr.y * C.Nr.x - C.Nr.x - 1)	std::cerr << i << " " << C.h_CI[i] << "\n";std::cin.get();
	//HANDLE_ERROR(cudaMemcpy(C.h_CIs + P.N, C.d_IP, P.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//std::cin.get();
	//for (int i = 0; i < P.N; ++i)std::cerr << i << " " << C.h_CI[i] << " " << C.h_CIs[i] << " " << C.h_CIs[i + P.N] << "\n";
	//std::cin.get();
	C.dtmpN = 0;
	C.d_tmp = nullptr;
	cub::DeviceRadixSort::SortPairs(C.d_tmp, C.dtmpN, C.d_CI, C.d_CIs, C.d_IP, C.d_CIs + P.N, P.N);
	//thrust::sort_by_key(C.d_CIs, C.d_CIs + P.N, C.d_CIs + P.N);
	//std::cerr << "Size " << C.dtmpN << " " << C.dtmpN /P.N << " " << C.dtmpN % P.N << "\n";
	//std::cin.get();

	if (C.dtmpN > C.dtmpN_old)
	{
		cudaFree(C.d_tmp_old);
		C.dtmpN_old = C.dtmpN;
		HANDLE_ERROR(cudaMalloc((void**)&C.d_tmp_old, C.dtmpN_old));
		C.d_tmp = C.d_tmp_old;
	}
	else
	{
		C.dtmpN = C.dtmpN_old;
		C.d_tmp = C.d_tmp_old;
	}
		
	//std::cerr << "Size " << C.dtmpN << " " << C.dtmpN / P.N << " " << C.dtmpN % P.N << "\n";
	cub::DeviceRadixSort::SortPairs(C.d_tmp, C.dtmpN, C.d_CI, C.d_CIs, C.d_IP, C.d_CIs + P.N, P.N);
	//C.d_tmp = nullptr;//!make C.dtmpN_old
	
	//HANDLE_ERROR(cudaMemcpy(C.h_CIs, C.d_CIs, 2 * P.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < P.N; ++i)std::cerr << i << " " << C.h_CIs[i] << " " << C.h_CIs[i + P.N] << " " << C.h_CI[i] << "\n";std::cin.get();
	//std::cerr << "C.d_pnC " << C.d_pnC <<" "<< C.N << "\n"; std::cin.get();
	HANDLE_ERROR(cudaMemset((void*)C.d_pnC, 0, 2 * C.N * sizeof(uint)));
	d_DetermineCellPointer << <A.bloks, SMEMDIM >> > (C.d_CIs, C.d_pnC, P.N, C.N);
	//HANDLE_ERROR(cudaMemcpy(C.h_pnC, C.d_pnC, 2 * C.N * sizeof(uint), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < P.N; ++i)
	//	std::cerr << i << " " << C.h_pnC[i] << " " << C.h_pnC[i + C.N] << "\n";
	//std::cerr << "Cells " << A.bloks << " " << P.d_R << " " << P.N << " " << C.d_CI << " " << C._1d_a << " " << C.Nr.x << " " << C.Nr.y << " " << C.Nr.z << "\n";
	//std::cerr << "Cells " << C.d_tmp << " " << C.dtmpN << " " << C.d_CI << " " << C.d_CIs << " " << C.d_IP << " " << C.d_CIs + P.N << " " << P.N << "\n";
	//std::cerr << "Cells " << C.d_CIs << " " << C.d_pnC << " " << P.N << " " << C.N << "\n";	std::cin.get();
}
