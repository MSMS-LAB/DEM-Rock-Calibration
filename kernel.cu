#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include "pcuda_helper.h"
#include "md_data_types.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include "md.h"
#include "mp.h"
#include <chrono>
float normm(float3 r)
{
    return sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
}

float normm(float* r, uint i, uint N)
{
    return sqrt(r[i] * r[i] + r[i + N] * r[i + N] + r[i + 2 * N] * r[i + 2 * N]);
}
int main(int argc, char* argv[])
{
	bool l_flags[]={false, false, false, false, false, false, false, false, false};
	std::string l_MaterialString, l_param, l_indexstring;
	int deviceCount, idevice, l_2Shift=0, l_i;
    cudaGetDeviceCount(&deviceCount);   
    std::cout << "There are " <<  deviceCount << " GPUs"<< std::endl;
	if(argc > 1)
	{
		for(uint i = 1; i<argc ; ++i )
		{
			l_param = std::string(argv[i]);
			if(l_param.compare(0, 1, "G") == 0)
			{
				idevice = argv[i][1] - '0';
				if(idevice>=0 && idevice<deviceCount)
				{
					std::cout<<"Correct GPU request! Requested: "<<idevice<<"\n";
					cudaSetDevice(idevice); 
					l_flags[0] = true;
				} else
				{
					std::cerr<<"Incorrect GPU request! Requested:"<<idevice<<"\n";
					return 1;
				}
			}
			if(l_param.compare(0, 1, "M") == 0)
			{
				l_MaterialString = l_param;
				l_flags[1] = true;
				std::cerr<<"Set Calculate One Sample\n";
			}
			if(l_param.size()>=2 && l_param.compare(0, 2, "CS") == 0)
			{
				l_flags[2] = true;
				l_2Shift = 2;
			}
			if(l_param.size()>=l_2Shift+1 && l_param.compare(l_2Shift, 1, "U") == 0)
			{
				l_flags[5] = true;
				++l_2Shift; 
			}
			if(l_param.size()>=l_2Shift+1 && l_param.compare(l_2Shift, 1, "B") == 0)
			{
				l_flags[6] = true;
				++l_2Shift; 
			}
			if(l_param.compare(0, 3, "DCG") == 0)
			{				
				l_flags[3] = true;
				std::cerr<<"Set DetermineConjGradient\n";
			}
			if(l_param.compare(0, 3, "TSS") == 0)
			{				
				l_flags[4] = true;
				std::cerr<<"Set TestSeveralSamples\n";
			}
			if(l_param.compare(0, 4, "FUSC") == 0)
			{				
				l_flags[7] = true;
				std::cerr<<"Set Force uniaxial sample creation\n";
			}
			if(l_param.compare(0, 4, "FBSC") == 0)
			{				
				l_flags[8] = true;
				std::cerr<<"Set Force brazil sample creation\n";
			}
			if(l_param.compare(0, 1, "I") == 0)
			{
				l_indexstring = l_param;
				l_indexstring.erase(0, 1);				
				std::cout<<"INDEX "<< l_indexstring <<"\n";
			}
		}
	}
	//double j = 0;
	//for(int i=0; i<1000000000;++i)
	//	j+=pow(i, 1.0/double(i));
	//for(l_i=0; l_i<7;++l_i)std::cerr<<"Flag "<<l_i<<" "<<l_flags[l_i]<<"\n";std::cerr<<"M "<<l_MaterialString<<"\n";
	//exit(0);
	
	if(!l_flags[0])
	{
		std::cerr<<"Selected by default GPU 0\n";
		cudaSetDevice(0); // Select the second GPU
	}
	
	if(l_flags[1]) CalculateOneSample(l_MaterialString, l_flags[2], l_flags[5], l_flags[6], l_flags[7], l_flags[8]);
	if(l_flags[3]) DetermineConjGradient();
	if(l_flags[4]) TestSeveralSamples();
    //std::cin.get();       
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus;
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

