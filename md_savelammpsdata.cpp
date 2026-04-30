#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <cstring>

void SaveLammpsDATASimple(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po,
	char* Name, bool flagv)
{
	double Et, Er;
	uint i, j, k, ILn = 0;
	for (i = 0; i < IL.N; ++i)
	{
		j = i / IL.IonP;
		//std::cerr << "I " << i << " " << j << " " << IL.h_IL[i] << "\n";
		if (IL.h_IL[i] < P.N && IL.h_ILtype[i]==1)
			++ILn;
	}
	//std::cin.get();
	std::ofstream file;
	file.open(Name, std::ios::out);
	file << "LAMMPS Description T0="<<Po.dt<<"  (1st line of file)\n\n";
	file << P.N << " atoms\n" << ILn << " bonds\n" << 0 << " angles\n" << 0 << " dihedrals\n" << 0 << " impropers\n";
	file << 1 << " atom types\n" << 2 << " bond types\n";
	file << S.A.x * length_const << " " << S.B.x * length_const << " xlo xhi\n" << S.A.y * length_const << " " << S.B.y * length_const << " ylo yhi\n" << S.A.z * length_const << " " << S.B.z * length_const << " zlo zhi\n";
	file << "Masses\n\n" << 1 << " " << 1.0 << "\n";
	//file << "Nonbond Coeffs\n" << 1 << " " << 1.0 << "\n" << 2 << " " << 1.0 << "\n" << 3 << " " << 1.0 << "\n" << 4 << " " << 1.0 << "\n" << 5 << " " << 1.0 << "\n";
	file << "Bond Coeffs\n\n" << 2 << " " << 1.0 << " " << 2.0 << "\n\n";
	file << "Atoms\n\n";
	for (i = 0; i < P.N; ++i)
	{
		file << i+1 << " " << "1 ";
		file << P.h_R[i] * length_const << " " << P.h_R[i + P.N] * length_const << " " << P.h_R[i + 2*P.N] * length_const << "\n";
			//<< " " << 0.5 * Po.m * P.h_V[i] * P.h_V[i] << " " << 0.5 * Po.m * P.h_V[i + P.N] * P.h_V[i + P.N]//<<"\n";
			//<< " " << P.h_V[i] << " " << P.h_V[i + P.N]
			//<< " " << P.h_F[i] << " " << P.h_F[i + P.N] << "\n";
	}
	file << "Velocities\n\n";
	for (i = 0; i < P.N; ++i)
	{
		if (flagv)
		{
			Et = 0.5 * Po.p_m * (P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N] + P.h_V[i + 2 * P.N] * P.h_V[i + 2 * P.N]) * energy_const;
			Er = 0.5 * Po.p_I * (P.h_W[i] * P.h_W[i] + P.h_W[i + P.N] * P.h_W[i + P.N] + P.h_W[i + 2 * P.N] * P.h_W[i + 2 * P.N]) * energy_const;
		}
		else
		{
			Et = 0;
			Er = 0;
		}
		
		//ek = P.h_IM[i] * (P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N]);
		file << i + 1 << " " << Et << " " << Er << " " << Et + Er << "\n";
		//std::cerr << "FI " << Padd.h_Fbound[i] << " " << Padd.h_LammpsSumF[i] << "\n"; std::cin.get();
		
		
	}
	file << "\nBonds\n\n";
	k = 1;
	for (i = 0; i < IL.N; ++i)
	{
		j = i / IL.IonP;
		if (IL.h_IL[i] < P.N && IL.h_ILtype[i] == 1)
		{			
			file << k << " " << IL.h_ILtype[i] + 1 << " " << j + 1 << " " << IL.h_IL[i] + 1 << "\n";
			++k;
		}
			
	}	
	file.close();
	//std::cerr << "Save PBM "<<"\n";
	//std::cerr << "LL " << P.N << " " << pnid2 << " | " << P.N + pnid2 << " " << P.NI << " " << j << "\n";
}
