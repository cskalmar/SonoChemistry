#ifndef SC_SYSTEMDEFINITION_H
#define SC_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

//&F[3] = omega;

// SYSTEM
template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(int tid, int NT, Precision* F, Precision* X, Precision T, Precision* cPAR, Precision* sPAR, int* sPARi, Precision* ACC, int* ACCi)
{
	for (int i = 0; i < NumberOfMolecules + 3 + 1; i++)
		F[i] = 0.0;

	Precision X_conc[NumberOfMolecules];
	Precision H[NumberOfMolecules];
	Precision S_0[NumberOfMolecules];

	Precision Temp 		= X[2] * sPAR[0];
	for (int k = 0; k < NumberOfMolecules; k++)
		X_conc[k] 		= X[k+3] * cPAR[16];

	Precision M 		= sum(X_conc, NumberOfMolecules);
	Precision tmp		= 1.0 / M;

	Precision p			= M * sPAR[1] * Temp * 0.1;
	Precision C_p_mean	= 0.0;
	CalculateThermoDynamics(C_p_mean, H, S_0, X_conc, Temp, sPAR[1]);
	C_p_mean			*= tmp;

	Precision lambda_mean = 0.0;
	for (int k = 0; k < NumberOfMolecules; k++) lambda_mean += X_conc[k] * const_lambda[k];
	lambda_mean			*= tmp;
	Precision Heat		= ThermalConduction(X[0] * cPAR[15], X[1] * cPAR[14] * cPAR[15], lambda_mean, sPAR, Temp, C_p_mean * M);
	Precision m_net_mol	= Evaporation(sPAR, Temp, p * X_conc[5] * tmp);

	Reactions(&F[3], Temp, X_conc, sPAR, S_0, H);
	Precision rX0		= 1.0 / X[0];
	Precision rX0pc15	= rX0 / cPAR[15];
	Precision dTdt		= (-SumCoeffProd(H, &F[3], NumberOfMolecules) - p * 30.0 * X[1] * rX0 * cPAR[14] + 30.0 * rX0pc15 * Heat) / (M * (C_p_mean - sPAR[1]));
	F[2]				= dTdt / cPAR[14] / sPAR[0];

	Precision p_inf 	= sPAR[2] + cPAR[6] * sin(2.0 * PI * T) + cPAR[7] * sin(cPAR[8] * T + cPAR[9]);
	Precision p_inf_dot = cPAR[10] * cos(2.0 * PI * T) + cPAR[11] * cos(cPAR[8] * T + cPAR[9]);
	Precision p_L 		= p - cPAR[12] * rX0 - cPAR[13] * X[1] * rX0;
	Precision dpdt 		= p * (sum(&F[3], NumberOfMolecules) * tmp + dTdt / Temp - 3.0 * X[1] * rX0 * cPAR[14]);

	Precision Nom 		= (p_L - p_inf) * cPAR[0] * rX0 + (p - p_inf) * X[1] * cPAR[1] * rX0 + (dpdt - p_inf_dot) * cPAR[2] - (1.0 - cPAR[3] * X[1]) * 1.5 * X[1] * X[1] * rX0;
	Precision Den 		= 1.0 - cPAR[4] * X[1] + cPAR[5] * rX0;

	F[0] 				= X[1];
	F[1] 				= Nom / Den;

	for (int k = 0; k < NumberOfMolecules; k++)
		F[k+3] 			-= X_conc[k] * 3.0 * X[1] * rX0 * cPAR[14];

	F[8] 				+= m_net_mol * 3.0 * rX0pc15 * 1.0e-6;

	tmp					= 1.0 / (cPAR[14] * cPAR[16]);
	for (int k = 0; k < NumberOfMolecules; k++)
		F[k+3] 			*= tmp;

	if (ACCi[0] == 1)
	{
		Precision d2Rdt2 = F[1] * cPAR[15] * cPAR[14] * cPAR[14];
		Precision R		 = X[0] * cPAR[15];
		Precision R_dot	 = X[1] * cPAR[15] * cPAR[14];
		Precision V		 = (4.0/3.0) * R * R * R * PI;
		Precision dVdt	 = 3.0 * V * R_dot * rX0pc15;

		tmp						= 1.0 / sPAR[7];
		F[NumberOfMolecules+3]	= -(p * (1.0 + R_dot * tmp) + R * tmp * dpdt) * dVdt;
		F[NumberOfMolecules+3]	+= 4.0 * PI * (cPAR[13] / cPAR[14]) * (R * R_dot * R_dot + R * R * R_dot * d2Rdt2 * tmp);
		F[NumberOfMolecules+3]	+= 4.0 * PI * tmp * R * R * R_dot * (R_dot * p + dpdt * R - 0.5 * sPAR[8] * R_dot * R_dot * R_dot - sPAR[8] * R * R_dot * d2Rdt2);
	}
}

// EVENTS
template <class Precision>
__forceinline__ __device__ void PerThread_EventFunction(int tid, int NT, Precision* EF, Precision  T, Precision dT, Precision* TD, Precision* X, Precision* cPAR, Precision* sPAR, int* sPARi, Precision* ACC, int* ACCi)
{
}

template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterEventDetection(int tid, int NT, int IDX, int& UDT, Precision &T, Precision &dT, Precision* TD, Precision* X, Precision* cPAR, Precision* sPAR, int* sPARi, Precision* ACC, int* ACCi)
{
}

// ACCESSORIES
template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(int tid, int NT, int& UDT, Precision& T, Precision& dT, Precision* TD, Precision* X, Precision* cPAR, Precision* sPAR, int* sPARi, Precision* ACC, int* ACCi)
{
	if (ACCi[0] == 1)
	{
		if (X[0] > ACC[2]) //x1_max_local
		{
			ACC[2] 				= X[0];
			ACC[6] 				= T; //tau_max_local
			Precision V_cPAR16 	= cPAR[16] * (4.0/3.0) * (X[0] * cPAR[15] * 100.0) * (X[0] * cPAR[15] * 100.0) * (X[0] * cPAR[15] * 100.0) * PI; //cm^3
			for (int k = 0; k < NumberOfMolecules; k++)
				ACC[k+14] 		= X[k+3] * V_cPAR16; //yield_k_local_max in moles
		}

		if (X[0] < ACC[3]) //x1_min_local
		{
			ACC[3] 				= X[0];
			ACC[7] 				= T; //tau_min_local
		}

		ACC[11]					= fmax(X[2], ACC[11]); //T_max_local
	}
}

template <class Precision>
__forceinline__ __device__ void PerThread_Initialization(int tid, int NT, int& DOIDX, Precision& T, Precision& dT, Precision* TD, Precision* X, Precision* cPAR, Precision* sPAR, int* sPARi, Precision* ACC, int* ACCi)
{
	T     		 			= TD[0]; // Reset the starting point of the simulation from the lower limit of the time domain

	if (ACCi[0] == 1)
	{
		X[NumberOfMolecules+3]	= 0.0; //Pi_w

		//Initializing local maxima/minima
		int Indices[]			= {2, 6, 7, 8, 11};
		for (auto k : Indices)
			ACC[k] 				= 0.0;
		for (int k = 14; k < 24; k++)
			ACC[k]				= 0.0;

		ACC[3]					= 1.0e10;
	}
}

template <class Precision>
__forceinline__ __device__ void PerThread_Finalization(int tid, int NT, int& DOIDX, Precision& T, Precision& dT, Precision* TD, Precision* X, Precision* cPAR, Precision* sPAR, int* sPARi, Precision* ACC, int* ACCi)
{
	if (ACCi[0] == 1)
	{
		ACC[0]			= fmax(ACC[2], ACC[0]); //x1 global max
		ACC[1]			+= ACC[2]; //for x1 global avg

		ACC[4]			= fmax(ACC[2] / ACC[3], ACC[4]); //x1_max_local/x1_min_local global max
		ACC[5]			+= ACC[2] / ACC[3]; //for x1max/x1min global avg

		ACC[8]			= (ACC[7] > ACC[6]) ? ACC[7] - ACC[6] : ACC[7] + 1 - ACC[6]; // tau_c local

		Precision tmp	= ACC[2] * ACC[2] * ACC[2] / ACC[8];
		ACC[9]			= fmax(tmp, ACC[9]); //(x1max^3/tau_c) global max
		ACC[10]			+= tmp; //for (x1max^3/tau_c) global avg

		ACC[12]			= fmax(ACC[11], ACC[12]); //x3 global max
		ACC[13]			+= ACC[11]; //for x3 global avg

		tmp				= 1.0 / X[NumberOfMolecules+3]; //Pi_w
		for (int k = 0; k < NumberOfMolecules; k++)
		{
			ACC[k+24]	= fmax(ACC[k+14], ACC[k+24]); //Y_k global max
			ACC[k+34]	+= ACC[k+14]; //for Y_k global avg

			ACC[k+44]	= fmax(ACC[k+14] * tmp, ACC[k+44]); //Y_k/Pi_w global max
			ACC[k+54]	+= ACC[k+14] * tmp; //for Y_k/Pi_w global avg
		}
	}
}

#endif