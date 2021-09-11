#ifndef SC_SYSTEMDEFINITION_H
#define SC_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	for (int i = 0; i < NumberOfMolecules + 3 + 1; i++)
		F[i] = 0.0;

	Precision X_conc[NumberOfMolecules]; 
	Precision x[NumberOfMolecules];
	Precision C_v[NumberOfMolecules];
	Precision C_p[NumberOfMolecules];
	Precision H[NumberOfMolecules];
	Precision S_0[NumberOfMolecules];
	Precision omega[NumberOfMolecules];

//	Precision TmpColumnVector[NumberOfReactions];

	Precision Temp 		= X[2] * sPAR[0];
	for (int k = 0; k < NumberOfMolecules; k++)	
		X_conc[k] 		= X[k+3] * cPAR[16];

	Precision M 		= sum(X_conc, NumberOfMolecules);
	Precision rM		= 1.0 / M;
	for (int k = 0; k < NumberOfMolecules; k++)
		x[k] 			= X_conc[k] * rM;
	Precision W_mean	= SumCoeffProd(x, 		&sPAR[1], NumberOfMolecules);
	Precision Rho_mean	= SumCoeffProd(X_conc, 	&sPAR[1], NumberOfMolecules);

	Precision p			= M * sPAR[11] * Temp * 0.1;
	CalculateThermoDynamics(C_v, C_p, H, S_0, Temp, sPAR[11]);
	Precision C_v_mean	= SumCoeffProd(C_v, x, NumberOfMolecules);
//	C_v_mean			= 2.1148e8;
	Precision C_p_mean	= SumCoeffProd(C_p, x, NumberOfMolecules);
	Precision c_p_mean	= C_p_mean / W_mean;

	Precision Heat		= ThermalConduction(X[0] * cPAR[15], X[1] * cPAR[14] * cPAR[15], x, sPAR, Temp, c_p_mean, Rho_mean);
	Precision m_net_mol	= Evaporation(sPAR, Temp, p, x);

	Reactions(omega, Temp, X_conc, sPAR, S_0, H);
	Precision rX0		= 1.0 / X[0];
	Precision rX0pc15	= rX0 / cPAR[15];
	Precision dTdt		= (-SumCoeffProd(H, omega, NumberOfMolecules) - p * 30.0 * X[1] * rX0 * cPAR[14] + \
								30.0 * rX0pc15 * Heat) / (M * C_v_mean);
	F[2]				= dTdt / cPAR[14] / sPAR[0];

	for (int k = 0; k < NumberOfMolecules; k++)
		F[k+3] 			= omega[k] - X_conc[k] * 3.0 * X[1] * rX0 * cPAR[14];
		
	F[8] 				+= m_net_mol * 3.0 * rX0pc15 * 1.0e-6;

	Precision tmp		= 1.0 / (cPAR[14] * cPAR[16]);
	for (int k = 0; k < NumberOfMolecules; k++)
		F[k+3] 			*= tmp;

	Precision p_inf 	= sPAR[12] + cPAR[6] * sin(2.0 * PI * T) + cPAR[7] * sin(cPAR[8] * T + cPAR[9]);
	Precision p_inf_dot = cPAR[10] * cos(2.0 * PI * T) + cPAR[11] * cos(cPAR[8] * T + cPAR[9]);
	Precision p_L 		= p - cPAR[12] * rX0 - cPAR[13] * X[1] * rX0;
	Precision dpdt 		= p * (sum(omega, NumberOfMolecules) * rM + dTdt / Temp - 3.0 * X[1] * rX0 * cPAR[14]);

	Precision Nom 		= (p_L - p_inf) / cPAR[0] * rX0 + (p - p_inf) * X[1] / cPAR[1] * rX0 \
		+ (dpdt - p_inf_dot) / cPAR[2] - (1.0 - cPAR[3] * X[1]) * 1.5 * X[1] * X[1] * rX0;
	Precision Den 		= 1.0 - cPAR[4] * X[1] + cPAR[5] * rX0;

	F[0] 				= X[1];
	F[1] 				= Nom / Den;

	Precision d2Rdt2 = F[1] * cPAR[15] * cPAR[14] * cPAR[14];
	Precision R		 = X[0] * cPAR[15];
	Precision R_dot	 = X[1] * cPAR[15] * cPAR[14];
	Precision V		 = 1.3333333333 * R * R * R * PI;
	Precision dVdt	 = 3.0 * V * R_dot * rX0pc15;

	Precision int_th, int_v, int_r;
	int_th 			= -(p * (1.0 + R_dot / sPAR[21]) + R / sPAR[21] * dpdt) * dVdt;
	int_v			= 4.0 * PI * (cPAR[13] / cPAR[14]) * (R * R_dot * R_dot + R * R * R_dot * d2Rdt2 / sPAR[21]);
	int_r			= 4.0 * PI / sPAR[21] * R * R * R_dot * (R_dot * p + dpdt * R - 0.5 * sPAR[22] * R_dot * R_dot * R_dot - sPAR[22] * R * R_dot * d2Rdt2);

	F[NumberOfMolecules+3] = int_th + int_v + int_r;
}

// EVENTS
template <class Precision>
__forceinline__ __device__ void PerThread_EventFunction(\
			int tid, int NT, Precision* EF, \
			Precision     T, Precision    dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
}

template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterEventDetection(\
			int tid, int NT, int IDX, int& UDT, \
			Precision    &T, Precision   &dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{	
}

// ACCESSORIES
template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
			int tid, int NT, int& UDT, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{
	if (ACCi[0] == 1) //Transient
	{
		if (X[0] > ACC[0])
		{
			ACC[0] = X[0]; //x_1_local_max
			Precision V 	= 1.3333333333 * (X[0] * cPAR[15] * 100.0) * (X[0] * cPAR[15] * 100.0) * (X[0] * cPAR[15] * 100.0) * PI;
			for (int k = 0; k < NumberOfMolecules; k++)
				ACC[2+k] 	= X[k+3] * cPAR[16] * V; //yield_i_local
		}
	}
	else //Converged
	{
		if (X[0] > ACC[0])
		{
			ACC[0] = X[0]; //x_1_local_max
			Precision V 	= 1.3333333333 * (X[0] * cPAR[15] * 100.0) * (X[0] * cPAR[15] * 100.0) * (X[0] * cPAR[15] * 100.0) * PI;
			for (int k = 0; k < NumberOfMolecules; k++)
				ACC[2+k] 	= X[k+3] * cPAR[16] * V; //yield_i_local
		}

		if (X[2] > ACC[63])
			ACC[63] = X[2]; // T_max
	}
}

template <class Precision>
__forceinline__ __device__ void PerThread_Initialization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	T      = TD[0]; // Reset the starting point of the simulation from the lower limit of the time domain

	X[NumberOfMolecules+3] = 0.0;
}

template <class Precision>
__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	if (ACCi[0] == 1) //Transient
	{
		Precision Pi_w					= X[NumberOfMolecules+3];
		ACC[52]	+= Pi_w / cPAR[14];

		for (int k = 0; k < NumberOfMolecules; k++)
		{
			Precision yield_i_local		= ACC[2+k];

			if (yield_i_local / ACC[52] > ACC[53+k])
			{
				ACC[53+k] 				= yield_i_local / ACC[52];
				ACCi[1+k]++;
			}
		}

		for (int k = 0; k < NumberOfMolecules + 2; k++)
			ACC[k] = 0.0;
	}
	else //Converged
	{
		Precision Pi_w					= X[NumberOfMolecules+3];
		
		for (int k = 0; k < NumberOfMolecules; k++)
		{
			Precision yield_i_local 	= ACC[2+k];

			if (yield_i_local > ACC[12+k])
				ACC[12+k] 				= yield_i_local; // yield_i_global

			ACC[22+k] 					+= yield_i_local; //yield_i_avg-hoz

			if (yield_i_local / Pi_w > ACC[32+k])
				ACC[32+k] 				= yield_i_local / Pi_w; //yield_i/Pi_w global max

			ACC[42+k] 					+= yield_i_local / Pi_w; //yield_i/Pi_w avg-hoz
		}

		if (ACC[0] > ACC[1])
			ACC[1] 						= ACC[0]; //x1_global_max

		ACC[0] = 0.0;
		for (int k = 0; k < NumberOfMolecules; k++)
			ACC[2+k]					= 0.0;
	}
}

#endif