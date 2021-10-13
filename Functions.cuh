#ifndef FUNCTIONS_H
#define FUNCTIONS_H

template <class Precision>
__forceinline__ __device__ Precision sum(Precision* vec, const int& length)
{
	Precision res = 0.0;
	for (int i = 0; i < length; i++){res += vec[i];}
	return res;
}

template <class Precision>
__forceinline__ __device__ int sum(int* vec, const int& length)
{
	int res = 0;
	for (int i = 0; i < length; i++){res += vec[i];}
	return res;
}

template <class Precision>
__forceinline__ __device__ Precision prod(Precision* vec, const int& length)
{
	Precision res = 1.0;
	for (int i = 0; i < length; i++){res *= vec[i];}
	return res;
}

/*
template <class Precision>
__forceinline__ __device__ Precision SumCoeffProd(Precision* vec1, Precision* vec2, int& length)
{
	Precision res = 0.0;
	for (int i = 0; i < length; i++)
	{
		res += vec1[i] * vec2[i];
	}
	return res;
}
*/

template <class Precision>
__forceinline__ __device__ Precision SumCoeffProd(Precision* vec1, Precision* vec2, const int& length)
{
	Precision res = 0.0;
	for (int i = 0; i < length; i++){res += vec1[i] * vec2[i];}
	return res;
}

template <class Precision>
__forceinline__ __device__ Precision SumCoeffProd(Precision* vec1, int* vec2, const int& length)
{
	Precision res = 0.0;
	for (int i = 0; i < length; i++){res += vec1[i] * vec2[i];}
	return res;
}

template <class Precision>
__forceinline__ __device__ Precision ProdCoeffPow(Precision* vec_base, Precision* vec_pow, const int& length)
{
	Precision res = 1.0;
	for (int i = 0; i < length; i++)
	{
		switch (vec_pow[i])
		{
			case 0.0:
				break;
			case 1.0:
				res *= vec_base[i];
				break;
			case 2.0:
				res *= vec_base[i] * vec_base[i];
				break;
			default:
				res *= pow(vec_base[i], vec_pow[i]);
		}
	}
	return res;
}

template <class Precision>
__forceinline__ __device__ Precision ProdCoeffPow(Precision* vec_base, int* vec_pow, const int& length)
{
	Precision res = 1.0;
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < vec_pow[i]; j++)
			res *= vec_base[i];
	}
	return res;
}

template <class Precision>
__forceinline__ __device__ void CalculateThermoDynamics(Precision& C_p, Precision* H, Precision* S_0, Precision* X_conc, const Precision& T, const Precision& R)
{
	Precision Temp;
	int RowOffset;
	if (T < const_TempRanges[0]){
		Temp = const_TempRanges[0];
		RowOffset = 1;}
	else if (T > const_TempRanges[1]){
		Temp = const_TempRanges[1];
		RowOffset = 0;}
	else if (T <= const_TempRanges[2]){
		Temp = T;
		RowOffset = 1;}
	else {
		Temp = T;
		RowOffset = 0;}

//	Temp				= (T < TempRangeLow) ? TempRangeLow : T;
//	Temp				= (Temp > TempRangeHigh) ? TempRangeHigh : Temp;
//	int RowOffset 		= (T <= TempRangeMid) ? 1 : 0;
//	int RowOffset 		= (Temp <= TempRangeMid) ? 1 : 0;

	Precision lnT 		= log(Temp);
	Precision Temp2		= Temp * Temp;
	Precision Temp3		= Temp * Temp * Temp;
	Precision Temp4		= Temp * Temp * Temp * Temp;
	Precision a_tmp[7];

	for (int k = 0; k < NumberOfMolecules; k++)
	{
		for (int i = 0; i < 7; i++)
			a_tmp[i]	= const_a[(2*k+RowOffset) * 7 + i];

		C_p 			+= R * 		 (	a_tmp[0]  		  +	a_tmp[1] * Temp  		  +	a_tmp[2] * Temp2   			  	+ a_tmp[3] * Temp3 				 + a_tmp[4] * Temp4		  ) * X_conc[k];
		H[k] 			= R * Temp * (	a_tmp[0]    	  +	a_tmp[1] * Temp * 0.5 	  +	a_tmp[2] * Temp2 * 0.33333333   + a_tmp[3] * Temp3 * 0.25	 	 + a_tmp[4] * Temp4 * 0.2 ) + R * a_tmp[5];
		S_0[k] 			= R * 		 (	a_tmp[0] * lnT    +	a_tmp[1] * Temp 		  + a_tmp[2] * Temp2 * 0.5  		+ a_tmp[3] * Temp3 * 0.33333333  + a_tmp[4] * Temp4 * 0.25  + a_tmp[6]);
	}
}

template <class Precision>
__forceinline__ __device__ Precision ThermalConduction(const Precision& R, const Precision& R_dot, const Precision& lambda_mean, Precision* sPAR, const Precision& Temp, const Precision& rho_c_p)
{
	Precision khi			= 10.0 * lambda_mean / rho_c_p;
	Precision l_th			= fmin(sqrt(fabs(R * khi / R_dot)), R / PI);
	return lambda_mean * (sPAR[0] - Temp) / l_th;
}

template <class Precision>
__forceinline__ __device__ Precision Evaporation(Precision* sPAR, const Precision& Temp, const Precision& x5p)
{
	Precision m_eva_kg 		= sPAR[17] * rsqrt(2.0 * PI * sPAR[18] * sPAR[0]) 	* sPAR[19];
	Precision m_con_kg 		= sPAR[17] * rsqrt(2.0 * PI * sPAR[18] * Temp) 		* x5p;
	Precision m_net_mol 	= 1.0e3 / sPAR[6] * (m_eva_kg - m_con_kg);
	return m_net_mol;
}

template <class Precision>
__forceinline__ __device__ Precision BackwardRate(Precision* sPAR, Precision* S_0, Precision* H_0, const Precision& Temp, Precision& k_f, const int& i)
{
	Precision DeltaS_0 	= SumCoeffProd(S_0, &const_ReactionMatrix[i * NumberOfMolecules], NumberOfMolecules);
	Precision DeltaH_0 	= SumCoeffProd(H_0, &const_ReactionMatrix[i * NumberOfMolecules], NumberOfMolecules);
	Precision r_Temp_R	= 1.0 / (sPAR[11] * Temp);
	Precision K_p		= exp(DeltaS_0  * r_Temp_R * Temp - DeltaH_0 * r_Temp_R);
	Precision K_c		= K_p * pow( sPAR[12] * 10.0  * r_Temp_R , sum(&const_ReactionMatrix[i * NumberOfMolecules], NumberOfMolecules) );
	return k_f / K_c;
}

template <class Precision>
__forceinline__ __device__ Precision PressureDependentReaction(Precision* sPAR, Precision* X_conc, const Precision& Temp, const int& i)
{
	// Precision A_0, b_0, E_0;
	// Precision alfa;
	// Precision T_1_, T_3_, T_2; // -1.0 / value for T_1 and T_3
	Precision rTemp		= 1.0 / Temp;
	Precision exponent	= -1.0 / sPAR[20] * rTemp;
	Precision k_0, F_cent;
	switch (i)
	{
		case 4:
			// A_0		= 5.2669e19; b_0 = -1.3737; E_0 = 0.0;
			// alfa		= 0.67;
			// T_3_		= -1.0e30;
			// T_1_		= -1.0e-30;
			// T_2			= 1.0e+30;

			k_0			= 5.2669e19 * pow(Temp, -1.3737);
			// F_cent		= (1.0 - alfa) * exp(Temp * T_3_) + alfa * exp(Temp * T_1_) + exp(-T_2 * rTemp);
			// F_cent		= (1.0 - 0.67) * exp(-Temp * 1.0e30) + 0.67 * exp(-Temp * 1.0e-30) + exp(-1.0e30 * rTemp);
			F_cent		= 0.67;
			break;
		case 13:
			// A_0		= 1.9928e18; b_0 = -1.178; E_0 = -5.2382e3;
			// alfa		= 0.43;
			// T_3_		= -1.0e30;
			// T_1_		= -1.0e-30;
			// T_2			= 1.0e+30;

			k_0			= 1.9928e18 * pow(Temp, -1.178) * exp(-5.2382e3 * exponent);
			// F_cent		= (1.0 - alfa) * exp(Temp * T_3_) + alfa * exp(Temp * T_1_) + exp(-T_2 * rTemp);
			// F_cent		= (1.0 - 0.43) * exp(-Temp * 1.0e30) + 0.43 * exp(-Temp * 1.0e-30) + exp(-1.0e+30 * rTemp);
			F_cent		= 0.43;
			break;
		case 21:
			// A_0		= 2.275e28; b_0	= -4.37; E_0 = 27297.0;
			// alfa		= 0.6417;
			// T_3_		= -2557.544757;
			// T_1_		= -0.000115198;
			// T_2			= 6.0608e3;

			k_0			= 2.275e28 * pow(Temp, -4.37) * exp(27297.0 * exponent);
			// F_cent		= (1.0 - alfa) * exp(Temp * T_3_) + alfa * exp(Temp * T_1_) + exp(-T_2 * rTemp);
			F_cent		= (1.0 - 0.6417) * exp(-Temp * 2557.544757) + 0.6417 * exp(-Temp * 0.000115198) + exp(-6.0608e3 * rTemp);
			break;
	}

	Precision k_inf 	= const_A[i] * pow(Temp, const_b[i]) * exp(const_E[i] * exponent);
	Precision M_corr	= SumCoeffProd(&const_ThirdBodyMatrix[i * NumberOfMolecules], X_conc, NumberOfMolecules);
	Precision P_r		= k_0 / k_inf * M_corr;

	// F_cent				= (1.0 - alfa) * exp(Temp * T_3_) + alfa * exp(Temp * T_1_) + exp(-T_2 * rTemp);

	Precision log10F_c	= log10(F_cent);
	Precision n			= 0.75 - 1.27 * log10F_c;
	// Precision c			= -0.4 - 0.67 * log10F_c;
	Precision log10Pr_plus_c	= log10(P_r) - 0.4 - 0.67 * log10F_c;
	Precision logF		= log10F_c / ( 1.0 + pow( log10Pr_plus_c / (n - 0.14 * log10Pr_plus_c), 2.0 ) );

	return k_inf * P_r / (1.0 + P_r) * pow(10.0, logF);
}

template <class Precision>
__forceinline__ __device__ void Reactions(Precision* omega, const Precision& Temp, Precision* X_conc, Precision* sPAR, Precision* S_0, Precision* H_0)
{
	Precision k_f, k_b, q_i;
	Precision exponent = -1.0 / (sPAR[20] * Temp);

	for (int i = 0; i < NumberOfReactions; i++)
	{
		if ( (i == 4) || (i == 13) || (i == 21) ) // Pressure dependent reactions
			k_f = PressureDependentReaction(sPAR, X_conc, Temp, i);
		else
		{
			k_f = const_A[i];
			if (const_b[i] != 0.0)
				k_f *= pow(Temp, const_b[i]);
			if (const_E[i] != 0.0)
				k_f *=  exp(const_E[i] * exponent);
		}
		k_b = BackwardRate(sPAR, S_0, H_0, Temp, k_f, i);

		q_i = k_f * ProdCoeffPow(X_conc, &const_ReactionMatrix_forward[i * NumberOfMolecules], NumberOfMolecules) - k_b * ProdCoeffPow(X_conc, &const_ReactionMatrix_backward[i * NumberOfMolecules], NumberOfMolecules);

		if ( (i == 0) || (i == 1) || (i == 2) || (i == 4) || (i == 10) || (i == 13) || (i == 21) || (i == 28) ) // Third body reactions
			q_i *= SumCoeffProd(&const_ThirdBodyMatrix[i * NumberOfMolecules], X_conc, NumberOfMolecules);

		for (int k = 0; k < NumberOfMolecules; k++)
			omega[k] += q_i * const_ReactionMatrix[i * NumberOfMolecules + k];
	}
}

#endif