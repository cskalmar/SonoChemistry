#ifndef FUNCTIONS_H
#define FUNCTIONS_H

template <class Precision>
__forceinline__ __device__ Precision sum(Precision* vec, const int& length)
{
	Precision res = 0.0;
	for (int i = 0; i < length; i++)
	{
		res += vec[i];
	}
	return res;
}

template <class Precision>
__forceinline__ __device__ int sum(int* vec, const int& length)
{
	int res = 0;
	for (int i = 0; i < length; i++)
	{
		res += vec[i];
	}
	return res;
}

template <class Precision>
__forceinline__ __device__ Precision prod(Precision* vec, const int& length)
{
	Precision res = 1.0;
	for (int i = 0; i < length; i++)
	{
		res *= vec[i];
	}
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
	for (int i = 0; i < length; i++)
	{
		res += vec1[i] * vec2[i];
	}
	return res;
}

template <class Precision>
__forceinline__ __device__ Precision SumCoeffProd(Precision* vec1, int* vec2, const int& length)
{
	Precision res = 0.0;
	for (int i = 0; i < length; i++)
	{
		res += vec1[i] * vec2[i];
	}
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
__forceinline__ __device__ void CalculateThermoDynamics(Precision* C_v, Precision* C_p, Precision* H, Precision* S_0, const Precision& T, const Precision& R)
{
	Precision TempRangeLow	 	= 200.0;
	Precision TempRangeHigh		= 6000.0;
	Precision TempRangeMid	 	= 1000.0;

	Precision Temp;
	int RowOffset;
	if (T < TempRangeLow){
		Temp = TempRangeLow;
		RowOffset = 1;}
	else if (T > TempRangeHigh){
		Temp = TempRangeHigh;
		RowOffset = 0;}
	else if (T <= TempRangeMid){
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
	Precision a_tmp[7];

	for (int k = 0; k < NumberOfMolecules; k++)
	{
		for (int i = 0; i < 7; i++)
			a_tmp[i]	= const_a[(2*k+RowOffset) * 7 + i];

		C_p[k] 			= R * 		 (	a_tmp[0]  		  +	a_tmp[1] * Temp  		  +	a_tmp[2] * Temp * Temp   			  	+ a_tmp[3] * Temp * Temp * Temp 			 + a_tmp[4] * Temp * Temp * Temp * Temp		  );
		H[k] 			= R * Temp * (	a_tmp[0]    	  +	a_tmp[1] * Temp * 0.5 	  +	a_tmp[2] * Temp * Temp * 0.33333333   	+ a_tmp[3] * Temp * Temp * Temp * 0.25	 	 + a_tmp[4] * Temp * Temp * Temp * Temp * 0.2 ) + R * a_tmp[5];
		S_0[k] 			= R * 		 (	a_tmp[0] * lnT    +	a_tmp[1] * Temp 		  + a_tmp[2] * Temp * Temp * 0.5  			+ a_tmp[3] * Temp * Temp * Temp * 0.33333333 + a_tmp[4] * Temp * Temp * Temp * Temp * 0.25  + a_tmp[6]);
		C_v[k]			= C_p[k] - R;
	}
}

template <class Precision>
__forceinline__ __device__ Precision ThermalConduction(const Precision& R, const Precision& R_dot, Precision* x, Precision* sPAR, const Precision& Temp, const Precision& c_p, const Precision& rho)
{
	Precision lambda_mean	= x[2] * sPAR[13] + x[4] * sPAR[14] +  x[5] * sPAR[15] +  x[7] * sPAR[16];
	Precision khi			= 10.0 * lambda_mean / rho / c_p;
	Precision l_th			= fmin(sqrt(fabs(R * khi / R_dot)), R / PI);
	Precision Heat 			= lambda_mean * (sPAR[0] - Temp) / l_th;
//	Heat					= 0.0;
	return Heat;
}

template <class Precision>
__forceinline__ __device__ Precision Evaporation(Precision* sPAR, const Precision& Temp, const Precision& p, Precision* x)
{
	Precision m_eva_kg 		= sPAR[17] * rsqrt(2.0 * PI * sPAR[18] * sPAR[0]) 	* sPAR[19];
	Precision m_con_kg 		= sPAR[17] * rsqrt(2.0 * PI * sPAR[18] * Temp) 		* x[5] * p;
	Precision m_net_mol 	= 1.0e3 / sPAR[6] * (m_eva_kg - m_con_kg);
//	m_net_mol				= 0.0;
	return m_net_mol;
}

template <class Precision>
__forceinline__ __device__ Precision BackwardRate(Precision* sPAR, Precision* S_0, Precision* H_0, const Precision& Temp, Precision& k_f, const int& i)
{
	Precision DeltaS_0 	= SumCoeffProd(S_0, &const_ReactionMatrix[i * NumberOfMolecules], NumberOfMolecules);
	Precision DeltaH_0 	= SumCoeffProd(H_0, &const_ReactionMatrix[i * NumberOfMolecules], NumberOfMolecules);
	Precision K_p		= exp(DeltaS_0 / sPAR[11] - DeltaH_0 / sPAR[11] / Temp);
	Precision K_c		= K_p * pow( sPAR[12] * 10.0 / sPAR[11] / Temp , sum(&const_ReactionMatrix[i * NumberOfMolecules], NumberOfMolecules) );
	return k_f / K_c;
}

template <class Precision>
__forceinline__ __device__ Precision PressureDependentReaction(Precision* sPAR, Precision* X_conc, const Precision& Temp, const int& i)
{
	Precision A_0, b_0, E_0;
	Precision alfa;
	Precision T_1_, T_3_, T_2; // -1.0 / value for T_1 and T_3
	switch (i)
	{
		case 4:
			A_0			= 5.2669e19;
			b_0			= -1.3737;
			E_0			= 0.0;
			alfa		= 0.67;
			T_3_		= -1.0e30;
			T_1_		= -1.0e-30;
			T_2			= 1.0e+30;
			break;
		case 13:
			A_0			= 1.9928e18;
			b_0			= -1.178;
			E_0			= -5.2382e3;
			alfa		= 0.43;
			T_3_		= -1.0e30;
			T_1_		= -1.0e-30;
			T_2			= 1.0e+30;
			break;
		case 21:
			A_0			= 2.275e28;
			b_0			= -4.37;
			E_0			= 27297.0;
			alfa		= 0.6417;
			T_3_		= -2557.544757;
			T_1_		= -0.000115198;
			T_2			= 6.0608e3;
			break;
	}

	Precision exponent	= -1.0 / sPAR[20] / Temp;
	Precision k_inf 	= const_A[i] * pow(Temp, const_b[i]) * exp(const_E[i] * exponent);
	Precision k_0		= A_0 * pow(Temp, b_0) * exp(E_0 * exponent);
	Precision M_corr	= SumCoeffProd(&const_ThirdBodyMatrix[i * NumberOfMolecules], X_conc, NumberOfMolecules);
	Precision P_r		= k_0 / k_inf * M_corr;

//	Precision F_cent	= alfa;
	Precision F_cent	= (1.0 - alfa) * exp(Temp * T_3_) + alfa * exp(Temp * T_1_) + exp(-T_2 / Temp);

	Precision d			= 0.14;
	Precision log10F_c	= log10(F_cent);
	Precision n			= 0.75 - 1.27 * log10F_c;
	Precision c			= -0.4 - 0.67 * log10F_c;
	Precision log10Pr	= log10(P_r);
	Precision logF		= log10F_c / ( 1.0 + pow( (log10Pr + c) / (n - d * (log10Pr + c)), 2.0 ) );
	Precision F 		= pow(10.0, logF);

	Precision k_f		= k_inf * P_r / (1.0 + P_r) * F;
	return k_f;
}

template <class Precision>
__forceinline__ __device__ void Reactions(Precision* omega, const Precision& Temp, Precision* X_conc, Precision* sPAR, Precision* S_0, Precision* H_0)
{

	Precision q[NumberOfReactions];
	for (int i = 0; i < NumberOfReactions; i++)
		q[i] = 0.0;

	Precision k_f, k_b;

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

		q[i] = k_f * ProdCoeffPow(X_conc, &const_ReactionMatrix_forward[i * NumberOfMolecules], NumberOfMolecules) - k_b * ProdCoeffPow(X_conc, &const_ReactionMatrix_backward[i * NumberOfMolecules], NumberOfMolecules);

		if ( (i == 0) || (i == 1) || (i == 2) || (i == 4) || (i == 10) || (i == 13) || (i == 21) || (i == 28) ) // Third body reactions
			q[i] *= SumCoeffProd(&const_ThirdBodyMatrix[i * NumberOfMolecules], X_conc, NumberOfMolecules);
	}

	for (int k = 0; k < NumberOfMolecules; k++)
	{
		omega[k] = 0.0;
		for (int i = 0; i < NumberOfReactions; i++)
			omega[k] += q[i] * const_ReactionMatrix[i * NumberOfMolecules + k];
	}

}

#endif