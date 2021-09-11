#ifndef PARAMETERS_H
#define PARAMETERS_H

class Parameters{
public:
    double p_A_1;
    double p_A_2;
    double f_1;
    double f_2;
    double theta; 
    double R_E;

    const int K             = NumberOfMolecules;
    const int N             = 5;
    const int I             = NumberOfReactions;

    double P_inf            = 1.0e5;
    double T_inf            = 300.0;

    double ro_L             = 998.2;
    double c_L              = 1483.0;
    double sigma            = 71.97e-3;
    double mu_L             = 1.0e-3;

    double R                = 8.31446e7;
    double R_SI             = R * 1.0e-7;
    double R_c              = 1.987;

    vector<double> lambda   = {0.0,     0.0,    0.1805, 0.0,    0.02658,    0.016,  0.0,    0.5863, 0.0,    0.0};

    vector<double> W        = {16.0,    1.0,    2.0,    17.0,   32.0,       18.0,   33.0,   34.0,   48.0,   17.0};

    double R_v              = R_SI / W[5] * 1.0e3;
    double p_v_sat          = 2338.1;
    double alfa_M           = 0.35;

    double p_E;
    double V_E;
    double T_E;
    double n_t_E;
    double M_Eq;
    vector<double> C;

    Parameters();
    Parameters(double, double, double);
    Parameters(double, double, double, double, double, double);
};

Parameters::Parameters()
{
    p_A_2                   = 0.0;
    f_2                     = 1.0;
    theta                   = 0.0;
}

Parameters::Parameters(double p_A_1_in, double f_1_in, double R_E_in)
{
    p_A_1                   = p_A_1_in;
    f_1                     = f_1_in;
    R_E                     = R_E_in;

    p_A_2                   = 0.0;
    f_2                     = 1.0;
    theta                   = 0.0;
}

Parameters::Parameters(double p_A_1_in, double p_A_2_in, double f_1_in, double f_2_in, double theta_in, double R_E_in)
{
    p_A_1                   = p_A_1_in;
    p_A_2                   = p_A_2_in;
    f_1                     = f_1_in;
    f_2                     = f_2_in;
    R_E                     = R_E_in;
    theta                   = theta_in;
}

#endif