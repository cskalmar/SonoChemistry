#ifndef INITIALCONDITIONS_H
#define INITIALCONDITIONS_H

void SetInitialConditions(Parameters& par, vector<double>& IC)
{
    par.p_E = par.P_inf + 2.0 * par.sigma / par.R_E;
    par.V_E = 4.0 / 3.0 * pow(par.R_E,3) * PI;
    par.T_E = par.T_inf;
    par.n_t_E = par.p_E * par.V_E / par.R_SI / par.T_E;
    par.M_Eq = par.n_t_E / (par.V_E * 1.0e6);

    IC[0] = 1.0;
    IC[1] = 0.0;
    IC[2] = 1.0;

    vector<double> X_0(par.K, 0.0);
    X_0[5] = par.p_v_sat / par.p_E;
    X_0[4] = 1.0 - X_0[5];
    for (int i = 0; i < par.K; i++)
        IC[i+3] = X_0[i] * par.n_t_E / (4.0 / 3.0 * pow(IC[0] * par.R_E, 3) * PI) * 1.0e-6 / par.M_Eq;

    par.C.resize(17);
    par.C[0] = par.ro_L * pow(par.R_E * par.f_1, 2);
    par.C[1] = par.R_E * par.f_1 * par.ro_L * par.c_L;
    par.C[2] = par.R_E * pow(par.f_1, 2) * par.ro_L * par.c_L;
    par.C[3] = par.R_E * par.f_1 / 3.0 / par.c_L;
    par.C[4] = par.R_E * par.f_1 / par.c_L;
    par.C[5] = 4.0 * par.mu_L / par.c_L / par.ro_L / par.R_E;
    par.C[6] = par.p_A_1;
    par.C[7] = par.p_A_2;
    par.C[8] = 2.0 * PI * par.f_2 / par.f_1;
    par.C[9] = par.theta;
    par.C[10] = par.f_1 * 2.0 * PI * par.p_A_1;
    par.C[11] = par.f_2 * 2.0 * PI * par.p_A_2;
    par.C[12] = 2.0 * par.sigma / par.R_E;
    par.C[13] = 4.0 * par.mu_L * par.f_1;
    par.C[14] = par.f_1;
    par.C[15] = par.R_E;
    par.C[16] = par.M_Eq;
}

#endif