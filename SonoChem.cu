#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
//#include <eigen/Eigen/Dense>

using namespace std;
//using namespace Eigen;

#define PI 3.14159265358979323846
#define NumberOfMolecules 10
#define NumberOfReactions 34
#include "Functions.cuh"
#include "SonoChem_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"
#include "Parameters.cuh"
#include "InitialConditions.cuh"

// Solver Configuration
#define SOLVER RKCK45     // RK4, RKCK45
#define PRECISION double  // float, double
const int NT   = 1; 	 // NumberOfThreads
const int SD   = NumberOfMolecules + 3 + 1;     // SystemDimension
const int NCP  = 17;     // NumberOfControlParameters
const int NSP  = 23;     // NumberOfSharedParameters
const int NISP = 0;      // NumberOfIntegerSharedParameters
const int NE   = 0;      // NumberOfEvents
const int NA   = 3 * NumberOfMolecules + 3;     // NumberOfAccessories
const int NIA  = 2 * NumberOfMolecules + 1;     // NumberOfIntegerAccessories
const int NDO  = 80000;   // NumberOfPointsOfDenseOutput

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const Parameters&, const vector<double>&);

int main()
{
	int NumberOfProblems = 1;
	int BlockSize        = 1;
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> Solver_SC(SelectedDevice);
	
	Solver_SC.SolverOption(PreferSharedMemory, 1);
	Solver_SC.SolverOption(ThreadsPerBlock, BlockSize);
	Solver_SC.SolverOption(InitialTimeStep, 1.0e-2);
	Solver_SC.SolverOption(ActiveNumberOfThreads, NT);
	
	Solver_SC.SolverOption(DenseOutputMinimumTimeStep, 0.0);
	Solver_SC.SolverOption(DenseOutputSaveFrequency, 1);
	
	Solver_SC.SolverOption(MaximumTimeStep, 1.0e3);
	Solver_SC.SolverOption(MinimumTimeStep, 1.0e-14);
	Solver_SC.SolverOption(TimeStepGrowLimit, 10.0);
	Solver_SC.SolverOption(TimeStepShrinkLimit, 0.2);
	
	for (int i = 0; i < SD; i++)
	{
		Solver_SC.SolverOption(RelativeTolerance, i, 1e-8);
		Solver_SC.SolverOption(AbsoluteTolerance, i, 1e-8);
	}

	Parameters par(1.5e5, 80.0e3, 6.0e-6); //p_A_1, f_1, R_E
	vector<double> IC(SD, 0.0);
	SetInitialConditions(par, IC);
	FillSolverObject(Solver_SC, par, IC);

	clock_t SimulationStart = clock();
	
	Solver_SC.SynchroniseFromHostToDevice(All);
	Solver_SC.InsertSynchronisationPoint();
	Solver_SC.SynchroniseSolver();
		
	int TransientSimulations = 3;
	int ConvergentSimulations = 0;
	for (int i = 0; i < TransientSimulations; i++)
	{
		cout << "Transient: " << i+1 << " / " << TransientSimulations << endl;
		Solver_SC.Solve();
		Solver_SC.InsertSynchronisationPoint();
		Solver_SC.SynchroniseSolver();
	}
	Solver_SC.SynchroniseFromDeviceToHost(All);
	
	clock_t TransientEnd = clock();
	cout << "Transient simulation time: " << 1.0*(TransientEnd-SimulationStart) / CLOCKS_PER_SEC << " s" << endl << endl;

	Solver_SC.SynchroniseFromHostToDevice(All);
	for (int i = 0; i < ConvergentSimulations; i++)
	{
		cout << "Converged: " << i+1 << " / " << ConvergentSimulations << endl;
		Solver_SC.Solve();
		Solver_SC.InsertSynchronisationPoint();
		Solver_SC.SynchroniseSolver();
	}
	
	Solver_SC.SynchroniseFromDeviceToHost(All);
	clock_t ConvergedEnd = clock();
	cout << "Converged simulation time: " << 1.0*(ConvergedEnd-TransientEnd) / CLOCKS_PER_SEC << " s" << endl << endl;
	cout << "Total simulation time: " << 1.0*(ConvergedEnd-SimulationStart) / CLOCKS_PER_SEC << " s" << endl << endl;

	Solver_SC.Print(DenseOutput, 0);
	
	cout << "Test finished!" << endl << endl;

	int anyag = 3;

/*	cout << "Pi_w:\t" << Solver_SC.GetHost<PRECISION>(0, ActualState, NumberOfMolecules+3) << endl << endl;
	for (int i = 0; i < 2; i++)
		cout << i << "\t" << Solver_SC.GetHost<PRECISION>(0, Accessories, i) << endl;
	cout << endl;
	for (int i = 2; i < 12; i++)
		cout << i << "\t" << Solver_SC.GetHost<PRECISION>(0, Accessories, i) << endl;
	cout << endl;
	for (int i = 12; i < 22; i++)
		cout << i << "\t" << Solver_SC.GetHost<PRECISION>(0, Accessories, i) << endl;
	cout << endl;
	for (int i = 22; i < 32; i++)
		cout << i << "\t" << Solver_SC.GetHost<PRECISION>(0, Accessories, i) << endl;
	cout << endl;
	for (int i = 32; i < NA; i++)
		cout << i << "\t" << Solver_SC.GetHost<PRECISION>(0, Accessories, i) << endl;
	cout << endl;*/
}

// AUXILIARY FUNCTION -----------------------------------------------------------------------------

void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const Parameters& par, const vector<double>& IC)
{
	Solver.SetHost(0, TimeDomain,  0, 0.0 );
	Solver.SetHost(0, TimeDomain,  1, 1.0 );
	
	for (int i = 0; i < IC.size(); i++) 
		Solver.SetHost(0, ActualState, i, IC[i] );
		
	Solver.SetHost(0, ActualTime, 0.0 );
		
	for (int i = 0; i < par.C.size(); i++)
		Solver.SetHost(0, ControlParameters, i, par.C[i] );

	Solver.SetHost(0, DenseIndex, 0 );

	for (int i = 0; i < 33; i++)
		Solver.SetHost(0, Accessories, i, 0.0);
	
	Solver.SetHost(0, IntegerAccessories, 0, 1);
	for (int i = 1; i < 2 * NumberOfMolecules + 1; i++)
		Solver.SetHost(0, IntegerAccessories, i, 1);

	Solver.SetHost(SharedParameters, 0, par.T_inf );
	for (int i = 0; i < par.K; i++)
		Solver.SetHost(SharedParameters, i+1, par.W[i] );
	Solver.SetHost(SharedParameters, 11, par.R );
	Solver.SetHost(SharedParameters, 12, par.P_inf );
	Solver.SetHost(SharedParameters, 13, par.lambda[2] );
	Solver.SetHost(SharedParameters, 14, par.lambda[4] );
	Solver.SetHost(SharedParameters, 15, par.lambda[5] );
	Solver.SetHost(SharedParameters, 16, par.lambda[7] );
	Solver.SetHost(SharedParameters, 17, par.alfa_M );
	Solver.SetHost(SharedParameters, 18, par.R_v );
	Solver.SetHost(SharedParameters, 19, par.p_v_sat );
	Solver.SetHost(SharedParameters, 20, par.R_c );
	Solver.SetHost(SharedParameters, 21, par.c_L );
	Solver.SetHost(SharedParameters, 22, par.ro_L );
}

void Linspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    PRECISION Increment;
	
	x[0]   = B;
	
	if ( N>1 )
	{
		x[N-1] = E;
		Increment = (E-B)/(N-1);
		
		for (int i=1; i<N-1; i++)
		{
			x[i] = B + i*Increment;
		}
	}
}