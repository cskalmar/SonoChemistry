#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

// asdf

using namespace std;

#define PI 3.14159265358979323846
#define NumberOfMolecules 10
#define NumberOfReactions 34
#include "Functions.cuh"
#include "SonoChem_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"
#include "Parameters.cuh"
#include "InitialConditions.cuh"

//-----------------------------------------------------------------------
// Problem definition

const int NumberOf_PA         = 2;
const int NumberOf_f          = 2;
const int NumberOf_RE         = 1;

//-----------------------------------------------------------------------
// Solver Configuration
#define SOLVER RKCK45     // RK4, RKCK45
#define PRECISION double  // float, double
const int NT   = NumberOf_PA * NumberOf_f; 	 // NumberOfThreads
const int SD   = NumberOfMolecules + 3 + 1;     // SystemDimension
const int NCP  = 17;     // NumberOfControlParameters
const int NSP  = 23;     // NumberOfSharedParameters
const int NISP = 0;      // NumberOfIntegerSharedParameters
const int NE   = 0;      // NumberOfEvents
const int NA   = 64;     // NumberOfAccessories
const int NIA  = 11;     // NumberOfIntegerAccessories
const int NDO  = 0;   // NumberOfPointsOfDenseOutput

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const PRECISION&);

int main()
{
//-----------------------------------------------------------------------
//  GPU configuration

	int BlockSize        = 32;

	ListCUDADevices();

	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);

	PrintPropertiesOfSpecificDevice(SelectedDevice);

//-----------------------------------------------------------------------
//  Solver object configuration

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

//-----------------------------------------------------------------------
//  Simulations
	vector<PRECISION> PA_vec(NumberOf_PA, 0.0);
	//	Linspace(PA_vec, 1.72549e5, 0.0e5, NumberOf_PA);
		// Linspace(PA_vec, 2.0e5, 1.890196e5, NumberOf_PA);
		Linspace(PA_vec, 2.0e5, 1.890196e5, NumberOf_PA);
	vector<PRECISION> f_vec(NumberOf_f, 0.0);
	//	Logspace(f_vec, 89.943e3, 1000.0e3, NumberOf_f);
		Logspace(f_vec, 20.0e3, 1000.0e3, NumberOf_f);
	vector<PRECISION> RE_vec(NumberOf_RE, 0.0);
		Linspace(RE_vec, 8.0e-6, 2.0e-6, NumberOf_RE);

	vector< vector<PRECISION> > CollectedData;
	CollectedData.resize( NT , vector<PRECISION>( 3 + 2 * NumberOfMolecules + 4 * NumberOfMolecules + 2 , 0.0 ) );

    clock_t SimulationStart, TransientEnd, ConvergedEnd;
	SimulationStart = clock();

    for (int LaunchCounter = 0; LaunchCounter < RE_vec.size(); LaunchCounter++)
    {
		cout << "Filling solver object for R_E = " << RE_vec[LaunchCounter] * 1.0e6 << " mum..." << endl;
		FillSolverObject(Solver_SC, PA_vec, f_vec, RE_vec[LaunchCounter]);
		cout << "Solver object filled successfully." << endl << endl;
		Solver_SC.SynchroniseFromHostToDevice(All);

		for (int tid = 0; tid < NT; tid++)
		{
			CollectedData[tid][0] = Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 6) / 1.0e5;
			CollectedData[tid][1] = Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 14) / 1.0e3;
			CollectedData[tid][2] = Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 15) * 1.0e6;

			//cout << Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 6) / 1.0e5 << "  ";
		}

    	int TransientSimulations = 8;
    	int ConvergentSimulations = 4;

		cout << "Simulation started with R_E = " << RE_vec[LaunchCounter] * 1.0e6 << " mum." << endl << endl;

		cout << "Transient simulation started. Out of " << TransientSimulations << ":" << endl;
    	for (int i = 0; i < TransientSimulations; i++)
    	{
    		Solver_SC.Solve();
    		Solver_SC.InsertSynchronisationPoint();
    		Solver_SC.SynchroniseSolver();
			cout << i+1 << " ";
    	}
		TransientEnd = clock();
		cout << endl << "Transient finished." << endl;
		cout << "Transient simulation time: " << 1.0*(TransientEnd-SimulationStart) / CLOCKS_PER_SEC << " s." << endl << endl;

		Solver_SC.SynchroniseFromDeviceToHost(All);
		for (int tid = 0; tid < NT; tid++)
		{
			Solver_SC.SetHost(tid, IntegerAccessories, 0, 0);
			for (int col = 3; col < 13; col++)
				CollectedData[tid][col] = Solver_SC.GetHost<PRECISION>(tid, Accessories, col+50); //yield_i/Pi_j global max
			for (int col = 13; col < 23; col++)
				CollectedData[tid][col] = Solver_SC.GetHost<PRECISION>(tid, IntegerAccessories, col-12); //trans. num.
		}
		Solver_SC.SynchroniseFromHostToDevice(All);

		cout << "Convergent simulation started. Out of " << ConvergentSimulations << ":" << endl;
    	for (int i = 0; i < ConvergentSimulations; i++)
    	{
    		Solver_SC.Solve();
    		Solver_SC.InsertSynchronisationPoint();
    		Solver_SC.SynchroniseSolver();
			cout << i+1 << " " ;
    	}

    	Solver_SC.SynchroniseFromDeviceToHost(All);
		for (int tid = 0; tid < NT; tid++)
		{
			for (int col = 23; col < 33; col++)
				CollectedData[tid][col] = Solver_SC.GetHost<PRECISION>(tid, Accessories, col-11); // Yield_i global max

			for (int col = 33; col < 43; col++)
				CollectedData[tid][col] = Solver_SC.GetHost<PRECISION>(tid, Accessories, col-11) / ConvergentSimulations; // Yield_i avg

			for (int col = 43; col < 53; col++)
				CollectedData[tid][col] = Solver_SC.GetHost<PRECISION>(tid, Accessories, col-11); // Yield_i_local/Pi_w global max

			for (int col = 53; col < 63; col++)
				CollectedData[tid][col] = Solver_SC.GetHost<PRECISION>(tid, Accessories, col-11) / ConvergentSimulations; // Yield_i_local/Pi_w avg

			CollectedData[tid][63] 		= Solver_SC.GetHost<PRECISION>(tid, Accessories, 1);
			CollectedData[tid][64] 		= Solver_SC.GetHost<PRECISION>(tid, Accessories, 63);
		}

	    ConvergedEnd = clock();
		cout << endl << "Convergent finished." << endl;
	    cout << "Converged simulation time: " << 1.0*(ConvergedEnd-TransientEnd) / CLOCKS_PER_SEC << " s." << endl << endl;

		stringstream StreamFilename;
		StreamFilename.precision(2);
		StreamFilename.setf(ios::fixed);
		StreamFilename << "SC_results_RE_" << RE_vec[LaunchCounter] * 1.0e6 << "_plus.txt";

		string Filename = StreamFilename.str();
		remove( Filename.c_str() );

		ofstream DataFile;
		DataFile.open ( Filename.c_str(), std::fstream::app );
		int Width = 18;
		DataFile.precision(10);
		DataFile.flags(ios::scientific);

		for (int tid = 0; tid < NT; tid++)
			{
				for (int col = 0; col < 65; col++)
				{
					if ( col < (65-1) )
					{
						DataFile.width(Width); DataFile << CollectedData[tid][col] << ',';
					} else
					{
						DataFile.width(Width); DataFile << CollectedData[tid][col];
					}
				}
				DataFile << '\n';
			}

		DataFile.close();

		cout << "Simulation finished for R_E = " << RE_vec[LaunchCounter] * 1.0e6 << " mum. Filename: " << Filename << endl << endl;
    }

    cout << "Total simulation time: " << 1.0*(ConvergedEnd-SimulationStart) / CLOCKS_PER_SEC << " s" << endl << endl;

	cout << "Test finished!" << endl << endl;
}

// AUXILIARY FUNCTION -----------------------------------------------------------------------------

void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& PA_vec, const vector<PRECISION>& f_vec, const PRECISION& R_E)
{
	int ProblemNumber = 0;

	for (auto const& PA_act: PA_vec)
	{
		for (auto const& f_act: f_vec)
		{
			Parameters par(PA_act, f_act, R_E);
			vector<PRECISION> IC(par.K + 3 + 1, 0.0);
			SetInitialConditions(par, IC);

			for (int i = 0; i < IC.size(); i++)
				Solver.SetHost(ProblemNumber, ActualState, i, IC[i] );

			for (int i = 0; i < par.C.size(); i++)
				Solver.SetHost(ProblemNumber, ControlParameters, i, par.C[i] );

			Solver.SetHost(ProblemNumber, TimeDomain, 0, 0.0 );
			Solver.SetHost(ProblemNumber, TimeDomain, 1, 1.0 );
			Solver.SetHost(ProblemNumber, ActualTime, 0.0 );

			for (int i = 0; i < 64; i++)
				Solver.SetHost(ProblemNumber, Accessories, i, 0.0);

			Solver.SetHost(ProblemNumber, IntegerAccessories, 0, 1);
			for (int i = 1; i < 11; i++)
				Solver.SetHost(ProblemNumber, IntegerAccessories, i, 1);

			ProblemNumber++;
		}
	}

	Parameters par;
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

void Logspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    x[0] = B;

	if ( N>1 )
	{
		x[N-1] = E;
		PRECISION ExpB = log10(B);
		PRECISION ExpE = log10(E);
		PRECISION ExpIncr = (ExpE-ExpB)/(N-1);
		for (int i=1; i<N-1; i++)
		{
			x[i] = pow(10,ExpB + i*ExpIncr);
		}
	}
}