#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;

#define PI 3.14159265358979323846
#define NumberOfMolecules 10
#define NumberOfReactions 34
#define NumberOfAccessories 64

//-----------------------------------------------------------------------
// Problem definition

const int NumberOf_PA         = 2; //TODO
const int NumberOf_f          = 2; //TODO
const int NumberOf_RE         = 1; //TODO

//-----------------------------------------------------------------------
// Solver Configuration
#define SOLVER RKCK45     // RK4, RKCK45
#define PRECISION double  // float, double
const int NT   = NumberOf_PA * NumberOf_f; 	 	// NumberOfThreads
const int SD   = NumberOfMolecules + 3 + 1;     // SystemDimension
const int NCP  = 17;     						// NumberOfControlParameters
const int NSP  = 9;     						// NumberOfSharedParameters
const int NISP = 0;     						// NumberOfIntegerSharedParameters
const int NE   = 0;      						// NumberOfEvents
const int NA   = NumberOfAccessories;     		// NumberOfAccessories
const int NIA  = 1;     						// NumberOfIntegerAccessories
const int NDO  = 0;   							// NumberOfPointsOfDenseOutput

//-----------------------------------------------------------------------
// Constant memory allocation
__constant__ PRECISION const_a[2*NumberOfMolecules*7];
__constant__ PRECISION const_ThirdBodyMatrix[NumberOfReactions * NumberOfMolecules];
__constant__ int const_ReactionMatrix_forward[NumberOfReactions * NumberOfMolecules];
__constant__ int const_ReactionMatrix_backward[NumberOfReactions * NumberOfMolecules];
__constant__ int const_ReactionMatrix[NumberOfReactions * NumberOfMolecules];
__constant__ PRECISION const_A[NumberOfReactions];
__constant__ PRECISION const_b[NumberOfReactions];
__constant__ PRECISION const_E[NumberOfReactions];
__constant__ PRECISION const_TempRanges[3*NumberOfMolecules];
__constant__ PRECISION const_lambda[NumberOfMolecules];
__constant__ PRECISION const_W[NumberOfMolecules];

//-----------------------------------------------------------------------
// Includes

#include "Functions.cuh"
#include "SonoChem_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"
#include "Parameters.cuh"
#include "InitialConditions.cuh"

//-----------------------------------------------------------------------

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
		Linspace(PA_vec, 1.8e5, 1.8e5, NumberOf_PA); //TODO
		// Linspace(PA_vec, 2.0e5, 1.890196e5, NumberOf_PA);
	vector<PRECISION> f_vec(NumberOf_f, 0.0);
	//	Logspace(f_vec, 89.943e3, 1000.0e3, NumberOf_f);
		Logspace(f_vec, 50.0e3, 50.0e3, NumberOf_f); //TODO
	vector<PRECISION> RE_vec(NumberOf_RE, 0.0);
		Linspace(RE_vec, 8.0e-6, 8.0e-6, NumberOf_RE); //TODO

	vector< vector<PRECISION> > CollectedData;
	CollectedData.resize( NT , vector<PRECISION>(51 , 0.0));

    clock_t SimulationStart, TransientEnd, ConvergedEnd;
	SimulationStart = clock();

    for (int LaunchCounter = 0; LaunchCounter < RE_vec.size(); LaunchCounter++)
    {
		cout << "Filling solver object for R_E = " << RE_vec[LaunchCounter] * 1.0e6 << " mum..." << endl;
		FillSolverObject(Solver_SC, PA_vec, f_vec, RE_vec[LaunchCounter]); //TODO: function
		cout << "Solver object filled successfully." << endl << endl;
		Solver_SC.SynchroniseFromHostToDevice(All);

		for (int tid = 0; tid < NT; tid++)
		{
			CollectedData[tid][0] = Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 6) / 1.0e5;    // p_A [bar]
			CollectedData[tid][1] = Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 14) / 1.0e3;   // f [kHz]
			CollectedData[tid][2] = Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 15) * 1.0e6;   // R_E [mum]
		}

    	int TransientSimulations = 1; //TODO
    	int ConvergentSimulations = 1;

		cout << "Simulation started with R_E = " << RE_vec[LaunchCounter] * 1.0e6 << " mum." << endl << endl;

		cout << "Transient simulation started." << endl;
    	for (int i = 0; i < TransientSimulations; i++)
    	{
    		Solver_SC.Solve();
    		Solver_SC.InsertSynchronisationPoint();
    		Solver_SC.SynchroniseSolver();
			cout << TransientSimulations - i << " ";
    	}
		TransientEnd = clock();
		cout << endl << "Transient finished." << endl;
		cout << "Transient simulation time: " << 1.0*(TransientEnd-SimulationStart) / CLOCKS_PER_SEC << " s." << endl << endl;

		Solver_SC.SynchroniseFromDeviceToHost(All);
		for (int tid = 0; tid < NT; tid++) Solver_SC.SetHost(tid, IntegerAccessories, 0, 1);
		Solver_SC.SynchroniseFromHostToDevice(All);

		cout << "Convergent simulation started." << endl;
    	for (int i = 0; i < ConvergentSimulations; i++)
    	{
    		Solver_SC.Solve();
    		Solver_SC.InsertSynchronisationPoint();
    		Solver_SC.SynchroniseSolver();
			cout << ConvergentSimulations - i << " ";
    	}

    	Solver_SC.SynchroniseFromDeviceToHost(All);
	    ConvergedEnd = clock();
		cout << endl << "Convergent finished." << endl;
	    cout << "Converged simulation time: " << 1.0*(ConvergedEnd-TransientEnd) / CLOCKS_PER_SEC << " s." << endl;

		double rCS	= 1.0 / ConvergentSimulations;
		for (int tid = 0; tid < NT; tid++)
		{
			CollectedData[tid][3] 	= Solver_SC.GetHost<PRECISION>(tid, Accessories, 0) - 1.0; // Relative expansion global max
			CollectedData[tid][4] 	= Solver_SC.GetHost<PRECISION>(tid, Accessories, 1) * rCS - 1.0; // Relative expansion global avg

			CollectedData[tid][5] 	= Solver_SC.GetHost<PRECISION>(tid, Accessories, 4); // Compression ratio global max
			CollectedData[tid][6] 	= Solver_SC.GetHost<PRECISION>(tid, Accessories, 5) * rCS; // Compression ratio global avg

			CollectedData[tid][7] 	= Solver_SC.GetHost<PRECISION>(tid, Accessories, 9); // Compression speed global max (dimensionless)
			// CollectedData[tid][7] = Solver_SC.GetHost<PRECISION>(tid, Accessories, 9) * pow((Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 15) * 1.0e6, 3.0)) * Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 14); // Compression speed global max (mum^3/s)
			CollectedData[tid][8]	= Solver_SC.GetHost<PRECISION>(tid, Accessories, 10) * rCS; // Compression speed global avg (dimensionless)
			// CollectedData[tid][8] = Solver_SC.GetHost<PRECISION>(tid, Accessories, 10) * rCS * pow((Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 15) * 1.0e6, 3.0)) * Solver_SC.GetHost<PRECISION>(tid, ControlParameters, 14); // Compression speed global avg (mum^3/s)

			CollectedData[tid][9] 	= Solver_SC.GetHost<PRECISION>(tid, Accessories, 12) 		* Solver_SC.GetHost<PRECISION>(SharedParameters, 0); // Max temperature global max (K)
			CollectedData[tid][10] 	= Solver_SC.GetHost<PRECISION>(tid, Accessories, 13) * rCS 	* Solver_SC.GetHost<PRECISION>(SharedParameters, 0); // Max temperature global avg (K)

			for (int k = 0; k < NumberOfMolecules; k++)
			{
				CollectedData[tid][k+11]	= Solver_SC.GetHost<PRECISION>(tid, Accessories, k+24); //Yield_k global max
				CollectedData[tid][k+21]	= Solver_SC.GetHost<PRECISION>(tid, Accessories, k+34) * rCS; //Yield_k global avg

				CollectedData[tid][k+31]	= Solver_SC.GetHost<PRECISION>(tid, Accessories, k+44); //Yield_k/Pi_w global max
				CollectedData[tid][k+41]	= Solver_SC.GetHost<PRECISION>(tid, Accessories, k+54) * rCS; //Yield_k/Pi_w global avg
			}
		}

		stringstream StreamFilename;
		StreamFilename.precision(2);
		StreamFilename.setf(ios::fixed);
		StreamFilename << "Results/RE_" << RE_vec[LaunchCounter] * 1.0e6 << ".txt";

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

	for (int i = 0; i < SD; i++)
	{
		cout << "x" << i << ":\t" << Solver_SC.GetHost<PRECISION>(0, ActualState, i) << endl;
	}
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

			for (int i = 0; i < NumberOfAccessories; i++)
				Solver.SetHost(ProblemNumber, Accessories, i, 0.0);

			Solver.SetHost(ProblemNumber, IntegerAccessories, 0, 0);

			ProblemNumber++;
		}
	}

	Parameters par;
	Solver.SetHost(SharedParameters, 0, par.T_inf );
	Solver.SetHost(SharedParameters, 1, par.R );
	Solver.SetHost(SharedParameters, 2, par.P_inf );
	Solver.SetHost(SharedParameters, 3, par.alfa_M );
	Solver.SetHost(SharedParameters, 4, par.R_v );
	Solver.SetHost(SharedParameters, 5, par.p_v_sat );
	Solver.SetHost(SharedParameters, 6, par.R_c );
	Solver.SetHost(SharedParameters, 7, par.c_L );
	Solver.SetHost(SharedParameters, 8, par.ro_L );

	cudaMemcpyToSymbol(const_a,                         &par.h_a,                       2*NumberOfMolecules*7 * sizeof(PRECISION));
	cudaMemcpyToSymbol(const_ThirdBodyMatrix,           &par.h_ThirdBodyMatrix,         NumberOfReactions * NumberOfMolecules * sizeof(PRECISION));
	cudaMemcpyToSymbol(const_ReactionMatrix_forward,    &par.h_ReactionMatrix_forward,  NumberOfReactions * NumberOfMolecules * sizeof(int));
	cudaMemcpyToSymbol(const_ReactionMatrix_backward,   &par.h_ReactionMatrix_backward, NumberOfReactions * NumberOfMolecules * sizeof(int));
	cudaMemcpyToSymbol(const_ReactionMatrix,            &par.h_ReactionMatrix,          NumberOfReactions * NumberOfMolecules * sizeof(int));
	cudaMemcpyToSymbol(const_A,                         &par.h_A,                       NumberOfReactions * sizeof(PRECISION));
	cudaMemcpyToSymbol(const_b,                         &par.h_b,                       NumberOfReactions * sizeof(PRECISION));
	cudaMemcpyToSymbol(const_E,                         &par.h_E,                       NumberOfReactions * sizeof(PRECISION));
	cudaMemcpyToSymbol(const_TempRanges,				&par.h_TempRanges,				3*NumberOfMolecules * sizeof(PRECISION));
	cudaMemcpyToSymbol(const_lambda,					&par.h_lambda,					NumberOfMolecules * sizeof(PRECISION));
	cudaMemcpyToSymbol(const_W,							&par.h_W,						NumberOfMolecules * sizeof(PRECISION));
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