INCL_DIR = -I$(HOME)/MPGOS/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -w -maxrregcount=255

all: SonoChem_KCs.exe

# SonoChem.exe: SonoChem_BookChapterSimulations.cu
# 	nvcc -o	SonoChem_KCs.exe SonoChem_BookChapterSimulations.cu $(INCL_DIR) $(CMPL_OPT)

# SonoChem.exe: SonoChem.cu
# 	nvcc -o	SonoChem_KCs.exe SonoChem.cu $(INCL_DIR) $(CMPL_OPT)

SonoChem_KCs.exe: SonoChem_ParameterStudy.cu
	nvcc -o	SonoChem_KCs.exe SonoChem_ParameterStudy.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f SonoChem_KCs.exe