
NVCC        = nvcc

NVCC_FLAGS  = --ptxas-options=-v -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = 2Dconvolution
OBJ	        = 2Dconvolution_cu.o 2Dconvolution_cpp.o

default: $(EXE)

2Dconvolution_cu.o: 2Dconvolution.cu 2Dconvolution.h  2Dconvolution_kernel.cu
	$(NVCC) -c -o $@ 2Dconvolution.cu $(NVCC_FLAGS)

2Dconvolution_cpp.o: 2Dconvolution_gold.cpp 2Dconvolution.h
	$(NVCC) -c -o $@ 2Dconvolution_gold.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
