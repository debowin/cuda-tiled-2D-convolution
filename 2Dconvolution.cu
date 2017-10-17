/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "2Dconvolution.h"

// includes, kernels
__constant__ float Mc[KERNEL_SIZE * KERNEL_SIZE];

#include "2Dconvolution_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
int ReadParamsFile(int* params, char* file_name, int num_params);
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
bool CompareMatrices(Matrix A, Matrix B);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	
	srand(2012);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
		N  = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
		P  = AllocateMatrix(N.height, N.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = (int*) malloc(2*sizeof(int)); 
		unsigned int data_read = ReadParamsFile(params, argv[1], 2);
		if(data_read != 2){
			printf("Error reading parameter file\n");
			return 1;
		}

		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0);
		N  = AllocateMatrix(params[0], params[1], 0);		
		P  = AllocateMatrix(params[0], params[1], 0);
		free(params);
		(void)ReadFile(&M, argv[2]);
		(void)ReadFile(&N, argv[3]);
	}

	// M * N on the device
    ConvolutionOnDevice(M, N, P);
    
    // compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    computeGold(reference.elements, M.elements, N.elements, N.height, N.width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken by CPU Gold %lf milliseconds.\n", elapsedTime);
        
    // in this case check if the result is equivalent to the expected soluion
    bool res = CompareMatrices(reference, P);
    printf("Test %s\n", (res) ? "PASSED" : "FAILED");
    
    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}   

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    cudaEvent_t startBig, stopBig, start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startBig);
    cudaEventCreate(&stopBig);
    cudaEventRecord(startBig);
    // Load M and N to the device
    int size = M.width * M.height * sizeof(float);
    cudaMemcpyToSymbol(Mc, M.elements, size);
 
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);

    // Setup the execution configuration
    dim3 blockSize, gridSize;
    blockSize.x = BLOCK_SIZE, blockSize.y = BLOCK_SIZE, blockSize.z = 1;
    gridSize.x = ceil((float)P.width/TILE_SIZE),
    gridSize.y = ceil((float)P.height/TILE_SIZE),
    gridSize.z = 1;

    // Launch the device computation threads!
    cudaEventRecord(start);
    ConvolutionKernel<<<gridSize, blockSize>>>(Nd, Pd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelElapsedTime = 0;
    cudaEventElapsedTime(&kernelElapsedTime, start, stop);
    printf("Time taken by GPU Kernel %lf milliseconds.\n", kernelElapsedTime);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd);

    // Free device matrices
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);

    cudaEventRecord(stopBig);
    cudaEventSynchronize(stopBig);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, startBig, stopBig);
    printf("Overhead Time taken by GPU %lf milliseconds.\n", elapsedTime-kernelElapsedTime);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
    unsigned int data_read = M->width * M->height;
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < data_read; i++) 
        fscanf(input, "%f", &(M->elements[i]));
    return data_read;
}

// Read params of input matrices
int ReadParamsFile(int* params, char* file_name, int num_params)
{
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < num_params; i++) 
        fscanf(input, "%d", &(params[i]));
    return num_params;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    unsigned int size = M.width * M.height;
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < size; i++) {
        fprintf(output, "%f ", M.elements[i]);
    }
}

// returns true iff A and B have same elements in same order
bool CompareMatrices(Matrix A, Matrix B) {
    unsigned int size = A.width * A.height;

    if ( (A.width != B.width) || (A.height != B.height) )
        return false;

    for (unsigned i = 0; i < size; i++)
        if (abs(A.elements[i] - B.elements[i]) > 0.001f)
            return false;
    return true;
}

