#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// Matrix multiplication kernel thread specification
__global__ void ConvolutionKernel(Matrix N, Matrix P)
{
    __shared__ float tileNs[BLOCK_SIZE][BLOCK_SIZE];
    // get thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // get the output indices
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;

    // shift to obtain input indices
    int row_i = row_o - KS_DIV_2;
    int col_i = col_o - KS_DIV_2;

    // Load tile elements
    if(row_i >= 0 && row_i < N.height && col_i >= 0 && col_i < N.width)
        tileNs[ty][tx] = N.elements[row_i*N.width + col_i];
    else
        tileNs[ty][tx] = 0.0f;

    // Wait until all tile elements are loaded
    __syncthreads();

    // only compute if you're an output tile element
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        for(int y=0; y<KERNEL_SIZE; y++)
            for(int x=0; x<KERNEL_SIZE; x++)
                pValue += Mc[y*KERNEL_SIZE + x] * tileNs[y+ty][x+tx];
        // only write values if you are inside matrix bounds
        if(row_o < P.height && col_o < P.width)
            P.elements[row_o*P.width + col_o] = pValue;
    }
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
