#include <stdlib.h>
#include "2Dconvolution.h"

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float*, const float*, const float*, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A convolved with B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param kernel_size         height and width of matrix A
//! @param hB         height of matrices B and C
//! @param wB         width of matrices B and C
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float* B, unsigned int hB, unsigned int wB)
{	
	// For each element in the result matrix matrix
	for (unsigned int i = 0; i < hB; ++i){
        for (unsigned int j = 0; j < wB; ++j) {
			float sum = 0;
			// check the start and end values of m and n to prevent overrunning the 
			//  matrix edges
			unsigned int mbegin = (i < KS_DIV_2)? KS_DIV_2 - i : 0;
			unsigned int mend = (i > (hB - (KS_DIV_2+1)))?
									hB - i + KS_DIV_2 : KERNEL_SIZE;
			unsigned int nbegin = (j < KS_DIV_2)? KS_DIV_2 - j : 0;
			unsigned int nend = (j > (wB - (KS_DIV_2+1)))?
									(wB-j) + KS_DIV_2 : KERNEL_SIZE;
			// overlay A over B centered at element (i,j).  For each 
			//  overlapping element, multiply the two and accumulate
			for(unsigned int m = mbegin; m < mend; ++m) {
				for(unsigned int n = nbegin; n < nend; n++) {
				  sum += A[m * KERNEL_SIZE + n] * 
							B[wB*(i + m - KS_DIV_2) + (j+n - KS_DIV_2)];
				}
			}
			// store the result
			C[i*wB + j] = (float)sum;
        }
	}
}
