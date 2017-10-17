#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Thread block size
// KERNEL_SIZE must be an odd number
#define KERNEL_SIZE 5
#define KS_DIV_2 (KERNEL_SIZE >> 1)
#define TILE_SIZE 12
#define BLOCK_SIZE (TILE_SIZE + KERNEL_SIZE - 1)

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

#endif // _MATRIXMUL_H_

