
#ifndef _MULTIPLY_H_
#define _MULTIPLY_H_

#include <cuda.h>
#include <cuda_runtime.h>

const int block_size = 16;

__global__ void matrixMul(double *C, double *A, double *B, int Matrix_A_width, int Matrix_B_width);

__global__ void matrixMultWithNumber(double *a, double *b, const double v, int memsize);

#endif

