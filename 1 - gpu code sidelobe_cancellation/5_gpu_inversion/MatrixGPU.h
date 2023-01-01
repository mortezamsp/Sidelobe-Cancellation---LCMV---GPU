
#ifndef _MATRIXGPU_H_
#define _MATRIXGPU_H_

#include <cuda.h>
#include <cuda_runtime.h>

const int block_size = 16;

__global__ void matrixMul(double *C, double *A, double *B, int Matrix_A_width, int Matrix_B_width);
__global__ void matrixMultWithNumber(double *a, double *b, const double v, int memsize);

__global__ void MatrixAdd(double *a, double *b, int memsize);
__global__ void MatrixAddWithNumber(double *a, const double b, int memsize);

__global__ void GEStep1A(double * AI, int i, int n2, int lda2);
__global__ void GEStep2(double * AI,double diag,int i, int n2, int lda2);
__global__ void GEStep3(double * AI,int i, int n2, int lda2);
void invertge(double * AI_d, int lda, int n);

#endif

