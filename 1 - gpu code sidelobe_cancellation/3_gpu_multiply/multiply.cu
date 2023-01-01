
/*`copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include "multiply.h"


__global__ void matrixMul(double *C, double *A, double *B, int Matrix_A_width, int Matrix_B_width)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = Matrix_A_width * block_size * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + Matrix_A_width - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = block_size;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = block_size * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = block_size * Matrix_B_width;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    double Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ double As[block_size][block_size];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ double Bs[block_size][block_size];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + Matrix_A_width * ty + tx];
        Bs[ty][tx] = B[b + Matrix_B_width * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < block_size; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = Matrix_B_width * block_size * by + block_size * bx;
    C[c + Matrix_B_width * ty + tx] = Csub;
}


__global__ void matrixMultWithNumber(double *a, double *b, const double v, int memsize)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if(tidx > memsize)
		return;
	b[tidx] = a[tidx] * v;
}

