//----------------------------copyrite 10---------------------------------------------------
/*`copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
//--------------------------copyright 2--------------------------------------------------------
// This file is part of BOINC.
// http://boinc.berkeley.edu
// Copyright (C) 2008 University of California
//
// BOINC is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// BOINC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with BOINC.  If not, see <http://www.gnu.org/licenses/>.
//
// This file contains kernel definition for matrix inversion. The external function
// "invert" serves as an interface between cuda_kernel.cu and cuda.cpp
//
// See http://boinc.berkeley.edu/trac/wiki/GPUApp for any compiling issues
// Contributor: Tuan Le (tuanle86@berkeley.edu)

// When VERIFY is defined, the sum of squared errors is calculated between the
// identity matrix and the product A * incerse(A). For debugging...
//----------------------------------------------------------------------------------------------

#include <stdio.h>
#include "MatrixGPU.h"
#include <math.h>

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


__global__ void MatrixAdd(double *a, double *b, int memsize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid > memsize)
		return;
	a[tid] = a[tid] + b[tid];
}

__global__ void MatrixAddWithNumber(double *a, const double b, int memsize)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid > memsize)
                return;
        a[tid] = a[tid] + b;
}




__global__ void GEStep1A(double * AI, int i, int n2, int lda2) 
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k>i && k < n2 && AI[i*lda2+k]!=0) {
        double multiplyer = -AI[i*lda2+k]/AI[i*lda2+i];
        int n = n2 / 2;
        for (int j = i+1; j < n; j++) {
            AI[j*lda2+k] += multiplyer*AI[j*lda2+i];
        }
    }
}

__global__ void GEStep2(double * AI,double diag,int i, int n2, int lda2) 
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n2) {
        AI[i*lda2+k] /= diag;
    }
}

__global__ void GEStep3(double * AI,int i, int n2, int lda2) 
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > i && k < n2) {
        double multiplyer = -AI[i*lda2+k];
        for (int j = 0; j < i; j++) {
            AI[j*lda2+k] += multiplyer*AI[j*lda2+i];
        }
    }
}

/* Helper function for invert. Kernel calls are made in this function */
void invertge(double * AI_d, int lda, int n) 
{
    int lda2 = lda * 2;
    // perform elementary row operations till A in AI becomes identity matrix
    for (int i = 0; i < n; i++) {
        GEStep1A<<<1, 256>>>(AI_d,i,n*2, lda2);
        cudaThreadSynchronize();
    }

    for (int i = n-1; i >= 0; i--) {
        double diag = 1.0;
        cudaMemcpy(&diag, &AI_d[i*lda2+i], sizeof(double), cudaMemcpyDeviceToHost);
        GEStep2<<<1, 256>>>(AI_d,diag,i,n*2, lda2);

        GEStep3<<<2, 256>>>(AI_d,i,n*2, lda2);
        cudaThreadSynchronize();
    }
}

