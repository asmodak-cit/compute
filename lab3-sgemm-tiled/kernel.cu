/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

    __shared__ float Ads[TILE_SIZE][TILE_SIZE];
    __shared__ float Bds[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0;
    for (int i = 0; i < ((k + TILE_SIZE - 1)/TILE_SIZE); ++i) {

        // Collaborative loading of tiles in to shared memory.

        if ((row < m) && ((i*TILE_SIZE + tx) < k)) {
            Ads[ty][tx] = A[row*k + i*TILE_SIZE + tx];
        } else {
            Ads[ty][tx] = 0.0f;
        }

        if (((i*TILE_SIZE + ty) < k) && (col < n)) {
            Bds[ty][tx] = B[(i*TILE_SIZE + ty)*n + col];
        } else {
            Bds[ty][tx] = 0.0f;
        }

        __syncthreads();


        for (int j = 0; j < TILE_SIZE; ++j) {
            sum += Ads[ty][j] * Bds[j][tx];
        }
        __syncthreads();

    }

    if ((row < m) && (col < n)) {
        C[row*n + col] = sum;
    }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);  // Each block will have BLOCK_SIZE threads in each dimension
    dim3 dim_grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<dim_grid, dim_block>>>(m, n, k, A, B, C);
}


