/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }

    float* A_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    float* B_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc( sizeof(float)*n );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    int size = sizeof(float)*n;

    float* A_d;
    cuda_ret = cudaMalloc((void**)&A_d, size);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMalloc failed for A_d: %s\n", cudaGetErrorString(cuda_ret));
        return -1;
    }

    float* B_d;
    cuda_ret = cudaMalloc((void**)&B_d, size);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMalloc failed for B_d: %s\n", cudaGetErrorString(cuda_ret));
        return -1;
    }

    float* C_d;
    cuda_ret = cudaMalloc((void**)&C_d, size);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMalloc failed for C_d: %s\n", cudaGetErrorString(cuda_ret));
        return -1;
    }


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    cuda_ret = cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_ret));
        return -1;
    }

    cuda_ret = cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_ret));
        return -1;
    }

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    cuda_ret = cudaPeekAtLastError();
    if (cuda_ret != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(cuda_ret));
        return -1;
    }

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    cuda_ret = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_ret));
        return -1;
    }

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE

    cuda_ret = cudaFree(A_d);
    if (cuda_ret != cudaSuccess) {
        printf("CUDA Free Error: %s\n", cudaGetErrorString(cuda_ret));
    }
    cuda_ret = cudaFree(B_d);
    if (cuda_ret != cudaSuccess) {
        printf("CUDA Free Error: %s\n", cudaGetErrorString(cuda_ret));
    }
    cuda_ret = cudaFree(C_d);
    if (cuda_ret != cudaSuccess) {
        printf("CUDA Free Error: %s\n", cudaGetErrorString(cuda_ret));
    }

    return 0;

}

