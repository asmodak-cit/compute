/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

#define FILTER_RADIUS ((FILTER_SIZE - 1) / 2)

__global__ void convolution(Matrix N, Matrix P)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

    //INSERT KERNEL CODE HERE

    
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((out_row < P.height) && (out_col < P.width)) {

        
        float sum = 0.0f;

        for (int mask_row = 0; mask_row < FILTER_SIZE; mask_row++) {
			for (int mask_col = 0; mask_col < FILTER_SIZE; mask_col++) {
				
				
				int in_row = out_row - FILTER_RADIUS + mask_row;
				int in_col = out_col - FILTER_RADIUS + mask_col;
			    if (in_row >= 0 && in_row < N.height && in_col >= 0 && in_col < N.width) {

		            sum += M_c[mask_row][mask_col] * N.elements[in_row*N.width + in_col];                    

				}
				

			}
        }

        P.elements[out_row*P.width + out_col] = sum;		

    }


}
