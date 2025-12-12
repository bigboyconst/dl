#ifndef CULA_MATMUL_CU
#define CULA_MATMUL_CU
#include "Matrix.hpp"

#define CULA_TILE_SIZE 16

// Credit: https://kharshit.github.io/blog/2024/06/07/matrix-multiplication-cuda
__global__ void cuLA_matMul(cuLA::Matrix A, cuLA::Matrix B, cuLA::Matrix C) {
	__shared__ float shared_A[CULA_TILE_SIZE][CULA_TILE_SIZE];
	__shared__ float shared_B[CULA_TILE_SIZE][CULA_TILE_SIZE];

	int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
	int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

	float CValue = 0.0f;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < (A.cols + CULA_TILE_SIZE - 1) / CULA_TILE_SIZE; m++) {
		if (row < A.rows && (m * CULA_TILE_SIZE + col) < A.cols) {
			shared_A[row][col] = A.elements[globalRow * A.cols + m * CULA_TILE_SIZE + col];
		}
		else {
			shared_A[row][col] = 0.0f;
		}

		if (col < B.cols && (m * CULA_TILE_SIZE + row) < B.rows) {
			shared_B[row][col] = B.elements[(m * CULA_TILE_SIZE + row) * B.cols + globalCol];
		}
		else {
			shared_B[row][col] = 0.0f;
		}

		for (int k = 0; k < CULA_TILE_SIZE; k++) {
			CValue += shared_A[row][k] * shared_B[k][col];
		}

		// Synchronize
		__syncthreads();
	}

	if (globalRow < C.rows && globalCol < C.cols) {
		C.elements[globalRow * C.cols + globalCol] = CValue;
	}
} 
#endif // CULA_MATMUL_CU