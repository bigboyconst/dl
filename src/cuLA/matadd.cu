#ifndef CULA_MATADD_CU
#define CULA_MATADD_CU
#include "Matrix.hpp"

__global__ void cuLA_matAdd(cuLA::Matrix A, cuLA::Matrix B, cuLA::Matrix C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * A.cols + col;

	if (row < A.rows && col < A.cols) {
		C.elements[index] = A.elements[index] + B.elements[index];
	}
}

__global__ void cuLA_matSub(cuLA::Matrix A, cuLA::Matrix B, cuLA::Matrix C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * A.cols + col;

	if (row < A.rows && col < A.cols) {
		C.elements[index] = A.elements[index] - B.elements[index];
	}
}

#endif // CULA_MATADD_CU