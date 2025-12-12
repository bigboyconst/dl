#ifndef CULA_MATADD_CU
#define CULA_MATADD_CU
#include "Matrix.hpp"

__global__ void cuLA_matAdd(cuLA::Matrix A, cuLA::Matrix B, cuLA::Matrix C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < A.rows && j < A.cols) {
		C.at_ref(row, col) = A.at(row, col) + B.at(row, col);
	}
}

__global__ void cuLA_matSub(cuLA::Matrix A, cuLA::Matrix B, cuLA::Matrix C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < A.rows && j < A.cols) {
		C.at_ref(row, col) = A.at(row, col) - B.at(row, col);
	}
}

#endif // CULA_MATADD_CU