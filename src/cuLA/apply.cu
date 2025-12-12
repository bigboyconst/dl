#ifndef CULA_APPLY_CU
#define CULA_APPLY_CU

#include "Vector.hpp"
#include "Matrix.hpp"

__global__ void cuLA_vecApply(cuLA::Vector v, float(*fn)(float)) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < v.size) {
		v.elements[i] = fn(v.elements[i]);
	}
}

__global__ void cuLA_matApply(cuLA::Matrix A, float(*fn)(float)) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * A.cols + col;

	if (row < A.rows && col < A.cols) {
		A.elements[index] = fn(A.elements[index]);
	}
}

#endif // CULA_APPLY_CU