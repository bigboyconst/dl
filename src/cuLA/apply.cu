#ifndef CULA_APPLY_CU
#define CULA_APPLY_CU

#include "Vector.hpp"
#include "Matrix.hpp"

__global__ void cuLA_vecApply(cuLA::Vector v, float(*fn)(float)) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < v.size) {
		v.at_ref(i) = fn(v.at(i));
	}
}

__global__ void cuLA_matApply(cuLA::Matrix A, float(*fn)(float)) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < A.rows && col < A.cols) {
		A.at_ref(i, j) = fn(A.at(i, j));
	}
}

#endif // CULA_APPLY_CU