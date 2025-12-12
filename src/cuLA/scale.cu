#ifndef CULA_SCALE_CU
#define CULA_SCALE_CU

#include "Matrix.hpp"
#include "Vector.hpp"

__global__ void cuLA_vecScale(cuLA::Vector v, float val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < v.size) {
		v.at_ref(i) *= val;
	}
}

__global__ void cuLA_matScale(cuLA::Matrix A, float val) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < A.rows && col < A.cols) {
		A.at_ref(row, col) *= val;
	}
}

#endif // CULA_SCALE_CU