#ifndef CULA_VECADD_CU
#define CULA_VECADD_CU

#include "Vector.hpp"

using namespace cuLA;

__global__ void cuLA_vecAdd(Vector u, Vector v, Vector w) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < u.size) {
		w.elements[i] = u.elements[i] + v.elements[i];
	}
}

__global__ void cuLA_vecSub(Vector u, Vector v, Vector w) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < u.size) {
		w.elements[i] = u.elements[i] - v.elements[i];
	}
}

#endif // CULA_VECADD_CU