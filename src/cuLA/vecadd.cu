#ifndef CULA_VECADD_CU
#define CULA_VECADD_CU

#include "Vector.hpp"

using namespace cuLA;

__global__ void cuLA_vecAdd(Vector u, Vector v, Vector w) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < u.size) {
		w.at_ref(i) = u.at(i) + v.at(i);
	}
}

__global__ void cuLA_vecSub(Vector u, Vector v, Vector w) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < u.size) {
		w.at_ref(i) = u.at(i) - v.at(i);
	}
}

#endif // CULA_VECADD_CU