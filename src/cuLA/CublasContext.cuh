#ifndef CULA_CUBLASCONTEXT_CUH
#define CULA_CUBLASCONTEXT_CUH

#include "cublas_v2.h"

class CublasContext {
public:
	cublasHandle_t handle;

	CublasContext() {
		cublasCreate(&handle);
	}

	~CublasContext() {
		cublasDestroy(handle);
	}
}

#endif // CULA_CUBLASCONTEXT_CUH