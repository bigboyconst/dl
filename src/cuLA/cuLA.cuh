#ifndef CULA_CULA_CUH
#define CULA_CULA_CUH

#include "CublasContext.cuh"

#include "Vector.cuh"
#include "Matrix.cuh"

#include <cublas_v2.h>

void cuLA_matMul(
	CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C
) {
	float alpha = 1.0f;
	float beta = 0.0f;

	cublasSgemm(
		ctx.handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		A.rows, B.cols, A.cols,
		&alpha,
		A.data, A.rows,
		B.data, B.rows,
		&beta,
		C.data, C.rows
	);
}

#pragma region Vector Implementations



#pragma endregion

// =========================================

#pragma region Matrix Implementations



#pragma endregion

#endif // CULA_CULA_CUH