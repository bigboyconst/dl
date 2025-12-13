#include "cuLA.cuh"

#include <cuda_runtime.h>
#include "cublas_v2.h"

__global__ void mat_sub(const float* A, 
	const float* B,
	float* C, 
	int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		C[i] = A[i] - B[i];
	}
}

__global__ void mat_add(const float* A, 
	const float* B, 
	float* C, 
	int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

void cuLA_matAdd(
	CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C) {

}

void cuLA_matMul(
	CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C) {
	float alpha = 1.0f;
	float beta = 0.0f;

	cublasSgemm(
		ctx.handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		A.rows, B.cols, A.cols,
		&alpha,
		A.device_data, A.rows,
		B.device_data, B.rows,
		&beta,
		C.device_data, C.rows
	);
}

void cuLA_matVecMul(
	CublasContext& ctx,
	const Matrix& A,
	const Vector& x,
	Vector& y) {
	float alpha = 1.0f;
	float beta = 0.0f;

	cublasSgemv(
		ctx.handle,
		CUBLAS_OP_N,
		A.rows, A.cols,
		&alpha,
		A.device_data, A.rows,
		x.device_data, 1,
		&beta,
		y.device_data, 1
	);
}

void cuLA_linear(
	CublasContext& ctx,
	const Matrix& A,
	const Vector& x,
	Vector& y,
	const float a = 1.0f,
	const float b = 1.0f,
	bool transp = false) {
	cublasSgemv(
		ctx.handle, 
		transp ? CUBLAS_OP_T : CUBLAS_OP_N,
		A.rows, A.cols,
		&a,
		A.device_data, A.rows,
		x.device_data, 1,
		&b,
		y.device_data, 1
	);
}



#pragma region Vector Implementations

Vector::Vector(int count) : count(count) {
	cudaMalloc(&device_data, count * sizeof(float));
}

Vector::~Vector() {
	cudaFree(device_data);
}

void Vector::upload(const float* h_data) {
	cublasSetVector(
		count, sizeof(float),
		h_data, 1,
		device_data, 1
	);
}

void Vector::download(float* h_data) const {
	cublasSetVector(
		count, sizeof(float),
		device_data, 1,
		h_data, 1
	);
}

#pragma endregion

// =========================================

#pragma region Matrix Implementations

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
	cudaMalloc(&device_data, r * c * sizeof(float));
}

Matrix::~Matrix() {
	cudaFree(device_data);
}

void Matrix::upload(const float* h_data) {
	cublasSetMatrix(
		rows, cols, sizeof(float),
		h_data, rows,
		device_data, rows
	);
}

void Matrix::download(float* h_data) const {
	cublasSetMatrix(
		rows, cols, sizeof(float),
		device_data, rows,
		h_data, rows
	);
}

#pragma endregion