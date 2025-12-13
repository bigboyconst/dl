#ifndef CULA_CULA_CUH
#define CULA_CULA_CUH

#include "CublasContext.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"

__global__ void mat_sub(const float* A, 
	const float* B,
	float* C, 
	int n);

__global__ void mat_add(const float* A, 
	const float* B, 
	float* C, 
	int n);

void cuLA_matAdd(
	CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C);

void cuLA_matMul(
	CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C);

void cuLA_matVecMul(
	CublasContext& ctx,
	const Matrix& A,
	const Vector& x,
	Vector& y);

void cuLA_linear(
	CublasContext& ctx,
	const Matrix& A,
	const Vector& x,
	Vector& y,
	const float a = 1.0f,
	const float b = 1.0f,
	bool transp = false);

// #pragma region Vector Implementations

// Vector::Vector(int count) : count(count) {
// 	cudaMalloc(&device_data, count * sizeof(float));
// }

// Vector::~Vector() {
// 	cudaFree(device_data);
// }

// #pragma endregion

// // =========================================

// #pragma region Matrix Implementations

// Matrix::Matrix(int r, int c) : rows(r), cols(c) {
// 	cudaMalloc(&device_data, r * c * sizeof(float));
// }

// Matrix::~Matrix() {
// 	cudaFree(device_data);
// }

// #pragma endregion

#endif // CULA_CULA_CUH