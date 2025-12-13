#ifndef CULA_CULA_CUH
#define CULA_CULA_CUH

#include "CublasContext.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"

#define CULA_THREAD_COUNT 256
#define CULA_BLOCKS(n, threads) ((n) + (threads) - 1) / (threads)

namespace cuLA {
	inline CublasContext ctx;
}

__global__ void vec_add(const float* a, 
	const float* b, 
	float* c, 
	int n);

__global__ void vec_sub(const float* a, 
	const float* b, 
	float* c, 
	int n);

__global__ void vec_mul(const float* a, 
	const float* b, 
	float* c, 
	int n);

__global__ void vec_div(const float* a, 
	const float* b, 
	float* c, 
	int n);

__global__ void vec_scale(const float* a,
	const float x,
	float* c,
	int n);

__global__ void mat_add(const float* A, 
	const float* B, 
	float* C, 
	int n);

__global__ void mat_sub(const float* A, 
	const float* B,
	float* C, 
	int n);

__global__ void mat_scale(const float* A,
	const float x,
	float* C,
	int n);

__global__ void mat_apply(const float* A, 
	float(*fn)(float),
	float* C,
	int n);

void cuLA_vecAdd(CublasContext& ctx,
	const Vector& a,
	const Vector& b,
	Vector& c);

void cuLA_vecSub(CublasContext& ctx,
	const Vector& a,
	const Vector& b,
	Vector& c);

void cuLA_vecMul(CublasContext& ctx,
	const Vector& a,
	const Vector& b,
	Vector& c);

void cuLA_vecDiv(CublasContext& ctx,
	const Vector& a,
	const Vector& b,
	Vector& c);

void cuLA_vecScale(CublasContext& ctx,
	const Vector& a,
	const float x,
	Vector& c);

void cuLA_matAdd(CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C);

void cuLA_matSub(CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C);

void cuLA_matScale(CublasContext& ctx,
	const Matrix& A,
	const float val,
	Matrix& C);

void cuLA_matApply(CublasContext& ctx,
	const Matrix& A,
	float(*fn)(float),
	Matrix& C);

void cuLA_matMul(CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C);

void cuLA_matVecMul(CublasContext& ctx,
	const Matrix& A,
	const Vector& x,
	Vector& y);

void cuLA_linear(CublasContext& ctx,
	const Matrix& A,
	const Vector& x,
	Vector& y,
	const float a = 1.0f,
	const float b = 1.0f,
	bool transp = false);

#pragma region Vector operators

Vector operator+(const Vector& v1, const Vector& v2);

Vector operator-(const Vector& v1, const Vector& v2);

Vector operator*(const Vector& v1, const Vector& v2);

Vector operator*(const float x, const Vector& v1);

Vector operator*(const Vector& v1, const float x);

Vector operator/(const Vector& v1, const Vector& v2);

Vector operator/(const Vector& v, const float x);

#pragma endregion

#pragma region Matrix operators

Matrix operator+(const Matrix& v1, const Matrix& v2);

Matrix operator-(const Matrix& v1, const Matrix& v2);

Matrix operator*(const Matrix& v1, const Matrix& v2);

Matrix operator*(const float x, const Matrix& v1);

Vector operator*(const Matrix& mat, const Vector& vec);

Matrix operator*(const Matrix& v1, const float x);

Matrix operator/(const Matrix& v, const float x);

#pragma endregion

#endif // CULA_CULA_CUH