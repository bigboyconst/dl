#include "cuLA.cuh"

#include <cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"

__global__ void vec_add(const float* a,
	const float* b,
	float* c,
	int n) {
	int i = threadIdx.x;

	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

__global__ void vec_sub(const float* a,
	const float* b,
	float* c,
	int n) {
	int i = threadIdx.x;

	if (i < n) {
		c[i] = a[i] - b[i];
	}
}

__global__ void vec_mul(const float* a, 
	const float* b,
	float* c,
	int n) {
	int i = threadIdx.x;

	if (i < n) {
		c[i] = a[i] * b[i];
	}
}

__global__ void vec_div(const float* a, 
	const float* b, 
	float* c,
	int n) {
	int i = threadIdx.x;

	if (i < n) {
		c[i] = a[i] / b[i];
	}
}

__global__ void vec_scale(const float* a,
	const float x,
	float* c,
	int n) {
	int i = threadIdx.x;

	if (i < n) {
		c[i] = a[i] * x;
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

__global__ void mat_sub(const float* A, 
	const float* B,
	float* C, 
	int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		C[i] = A[i] - B[i];
	}
}

__global__ void mat_scale(const float* A,
	const float x,
	float* C,
	int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		C[i] = A[i] * x;
	}
}

__global__ void mat_apply(const float* A,
	float(*fn)(float),
	float* C,
	int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		C[i] = fn(A[i]);
	}
}

void cuLA_vecAdd(CublasContext& ctx,
	const Vector& a,
	const Vector& b,
	Vector& c) {
	assert(a.count == b.count);

	int n = a.count;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	vec_add<<<blocks, threads>>>(a.device_data, b.device_data, c.device_data, n);
}

void cuLA_vecSub(CublasContext& ctx,
	const Vector& a,
	const Vector& b,
	Vector& c) {
	assert(a.count == b.count);

	int n = a.count;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	vec_sub<<<blocks, threads>>>(a.device_data, b.device_data, c.device_data, n);
}

void cuLA_vecMul(CublasContext& ctx,
	const Vector& a,
	const Vector& b,
	Vector& c) {
	assert(a.count == b.count);

	int n = a.count;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	vec_mul<<<blocks, threads>>>(a.device_data, b.device_data, c.device_data, n);
}

void cuLA_vecDiv(CublasContext& ctx,
	const Vector& a,
	const Vector& b,
	Vector& c) {
	assert(a.count == b.count);

	int n = a.count;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	vec_div<<<blocks, threads>>>(a.device_data, b.device_data, c.device_data, n);
}

void cuLA_vecScale(CublasContext& ctx,
	const Vector& a,
	const float x,
	Vector& c) {
	int n = a.count;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	vec_scale<<<blocks, threads>>>(a.device_data, x, c.device_data, n);
}

void cuLA_matAdd(
	CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C) {
	assert(A.rows == B.rows && A.cols == B.cols);
	
	int n = A.rows * A.cols;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	mat_add<<<blocks, threads>>>(A.device_data, B.device_data, C.device_data, n);
}

void cuLA_matSub(
	CublasContext& ctx,
	const Matrix& A,
	const Matrix& B,
	Matrix& C) {
	assert(A.rows == B.rows && A.cols == B.cols);

	int n = A.rows * A.cols;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	mat_sub<<<blocks, threads>>>(A.device_data, B.device_data, C.device_data, n);
}

void cuLA_matScale(
	CublasContext& ctx,
	const Matrix& A,
	const float val,
	Matrix& C) {
	int n = A.rows * A.cols;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	mat_scale<<<blocks, threads>>>(A.device_data, val, C.device_data, n);
}

void cuLA_matApply(
	CublasContext& ctx,
	const Matrix& A,
	float(*fn)(float),
	Matrix& C) {
	int n = A.rows * A.cols;
	int threads = CULA_THREAD_COUNT;
	int blocks = CULA_BLOCKS(n, threads);

	mat_apply<<<blocks, threads>>>(A.device_data, fn, C.device_data, n);
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
	const float a,
	const float b,
	bool transp) {
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
	cublasGetVector(
		count, sizeof(float),
		device_data, 1,
		h_data, 1
	);
}

Matrix Vector::outer(const Vector& other) const {
	Matrix m_v = {count, 1, device_data};
	Matrix m_o = {other.count, 1, other.device_data};
	Matrix output(count, other.count);
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSgemm(
		cuLA::ctx.handle,
		CUBLAS_OP_N, CUBLAS_OP_T,
		count, 1, other.count,
		&alpha,
		m_v.device_data, count,
		m_o.device_data, other.count,
		&beta,
		output.device_data, count
	);

	return output;
}

Vector operator+(const Vector& v1, const Vector& v2) {
	Vector v(v1.count);
	cuLA_vecAdd(cuLA::ctx, v1, v2, v);
	return v;
}

Vector operator-(const Vector& v1, const Vector& v2) {
	Vector v(v1.count);
	cuLA_vecSub(cuLA::ctx, v1, v2, v);
	return v;
}

Vector operator*(const Vector& v1, const Vector& v2) {
	Vector v(v1.count);
	cuLA_vecMul(cuLA::ctx, v1, v2, v);
	return v;
}

Vector operator*(const float x, const Vector& v1) {
	Vector v(v1.count);
	cuLA_vecScale(cuLA::ctx, v1, x, v);
	return v;
}

Vector operator*(const Vector& v1, const float x) {
	return x * v1;
}

Vector operator/(const Vector& v1, const Vector& v2) {
	Vector v(v1.count);
	cuLA_vecDiv(cuLA::ctx, v1, v2, v);
	return v;
}

Vector operator/(const Vector& v, const float x) {
	return v * (1.0f / x);
}



Vector& Vector::operator+=(const Vector& other) {
	*this = *this + other;
	return *this;
}

Vector& Vector::operator-=(const Vector& other) {
	*this = *this - other;
	return *this;
}

Vector& Vector::operator*=(const Vector& other) {
	*this = *this * other;
	return *this;
}

Vector& Vector::operator*=(const float x) {
	*this = *this * x;
	return *this;
}

Vector& Vector::operator/=(const Vector& other) {
	*this = *this / other;
	return *this;
}

Vector& Vector::operator/=(const float x) {
	*this = *this / x;
	return *this;
}

// TODO: Compound assignment

#pragma endregion

// =========================================

#pragma region Matrix Implementations

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
	cudaMalloc(&device_data, r * c * sizeof(float));
}

Matrix::Matrix(int r, int c, float* data) : rows(r), cols(c) {
	cudaMalloc(&device_data, r * c * sizeof(float));

	cublasSetMatrix(
		r, c, sizeof(float),
		data, rows,
		device_data, rows
	);
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
	cublasGetMatrix(
		rows, cols, sizeof(float),
		device_data, rows,
		h_data, rows
	);
}

Vector Matrix::linear(const Vector& x, 
	const Vector& b, 
	const float alpha, 
	const float beta, 
	bool transp) const {
	Vector v = b;
	cuLA_linear(cuLA::ctx, *this, x, v, alpha, beta, transp);
	return v;
}

Matrix operator+(const Matrix& v1, const Matrix& v2) {
	Matrix v(v1.rows, v1.cols);
	cuLA_matAdd(cuLA::ctx, v1, v2, v);
	return v;
}

Matrix operator-(const Matrix& v1, const Matrix& v2) {
	Matrix v(v1.rows, v1.cols);
	cuLA_matSub(cuLA::ctx, v1, v2, v);
	return v;
}

Matrix operator*(const Matrix& v1, const Matrix& v2) {
	Matrix v(v1.rows, v2.cols);
	cuLA_matMul(cuLA::ctx, v1, v2, v);
	return v;
}

Matrix operator*(const float x, const Matrix& v1) {
	Matrix v(v1.rows, v1.cols);
	cuLA_matScale(cuLA::ctx, v1, x, v);
	return v;
}

Vector operator*(const Matrix& mat, const Vector& vec) {
	Vector v(mat.rows);
	cuLA_matVecMul(cuLA::ctx, mat, vec, v);
	return v;
}

Matrix operator*(const Matrix& v1, const float x) {
	return x * v1;
}

Matrix operator/(const Matrix& v, const float x) {
	return v * (1.0f / x);
}



Matrix& Matrix::operator+=(const Matrix& other) {
	*this = *this + other;
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
	*this = *this - other;
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& other) {
	*this = *this * other;
	return *this;
}

Matrix& Matrix::operator*=(const float x) {
	*this = *this * x;
	return *this;
}

Matrix& Matrix::operator/=(const float x) {
	*this = *this / x;
	return *this;
}

#pragma endregion