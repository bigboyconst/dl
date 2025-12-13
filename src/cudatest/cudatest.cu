#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include "cublas_v2.h"
#include <chrono>

#include "../cuLA/cuLA.cuh"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define CUDA_CHECK_ERROR(call) \
do { \
	cudaError_t __stat = (call); \
	if (__stat != cudaSuccess) { \
		std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
				  << cudaGetErrorString(__stat) << "\n"; \
	    exit(EXIT_FAILURE); \
	} \
} while (0)

#define CUBLAS_CHECK_ERROR(call) \
do { \
	cublasStatus_t __stat = (call); \
	if (__stat != CUBLAS_STATUS_SUCCESS) { \
		std::cerr << "CUBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
				  << cublasGetStatusString(__stat) << "\n"; \
		exit(EXIT_FAILURE); \
	} \
} while (0)

// For validation
void matmulCPU(const float* A, const float* B, float* C,
	int m, int K, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float cval = 0.0f;
			for (int k = 0; k < K; k++) {
				cval += A[IDX2C(i, k, m)] * B[IDX2C(k, j, K)];
			}
			C[IDX2C(i, j, m)] = cval;
		}
	}
}

// Function to initialize matrix elements
void initializeMatrix(Matrix& mat) {
	float* data = (float*)malloc(mat.size_bytes());
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            data[IDX2C(i, j, mat.rows)] = static_cast<float>(rand() % 100);
        }
    }
    mat.upload(data);
    free(data);
}

void initializeVector(Vector& vec) {
	float* data = (float*)malloc(vec.size_bytes());
	for (int i = 0; i < vec.count; i++) {
		data[i] = static_cast<float>(rand() % 100);
	}
	vec.upload(data);
	free(data);
}

void printMatrix(const Matrix& mat) {
	float* data = (float*)malloc(mat.size_bytes());
	mat.download(data);
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			std::cout << data[IDX2C(i, j, mat.rows)];
			if (j != mat.cols - 1) {
				std::cout << ", ";
			}
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";
	free(data);
}

void printValues(const float* data, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			std::cout << data[IDX2C(i, j, rows)];
			if (j != cols - 1) {
				std::cout << ", ";
			}
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";
}

void compareGPUtoCPU() {
	srand(1234);

	int M = 1024;
	int K = 768;
	int N = 1024;

	CublasContext ctx = CublasContext();

    Matrix A(M, K);
    Matrix B(K, N);
    Matrix C(M, N);

    initializeMatrix(A);
    initializeMatrix(B);

   	auto start = std::chrono::high_resolution_clock::now();
    cuLA_matMul(ctx, A, B, C);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    printf("GPU Matrix multiplication:\n");
    printf("Duration: %f ms\n", duration.count() * 1000.0f);
    
    // printMatrix(C);

    float* hostA = (float*)malloc(A.size_bytes());
    float* hostB = (float*)malloc(B.size_bytes());
    
    A.download(hostA);
    B.download(hostB);

    float* hostC = (float*)malloc(C.size_bytes());

    start = std::chrono::high_resolution_clock::now();
    matmulCPU(hostA, hostB, hostC, M, K, N);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;

    printf("CPU Matrix multiplication:\n");
    printf("Duration: %f ms\n", duration.count() * 1000.0f);

    // printValues(hostC, M, N);

    free(hostC);
    free(hostB);
    free(hostA);
}

int main() {
	srand(1234);

	Matrix W(10, 12);
	Vector a(12);
	Vector b(10);

	initializeMatrix(W);
	initializeVector(a);
	initializeVector(b);

	W.print();

	std::cout << "\n\n";

	a.print();

	std::cout << "\n\n";

	b.print();

	std::cout << "\n\n";

	Vector z = W * a + b;

	z.print();

    return 0;
}
