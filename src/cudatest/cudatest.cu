#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include "../cuLA/Matrix.cuh"
#include "cublas_v2.h"
#include <chrono>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

cublasHandle_t ctx;

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
void matmulCPU(const Matrix& A, const Matrix& B, Matrix& C) {
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < B.cols; j++) {
			float cval = 0.0f;
			for (int k = 0; k < A.cols; k++) {
				cval += A.at(i, k) * B.at(k, j);
			}
			C.at_ref(i, j) = cval;
		}
	}
}

void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
	float alpha = 1.0f, beta = 0.0f;

	CUBLAS_CHECK_ERROR(cublasSgemm(
		ctx,
		CUBLAS_OP_N, CUBLAS_OP_N,
		A.rows, B.cols, A.cols,
		&alpha,
		A.data, A.rows,
		B.data, B.rows,
		&beta,
		C.data, C.rows
	));
}

// Function to initialize matrix elements
void initializeMatrix(Matrix& mat) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            mat.at_ref(i, j) = static_cast<float>(rand() % 100);
        }
    }
}

void printMatrix(const Matrix& mat) {
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			std::cout << mat.at(i, j);
			if (j != mat.cols - 1) {
				std::cout << ", ";
			}
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";
}

int main() {
	srand(1234);

	int M = 1024;
	int K = 768;
	int N = 1024;

    Matrix hostA = {M, K, (float*)malloc(M * K * sizeof(float))};
    Matrix hostB = {K, N, (float*)malloc(K * N * sizeof(float))};
    Matrix hostC = {M, N, (float*)malloc(M * N * sizeof(float))};

    if (!hostA.data || !hostB.data || !hostC.data) {
    	fprintf(stderr, "Memory allocation failed\n Buy more ram ig lol\n");
    	exit(EXIT_FAILURE);
    }

    initializeMatrix(hostA);
    initializeMatrix(hostB);
    
    Matrix devA;
    Matrix devB;
    Matrix devC;

    devA.rows = hostA.rows; devA.cols = hostA.cols;
    devB.rows = hostB.rows; devB.cols = hostB.cols;
    devC.rows = hostC.rows; devC.cols = hostC.cols;

    CUDA_CHECK_ERROR(cudaMalloc(&devA.data, hostA.size_bytes()));
    CUDA_CHECK_ERROR(cudaMalloc(&devB.data, hostB.size_bytes()));
    CUDA_CHECK_ERROR(cudaMalloc(&devC.data, hostC.size_bytes()));

    CUBLAS_CHECK_ERROR(cublasCreate(&ctx));

    CUBLAS_CHECK_ERROR(cublasSetMatrix(
    	hostA.rows, hostA.cols, sizeof(float), 
    	hostA.data, hostA.rows,
    	devA.data, hostA.rows
	));

    CUBLAS_CHECK_ERROR(cublasSetMatrix(
    	hostB.rows, hostB.cols, sizeof(float),
    	hostB.data, hostB.rows,
    	devB.data, hostB.rows
	));

   	auto start = std::chrono::high_resolution_clock::now();
    matmul(devA, devB, devC);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    CUBLAS_CHECK_ERROR(
    	cublasGetMatrix(
    		devC.rows, devC.cols, sizeof(float), 
    		devC.data, devC.rows, 
    		hostC.data, hostC.rows
	));

    printf("GPU Matrix multiplication:\n");
    printf("Duration: %f ms\n", duration.count() * 1000.0f);

    Matrix C2 = {M, N, (float*)malloc(M * N * sizeof(float))};

    printf("CPU Matrix multiplication:\n");
    
    auto start2 = std::chrono::high_resolution_clock::now();
    matmulCPU(hostA, hostB, C2);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d2 = end2 - start2;
    printf("Duration: %f ms\n", d2.count() * 1000.0f);

    cudaFree(devA.data);
    cudaFree(devB.data);
    cudaFree(devC.data);

    cublasDestroy(ctx);

	free(hostA.data);
	free(hostB.data);
	free(hostC.data);
	free(C2.data);
}
