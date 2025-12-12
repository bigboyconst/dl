#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <chrono>
#include "../cuLA/Matrix.hpp"
#include "../cuLA/matmul.cu"

#define TILE_SIZE CULA_TILE_SIZE

using namespace cuLA;

// Error checking macro
#define CUDA_CHECK_ERROR(call) \
	do { \
	    cudaError_t err = call; \
	    if (err != cudaSuccess) { \
	        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
	                  << cudaGetErrorString(err) << std::endl; \
	        std::exit(EXIT_FAILURE); \
	    } \
	} while (0)

void runKernel(void(*kernel)(Matrix, Matrix, Matrix), const Matrix &A, const Matrix &B, Matrix &C, dim3 gridDim, dim3 blockDim) {
    // Load A, B to device memory
    Matrix d_A, d_B, d_C;

    size_t size_A = A.cols * A.rows * sizeof(float);
    size_t size_B = B.cols * B.rows * sizeof(float);
    size_t size_C = C.cols * C.rows * sizeof(float);

    d_A.cols = A.cols; d_A.rows = A.rows;
    d_B.cols = B.cols; d_B.rows = B.rows;
    d_C.cols = C.cols; d_C.rows = C.rows;

    // Allocate device memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_A.elements, size_A));
    CUDA_CHECK_ERROR(cudaMalloc(&d_B.elements, size_B));
    CUDA_CHECK_ERROR(cudaMalloc(&d_C.elements, size_C));

    // Copy A, B to device memory
    CUDA_CHECK_ERROR(cudaMemcpy(d_A.elements, A.elements, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B.elements, B.elements, size_B, cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    // Synchronize device memory
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Kernel execution time: " << duration.count() * 1000.0f << " ms" << std::endl;

    // Copy C from device memory to host memory
    CUDA_CHECK_ERROR(cudaMemcpy(C.elements, d_C.elements, size_C, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(d_A.elements));
    CUDA_CHECK_ERROR(cudaFree(d_B.elements));
    CUDA_CHECK_ERROR(cudaFree(d_C.elements));
}

// For validation
void matmulCPU(const Matrix& A, const Matrix& B, Matrix& C) {
	for (size_t i = 0; i < A.rows; i++) {
		for (size_t j = 0; j < B.cols; j++) {
			float cval = 0.0f;
			for (size_t k = 0; k < A.cols; k++) {
				cval += A.at(i, k) * B.at(k, j);
			}
			C.at_ref(i, j) = cval;
		}
	}
}

// Function to initialize matrix elements
void initializeMatrix(Matrix &mat) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            mat.at_ref(i, j) = static_cast<float>(rand() % 100);
        }
    }
}

int main() {
    std::cout << "CUDA Matrix Multiplication Comparison\n";

    size_t M = 10; // Rows of A and C
    size_t K = 5; // Columns of A and rows of B
    size_t N = 10; // Columns of B and C

    // Allocate matrices A, B, and C
    Matrix A = {M, K, new float[M * K]}; // 1024x1024
    Matrix B = {K, N, new float[K * N]}; // 768x1024
    Matrix C = {M, N, new float[M * N]}; // 1024x1024
    Matrix C_cpu = {M, N, new float[M * N]};

    initializeMatrix(A);
    initializeMatrix(B);

    dim3 gridDim((C.cols + TILE_SIZE - 1) / TILE_SIZE, (C.rows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    std::cout << "Running shared memory kernel...\n";
    runKernel(cuLA_matMul, A, B, C, gridDim, blockDim);

    matmulCPU(A, B, C_cpu);

    for (size_t i = 0; i < C.rows; i++) {
    	for (size_t j = 0; j < C.cols; j++) {
    		std::cout << C.at(i, j) - C_cpu.at(i, j) << ", ";
    	}
    	std::cout << "\n";
    }

    delete[] A.elements;
    delete[] B.elements;
    delete[] C.elements;

    return 0;
}
