#include <iostream>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	constexpr int N = 200;

	float hostA[N], hostB[N], hostC[N];

	for (int i = 0; i < N; i++) {
		hostA[i] = i + 1;
		hostB[i] = 2 * i;
	}

	float* devA;
	float* devB;
	float* devC;

	size_t bytes = N * sizeof(float);

	cudaMalloc(&devA, bytes);
	cudaMalloc(&devB, bytes);
	cudaMalloc(&devC, bytes);

	cudaMemcpy(devA, hostA, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, hostB, bytes, cudaMemcpyHostToDevice);

	int threads = 256;
	int blocks = (N + threads - 1) / threads;

	add<<<blocks, threads>>>(devA, devB, devC, N);

	cudaDeviceSynchronize();

	cudaMemcpy((void*)hostC, (void*)devC, bytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("[%d] %f + %f = %f\n", i, hostA[i], hostB[i], hostC[i]);
	}

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	return 0;
}