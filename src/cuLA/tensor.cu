#include <cuda_runtime.h>
#include <numeric>
#include <functional>
#include "tensor.cuh"
#include "cuda_utils.cuh"

Tensor::Tensor(const vector<int>& shape, Device device) : shape(shape), device(device) {

}

Tensor::~Tensor() {

}

size_t Tensor::size() const {
	return static_cast<size_t>(std::accumulate(
		shape.begin(),
		shape.end(),
		1,
		std::multiplies<int>()
	));
}

size_t Tensor::size_bytes() const {
	return size() * sizeof(float);
}

void Tensor::to(Device target) {
	if (device == target) return;

	if (target == Device::CUDA) {
		CUDA_CHECK_ERROR(cudaMalloc(&device_ptr, size_bytes()));
		CUDA_CHECK_ERROR(cudaMemcpy(device_ptr, host_ptr, size_bytes(), cudaMemcpyHostToDevice));
	}
	else {
		host_ptr = new float[size()];
		cudaMemcpy(host_ptr, device_ptr, size_bytes(), cudaMemcpyDeviceToHost);
	}

	device = target;
}

void Tensor::allocate() {

}