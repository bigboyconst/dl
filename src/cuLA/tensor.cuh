#ifndef CULA_TENSOR_CUH
#define CULA_TENSOR_CUH

#include <vector>

template<typename T>
using vector = std::vector<T>;

enum class Device {
	CPU,
	CUDA
};

class Tensor {
public:
	float* device_ptr = nullptr;
	float* host_ptr = nullptr;

	vector<int> shape;
	Device device;

	Tensor(const vector<int>& shape, Device device = Device::CUDA);
	~Tensor();

	size_t size() const;
	size_t size_bytes() const;

	void to(Device target);
	void allocate();
};

#endif // CULA_TENSOR_CUH