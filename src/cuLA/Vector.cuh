#ifndef CULA_VECTOR_CUH
#define CULA_VECTOR_CUH

struct Vector {
	int count;
	float* data;

	size_t size() const {
		return count;
	}

	size_t size_bytes() const {
		return size() * sizeof(float);
	}
};

#endif // CULA_VECTOR_CUH