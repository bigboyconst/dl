#ifndef CULA_MATRIX_CUH
#define CULA_MATRIX_CUH

struct Matrix {
	int rows;
	int cols;
	float* data;

	inline size_t size() const {
		return rows * cols;
	}

	inline size_t size_bytes() const {
		return size() * sizeof(float);
	}

	inline float at(int i, int j) const {
		return data[j * rows + i];
	}

	inline float& at_ref(int i, int j) {
		return data[j * rows + i];
	}
};

#endif // CULA_MATRIX_CUH