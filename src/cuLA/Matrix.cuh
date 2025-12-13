#ifndef CULA_MATRIX_CUH
#define CULA_MATRIX_CUH

struct Matrix {
	int rows;
	int cols;
	float* device_data;

	Matrix(int r, int c);

	~Matrix();

	inline size_t size() const {
		return rows * cols;
	}

	inline size_t size_bytes() const {
		return size() * sizeof(float);
	}

	void upload(const float* h_data);

	void download(float* h_data) const;
};

#endif // CULA_MATRIX_CUH