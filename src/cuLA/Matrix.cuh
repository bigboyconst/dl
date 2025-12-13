#ifndef CULA_MATRIX_CUH
#define CULA_MATRIX_CUH

#include <iostream>

// forward-declaration of vector

struct Vector;

struct Matrix {
	int rows;
	int cols;
	float* device_data;

	Matrix(int r, int c);

	Matrix(int r, int c, float* data);

	~Matrix();

	inline size_t size() const {
		return rows * cols;
	}

	inline size_t size_bytes() const {
		return size() * sizeof(float);
	}

	inline void print() const {
		float* data = (float*)malloc(size_bytes());
		download(data);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				std::cout << data[j * rows + i];
				if (j != cols - 1) {
					std::cout << ", ";
				}
			}
			std::cout << "\n";
		}
		free(data);
	}

	void upload(const float* h_data);

	void download(float* h_data) const;

	Vector linear(const Vector& x, 
		const Vector& b,
		const float alpha = 1.0f,
		const float beta = 1.0f,
		bool transp = false) const;

	Matrix& operator+=(const Matrix& other);

	Matrix& operator-=(const Matrix& other);

	Matrix& operator*=(const Matrix& other);

	Matrix& operator/=(const Matrix& other);

	Matrix& operator*=(const float x);

	Matrix& operator/=(const float x);
};

#endif // CULA_MATRIX_CUH