#ifndef CULA_VECTOR_CUH
#define CULA_VECTOR_CUH

#include "Matrix.cuh"
#include <iostream>

struct Vector {
	int count;
	float* device_data;

	Vector(int count);

	~Vector();

	inline size_t size() const {
		return count;
	}

	inline size_t size_bytes() const {
		return size() * sizeof(float);
	}

	inline void print() const {
		float* data = (float*)malloc(size_bytes());
		download(data);
		std::cout << "<";
		for (int i = 0; i < count; i++) {
			std::cout << data[i];
			if (i != count - 1) {
				std::cout << ", ";
			}
			else {
				std::cout << ">\n";
			}
		}
		free(data);
	}

	void upload(const float* h_data);

	void download(float* h_data) const;

	Matrix outer(const Vector& other) const;

	Vector apply(float(*fn)(float)) const;

	Vector& operator+=(const Vector& other);

	Vector& operator-=(const Vector& other);

	Vector& operator*=(const Vector& other);

	Vector& operator/=(const Vector& other);

	Vector& operator*=(const float x);

	Vector& operator/=(const float x);
};

#endif // CULA_VECTOR_CUH