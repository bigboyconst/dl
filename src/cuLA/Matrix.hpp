#ifndef CULA_MATRIX_HPP
#define CULA_MATRIX_HPP
#include <iostream>

namespace cuLA {
	struct Matrix {
		size_t rows;
		size_t cols;

		float* elements;

		float at(size_t i, size_t j) const {
			return elements[i * cols + j];
		}

		float& at_ref(size_t i, size_t j) {
			return elements[i * cols + j];
		}
	};
}
#endif // CULA_MATRIX_HPP