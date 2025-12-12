#ifndef CULA_VECTOR_HPP
#define CULA_VECTOR_HPP

namespace cuLA {
	struct Vector {
		size_t size;
		float* elements;

		float at(size_t i) {
			return elements[i];
		}

		float& at_ref(size_t i) {
			return elements[i];
		}
	};
}

#endif // CULA_VECTOR_HPP