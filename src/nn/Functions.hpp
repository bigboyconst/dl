#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <cmath>

namespace Functions {
	inline float sigmoid(float x) {
		return 1.0f / (1.0f + std::exp(-x));
	}

	inline float dSigmoid(float x) {
		float s = sigmoid(x);
		return s * (1.0f - s);
	}

	inline float relu(float x) {
		return std::max(0.0f, x);
	}

	inline float dRelu(float x) {
		return x > 0.0f ? 1.0f : 0.0f;
	}
}

#endif // FUNCTIONS_HPP