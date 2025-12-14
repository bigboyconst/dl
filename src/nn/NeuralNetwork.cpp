#include "NeuralNetwork.hpp"
#include "Functions.hpp"
#include <cuLA.cuh>

template<typename... Args>
NeuralNetwork::NeuralNetwork(Args... args) {
	int layerSizes[] = { static_cast<int>(args)... };
	constexpr int n = sizeof...(args);

	for (int layer = 0; layer < n - 1; layer++) {
		Weights.emplace_back(layerSizes[layer + 1], layerSizes[layer]);
		Biases.emplace_back(layerSizes[layer + 1]);
	}
}

Vector NeuralNetwork::forward(const Vector& input) {
	activations.clear();
	zs.clear();

	activations.push_back(input);

	Vector a = input;

	for (int i = 0; i < Weights.size(); i++) {
		Vector z = Weights[i] * a + Biases[i];
		zs.push_back(z);

		a = z.apply(Functions::sigmoid);

		activations.push_back(a);
	}

	return a;
}

void NeuralNetwork::backprop(const Vector& input, const Vector& target, float learningRate) {
	Vector output = forward(input);

	// MSE loss = 2 * (output - target)
	Vector lossDerivative = 2.0f * (output - target);

	vector<Matrix> nablaW = {};
	vector<Vector> nablaB = {};

	for (int i = 0; i < Weights.size(); i++) {
		nablaW.emplace_back(Weights[i].rows, Weights[i].cols);
		nablaB.emplace_back(Biases[i].count);
	}

	Vector delta = lossDerivative;
	nablaB[nablaB.size() - 1] = delta;
	nablaW[nablaW.size() - 1] = delta.outer(activations[activations.size() - 2]);

	for (int l = Weights.size() - 2; l >= 0; l--) {
		delta = (Weights[l + 1].transpose_mul(delta)) * zs[l].apply(Functions::dSigmoid);
		nablaB[l] = delta;
		nablaW[l] = delta.outer(activations[l]);
	}

	for (int i = 0; i < Weights.size(); i++) {
		Weights[i] -= learningRate * nablaW[i];
		Biases[i] -= learningRate * nablaB[i];
	}
}